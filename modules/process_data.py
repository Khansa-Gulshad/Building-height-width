# process_data.py
import os
os.environ['USE_PYGEOS'] = '0'

import io
import csv
import time
import math
import argparse
import requests
import numpy as np

import geopandas as gpd
import torch
from PIL import Image, ImageFile
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

# use your 3-class helpers from segmentation.py
from segmentation import remap_to_three, save_three_class_mask, save_overlay
from modules.config import PROJECT_DIR, city_to_dir

ImageFile.LOAD_TRUNCATED_IMAGES = True


# =========================
# FOLDERS
# =========================
def prepare_folders(city: str):
    base = os.path.join("/mnt/project/pt01183/results", city)
    for sub in ["seg_3class", "seg_qa", "sample_images"]:
        os.makedirs(os.path.join(base, sub), exist_ok=True)


# =========================
# MODEL LOADING
# =========================
def get_models():
    """
    Load Mask2Former (Cityscapes) on GPU if available.
    """
    processor = AutoImageProcessor.from_pretrained(
        "facebook/mask2former-swin-large-cityscapes-semantic"
    )
    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        "facebook/mask2former-swin-large-cityscapes-semantic"
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    return processor, model


def segment_image(image_pil, processor, model):
    """
    Run semantic segmentation and return torch tensor (H,W) int labels.
    """
    inputs = processor(images=image_pil, return_tensors="pt")
    with torch.no_grad():
        if torch.cuda.is_available():
            inputs = {k: v.to('cuda') for k, v in inputs.items()}
            outputs = model(**inputs)
            seg = processor.post_process_semantic_segmentation(
                outputs, target_sizes=[image_pil.size[::-1]]
            )[0].to('cpu')
        else:
            outputs = model(**inputs)
            seg = processor.post_process_semantic_segmentation(
                outputs, target_sizes=[image_pil.size[::-1]]
            )[0]
    return seg  # torch (H,W)


# =========================
# STREET VIEW FETCH (DIRECT)
# =========================
def fetch_gsv_image_by_location(
    lat, lon, heading, pitch=6, fov=70, size="640x640",
    api_key=None, retries=3, backoff=1.6, timeout=20
):
    """
    Fetch a Street View image for given pose. Returns PIL.Image RGB.
    """
    assert api_key, "GSV API key required"
    url = (
        "https://maps.googleapis.com/maps/api/streetview"
        f"?size={size}&location={lat},{lon}&heading={heading}&pitch={pitch}&fov={fov}&key={api_key}"
    )
    last_err = None
    for attempt in range(retries):
        try:
            resp = requests.get(url, timeout=timeout)
            resp.raise_for_status()
            return Image.open(io.BytesIO(resp.content)).convert("RGB")
        except Exception as e:
            last_err = e
            time.sleep((backoff ** attempt))
    raise last_err


# =========================
# SINGLE VIEW → 3-CLASS MASK
# =========================
def process_facade_view(img_pil, processor, model):
    """
    Returns uint8 mask with values {0,1,2,3}; we use {1=building,2=sky,3=ground}.
    """
    seg_full = segment_image(img_pil, processor, model)  # torch (H,W)
    mask_full_np = seg_full.cpu().numpy().astype(np.int32)
    mask3 = remap_to_three(mask_full_np).astype(np.uint8)
    return mask3


# =========================
# PER-POINT RUNNER (two headings: road_angle ± 90°)
# =========================
def _round_heading(h):  # nicer filenames
    return int(round(h)) % 360

def download_facade_masks_for_point(
    row, city, access_token, processor, model,
    pitch_deg=6, fov_deg=70, size="640x640",
    save_sample=False
):
    """
    row: GeoDataFrame row with:
      - id
      - geometry (Point; x=lon, y=lat) in WGS84
      - road_angle (deg from North)
    Fetch two images: headings = road_angle ± 90°, segment, save masks (+ overlay).
    """
    lat, lon = row.geometry.y, row.geometry.x
    try:
        ra = float(row.road_angle)
        if math.isnan(ra):
            ra = 0.0
    except Exception:
        ra = 0.0

    headings = [ (ra + 90) % 360, (ra + 270) % 360 ]
    records = []

    for h in headings:
        image_id = f"{row.id}_{_round_heading(h)}"
        try:
            img = fetch_gsv_image_by_location(
                lat, lon,
                heading=h, pitch=pitch_deg, fov=fov_deg, size=size,
                api_key=access_token
            )
            mask3 = process_facade_view(img, processor, model)

            # save mask (grayscale PNG; values {0,1,2,3})
            mask_path = save_three_class_mask(city, image_id, mask3)

            # optional QA overlay
            if save_sample:
                save_overlay(city, image_id, np.array(img), mask3)

            records.append([image_id, mask_path, h, pitch_deg, fov_deg])
        except Exception:
            records.append([image_id, "ERROR", h, pitch_deg, fov_deg])

    return records  # list of [image_id, mask_path|ERROR, heading, pitch, fov]


# =========================
# BATCH DRIVER (uses *your* points with road_angle)
# =========================
def download_images_for_points(
    gdf, access_token, city,
    pitch_deg=6, fov_deg=70, size="640x640",
    save_sample=False, max_workers=1
):
    """
    Runs façade segmentation on a points GDF that ALREADY has 'road_angle'.
    gdf must be in WGS84 (EPSG:4326).
    """
    prepare_folders(city)
    processor, model = get_models()

    manifest = []
    max_workers = max(1, int(max_workers))  # single-GPU → usually keep 1

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = []
        for _, row in gdf.iterrows():
            futures.append(
                ex.submit(
                    download_facade_masks_for_point,
                    row, city, access_token, processor, model,
                    pitch_deg, fov_deg, size, save_sample
                )
            )

        for f in tqdm(as_completed(futures), total=len(futures), desc="Façade masks (±90°)"):
            try:
                recs = f.result()
                manifest.extend(recs)
            except Exception:
                manifest.append(["POINT_ERROR", "ERROR", None, pitch_deg, fov_deg])

    # write manifest
    out_dir = os.path.join("/mnt/project/pt01183/results", city, "seg_3class")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "manifest.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["image_id", "mask_path", "heading_deg", "pitch_deg", "fov_deg"])
        for row in manifest:
            w.writerow(row)

    return manifest


# =========================
# CLI (reads your existing points file; no building/sweeps)
# =========================
def main():
    ap = argparse.ArgumentParser(description="Façade 3-class segmentation from points with road_angle (±90° only)")
    ap.add_argument("--city", type=str, required=True, help="City name (for results path)")
    ap.add_argument("--api_key", type=str, default=os.getenv("GSV_API_KEY"), help="Google Street View API key")
    ap.add_argument("--points", type=str, required=True, help="Path to GeoPackage/GeoJSON with points (must have 'road_angle')")
    ap.add_argument("--layer", type=str, default=None, help="Layer name if reading from GeoPackage")
    ap.add_argument("--pitch", type=float, default=6.0, help="Camera pitch (deg)")
    ap.add_argument("--fov", type=float, default=70.0, help="Field of view (deg)")
    ap.add_argument("--size", type=str, default="640x640", help="Image size for Static API")
    ap.add_argument("--save_qa", action="store_true", help="Save overlay QA images")
    ap.add_argument("--workers", type=int, default=1, help="Thread workers (keep 1 for single GPU)")
    args = ap.parse_args()

    if not args.api_key:
        raise SystemExit("Provide --api_key or set env GSV_API_KEY")

    # read your existing points (already created by road_network.py pipeline)
    gdf = gpd.read_file(args.points, layer=args.layer) if args.layer else gpd.read_file(args.points)
    if gdf.crs is None or gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(4326)

    if "road_angle" not in gdf.columns:
        raise SystemExit("The provided points file must contain 'road_angle'.")

    if "id" not in gdf.columns:
        gdf = gdf.reset_index(drop=True)
        gdf["id"] = np.arange(len(gdf), dtype=int)

    # run
    download_images_for_points(
        gdf=gdf,
        access_token=args.api_key,
        city=args.city,
        pitch_deg=args.pitch,
        fov_deg=args.fov,
        size=args.size,
        save_sample=args.save_qa,
        max_workers=args.workers
    )

if __name__ == "__main__":
    main()
