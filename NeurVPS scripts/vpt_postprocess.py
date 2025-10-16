#!/usr/bin/env python3
"""
vpt_postprocess.py — Convert NeurVPS 3D VP outputs (vpts_pd) to 2D pixel coordinates.

What it does:
- Reads a split list (e.g., valid.txt) to keep filename order.
- Loads each prediction /path/to/vpts/{index:06d}.npz and extracts `vpts_pd` (3x3).
- Applies SIHE-style projection:
    f = 1 / tan(FOV/2 in radians)
    x_px =  v[0]/v[2] * f * 256 + 256
    y_px = -v[1]/v[2] * f * 256 + 256
- Scales from 512×512 network space to your original width (org_width).
- Reorders VPs so: [v1_right, v2_left, v3_vertical].
- Writes:
    - a CSV summary
    - per-image JSON (with both 3D and ordered 2D VPs)
- (Optional) Draws simple overlays if --img-root and --overlays are given.

Examples
--------
python scripts/vpt_postprocess.py \
  --list     data/wflike/valid.txt \
  --preds    data/vpts \
  --outdir   data/vpts/json \
  --csv      data/vpts/preds.csv \
  --fov-deg  100 \
  --org-width 640 \
  --img-root data/images \
  --overlays data/vpts/overlays
"""
import argparse, os, json, csv, math
from pathlib import Path
import numpy as np

def to_pixel_new(v, focal_length):
    # project 3D direction v=(x,y,z) to 512x512 image plane (NeurVPS space)
    x =  v[0] / v[2] * focal_length * 256 + 256
    y = -v[1] / v[2] * focal_length * 256 + 256
    return x, y

def order_vpt(vps_2D, w):
    """
    Make v3 vertical (largest |y - h/2|), and v1/v2 the right/left horizontals.
    Assumes square canvas; uses w for both width & height.
    """
    h = w
    vps_2D = np.asarray(vps_2D, dtype=float)
    out = np.zeros((3,2), dtype=float)
    dy = np.abs(vps_2D[:,1] - h/2)

    # pick vertical (largest dy; if tie, pick the one farther in x)
    candidates = np.where(np.max(dy) - dy < 1)[0]
    if candidates.size == 1:
        v3_id = candidates[0]
    else:
        dxs = [abs(vps_2D[i,0] - w/2) for i in candidates]
        v3_id = candidates[int(np.argmax(dxs))]

    # remaining two are horizontals; right has larger x
    others = [i for i in (0,1,2) if i != v3_id]
    if vps_2D[others[0],0] > vps_2D[others[1],0]:
        v1, v2 = vps_2D[others[0]], vps_2D[others[1]]
    else:
        v1, v2 = vps_2D[others[1]], vps_2D[others[0]]
    v3 = vps_2D[v3_id]

    out[0] = v1  # right horizontal
    out[1] = v2  # left  horizontal
    out[2] = v3  # vertical
    return out

def transform_vpt(vpts_3d, fov_deg, org_w):
    """3D->2D in 512x512, then scale to org_w, then reorder."""
    f = 1.0 / math.tan(math.radians(fov_deg/2.0))
    pts = np.array([to_pixel_new(v, f) for v in vpts_3d], dtype=float)  # 512x512
    pts *= (org_w / 512.0)
    return order_vpt(pts, w=org_w)

def draw_overlays(rows, json_dir, img_root, overlays_dir, size=None):
    """Optional: draw simple markers on images using Pillow (if installed)."""
    try:
        from PIL import Image as I, ImageDraw as D
    except Exception:
        print("[warn] Pillow not installed; skipping overlays.")
        return
    overlays_dir.mkdir(parents=True, exist_ok=True)
    for _, rel, v1x, v1y, v2x, v2y, v3x, v3y in rows:
        stem = Path(rel).stem
        # Try to find the original image; we assume JPG with this name inside img_root
        # (Adjust this if your originals live elsewhere or use PNGs.)
        # As a fallback, try PNG too.
        for cand in [img_root/f"{stem}.jpg", img_root/f"{stem}.png"]:
            if cand.exists():
                img_path = cand
                break
        else:
            # if the resized PNG exists in the wflike set, use that instead
            # (this keeps the overlay consistent with eval input size)
            wf_png = Path(json_dir).parent.parent / "wflike" / "A" / f"{stem}.png"
            img_path = wf_png if wf_png.exists() else None

        if not img_path:
            print(f"[warn] cannot locate base image for {stem}; skipping overlay")
            continue

        img = I.open(img_path).convert("RGB")
        if size is not None:
            img = img.resize((size, size))
        W = img.size[0]
        draw = D.Draw(img)
        for (x,y) in [(v1x,v1y),(v2x,v2y),(v3x,v3y)]:
            if 0 <= x < W and 0 <= y < W:
                draw.ellipse((x-4,y-4,x+4,y+4), outline="white", width=2)
            cx, cy = max(0,min(W-1,int(x))), max(0,min(W-1,int(y)))
            draw.line((cx-6,cy, cx+6,cy), fill="red", width=1)
            draw.line((cx,cy-6, cx,cy+6), fill="red", width=1)
        out = overlays_dir / f"{stem}_overlay.png"
        img.save(out)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list",      required=True, help="Split file (e.g., valid.txt)")
    ap.add_argument("--preds",     required=True, help="Folder with 000xxx.npz predictions")
    ap.add_argument("--outdir",    required=True, help="Folder to write per-image JSON")
    ap.add_argument("--csv",       required=True, help="CSV summary to write")
    ap.add_argument("--fov-deg",   type=float, default=100.0, help="Field of view in degrees")
    ap.add_argument("--org-width", type=float, default=640.0, help="Output pixel width")
    ap.add_argument("--img-root",  default=None, help="Optional original images root (for overlays)")
    ap.add_argument("--overlays",  default=None, help="Optional folder to save overlay PNGs")
    ap.add_argument("--overlay-size", type=int, default=None, help="Force overlay canvas to NxN (e.g., 640)")
    args = ap.parse_args()

    split = Path(args.list)
    preds = Path(args.preds)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    with split.open() as f:
        rel_paths = [ln.strip() for ln in f if ln.strip()]
    print(f"[info] items: {len(rel_paths)}")

    rows = []
    for i, rel in enumerate(rel_paths):
        npz_path = preds / f"{i:06d}.npz"
        if not npz_path.exists():
            print("[warn] missing:", npz_path)
            continue
        d = np.load(npz_path)
        if "vpts_pd" not in d:
            print("[warn] no vpts_pd in", npz_path, "keys:", list(d.keys()))
            continue
        v3d = np.array(d["vpts_pd"], dtype=float)
        v2d = transform_vpt(v3d, args.fov_deg, args.org_width)

        # write JSON per image (use the stem from the list)
        stem = Path(rel).stem
        payload = {
            "image_rel": rel,
            "fov_deg": args.fov_deg,
            "org_width": args.org_width,
            "vpts_2d_ordered": {
                "v1_right":    v2d[0].tolist(),
                "v2_left":     v2d[1].tolist(),
                "v3_vertical": v2d[2].tolist(),
            },
            "vpts_3d": v3d.tolist(),
        }
        with (outdir / f"{stem}.json").open("w") as f:
            json.dump(payload, f)

        rows.append([i, rel, *v2d[0], *v2d[1], *v2d[2]])

    # CSV summary
    csv_path = Path(args.csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["idx","image_rel","v1x","v1y","v2x","v2y","v3x","v3y"])
        w.writerows(rows)
    print("[done] wrote", len(rows), "predictions")
    print("[csv]", csv_path)
    print("[json dir]", outdir)

    # overlays (optional)
    if args.img_root and args.overlays:
        draw_overlays(
            rows=rows,
            json_dir=outdir,
            img_root=Path(args.img_root),
            overlays_dir=Path(args.overlays),
            size=args.overlay_size,
        )
        print("[overlays]", args.overlays)

if __name__ == "__main__":
    main()
