import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import modules.config as cfg  # single source of truth for paths

# ---------- ID remap (Cityscapes-ish) ----------
SOURCE_BUILDING_IDS = {2, 3}      # building, wall -> building
SOURCE_SKY_IDS      = {10}
SOURCE_GROUND_IDS   = {0, 1, 9}   # road, sidewalk, terrain

FULL_PALETTE = np.array([
    [128,  64, 128],  # 0 road
    [244,  35, 232],  # 1 sidewalk
    [ 70,  70,  70],  # 2 building
    [102, 102, 156],  # 3 wall
    [190, 153, 153],  # 4 fence
    [153, 153, 153],  # 5 pole
    [250, 170,  30],  # 6 traffic light
    [220, 220,   0],  # 7 traffic sign
    [  0, 255,   0],  # 8 vegetation
    [152, 251, 152],  # 9 terrain
    [ 70, 130, 180],  # 10 sky
    [220,  20,  60],  # 11 person
    [255,   0,   0],  # 12 rider
    [  0,   0, 142],  # 13 car
    [  0,   0,  70],  # 14 truck
    [  0,  60, 100],  # 15 bus
    [  0,  80, 100],  # 16 train
    [  0,   0, 230],  # 17 motorcycle
    [119,  11,  32],  # 18 bicycle
], dtype=np.uint8)

# >>> ADD THIS (you referenced PALETTE_3 below) <<<
PALETTE_3 = np.array([
    [0,   0,   0],    # 0 (unused)
    [180, 180, 180],  # 1 building
    [90,  180, 255],  # 2 sky
    [140, 90,  40],   # 3 ground
], dtype=np.uint8)

def colorize_full(mask_full: np.ndarray) -> np.ndarray:
    idx = np.clip(mask_full, 0, FULL_PALETTE.shape[0]-1)
    return FULL_PALETTE[idx]


def save_full_color(city: str, image_id: str, mask_full: np.ndarray, out_root: str | None = None):
    if out_root is None:
        out_root = cfg.PROJECT_DIR
    out_dir = os.path.join(out_root, cfg.city_to_dir(city), "seg_full_vis")
    _ensure_dir(out_dir)
    path = os.path.join(out_dir, f"{image_id}.png")
    Image.fromarray(colorize_full(mask_full)).save(path)
    return path

def remap_to_three(mask_full: np.ndarray) -> np.ndarray:
    m = np.zeros_like(mask_full, dtype=np.uint8)
    for i in SOURCE_BUILDING_IDS: m[mask_full == i] = 1
    for i in SOURCE_SKY_IDS:      m[mask_full == i] = 2
    for i in SOURCE_GROUND_IDS:   m[mask_full == i] = 3
    return m

def colorize_three(mask3: np.ndarray) -> np.ndarray:
    idx = np.clip(mask3, 0, 3)
    return PALETTE_3[idx]
    
def save_three_color(city: str, image_id: str, mask3: np.ndarray, out_root: str | None = None):
    """
    Save a human-friendly colorized 3-class PNG:
    1=building (gray), 2=sky (light blue), 3=ground (brown).
    """
    if out_root is None:
        out_root = cfg.PROJECT_DIR
    out_dir = os.path.join(out_root, cfg.city_to_dir(city), "seg_3class_vis")
    _ensure_dir(out_dir)
    colored = colorize_three(mask3)
    path = os.path.join(out_dir, f"{image_id}.png")
    Image.fromarray(colored).save(path)
    return path

def overlay_rgb_with_mask(rgb: np.ndarray, mask3: np.ndarray, alpha=0.4) -> np.ndarray:
    if not isinstance(rgb, np.ndarray):
        rgb = np.array(rgb)
    rgb = rgb.astype(np.float32)
    color = colorize_three(mask3).astype(np.float32)
    out = (1.0 - alpha) * rgb + alpha * color
    return np.clip(out, 0, 255).astype(np.uint8)

def _ensure_dir(p: str): os.makedirs(p, exist_ok=True)

def save_three_class_mask(city: str, image_id: str, mask3: np.ndarray, out_root: str | None = None):
    if out_root is None:
        out_root = cfg.PROJECT_DIR
    out_dir = os.path.join(out_root, cfg.city_to_dir(city), "seg_3class")
    _ensure_dir(out_dir)
    path = os.path.join(out_dir, f"{image_id}.png")
    Image.fromarray(mask3, mode="L").save(path)
    return path

def save_overlay(city: str, image_id: str, rgb: np.ndarray, mask3: np.ndarray, out_root: str | None = None):
    if out_root is None:
        out_root = cfg.PROJECT_DIR
    qa_dir = os.path.join(out_root, cfg.city_to_dir(city), "seg_qa")
    _ensure_dir(qa_dir)
    ov = overlay_rgb_with_mask(rgb, mask3, alpha=0.4)
    path = os.path.join(qa_dir, f"{image_id}_overlay.jpg")
    Image.fromarray(ov).save(path, quality=92)
    return path

def visualize_results(city, image_id, image, segmentation_3class, num, out_root: str | None = None):
    if out_root is None:
        out_root = cfg.PROJECT_DIR
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(9, 4), sharey=True)
    ax1.imshow(image); ax1.set_title("Image"); ax1.axis("off")
    ax2.imshow(colorize_three(segmentation_3class)); ax2.set_title("Segmentation (3 classes)"); ax2.axis("off")
    out_dir = os.path.join(out_root, cfg.city_to_dir(city), "sample_images")
    _ensure_dir(out_dir)
    fig.savefig(os.path.join(out_dir, f"{image_id}-{num}.png"), bbox_inches='tight', dpi=110)
    plt.close(fig)

def save_images(city, image_id, images, pickles, out_root: str | None = None):
    if out_root is None:
        out_root = cfg.PROJECT_DIR
    for i, (img, mask3) in enumerate(zip(images, pickles), start=1):
        img_np = np.array(img) if not isinstance(img, np.ndarray) else img
        visualize_results(city, image_id, img_np, mask3, i, out_root=out_root)
