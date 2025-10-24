import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import modules.config as cfg  # single source of truth for paths

# optional blur (avoid hard dep on scipy)
try:
    from scipy.ndimage import gaussian_filter  # type: ignore
except Exception:
    gaussian_filter = None
    
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
    [140,  90,  40],  # 0 ground (brown)
    [180, 180, 180],  # 1 building (gray)
    [ 90, 180, 255],  # 2 sky (light blue)
], dtype=np.uint8)


def _ensure_dir(p): os.makedirs(p, exist_ok=True)

def save_rgb(city: str, image_id: str, img_pil, out_root=None):
    if out_root is None: out_root = cfg.PROJECT_DIR
    out_dir = os.path.join(out_root, cfg.city_to_dir(city), "save_rgb", "imgs")
    _ensure_dir(out_dir)
    path = os.path.join(out_dir, f"{image_id}.jpg")
    img_pil.save(path, quality=95)   # overwrites if exists
    return path

def colorize_full(mask_full: np.ndarray) -> np.ndarray:
    idx = np.clip(mask_full, 0, FULL_PALETTE.shape[0]-1)
    return FULL_PALETTE[idx]

def remap_to_three(mask_full: np.ndarray) -> np.ndarray:
    m = np.zeros_like(mask_full, dtype=np.uint8)           # 0=ground
    for i in SOURCE_BUILDING_IDS: m[mask_full == i] = 1     # 1=building
    for i in SOURCE_SKY_IDS:      m[mask_full == i] = 2     # 2=sky
    return m

def colorize_three(mask3: np.ndarray) -> np.ndarray:
    idx = np.clip(mask3, 0, 2)
    return PALETTE_3[idx]

def save_full_color(city: str, image_id: str, mask_full: np.ndarray, out_root=None):
    if out_root is None: out_root = cfg.PROJECT_DIR
    out_dir = os.path.join(out_root, cfg.city_to_dir(city), "seg_full_vis")
    _ensure_dir(out_dir)
    path = os.path.join(out_dir, f"{image_id}.png")
    Image.fromarray(colorize_full(mask_full)).save(path)    # overwrites
    return path

def save_three_color(city: str, image_id: str, mask3: np.ndarray, out_root=None):
    if out_root is None: out_root = cfg.PROJECT_DIR
    out_dir = os.path.join(out_root, cfg.city_to_dir(city), "seg_3class_vis")
    _ensure_dir(out_dir)
    path = os.path.join(out_dir, f"{image_id}.png")
    Image.fromarray(colorize_three(mask3)).save(path)       # overwrites
    return path

def save_three_class_npz(city: str, image_id: str, mask3: np.ndarray, out_root=None):
    if out_root is None: out_root = cfg.PROJECT_DIR
    out_dir = os.path.join(out_root, cfg.city_to_dir(city), "seg")
    _ensure_dir(out_dir)
    path = os.path.join(out_dir, f"{image_id}_seg.npz")
    np.savez(path, seg=mask3.astype(np.uint8))              # overwrites
    return path

def save_all_products(city: str, image_id: str, img_pil, mask_full_tensor, out_root=None):
    # enforce 640×640 for both image and labels
    if img_pil.size != (640, 640):
        img_pil = img_pil.resize((640, 640), Image.BILINEAR)

    # to numpy H×W uint8 (label map 0..18)
    if hasattr(mask_full_tensor, "detach"):
        mask_full = mask_full_tensor.detach().cpu().numpy().astype(np.uint8)
    else:
        mask_full = np.asarray(mask_full_tensor, dtype=np.uint8)

    if mask_full.shape[::-1] != (640, 640):
        mask_full = np.array(Image.fromarray(mask_full, mode="L")
                             .resize((640, 640), Image.NEAREST), dtype=np.uint8)

    # 1) RGB
    save_rgb(city, image_id, img_pil, out_root)

    # 2) full-class segmented (color)
    save_full_color(city, image_id, mask_full, out_root)

    # 3) three-class: color + npz labels
    mask3 = remap_to_three(mask_full)          # 0 ground, 1 building, 2 sky
    save_three_color(city, image_id, mask3, out_root)
    save_three_class_npz(city, image_id, mask3, out_root)

