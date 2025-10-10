import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# ---------- ID remap: adapt if your upstream model uses different IDs ----------
# Your current palette suggests Cityscapes-like indexing:
# 0: road, 1: sidewalk, 2: building, 9: terrain, 10: sky
SOURCE_BUILDING_IDS = {2, 3}         # building + wall -> building plane
SOURCE_SKY_IDS      = {10}
SOURCE_GROUND_IDS   = {0, 1, 9}      # road, sidewalk, terrain (add more if needed)

# ---------- Colors for 3-class visualization ----------
# 0 is reserved (unused), 1=building, 2=sky, 3=ground
PALETTE_3 = np.array([
    [0,   0,   0],      # 0 (unused)
    [180, 180, 180],    # 1 building (gray)
    [ 90, 180, 255],    # 2 sky (light blue)
    [140,  90,  40],    # 3 ground (brown)
], dtype=np.uint8)

def remap_to_three(mask_full: np.ndarray) -> np.ndarray:
    """
    Convert a multi-class mask (H,W) to 3-class uint8 mask with values in {0,1,2,3}.
    1=building, 2=sky, 3=ground. 0 stays unused/background if any pixels don't match.
    """
    m = np.zeros_like(mask_full, dtype=np.uint8)
    for i in SOURCE_BUILDING_IDS:
        m[mask_full == i] = 1
    for i in SOURCE_SKY_IDS:
        m[mask_full == i] = 2
    for i in SOURCE_GROUND_IDS:
        m[mask_full == i] = 3
    return m

def colorize_three(mask3: np.ndarray) -> np.ndarray:
    """
    Map 3-class mask -> RGB for visualization.
    """
    idx = np.clip(mask3, 0, 3)
    return PALETTE_3[idx]

def overlay_rgb_with_mask(rgb: np.ndarray, mask3: np.ndarray, alpha=0.4) -> np.ndarray:
    """
    Blend colorized 3-class mask over the RGB image for QA visualization.
    """
    rgb = np.asarray(rgb, dtype=np.float32)
    color = colorize_three(mask3).astype(np.float32)
    out = (1.0 - alpha) * rgb + alpha * color
    return np.clip(out, 0, 255).astype(np.uint8)

# ---------- I/O helpers ----------
def _ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def save_three_class_mask(city: str, image_id: str, mask3: np.ndarray, out_root=PROJECT_DIR):
    out_dir = os.path.join(out_root, city_to_dir(city), "seg_3class")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{image_id}.png")
    Image.fromarray(mask3, mode="L").save(path)
    return path

def save_overlay(city: str, image_id: str, rgb: np.ndarray, mask3: np.ndarray, out_root=PROJECT_DIR):
    qa_dir = os.path.join(out_root, city_to_dir(city), "seg_qa")
    os.makedirs(qa_dir, exist_ok=True)
    _ensure_dir(qa_dir)
    ov = overlay_rgb_with_mask(rgb, mask3, alpha=0.4)
    path = os.path.join(qa_dir, f"{image_id}_overlay.jpg")
    Image.fromarray(ov).save(path, quality=92)
    return path

# ---------- Back-compat friendly APIs ----------
def visualize_results(city, image_id, image, segmentation_3class, num):
    """
    Simple side-by-side visualization (image + 3-class colorized).
    """
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(9, 4), sharey=True)
    ax1.imshow(image)
    ax1.set_title("Image")
    ax1.axis("off")

    seg_color = colorize_three(segmentation_3class)
    ax2.imshow(seg_color)
    ax2.set_title("Segmentation (3 classes)")
    ax2.axis("off")

    out_dir = f"/mnt/project/pt01183/results/{city}/sample_images"
    _ensure_dir(out_dir)
    fig.savefig(os.path.join(out_dir, f"{image_id}-{num}.png"),
                bbox_inches='tight', dpi=110)
    plt.close(fig)

def save_images(city, image_id, images, pickles):
    """
    Keep the original signature but assume 'pickles' are already 3-class masks.
    Saves the side-by-side visualization only (used for quick QA batches).
    """
    for i, (img, mask3) in enumerate(zip(images, pickles), start=1):
        # 'img' can be PIL Image or np.ndarray
        img_np = np.array(img) if not isinstance(img, np.ndarray) else img
        visualize_results(city, image_id, img_np, mask3, i)

