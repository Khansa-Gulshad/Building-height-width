# filesIO.py
import os, json
import numpy as np
import skimage.io

def _strip_to_float(s, default=None):
    """Parse floats even if the value has inline comments like '10 ; note'."""
    if s is None:
        return default
    s = str(s).split(';')[0].split('#')[0].strip()
    return float(s)

def order_vpt(vps_2d, w):
    """
    Reorder 3 vanishing points so:
      v3 = vertical (|y - h/2| largest), v1 = right-H, v2 = left-H.
    Assumes square canvas (h = w); if your images are not square,
    pass their width to approximate. vps_2d shape (3,2) in pixels.
    """
    vps_2d = np.asarray(vps_2d, float)
    h = w
    dy = np.abs(vps_2d[:, 1] - h / 2.0)
    v3_id = int(np.argmax(dy))
    others = [i for i in (0, 1, 2) if i != v3_id]
    # right has larger x
    if vps_2d[others[0], 0] >= vps_2d[others[1], 0]:
        v1 = vps_2d[others[0]]
        v2 = vps_2d[others[1]]
    else:
        v1 = vps_2d[others[1]]
        v2 = vps_2d[others[0]]
    v3 = vps_2d[v3_id]
    return np.vstack([v1, v2, v3])

def load_vps_2d(vpt_path, img_width=None, focal_length_px=None):
    """
    Return (3,2) array [v1_right, v2_left, v3_vertical] in PIXELS of your working image.

    Supports:
      - JSON with:
          {"vpts_2d_ordered": {"v1_right":[x,y],"v2_left":[x,y],"v3_vertical":[x,y]}}
      - NPZ with keys: 'vpts_2d' or 'vpts_re' (already 2D) -> returned as-is
      - NPZ with key 'vpts_pd' (3D directions) -> projected to 2D via K
        (requires img_width and focal_length_px)

    Notes:
      * JSON is the recommended source (already ordered + scaled).
      * For 'vpts_2d'/'vpts_re' we assume they are already in the desired order.
      * For 'vpts_pd' we project and then order with order_vpt().
    """
    ext = os.path.splitext(vpt_path)[1].lower()

    # --- JSON path (preferred)
    if ext == ".json":
        j = json.load(open(vpt_path, "r"))
        o = j["vpts_2d_ordered"]
        return np.array([o["v1_right"], o["v2_left"], o["v3_vertical"]], dtype=float)

    # --- NPZ path
    if ext == ".npz":
        with np.load(vpt_path) as npz:
            # Already 2D?
            for k in ("vpts_2d", "vpts_re"):
                if k in npz:
                    arr = np.array(npz[k], dtype=float)
                    if arr.shape == (3, 2):
                        return arr
            # 3D directions -> project
            if "vpts_pd" in npz:
                if img_width is None or focal_length_px is None:
                    raise ValueError(
                        "For NPZ with 'vpts_pd', you must pass img_width and focal_length_px"
                    )
                v3d = np.asarray(npz["vpts_pd"], float)  # (3,3)
                v3d /= (np.linalg.norm(v3d, axis=1, keepdims=True) + 1e-9)
                cx = cy = img_width / 2.0
                K = np.array([[focal_length_px, 0, cx],
                              [0, focal_length_px, cy],
                              [0, 0, 1]], float)
                v2d_h = (K @ v3d.T).T
                v2d = v2d_h[:, :2] / (v2d_h[:, 2:]+1e-9)
                v2d = order_vpt(v2d, w=img_width)
                return v2d

    raise ValueError(f"Unrecognized or unsupported VPS file: {vpt_path}")

def load_line_array(filename):
    """
    Load LCNN line segments and scores.
    Returns:
      lines: (N,2,2) in [y,x] order; scores: (N,)
    """
    with np.load(filename) as npz:
        if "nlines" in npz and "nscores" in npz:
            return npz["nlines"], npz["nscores"]
        # fallback to demo-style keys
        if "lines" in npz and "score" in npz:
            return npz["lines"], npz["score"]
    raise ValueError(f"Unrecognized lines format: {filename}")

def load_seg_array(filename):
    """
    Load semantic segmentation label map as HxW int array.
    Supports:
      - NPZ with key 'seg' (or fallback: 'label'/'labels'/'mask')
      - PNG/JPG (assumed single-channel label image)
    """
    ext = os.path.splitext(filename)[1].lower()
    if ext == ".npz":
        with np.load(filename) as npz:
            if "seg" in npz:
                return np.array(npz["seg"])
            for k in ("label", "labels", "mask"):
                if k in npz:
                    return np.array(npz[k])
        raise ValueError(f"Unrecognized seg npz format: {filename}")
    else:
        arr = skimage.io.imread(filename)
        # if RGB, take first channel (customize if needed)
        if arr.ndim == 3:
            arr = arr[..., 0]
        return arr.astype(np.int32)

def load_zgts(filename):
    """
    Load ground-truth height map if present (HxW, float or int).
    NPZ with key 'height'.
    """
    with np.load(filename) as npz:
        if "height" in npz:
            return np.array(npz["height"])
    raise ValueError(f"Unrecognized zgt format: {filename}")
