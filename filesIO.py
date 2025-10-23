import os, json
import numpy as np
import skimage.io
import matplotlib.pyplot as plt

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
    vps_2d: (3,2) array in pixels, w=h=image width.
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
    Returns (3,2) float array [v1_right, v2_left, v3_vertical] in PIXELS of your working image.
    - If *.json: reads the already-projected 2D VPs (recommended).
    - If *.npz : projects 3D directions using intrinsics (needs img_width & focal_length_px).
    """
    path = vpt_path.lower()
    if path.endswith(".json"):
        j = json.load(open(vpt_path, "r"))
        o = j["vpts_2d_ordered"]
        return np.array([o["v1_right"], o["v2_left"], o["v3_vertical"]], dtype=float)

    if path.endswith(".npz"):
        if img_width is None or focal_length_px is None:
            raise ValueError("img_width and focal_length_px are required for .npz -> 2D projection")
        d = np.load(vpt_path)
        if "vpts_pd" not in d:
            raise IOError(f"{vpt_path}: missing 'vpts_pd'")
        v3d = np.asarray(d["vpts_pd"], float)  # (3,3) directions
        v3d /= (np.linalg.norm(v3d, axis=1, keepdims=True) + 1e-9)

        # Intrinsics with principal point at image center (w=h)
        cx = cy = img_width / 2.0
        K = np.array([[focal_length_px, 0, cx],
                      [0, focal_length_px, cy],
                      [0, 0, 1]], float)

        v2d_h = (K @ v3d.T).T                   # (3,3) homogeneous
        v2d = v2d_h[:, :2] / (v2d_h[:, 2:]+1e-9)  # (3,2) pixels
        v2d = order_vpt(v2d, w=img_width)
        return v2d

    raise IOError(f"Unsupported vpt file: {vpt_path}")

def load_vps_2d(filename):
    """
    Load vanishing points (2D pixel coords) as a (3,2) array ordered:
      [v1_right, v2_left, v3_vertical].
    Supports:
      - JSON with {"vpts_2d_ordered": {"v1_right":[x,y],"v2_left":[x,y],"v3_vertical":[x,y]}}
      - NPZ with keys: 'vpts_re'  (already 2D)
      - NPZ with keys: 'vpts_2d'  (already 2D)
      - NPZ with 'vpts_pd' (3D dirs)  -> not converted here; prefer using your JSON.
    """
    ext = os.path.splitext(filename)[1].lower()
    if ext == ".json":
        d = json.load(open(filename, "r"))
        o = d.get("vpts_2d_ordered", {})
        arr = np.array([o["v1_right"], o["v2_left"], o["v3_vertical"]], dtype=float)
        return arr

    with np.load(filename) as npz:
        for k in ("vpts_re", "vpts_2d"):
            if k in npz:
                arr = np.array(npz[k], dtype=float)
                if arr.shape == (3,2):
                    return arr
        if "vpts_pd" in npz:
            # 3D directions present but not projected here (needs intrinsics & FOV).
            raise ValueError(
                f"{filename} contains 'vpts_pd' (3D). "
                "Use the JSON we wrote in /data/vpts/json or convert 3D→2D before calling load_vps_2d."
            )

    raise ValueError(f"Unrecognized VPS format: {filename}")

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
      - NPZ with key 'seg'
      - PNG/JPG (assumed single-channel label image)
    """
    ext = os.path.splitext(filename)[1].lower()
    if ext == ".npz":
        with np.load(filename) as npz:
            if "seg" in npz:
                return np.array(npz["seg"])
            # fallback keys if your generator used a different name
            for k in ("label", "labels", "mask"):
                if k in npz:
                    return np.array(npz[k])
        raise ValueError(f"Unrecognized seg npz format: {filename}")
    else:
        arr = skimage.io.imread(filename)
        # if RGB, take first channel (you can customize if needed)
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
