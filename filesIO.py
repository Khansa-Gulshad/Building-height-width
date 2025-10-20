import os, json
import numpy as np
import skimage.io
import matplotlib.pyplot as plt

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
