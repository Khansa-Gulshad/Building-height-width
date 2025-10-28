# -*- encoding:utf-8 -*-
import os, csv
import numpy as np

from horizontalClassification import filter_horizontal_lines, is_bottom_like, is_roof_like
from horizontalLines import horizontalLinePostprocess
from lineClassification import lineCoeff, intersection
from skimage.io import imread

# --- small helpers ---

def K_from_fov(W, H, fov_deg):
    f = (W/2.0)/np.tan(np.deg2rad(fov_deg/2.0))
    return np.array([[f,0,W/2.0],
                     [0,f,H/2.0],
                     [0,0,1.0]], float)

def horizon_from_vpts(vpts_xy):
    """Return homogeneous line (A,B,C) for the horizon passing pB′=vpts[0], pC′=vpts[1].
       vpts_xy is (3,2) in [x,y] order."""
    pB = np.array([vpts_xy[0,0], vpts_xy[0,1], 1.0])
    pC = np.array([vpts_xy[1,0], vpts_xy[1,1], 1.0])
    L = np.cross(pB, pC)                      # homogeneous line
    n = np.linalg.norm(L[:2])
    return L / (n if n>0 else 1.0)

def point_to_line_dist(px_xy, line_ABC):
    """Perpendicular pixel distance from point [x,y] to line Ax+By+C=0."""
    A,B,C = line_ABC
    x,y = px_xy
    return abs(A*x + B*y + C) / max(1e-9, np.hypot(A,B))

def pick_base_horizontals(lines, scores, segimg, vpts_xy, config):
    """Classify, refine, extend, then keep likely ground/base horizontals."""
    # 1) Classify by horizontal VPs (upstream; gives raw candidate horizontals)
    #    We only need the raw lines; refinement/extension will be done next.
    #    (If you want, you can use return_roof_base=True and filter by bottom directly.)
    hori0, hori1 = filter_horizontal_lines(
        imgfile=None, lines=lines, line_scores=scores,
        segimg=segimg, vpts=vpts_xy, config=config,
        return_roof_base=False, verbose=False
    )

    # 2) Snap + extend inside building for both HVP families
    tau_h = float(config["LINE"]["HORIZ_TOL_DEG"])
    pB = np.array([vpts_xy[0,1], vpts_xy[0,0]], float)   # [row, col] order for refinement helpers
    pC = np.array([vpts_xy[1,1], vpts_xy[1,0]], float)
    base_segments = []

    if len(hori0):
        base_segments += horizontalLinePostprocess(hori0, segimg, pB, pC, tau_h, config)
    if len(hori1):
        base_segments += horizontalLinePostprocess(hori1, segimg, pB, pC, tau_h, config)

    # 3) Filter to those likely at the bottom (touching ground) for façade width
    #    (Re-use your optional helpers; if missing, keep all and let user post-filter.)
    ground_labels = [int(x) for x in str(config["SEGMENTATION"]["GroundLabel"]).split(',')]
    bottom_like = []
    for (L, R) in base_segments:
        # use the original boolean from horizontalClassification.py if imported,
        # otherwise implement a simple check around (L+R)/2 for ground adjacency below the line.
        if is_bottom_like(segimg, L, R, ground_labels):
            bottom_like.append((L, R))

    # Fallback: if none labelled bottom-like, keep everything we have
    return bottom_like if bottom_like else base_segments

def width_from_segment(L_rc, R_rc, horizon_ABC, zc_m):
    """Meters from a single horizontal ground segment."""
    # swap [row,col]->[x,y]
    L_xy = np.array([L_rc[1], L_rc[0]], float)
    R_xy = np.array([R_rc[1], R_rc[0]], float)

    # pixel length
    w_px = np.linalg.norm(R_xy - L_xy)
    if w_px < 1e-6:
        return 0.0

    # local meters/px using distance to horizon (average at endpoints)
    dL = point_to_line_dist(L_xy, horizon_ABC)
    dR = point_to_line_dist(R_xy, horizon_ABC)
    sL = zc_m / max(dL, 1e-6)
    sR = zc_m / max(dR, 1e-6)
    m_per_px = 0.5*(sL + sR)

    return w_px * m_per_px

# --- main API ---

def compute_widths_config(fname_dict, seg_img, lines, scores, vpts, config,
                          intrins=None, img_size=None, pitch_deg=None,
                          verbose=False, out_csv=None):
    """
    Inputs
      - fname_dict: {"img": <path>} (only used for reporting/saving)
      - seg_img: HxW label map (numpy)
      - lines, scores: LCNN outputs in [row,col] order
      - vpts: (3,2) array in [x,y] -> [pB′, pC′, pA′]
      - config: parsed INI (ConfigParser)
      - intrins: optional K; if None, built from FoV in config
      - img_size: (W,H); if None, read from image (if path given) or seg_img
      - pitch_deg: optional (not required here)
    Returns
      widths_m: list of floats
      base_segments: list of [(L_rc, R_rc), ...]
      meta: dict with bookkeeping
    """

    zc = float(config["STREET_VIEW"]["CameraHeight"])
    if img_size is None:
        if fname_dict.get("img") and os.path.exists(fname_dict["img"]):
            H,W = imread(fname_dict["img"]).shape[:2]
        else:
            H,W = seg_img.shape[:2]
    else:
        W,H = img_size

    if intrins is None:
        fov = float(config["STREET_VIEW"]["HVFoV"])
        K = K_from_fov(W, H, fov)
    else:
        K = intrins

    # 1) horizon
    horizon = horizon_from_vpts(vpts)  # uses detected pB′, pC′

    # 2) get good base horizontals on building & on ground
    base_segments = pick_base_horizontals(lines, scores, seg_img, vpts, config)

    # 3) measure widths
    widths_m = []
    for (L, R) in base_segments:
        w = width_from_segment(L, R, horizon, zc)
        if w > 0:
            widths_m.append(w)

    # 4) save optional CSV
    if out_csv:
        os.makedirs(os.path.dirname(out_csv), exist_ok=True)
        write_header = not os.path.exists(out_csv)
        with open(out_csv, "a", newline="") as f:
            wcsv = csv.writer(f)
            if write_header:
                wcsv.writerow(["image", "segment_idx", "width_m",
                               "L_row","L_col","R_row","R_col"])
            for i, ((L,R), wval) in enumerate(zip(base_segments, widths_m)):
                wcsv.writerow([
                    fname_dict.get("img",""), i, float(wval),
                    float(L[0]), float(L[1]), float(R[0]), float(R[1])
                ])

    meta = {
        "W": W, "H": H,
        "K": K, "horizon_ABC": horizon,
        "camera_height_m": zc,
        "pitch_deg": pitch_deg
    }
    return widths_m, base_segments, meta
