# -*- encoding:utf-8 -*-
import os, csv
import numpy as np

from horizontalClassification import filter_horizontal_lines, is_bottom_like, is_roof_like
from horizontalLines import horizontalLinePostprocess
from skimage.io import imread
from lineDrawingConfig import PLTOPTS, colors_tables  # reuse your palette
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import skimage.io

def _cluster_rows_1d(yvals, eps=20.0):
    """Simple 1D clustering by sorted gaps; eps in pixels."""
    idx = np.argsort(yvals)
    groups, cur = [], [idx[0]] if len(idx) else []
    for i in range(1, len(idx)):
        if abs(yvals[idx[i]] - yvals[idx[i-1]]) <= eps:
            cur.append(idx[i])
        else:
            groups.append(cur); cur = [idx[i]]
    if cur: groups.append(cur)
    return groups

def _draw_width_overlay(img_path, seg_img, base_segments, widths_m, groups, out_path):
    im = skimage.io.imread(img_path); H, W = im.shape[:2]
    fig = plt.figure(figsize=(W/100, H/100), dpi=100)
    ax = plt.axes([0,0,1,1]); ax.imshow(im); ax.imshow(seg_img, alpha=0.35); ax.axis("off")

    legends = []
    for gi, gidx in enumerate(groups):
        color = colors_tables[gi % len(colors_tables)]
        vals  = [widths_m[k] for k in gidx]
        med   = float(np.median(vals)) if len(vals) else 0.0
        mean  = float(np.mean(vals)) if len(vals) else 0.0
        handle = None
        for k in gidx:
            L,R = base_segments[k]
            ax.plot([L[1], R[1]], [L[0], R[0]], c=color, linewidth=2, zorder=3)
            ax.scatter([L[1], R[1]], [L[0], R[0]], **PLTOPTS)
        if len(gidx):
            txt = f"avg_width = {mean:.3f}m, median_width = {med:.3f}m"
            handle, = ax.plot([], [], c=color, linewidth=4, label=txt)
            legends.append(handle)

    if legends:
        ax.legend(handles=legends, loc="lower left", framealpha=0.85)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path.replace(".png",".svg"), bbox_inches="tight", pad_inches=0)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


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
    L = np.cross(pB, pC)
    n = np.linalg.norm(L[:2])
    return L / (n if n>0 else 1.0)

def point_to_line_dist(px_xy, line_ABC):
    """Perpendicular pixel distance from point [x,y] to line Ax+By+C=0."""
    A,B,C = line_ABC
    x,y = px_xy
    return abs(A*x + B*y + C) / max(1e-9, np.hypot(A,B))

def _get_tau_h_deg(config):
    # Prefer your horizontal-specific key; fall back to classifier angle if absent.
    if "LINE" in config and "HORIZ_TOL_DEG" in config["LINE"]:
        return float(config["LINE"]["HORIZ_TOL_DEG"])
    return float(config["LINE_CLASSIFY"]["AngleThres"])

def pick_facade_horizontals(
    lines, scores, segimg, vpts_xy, config,
    min_width_px=20.0,
    avoid_ground_px=20,   # vertical buffer above ground mask
    avoid_sky_px=10,      # vertical buffer below sky mask
):
    """
    Classify, snap, and extend horizontal façade bands.
    - keeps segments fully inside the building mask
    - rejects ones touching (or very near) ground/sky
    - no preference for bottom-like lines
    """
    # 1) LCNN -> horizontal candidates by VP angle
    hori0, hori1 = filter_horizontal_lines(
        imgfile=None, lines=lines, line_scores=scores,
        segimg=segimg, vpts=vpts_xy, config=config,
        return_roof_base=False, verbose=False
    )

    # 2) snap + extend within building mask
    tau_h = _get_tau_h_deg(config)
    pB_rc = np.array([vpts_xy[0,1], vpts_xy[0,0]], float)
    pC_rc = np.array([vpts_xy[1,1], vpts_xy[1,0]], float)

    seg = segimg
    H, W = seg.shape
    building_label = int(config["SEGMENTATION"]["BuildingLabel"])
    sky_label      = int(config["SEGMENTATION"]["SkyLabel"])
    ground_labels  = [int(x) for x in str(config["SEGMENTATION"]["GroundLabel"]).split(',')]

    # quick col-wise buffers: nearest ground/sky row per column
    ground_rows = np.full(W, -1, int)
    sky_rows    = np.full(W,  H, int)
    for c in range(W):
        g = np.where(np.isin(seg[:, c], ground_labels))[0]
        s = np.where(seg[:, c] == sky_label)[0]
        if g.size: ground_rows[c] = g.max()           # highest ground pixel
        if s.size: sky_rows[c]    = s.min()           # lowest sky pixel

    def clear_of_ground_sky(L, R):
        # require both endpoints and midpoint to be (i) building,
        # (ii) at least avoid_* pixels away from ground/sky
        for P in (L, R, 0.5*(L+R)):
            r = int(round(P[0])); c = int(round(P[1]))
            if not (0 <= r < H and 0 <= c < W): return False
            if seg[r, c] != building_label:          return False
            if ground_rows[c] >= 0 and r > ground_rows[c] - avoid_ground_px: return False
            if sky_rows[c]    <  H and r < sky_rows[c]    + avoid_sky_px:    return False
        return True

    cand = []
    if len(hori0):
        cand += horizontalLinePostprocess(hori0, seg, pB_rc, pC_rc, tau_h, config)
    if len(hori1):
        cand += horizontalLinePostprocess(hori1, seg, pB_rc, pC_rc, tau_h, config)

    # min length in pixels
    cand = [(L, R) for (L, R) in cand
            if np.linalg.norm((R - L)[[1,0]]) >= min_width_px]

    # remove segments touching ground/sky (use buffers)
    facade_segments = [(L, R) for (L, R) in cand if clear_of_ground_sky(L, R)]

    return facade_segments

def width_from_segment(L_rc, R_rc, horizon_ABC, zc_m):
    """Meters from a single horizontal ground segment."""
    L_xy = np.array([L_rc[1], L_rc[0]], float)
    R_xy = np.array([R_rc[1], R_rc[0]], float)

    w_px = np.linalg.norm(R_xy - L_xy)
    if w_px < 1e-6:
        return 0.0

    # local meters/px via distance to horizon at endpoints
    dL = point_to_line_dist(L_xy, horizon_ABC)
    dR = point_to_line_dist(R_xy, horizon_ABC)
    sL = zc_m / max(dL, 1e-6)
    sR = zc_m / max(dR, 1e-6)
    m_per_px = 0.5*(sL + sR)

    return w_px * m_per_px

# --- main API ---

def compute_widths_config(fname_dict, seg_img, lines, scores, vpts, config,
                          intrins=None, img_size=None, pitch_deg=None,
                          verbose=False, out_csv=None, out_img_dir=None):
    """
    Inputs
      - fname_dict: {"img": <path>} (only used for reporting/saving)
      - seg_img: HxW label map (numpy)
      - lines, scores: LCNN outputs in [row,col] order
      - vpts: (3,2) array in [x,y] -> [pB′, pC′, pA′]
      - config: parsed INI (ConfigParser)
      - intrins: optional K; if None, built from FoV in config (kept in meta)
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
            H, W = imread(fname_dict["img"]).shape[:2]
        else:
            H, W = seg_img.shape[:2]
    else:
        W, H = img_size

    if intrins is None:
        fov = float(config["STREET_VIEW"]["HVFoV"])
        K = K_from_fov(W, H, fov)
    else:
        K = intrins

    # 1) horizon from detected pB′, pC′
    horizon = horizon_from_vpts(vpts)

    # 2) pick good base horizontals
    base_segments = pick_facade_horizontals(lines, scores, seg_img, vpts, config)

    # 3) measure widths
    widths_m = []
    for (L, R) in base_segments:
        w = width_from_segment(L, R, horizon, zc)
        if w > 0:
            widths_m.append(w)
    # 3b) group horizontals by mid-row (≈ DBSCAN on 1-D y)
    ymid = np.array([0.5*(L[0]+R[0]) for (L,R) in base_segments], float)
    eps = float(config["HEIGHT_MEAS"].get("MaxDBSANDist", "50"))  # reuse ε
    groups = _cluster_rows_1d(ymid, eps=eps) if len(ymid) else []

    # 4) optional CSV
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
    # 5) optional overlay
    if verbose and out_img_dir and fname_dict.get("img"):
        stem = os.path.splitext(os.path.basename(fname_dict["img"]))[0]
        out_img = os.path.join(out_img_dir, f"{stem}_wdr.png")
        _draw_width_overlay(fname_dict["img"], seg_img, base_segments, widths_m, groups, out_img)
        
    meta = {
        "W": W, "H": H,
        "K": K, "horizon_ABC": horizon,
        "camera_height_m": zc,
        "pitch_deg": pitch_deg,
        "groups": groups
    }
    return widths_m, base_segments, meta
