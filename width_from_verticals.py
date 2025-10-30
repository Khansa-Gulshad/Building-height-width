# -*- coding: utf-8 -*-
import os, csv
import numpy as np
import skimage.io
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------- IO: load verticals saved by height step ----------
def load_height_verticals(stem, metrics_dir):
    """
    Return list of segments [[r1,c1],[r2,c2]] (float, RC order) or [] if not found.
    Adjust keys to match your *_vertical_refs.npz.
    """
    f = os.path.join(metrics_dir, f"{stem}_vertical_refs.npz")
    if not os.path.exists(f): return []
    z = np.load(f, allow_pickle=True)
    # Inspect once if unsure: print("keys:", z.files)
    for key in ("vlines_rc","vline_pairs_rc","lines","lines_rc"):
        if key in z:
            arr = z[key].astype(float)  # (N,2,2) in [row, col]
            return [arr[i] for i in range(arr.shape[0])]
    return []

# ---------- utilities ----------
def _seg_label_at(seg, p_rc):
    r = int(round(p_rc[0])); c = int(round(p_rc[1]))
    H,W = seg.shape
    if r < 0 or r >= H or c < 0 or c >= W: return -999
    return int(seg[r,c])

def _line_len_px(seg_rc):
    a,b = seg_rc[0], seg_rc[1]
    # measure in (x,y) == (col,row)
    return float(np.linalg.norm((b - a)[[1,0]]))

def _angle_to_vertical_deg(seg_rc):
    a,b = seg_rc[0], seg_rc[1]
    v = (b - a)[[1,0]]  # (x,y)
    n = np.linalg.norm(v)
    if n < 1e-9: return 90.0
    v /= n
    # vertical axis in (x,y) is (0,-1) if rows increase downward; use |y| large means vertical
    cosang = abs(v[1])  # projection onto y-axis
    cosang = np.clip(cosang, -1.0, 1.0)
    ang = np.degrees(np.arccos(cosang))  # 0 = perfect vertical
    return ang

def _overlap_with_building(seg_img, seg_rc, building_label, nsamples=50):
    a,b = seg_rc[0], seg_rc[1]
    hits = 0
    for t in np.linspace(0,1,nsamples):
        p = (1-t)*a + t*b
        if _seg_label_at(seg_img,p) == building_label:
            hits += 1
    return hits / float(nsamples)

# ---------- pick façade edges from height verticals ----------
def filter_vertical_facade_lines(vlines, seg_img, tau_vert_deg, min_len_px, min_overlap, building_label):
    out = []
    for l in vlines:
        if _line_len_px(l) < min_len_px: continue
        if _angle_to_vertical_deg(l) > tau_vert_deg: continue
        if _overlap_with_building(seg_img, l, building_label) < min_overlap: continue
        out.append(l.astype(float))
    return out

def pick_left_right_edges(vlines, seg_img):
    """
    Choose two façade edges: the 'leftmost' and 'rightmost' by column of the line midpoint.
    If multiple cluster on a side, take the longest there.
    """
    if not vlines: return None, None
    mids = np.array([0.5*(v[0]+v[1]) for v in vlines])    # RC
    cols = mids[:,1]
    # left side: smallest col; right side: largest col
    li = int(np.argmin(cols)); ri = int(np.argmax(cols))
    left = vlines[li]; right = vlines[ri]
    if li == ri:
        # degenerate: pick the farthest other by |col - cols[li]|
        j = int(np.argmax(np.abs(cols - cols[li])))
        if cols[j] < cols[li]: left, right = vlines[j], vlines[li]
        else:                  left, right = vlines[li], vlines[j]
    # break ties within each side by taking the longer one near that side
    def _pick_side(target_col, cand):
        idx = np.argsort(np.abs(cols - target_col))[:3]  # 3 nearest by column
        if len(idx)==0: return None
        best = max([vlines[k] for k in idx], key=_line_len_px)
        return best
    left  = _pick_side(np.min(cols), vlines)  or left
    right = _pick_side(np.max(cols), vlines)  or right
    return left, right

# ---------- find foot point (lowest façade contact) ----------
def footpoint_on_facade_or_ground(line_rc, seg_img, building_label, ground_labels):
    """
    Return the lowest in-image point along this line that is still on the façade boundary.
    Strategy: sample many points on the segment; keep those labeled building; take the one with max row.
    If none found, return the lower endpoint that lies on/near building boundary.
    """
    a,b = line_rc[0], line_rc[1]
    H,W = seg_img.shape
    samples = []
    for t in np.linspace(0,1,200):
        p = (1-t)*a + t*b
        r = int(round(p[0])); c=int(round(p[1]))
        if r<0 or r>=H or c<0 or c>=W: continue
        if seg_img[r,c] == building_label:
            samples.append(p)
    if samples:
        # pick with largest row (closest to ground in image coords)
        return max(samples, key=lambda q: q[0])
    # fallback: choose the endpoint that is inside building or closest to it
    for P in (a,b):
        if _seg_label_at(seg_img,P) == building_label:
            return P.copy()
    return 0.5*(a+b)  # worst-case

# ---------- pixel -> ground via pitch + K ----------
def ipm_project_to_ground(u, v, K, pitch_deg, cam_h_m):
    """
    Project pixel (u=x, v=y) to ground plane Y=0.
    Camera at (0,h,0); +y is up. Assumes roll already corrected.
    Uses rotation about camera x-axis by -pitch.
    """
    # build normalized camera ray
    fx, fy = K[0,0], K[1,1]
    cx, cy = K[0,2], K[1,2]
    # image y points down; convert to camera with y up: use (u - cx, cy - v)
    x = (u - cx) / fx
    y = (cy - v) / fy
    z = 1.0
    rc = np.array([x, y, z], dtype=float)
    rc /= np.linalg.norm(rc) + 1e-12

    # rotate by -pitch around x
    th = np.deg2rad(float(pitch_deg))
    c, s = np.cos(-th), np.sin(-th)
    Rx = np.array([[1,0,0],[0,c,-s],[0,s,c]], float)
    rw = Rx @ rc

    # intersect C + λ rw with Y=0; C=(0,h,0), rw=(rx, ry, rz)
    ry = rw[1]
    if abs(ry) < 1e-9:
        return None  # ray parallel to ground
    lam = cam_h_m / max(1e-12, ry)
    X = lam * rw[0]
    Z = lam * rw[2]
    return np.array([X, 0.0, Z], float)

# ---------- K from FoV ----------
def K_from_fov(W,H,fov_deg):
    f = (W/2.0)/np.tan(np.deg2rad(fov_deg/2.0))
    return np.array([[f,0,W/2.0],[0,f,H/2.0],[0,0,1.0]], float)

# ---------- overlay ----------
def draw_overlay(img_path, seg_img, left_rc, right_rc, P1, P2, out_path):
    im = skimage.io.imread(img_path); H,W = im.shape[:2]
    fig = plt.figure(figsize=(W/100, H/100), dpi=100)
    ax = plt.axes([0,0,1,1]); ax.imshow(im); ax.axis("off")
    # verticals
    for l, col in ((left_rc,"tab:blue"), (right_rc,"tab:orange")):
        a,b = l[0], l[1]
        ax.plot([a[1], b[1]],[a[0], b[0]], c=col, lw=2)
        ax.scatter([a[1], b[1]],[a[0], b[0]], s=12, c=col)
    # footpoints
    def _footcolor(c): return {"tab:blue":"cyan","tab:orange":"yellow"}[c]
    for (l,col,P) in ((left_rc,"tab:blue",P1),(right_rc,"tab:orange",P2)):
        fp = footpoint_on_facade_or_ground(l, seg_img, 1, [0])  # labels not used here visually
        ax.scatter([fp[1]],[fp[0]], s=40, c=_footcolor(col), marker='x', linewidths=2)
    # save
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path.replace(".png",".svg"), bbox_inches="tight", pad_inches=0)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0); plt.close(fig)

# ---------- main per-image ----------
def compute_width_from_height_verticals(stem, paths, cfg, out_csv=None, out_dir=None, verbose=False):
    """
    paths = dict(
      img = "/w/PROJ/.../imgs/<stem>.jpg",
      seg = "/w/PROJ/.../seg/<stem>[_seg].npz",
      metrics_dir = "/w/PROJ/.../metrics"
    )
    """
    img_path = paths["img"]
    seg_img  = np.load(paths["seg"])["seg"]
    vlines   = load_height_verticals(stem, paths["metrics_dir"])

    if len(vlines) == 0:
        return None

    building_label = int(cfg["SEGMENTATION"]["BuildingLabel"])
    ground_labels  = [int(x) for x in str(cfg["SEGMENTATION"]["GroundLabel"]).split(',')]
    tau_vert_deg   = float(cfg["LINE"].get("VERT_TOL_DEG","10"))
    min_len_px     = 30.0
    min_overlap    = 0.5

    vfac = filter_vertical_facade_lines(vlines, seg_img, tau_vert_deg, min_len_px, min_overlap, building_label)
    if len(vfac) < 2:
        return None

    left, right = pick_left_right_edges(vfac, seg_img)
    if left is None or right is None:
        return None

    # footpoints
    fl = footpoint_on_facade_or_ground(left,  seg_img, building_label, ground_labels)
    fr = footpoint_on_facade_or_ground(right, seg_img, building_label, ground_labels)

    # intrinsics & pitch
    if "K" in paths:
        K = paths["K"]
    else:
        H,W = seg_img.shape[:2]
        fov = float(cfg["STREET_VIEW"]["HVFoV"])
        K = K_from_fov(W,H,fov)
    pitch_deg = float(cfg["STREET_VIEW"]["Pitch"])
    cam_h     = float(cfg["STREET_VIEW"]["CameraHeight"])

    # IPM to ground
    P1 = ipm_project_to_ground(fl[1], fl[0], K, pitch_deg, cam_h)
    P2 = ipm_project_to_ground(fr[1], fr[0], K, pitch_deg, cam_h)
    if P1 is None or P2 is None:
        return None
    width_m = float(np.linalg.norm(P1 - P2))

    # CSV
    if out_csv:
        os.makedirs(os.path.dirname(out_csv), exist_ok=True)
        write_header = not os.path.exists(out_csv)
        with open(out_csv, "a", newline="") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(["image","width_m","fl_row","fl_col","fr_row","fr_col"])
            w.writerow([img_path, width_m, float(fl[0]), float(fl[1]), float(fr[0]), float(fr[1])])

    # overlay
    if verbose and out_dir:
        os.makedirs(out_dir, exist_ok=True)
        out_img = os.path.join(out_dir, f"{stem}_wipm.png")
        draw_overlay(img_path, seg_img, left, right, P1, P2, out_img)

    return dict(width_m=width_m, left=left, right=right, fl=fl, fr=fr, P1=P1, P2=P2)
