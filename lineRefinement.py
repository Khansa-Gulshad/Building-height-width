# -*-encoding:utf-8-*-

# -*-encoding:utf-8-*-

import numpy as np
import matplotlib.pyplot as plt
import skimage
from filesIO import _strip_to_float
from lineDrawingConfig import *  # assumes PLTOPTS, config sections exist

# ---------------------------
# helpers
# ---------------------------

def _parse_label_list(cfg, key):
    """Parse comma-separated label list (e.g. '2,3') -> set[int]."""
    raw = str(cfg["SEGMENTATION"][key])
    return set(int(x.strip()) for x in raw.split(",") if x.strip() != "")

def _get_edge_thres(cfg):
    val = str(cfg["LINE_REFINE"]["Edge_Thres"]).split(",")[0]
    return int(val)

def pointOnLine(a, b, p):
    """
    Project point p onto the infinite line through segment (a-b).
    All points are (y,x).
    """
    l2 = _strip_to_float(np.sum((a - b) ** 2))
    if l2 == 0:
        return a.copy()
    t = float(np.sum((p - a) * (b - a))) / l2
    return a + t * (b - a)

def lineRefinementWithVPT(line, vpt_yx):
    """
    Slightly rotate the line around its midpoint so its direction aligns with
    the ray from midpoint to a vanishing point. vpt_yx is (y,x).
    """
    a = line[0].astype(float)
    b = line[1].astype(float)
    mpt = (a + b) / 2.0
    line[0] = pointOnLine(vpt_yx, mpt, a)
    line[1] = pointOnLine(vpt_yx, mpt, b)
    return line

# ---------------------------
# vertical extension (kept, fixed)
# ---------------------------

def extendLines(pt1, pt2, segmt, config):
    """
    Extend a vertical-ish line within the building mask:
      - upward until hitting sky
      - downward until just above ground
    Returns two endpoints (y,x); empty lists if invalid.
    """
    sky_label       = int(config["SEGMENTATION"]["SkyLabel"])
    building_labels = _parse_label_list(config, "BuildingLabel")
    ground_labels   = _parse_label_list(config, "GroundLabel")
    edge_thres      = _get_edge_thres(config)

    # order so pt_up has smaller y
    if pt1[0] > pt2[0]:
        pt_up  = pt2.astype(float)
        pt_down= pt1.astype(float)
    else:
        pt_up  = pt1.astype(float)
        pt_down= pt2.astype(float)

    if np.linalg.norm(pt_down - pt_up) == 0:
        return [], []

    direction   = (pt_down - pt_up) / (np.linalg.norm(pt_down - pt_up) + 1e-8)
    pt_up_end   = pt_up.copy()
    pt_down_end = pt_down.copy()
    pt_middle   = (pt_up + pt_down) / 2.0

    rows, cols = segmt.shape
    # clamp initial points inside image
    pt_up_end[0]   = np.clip(pt_up_end[0],   0, rows - 2)
    pt_up_end[1]   = np.clip(pt_up_end[1],   0, cols - 2)
    pt_down_end[0] = np.clip(pt_down_end[0], 0, rows - 2)
    pt_down_end[1] = np.clip(pt_down_end[1], 0, cols - 2)

    if pt_middle[0] >= rows - 1 or pt_middle[1] >= cols - 1:
        return [], []

    # all three anchor pixels must be building
    def _in_build(yx):
        y, x = int(yx[0] + 0.5), int(yx[1] + 0.5)
        if y < 0 or y >= rows or x < 0 or x >= cols:
            return False
        return int(segmt[y, x]) in building_labels

    if not (_in_build(pt_up_end) and _in_build(pt_down_end) and _in_build(pt_middle)):
        return [], []

    # extend upward until hitting sky
    while True:
        nxt = pt_up_end - direction
        y, x = int(nxt[0] + 0.5), int(nxt[1] + 0.5)
        if y < 0 or y >= rows - 1 or x < 0 or x >= cols - 1:
            break
        if int(segmt[y, x]) == sky_label:
            break
        pt_up_end = nxt
    # step back one to stay within building
    pt_up_end = pt_up_end + direction

    # extend downward until just before ground (and not leaving building -> ground)
    out_of_building = False
    while True:
        nxt = pt_down_end + direction
        y, x = int(nxt[0] + 0.5), int(nxt[1] + 0.5)
        if y < 0 or y >= rows - 1 or x < 0 or x >= cols - 1:
            break
        lab = int(segmt[y, x])
        if lab not in building_labels and lab not in ground_labels:
            out_of_building = True
        else:
            if lab in building_labels:
                out_of_building = False
        if lab in ground_labels and not out_of_building:
            # step back one to stay above ground inside building
            pt_down_end = pt_down_end
            break
        pt_down_end = nxt

    # stay away from image edges
    if (pt_up_end[0] > rows - 1 - edge_thres or pt_up_end[0] < edge_thres or
        pt_up_end[1] < edge_thres or pt_up_end[1] > cols - 1 - edge_thres or
        pt_down_end[0] > rows - 1 - edge_thres or pt_down_end[0] < edge_thres or
        pt_down_end[1] < edge_thres or pt_down_end[1] > cols - 1 - edge_thres):
        return [], []

    return pt_up_end, pt_down_end

def verticalLineExtending(img_name, vertical_lines, segimg, vptz_yx, config, verbose=True):
    """
    Refine each vertical line w.r.t. vertical VP and extend to roof/ground.
    vptz_yx: vertical VP as (y,x).
    """
    if verbose:
        plt.close()
        org_img = skimage.io.imread(img_name)
        plt.imshow(org_img)

    extd_lines = []
    for line in vertical_lines:
        line = lineRefinementWithVPT(line, vptz_yx)  # refine orientation
        a, b = line[0], line[1]
        extd_a, extd_b = extendLines(a, b, segimg, config)
        if len(extd_a) == 0 or len(extd_b) == 0:
            continue
        extd_lines.append([extd_a, extd_b])

        if verbose:
            plt.plot([extd_a[1], extd_b[1]], [extd_a[0], extd_b[0]], c='y', linewidth=2)
            try:
                plt.scatter(extd_a[1], extd_a[0], **PLTOPTS)
                plt.scatter(extd_b[1], extd_b[0], **PLTOPTS)
            except Exception:
                pass

    if verbose:
        plt.close()

    return extd_lines

def verticalLineExtendingWithBRLines(img_name, vertical_lines, roof_lines, bottom_lines, segimg, config, verbose=True):
    """
    (Unchanged behavior) Extend vertical lines using explicit roof/bottom horizontals.
    """
    if verbose:
        org_img = skimage.io.imread(img_name)
        plt.close()
        plt.imshow(org_img)

    rows, cols = segimg.shape
    extd_lines = []
    for vl in vertical_lines:
        pt_rl = []
        # closest intersection to roof
        for rl in roof_lines:
            vl_direction = vl[0] - vl[1]
            rl_direction = rl[0] - rl[1]
            A = np.transpose(np.vstack([vl_direction, -rl_direction]))
            b = np.transpose(rl[0] - vl[0])
            try:
                x = np.matmul(np.linalg.inv(A), b)
            except np.linalg.LinAlgError:
                continue
            pt = vl_direction * x[0] + vl[0]
            pt = (pt + 0.5).astype(int)
            if abs(x[0]) > 2:
                continue
            if pt[0] < 10 or pt[0] > rows - 10 or pt[1] < 10 or pt[1] > cols - 10:
                continue
            if np.std(segimg[pt[0]-10:pt[0]+10, pt[1]-10:pt[1]+10]) == 0:
                continue
            if len(pt_rl) == 0 or pt_rl[0] > pt[0]:
                pt_rl = pt
        if len(pt_rl) == 0:
            continue

        pt_bl = []
        for bl in bottom_lines:
            vl_direction = vl[0] - vl[1]
            bl_direction = bl[0] - bl[1]
            A = np.transpose(np.vstack([vl_direction, -bl_direction]))
            b = np.transpose(bl[0] - vl[0])
            try:
                x = np.matmul(np.linalg.inv(A), b)
            except np.linalg.LinAlgError:
                continue
            pt = vl_direction * x[0] + vl[0]
            pt = (pt + 0.5).astype(int)
            if abs(x[0]) > 2:
                continue
            if pt[0] < 10 or pt[0] > rows - 10 or pt[1] < 10 or pt[1] > cols - 10:
                continue
            if np.std(segimg[pt[0]-10:pt[0]+10, pt[1]-10:pt[1]+10]) == 0:
                continue
            if len(pt_bl) == 0 or pt_bl[0] < pt[0]:
                pt_bl = pt
        if len(pt_bl) == 0:
            continue

        extd_lines.append([pt_rl, pt_bl])

    return extd_lines

# ---------------------------
# NEW: horizontal extension (for width)
# ---------------------------

def refine_horizontal_with_best_vp(line, vpt1_xy, vpt2_xy):
    """
    Refine a horizontal candidate toward the closer horizontal VP.
    vpt*_xy are (x,y); this converts them to (y,x) for lineRefinementWithVPT.
    """
    a, b = line
    # choose VP by smaller angle at midpoint
    def _ang_to_vp(vp_xy):
        m  = (a + b) / 2.0
        d  = (b - a) / (np.linalg.norm(b - a) + 1e-8)
        v  = np.array([vp_xy[1] - m[0], vp_xy[0] - m[1]], float)  # (y,x) diff
        v /= (np.linalg.norm(v) + 1e-8)
        cosang = np.clip(abs(d[0] * v[0] + d[1] * v[1]), 0.0, 1.0)
        return _strip_to_float(np.degrees(np.arccos(cosang)))
    ang1 = _ang_to_vp(vpt1_xy)
    ang2 = _ang_to_vp(vpt2_xy)
    vpt_best_yx = np.array([vpt1_xy[1], vpt1_xy[0]]) if ang1 <= ang2 else np.array([vpt2_xy[1], vpt2_xy[0]])
    return lineRefinementWithVPT([a.copy(), b.copy()], vpt_best_yx)

def extendHorizontalWithinBuilding(pt1, pt2, segmt, config):
    """
    Extend a horizontal-ish segment left/right within the building mask.
    Returns two endpoints (y,x); empty lists if invalid.
    """
    building_labels = _parse_label_list(config, "BuildingLabel")
    edge_thres      = _get_edge_thres(config)

    a = pt1.astype(float)
    b = pt2.astype(float)

    d = (b - a)
    if abs(d[1]) < abs(d[0]):  # not horizontal enough (dx << dy)
        return [], []
    direction = d / (np.linalg.norm(d) + 1e-8)

    rows, cols = segmt.shape

    def _in_build(yx):
        y, x = int(yx[0] + 0.5), int(yx[1] + 0.5)
        if y < 0 or y >= rows or x < 0 or x >= cols:
            return False
        return int(segmt[y, x]) in building_labels

    # start from sorted endpoints (left/right by x)
    if a[1] <= b[1]:
        left, right = a.copy(), b.copy()
    else:
        left, right = b.copy(), a.copy()

    # extend left
    while True:
        nxt = left - direction
        y, x = int(nxt[0] + 0.5), int(nxt[1] + 0.5)
        if y < 0 or y >= rows or x < 0 or x >= cols:
            break
        if not _in_build(nxt):
            break
        left = nxt
    # step back one pixel
    left = left + direction

    # extend right
    while True:
        nxt = right + direction
        y, x = int(nxt[0] + 0.5), int(nxt[1] + 0.5)
        if y < 0 or y >= rows or x < 0 or x >= cols:
            break
        if not _in_build(nxt):
            break
        right = nxt
    right = right - direction

    # keep away from edges
    if (left[0] < edge_thres or left[0] > rows - 1 - edge_thres or
        right[0] < edge_thres or right[0] > rows - 1 - edge_thres or
        left[1] < edge_thres or left[1] > cols - 1 - edge_thres or
        right[1] < edge_thres or right[1] > cols - 1 - edge_thres):
        return [], []

    return left, right

def horizontalLineExtending(img_name, horizontal_lines, segimg, vpt1_xy, vpt2_xy, config, refine_with_vp=True, verbose=True):
    """
    Take horizontal candidates (e.g., 'bottom_lines') and extend them across the façade.
    vpt1_xy, vpt2_xy are the two horizontal vanishing points as (x,y).
    Returns list of [left_endpoint, right_endpoint].
    """
    if verbose:
        plt.close()
        org_img = skimage.io.imread(img_name)
        plt.imshow(org_img)

    extd = []
    for line in horizontal_lines:
        a, b = line[0].astype(float), line[1].astype(float)
        if refine_with_vp:
            line_ref = refine_horizontal_with_best_vp([a, b], vpt1_xy, vpt2_xy)
            a, b = line_ref[0], line_ref[1]
        L, R = extendHorizontalWithinBuilding(a, b, segimg, config)
        if len(L) == 0 or len(R) == 0:
            continue
        extd.append([L, R])

        if verbose:
            plt.plot([L[1], R[1]], [L[0], R[0]], c='c', linewidth=2)
            try:
                plt.scatter(L[1], L[0], **PLTOPTS)
                plt.scatter(R[1], R[0], **PLTOPTS)
            except Exception:
                pass

    if verbose:
        plt.close()

    return extd
