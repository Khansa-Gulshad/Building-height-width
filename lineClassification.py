# -*-encoding:utf-8-*-

import copy
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

from lineDrawingConfig import *
from lineRefinement import lineRefinementWithVPT, pointOnLine

# --------------------------
# small helpers
# --------------------------

def _parse_label_list(config, key):
    """Parse comma-separated label list (e.g., '2,3') into a set of ints."""
    return set(int(x.strip()) for x in str(config["SEGMENTATION"][key]).split(",") if x.strip())

def _seg_dir(p1, p2):  # p1,p2 are (y,x)
    vx = p2[1] - p1[1]
    vy = p2[0] - p1[0]
    n = (vx * vx + vy * vy) ** 0.5 + 1e-8
    return np.array([vx / n, vy / n], float)  # [dx, dy]

def _angle_to_vp(p1, p2, vp_xy):
    """
    Angle between segment direction (p1->p2) and the image ray from the segment midpoint to the vanishing point.
    p1,p2: (y,x); vp_xy: (x,y). Returns degrees in [0, 90].
    """
    m = np.array([(p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0], float)  # (y,x)
    d = _seg_dir(p1, p2)                                                 # [dx, dy]
    v = np.array([vp_xy[0] - m[1], vp_xy[1] - m[0]], float)              # (x - mx, y - my)
    nv = np.linalg.norm(v) + 1e-8
    v /= nv
    cosang = np.clip(abs(d[0] * v[0] + d[1] * v[1]), 0.0, 1.0)
    return float(np.degrees(np.arccos(cosang)))

def _midpoint_in_building(seg_img, a, b, building_labels):
    """Check midpoint inside building mask (seg ids ∈ building_labels). a,b are (y,x)."""
    rows, cols = seg_img.shape
    my = int(round((a[0] + b[0]) / 2.0))
    mx = int(round((a[1] + b[1]) / 2.0))
    if not (0 <= my < rows and 0 <= mx < cols):
        return False
    return int(seg_img[my, mx]) in building_labels

# --------------------------
# unified VP-based classifier (new)
# --------------------------

def classify_lines(lines, scores, vps2d_dict, seg_img, config):
    """
    Classify raw LCNN segments into vertical and horizontal, filtered to building area.

    Args:
        lines:   (N,2,2) array-like of [[y1,x1],[y2,x2]]
        scores:  (N,) confidences
        vps2d_dict: {'v1_right': (x,y), 'v2_left': (x,y), 'v3_vertical': (x,y)}
        seg_img: HxW segmentation (int labels)
        config:  configparser

    Returns:
        dict with lists of indices:
          {'vertical_idx': [...], 'hori0_idx': [...], 'hori1_idx': [...], 'horizontal_idx': [...]}
    """
    min_score     = float(config["LINE_CLASSIFY"]["LineScore"])
    tol_vert_deg  = float(config["LINE_CLASSIFY"]["AngleThres"])
    tol_horiz_deg = float(config["LINE_CLASSIFY"].get("HorizAngleThres", tol_vert_deg))

    building_labels = _parse_label_list(config, "BuildingLabel")

    v1 = tuple(vps2d_dict["v1_right"])
    v2 = tuple(vps2d_dict["v2_left"])
    v3 = tuple(vps2d_dict["v3_vertical"])

    out = dict(vertical_idx=[], hori0_idx=[], hori1_idx=[], horizontal_idx=[])
    for i, (p1, p2) in enumerate(lines):
        if scores[i] < min_score:
            continue
        if not _midpoint_in_building(seg_img, p1, p2, building_labels):
            continue

        ang_v  = _angle_to_vp(p1, p2, v3)
        ang_h1 = _angle_to_vp(p1, p2, v1)
        ang_h2 = _angle_to_vp(p1, p2, v2)

        if ang_v <= tol_vert_deg:
            out["vertical_idx"].append(i)

        best_h = min(ang_h1, ang_h2)
        if best_h <= tol_horiz_deg:
            out["horizontal_idx"].append(i)
            if ang_h1 <= ang_h2:
                out["hori0_idx"].append(i)  # towards v1_right
            else:
                out["hori1_idx"].append(i)  # towards v2_left

    return out

# --------------------------
# original SIHE-style helpers (kept for compatibility)
# --------------------------

def classifyWithVPTs(n1, n2, vpt, config):
    """
    Original SIHE helper: classify a single line vs one VP by angle at the midpoint.
    n1,n2 are (y,x); vpt is (x,y).
    """
    t_angle = float(config["LINE_CLASSIFY"]["AngleThres"])
    p1 = np.array([n1[1], n1[0]], float)  # to (x,y)
    p2 = np.array([n2[1], n2[0]], float)
    mpt = [(p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0]
    d1  = p2 - p1
    d2  = np.array(vpt, float) - np.array(mpt, float)
    denom = (np.linalg.norm(d1) * np.linalg.norm(d2)) + 1e-8
    angle = np.degrees(np.arccos(np.clip(np.dot(d1, d2) / denom, -1.0, 1.0)))
    return bool(angle < t_angle or 180 - angle < t_angle)

def check_if_line_lies_in_building_area(seg_img, a, b, config)->bool:
    """
    Original SIHE check: sample around endpoints and midpoint to ensure we're in the building region.
    a,b are (y,x).
    """
    building_labels = _parse_label_list(config, "BuildingLabel")

    middle = (a + b) / 2.0
    norm_direction = (a - b) / (np.linalg.norm(a - b) + 1e-8)
    ppd_dir = np.asarray([norm_direction[1], -norm_direction[0]])

    ratio = 10
    ppd_dir = ratio * ppd_dir
    point_check_list = np.vstack([
        a,
        a - ppd_dir,
        a + ppd_dir,
        b,
        b - ppd_dir,
        b + ppd_dir,
        middle,
        middle - ppd_dir,
        middle + ppd_dir
    ])

    rows, cols = seg_img.shape
    total_num = 0
    local_num = 0
    flag = True
    for pcl in point_check_list:
        total_num += 1
        y_int = int(pcl[0] + 0.5)
        x_int = int(pcl[1] + 0.5)
        if x_int < 0 or x_int > cols - 1 or y_int < 0 or y_int > rows - 1:
            local_num += 1
            continue
        if int(seg_img[y_int, x_int]) in building_labels:
            local_num += 1
        if total_num % 3 == 0 and local_num == 0:
            flag = False
            break
        else:
            if total_num % 3 == 0:
                local_num = 0
    return flag

def check_if_bottom_lines(seg_img, a, b, config)->bool:
    """Is a horizontal line touching 'ground' labels around endpoints/midpoint?"""
    ground_labels = _parse_label_list(config, "GroundLabel")

    middle = (a + b) / 2.0
    norm_direction = (a - b) / (np.linalg.norm(a - b) + 1e-8)
    ppd_dir = np.asarray([norm_direction[1], -norm_direction[0]])

    ratio = 10
    ppd_dir = ratio * ppd_dir
    point_check_list = np.vstack([
        a, a - ppd_dir, a + ppd_dir,
        b, b - ppd_dir, b + ppd_dir,
        middle, middle - ppd_dir, middle + ppd_dir
    ])
    rows, cols = seg_img.shape

    for pcl in point_check_list:
        y_int = int(pcl[0] + 0.5)
        x_int = int(pcl[1] + 0.5)
        if x_int < 0 or x_int > cols - 1 or y_int < 0 or y_int > rows - 1:
            continue
        if int(seg_img[y_int, x_int]) in ground_labels:
            return True
    return False

def check_if_roof_lines(seg_img, a, b, config)->bool:
    """Is a horizontal line touching the 'sky' label around endpoints/midpoint?"""
    sky_label = int(config["SEGMENTATION"]["SkyLabel"])

    middle = (a + b) / 2.0
    norm_direction = (a - b) / (np.linalg.norm(a - b) + 1e-8)
    ppd_dir = np.asarray([norm_direction[1], -norm_direction[0]])

    ratio = 10
    ppd_dir = ratio * ppd_dir
    point_check_list = np.vstack([
        a, a - ppd_dir, a + ppd_dir,
        b, b - ppd_dir, b + ppd_dir,
        middle, middle - ppd_dir, middle + ppd_dir
    ])
    rows, cols = seg_img.shape

    for pcl in point_check_list:
        y_int = int(pcl[0] + 0.5)
        x_int = int(pcl[1] + 0.5)
        if x_int < 0 or x_int > cols - 1 or y_int < 0 or y_int > rows - 1:
            continue
        if int(seg_img[y_int, x_int]) == sky_label:
            return True
    return False

def lineCoeff(p1, p2):
    """Coefficients (A,B,C) of the infinite line through p1,p2 where p are (y,x)."""
    # convert to (x,y) for algebra if you want; here we keep (y,x) but formula accounts for it
    A = (p1[1] - p2[1])
    B = (p2[0] - p1[0])
    C = (p1[0] * p2[1] - p2[0] * p1[1])
    return A, B, -C

def intersection(L1, L2):
    """Intersection point (x,y) of two lines given by (A,B,C). Returns (x,y) or False if parallel."""
    D  = L1[0] * L2[1] - L1[1] * L2[0]
    Dx = L1[2] * L2[1] - L1[1] * L2[2]
    Dy = L1[0] * L2[2] - L1[2] * L2[0]
    if D != 0:
        x = Dx / D
        y = Dy / D
        return x, y
    else:
        return False

def dist_comparaison(first_line, second_line, thres):
    """
    Compare distance of two nearly-parallel lines; if close, merge onto first's direction.
    Lines are [[y,x],[y,x]].
    """
    a_0 = copy.deepcopy(first_line[0])
    b_0 = copy.deepcopy(first_line[1])
    a_1 = copy.deepcopy(second_line[0])
    b_1 = copy.deepcopy(second_line[1])

    pt_0 = pointOnLine(a_0, b_0, (a_1 + b_1) / 2.0)
    dist_0 = np.linalg.norm(pt_0 - (a_1 + b_1) / 2.0)

    pt_1 = pointOnLine(a_1, b_1, (a_0 + b_0) / 2.0)
    dist_1 = np.linalg.norm(pt_1 - (a_0 + b_0) / 2.0)

    if dist_0 < thres or dist_1 < thres:
        a_1_refine = pointOnLine(a_0, b_0, a_1)
        b_1_refine = pointOnLine(a_0, b_0, b_1)

        if a_0[0] > a_1_refine[0]:
            a_0 = a_1_refine
        if b_0[0] < b_1_refine[0]:
            b_0 = b_1_refine
        return True, [a_0, b_0]

    return False, first_line

# --------------------------
# main API (used by pipeline)
# --------------------------

def filter_lines_outof_building_ade20k(
    imgfile, lines, line_scores, segimg, vpts, config,
    use_vertical_vpt_only=0, verbose=True
):
    """
    Filter LCNN line segments by building mask, classify using VPs, and refine/merge verticals.
    Also (optionally) classify horizontals into bottom/roof using segmentation.

    Args:
        imgfile:  path to RGB (used only for visualization)
        lines:    (N,2,2) [[y1,x1],[y2,x2]] in original pixels
        line_scores: (N,)
        segimg:   HxW segmentation labels (int)
        vpts:     (3,2) np.array in (x,y), ordered [v1_right, v2_left, v3_vertical]
        config:   configparser
        use_vertical_vpt_only: if True, skip horizontals and only return verticals
        verbose:  draw debug plot

    Returns:
        vert_line_merge, hori0_lines, hori1_lines, bottom_lines, roof_lines
    """

    # build dict for classifier
    vps2d_dict = {
        "v1_right":    (float(vpts[0, 0]), float(vpts[0, 1])),
        "v2_left":     (float(vpts[1, 0]), float(vpts[1, 1])),
        "v3_vertical": (float(vpts[2, 0]), float(vpts[2, 1])),
    }

    idx = classify_lines(lines, line_scores, vps2d_dict, segimg, config)

    # buckets
    vert_indices = idx["vertical_idx"]
    hori0_lines = [lines[i] for i in idx["hori0_idx"]] if not use_vertical_vpt_only else []
    hori1_lines = [lines[i] for i in idx["hori1_idx"]] if not use_vertical_vpt_only else []

    # segment-based roof/bottom classification (optional)
    bottom_lines, roof_lines = [], []
    if not use_vertical_vpt_only:
        for line in (hori0_lines + hori1_lines):
            a, b = line
            if check_if_roof_lines(segimg, a, b, config):
                roof_lines.append(line)
            elif check_if_bottom_lines(segimg, a, b, config):
                bottom_lines.append(line)

    # --- vertical refinement & merge (same as SIHE) ---
    vert_lines = [lines[i] for i in vert_indices]

    # refine verticals toward v3 (pass (y,x) to refinement)
    vptz_yx = np.asarray([vpts[2, 1], vpts[2, 0]], float)
    vert_line_refine = []
    for line in vert_lines:
        a, b = line[0], line[1]
        line_ref = lineRefinementWithVPT([a, b], vptz_yx)
        vert_line_refine.append(line_ref)

    # merge close collinear verticals
    vert_line_merge = []
    for i in range(len(vert_line_refine)):
        li = vert_line_refine[i]
        lens = np.linalg.norm(li[0] - li[1])
        if (li[0][0] < 0 and li[1][0] < 0) or lens < 10:
            continue
        for j in range(i + 1, len(vert_line_refine)):
            lj = vert_line_refine[j]
            if lj[0][0] < 0 and lj[1][0] < 0:
                continue
            is_merging, li = dist_comparaison(li, lj, 5)
            if is_merging:
                vert_line_refine[j] = [np.asarray([-1, -1]), np.asarray([-1, -1])]
        a, b = li[0], li[1]
        if a[1] < 0:
            continue
        vert_line_merge.append(li)

    # --- debug viz ---
    if verbose:
        try:
            org_img = plt.imread(imgfile)
            plt.figure()
            plt.imshow(org_img)
            # verticals (blue)
            for a, b in vert_line_merge:
                plt.plot([a[1], b[1]], [a[0], b[0]], c="b", linewidth=2)
            # horizontals: v1 bucket (green), v2 bucket (red)
            for a, b in hori0_lines:
                plt.plot([a[1], b[1]], [a[0], b[0]], c="g", linewidth=1)
            for a, b in hori1_lines:
                plt.plot([a[1], b[1]], [a[0], b[0]], c="r", linewidth=1)
            # bottom (cyan), roof (magenta)
            for a, b in bottom_lines:
                plt.plot([a[1], b[1]], [a[0], b[0]], c="#00FFFF", linewidth=1)
            for a, b in roof_lines:
                plt.plot([a[1], b[1]], [a[0], b[0]], c="#FF00FF", linewidth=1)
            plt.axis("off"); plt.close()
        except Exception:
            plt.close()

    return vert_line_merge, hori0_lines, hori1_lines, bottom_lines, roof_lines

# --------------------------
# unchanged clustering helper
# --------------------------

def clausterLinesWithCenters(ht_set, config, using_height=False):
    """
    Group line segments (and heights) via DBSCAN.
    ht_set entries: [height, a, b, ...]
    """
    X = []
    if using_height:
        for ht, a, b, *_ in ht_set:
            X.append([(a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0, ht])
    else:
        for ht, a, b, *_ in ht_set:
            X.append([(a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0])
    X = np.asarray(X)

    max_DBSAN_dist = float(config["HEIGHT_MEAS"]["MaxDBSANDist"])
    try:
        clustering = DBSCAN(eps=max_DBSAN_dist, min_samples=1).fit(X)
    except Exception:
        print("!!! error in clustering: Expected 2D array, got 1D array instead. Return no results")
        return None

    clustered_lines = []
    max_val = int(np.max(clustering.labels_)) + 1
    for label in range(max_val):
        new_list = []
        new_ht_list = []
        for i in range(len(clustering.labels_)):
            if clustering.labels_[i] == label:
                new_list.append(ht_set[i])
                new_ht_list.append(ht_set[i][0])
        medi_val = float(np.median(np.asarray(new_ht_list)))  # median height
        mean_val = float(np.mean(np.asarray(new_ht_list)))    # mean height
        new_list.append(medi_val)
        new_list.append(mean_val)
        clustered_lines.append(new_list)

    return clustered_lines

