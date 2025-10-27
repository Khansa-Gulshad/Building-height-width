# -*- encoding:utf-8 -*-
# horizontalClassification.py
import copy
import numpy as np

# ---------- Angle test vs horizontal VPs ----------
def classifyWithHVPs(n1, n2, vpt_xy, angle_thres_deg):
    """
    Return True if the line segment (n1,n2) is aligned with the given horizontal VP.
    n1, n2 are [row, col]; vpt_xy is [x, y].
    Angle is measured between segment direction and ray (midpoint -> vpt).
    """
    # swap [row, col] -> [x, y] for consistent VP math
    p1 = np.array([n1[1], n1[0]], dtype=float)
    p2 = np.array([n2[1], n2[0]], dtype=float)
    mpt = 0.5 * (p1 + p2)

    d_line = p2 - p1
    d_vp   = vpt_xy - mpt

    # guard
    if np.linalg.norm(d_line) < 1e-6 or np.linalg.norm(d_vp) < 1e-6:
        return False

    # angle in degrees
    cosv = np.dot(d_line, d_vp) / (np.linalg.norm(d_line) * np.linalg.norm(d_vp))
    cosv = np.clip(cosv, -1.0, 1.0)
    ang  = np.degrees(np.arccos(cosv))
    return (ang < angle_thres_deg) or (180.0 - ang < angle_thres_deg)


# ---------- Building mask check (same logic you use for verticals) ----------
def check_if_line_lies_in_building_area(seg_img, a, b, building_label, ratio=10):
    """
    Sample 9 points around the segment (endpoints/midpoint +/- perpendicular offsets).
    Require that in each triplet at least one is building. Returns True/False.
    a, b are [row, col].
    """
    middle = (a + b) / 2.0
    v = (a - b).astype(float)
    if np.linalg.norm(v) < 1e-6:
        return False
    v = v / np.linalg.norm(v)
    # perpendicular in [row, col]
    ppd = np.array([v[1], -v[0]], dtype=float) * ratio

    pts = np.vstack([
        a, a - ppd, a + ppd,
        b, b - ppd, b + ppd,
        middle, middle - ppd, middle + ppd
    ])

    rows, cols = seg_img.shape
    total = 0; local = 0
    for i, pcl in enumerate(pts, start=1):
        r = int(pcl[0] + 0.5); c = int(pcl[1] + 0.5)
        if 0 <= r < rows and 0 <= c < cols and seg_img[r, c] == building_label:
            local += 1
        total += 1
        # every 3 samples (center and its ± offsets) must see building at least once
        if (total % 3) == 0:
            if local == 0:
                return False
            local = 0
    return True


# ---------- Optional roof/base tests (if you want them now) ----------
# def is_bottom_like(seg_img, a, b, ground_labels, ratio=10):
#     """True if any of the 9 sample points hits ground."""
#     middle = (a + b) / 2.0
#     v = (a - b).astype(float); v /= (np.linalg.norm(v) + 1e-9)
#     ppd = np.array([v[1], -v[0]]) * ratio
#     pts = np.vstack([a, a-ppd, a+ppd, b, b-ppd, b+ppd, middle, middle-ppd, middle+ppd])
#     rows, cols = seg_img.shape
#     for pcl in pts:
#         r = int(pcl[0] + 0.5); c = int(pcl[1] + 0.5)
#         if 0 <= r < rows and 0 <= c < cols and seg_img[r, c] in ground_labels:
#             return True
#     return False

# def is_roof_like(seg_img, a, b, sky_label, ratio=10):
#     """True if any of the 9 sample points hits sky."""
#     middle = (a + b) / 2.0
#     v = (a - b).astype(float); v /= (np.linalg.norm(v) + 1e-9)
#     ppd = np.array([v[1], -v[0]]) * ratio
#     pts = np.vstack([a, a-ppd, a+ppd, b, b-ppd, b+ppd, middle, middle-ppd, middle+ppd])
#     rows, cols = seg_img.shape
#     for pcl in pts:
#         r = int(pcl[0] + 0.5); c = int(pcl[1] + 0.5)
#         if 0 <= r < rows and 0 <= c < cols and seg_img[r, c] == sky_label:
#             return True
#     return False


# ---------- Main: filter + classify horizontals ----------
def filter_horizontal_lines(imgfile, lines, line_scores, segimg, vpts, config,
                            return_roof_base=False, verbose=False):
    """
    Classify raw LCNN lines into horizontal groups tied to the two horizontal VPs.

    Inputs
    - lines: list of [a, b], with a,b in [row, col]
    - line_scores: list of detector scores (same length as lines)
    - segimg: HxW label map
    - vpts: np.array shape (3,2) in [x, y] order => [pB′, pC′, pA′]
    - config: dict-like; uses
        LINE_CLASSIFY.AngleThres
        LINE_CLASSIFY.LineScore
        SEGMENTATION.BuildingLabel / SkyLabel / GroundLabel
    - return_roof_base: also classify into roof/bottom sets (optional)

    Returns
    - hori0_lines: list of [a, b] aligned to vpts[0] (pB′)
    - hori1_lines: list of [a, b] aligned to vpts[1] (pC′)
    - (optionally) roof_lines, bottom_lines
    """
    angle_thres = float(config["LINE_CLASSIFY"]["AngleThres"])
    score_thres = float(config["LINE_CLASSIFY"]["LineScore"])
    building_label = int(config["SEGMENTATION"]["BuildingLabel"])
    sky_label = int(config["SEGMENTATION"]["SkyLabel"])
    ground_labels = [int(x) for x in str(config["SEGMENTATION"]["GroundLabel"]).split(',')]

    pB_xy = np.asarray([vpts[0, 0], vpts[0, 1]], dtype=float)
    pC_xy = np.asarray([vpts[1, 0], vpts[1, 1]], dtype=float)

    hori0_lines, hori1_lines = [], []
    roof_lines, bottom_lines = [], []

    for (a, b), s in zip(lines, line_scores):
        if s < score_thres:
            continue
        if not check_if_line_lies_in_building_area(segimg, a, b, building_label):
            continue

        is_h0 = classifyWithHVPs(a, b, pB_xy, angle_thres)
        is_h1 = classifyWithHVPs(a, b, pC_xy, angle_thres)

        # If it matches both (rare), keep the closer one by angle:
        if is_h0 and is_h1:
            # compute exact angles and pick min
            def ang_to(vp_xy):
                p1 = np.array([a[1], a[0]], dtype=float)
                p2 = np.array([b[1], b[0]], dtype=float)
                m  = 0.5*(p1+p2)
                dL = p2 - p1
                dV = vp_xy - m
                c = np.dot(dL, dV) / ((np.linalg.norm(dL) + 1e-9) * (np.linalg.norm(dV) + 1e-9))
                return np.degrees(np.arccos(np.clip(c, -1.0, 1.0)))
            if ang_to(pB_xy) <= ang_to(pC_xy):
                is_h1 = False
            else:
                is_h0 = False

        if is_h0:
            hori0_lines.append([a, b])
            if return_roof_base:
                if is_roof_like(segimg, a, b, sky_label):
                    roof_lines.append([a, b])
                elif is_bottom_like(segimg, a, b, ground_labels):
                    bottom_lines.append([a, b])

        elif is_h1:
            hori1_lines.append([a, b])
            if return_roof_base:
                if is_roof_like(segimg, a, b, sky_label):
                    roof_lines.append([a, b])
                elif is_bottom_like(segimg, a, b, ground_labels):
                    bottom_lines.append([a, b])

    if return_roof_base:
        return hori0_lines, hori1_lines, roof_lines, bottom_lines
    return hori0_lines, hori1_lines
