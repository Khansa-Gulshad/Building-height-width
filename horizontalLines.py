# -*- encoding:utf-8 -*-
import numpy as np

# reuse pointOnLine from your vertical file, or copy it here
def pointOnLine(a, b, p):
    l2 = np.sum((a - b) ** 2)
    if l2 == 0: return a.copy()
    t = np.sum((p - a) * (b - a)) / l2
    return a + t * (b - a)

def lineRefinementWithHVP(line, hvp):
    """
    Snap a roughly-horizontal segment to the chosen horizontal vanishing point (pH′).
    Rotate around the midpoint so the direction is collinear with (hvp ↔ midpoint).
    """
    a, b = line[0].astype(float), line[1].astype(float)
    mpt = (a + b) / 2.0
    line = line.astype(float)
    line[0] = pointOnLine(hvp, mpt, a)
    line[1] = pointOnLine(hvp, mpt, b)
    return line

def is_horizontal_by_vp(line, pB, pC, tau_h_deg):
    """
    Keep if min angle to rays (midpoint→pB′, midpoint→pC′) < τh.
    Returns (keep:bool, chosen_hvp:np.array).
    """
    a, b = line[0].astype(float), line[1].astype(float)
    m = (a + b) / 2.0
    v = (b - a).astype(float)

    def angle_deg(u, w):
        u = u / (np.linalg.norm(u) + 1e-9)
        w = w / (np.linalg.norm(w) + 1e-9)
        cosv = np.clip(np.dot(u, w), -1.0, 1.0)
        return np.degrees(np.arccos(cosv))

    phiB = angle_deg(v, pB - m)
    phiC = angle_deg(v, pC - m)
    if min(phiB, phiC) < tau_h_deg:
        return True, (pB if phiB <= phiC else pC)
    return False, None

def extendHoriz(line, segmt, config):
    """
    Extend a horizontal (already HVP-snapped) segment left/right inside the building mask.
    Returns (L, R) endpoints or ([], []) if rejected.
    """
    building_label = int(config["SEGMENTATION"]["BuildingLabel"])
    # Ground/Sky not strictly needed for horizontals; we stop at non-building.
    et = config["LINE_REFINE"].get("Edge_Thres", "5")
    edge_thres = int(et.split(',')[0]) if isinstance(et, str) else int(et)

    a, b = line[0].astype(float), line[1].astype(float)
    # ensure left/right order by column (index 1)
    if a[1] <= b[1]:
        L = a.copy(); R = b.copy()
    else:
        L = b.copy(); R = a.copy()

    rows, cols = segmt.shape
    # unit direction pointing rightward along the segment
    d = (R - L); n = np.linalg.norm(d)
    if n == 0: return [], []
    u = d / n

    # quick helper for safe label lookup
    def lab(pt):
        r = int(pt[0] + 0.5); c = int(pt[1] + 0.5)
        if r < 0 or r >= rows or c < 0 or c >= cols: return -999
        return segmt[r, c]

    # must start on building
    m = (L + R) / 2.0
    if lab(L) != building_label or lab(R) != building_label or lab(m) != building_label:
        return [], []

    # extend left: subtract u while staying on building
    Lend = L.copy()
    while True:
        cand = Lend - u
        r = int(cand[0] + 0.5); c = int(cand[1] + 0.5)
        # bounds
        if r < 0 or c < 0 or r >= rows or c >= cols: break
        if segmt[r, c] != building_label: break
        Lend = cand
    # extend right: add u while staying on building
    Rend = R.copy()
    while True:
        cand = Rend + u
        r = int(cand[0] + 0.5); c = int(cand[1] + 0.5)
        if r < 0 or c < 0 or r >= rows or c >= cols: break
        if segmt[r, c] != building_label: break
        Rend = cand

    # border safety
    if (Lend[0] < edge_thres or Lend[0] > rows-1-edge_thres or
        Rend[0] < edge_thres or Rend[0] > rows-1-edge_thres or
        Lend[1] < edge_thres or Lend[1] > cols-1-edge_thres or
        Rend[1] < edge_thres or Rend[1] > cols-1-edge_thres):
        return [], []

    return Lend, Rend

def horizontalLinePostprocess(lines, segimg, pB, pC, tau_h_deg, config):
    """
    Full horizontal pipeline:
    1) keep only segments that pass the horizontal-VP angle test,
    2) snap to chosen horizontal VP,
    3) extend left/right inside the building mask.
    """
    out = []
    for l in lines:
        keep, hvp = is_horizontal_by_vp(l, pB, pC, tau_h_deg)
        if not keep: continue
        l2 = lineRefinementWithHVP(l.copy(), hvp)
        L, R = extendHoriz(l2, segimg, config)
        if len(L) == 0 or len(R) == 0: continue
        out.append([L, R])
    return out
