import numpy as np
import cv2

import numpy as np

def robust_keep_by_height(Hm, min_h=2.5, max_h=120.0, k_mad=4.5):
    """
    Keep heights within [min_h, max_h] and within k*MAD of the median.
    Returns a boolean mask of length len(Hm).
    RELAXED defaults (broader than before) to avoid dropping useful verticals.
    """
    Hm = np.asarray(Hm, float)
    finite = np.isfinite(Hm)
    Hf = Hm[finite]
    if len(Hf) == 0:
        return np.zeros_like(Hm, dtype=bool)
    med = np.median(Hf)
    mad = np.median(np.abs(Hf - med)) + 1e-9
    z = np.abs(Hm - med) / mad
    return finite & (Hm >= min_h) & (Hm <= max_h) & (z <= k_mad)


def filter_vertical_refs_with_counts(
    stem, Vt, Vb, Hm, heights_csv_path="/w/PROJ/heights.csv",
    min_count=1,              # was 3 → keep more data by default
    min_h=2.5, max_h=120.0,   # was 80.0 → allow taller façades
    k_mad=4.5,                # was 3.5 → more tolerant to spread
    pix_h_min=10,             # was 15 → accept slightly shorter slivers
    mode="relaxed",           # "relaxed" | "strict" | "count_only"
    verbose=False
):
    """
    Filter per-image vertical refs using (1) group 'count' from heights.csv,
    (2) robust height sanity (range + MAD), and (3) simple pixel checks.
    Set mode="count_only" to keep only the count filter + pixel sanity.
    """
    import csv, os

    Vt = np.asarray(Vt, float)
    Vb = np.asarray(Vb, float)
    Hm = np.asarray(Hm, float)

    # --- choose presets by mode ---
    if mode == "strict":
        # Use your earlier, tighter gates
        min_count = max(min_count, 3)
        min_h, max_h, k_mad, pix_h_min = 2.5, 80.0, 3.5, 15
    elif mode == "count_only":
        # Disable height-based rejection; keep pixel sanity only
        pass  # keep current arguments; we’ll skip the robust step below
    else:
        # "relaxed" -> use the function defaults defined above
        pass

    # --- 1) map median_m -> count for this image ---
    keep = np.ones(len(Hm), dtype=bool)
    counts = {}
    if os.path.exists(heights_csv_path):
        with open(heights_csv_path) as f:
            r = csv.DictReader(f)
            for row in r:
                if row["image"].endswith(f"/{stem}.jpg"):
                    m = round(float(row["median_m"]), 6)
                    counts[m] = max(counts.get(m, 0), int(row.get("count", "0")))

    # drop entries whose group count < min_count (if we can match)
    for i, h in enumerate(Hm):
        m = round(float(h), 6)
        if m in counts and counts[m] < min_count:
            keep[i] = False
            if verbose:
                print(f"[{stem}] drop idx {i}: group count {counts[m]} < {min_count}")

    # --- 2) robust height sanity (unless count_only) ---
    if mode != "count_only":
        keep &= robust_keep_by_height(Hm, min_h=min_h, max_h=max_h, k_mad=k_mad)

    # --- 3) pixel sanity: top<bottom and minimum pixel height ---
    for i in range(len(keep)):
        if not keep[i]:
            continue
        if not (Vt[i][1] < Vb[i][1]):  # enforce top.y < bottom.y
            keep[i] = False
            if verbose:
                print(f"[{stem}] drop idx {i}: top below bottom")
            continue
        pix_h = float(np.linalg.norm(np.array(Vb[i]) - np.array(Vt[i])))
        if pix_h < pix_h_min:
            keep[i] = False
            if verbose:
                print(f"[{stem}] drop idx {i}: pix_h {pix_h:.1f} < {pix_h_min}")

    return Vt[keep], Vb[keep], Hm[keep]

def pick_two_verticals_farthest_x(Vt_xy, Vb_xy, heights_m):
    """
    Select two verticals on the same façade that are farthest apart horizontally.
    Inputs:
      Vt_xy: (N,2) array of tops [x,y] for each vertical (in image coords)
      Vb_xy: (N,2) array of bottoms [x,y] for each vertical
      heights_m: (N,) array; the metric height estimated for each vertical
    Returns:
      idxL, idxR: indices of the left and right chosen verticals
      height_m_for_scale: a single height to set meters-per-pixel (mean of the two)
    """
    # Compute each vertical's x-position as the average of its top/bottom x  (robust)
    x_centers = 0.5 * (Vt_xy[:, 0] + Vb_xy[:, 0])  # (N,)

    # Pick leftmost and rightmost indices
    idxL = int(np.argmin(x_centers))               # index of the minimum x
    idxR = int(np.argmax(x_centers))               # index of the maximum x
    if idxL == idxR and len(x_centers) >= 2:       # degenerate case: all same x?
        idxR = (idxL + 1) % len(x_centers)         # fall back to a neighbor

    # Use the average metric height from these two as the scale (robust to noise)
    height_m_for_scale = float(0.5 * (heights_m[idxL] + heights_m[idxR]))
    return idxL, idxR, height_m_for_scale


def rectify_facade_from_two_verticals(
    img_rgb,            # original RGB image (H,W,3), dtype uint8
    seg_mask,           # segmentation mask (H,W), ints
    Vt1, Vb1,           # [x,y] top/bottom of chosen left vertical  (image coords)
    Vt2, Vb2,           # [x,y] top/bottom of chosen right vertical (image coords)
    known_height_m,     # metric height (meters) to scale the rectified image
    target_h_px=None,   # <-- None/"native"/"auto" => match native façade pixel height
    min_h_px=64,        # safety clamp (avoid tiny outputs)
    max_h_px=None       # optional upper clamp (e.g., 2000)
):
    """
    Returns:
      rect_rgb, rect_mask, meters_per_px, width_m, H, (target_w_px, target_h_px)
    """
    # --- 1) Order the 4 source points as a convex quad in (TL, TR, BR, BL) ---
    left_is_1 = (Vt1[0] <= Vt2[0])
    if left_is_1:
        TL, TR, BR, BL = Vt1, Vt2, Vb2, Vb1
    else:
        TL, TR, BR, BL = Vt2, Vt1, Vb1, Vb2
    src = np.float32([TL, TR, BR, BL])

    # --- 2) Choose the target rectangle size (target_w_px, target_h_px) ---
    h_left_px  = np.linalg.norm(np.array(BL) - np.array(TL))
    h_right_px = np.linalg.norm(np.array(BR) - np.array(TR))
    avg_h_px   = 0.5 * (h_left_px + h_right_px)            # native façade pixel height in the input

    top_span_px    = np.linalg.norm(np.array(TR) - np.array(TL))
    bottom_span_px = np.linalg.norm(np.array(BR) - np.array(BL))
    avg_w_px       = 0.5 * (top_span_px + bottom_span_px)  # native façade pixel width in the input

    # Native sampling option
    if target_h_px is None or str(target_h_px).lower() in ("native", "auto"):
        target_h_px = int(round(avg_h_px))
        target_h_px = max(int(min_h_px), target_h_px)
        if max_h_px is not None:
            target_h_px = min(int(max_h_px), target_h_px)
    else:
        target_h_px = int(target_h_px)

    ratio = (avg_w_px / max(1e-6, avg_h_px))               # preserve aspect ratio
    target_w_px = int(max(8, round(ratio * float(target_h_px))))

    dst = np.float32([
        [0,               0],
        [target_w_px - 1, 0],
        [target_w_px - 1, target_h_px - 1],
        [0,               target_h_px - 1]
    ])

    # --- 3) Homography ---
    H = cv2.getPerspectiveTransform(src, dst)

    # --- 4) Warp ---
    rect_rgb  = cv2.warpPerspective(img_rgb, H, (target_w_px, target_h_px), flags=cv2.INTER_LINEAR)
    rect_mask = cv2.warpPerspective(seg_mask.astype(np.int32), H, (target_w_px, target_h_px),
                                    flags=cv2.INTER_NEAREST)

    # --- 5) Metric scale ---
    meters_per_px = float(known_height_m) / float(target_h_px)  # m/px in rectified frame
    width_m = float(target_w_px) * meters_per_px

    return rect_rgb, rect_mask, meters_per_px, width_m, H, (target_w_px, target_h_px)


def facade_areas_from_rectified_mask(rect_mask, meters_per_px,
                                     building_ids=(2,), window_ids=(3,), door_ids=(4,)):
    """
    Compute areas (m^2) after rectification by counting pixels per class.
    Adjust 'building_ids', 'window_ids', 'door_ids' to your label map.

    Returns:
      area_facade_m2, area_windows_m2, area_doors_m2
    """
    # Count pixels for each category (True==1, False==0)
    px_facade  = int(np.isin(rect_mask, building_ids).sum())  # façade pixels in rectified image
    px_windows = int(np.isin(rect_mask, window_ids).sum())    # window pixels
    px_doors   = int(np.isin(rect_mask, door_ids).sum())      # door pixels

    # Convert pixel counts -> area using (m/px)^2
    px_area_to_m2 = (meters_per_px ** 2)
    area_facade_m2  = px_facade  * px_area_to_m2
    area_windows_m2 = px_windows * px_area_to_m2
    area_doors_m2   = px_doors   * px_area_to_m2
    return area_facade_m2, area_windows_m2, area_doors_m2
