import numpy as np
import cv2

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
    seg_mask,           # segmentation mask (H,W), ints; 1==building or use your IDs
    Vt1, Vb1,           # [x,y] top/bottom of chosen left vertical  (image coords)
    Vt2, Vb2,           # [x,y] top/bottom of chosen right vertical (image coords)
    known_height_m,     # metric height (meters) to scale the rectified image
    target_h_px=1000    # desired rectified height in pixels (sampling resolution)
):
    """
    Returns:
      rect_rgb        : rectified RGB image (target_h_px tall)
      rect_mask       : rectified mask (same size as rect_rgb)
      meters_per_px   : scalar, metric scale in the rectified image
      width_m         : whole-quad width in meters (useful quick check)
      H               : 3x3 homography mapping image -> rectified
      (tw, th)        : rectified size (width px, height px)
    """

    # --- 1) Order the 4 source points as a convex quad in (TL, TR, BR, BL) ---

    # Decide which vertical is left vs right by comparing their top x
    left_is_1 = (Vt1[0] <= Vt2[0])  # True if vertical 1 is to the left

    if left_is_1:
        TL, TR, BR, BL = Vt1, Vt2, Vb2, Vb1  # TL=top of left, TR=top of right, BR=bottom of right, BL=bottom of left
    else:
        TL, TR, BR, BL = Vt2, Vt1, Vb1, Vb2  # swap roles if vertical 2 is left

    src = np.float32([TL, TR, BR, BL])       # source quad, image coords [x,y]

    # --- 2) Choose the target rectangle size (target_w_px, target_h_px) ---

    # Pixel height of the quad on the image (average of the two vertical lengths)
    h_left_px  = np.linalg.norm(np.array(BL) - np.array(TL))  # length of left vertical in pixels
    h_right_px = np.linalg.norm(np.array(BR) - np.array(TR))  # length of right vertical in pixels
    avg_h_px   = 0.5 * (h_left_px + h_right_px)               # average façade pixel-height in the image

    # Pixel width of the quad on the image (average of top and bottom spans)
    top_span_px    = np.linalg.norm(np.array(TR) - np.array(TL))  # top horizontal span in pixels
    bottom_span_px = np.linalg.norm(np.array(BR) - np.array(BL))  # bottom horizontal span in pixels
    avg_w_px       = 0.5 * (top_span_px + bottom_span_px)         # average façade pixel-width in the image

    # Preserve the façade's aspect ratio when we rectifiy:
    #    target_w_px / target_h_px  ≈  avg_w_px / avg_h_px
    # -> target_w_px  ≈  (avg_w_px / avg_h_px) * target_h_px
    ratio = (avg_w_px / max(1e-6, avg_h_px))                      # guard divide-by-zero
    target_w_px = int(max(8, round(ratio * float(target_h_px))))  # pick a reasonable integer width >=8

    # Destination rectangle corners in rectified pixel coords
    dst = np.float32([
        [0,               0],                # top-left  maps to x=0,               y=0
        [target_w_px - 1, 0],                # top-right maps to x=target_w_px-1,  y=0
        [target_w_px - 1, target_h_px - 1],  # bottom-right
        [0,               target_h_px - 1]   # bottom-left
    ])

    # --- 3) Compute homography that maps image -> rectified rectangle ---

    H = cv2.getPerspectiveTransform(src, dst)  # 3x3 matrix; maps [x,y,1]^T in image to rectified coords

    # --- 4) Warp RGB and mask into the rectified frame ---

    rect_rgb  = cv2.warpPerspective(img_rgb, H, (target_w_px, target_h_px), flags=cv2.INTER_LINEAR)      # bilinear for images
    rect_mask = cv2.warpPerspective(seg_mask.astype(np.int32), H, (target_w_px, target_h_px),
                                    flags=cv2.INTER_NEAREST)                                             # nearest for labels

    # --- 5) Set metric scale in the rectified image ---

    # By construction, the rectified vertical extent equals target_h_px pixels,
    # and that should correspond to the real-world height 'known_height_m'.
    meters_per_px = float(known_height_m) / float(target_h_px)  # m/px everywhere in rectified façade

    # Useful sanity: whole façade width in meters (can compare to building drawings)
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
