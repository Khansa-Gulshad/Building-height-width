# -*- encoding:utf-8 -*-

import os
import copy
import numpy as np
import matplotlib.pyplot as plt
import skimage.io

from lineClassification import *   # lineCoeff, intersection, filters, clustering, etc.
from lineDrawingConfig import *    # colors_tables, PLTOPTS, c(), etc.
from lineRefinement import *       # verticalLineExtending, etc.
from filesIO import *              # load_vps_2d, load_line_array, load_seg_array, load_zgts


def gt_measurement(zgt_img, a, b, verbose=False):
    """
    If there is a ground truth image with each pixel value representing vertical z, measure GT height of a line [a,b].
    a/b are [y,x].
    """
    if a[1] > b[1]:
        a, b = copy.deepcopy(b), copy.deepcopy(a)

    a = np.cast["int"](a + [0.5, 0.5])
    b = np.cast["int"](b + [0.5, 0.5])

    rows, cols = zgt_img.shape
    row_clamp = lambda x: min(rows - 1, max(0, x))
    col_clamp = lambda x: min(cols - 1, max(0, x))
    fix_pt = lambda pt: np.asarray([col_clamp(pt[0]), row_clamp(pt[1])])

    a = fix_pt(a)
    b = fix_pt(b)

    if zgt_img[a[1], a[0]] == 0 or zgt_img[b[1], b[0]] == 0:
        gt_org = 0
    else:
        gt_org = abs(zgt_img[a[1], a[0]] - zgt_img[b[1], b[0]])

    direction = (a - b) / (np.linalg.norm(a - b) + 1e-8)

    b_expd = copy.deepcopy(b)
    cnt = 1
    while zgt_img[b_expd[1], b_expd[0]] == 0 and a[1] < b_expd[1]:
        b_expd = np.cast["int"](b + cnt * direction)
        cnt += 1

    a_expd = copy.deepcopy(a)
    cnt = 1
    if zgt_img[a_expd[1], a_expd[0]] == 0:
        while 0 < a_expd[0] <= cols - 1 and a_expd[1] <= rows - 1 and zgt_img[a_expd[1], a_expd[0]] == 0:
            a_expd = np.cast["int"](a - cnt * direction)
            cnt += 1
    else:
        while 0 < a_expd[0] <= cols - 1 and a_expd[1] >= 0 and zgt_img[a_expd[1], a_expd[0]] != 0:
            a_expd = np.cast["int"](a + cnt * direction)
            cnt += 1
        a_expd = np.cast["int"](a + (cnt - 2) * direction)

    gt_expd = abs(zgt_img[a_expd[1], a_expd[0]] - zgt_img[b_expd[1], b_expd[0]])

    if verbose:
        plt.close()
        plt.figure()
        plt.imshow(zgt_img)
        plt.plot([a[0], b[0]], [a[1], b[1]], c=c(0), linewidth=2)
        plt.scatter(a[0], a[1], **PLTOPTS)
        plt.scatter(b[0], b[1], **PLTOPTS)
        plt.close()

    return gt_org, gt_expd


def sv_measurement_along(v_dir, v_a, v_b, x1, x2, zc=2.5):
    """
    Measure metric length of segment (x1,x2) aligned with direction v_dir,
    using the vanishing line from (v_a, v_b). All vectors are 3D homogeneous in normalized camera frame.
    """
    vline = np.cross(v_a, v_b)
    p4 = vline / (np.linalg.norm(vline) + 1e-8)

    zc_scaled = zc * np.linalg.det([v_a, v_b, v_dir])
    alpha = -np.linalg.det([v_a, v_b, p4]) / (zc_scaled + 1e-12)
    p3 = alpha * v_dir

    num = np.linalg.norm(np.cross(x1, x2))
    den = (np.dot(p4, x1) * (np.linalg.norm(np.cross(p3, x2)) + 1e-12))
    L = -num / (den + 1e-12)
    return abs(L)


def sv_measurement(v1, v2, v3, x1, x2, zc=2.5):
    """ Original height formula with three VPs (normalized camera frame). """
    vline = np.cross(v1, v2)
    p4 = vline / np.linalg.norm(vline)

    zc = zc * np.linalg.det([v1, v2, v3])
    alpha = -np.linalg.det([v1, v2, p4]) / (zc + 1e-12)
    p3 = alpha * v3

    zx = -np.linalg.norm(np.cross(x1, x2)) / (np.dot(p4, x1) * (np.linalg.norm(np.cross(p3, x2)) + 1e-12))
    return abs(zx)


def sv_measurement1(v, vline, x1, x2, zc=2.5):
    """ Height using vertical VP + horizontal vanishing line (normalized camera frame). """
    p4 = vline / np.linalg.norm(vline)
    alpha = -1.0 / (np.dot(p4, v) * zc + 1e-12)
    p3 = alpha * v
    zx = -np.linalg.norm(np.cross(x1, x2)) / (np.dot(p4, x1) * (np.linalg.norm(np.cross(p3, x2)) + 1e-12))
    return abs(zx)


def singleViewMeasWithCrossRatio(hori_v1, hori_v2, vert_v1, pt_top, pt_bottom, zc=2.5):
    """ Cross-ratio height with two horizontal VPs + vertical VP (image coords, not normalized). """
    line_vl = lineCoeff(hori_v1, hori_v2)
    line_building_vert = lineCoeff(pt_top, pt_bottom)
    C = intersection(line_vl, line_building_vert)

    dist_AC = np.linalg.norm(np.asarray([vert_v1 - C]))
    dist_AB = np.linalg.norm(np.asarray([vert_v1 - pt_top]))
    dist_BD = np.linalg.norm(np.asarray([pt_top - pt_bottom]))
    dist_CD = np.linalg.norm(np.asarray([C - pt_bottom]))

    return dist_BD * dist_AC / (dist_CD * dist_AB + 1e-12) * zc


def singleViewMeasWithCrossRatio_vl(hori_vline, vert_v1, pt_top, pt_bottom, zc=2.5):
    """ Cross-ratio height with vertical VP + horizontal vanishing line (image coords). """
    line_vl = hori_vline
    line_building_vert = lineCoeff(pt_top, pt_bottom)
    C = intersection(line_vl, line_building_vert)

    dist_AC = np.linalg.norm(np.asarray([vert_v1 - C]))
    dist_AB = np.linalg.norm(np.asarray([vert_v1 - pt_top]))
    dist_BD = np.linalg.norm(np.asarray([pt_top - pt_bottom]))
    dist_CD = np.linalg.norm(np.asarray([C - pt_bottom]))

    return dist_BD * dist_AC / (dist_CD * dist_AB + 1e-12) * zc


def vp_calculation_with_pitch(w, h, pitch, focal_length):
    """ For street-view style inputs (known pitch), compute vertical VP + horizon line in image coords. """
    v = np.array([w / 2, 0.0, 1.0])
    vline = np.array([0.0, 1.0, 0.0])

    if pitch == 0:
        v[:] = [0, -1, 0]
        vline[:] = [0, 1, h / 2]
    else:
        v[1] = h / 2 - (focal_length / np.tan(np.deg2rad(pitch)))
        vline[2] = (h / 2 + focal_length * np.tan(np.deg2rad(pitch)))
    return v, vline


def heightCalc(fname_dict, intrins, config, img_size=None, pitch=None,
               use_pitch_only=0, use_detected_vpt_only=0, verbose=False):
    """
    Main: estimate height (always), width (if detected VPs available), and an area per building.
    """
    if img_size is None:
        img_size = [640, 640]

    try:
        vpt_fname = fname_dict["vpt"]
        img_fname = fname_dict["img"]
        line_fname = fname_dict["line"]
        seg_fname = fname_dict["seg"]
        zgt_fname = fname_dict.get("zgt", "")

        # ===== 1) Vanishing points (image coords) =====
        w, h = img_size
        focal_length = intrins[0, 0]

        if use_pitch_only:
            # Only vertical VP + horizon
            vps = np.zeros([3, 2])
            vertical_v, vline = vp_calculation_with_pitch(w, h, pitch, focal_length)
            if vertical_v[2] == 0:
                vertical_v[0], vertical_v[1] = 320, -9999999
            vps[2, :] = vertical_v[:2]

        elif ".npz" in vpt_fname:
            vps = load_vps_2d(vpt_fname)  # expects shape (3,2): [v1_right, v2_left, v3_vertical]
            if not use_detected_vpt_only:
                # Replace the vertical with pitch-derived one (keeps horizon line too)
                vertical_v, vline = vp_calculation_with_pitch(w, h, pitch, focal_length)
                if vertical_v[2] == 0:
                    vertical_v[0], vertical_v[1] = 320, -9999999
                vps[2, :] = vertical_v[:2]
        else:
            raise IOError("vpt file not found or unsupported")

        # ===== 2) Load LCNN lines + seg =====
        line_segs, scores = load_line_array(line_fname)  # lines: (N,2,2) in [y,x]
        seg_img = load_seg_array(seg_fname)              # HxW labels

        # (Optional) quick viz of inputs
        if verbose:
            org_image = skimage.io.imread(img_fname)
            plt.close()
            plt.figure()
            plt.imshow(org_image)
            plt.imshow(seg_img, alpha=0.5)
            # draw VPs
            for i_v in range(3):
                x, y = vps[i_v]
                plt.scatter(x, y, s=30)
            plt.title(os.path.basename(img_fname))
            plt.close()

        # ===== 3) Classify & refine lines =====
       verticals, hori0_lines, hori1_lines, bottom_lines, roof_lines = filter_lines_outof_building_ade20k(
            img_fname, line_segs, scores, seg_img, vps, config,
            use_vertical_vpt_only=use_pitch_only, verbose=verbose
        )
        # extend verticals (roof<->ground)
        verticals = verticalLineExtending(img_fname, verticals, seg_img, [vps[2, 1], vps[2, 0]], config)

        # ===== 4) Heights (always) =====
        invK = np.linalg.inv(intrins)

        # ---- Precompute normalized camera-frame VPs (only if using detected VPs)
        vps0_d3 = vps1_d3 = vps2_d3 = None
        if use_detected_vpt_only:
            vps0_d3 = np.matmul(invK, np.array([vps[0, 0], vps[0, 1], 1.0]))
            vps1_d3 = np.matmul(invK, np.array([vps[1, 0], vps[1, 1], 1.0]))
            vps2_d3 = np.matmul(invK, np.array([vps[2, 0], vps[2, 1], 1.0]))

        ht_set = []
        seen = set()

        for line in verticals:
            a, b = line[0], line[1]  # [y,x]
            key = (int(a[0]), int(a[1]), int(b[0]), int(b[1]))
            if key in seen:
                continue
            seen.add(key)

            a_d3 = np.matmul(invK, np.array([a[1], a[0], 1.0]))
            b_d3 = np.matmul(invK, np.array([b[1], b[0], 1.0]))

            if use_detected_vpt_only:
                ht = sv_measurement(vps0_d3, vps1_d3, vps2_d3, b_d3, a_d3,
                                    zc=float(config["STREET_VIEW"]["CameraHeight"]))
            else:
                # cross-ratio version with pitch-derived horizon/vertical VP (image coords)
                ht = singleViewMeasWithCrossRatio_vl(vline, vertical_v[:2],
                                                     np.asarray([a[1], a[0]]),
                                                     np.asarray([b[1], b[0]]),
                                                     zc=float(config["STREET_VIEW"]["CameraHeight"]))

            if int(config["GROUND_TRUTH"]["Exist"]):
                zgt_img = load_zgts(zgt_fname)
                ht_gt_org, ht_gt_expd = gt_measurement(zgt_img, np.asarray([a[1], a[0]]),
                                                       np.asarray([b[1], b[0]]))
            else:
                ht_gt_org = ht_gt_expd = 0.0

            ht_set.append([ht, a, b, ht_gt_org, ht_gt_expd])

        # ===== 5) Widths (only when using detected VPs) =====
        wd_set = []
        if use_detected_vpt_only and (vps0_d3 is not None):
            cam_h = float(config["STREET_VIEW"]["CameraHeight"])

            def add_widths_from(lines, v_dir, v_a, v_b, grp_id):
                for ln in lines:
                    ax, bx = ln[0], ln[1]      # [y,x]
                    a_cam = np.matmul(invK, np.array([ax[1], ax[0], 1.0]))
                    b_cam = np.matmul(invK, np.array([bx[1], bx[0], 1.0]))
                    wval = sv_measurement_along(v_dir, v_a, v_b, a_cam, b_cam, zc=cam_h)
                    wd_set.append([wval, ax, bx, grp_id])

            # hori0 aligned with v1_right → use vanishing line from (v2_left, v3_vertical)
            add_widths_from(hori0_lines, vps0_d3, vps1_d3, vps2_d3, grp_id=0)
            # hori1 aligned with v2_left  → use vanishing line from (v1_right, v3_vertical)
            add_widths_from(hori1_lines, vps1_d3, vps0_d3, vps2_d3, grp_id=1)
        else:
            if verbose:
                print("[width] skipped: need detected VPs (use_detected_vpt_only=1).")

        # ===== 6) Group (cluster) & visualize =====
        if verbose:
            print("path:%s" % img_fname)

        grouped_heights = clausterLinesWithCenters(ht_set, config, using_height=True)
        if grouped_heights is None:
            print("no suitable vertical lines found in", img_fname)
            return None

        grouped_widths = None
        if len(wd_set) > 0:
            grouped_widths = clausterLinesWithCenters(wd_set, config, using_height=True)

        # plot grouped verticals
        if verbose:
            plt.close()
            plt.figure(figsize=(10, 8))
            org_img = skimage.io.imread(img_fname)
            plt.imshow(org_img)
            plt.imshow(seg_img, alpha=0.5)

            heights = []
            ax_legends = []
            for i_g, grp in enumerate(grouped_heights):
                heights.append([grp[-2], grp[-1]])  # [median, mean]
                color = colors_tables[i_g % len(colors_tables)]
                last_line = None
                for it in grp[:-2]:
                    _, a, b, *_ = it
                    last_line, = plt.plot([a[1], b[1]], [a[0], b[0]], c=color, linewidth=2)
                    plt.scatter(a[1], a[0], **PLTOPTS)
                    plt.scatter(b[1], b[0], **PLTOPTS)
                if last_line is not None:
                    ax_legends.append(last_line)

            if ax_legends:
                plt.legend(ax_legends, [f'avg={y:.3f}m, med={x:.3f}m' for x, y in heights])
            out_dir = os.path.dirname(img_fname.replace('imgs', 'ht_results'))
            os.makedirs(out_dir, exist_ok=True)
            out_svg = img_fname.replace('imgs', 'ht_results').replace('.jpg', '_htre.svg')
            out_png = out_svg.replace('.svg', '.png')
            plt.savefig(out_svg, bbox_inches="tight")
            plt.savefig(out_png, bbox_inches="tight")
            plt.close()

        # ===== 7) Pair height groups ↔ width groups (simple nearest-centroid), compute area =====
        def _centroid_of_group(grp):
            pts = []
            for it in grp[:-2]:
                _, a, b, *_ = it
                pts.append(((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0))  # (y,x)
            if not pts:
                return np.array([np.nan, np.nan])
            return np.mean(np.array(pts), axis=0)

        areas = []
        if grouped_widths is not None:
            h_centers = [_centroid_of_group(g) for g in grouped_heights]
            w_centers = [_centroid_of_group(g) for g in grouped_widths]

            for hi, hg in enumerate(grouped_heights):
                if not np.isfinite(h_centers[hi]).all():
                    continue
                h_med, h_mean = hg[-2], hg[-1]
                dists = [np.linalg.norm(h_centers[hi] - wc) if np.isfinite(wc).all() else 1e9
                         for wc in w_centers]
                if not dists or min(dists) == 1e9:
                    continue
                wi = int(np.argmin(dists))
                wg = grouped_widths[wi]
                w_med, w_mean = wg[-2], wg[-1]
                areas.append({
                    "height_median": h_med, "height_mean": h_mean,
                    "width_median": w_med,  "width_mean":  w_mean,
                    "area_median":  h_med * w_med,
                    "area_mean":    h_mean * w_mean,
                    "pair_dist_px": float(dists[wi]),
                })

            print("[areas] per matched building (median×median):")
            for k, a in enumerate(areas):
                print(f"  #{k}: H_med={a['height_median']:.3f} m, "
                      f"W_med={a['width_median']:.3f} m, "
                      f"Area_med={a['area_median']:.3f} m^2  "
                      f"(pair_dist_px={a['pair_dist_px']:.1f})")
        else:
            print("[areas] skipped (no grouped widths).")

        return grouped_heights, grouped_widths, areas

    except Exception as e:
        print("heightCalc error:", str(e))
        return None

    except IOError:
        print("file does not exist\n")
