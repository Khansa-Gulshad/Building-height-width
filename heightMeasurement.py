#-*- encoding:utf-8 -*-

# -*- encoding:utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
from lineClassification import *
from lineDrawingConfig import *
from lineRefinement import *
from filesIO import *
import skimage.io
import copy
import csv
import os


def gt_measurement(zgt_img, a, b, verbose=False):
    """
    Measure ground-truth height from a z-map image.
    a,b are [row, col].
    """
    if a[1] > b[1]:
        temp = copy.deepcopy(a)
        a = b
        b = temp

    a = np.cast["int"](a + [0.5, 0.5])
    b = np.cast["int"](b + [0.5, 0.5])

    rows, cols = zgt_img.shape
    row_check = lambda x: min(rows - 1, max(0, x))
    cols_check = lambda x: min(cols - 1, max(0, x))
    pt_check = lambda pt: np.asarray([cols_check(pt[0]), row_check(pt[1])])

    a = pt_check(a)
    b = pt_check(b)

    if zgt_img[a[1], a[0]] == 0 or zgt_img[b[1], b[0]] == 0:
        gt_org = 0
    else:
        gt_org = abs(zgt_img[a[1], a[0]] - zgt_img[b[1], b[0]])

    direction = (a - b) / np.linalg.norm(a - b)

    b_expd = copy.deepcopy(b)
    count = 1
    while zgt_img[b_expd[1], b_expd[0]] == 0 and a[1] < b_expd[1]:
        b_expd = np.cast["int"](b + count * direction)
        count += 1

    a_expd = copy.deepcopy(a)
    count = 1
    if zgt_img[a_expd[1], a_expd[0]] == 0:
        while (a_expd[0] > 0 and a_expd[0] <= cols - 1 and
               a_expd[1] <= rows - 1 and zgt_img[a_expd[1], a_expd[0]] == 0):
            a_expd = np.cast["int"](a - count * direction)
            count += 1
    else:
        while (a_expd[0] > 0 and a_expd[0] <= cols - 1 and
               a_expd[1] >= 0 and zgt_img[a_expd[1], a_expd[0]] != 0):
            a_expd = np.cast["int"](a + count * direction)
            count += 1
        a_expd = np.cast["int"](a + (count - 2) * direction)

    gt_expd = abs(zgt_img[a_expd[1], a_expd[0]] - zgt_img[b_expd[1], b_expd[0]])

    if verbose:
        print("here---------------:")
        print(a_expd)
        print(b_expd)
        print(zgt_img[a_expd[1], a_expd[0]])
        print(gt_org, gt_expd)

        plt.close()
        plt.figure()
        plt.imshow(zgt_img)
        plt.plot([a[0], b[0]], [a[1], b[1]], c=c(0), linewidth=2)
        plt.scatter(a[0], a[1], **PLTOPTS)
        plt.scatter(b[0], b[1], **PLTOPTS)

    return gt_org, gt_expd


def sv_measurement(v1, v2, v3, x1, x2, zc=2.5):
    """
    Three-VP SV metrology (projective form).
    """
    vline = np.cross(v1, v2)
    p4 = vline / np.linalg.norm(vline)

    zc = zc * np.linalg.det([v1, v2, v3])
    alpha = -np.linalg.det([v1, v2, p4]) / zc
    p3 = alpha * v3

    zx = -np.linalg.norm(np.cross(x1, x2)) / (np.dot(p4, x1) * np.linalg.norm(np.cross(p3, x2)))
    return abs(zx)


def sv_measurement1(v, vline, x1, x2, zc=2.5):
    """
    Vertical VP + horizontal vanishing line version.
    """
    p4 = vline / np.linalg.norm(vline)
    alpha = -1 / (np.dot(p4, v) * zc)
    p3 = alpha * v
    zx = -np.linalg.norm(np.cross(x1, x2)) / (np.dot(p4, x1) * np.linalg.norm(np.cross(p3, x2)))
    return abs(zx)


def singleViewMeasWithCrossRatio(hori_v1, hori_v2, vert_v1, pt_top, pt_bottom, zc=2.5):
    """
    Cross-ratio formulation with two horizontal VPs and a vertical VP.
    Coordinates here are 2D [x, y] on image plane.
    """
    line_vl = lineCoeff(hori_v1, hori_v2)
    line_building_vert = lineCoeff(pt_top, pt_bottom)
    C = intersection(line_vl, line_building_vert)

    dist_AC = np.linalg.norm(np.asarray([vert_v1 - C]))
    dist_AB = np.linalg.norm(np.asarray([vert_v1 - pt_top]))
    dist_BD = np.linalg.norm(np.asarray([pt_top - pt_bottom]))
    dist_CD = np.linalg.norm(np.asarray([C - pt_bottom]))

    height = dist_BD * dist_AC / (dist_CD * dist_AB) * zc
    return height


def singleViewMeasWithCrossRatio_vl(hori_vline, vert_v1, pt_top, pt_bottom, zc=2.5):
    """
    Cross-ratio with vertical VP + horizontal vanishing line.
    """
    line_vl = hori_vline
    line_building_vert = lineCoeff(pt_top, pt_bottom)
    C = intersection(line_vl, line_building_vert)

    dist_AC = np.linalg.norm(np.asarray([vert_v1 - C]))
    dist_AB = np.linalg.norm(np.asarray([vert_v1 - pt_top]))
    dist_BD = np.linalg.norm(np.asarray([pt_top - pt_bottom]))
    dist_CD = np.linalg.norm(np.asarray([C - pt_bottom]))

    height = dist_BD * dist_AC / (dist_CD * dist_AB) * zc
    return height


def vp_calculation_with_pitch(w, h, pitch, focal_length):
    """
    Compute vertical VP and horizontal vanishing line from known pitch (Google SV).
    Returns v (homog) and vline (homog line).
    """
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
    Main height estimation + export vertical refs for width.
    """
    if img_size is None:
        img_size = [640, 640]

    try:
        vpt_fname = fname_dict["vpt"]
        img_fname = fname_dict["img"]
        line_fname = fname_dict["line"]
        seg_fname = fname_dict["seg"]
        zgt_fname = fname_dict["zgt"]

        # ---------- VPs ----------
        w = img_size[0]
        h = img_size[1]
        focal_length = intrins[0, 0]

        if use_pitch_only:
            vps = np.zeros([3, 2])
            vertical_v, vline = vp_calculation_with_pitch(w, h, pitch, focal_length)
            if vertical_v[2] == 0:  # special case
                vertical_v[0] = 320
                vertical_v[1] = -9999999
            vps[2, :] = vertical_v[:2]
        elif '.npz' in vpt_fname:
            vps = load_vps_2d(vpt_fname)
            if not use_detected_vpt_only:
                vertical_v, vline = vp_calculation_with_pitch(w, h, pitch, focal_length)
                if vertical_v[2] == 0:  # special case
                    vertical_v[0] = 320
                    vertical_v[1] = -9999999
                vps[2, :] = vertical_v[:2]

        # ---------- inputs ----------
        line_segs, scores = load_line_array(line_fname)
        seg_img = load_seg_array(seg_fname)

        # optional overlays
        org_image = skimage.io.imread(img_fname)
        for i, t in enumerate([0.94]):
            plt.gca().set_axis_off()
            plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
            plt.margins(0, 0)
            for (a, b), s in zip(line_segs, scores):
                if s < t:
                    continue
                plt.plot([a[1], b[1]], [a[0], b[0]], c=c(s), linewidth=2, zorder=s)
                plt.scatter(a[1], a[0], **PLTOPTS)
                plt.scatter(b[1], b[0], **PLTOPTS)
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            plt.imshow(org_image)
            plt.imshow(seg_img, alpha=0.5)
            if use_pitch_only:
                x, y = vertical_v[:2]
                plt.scatter(x, y)
                plt.plot([0, w], [vline[2], vline[2]], c='b', linewidth=5)
            else:
                for i in range(len(vps)):
                    x, y = vps[i]
                    plt.scatter(x, y)
            integrated_save_name = img_fname.replace(".jpg", f"-{t:.02f}_inls.svg")
            integrated_save_name = integrated_save_name.replace("imgs", "inls")
            integrated_save_dir = os.path.dirname(integrated_save_name)
            if not os.path.exists(integrated_save_dir):
                os.makedirs(integrated_save_dir)
            plt.close()

        # ---------- process lines ----------
        if verbose:
            plt.close()
            plt.imshow(org_image)
            plt.imshow(seg_img, alpha=0.5)

        verticals = filter_lines_outof_building_ade20k(
            img_fname, line_segs, scores, seg_img, vps, config, use_pitch_only)
        verticals = verticalLineExtending(
            img_fname, verticals, seg_img, [vps[2, 1], vps[2, 0]], config)

        # ---------- heights ----------
        invK = np.linalg.inv(intrins)
        ht_set = []
        check_list = []

        for line in verticals:
            a = line[0]
            b = line[1]

            if len(check_list) != 0:
                flag = 0
                for a0, a1, b0, b1 in check_list:
                    if a0 == a[0] and a1 == a[1] and b0 == b[0] and b1 == b[1]:
                        flag = 1
                        break
                if flag:
                    continue
            check_list.append([a[0], a[1], b[0], b[1]])

            # [row,col] -> normalized camera rays via K^-1
            a_d3 = np.asarray([a[1], a[0], 1])
            a_d3 = np.matmul(invK, np.transpose(a_d3))
            b_d3 = np.asarray([b[1], b[0], 1])
            b_d3 = np.matmul(invK, np.transpose(b_d3))

            if use_detected_vpt_only:
                vps0 = np.asarray([vps[0, 0], vps[0, 1], 1])
                vps1 = np.asarray([vps[1, 0], vps[1, 1], 1])
                vps0 = np.matmul(invK, np.transpose(vps0))
                vps1 = np.matmul(invK, np.transpose(vps1))
                vps2 = np.asarray([vps[2, 0], vps[2, 1], 1])
                vps2 = np.matmul(invK, np.transpose(vps2))
                ht = sv_measurement(vps0, vps1, vps2, b_d3, a_d3,
                                    zc=float(config["STREET_VIEW"]["CameraHeight"]))
            else:
                ht = singleViewMeasWithCrossRatio_vl(
                    vline, vertical_v[:2], np.asarray([a[1], a[0]]),
                    np.asarray([b[1], b[0]]),
                    zc=float(config["STREET_VIEW"]["CameraHeight"]))

            gt_exist = int(config["GROUND_TRUTH"]["Exist"])
            if gt_exist:
                zgt_img = load_zgts(zgt_fname)
                ht_gt_org, ht_gt_expd = gt_measurement(
                    zgt_img, np.asarray([a[1], a[0]]), np.asarray([b[1], b[0]]))
            else:
                ht_gt_org, ht_gt_expd = ht * 0, ht * 0

            ht_set.append([ht, a, b, ht_gt_org, ht_gt_expd])

        if verbose:
            plt.close()
            plt.figure(figsize=(10, 8))
            plt.imshow(org_image)
            plt.imshow(seg_img, alpha=0.5)
        print("path:%s" % img_fname)

        # ---------- group by spatial proximity (+height if enabled) ----------
        grouped_lines = clausterLinesWithCenters(ht_set, config, using_height=True)
        if grouped_lines is None:
            print('no suitable vertical lines founded in image ' + img_fname)
            return

        list_len = len(grouped_lines)
        heights = []
        ax_legends = []
        if len(colors_tables) < list_len:
            print("warning: lines with the same color might be different groups.")

        for i in range(list_len):
            ax_line = None
            lines = grouped_lines[i]
            list_len_lines = len(lines)
            heights.append([lines[-2], lines[-1]])
            for j in range(list_len_lines - 2):
                a = lines[j][1]
                b = lines[j][2]
                if verbose:
                    ax_line, = plt.plot(
                        [a[1], b[1]], [a[0], b[0]],
                        c=colors_tables[i % len(colors_tables)], linewidth=2
                    )
                    plt.scatter(a[1], a[0], **PLTOPTS)
                    plt.scatter(b[1], b[0], **PLTOPTS)
            if verbose and ax_line is not None:
                ax_legends.append(ax_line)

        # --- BEGIN: export vertical references for width ---
        def rc_to_xy(pt_rc):
            # convert [row, col] -> [x, y]
            return np.array([float(pt_rc[1]), float(pt_rc[0])], dtype=float)

        vertical_refs_xy = []   # list of (Vt_xy, Vb_xy, median_h_m)

        if grouped_lines and len(grouped_lines) > 0:
            for gi, grp in enumerate(grouped_lines):
                # grp: [ [ht, a_rc, b_rc, (gt_org), (gt_expd)], ..., median, mean ]
                median_h_m = float(grp[-2])
                lines_in_grp = grp[:-2]
                if len(lines_in_grp) == 0:
                    continue

                # pick member closest to cluster median
                sel = int(np.argmin([abs(li[0] - median_h_m) for li in lines_in_grp]))
                _, a_rc, b_rc, *_ = lines_in_grp[sel]

                # enforce top/bottom by row (y)
                if a_rc[0] > b_rc[0]:
                    a_rc, b_rc = b_rc, a_rc

                Vt_xy = rc_to_xy(a_rc)   # [x,y]
                Vb_xy = rc_to_xy(b_rc)   # [x,y]
                vertical_refs_xy.append((Vt_xy, Vb_xy, median_h_m))
        else:
            print("[warn] grouped_lines empty; no vertical references exported.")

        # save a sidecar so the width step can load it
        try:
            side_npz = img_fname.replace('save_rgb/imgs', 'metrics').replace('.jpg', '_vertical_refs.npz')
            os.makedirs(os.path.dirname(side_npz), exist_ok=True)
            if len(vertical_refs_xy) > 0:
                Vt = np.array([v[0] for v in vertical_refs_xy], dtype=float)   # (N,2)
                Vb = np.array([v[1] for v in vertical_refs_xy], dtype=float)   # (N,2)
                Hm = np.array([v[2] for v in vertical_refs_xy], dtype=float)   # (N,)
            else:
                Vt = np.zeros((0, 2), float)
                Vb = np.zeros((0, 2), float)
                Hm = np.zeros((0,), float)
            np.savez(side_npz, Vt_xy=Vt, Vb_xy=Vb, median_h_m=Hm,
                     heights_csv="/w/PROJ/heights.csv")
            print("vertical refs ->", side_npz)
        except Exception as e:
            print("[warn] could not save vertical refs:", e)
        # --- END: export vertical references for width ---

        # ---------- write heights CSV ----------
        try:
            out_csv = "/w/PROJ/heights.csv"
            group_counts = [len(g) - 2 for g in grouped_lines]  # each group ends with [median, mean]
            write_header = not os.path.exists(out_csv)
            os.makedirs(os.path.dirname(out_csv), exist_ok=True)
            with open(out_csv, "a", newline="") as f:
                w = csv.writer(f)
                if write_header:
                    w.writerow(["image", "group_idx", "median_m", "mean_m", "count"])
                for gi, (med, mean) in enumerate(heights):
                    w.writerow([img_fname, gi, float(med), float(mean), int(group_counts[gi])])
            print("heights ->", out_csv)
        except Exception as e:
            print("[warn] could not write heights CSV:", e)

        if verbose:
            plt.legend(ax_legends, ['average_height = %.4fm, median_height = %.4fm' % (y, x) for x, y in heights])
            result_save_name = img_fname.replace('imgs', 'ht_results')
            result_save_name = result_save_name.replace('.jpg', '_htre.svg')
            result_save_name2 = result_save_name.replace('.svg', '.png')
            re_save_dir = os.path.dirname(result_save_name)
            if not os.path.exists(re_save_dir):
                os.makedirs(re_save_dir)
            plt.savefig(result_save_name, bbox_inches="tight")
            plt.savefig(result_save_name2, bbox_inches="tight")
            plt.close()

    except IOError:
        print("file does not exist\n")

