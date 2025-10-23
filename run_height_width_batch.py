import os, glob, json, argparse, configparser
import numpy as np
from PIL import Image
from heightWidth_areaMeasurement import heightCalc

REPO = "/users/project1/pt01183/Building-height-width"
SIHE = os.path.join(REPO, "SIHE")

LINES_DIR  = os.path.join(SIHE, "data/lines")
IMG_JPG    = os.path.join(SIHE, "data/images_real")
IMG_PNG512 = os.path.join(REPO, "data/wflike/A")
CITY_DIR   = os.path.join(REPO, "Gdańsk, Poland")
SEG_DIR    = os.path.join(CITY_DIR, "seg")  # <stem>_seg.npz
VP_JSON    = os.path.join(REPO, "data/vpts/json")
VP_PERNPZ  = os.path.join(REPO, "data/vpts/per_npz")

os.makedirs(VP_PERNPZ, exist_ok=True)

def build_K(w, h, fov_deg):
    fx = 0.5 * w / np.tan(np.deg2rad(fov_deg / 2.0))
    fy = fx
    cx, cy = w / 2.0, h / 2.0
    return np.array([[fx, 0, cx],[0, fy, cy],[0, 0, 1.0]], float)

def find_image(stem):
    # prefer full jpg
    jpg = os.path.join(IMG_JPG, f"{stem}.jpg")
    if os.path.exists(jpg): return jpg
    # fallback to 512 png (what NeurVPS consumed)
    png = os.path.join(IMG_PNG512, f"{stem}.png")
    if os.path.exists(png): return png
    return None

def json_to_npz(stem):
    jf = os.path.join(VP_JSON, f"{stem}.json")
    if not os.path.exists(jf): return None
    with open(jf, "r") as f:
        j = json.load(f)
    # expect ordered dict keys in this order
    order = ["v1_right", "v2_left", "v3_vertical"]
    vps2d = np.array([j["vpts_2d_ordered"][k] for k in order], dtype=float)  # shape (3,2)
    out = os.path.join(VP_PERNPZ, f"{stem}.npz")
    # save with several keys for safety with load_vps_2d
    np.savez(out, vps_2d=vps2d, vpts_2d=vps2d, vps=vps2d, vpts=vps2d)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(REPO, "config/estimation_config.ini"))
    ap.add_argument("--fov", type=float, default=100.0)     # the FOV you used to fetch GSV (default 70)
    ap.add_argument("--pitch", type=float, default=15.0)    # the pitch you used
    ap.add_argument("--verbose", type=int, default=0)
    # mode flags
    ap.add_argument("--use_pitch_only", type=int, default=0)         # 1 = heights only
    ap.add_argument("--use_detected_vpt_only", type=int, default=0)  # 1 = pure detected (all 3 VPs)
    args = ap.parse_args()

    # load config
    cfg = configparser.ConfigParser()
    cfg.read(args.config)

    stems = [os.path.splitext(os.path.basename(p))[0] for p in sorted(glob.glob(os.path.join(LINES_DIR, "*.npz")))]
    if not stems:
        print("No line NPZs found in", LINES_DIR)
        return

    # results csv
    metrics_dir = os.path.join(REPO, "data/metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    out_csv = os.path.join(metrics_dir, "height_width_area.csv")
    with open(out_csv, "w") as fh:
        fh.write("stem,height_median,height_mean,width_median,width_mean,area_median,area_mean\n")

    done = 0
    for stem in stems:
        img = find_image(stem)
        if img is None:
            print(f"[skip] no image for {stem}")
            continue
        seg = os.path.join(SEG_DIR, f"{stem}_seg.npz")
        if not os.path.exists(seg):
            print(f"[skip] no seg for {stem} ({seg})")
            continue
        line = os.path.join(LINES_DIR, f"{stem}.npz")
        if not os.path.exists(line):
            print(f"[skip] no lines for {stem}")
            continue
        vpt_npz = json_to_npz(stem)
        if vpt_npz is None:
            print(f"[skip] no VP json for {stem}")
            continue

        # image size + intrinsics
        with Image.open(img) as im:
            W, H = im.size
        K = build_K(W, H, args.fov)

        fname_dict = {
            "vpt":  vpt_npz,
            "img":  img,
            "line": line,
            "seg":  seg
        }

        res = heightCalc(
            fname_dict=fname_dict,
            intrins=K,
            config=cfg,
            img_size=[W, H],
            pitch=args.pitch,
            use_pitch_only=args.use_pitch_only,
            use_detected_vpt_only=args.use_detected_vpt_only,
            verbose=bool(args.verbose)
        )
        if res is None:
            print(f"[fail] {stem}")
            continue

        grouped_heights, grouped_widths, areas = res
        # write one row per matched area (or a stub if none)
        if areas:
            for a in areas:
                row = f"{stem},{a['height_median']:.4f},{a['height_mean']:.4f},{a.get('width_median',0):.4f},{a.get('width_mean',0):.4f},{a.get('area_median',0):.4f},{a.get('area_mean',0):.4f}\n"
                with open(out_csv, "a") as fh:
                    fh.write(row)
        else:
            # still log heights-only case
            with open(out_csv, "a") as fh:
                fh.write(f"{stem},,,,\n")

        done += 1

    print(f"[done] processed {done} images -> {out_csv}")

if __name__ == "__main__":
    main()
