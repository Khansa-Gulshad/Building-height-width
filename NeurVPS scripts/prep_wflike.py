
#!/usr/bin/env python3
"""
make_wflike_dataset.py

Create a "Wireframe-like" dataset view from a folder of images:
- Converts input images to 512x512 PNGs under <dst>/A/
- Writes dummy *_label.npz (unused by eval, but keeps the loader happy)
- Creates split files <dst>/{valid.txt,test.txt,val.txt} with relative paths

Usage:
  python scripts/make_wflike_dataset.py \
      --src /absolute/path/to/images \
      --dst /absolute/path/to/wflike \
      --size 512

Dependencies: pillow, numpy

How to run it:
# ensure deps once inside the container (or your env)
pip install --user pillow numpy

# run the script
python scripts/make_wflike_dataset.py \
  --src "$SIHE/data/images" \
  --dst "$SIHE/data/wflike" \
  --size 512

This will create:
$SIHE/data/wflike/
  A/
    2_190.png
    2_190_label.npz
    ...
  valid.txt
  test.txt
  val.txt
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


def find_images(src: Path, exts=(".jpg", ".jpeg", ".png")):
    files = []
    for ext in exts:
        files.extend(sorted(src.glob(f"*{ext}")))
        files.extend(sorted(src.glob(f"*{ext.upper()}")))
    # De-duplicate while preserving order
    seen = set()
    unique = []
    for p in files:
        if p.resolve() not in seen:
            seen.add(p.resolve())
            unique.append(p)
    return unique


def main():
    ap = argparse.ArgumentParser(description="Build a wireframe-like dataset view.")
    ap.add_argument("--src", required=True, type=Path, help="Folder with input images (JPG/PNG).")
    ap.add_argument("--dst", required=True, type=Path, help="Output dataset root (will create <dst>/A).")
    ap.add_argument("--size", type=int, default=512, help="Output square size (default: 512).")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing PNG/NPZ if present.")
    args = ap.parse_args()

    src: Path = args.src
    dst_root: Path = args.dst
    out_size = int(args.size)

    if not src.is_dir():
        print(f"[ERROR] --src not found or not a directory: {src}", file=sys.stderr)
        sys.exit(2)

    # Output layout:
    #   <dst_root>/
    #     A/
    #       <stem>.png
    #       <stem>_label.npz
    #     valid.txt, test.txt, val.txt
    out_A = dst_root / "A"
    out_A.mkdir(parents=True, exist_ok=True)

    images = find_images(src)
    if not images:
        print(f"[WARN] No images found under {src} (expected *.jpg|*.jpeg|*.png)")
        # still write empty split files for reproducibility
        for name in ("valid.txt", "test.txt", "val.txt"):
            (dst_root / name).write_text("", encoding="utf-8")
        sys.exit(0)

    written = 0
    for jf in images:
        try:
            # Load & transpose with EXIF so rotations are correct
            img = Image.open(jf).convert("RGB")
            img = ImageOps.exif_transpose(img)
            img = img.resize((out_size, out_size), Image.BILINEAR)

            stem = jf.stem
            out_png = out_A / f"{stem}.png"
            out_npz = out_A / f"{stem}_label.npz"

            if args.overwrite or not out_png.exists():
                img.save(out_png)

            if args.overwrite or not out_npz.exists():
                # Dummy labels (unused by eval; keeps loader happy)
                np.savez(out_npz, vpts=np.zeros((3, 3), dtype="f4"))

            written += 1
        except Exception as e:
            print(f"[WARN] Failed to process {jf}: {e}", file=sys.stderr)

    # Build split files with RELATIVE paths from <dst_root>
    rels = []
    for png in sorted(out_A.glob("*.png")):
        rels.append(png.relative_to(dst_root).as_posix())  # e.g., "A/2_190.png"

    splits = ["valid.txt", "test.txt", "val.txt"]
    for split in splits:
        (dst_root / split).write_text("\n".join(rels) + ("\n" if rels else ""), encoding="utf-8")

    print(f"[ok] prepared {written} images at {out_A}")
    print(f"[ok] wrote splits: {', '.join(splits)} under {dst_root}")


if __name__ == "__main__":
    main()
