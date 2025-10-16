#!/usr/bin/env python3
"""
make_splits.py

Create split files (valid.txt, test.txt, val.txt) for a Wireframe-like dataset.
It scans <root>/<subdir> for images (default: *.png) and writes RELATIVE paths.

Usage:
  python scripts/make_splits.py --root /abs/path/to/wflike --subdir A --pattern *.png

  Run it:
  python scripts/make_splits.py \
  --root "$SIHE/data/wflike" \
  --subdir A \
  --pattern "*.png"
"""
import argparse
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path, help="Dataset root (contains split files and subdir).")
    ap.add_argument("--subdir", default="A", help="Subfolder within root that holds the images.")
    ap.add_argument("--pattern", default="*.png", help="Glob pattern for images (default: *.png).")
    args = ap.parse_args()

    root: Path = args.root
    sub: Path = root / args.subdir
    if not sub.is_dir():
        raise SystemExit(f"[ERROR] Not a directory: {sub}")

    # Collect and sort images, then make relative paths from root
    imgs = sorted(sub.glob(args.pattern))
    rels = [p.relative_to(root).as_posix() for p in imgs]

    # Write the three splits (same list is fine for eval/prediction)
    for name in ("valid.txt", "test.txt", "val.txt"):
        (root / name).write_text("\n".join(rels) + ("\n" if rels else ""), encoding="utf-8")
        print(f"[ok] wrote {name} with {len(rels)} entries")

    if rels:
        print("[sample]", *rels[:5], sep="\n  ")

if __name__ == "__main__":
    main()
