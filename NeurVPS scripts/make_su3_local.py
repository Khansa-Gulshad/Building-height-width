#!/usr/bin/env python3
"""
make_su3_local.py

Clone NeurVPS's su3.yaml into a local YAML and point it to your dataset root.
You can also optionally override focal length and number of VPs.

Usage:
  python scripts/make_su3_local.py \
    --src /abs/path/to/neurvps/config/su3.yaml \
    --out /abs/path/to/SIHE/misc/vps_models/su3_ds.yaml \
    --datadir /abs/path/to/SIHE/data/wflike \
    [--focal 2.1875] \
    [--num_vpts 3]

    Run it like this:
    python scripts/make_su3_local.py \
  --src   "$NEURVPS/config/su3.yaml" \
  --out   "$SIHE/misc/vps_models/su3_ds.yaml" \
  --datadir "$SIHE/data/wflike"
"""
import argparse
import yaml
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="Path to NeurVPS su3.yaml")
    ap.add_argument("--out", required=True, help="Where to write the local YAML")
    ap.add_argument("--datadir", required=True, help="Dataset root to set in YAML")
    ap.add_argument("--focal", type=float, default=None, help="Optional focal_length override")
    ap.add_argument("--num_vpts", type=int, default=None, help="Optional num_vpts override (usually 3)")
    args = ap.parse_args()

    src = Path(args.src)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    cfg = yaml.safe_load(src.read_text()) or {}
    # Ensure expected sections exist
    cfg.setdefault("io", {})
    cfg["io"]["datadir"] = args.datadir
    if args.focal is not None:
        cfg["io"]["focal_length"] = float(args.focal)
    if args.num_vpts is not None:
        cfg["io"]["num_vpts"] = int(args.num_vpts)

    out.write_text(yaml.safe_dump(cfg, sort_keys=False))
    print(f"[ok] wrote {out}")
    print(f"  io.datadir     = {cfg['io']['datadir']}")
    if args.focal is not None:
        print(f"  io.focal_length= {cfg['io']['focal_length']}")
    if args.num_vpts is not None:
        print(f"  io.num_vpts    = {cfg['io']['num_vpts']}")

if __name__ == "__main__":
    main()
