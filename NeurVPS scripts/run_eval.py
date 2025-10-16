#!/usr/bin/env python3
"""
run_eval.py — wrapper to run NeurVPS eval.py and dump predictions.

Typical use (inside your Apptainer container with binds in place):
  python scripts/run_eval.py \
    --neurvps /w/neurvps \
    --yaml    /w/SIHE/misc/vps_models/su3_ds.yaml \
    --ckpt    /w/SIHE/misc/vps_models/neurvps_sihe_checkpoint.pth.tar \
    --dump    /w/SIHE/data/vpts \
    --gpus    0 \
    --out     /w/SIHE/data/vpts/su3_error.npz

Optional: if you want to generate the local YAML on the fly and point it
to a dataset root, pass --datadir and --yaml-out:
  python scripts/run_eval.py \
    --neurvps /w/neurvps \
    --yaml    /w/neurvps/config/su3.yaml \
    --yaml-out /w/SIHE/misc/vps_models/su3_ds.yaml \
    --datadir /w/SIHE/data/wflike \
    --ckpt    /w/SIHE/misc/vps_models/neurvps_sihe_checkpoint.pth.tar \
    --dump    /w/SIHE/data/vpts \
    --gpus    0
"""
import argparse, subprocess, sys, os
from pathlib import Path

def make_local_yaml(src_yaml: Path, out_yaml: Path, datadir: str):
    import yaml
    out_yaml.parent.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(src_yaml.read_text()) or {}
    cfg.setdefault("io", {})
    cfg["io"]["datadir"] = datadir
    out_yaml.write_text(yaml.safe_dump(cfg, sort_keys=False))
    print(f"[cfg] wrote {out_yaml} (io.datadir={datadir})")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--neurvps", required=True, help="Path to NeurVPS repo root (has eval.py)")
    ap.add_argument("--yaml", required=True, help="Path to YAML (su3.yaml or your local su3_ds.yaml)")
    ap.add_argument("--ckpt", required=True, help="Path to checkpoint .pth/.tar")
    ap.add_argument("--dump", required=True, help="Output dir for per-image .npz")
    ap.add_argument("--out", default=None, help="Optional AA-curve .npz (default: <dump>/su3_error.npz)")
    ap.add_argument("--gpus", default="0", help="Comma-separated GPU ids (default 0)")
    # optional on-the-fly YAML clone:
    ap.add_argument("--yaml-out", default=None, help="Write a local YAML here (clone of --yaml)")
    ap.add_argument("--datadir", default=None, help="If given, set io.datadir in --yaml-out")
    args = ap.parse_args()

    eval_py = Path(args.neurvps) / "eval.py"
    if not eval_py.exists():
        print(f"[error] eval.py not found at {eval_py}", file=sys.stderr)
        sys.exit(1)

    yaml_to_use = Path(args.yaml)
    if args.yaml_out and args.datadir:
        # create a local YAML with datadir set
        yaml_to_use = Path(args.yaml_out)
        make_local_yaml(Path(args.yaml), yaml_to_use, args.datadir)

    dump_dir = Path(args.dump)
    dump_dir.mkdir(parents=True, exist_ok=True)
    out_curve = args.out or str(dump_dir / "su3_error.npz")

    cmd = [
        sys.executable, str(eval_py),
        str(yaml_to_use),
        str(args.ckpt),
        "--dump", str(dump_dir),
        "-o",     str(out_curve),
        "-d",     str(args.gpus),
    ]
    print("[run]", " ".join(cmd))
    r = subprocess.run(cmd)
    if r.returncode != 0:
        sys.exit(r.returncode)

    # quick summary
    created = sorted(p.name for p in dump_dir.glob("*.npz"))
    print(f"[ok] wrote {len(created)} files to {dump_dir}")
    for n in created[:10]:
        print("  ", n)
    if len(created) > 10:
        print("  ...")

if __name__ == "__main__":
    main()
