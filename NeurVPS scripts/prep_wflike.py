#!/usr/bin/env python3
import os, glob, numpy as np
from PIL import Image

SRC = "data/images"          # your JPGs
DST = "data/wflike/A"        # dataset-style folder
os.makedirs(DST, exist_ok=True)

n = 0
for jf in sorted(glob.glob(os.path.join(SRC, "*.jpg"))):
    im = Image.open(jf).convert("RGB").resize((512, 512), Image.BILINEAR)
    base = os.path.splitext(os.path.basename(jf))[0]
    pf = os.path.join(DST, base + ".png")
    im.save(pf)
    np.savez(pf.replace(".png", "_label.npz"), vpts=np.zeros((3,3), "f4"))  # dummy
    n += 1

# split files expected by NeurVPS Wireframe loader
root = "data/wflike"
rel = [os.path.join("A", os.path.basename(p)) for p in sorted(glob.glob(os.path.join(DST, "*.png")))]
for name in ("valid.txt", "test.txt", "val.txt"):
    with open(os.path.join(root, name), "w") as f:
        f.write("\n".join(rel) + "\n")
print(f"[ok] prepared {n} images and splits under {root}")
