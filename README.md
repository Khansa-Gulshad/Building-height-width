# Building Area (width*Height) estimation

## Repos used
- external/neurvps (pinned commit)
- sihe/misc/vps_models/{config.yaml, neurvps_sihe_checkpoint.pth.tar}

## Code Structure
Building-height-width/
├─ external/
│  └─ neurvps/                # the original NeurVPS repo (pinned)
├─ sihe/
│  └─ misc/vps_models/
│     ├─ config.yaml          # SIHE config (as you had)
│     ├─ neurvps_sihe_checkpoint.pth.tar   # SIHE retrained model (LFS)
│     └─ su3_ds.yaml          # local YAML we generated (datadir -> wflike)
├─ data/
│  ├─ images/                 # your original JPGs (inputs)
│  ├─ wflike/
│  │  ├─ A/                   # 512×512 PNGs + *_label.npz (dummy)
│  │  ├─ valid.txt
│  │  ├─ test.txt
│  │  └─ val.txt
│  └─ vpts/
│     ├─ 000000.npz, ...      # raw eval outputs (index-matched to valid.txt)
│     ├─ su3_error.npz
│     ├─ preds.csv            # our post-processed 2D points (ordered)
│     ├─ json/                # one JSON per image with 2D+3D VPs
│     └─ overlays/            # optional preview PNGs with red crosses
├─ apptainer/
│  └─ README.md               # notes on the CUDA11 container you used
├─ scripts/
│  ├─ 10_prep_wflike.py
│  ├─ 20_eval_neurvps.sh
│  ├─ 30_postprocess_vpts.py
│  └─ 40_overlay_one.py
├─ .gitattributes             # LFS tracking for big files
├─ .gitignore
└─ README.md

##  Apptainer environment (used for GPU eval)

- Module: `trytonp/apptainer/1.3.0`
- SIF built/pulled from: `nvcr.io/nvidia/pytorch:21.11-py3`
- In-container CUDA toolkit: 11.5 (`nvcc --version`)
- PyTorch: 1.11 (CUDA build 11.5)
- Run with `apptainer exec --nv ... pytorch_21.11-py3.sif`

We bind:  
- `$REPO/external/neurvps -> /w/neurvps`  
- `$REPO/sihe         -> /w/SIHE`  
- user base: `/users/scratch1/khansa/.pyuserbase -> /writable`

We install in-container (user-space):  
`docopt "tensorboardX<3" "protobuf<4" yacs pyyaml tqdm opencv-python-headless scikit-image scipy ninja pillow imageio numpy`
