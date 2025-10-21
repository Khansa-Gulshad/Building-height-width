# Building Area (Width*Height) estimation

## Repos used
- external/neurvps (pinned commit)
- sihe/misc/vps_models/{config.yaml, neurvps_sihe_checkpoint.pth.tar}

## Code Structure
```
Gdańsk, Poland/                 # example results
  
NeurVPS scripts/                # small utilities you ran/keep for reproducibility
  prep_wflike.py                # JPG -> 512×512 PNG + split lists
  make_su3_local.py             # NeurVPS's su3.yaml into a local YAML
  run_eval.py                   # wrapper to run NeurVPS eval.py and dump predictions
  vpt_postprocess.py            # 3D→2D VP transform (FOV, ordering, scaling)

config/
    estimation_config.ini
data/                           # default folder for placing the data
  images/                       # original street-view JPG images (inputs)
  wflike/                       # "wireframe-like" dataset view for NeurVPS
    valid.txt                   # split list (relative paths like A/xxx.png)
    test.txt
    val.txt
  vpts/                         # outputs from NeurVPS + post-processing
    000000.npz ...              # raw model outputs (contain vpts_pd)
    su3_error.npz               # AA curve file (unused for you; safe to ignore)
    json/                       # per-image 2D VP results (ordered + scaled)
      <image_stem>.json
    overlays/                   # optional visualization overlays (PNG)

external/                       # third-party code
  neurvps/                      # NeurVPS repo at commit 72d9502

modules/
  process_data.py              # to fetch and segment street view images
  road_network.py              # to fetch road network
  segmentation.py              # segmentation classes
  
sihe/                           # SIHE model + configs used
  vps_models/
    su3_ds.yaml                       # configuration (NeurVPS YAMLs that we used)
    neurvps_sihe_checkpoint.pth.tar   # SIHE retrained model (tracked with Git LFS)
    config.yaml                       # SIHE’s reference config (kept for record)
```


##  Apptainer environment (used for GPU eval)

- Module: `trytonp/apptainer/1.3.0`
- SIF built/pulled from: `nvcr.io/nvidia/pytorch:21.11-py3`
- In-container CUDA toolkit: 11.5 (`nvcc --version`)
- PyTorch: 1.11 (CUDA build 11.5)
- Run with `apptainer exec --nv ... pytorch_21.11-py3.sif`

We bind:  
- `$REPO/external/neurvps` -> /w/[neurvps](https://github.com/zhou13/neurvps)
- `$REPO/sihe`         -> /w/[SIHE](https://github.com/yzre/SIHE?tab=readme-ov-file)


We install in-container (user-space):  
`docopt "tensorboardX<3" "protobuf<4" yacs pyyaml tqdm opencv-python-headless scikit-image scipy ninja pillow imageio numpy`
