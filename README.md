# Building Area (width*Height) estimation

## Code Structure

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
