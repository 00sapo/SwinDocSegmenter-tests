# test-swin

## Requirements

1. Ninja (optional but recommended)
2. C++ compiler and CUDA compiler compatible among them (see for a matrix of compatible versions https://gist.github.com/ax3l/9489132#nvcc)
3. Python headers available:
   - you can install `python3-dev` packages or similar
   - you can set the CPATH environment variable to point to the python headers directory
   - you can use `pyenv exec pip install ...` when installing this packagehttps://gist.github.com/ax3l/9489132#nvcc

N.B. Tensorflow (DE-GAN) may requires NCCL
