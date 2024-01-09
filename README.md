# test-swin

## Usage

1. Ninja (optional but recommended)
2. C++ compiler and CUDA compiler compatible among them (see for a matrix of compatible versions https://gist.github.com/ax3l/9489132#nvcc)
3. Python headers available:
   - you can install `python3-dev` packages or similar
   - you can set the CPATH environment variable to point to the python headers directory
   - you can use `pyenv exec pip install ...` when installing this packages
4. `pdm` (consider installing it via `pipx`)
5. `pdm install --no-self --no-isolation`
6. Download the weights from the SwinDocSegmenter repo
7. Fix the paths to the weights in `test.py`
8. Run `pdm run test.py`
9. Check the `test.png` image

N.B. Tensorflow may requires NCCL (not sure)

## Examples

![Newspaper](./example_segmented0.png)

![Cortonese](./example_segmented1.png)
