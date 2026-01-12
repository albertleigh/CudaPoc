### CUDA Basic


### setup environment:

CMAKE:
CUDA TOOLKIT: https://developer.nvidia.com/cuda/toolkit
Set up environment variables like:
CUDAToolkit_ROOT = C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.1
CMAKE_CUDA_COMPILER = C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.1/bin/nvcc.exe


### handy commands:

- manully compile
- `nvcc add_cuda.cu -o add_cuda`
- `nvcc --arch <PTX virtual code arch version>, -code <physical machine code arch version>`

- to check gpu usage:
- `watch -n 1 nvidia-smi`
- `while ($true) { cls; nvidia-smi; Start-Sleep -Seconds 1 }`