### CUDA Basic

https://en.wikipedia.org/wiki/CUDA#GPUs_supported

| CUDA SDK Version  |    Tesla    | Fermi | Kepler (Early) | Kepler (Late) | Maxwell | Pascal |   Volta   | Turing | Ampere | Ada Lovelace | Hopper | Blackwell |
|:------------------|:-----------:|:-----:|:--------------:|:-------------:|:-------:|:------:|:---------:|:------:|:------:|:------------:|:------:|:---------:|
| **1.0**           |  1.0 – 1.1  |       |                |               |         |        |           |        |        |              |        |           |
| **1.1**           | 1.0 – 1.1+x |       |                |               |         |        |           |        |        |              |        |           |
| **2.0**           | 1.0 – 1.1+x |       |                |               |         |        |           |        |        |              |        |           |
| **2.1 – 2.3.1**   |  1.0 – 1.3  |       |                |               |         |        |           |        |        |              |        |           |
| **3.0 – 3.1**     |     1.0     |  2.0  |                |               |         |        |           |        |        |              |        |           |
| **3.2**           |     1.0     |  2.1  |                |               |         |        |           |        |        |              |        |           |
| **4.0 – 4.2**     |     1.0     |  2.1  |                |               |         |        |           |        |        |              |        |           |
| **5.0 – 5.5**     |     1.0     |       |      3.0       |      3.5      |         |        |           |        |        |              |        |           |
| **6.0**           |     1.0     |       |      3.2       |      3.5      |         |        |           |        |        |              |        |           |
| **6.5**           |     1.1     |       |                |      3.7      |   5.x   |        |           |        |        |              |        |           |
| **7.0 – 7.5**     |             |  2.0  |                |               |   5.x   |        |           |        |        |              |        |           |
| **8.0**           |             |  2.0  |                |               |         |  6.x   |           |        |        |              |        |           |
| **9.0 – 9.2**     |             |       |      3.0       |               |         |        | 7.0 – 7.2 |        |        |              |        |           |
| **10.0 – 10.2**   |             |       |      3.0       |               |         |        |           |  7.5   |        |              |        |           |
| **11.0**          |             |       |                |      3.5      |         |        |           |        |  8.0   |              |        |           |
| **11.1 – 11.4**   |             |       |                |      3.5      |         |        |           |        |  8.6   |              |        |           |
| **11.5 – 11.7.1** |             |       |                |      3.5      |         |        |           |        |  8.7   |              |        |           |
| **11.8**          |             |       |                |      3.5      |         |        |           |        |        |     8.9      |  9.0   |           |
| **12.0 – 12.6**   |             |       |                |               |   5.0   |        |           |        |        |              |  9.0   |           |
| **12.8**          |             |       |                |               |   5.0   |        |           |        |        |              |        |   12.0    |
| **12.9**          |             |       |                |               |   5.0   |        |           |        |        |              |        |   12.1    |
| **13.0**          |             |       |                |               |         |        |           |  7.5   |        |              |        |   12.1    |

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

- Nsight Compute(ncu) bash:
- sudo ${ncu} --print-details --nvtx --call-stack --set full ./add_cuda
- sudo ${ncu} --nvtx --call-stack --set full -f --export add_cuda.ncu-rep ./add_cuda
- Nsight Compute(ncu) pwsh administrator:
- & $ncu --print-details --nvtx --call-stack --set full .\cuda_poc.exe
- & $ncu --nvtx --call-stack --set full -f --export add_cuda.ncu-rep .\cuda_poc.exe
- Nsight Compute(ncu) cmd administrator:
- ncu --print-details --nvtx --call-stack --set full cuda_poc.exe
- ncu --nvtx --call-stack --set full -f --export add_cuda.ncu-rep cuda_poc.exe
- Nsight Compute(ncu) cmd administrator Roofline Chart only:
- ncu --print-details=all --section SpeedOfLight_RooflineChart cuda_poc.exe

### Sample roofline chart:
![Roofline Chart](./doc/roofline_chart_1.png)
