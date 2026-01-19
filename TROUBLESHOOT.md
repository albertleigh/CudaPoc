### fix File .venv\Scripts\Activate.ps1 cannot be loaded. The file .venv\Scripts\Activate.ps1 is not digitally signed.
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser -Force

### Override CUDA arch
cmake -DTARGET_CUDA_ARCH=86 -B cmake-build-debug
or replace `set(TARGET_CUDA_ARCH "86" CACHE STRING "Target CUDA Architecture")` with `set(TARGET_CUDA_ARCH "75")` in cmake_config.cmake.