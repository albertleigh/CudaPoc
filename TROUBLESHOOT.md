### fix File .venv\Scripts\Activate.ps1 cannot be loaded. The file .venv\Scripts\Activate.ps1 is not digitally signed.
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser -Force

### Override CUDA arch
cmake -DTARGET_CUDA_ARCH=86 -B cmake-build-debug