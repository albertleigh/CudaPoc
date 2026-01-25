#!/bin/bash
# Script to run NCCL MPI example with CUDA MPS (with sudo)
# This allows multiple MPI processes to share a single GPU efficiently

# 1. Export the GPU you want to use
export CUDA_VISIBLE_DEVICES=0
export NCCL_DEBUG=INFO

# Set MPS pipe directory (required for MPS)
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log
sudo mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY" 2>/dev/null
sudo chmod 777 "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY" 2>/dev/null

# Find MPS control binary
MPS_CONTROL="/usr/bin/nvidia-cuda-mps-control"
if [ ! -f "$MPS_CONTROL" ]; then
    echo "ERROR: nvidia-cuda-mps-control not found at $MPS_CONTROL"
    exit 1
fi

echo "=== Starting CUDA MPS Control Daemon (with sudo) ==="
echo "Using MPS control: $MPS_CONTROL"

# Stop any existing MPS daemon
echo quit | sudo "$MPS_CONTROL" 2>/dev/null || true
sleep 1

# Start MPS daemon with sudo
echo "Attempting to start MPS daemon with sudo..."
MPS_OUTPUT=$(sudo "$MPS_CONTROL" -d 2>&1)
MPS_STATUS=$?

if [ $MPS_STATUS -ne 0 ]; then
    echo "ERROR: Failed to start MPS daemon (exit code: $MPS_STATUS)"
    echo "Error output:"
    echo "$MPS_OUTPUT"
    echo ""
    echo "Checking MPS log for more details..."
    if [ -f "$CUDA_MPS_LOG_DIRECTORY/control.log" ]; then
        echo "--- MPS Control Log ---"
        sudo cat "$CUDA_MPS_LOG_DIRECTORY/control.log" 2>/dev/null || echo "Could not read log"
    fi
    echo ""
    echo "MPS may not be supported on your system or GPU may be in use."
    exit 1
fi

echo "MPS daemon started successfully"

# 2. Run the MPI program with 2 processes
echo ""
echo "=== Running nccl_mpi_example with 2 MPI ranks on 1 GPU ==="
mpirun -np 2 --allow-run-as-root ./cmake-build-debug/example/nccl_mpi_example

# 3. Cleanup: Stop MPS daemon
echo ""
echo "=== Stopping CUDA MPS Control Daemon ==="
echo quit | sudo "$MPS_CONTROL"
echo "MPS daemon stopped"

echo ""
echo "=== Success! ==="
