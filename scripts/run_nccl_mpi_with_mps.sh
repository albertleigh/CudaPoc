#!/bin/bash
# Script to run NCCL MPI example with CUDA MPS
# This allows multiple MPI processes to share a single GPU efficiently

# Don't exit on error for MPS setup
set +e

# 1. Export the GPU you want to use
export CUDA_VISIBLE_DEVICES=0
export NCCL_DEBUG=INFO

# Set MPS pipe directory (required for MPS)
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY" 2>/dev/null

# Find MPS control binary
MPS_CONTROL=""
for cuda_path in /usr /usr/local/cuda /usr/local/cuda-* /usr/lib/nvidia-cuda-toolkit; do
    if [ -f "$cuda_path/bin/nvidia-cuda-mps-control" ]; then
        MPS_CONTROL="$cuda_path/bin/nvidia-cuda-mps-control"
        break
    fi
done

if [ -z "$MPS_CONTROL" ]; then
    echo "WARNING: nvidia-cuda-mps-control not found"
    echo "MPS is not available on this system. Running without MPS..."
    echo ""
    echo "=== Running nccl_mpi_example with 2 MPI ranks on 1 GPU (without MPS) ==="
    mpirun -np 2 --allow-run-as-root ./cmake-build-debug/example/nccl_mpi_example
    exit 0
fi

echo "=== Starting CUDA MPS Control Daemon ==="
echo "Using MPS control: $MPS_CONTROL"

# Check GPU compute mode
GPU_MODE=$(nvidia-smi --query-gpu=compute_mode --format=csv,noheader 2>/dev/null | head -1)
echo "GPU compute mode: $GPU_MODE"

if [ "$GPU_MODE" = "Exclusive_Process" ]; then
    echo "WARNING: GPU is in Exclusive_Process mode. MPS requires Default mode."
    echo "To fix: sudo nvidia-smi -c DEFAULT"
    echo "Running without MPS for now..."
    echo ""
    echo "=== Running nccl_mpi_example with 2 MPI ranks on 1 GPU (without MPS) ==="
    mpirun -np 2 --allow-run-as-root ./cmake-build-debug/example/nccl_mpi_example
    exit 0
fi

# Stop any existing MPS daemon
echo quit | "$MPS_CONTROL" 2>/dev/null || true
sleep 1

# Start MPS daemon
echo "Attempting to start MPS daemon..."
MPS_OUTPUT=$("$MPS_CONTROL" -d 2>&1)
MPS_STATUS=$?

if [ $MPS_STATUS -ne 0 ]; then
    echo "WARNING: Failed to start MPS daemon (exit code: $MPS_STATUS)"
    echo "Output: $MPS_OUTPUT"
    echo ""
    echo "Possible reasons:"
    echo "  - MPS may require root privileges (try: sudo)"
    echo "  - GPU may be in use by another process"
    echo "  - GPU compute mode may need adjustment"
    echo ""
    echo "Running without MPS..."
    echo ""
    echo "=== Running nccl_mpi_example with 2 MPI ranks on 1 GPU (without MPS) ==="
    mpirun -np 2 --allow-run-as-root ./cmake-build-debug/example/nccl_mpi_example
    exit 0
fi

echo "MPS daemon started successfully"

# 2. Run the MPI program with 2 processes
echo ""
echo "=== Running nccl_mpi_example with 2 MPI ranks on 1 GPU ==="
mpirun -np 2 --allow-run-as-root ./cmake-build-debug/example/nccl_mpi_example

# 3. Cleanup: Stop MPS daemon
echo ""
echo "=== Stopping CUDA MPS Control Daemon ==="
echo quit | "$MPS_CONTROL"
echo "MPS daemon stopped"

echo ""
echo "=== Success! ==="
