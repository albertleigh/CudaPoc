#!/bin/bash
# Simple script to run NCCL MPI example
# This runs 2 MPI ranks on GPU 0

export CUDA_VISIBLE_DEVICES=0
export NCCL_DEBUG=INFO

echo "=== Running nccl_mpi_example with 2 MPI ranks on 1 GPU ==="
mpirun -np 2 --allow-run-as-root ./cmake-build-debug/example/nccl_mpi_example
