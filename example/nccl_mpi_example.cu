//
// Created by Albert Li on 1/20/26.
//
#include <mpi.h>
#include <nccl.h>
#include <cuda_runtime.h>

__global__ void my_kernel(float* data, int rank) {
  // A dummy workload to keep the GPU busy
  for (int i = 0; i < 1000000; i++) {
    data[0] += (float)rank;
  }
}

int main(int argc, char* argv[]) {
  MPI_Init(&argc, &argv);
  int rank, nranks;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nranks);

  // Both ranks target GPU 0
  cudaSetDevice(0);

  // Generate NCCL ID on rank 0 and broadcast it
  ncclUniqueId id;
  if (rank == 0) ncclGetUniqueId(&id);
  MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);

  ncclComm_t comm;
  cudaStream_t s;
  ncclCommInitRank(&comm, nranks, id, rank);
  cudaStreamCreate(&s);

  float *d_ptr;
  cudaMalloc(&d_ptr, sizeof(float));

  // Parallel Execution: Each process launches a kernel
  my_kernel<<<1, 1, 0, s>>>(d_ptr, rank);

  // NCCL Collective (AllReduce)
  ncclAllReduce(d_ptr, d_ptr, 1, ncclFloat, ncclSum, comm, s);

  cudaStreamSynchronize(s);
  printf("Rank %d complete\n", rank);

  ncclCommDestroy(comm);
  MPI_Finalize();
  return 0;
}