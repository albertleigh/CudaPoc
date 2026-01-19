//
// Created by Albert Li on 1/19/2026.
//
#include <cuda_runtime.h>
#include <nccl.h>
#include <iostream>
#include <thread>
#include <vector>
#include "cuda_utils.h"

using namespace cuda_poc;

void runRank(int rank, int nRanks, int device, ncclUniqueId id) {
  ncclComm_t comm;
  cudaStream_t stream;
  float *sendbuff, *recvbuff;
  int size = 1024;

  // Initialize communicator for this rank
  CUDA_CHECK(cudaSetDevice(device));
  NCCL_CHECK(ncclCommInitRank(&comm, nRanks, id, rank));

  // Setup memory and stream
  CUDA_CHECK(cudaStreamCreate(&stream));
  CUDA_CHECK(cudaMalloc(&sendbuff, size * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&recvbuff, size * sizeof(float)));

  // Fill buffer with dummy data
  std::vector<float> host_data(size, (float)(rank + 1));
  CUDA_CHECK(cudaMemcpy(sendbuff, host_data.data(), size * sizeof(float), cudaMemcpyHostToDevice));

  // Launch AllReduce
  NCCL_CHECK(ncclAllReduce(sendbuff, recvbuff, size, ncclFloat, ncclSum, comm, stream));

  // Synchronize and verify
  CUDA_CHECK(cudaStreamSynchronize(stream));
  std::cout << "Rank " << rank << " finished AllReduce." << "\n";

  // Verify result
  std::vector<float> host_result(size);
  CUDA_CHECK(cudaMemcpy(host_result.data(), recvbuff, size * sizeof(float), cudaMemcpyDeviceToHost));
  float expected = (nRanks * (nRanks + 1)) / 2.0f;  // Sum of 1+2+...+nRanks
  std::cout << "Rank " << rank << " first value: " << host_result[0] << " (expected: " << expected << ")" << "\n";

  // Cleanup
  CUDA_CHECK(cudaFree(sendbuff));
  CUDA_CHECK(cudaFree(recvbuff));
  CUDA_CHECK(cudaStreamDestroy(stream));
  NCCL_CHECK(ncclCommDestroy(comm));
}

int main() {
  // Check available GPUs
  int deviceCount;
  CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
  std::cout << "Found " << deviceCount << " CUDA device(s)" << "\n";

  // Use min of 2 ranks or available devices
  int nRanks = std::min(2, deviceCount);
  if (nRanks < 2) {
    std::cout << "Note: Running with " << nRanks
              << " rank(s). NCCL requires separate devices for multi-rank in single process." << "\n";
  }

  // Generate unique ID for all ranks
  ncclUniqueId id;
  NCCL_CHECK(ncclGetUniqueId(&id));

  // Launch each rank in a separate thread on different devices
  std::vector<std::thread> threads;
  for (int i = 0; i < nRanks; ++i) {
    threads.emplace_back(runRank, i, nRanks, i, id);  // Each rank on its own device
  }

  // Wait for all ranks to complete
  for (auto& thread : threads) {
    thread.join();
  }

  std::cout << "Success!" << "\n";
  return 0;
}
