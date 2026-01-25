//
// Created by ali on 1/25/26.
//
#include <gtest/gtest.h>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include "device_cuda.cuh"

namespace cuda_poc::day05cu {
using namespace cuda_poc::device;

// Helper to format bytes in hexadecimal
inline std::string to_hex(uint64_t value, int bytes) {
  std::ostringstream oss;
  oss << "0x" << std::hex << std::setfill('0');
  if (bytes == 8) {
    oss << std::setw(16) << value;
  } else if (bytes == 4) {
    oss << std::setw(8) << (value & 0xFFFFFFFF);
  } else if (bytes == 2) {
    oss << std::setw(4) << (value & 0xFFFF);
  }
  return oss.str();
}

// Kernel to test HtoBE64
__global__ void test_HtoBE64_kernel(const uint64_t* input, uint64_t* output, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    output[idx] = HtoBE64(input[idx]);
  }
}

// Kernel to test HtoBE32
__global__ void test_HtoBE32_kernel(const uint32_t* input, uint32_t* output, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    output[idx] = HtoBE32(input[idx]);
  }
}

// Kernel to test HtoBE16
__global__ void test_HtoBE16_kernel(const uint16_t* input, uint16_t* output, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    output[idx] = HtoBE16(input[idx]);
  }
}

class CudaPoc_Day06cu02 : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    std::cout << "\n========================================" << std::endl;
    std::cout << "  Device Byte Order Conversion Tests" << std::endl;
    std::cout << "========================================\n" << std::endl;
  }
};

TEST_F(CudaPoc_Day06cu02, HtoBE64_BasicConversion) {
  std::cout << "\n=== Testing HtoBE64: 64-bit Byte Order Conversion ===\n" << std::endl;

  // Test cases: input -> expected output (big-endian)
  std::vector<std::pair<uint64_t, uint64_t>> test_cases = {
      {0x0123456789ABCDEFULL, 0xEFCDAB8967452301ULL},  // Full byte reversal
      {0x0000000000000001ULL, 0x0100000000000000ULL},  // Single byte
      {0x00000000FFFFFFFFULL, 0xFFFFFFFF00000000ULL},  // Lower 32-bit set
      {0xFFFFFFFF00000000ULL, 0x00000000FFFFFFFFULL},  // Upper 32-bit set
      {0x0000000000000000ULL, 0x0000000000000000ULL},  // Zero
      {0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL},  // All bits set
  };

  const int n = test_cases.size();
  std::vector<uint64_t> h_input(n);
  std::vector<uint64_t> h_output(n);
  std::vector<uint64_t> h_expected(n);

  for (int i = 0; i < n; ++i) {
    h_input[i] = test_cases[i].first;
    h_expected[i] = test_cases[i].second;
  }

  // Allocate device memory
  uint64_t *d_input, *d_output;
  cudaMalloc(&d_input, n * sizeof(uint64_t));
  cudaMalloc(&d_output, n * sizeof(uint64_t));

  // Copy to device
  cudaMemcpy(d_input, h_input.data(), n * sizeof(uint64_t), cudaMemcpyHostToDevice);

  // Launch kernel
  test_HtoBE64_kernel<<<1, n>>>(d_input, d_output, n);
  cudaDeviceSynchronize();

  // Copy results back
  cudaMemcpy(h_output.data(), d_output, n * sizeof(uint64_t), cudaMemcpyDeviceToHost);

  // Verify and display results
  std::cout << "Input (Host) -> Output (Big-Endian) [Expected]" << std::endl;
  std::cout << "------------------------------------------------" << std::endl;
  for (int i = 0; i < n; ++i) {
    std::cout << to_hex(h_input[i], 8) << " -> " << to_hex(h_output[i], 8) << " [" << to_hex(h_expected[i], 8) << "]";
    if (h_output[i] == h_expected[i]) {
      std::cout << " ✓" << std::endl;
      EXPECT_EQ(h_output[i], h_expected[i]);
    } else {
      std::cout << " ✗" << std::endl;
      FAIL() << "Mismatch at index " << i;
    }
  }

  // Cleanup
  cudaFree(d_input);
  cudaFree(d_output);
}

TEST_F(CudaPoc_Day06cu02, HtoBE32_BasicConversion) {
  std::cout << "\n=== Testing HtoBE32: 32-bit Byte Order Conversion ===\n" << std::endl;

  // Test cases: input -> expected output (big-endian)
  std::vector<std::pair<uint32_t, uint32_t>> test_cases = {
      {0x01234567, 0x67452301},  // Full byte reversal
      {0x00000001, 0x01000000},  // Single byte
      {0x000000FF, 0xFF000000},  // Single byte at LSB
      {0xFF000000, 0x000000FF},  // Single byte at MSB
      {0x0000FFFF, 0xFFFF0000},  // Lower 16-bit set
      {0xFFFF0000, 0x0000FFFF},  // Upper 16-bit set
      {0x00000000, 0x00000000},  // Zero
      {0xFFFFFFFF, 0xFFFFFFFF},  // All bits set
  };

  const int n = test_cases.size();
  std::vector<uint32_t> h_input(n);
  std::vector<uint32_t> h_output(n);
  std::vector<uint32_t> h_expected(n);

  for (int i = 0; i < n; ++i) {
    h_input[i] = test_cases[i].first;
    h_expected[i] = test_cases[i].second;
  }

  // Allocate device memory
  uint32_t *d_input, *d_output;
  cudaMalloc(&d_input, n * sizeof(uint32_t));
  cudaMalloc(&d_output, n * sizeof(uint32_t));

  // Copy to device
  cudaMemcpy(d_input, h_input.data(), n * sizeof(uint32_t), cudaMemcpyHostToDevice);

  // Launch kernel
  test_HtoBE32_kernel<<<1, n>>>(d_input, d_output, n);
  cudaDeviceSynchronize();

  // Copy results back
  cudaMemcpy(h_output.data(), d_output, n * sizeof(uint32_t), cudaMemcpyDeviceToHost);

  // Verify and display results
  std::cout << "Input (Host) -> Output (Big-Endian) [Expected]" << std::endl;
  std::cout << "------------------------------------------------" << std::endl;
  for (int i = 0; i < n; ++i) {
    std::cout << to_hex(h_input[i], 4) << " -> " << to_hex(h_output[i], 4) << " [" << to_hex(h_expected[i], 4) << "]";
    if (h_output[i] == h_expected[i]) {
      std::cout << " ✓" << std::endl;
      EXPECT_EQ(h_output[i], h_expected[i]);
    } else {
      std::cout << " ✗" << std::endl;
      FAIL() << "Mismatch at index " << i;
    }
  }

  // Cleanup
  cudaFree(d_input);
  cudaFree(d_output);
}

TEST_F(CudaPoc_Day06cu02, HtoBE16_BasicConversion) {
  std::cout << "\n=== Testing HtoBE16: 16-bit Byte Order Conversion ===\n" << std::endl;

  // Test cases: input -> expected output (big-endian)
  std::vector<std::pair<uint16_t, uint16_t>> test_cases = {
      {0x0123, 0x2301},  // Full byte reversal
      {0x0001, 0x0100},  // Single byte
      {0x00FF, 0xFF00},  // Lower byte set
      {0xFF00, 0x00FF},  // Upper byte set
      {0x0000, 0x0000},  // Zero
      {0xFFFF, 0xFFFF},  // All bits set
      {0xABCD, 0xCDAB},  // Arbitrary pattern
  };

  const int n = test_cases.size();
  std::vector<uint16_t> h_input(n);
  std::vector<uint16_t> h_output(n);
  std::vector<uint16_t> h_expected(n);

  for (int i = 0; i < n; ++i) {
    h_input[i] = test_cases[i].first;
    h_expected[i] = test_cases[i].second;
  }

  // Allocate device memory
  uint16_t *d_input, *d_output;
  cudaMalloc(&d_input, n * sizeof(uint16_t));
  cudaMalloc(&d_output, n * sizeof(uint16_t));

  // Copy to device
  cudaMemcpy(d_input, h_input.data(), n * sizeof(uint16_t), cudaMemcpyHostToDevice);

  // Launch kernel
  test_HtoBE16_kernel<<<1, n>>>(d_input, d_output, n);
  cudaDeviceSynchronize();

  // Copy results back
  cudaMemcpy(h_output.data(), d_output, n * sizeof(uint16_t), cudaMemcpyDeviceToHost);

  // Verify and display results
  std::cout << "Input (Host) -> Output (Big-Endian) [Expected]" << std::endl;
  std::cout << "------------------------------------------------" << std::endl;
  for (int i = 0; i < n; ++i) {
    std::cout << to_hex(h_input[i], 2) << " -> " << to_hex(h_output[i], 2) << " [" << to_hex(h_expected[i], 2) << "]";
    if (h_output[i] == h_expected[i]) {
      std::cout << " ✓" << std::endl;
      EXPECT_EQ(h_output[i], h_expected[i]);
    } else {
      std::cout << " ✗" << std::endl;
      FAIL() << "Mismatch at index " << i;
    }
  }

  // Cleanup
  cudaFree(d_input);
  cudaFree(d_output);
}

TEST_F(CudaPoc_Day06cu02, RoundTrip_Conversion) {
  std::cout << "\n=== Testing Round-Trip Conversion ===\n" << std::endl;
  std::cout << "Converting to big-endian and back should yield original value\n" << std::endl;

  // Test 64-bit round trip
  uint64_t test64 = 0x0123456789ABCDEFULL;
  uint64_t* d_test64;
  cudaMalloc(&d_test64, sizeof(uint64_t));
  cudaMemcpy(d_test64, &test64, sizeof(uint64_t), cudaMemcpyHostToDevice);

  // Convert twice
  test_HtoBE64_kernel<<<1, 1>>>(d_test64, d_test64, 1);
  test_HtoBE64_kernel<<<1, 1>>>(d_test64, d_test64, 1);
  cudaDeviceSynchronize();

  uint64_t result64;
  cudaMemcpy(&result64, d_test64, sizeof(uint64_t), cudaMemcpyDeviceToHost);

  std::cout << "64-bit: " << to_hex(test64, 8) << " -> ... -> " << to_hex(result64, 8);
  if (result64 == test64) {
    std::cout << " ✓" << std::endl;
    EXPECT_EQ(result64, test64);
  } else {
    std::cout << " ✗" << std::endl;
    FAIL();
  }

  cudaFree(d_test64);

  // Test 32-bit round trip
  uint32_t test32 = 0x01234567;
  uint32_t* d_test32;
  cudaMalloc(&d_test32, sizeof(uint32_t));
  cudaMemcpy(d_test32, &test32, sizeof(uint32_t), cudaMemcpyHostToDevice);

  test_HtoBE32_kernel<<<1, 1>>>(d_test32, d_test32, 1);
  test_HtoBE32_kernel<<<1, 1>>>(d_test32, d_test32, 1);
  cudaDeviceSynchronize();

  uint32_t result32;
  cudaMemcpy(&result32, d_test32, sizeof(uint32_t), cudaMemcpyDeviceToHost);

  std::cout << "32-bit: " << to_hex(test32, 4) << " -> ... -> " << to_hex(result32, 4);
  if (result32 == test32) {
    std::cout << " ✓" << std::endl;
    EXPECT_EQ(result32, test32);
  } else {
    std::cout << " ✗" << std::endl;
    FAIL();
  }

  cudaFree(d_test32);

  // Test 16-bit round trip
  uint16_t test16 = 0x0123;
  uint16_t* d_test16;
  cudaMalloc(&d_test16, sizeof(uint16_t));
  cudaMemcpy(d_test16, &test16, sizeof(uint16_t), cudaMemcpyHostToDevice);

  test_HtoBE16_kernel<<<1, 1>>>(d_test16, d_test16, 1);
  test_HtoBE16_kernel<<<1, 1>>>(d_test16, d_test16, 1);
  cudaDeviceSynchronize();

  uint16_t result16;
  cudaMemcpy(&result16, d_test16, sizeof(uint16_t), cudaMemcpyDeviceToHost);

  std::cout << "16-bit: " << to_hex(test16, 2) << " -> ... -> " << to_hex(result16, 2);
  if (result16 == test16) {
    std::cout << " ✓" << std::endl;
    EXPECT_EQ(result16, test16);
  } else {
    std::cout << " ✗" << std::endl;
    FAIL();
  }

  cudaFree(d_test16);
}

TEST_F(CudaPoc_Day06cu02, BytePattern_Visualization) {
  std::cout << "\n=== Byte Pattern Visualization ===\n" << std::endl;
  std::cout << "Showing byte-by-byte transformation\n" << std::endl;

  uint64_t test64 = 0x0123456789ABCDEFULL;
  uint64_t *d_input64, *d_output64;
  cudaMalloc(&d_input64, sizeof(uint64_t));
  cudaMalloc(&d_output64, sizeof(uint64_t));
  cudaMemcpy(d_input64, &test64, sizeof(uint64_t), cudaMemcpyHostToDevice);

  test_HtoBE64_kernel<<<1, 1>>>(d_input64, d_output64, 1);
  cudaDeviceSynchronize();

  uint64_t result64;
  cudaMemcpy(&result64, d_output64, sizeof(uint64_t), cudaMemcpyDeviceToHost);

  std::cout << "64-bit transformation:" << std::endl;
  std::cout << "  Input:  ";
  for (int i = 7; i >= 0; --i) {
    std::cout << std::hex << std::setw(2) << std::setfill('0') << ((test64 >> (i * 8)) & 0xFF) << " ";
  }
  std::cout << "\n  Output: ";
  for (int i = 7; i >= 0; --i) {
    std::cout << std::hex << std::setw(2) << std::setfill('0') << ((result64 >> (i * 8)) & 0xFF) << " ";
  }
  std::cout << std::dec << "\n" << std::endl;

  cudaFree(d_input64);
  cudaFree(d_output64);
}

}  // namespace cuda_poc::day05cu