//
// Created by Albert Li on 1/18/2026.
//
#include <gtest/gtest.h>
#include <cute/layout.hpp>
#include <cute/swizzle.hpp>
#include <cute/tensor.hpp>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

// Helper to print 8-bit binary representation (nvcc may not support std::bitset)
inline std::string to_binary8(int value) {
  std::string result(8, '0');
  for (int i = 7; i >= 0; --i) {
    result[7 - i] = ((value >> i) & 1) ? '1' : '0';
  }
  return result;
}

namespace cuda_poc::day05cu {
using namespace cute;

// Utility class for visualizing swizzle patterns
class SwizzleVisualizer {
 public:
  // Print a bank conflict matrix showing which bank each address maps to
  template <int B, int M, int S>
  static void print_bank_map(int rows, int cols, int element_bytes = 4) {
    Swizzle<B, M, S> swizzle;

    std::cout << "\n=== Bank Map for Swizzle<" << B << "," << M << "," << S << "> ===\n";
    std::cout << "Matrix: " << rows << "x" << cols << ", Element size: " << element_bytes << " bytes\n";
    std::cout << "Total banks: 32 (4-byte banks)\n\n";

    // Create a matrix to store bank numbers
    std::vector<std::vector<int>> bank_matrix(rows, std::vector<int>(cols));

    // Calculate bank for each position
    for (int y = 0; y < rows; y++) {
      for (int x = 0; x < cols; x++) {
        int linear_idx = y * cols + x;
        int swizzled_idx = swizzle(linear_idx);
        int byte_offset = swizzled_idx * element_bytes;
        int bank = (byte_offset / 4) % 32;  // 32 banks, 4 bytes per bank
        bank_matrix[y][x] = bank;
      }
    }

    // Print the matrix with color-coding
    print_matrix_formatted(bank_matrix, rows, cols);
  }

  // Print before and after swizzle transformation
  // Note: CUTLASS Swizzle works on Layout coordinates, not raw integers
  template <int B, int M, int S>
  static void print_swizzle_transformation(int rows, int cols) {
    std::cout << "\n=== Swizzle Transformation: Swizzle<" << B << "," << M << "," << S << "> ===\n";
    std::cout << "Matrix: " << rows << "x" << cols << "\n\n";
    std::cout << "Note: CUTLASS Swizzle is meant for Layout composition, not direct integer transformation.\n";
    std::cout << "For manual XOR swizzle formula: addr ^ ((addr >> " << S << ") & 0x" << std::hex << ((1 << M) - 1)
              << std::dec << ")\n\n";

    std::vector<std::vector<int>> original(rows, std::vector<int>(cols));
    std::vector<std::vector<int>> manual_swizzled(rows, std::vector<int>(cols));

    // Manual XOR-based swizzle transformation
    for (int y = 0; y < rows; y++) {
      for (int x = 0; x < cols; x++) {
        int linear_idx = y * cols + x;
        original[y][x] = linear_idx;

        // Apply manual XOR swizzle formula
        int swizzled_idx = linear_idx ^ ((linear_idx >> S) & ((1 << M) - 1));
        manual_swizzled[y][x] = swizzled_idx;
      }
    }

    std::cout << "Original linear indices:" << std::endl;
    print_matrix_formatted(original, rows, cols);

    std::cout << "\nManual XOR-swizzled indices:" << std::endl;
    print_matrix_formatted(manual_swizzled, rows, cols);

    // Show which indices swap with which
    std::cout << "\nSwizzle pairs (indices that swap):" << std::endl;
    std::vector<bool> printed(rows * cols, false);
    for (int i = 0; i < rows * cols; i++) {
      if (!printed[i]) {
        int swizzled_i = i ^ ((i >> S) & ((1 << M) - 1));
        if (i != swizzled_i) {
          std::cout << "  " << std::setw(2) << i << " <-> " << std::setw(2) << swizzled_i << std::endl;
          printed[i] = printed[swizzled_i] = true;
        }
      }
    }
  }

  // Analyze bank conflicts for a specific access pattern
  template <int B, int M, int S>
  static void analyze_conflicts(int rows, int cols, int element_bytes = 4) {
    Swizzle<B, M, S> swizzle;

    std::cout << "\n=== Bank Conflict Analysis: Swizzle<" << B << "," << M << "," << S << "> ===\n";
    std::cout << "Matrix: " << rows << "x" << cols << ", Element size: " << element_bytes << " bytes\n\n";

    // Analyze column-wise access (typical transpose pattern)
    std::cout << "Column-wise access pattern (transpose read):" << std::endl;
    for (int col = 0; col < std::min(cols, 4); col++) {
      std::vector<int> banks;
      for (int row = 0; row < std::min(rows, 32); row++) {
        int linear_idx = row * cols + col;
        int swizzled_idx = swizzle(linear_idx);
        int byte_offset = swizzled_idx * element_bytes;
        int bank = (byte_offset / 4) % 32;
        banks.push_back(bank);
      }

      // Count unique banks
      std::vector<bool> bank_used(32, false);
      int conflicts = 0;
      for (int b : banks) {
        if (bank_used[b]) {
          conflicts++;
        }
        bank_used[b] = true;
      }

      int unique_banks = std::count(bank_used.begin(), bank_used.end(), true);
      std::cout << "  Column " << col << ": " << unique_banks << " unique banks, " << conflicts
                << " conflicts (worst-case " << (conflicts > 0 ? (banks.size() - unique_banks + 1) : 1) << "-way)"
                << std::endl;
    }
  }

  // Print the swizzle function details
  template <int B, int M, int S>
  static void print_swizzle_info() {
    Swizzle<B, M, S> swizzle;

    std::cout << "\n=== Swizzle<" << B << "," << M << "," << S << "> Information ===\n";
    std::cout << "Base bits (B): " << B << std::endl;
    std::cout << "Mask bits (M): " << M << std::endl;
    std::cout << "Shift amount (S): " << S << std::endl;
    std::cout << "\nTransformation formula:" << std::endl;
    std::cout << "  swizzled = addr ^ ((addr >> " << S << ") & 0x" << std::hex << ((1 << M) - 1) << std::dec << ")"
              << std::endl;
    std::cout << "\nBit-level operation:" << std::endl;
    std::cout << "  1. Take address" << std::endl;
    std::cout << "  2. Shift right by " << S << " bits" << std::endl;
    std::cout << "  3. Mask with " << M << " bits (0x" << std::hex << ((1 << M) - 1) << std::dec << ")" << std::endl;
    std::cout << "  4. XOR with original address\n" << std::endl;

    // Show more examples to see the swizzle effect
    std::cout << "Example transformations (showing where swizzle kicks in):" << std::endl;
    std::cout << "Note: CUTLASS Swizzle may work on layouts, not raw integers.\n" << std::endl;
    for (int i = 0; i < 32; i += 1) {
      int swizzled = swizzle(i);
      int manual_swizzle = i ^ ((i >> S) & ((1 << M) - 1));
      if (i != swizzled || i < 16) {
        // Show first 16 and any that change
        std::cout << "  " << std::setw(2) << i << " -> " << std::setw(2) << swizzled << " (manual: " << std::setw(2)
                  << manual_swizzle << ")" << " (binary: " << to_binary8(i) << " -> " << to_binary8(swizzled) << ")";
        if (i != swizzled)
          std::cout << " *";
        if (swizzled != manual_swizzle)
          std::cout << " DIFF!";
        std::cout << std::endl;
      }
    }
  }

  // Reverse-engineer swizzle parameters from a pattern
  static void analyze_swizzle_pattern(const std::vector<std::vector<int>>& original,
                                      const std::vector<std::vector<int>>& swizzled,
                                      int rows,
                                      int cols) {
    std::cout << "\n=== Analyzing Swizzle Pattern ===\n" << std::endl;

    // Collect all XOR differences
    std::map<int, int> xor_diffs;  // original_idx -> xor_diff
    for (int y = 0; y < rows; y++) {
      for (int x = 0; x < cols; x++) {
        int orig = original[y][x];
        int swiz = swizzled[y][x];
        int diff = orig ^ swiz;
        if (diff != 0) {
          xor_diffs[orig] = diff;
        }
      }
    }

    std::cout << "XOR differences (original ^ swizzled):" << std::endl;
    int count = 0;
    for (const auto& [orig, diff] : xor_diffs) {
      std::cout << "  " << std::setw(2) << orig << " ^ " << std::setw(2) << (orig ^ diff) << " = " << std::setw(2)
                << diff << " (binary: " << to_binary8(diff) << ")";
      if (++count % 4 == 0)
        std::cout << std::endl;
    }
    if (count % 4 != 0)
      std::cout << std::endl;

    // Try to determine S and M
    std::cout << "\nTrying to determine S and M parameters..." << std::endl;
    bool found = false;
    for (int S = 0; S <= 8 && !found; S++) {
      for (int M = 1; M <= 8 && !found; M++) {
        bool matches = true;
        int mask = (1 << M) - 1;

        // Test if this S,M produces the observed pattern
        for (const auto& [orig, expected_diff] : xor_diffs) {
          int computed_diff = (orig >> S) & mask;
          if (computed_diff != expected_diff) {
            matches = false;
            break;
          }
        }

        if (matches && !xor_diffs.empty()) {
          std::cout << "\nFound matching parameters:" << std::endl;
          std::cout << "  S (Shift) = " << S << std::endl;
          std::cout << "  M (Mask bits) = " << M << " (mask = 0x" << std::hex << mask << std::dec << ")" << std::endl;
          std::cout << "  Formula: swizzled = addr ^ ((addr >> " << S << ") & 0x" << std::hex << mask << std::dec << ")"
                    << std::endl;

          // Verify with a few examples
          std::cout << "\nVerification:" << std::endl;
          int verify_count = 0;
          for (const auto& [orig, _] : xor_diffs) {
            if (verify_count++ >= 8)
              break;
            int computed_swiz = orig ^ ((orig >> S) & mask);
            std::cout << "  " << std::setw(2) << orig << " -> " << std::setw(2) << computed_swiz;
            if (verify_count % 4 == 0)
              std::cout << std::endl;
          }
          if (verify_count % 4 != 0)
            std::cout << std::endl;

          found = true;
        }
      }
    }

    if (!found) {
      std::cout << "\nNo simple S,M pattern found. Pattern may not be a standard XOR swizzle." << std::endl;
    }
  }

 private:
  // Helper to print matrix in formatted way
  static void print_matrix_formatted(const std::vector<std::vector<int>>& matrix, int rows, int cols) {
    // Determine max value for formatting
    int max_val = 0;
    for (int y = 0; y < rows; y++) {
      for (int x = 0; x < cols; x++) {
        max_val = std::max(max_val, matrix[y][x]);
      }
    }
    int width = std::to_string(max_val).length() + 1;

    // Print header
    std::cout << "     ";
    for (int x = 0; x < cols; x++) {
      std::cout << std::setw(width) << x;
    }
    std::cout << std::endl;

    // Print matrix with row labels
    for (int y = 0; y < rows; y++) {
      std::cout << std::setw(3) << y << ": ";
      for (int x = 0; x < cols; x++) {
        std::cout << std::setw(width) << matrix[y][x];
      }
      std::cout << std::endl;
    }
  }
};

class CudaPoc_Day05cu01 : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    std::cout << "\n========================================" << std::endl;
    std::cout << "  CuTe Swizzle Pattern Visualization" << std::endl;
    std::cout << "========================================\n" << std::endl;
  }
};

TEST_F(CudaPoc_Day05cu01, SwizzleInfo_3_3_3) {
  SwizzleVisualizer::print_swizzle_info<3, 3, 3>();
}

TEST_F(CudaPoc_Day05cu01, BankMap_3_3_3_8x8) {
  // For float (4-byte), 8x8 tile
  SwizzleVisualizer::print_bank_map<3, 3, 3>(8, 8, 4);
}

TEST_F(CudaPoc_Day05cu01, BankMap_3_3_3_16x16) {
  // For float (4-byte), 16x16 tile
  SwizzleVisualizer::print_bank_map<3, 3, 3>(16, 16, 4);
}

TEST_F(CudaPoc_Day05cu01, BankMap_3_3_3_32x32) {
  // For float (4-byte), 32x32 tile - typical transpose size
  SwizzleVisualizer::print_bank_map<3, 3, 3>(32, 32, 4);
}

TEST_F(CudaPoc_Day05cu01, Transformation_3_3_3_8x8) {
  SwizzleVisualizer::print_swizzle_transformation<3, 3, 3>(8, 8);
}

TEST_F(CudaPoc_Day05cu01, ConflictAnalysis_3_3_3_32x32) {
  SwizzleVisualizer::analyze_conflicts<3, 3, 3>(32, 32, 4);
}

TEST_F(CudaPoc_Day05cu01, BankMap_2_1_3_8x8) {
  // For float (4-byte), 8x8 tile
  SwizzleVisualizer::print_bank_map<2, 1, 3>(8, 8, 4);
}

TEST_F(CudaPoc_Day05cu01, Transformation_2_1_3_8x8) {
  SwizzleVisualizer::print_swizzle_transformation<2, 1, 3>(8, 8);
}

// Compare different swizzle patterns
TEST_F(CudaPoc_Day05cu01, CompareSwizzlePatterns) {
  std::cout << "\n=== Comparing Different Swizzle Patterns ===\n" << std::endl;

  // Swizzle<2,2,3> - less aggressive
  std::cout << "--- Swizzle<2,2,3> ---" << std::endl;
  SwizzleVisualizer::analyze_conflicts<2, 2, 3>(32, 32, 4);

  // Swizzle<3,3,3> - standard for 4-byte elements
  std::cout << "\n--- Swizzle<3,3,3> ---" << std::endl;
  SwizzleVisualizer::analyze_conflicts<3, 3, 3>(32, 32, 4);

  // Swizzle<4,4,4> - more aggressive for 8-byte elements
  std::cout << "\n--- Swizzle<4,4,4> ---" << std::endl;
  SwizzleVisualizer::analyze_conflicts<4, 4, 4>(32, 32, 8);
}

// Demonstrate the XOR swizzle used in transpose_cuda.cu
TEST_F(CudaPoc_Day05cu01, SimpleXORSwizzle_32x32) {
  std::cout << "\n=== Simple XOR Swizzle (threadIdx.x ^ threadIdx.y) ===" << std::endl;
  std::cout << "Matrix: 32x32 (matching TRANSPOSE_BLOCK_DIM)\n" << std::endl;

  // Simulate the simple XOR swizzle from transpose_kernel_v2_swizzling
  std::vector<std::vector<int>> bank_matrix(32, std::vector<int>(32));

  for (int y = 0; y < 32; y++) {
    for (int x = 0; x < 32; x++) {
      int swizzled_x = x ^ y;
      int linear_idx = y * 32 + swizzled_x;
      int byte_offset = linear_idx * 4;  // float = 4 bytes
      int bank = (byte_offset / 4) % 32;
      bank_matrix[y][x] = bank;
    }
  }

  std::cout << "Bank map (first 8x8 corner):" << std::endl;
  for (int y = 0; y < 8; y++) {
    std::cout << std::setw(2) << y << ": ";
    for (int x = 0; x < 8; x++) {
      std::cout << std::setw(3) << bank_matrix[y][x];
    }
    std::cout << std::endl;
  }

  // Analyze conflicts
  std::cout << "\nConflict analysis for column-wise access:" << std::endl;
  for (int col = 0; col < 4; col++) {
    std::vector<bool> bank_used(32, false);
    int conflicts = 0;
    for (int row = 0; row < 32; row++) {
      int bank = bank_matrix[row][col];
      if (bank_used[bank]) {
        conflicts++;
      }
      bank_used[bank] = true;
    }
    int unique_banks = std::count(bank_used.begin(), bank_used.end(), true);
    std::cout << "  Column " << col << ": " << unique_banks << " unique banks, " << conflicts << " conflicts"
              << std::endl;
  }
}

// Reverse-engineer swizzle parameters from a given pattern

// === Analyzing Swizzle Pattern ===
//
// XOR differences (original ^ swizzled):
//    8 ^  9 =  1 (binary: 00000001)   9 ^  8 =  1 (binary: 00000001)  10 ^ 11 =  1 (binary: 00000001)  11 ^ 10 =  1
//    (binary: 00000001)
//   12 ^ 13 =  1 (binary: 00000001)  13 ^ 12 =  1 (binary: 00000001)  14 ^ 15 =  1 (binary: 00000001)  15 ^ 14 =  1
//   (binary: 00000001) 24 ^ 25 =  1 (binary: 00000001)  25 ^ 24 =  1 (binary: 00000001)  26 ^ 27 =  1 (binary:
//   00000001)  27 ^ 26 =  1 (binary: 00000001) 28 ^ 29 =  1 (binary: 00000001)  29 ^ 28 =  1 (binary: 00000001)  30 ^
//   31 =  1 (binary: 00000001)  31 ^ 30 =  1 (binary: 00000001) 40 ^ 41 =  1 (binary: 00000001)  41 ^ 40 =  1 (binary:
//   00000001)  42 ^ 43 =  1 (binary: 00000001)  43 ^ 42 =  1 (binary: 00000001) 44 ^ 45 =  1 (binary: 00000001)  45 ^
//   44 =  1 (binary: 00000001)  46 ^ 47 =  1 (binary: 00000001)  47 ^ 46 =  1 (binary: 00000001) 56 ^ 57 =  1 (binary:
//   00000001)  57 ^ 56 =  1 (binary: 00000001)  58 ^ 59 =  1 (binary: 00000001)  59 ^ 58 =  1 (binary: 00000001) 60 ^
//   61 =  1 (binary: 00000001)  61 ^ 60 =  1 (binary: 00000001)  62 ^ 63 =  1 (binary: 00000001)  63 ^ 62 =  1 (binary:
//   00000001)
//
// Trying to determine S and M parameters...
//
// Found matching parameters:
//   S (Shift) = 3
//   M (Mask bits) = 1 (mask = 0x1)
//   Formula: swizzled = addr ^ ((addr >> 3) & 0x1)
//
// Verification:
//    8 ->  9   9 ->  8  10 -> 11  11 -> 10
//   12 -> 13  13 -> 12  14 -> 15  15 -> 14
TEST_F(CudaPoc_Day05cu01, ReverseEngineerSwizzle_1) {
  const int rows = 8, cols = 8;

  // Original indices
  std::vector<std::vector<int>> original(rows, std::vector<int>(cols));
  for (int y = 0; y < rows; y++) {
    for (int x = 0; x < cols; x++) {
      original[y][x] = y * cols + x;
    }
  }

  // Your provided swizzled pattern
  std::vector<std::vector<int>> swizzled = {{0, 1, 2, 3, 4, 5, 6, 7},         {9, 8, 11, 10, 13, 12, 15, 14},
                                            {16, 17, 18, 19, 20, 21, 22, 23}, {25, 24, 27, 26, 29, 28, 31, 30},
                                            {32, 33, 34, 35, 36, 37, 38, 39}, {41, 40, 43, 42, 45, 44, 47, 46},
                                            {48, 49, 50, 51, 52, 53, 54, 55}, {57, 56, 59, 58, 61, 60, 63, 62}};

  SwizzleVisualizer::analyze_swizzle_pattern(original, swizzled, rows, cols);
}

// === Analyzing Swizzle Pattern ===
//
// XOR differences (original ^ swizzled):
//    8 ^  9 =  1 (binary: 00000001)   9 ^  8 =  1 (binary: 00000001)  10 ^ 11 =  1 (binary: 00000001)  11 ^ 10 =  1
//    (binary: 00000001)
//   12 ^ 13 =  1 (binary: 00000001)  13 ^ 12 =  1 (binary: 00000001)  14 ^ 15 =  1 (binary: 00000001)  15 ^ 14 =  1
//   (binary: 00000001) 16 ^ 18 =  2 (binary: 00000010)  17 ^ 19 =  2 (binary: 00000010)  18 ^ 16 =  2 (binary:
//   00000010)  19 ^ 17 =  2 (binary: 00000010) 20 ^ 22 =  2 (binary: 00000010)  21 ^ 23 =  2 (binary: 00000010)  22 ^
//   20 =  2 (binary: 00000010)  23 ^ 21 =  2 (binary: 00000010) 24 ^ 27 =  3 (binary: 00000011)  25 ^ 26 =  3 (binary:
//   00000011)  26 ^ 25 =  3 (binary: 00000011)  27 ^ 24 =  3 (binary: 00000011) 28 ^ 31 =  3 (binary: 00000011)  29 ^
//   30 =  3 (binary: 00000011)  30 ^ 29 =  3 (binary: 00000011)  31 ^ 28 =  3 (binary: 00000011) 32 ^ 36 =  4 (binary:
//   00000100)  33 ^ 37 =  4 (binary: 00000100)  34 ^ 38 =  4 (binary: 00000100)  35 ^ 39 =  4 (binary: 00000100) 36 ^
//   32 =  4 (binary: 00000100)  37 ^ 33 =  4 (binary: 00000100)  38 ^ 34 =  4 (binary: 00000100)  39 ^ 35 =  4 (binary:
//   00000100) 40 ^ 45 =  5 (binary: 00000101)  41 ^ 44 =  5 (binary: 00000101)  42 ^ 47 =  5 (binary: 00000101)  43 ^
//   46 =  5 (binary: 00000101) 44 ^ 41 =  5 (binary: 00000101)  45 ^ 40 =  5 (binary: 00000101)  46 ^ 43 =  5 (binary:
//   00000101)  47 ^ 42 =  5 (binary: 00000101) 48 ^ 54 =  6 (binary: 00000110)  49 ^ 55 =  6 (binary: 00000110)  50 ^
//   52 =  6 (binary: 00000110)  51 ^ 53 =  6 (binary: 00000110) 52 ^ 50 =  6 (binary: 00000110)  53 ^ 51 =  6 (binary:
//   00000110)  54 ^ 48 =  6 (binary: 00000110)  55 ^ 49 =  6 (binary: 00000110) 56 ^ 63 =  7 (binary: 00000111)  57 ^
//   62 =  7 (binary: 00000111)  58 ^ 61 =  7 (binary: 00000111)  59 ^ 60 =  7 (binary: 00000111) 60 ^ 59 =  7 (binary:
//   00000111)  61 ^ 58 =  7 (binary: 00000111)  62 ^ 57 =  7 (binary: 00000111)  63 ^ 56 =  7 (binary: 00000111)
//
// Trying to determine S and M parameters...
//
// Found matching parameters:
//   S (Shift) = 3
//   M (Mask bits) = 3 (mask = 0x7)
//   Formula: swizzled = addr ^ ((addr >> 3) & 0x7)
//
// Verification:
//    8 ->  9   9 ->  8  10 -> 11  11 -> 10
//   12 -> 13  13 -> 12  14 -> 15  15 -> 14
TEST_F(CudaPoc_Day05cu01, ReverseEngineerSwizzle_2) {
  const int rows = 8, cols = 8;

  // Original indices
  std::vector<std::vector<int>> original(rows, std::vector<int>(cols));
  for (int y = 0; y < rows; y++) {
    for (int x = 0; x < cols; x++) {
      original[y][x] = y * cols + x;
    }
  }

  // Your provided swizzled pattern
  std::vector<std::vector<int>> swizzled = {{0, 1, 2, 3, 4, 5, 6, 7},         {9, 8, 11, 10, 13, 12, 15, 14},
                                            {18, 19, 16, 17, 22, 23, 20, 21}, {27, 26, 25, 24, 31, 30, 29, 28},
                                            {36, 37, 38, 39, 32, 33, 34, 35}, {45, 44, 47, 46, 41, 40, 43, 42},
                                            {54, 55, 52, 53, 50, 51, 48, 49}, {63, 62, 61, 60, 59, 58, 57, 56}};

  SwizzleVisualizer::analyze_swizzle_pattern(original, swizzled, rows, cols);
}
}  // namespace cuda_poc::day05cu
