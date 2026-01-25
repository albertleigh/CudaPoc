//
// Created by ali on 1/25/26.
//

#pragma once
#include <cstdint>

namespace cuda_poc::device {

// Host-to-Big-Endian byte order conversion functions
// These use PTX instructions for efficient byte reordering on GPU

// Convert 64-bit value from host byte order to big-endian
static __device__ __forceinline__ uint64_t HtoBE64(uint64_t x) {
  uint64_t ret;
  asm("{"
      "\n\t.reg .b32 ign;"
      "\n\t.reg .b32 lo;"
      "\n\t.reg .b32 hi;"
      "\n\t.reg .b32 new_lo;"
      "\n\t.reg .b32 new_hi;"
      "\n\tmov.b64 {lo,hi}, %1;"
      "\n\tprmt.b32 new_hi, lo, ign, 0x0123;"
      "\n\tprmt.b32 new_lo, hi, ign, 0x0123;"
      "\n\tmov.b64 %0, {new_lo,new_hi};"
      "\n\t}"
      : "=l"(ret)
      : "l"(x));
  return ret;
}

// Convert 32-bit value from host byte order to big-endian
static __device__ __forceinline__ uint32_t HtoBE32(uint32_t x) {
  uint32_t ret;
  asm("{"
      "\n\t.reg .b32 ign;"
      "\n\tprmt.b32 %0, %1, ign, 0x0123;"
      "\n\t}"
      : "=r"(ret)
      : "r"(x));
  return ret;
}

// Convert 16-bit value from host byte order to big-endian
static __device__ __forceinline__ uint16_t HtoBE16(uint16_t x) {
  // TODO: simplify PTX using 16-bit instructions
  auto a = static_cast<uint32_t>(x);
  uint32_t d;
  asm volatile(
      "{"
      "\n\t.reg .b32 mask;"
      "\n\t.reg .b32 ign;"
      "\n\tmov.b32 mask, 0x4401;"
      "\n\tmov.b32 ign, 0x0;"
      "\n\tprmt.b32 %0, %1, ign, mask;"
      "\n\t}"
      : "=r"(d)
      : "r"(a));
  return static_cast<uint16_t>(d);
}

}  // namespace cuda_poc::device