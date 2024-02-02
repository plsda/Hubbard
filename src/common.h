#ifndef HUBBARD_COMMON_H

#include <vector>
#include <array>
#include <set>
#include <cmath>
#include <cstdint>
#include <cassert>
#include <cstddef>
#include <algorithm>
#include <numeric>
#include <sstream>

#include <iterator>
#include <execution>
#define STD_POLICY std::execution::par,
#define STD_POLICY_UNSEQ std::execution::par_unseq,

#define MAX_SITE_COUNT 16
#ifdef EIGEN_WORLD_VERSION
#define _EIGEN 
#endif

using u8 = uint8_t;
using u32 = uint32_t;
using u64 = uint64_t;

using real = float;
using idxt = std::ptrdiff_t;

using Det = uint32_t; // NOTE: Always unsigned
using HDet = uint16_t;

using Lookup16 = std::array<u8, 65536>;

inline constexpr Lookup16 gen_bitcount_lookup16()
{
   Lookup16 result{};
   // Lookup-table for the number of bits set in a 16-bit unsigned integer (https://graphics.stanford.edu/~seander/bithacks.html)
   for(u32 i = 0; i <= 0xFFFF; i++)
   {
      u32 tmp = i - ((i >> 1) & (0x55555555));
      tmp = (tmp & (0x33333333)) + ((tmp >> 2) & (0x33333333));
      result[i] = ((tmp + (tmp >> 4) & 0xF0F0F0F)*0x1010101) >> 24;
   }

   return result;
}

inline Lookup16 bitcount_lookup = gen_bitcount_lookup16();

#define HUBBARD_COMMON_H
#endif
