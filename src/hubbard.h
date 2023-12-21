#ifndef HUBBARD_H

#include <iostream>
#include <array>
#include <vector>
#include <set>
#include <cstdint>
#include <cstddef>
#include <cassert>
#include <cmath>
#include <numbers>
#include <numeric>
#include <execution>
#include <algorithm>
#include <ranges>
#include <iterator>
#include <span>

#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

#define MAX_SITE_COUNT 16

using u8 = uint8_t;
using u32 = uint32_t;
using u64 = uint64_t;

using real = float;
using idxt = std::ptrdiff_t;

using Det = uint32_t; // NOTE: Always unsigned
using HDet = uint16_t;

using MatR = Eigen::Matrix<real, Eigen::Dynamic, Eigen::Dynamic>;
using Arr2R = Eigen::Array<real, Eigen::Dynamic, Eigen::Dynamic>;
using Arr3R = Eigen::Tensor<real, 3>;

#define STD_POLICY std::execution::par,
#define STD_POLICY_UNSEQ std::execution::par_unseq,
#define PI std::numbers::pi_v<real>
const real LOG2 = std::log(2);

typedef std::array<u8, 65536> Lookup16;

enum class BCS
{
   OPEN = 1,
   PERIODIC = 2,
};

#include "utils.h"
#include "basis.h"

constexpr Lookup16 bitcount_lookup(gen_bitcount_lookup16());

#define HUBBARD_H
#endif
