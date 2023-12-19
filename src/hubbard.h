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

constexpr Lookup16 gen_bitcount_lookup16()
{
   Lookup16 result;
   // Lookup-table for the number of bits set in a 16-bit unsigned integer (https://graphics.stanford.edu/~seander/bithacks.html)
   for(u32 i = 0; i <= 0xFFFF; i++)
   {
      u32 tmp = i - ((i >> 1) & (0x55555555));
      tmp = (tmp & (0x33333333)) + ((tmp >> 2) & (0x33333333));
      result[i] = ((tmp + (tmp >> 4) & 0xF0F0F0F)*0x1010101) >> 24;
   }

   return result;
}

constexpr Lookup16 bitcount_lookup(gen_bitcount_lookup16());

int mod(int a, int b)
{
   assert(b >= 0);
   int rem = a % b;
   b &= rem >> std::numeric_limits<int>::digits;

   return b + rem;
}

struct SDet 
{
   Det det;
   int sign = 1;
};

struct WeightedDet
{
   Det det;
   real coeff;

   real operator*(WeightedDet other)
   {
      return coeff*other.coeff*(det == other.det);
   }
};

struct KBasis
{
   std::vector<Det> basis;
   std::vector<int> momenta;
   std::vector<int> block_sizes;
};

struct KConfigs
{
   std::vector<std::vector<std::vector<Det>>> configs;
   std::vector<int> block_sizes;
};

enum class BCS
{
   OPEN = 1,
   PERIODIC = 2,
};

template <class T>
class RangeItr
{
public:
   using difference_type = std::ptrdiff_t;
   using value_type = T;
   using pointer = T*;
   using reference = T&;
   using iterator_category = std::random_access_iterator_tag;

   T cur;

   explicit RangeItr(T idx) : cur(idx) {}
   RangeItr() = default;

   RangeItr<T>& operator+=(difference_type d)
   {
      cur += d;
      return *this;
   }

   RangeItr<T>& operator-=(difference_type d)
   {
      cur -= d;
      return *this;
   }

   RangeItr<T> operator+(difference_type d) const
   {
      RangeItr<T> result = *this;
      result += d;
      return result;
   }

   friend RangeItr<T> operator+(difference_type d, const RangeItr<T>& it)
   {
      return it + d;
   }

   RangeItr<T> operator-(difference_type d) const
   {
      RangeItr<T> result = *this;
      result -= d;
      return result;
   }

   difference_type operator-(const RangeItr<T>& it) const
   {
      return difference_type(cur) - difference_type(it.cur);
   }

   friend RangeItr<T> operator-(difference_type d, const RangeItr<T>& it)
   {
      return it - d;
   }

   RangeItr<T>& operator++()
   {
      cur++;
      return *this;
   }

   RangeItr<T> operator++(int)
   {
      RangeItr<T> result = *this;
      ++(*this);
      return result;
   }

   RangeItr<T>& operator--() 
   {
      cur--;
      return *this;
   }

   RangeItr<T> operator--(int)
   {
      RangeItr<T> result = *this;
      --(*this);
      return result;
   }


   T operator*()
   {
      return cur;
   }

   pointer operator->()
   {
      return &cur;
   }

   reference operator[](difference_type i)
   {
      return *(*this + i);
   }


   bool operator==(const RangeItr<T>& it) const
   {
      return cur == it.cur;
   }

   bool operator<(const RangeItr<T>& it) const
   {
      return cur < it.cur;
   }

   bool operator!=(const RangeItr<T>& it) const
   {
      return !(*this == it);
   }

   bool operator>(const RangeItr<T>& it) const
   {
      return it < *this;
   }

   bool operator<=(const RangeItr<T>& it) const
   {
      return !(it < *this);
   }

   bool operator>=(const RangeItr<T>& it) const
   {
      return !(*this < it);
   }

};

template <class T>
class Range
{
public:
   explicit Range(T begin_val, T end_val) : begin_val(begin_val), end_val(end_val) {}

   RangeItr<T> begin()
   {
      return RangeItr<T>(begin_val);
   }

   RangeItr<T> end()
   {
      return RangeItr<T>(end_val);
   }

private:
   T begin_val;
   T end_val;
};

bool is_close(real a, real ref, real abs_tol = 1e-8, real rel_tol = 1e-5)
{
   assert(!std::isnan(a));
   assert(!std::isnan(ref));

   return std::abs(a - ref) <= (abs_tol + rel_tol*std::abs(ref));
}

// Binomial coefficient with integer arguments
u64 choose(u32 n, u32 m)
{
   u64 result = 1;
   m = std::min(m, n - m);

   for(u64 i = 1; i <= m; i++, n--) 
   {
      if(result/i > std::numeric_limits<u64>::max()/u64(n))
      {
         return 0;
      }
   
      result = (result/i)*n + (result % i)*n/i;
   }
   
   return result;
}

struct HubbardParams
{
   /*
      Stores the parameters defining a Hubbard model and provides related utilities.

      Parameters
      ----------
      real T     : Hopping energy
      real U     : Interaction strength
      int Ns     : Site count
      int N      : Particle count
      int N_up   : Number of spin-up particles
      int N_down : Number of spin-down particles
   */

   real T;
   real U;
   int Ns;
   int N;
   int N_up;
   int N_down;

   HubbardParams() = default;

   explicit HubbardParams(real T, real U, int Ns, int N_up, int N_down) :
      T(T), U(U), Ns(Ns), N_up(N_up), N_down(N_down), N(N_up + N_down)
   {
      assert(Ns > 0 && N > 0 && N_up >= 0 && N_down >= 0);
   }

   // NOTE: Result may overflow for systems larger than 16 sites (but such systems are currently not supported)
   int basis_size() const
   {
      return choose(Ns, N_up)*choose(Ns, N_down);
   }

   void set_half_filling(int new_N_up)
   {
      assert(new_N_up > 0);

      Ns = 2*new_N_up;
      N = 2*new_N_up;
      N_up = new_N_up;
      N_down = new_N_up;
   }
   
};

struct IntArgs
{
   real lower{};
   real upper{};
   real abs_tol{};
   real rel_tol{};
   size_t min_steps{};
   size_t max_steps{};
};

template<typename F>
real adaptive_simpson(F f, real lower, real upper, real tolerance, size_t max_steps)
{
   real mid_point = 0.5*(lower + upper);
   real result = (upper - lower)/6.0 * (f(lower) + 4.0*f(mid_point) + f(upper));

   result = simpson_step(f, 1, result, lower, upper, tolerance, max_steps);

   return result;
}

template<typename F>
real simpson_step(F f, size_t step, real previous_estimate, real lower, real upper, real tolerance, size_t max_steps)
{
   real h = 0.5*(upper - lower);
   real mid_point = 0.5*(lower + upper);
   real f_mid = f(mid_point);

   real left_simpson = h/6.0 * (f(lower) + 4.0*f(0.5*(lower + mid_point)) + f_mid);
   real right_simpson = h/6.0 * (f_mid + 4.0*f(0.5*(mid_point + upper)) + f(upper));

   real result = left_simpson + right_simpson;
   real diff = result - previous_estimate;

   if(std::abs(diff) >= 15.0*tolerance && step < max_steps)
   {
      result = simpson_step(f, step + 1, left_simpson, lower, mid_point, 0.5*tolerance, max_steps) + 
               simpson_step(f, step + 1, right_simpson, mid_point, upper, 0.5*tolerance, max_steps);
   }
   else
   {
      result += diff / 15.0;
   }

   return result;
}

template <class F>
real quad(F integrand, IntArgs args)
{
   real result = adaptive_simpson(integrand, args.lower, args.upper, args.abs_tol, args.max_steps);

   return result;
}


#define HUBBARD_H
#endif
