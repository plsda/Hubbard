#ifndef UTILS_H

#define EXPAND(x) x 

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

template <class... Arrays>
void sort_multiple(Arrays&... arrays)
{
   const auto& sort_array = std::get<0>(std::forward_as_tuple(arrays...)); // Use the first array as a reference

   size_t len = sort_array.size();
   std::vector<size_t> sort_indices(len); // sort_indices[i] = index of the i'th element of the sorted array in the original array
   std::iota(sort_indices.begin(), sort_indices.end(), 0);
   std::sort(sort_indices.begin(), sort_indices.end(), [&sort_array](size_t i, size_t j){return sort_array[i] < sort_array[j];});

   for(int dest_idx = 0; dest_idx < len; dest_idx++)
   {
      size_t src_idx = sort_indices[dest_idx];

      int update_idx = dest_idx;
      for(; sort_indices[update_idx] != dest_idx; update_idx++) {}

      sort_indices[dest_idx] = dest_idx;
      sort_indices[update_idx] = src_idx;

      (std::swap(arrays[dest_idx], arrays[src_idx]),...);
   }
}

template <class T>
T pop_vec(std::vector<T>& vec, int i)
{
   T result = vec[i];
   std::swap(vec[i], vec.back());
   vec.pop_back();

   return result;
}

template <class T>
T pop_vec(std::vector<T>& vec, int i, int& one_past_last)
{
   T result = vec[i];
   std::swap(vec[i], vec[one_past_last-1]);
   one_past_last--;

   return result;
}

template <class T>
T pop_bit(T& a)
{
   T result = a & 1;
   a >>= 1;
   return result;
}

template <class T>
T& push_bit(T& a, u8 b = 1)
{
   a <<= 1;
   a |= b;
   return a;
}

int mod(int a, int b)
{
   assert(b >= 0);
   int rem = a % b;
   b &= rem >> std::numeric_limits<int>::digits;

   return b + rem;
}

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

#define UTILS_H
#endif
