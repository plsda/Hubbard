#ifndef UTILS_H

#define EXPAND(...) __VA_ARGS__

template <class T>
class RangeItr
{
public:
   using iterator_category = std::random_access_iterator_tag;
   using difference_type = std::ptrdiff_t;
   using value_type = T;
   using pointer = T*;
   using reference = T&;

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


   T operator*() { return cur; }

   pointer operator->() { return &cur; }

   reference operator[](difference_type i) { return *(*this + i); }


   bool operator==(const RangeItr<T>& it) const { return cur == it.cur; }

   bool operator<(const RangeItr<T>& it) const  { return cur < it.cur; }

   bool operator!=(const RangeItr<T>& it) const { return !(*this == it); }

   bool operator>(const RangeItr<T>& it) const  { return it < *this; }

   bool operator<=(const RangeItr<T>& it) const { return !(it < *this); }

   bool operator>=(const RangeItr<T>& it) const { return !(*this < it); }

};

template <class T>
class Range
{
public:
   explicit Range(T begin_val, T end_val) : begin_val(begin_val), end_val(end_val) {}
   explicit Range(T end_val) : begin_val(0), end_val(end_val) {}

   RangeItr<T> begin() { return RangeItr<T>(begin_val); }
   RangeItr<T> end() { return RangeItr<T>(end_val); }

private:
   T begin_val;
   T end_val;
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

template <class... Arrays>
void sort_multiple(Arrays&... arrays);

template <class T>
T pop_vec(std::vector<T>& vec, int i);

template <class T>
T pop_vec(std::vector<T>& vec, int i, int& one_past_last);

template <class T>
T pop_bit(T& a);

template <class T>
T& push_bit(T& a, u8 b = 1);

int mod(int a, int b);

bool is_close(real a, real ref, real abs_tol, real rel_tol);

// Binomial coefficient with integer arguments
u64 choose(u32 n, u32 m);

template<typename F>
real adaptive_simpson(F f, real lower, real upper, real tolerance, size_t max_steps);

template<typename F>
real simpson_step(F f, size_t step, real previous_estimate, real lower, real upper, real tolerance, size_t max_steps);

template <class F>
real quad(F integrand, IntArgs args);

#define UTILS_H
#endif
