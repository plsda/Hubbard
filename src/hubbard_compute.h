#ifndef HUBBARD_COMPUTE_H

#include "hubbard_common.h"
#include "utils.h"
#include "basis.h"

struct Error_stream
{
   std::stringstream errors;
   bool has_errors = false;

   void reset()
   {
      has_errors = false;
      errors.clear();
      errors.str({});
   }
};

inline Error_stream& endl(Error_stream& s)
{
   s.errors << std::endl;
   return s;
}

template <class T>
inline Error_stream& operator<<(Error_stream& s, T&& str)
{
   s.has_errors = true;
   s.errors << std::forward<T>(str);
   return s;
}

inline std::ostream& operator<<(std::ostream& os, const Error_stream& s)
{
   os << s.errors.rdbuf();
   return os;
}

class Hubbard_compute_device
{
public:
   Hubbard_compute_device(Error_stream* errors = 0);
   ~Hubbard_compute_device();
   real H_int_element(const Det* const bra_dets, const real* const bra_coeffs, int bra_count, 
                      const Det* const ket_dets, const real* const ket_coeffs, int ket_count,
                      const HubbardParams& params);
   real sym_eigs_smallest(real* elements, int dim); // NOTE: Elements are assumed to be stored in column-major order

   Error_stream* errors;
private:
   class Compute_context;
   std::unique_ptr<Compute_context> ctx;
};

#define HUBBARD_COMPUTE_H
#endif
