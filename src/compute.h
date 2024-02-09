#ifndef HUBBARD_COMPUTE_H

struct ErrorStream
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

inline ErrorStream& endl(ErrorStream& s)
{
   s.errors << std::endl;
   return s;
}

template <class T>
inline ErrorStream& operator<<(ErrorStream& s, T&& str)
{
   s.has_errors = true;
   s.errors << std::forward<T>(str);
   return s;
}

inline std::ostream& operator<<(std::ostream& os, const ErrorStream& s)
{
   os << s.errors.rdbuf();
   return os;
}

struct ComputeMemoryReqs 
{
   size_t total_host_memory_sz;
   size_t total_device_memory_sz;
};

class HubbardComputeDevice
{
public:
   HubbardComputeDevice(size_t host_workspace_init_size, size_t device_workspace_init_size, ErrorStream* errors = 0);
   HubbardComputeDevice(ComputeMemoryReqs workspace_init_sizes, ErrorStream* errors = 0) :
      HubbardComputeDevice(workspace_init_sizes.total_host_memory_sz, workspace_init_sizes.total_device_memory_sz, errors) {}
   HubbardComputeDevice(HubbardSizes init_sizes, ErrorStream* errors = 0) : HubbardComputeDevice(0, 0, errors)
   {
      assert(prepare(init_sizes));
   }

   ~HubbardComputeDevice();

   ComputeMemoryReqs get_memory_requirements(HubbardSizes sz);
   bool prepare(HubbardSizes sz);
   //void reset();

   real H_int_element(const Det* const bra_dets, const real* const bra_coeffs, int bra_count, 
                      const Det* const ket_dets, const real* const ket_coeffs, int ket_count,
                      const HubbardParams& params);
   real sym_eigs_smallest(real* elements, int dim); // NOTE: Elements are assumed to be stored in column-major order

   ErrorStream* errors;
private:
   class ComputeContext;
   std::unique_ptr<ComputeContext> ctx;
};

#define HUBBARD_COMPUTE_H
#endif
