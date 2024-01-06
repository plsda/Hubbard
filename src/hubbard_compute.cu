#include <utility>
#include <helper_cuda.h>
#include <cuda/atomic>
#include "hubbard_compute.h"

#define cuda_memcpy(src, dst, count) cudaMemcpy(dst, src, cudaMemcpyDefault)
#define cuda_memcpy_to_device(src, dst, sz) cudaMemcpy(dst, src, sz, cudaMemcpyHostToDevice)
#define cuda_memcpy_to_host(src, dst, sz) cudaMemcpy(dst, src, sz, cudaMemcpyDeviceToHost)

__device__
static int mod(int a, int b)
{
   int rem = a % b;
   b &= rem >> std::numeric_limits<int>::digits;

   return b + rem;
}

__device__
static u32 get_pop(Det det)
{
   // NOTE: 32-bit at most
   return __popc(reinterpret_cast<int&>(det));
}
__device__
static u32 count_state(Det det, u32 state_idx)
{
   return (det >> state_idx) & 1;
}

__device__
static u32 count_higher(u32 state_idx, Det det)
{
   u32 mask = u32(0xFFFFFFFF << (state_idx + 1));
   return get_pop(det & mask);
}


__device__
static SDet sadd(u32 state_idx, Det det, int sign)
{
   int neg_state_occ = 1 - count_state(det, state_idx);
   Det result = (det | (1 << state_idx))*neg_state_occ;

   sign *= neg_state_occ;
   sign *= 1 - 2*(count_higher(state_idx, det) & 1);

   return {result, sign};
}

__device__
static SDet sadd(u32 state_idx, SDet det)
{
   return sadd(state_idx, det.det, det.sign);
}

__device__
static SDet ssub(u32 state_idx, Det det, int sign)
{
   int state_occ = count_state(det, state_idx);
   Det result = (det & (~(1 << state_idx)))*state_occ;

   sign *= state_occ;
   sign *= 1 - 2*(count_higher(state_idx, det) & 1);

   return {result, sign};
}

__device__
static SDet ssub(u32 state_idx, SDet det)
{
   return ssub(state_idx, det.det, det.sign);
}

__host__ __device__
static std::pair<Det, Det> get_det_up_down(Det det, HubbardParams params)
{
   Det det_up = det & ((2 << (params.Ns - 1)) - 1);
   Det det_down = det >> params.Ns;
   return {det_up, det_down};
}

__device__
static Det det_config_ID(Det det, const HubbardParams& params)
{
   auto [det_up, det_down] = get_det_up_down(det, params);
   Det det_xor = det_up ^ det_down;
   Det det_and = det_up & det_down;

   Det result = (det_xor << params.Ns) | det_and;

   return result;
}


__device__
static bool cmp_det_config(Det det1, Det det2, HubbardParams params)
{
   return (det_config_ID(det1, params) == det_config_ID(det2, params));
}

#define CHECK_NO_CUDA_ERRORS assert(capture_cuda_error(cudaGetLastError(), __LINE__, __FILE__))
#define CAPTURE_LAST_CUDA_ERROR(...) capture_cuda_error(cudaGetLastError(), __LINE__, __FILE__, EXPAND(__VA_ARGS__))
#define CAPTURE_CUDA_ERROR(call, ...) capture_cuda_error(call, __LINE__, __FILE__, EXPAND(__VA_ARGS__))
bool capture_cuda_error(cudaError_t error, int line, const char* file, std::stringstream* const errors = 0)
{
   bool no_errors = true;
   if(error != cudaSuccess)
   {
      if(errors)
      {
         *errors << file << "(" << line << "): " << cudaGetErrorName(error) 
                 << ": " << cudaGetErrorString(error) << std::endl;
      }

      no_errors = false;
   }

   return no_errors;
}

bool init_compute(std::stringstream* const errors = 0)
{
   return (gpuDeviceInit(gpuGetMaxGflopsDeviceId()) >= 0);
}

__global__
void H_int_element_term(const Det* const __restrict__ bra_dets, const real* const __restrict__ bra_coeffs,
                        const Det* const __restrict__ ket_dets, const real* const __restrict__ ket_coeffs,
                        const int bra_count, HubbardParams params, real* __restrict__ result)
{
   int ket_det_idx = blockIdx.x;
   int q  = threadIdx.x;
   int k1 = blockIdx.y*blockDim.y + threadIdx.y;
   int k2 = blockIdx.z*blockDim.z + threadIdx.z;

   if((k1 < params.Ns) && (k2 < params.Ns))
   {
      Det ket_det = ket_dets[ket_det_idx];
      auto [new_ket_det, new_ket_det_sign] = sadd(mod(k1 + q, params.Ns),
                                                  ssub(k1,
                                                       sadd(mod(k2 - q, params.Ns) + params.Ns,
                                                            ssub(k2 + params.Ns, 
                                                                 ket_det))));

      if(new_ket_det_sign != 0 && cmp_det_config(bra_dets[0], new_ket_det, params))
      {
         real cur_res = 0;
         for(int bra_det_idx = 0; bra_det_idx < bra_count; bra_det_idx++)
         {
            Det bra_det = bra_dets[bra_det_idx];
            cur_res += new_ket_det_sign*bra_coeffs[bra_det_idx]*ket_coeffs[ket_det_idx]*(bra_det == new_ket_det);
         }

         // TODO: Use CUB to do block-wise reduction
         atomicAdd(result, cur_res);
      }
   }
}

real compute_H_int_element(const Det* const bra_dets, const real* const bra_coeffs, int bra_count, 
                           const Det* const ket_dets, const real* const ket_coeffs, int ket_count,
                           const HubbardParams& params)
{
   assert(bra_count > 0 && ket_count > 0);

   real result = 0;

   Det bra_ref_det = bra_dets[0];
   Det ket_ref_det = ket_dets[0];
   auto [bup, bdn] = get_det_up_down(bra_ref_det, params);
   auto [kup, kdn] = get_det_up_down(ket_ref_det, params);
   int spinless_diff = bitcount_lookup[(bup | bdn) ^ (kup | kdn)] +
                       bitcount_lookup[(bup & bdn) ^ (kup & kdn)];

   if(spinless_diff <= 4)
   {
      int yz_block_n = 2;
      int yz_block_dim = std::ceil(float(params.Ns)/float(yz_block_n));
      dim3 block_count = {unsigned int(ket_count),
                          unsigned int(yz_block_n),
                          unsigned int(yz_block_n)};
      dim3 threads_per_block = {unsigned int(params.Ns),
                                unsigned int(yz_block_dim),
                                unsigned int(yz_block_dim)};
      Det*  d_bra_dets;
      real* d_bra_coeffs;
      Det*  d_ket_dets;
      real* d_ket_coeffs;

      cudaMalloc(&d_bra_dets,   sizeof(Det)*bra_count);
      cudaMalloc(&d_bra_coeffs, sizeof(real)*bra_count);
      cudaMalloc(&d_ket_dets,   sizeof(Det)*ket_count);
      cudaMalloc(&d_ket_coeffs, sizeof(real)*ket_count);

      cuda_memcpy_to_device(bra_dets,   d_bra_dets,   sizeof(Det)*bra_count);
      cuda_memcpy_to_device(bra_coeffs, d_bra_coeffs, sizeof(real)*bra_count);
      cuda_memcpy_to_device(ket_dets,   d_ket_dets,   sizeof(Det)*ket_count);
      cuda_memcpy_to_device(ket_coeffs, d_ket_coeffs, sizeof(real)*ket_count);

      CHECK_NO_CUDA_ERRORS;

      real* d_result;
      cudaMalloc(&d_result, sizeof(real));
      cudaMemset(d_result, 0, sizeof(real));
      H_int_element_term<<<block_count, threads_per_block>>>(d_bra_dets, d_bra_coeffs,
                                                             d_ket_dets, d_ket_coeffs,
                                                             bra_count, params, d_result);
      cuda_memcpy_to_host(d_result, &result, sizeof(result));
      cudaFree(d_result);
      result *= params.U/params.Ns;

      cudaFree(d_bra_dets);
      cudaFree(d_bra_coeffs);
      cudaFree(d_ket_dets);
      cudaFree(d_ket_coeffs);

      CHECK_NO_CUDA_ERRORS;

   }

   return result;
}

real sym_eigs()
{
   assert(!"Not implemented!");
   return 0;
}
