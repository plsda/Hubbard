#include <utility>
#include <helper_cuda.h>
#include <cuda/atomic>
#include <cusolver_common.h>
#include <cusolverDn.h>

#include <cuda.h> // For cuMemGetAddressRange

#include "common.h"
#include "allocator.h"
#include "utils.h"
#include "basis.h"
#include "compute.h"

// library_types.h
constexpr cudaDataType real2cuda()
{
   switch(sizeof(real))
   {
      case 2: { return CUDA_R_16F; } break;
      case 4: { return CUDA_R_32F; } break;
      case 8: { return CUDA_R_64F; } break;
      default: { assert(!"Unsupported real type."); return CUDA_R_32F; };
   }
}
const cudaDataType CUDA_REAL = real2cuda();
const cudaDataType CUDA_COMP_TYPE = CUDA_REAL;

#define cuda_memcpy(dst, src, sz) cudaMemcpy(dst, src, sz, cudaMemcpyDefault)
#define cuda_memcpy_to_device(dst, src, sz) cudaMemcpy(dst, src, sz, cudaMemcpyHostToDevice)
#define cuda_memcpy_to_host(dst, src, sz) cudaMemcpy(dst, src, sz, cudaMemcpyDeviceToHost)

#define CHECK_NO_CUDA_ERRORS assert(capture_cuda_error(cudaGetLastError(), __LINE__, __FILE__))
#define CAPTURE_LAST_CUDA_ERROR(...) capture_cuda_error(cudaGetLastError(), __LINE__, __FILE__, EXPAND(__VA_ARGS__))
#define CAPTURE_CUDA_ERROR(call, ...) capture_cuda_error(call, __LINE__, __FILE__, EXPAND(__VA_ARGS__))
bool capture_cuda_error(cudaError_t error, int line, const char* file, ErrorStream* const errors = 0)
{
   bool no_errors = true;
   if(error != cudaSuccess)
   {
      if(errors)
      {
         *errors << file << "(" << line << "): " << cudaGetErrorName(error) 
                 << ": " << cudaGetErrorString(error) << "\n";
      }

      no_errors = false;
   }

   assert(no_errors);

   return no_errors;
}

void* cuda_malloc(size_t size)
{
   void* result;
   cudaError_t error = cudaMalloc(&result, size);
   if(error != cudaSuccess)
   {
      result = 0;
      assert(!"cuda_malloc failed!");
   }

   return result;
}
void cuda_free(void* ptr)
{
   cudaError_t error = cudaFree(ptr);
   assert("cuda_free failed!" && (error == cudaSuccess));
}
void* cuda_realloc(void* old_ptr, size_t size)
{
   void* result = 0;
   size_t old_size; 
   CUresult cu_result = cuMemGetAddressRange(NULL, &old_size, (CUdeviceptr)old_ptr);
   if(cu_result == CUDA_SUCCESS)
   {
      result = cuda_malloc(size);
      cudaError_t error = cudaMemcpy(result, old_ptr, std::min(old_size, size), cudaMemcpyDeviceToDevice);
      if(error == cudaSuccess)
      {
         cuda_free(old_ptr);
      }
      else
      {
         cuda_free(result);
         result = 0;
         assert(!"cudaMemcpy failed in cuda_realloc!");
      }
   }
   else
   {
      assert(!"cuMemGetAddressRange failed in cuda_realloc!");
   }

   return result;
}

using DArenaAllocator = allocation::ArenaAllocator<cuda_malloc, cuda_free, cuda_realloc>;
using DArenaCheckpoint = typename DArenaAllocator::Checkpoint;

struct HubbardComputeDevice::ComputeContext
{
   ComputeContext(size_t host_workspace_init_size, size_t device_workspace_init_size, ErrorStream* const errors = 0) :
      errors(errors), h_arena(host_workspace_init_size), d_arena(device_workspace_init_size)
   {
      cusolverStatus_t status = cusolverDnCreate(&cusolver_handle);
      assert(status == CUSOLVER_STATUS_SUCCESS);
      CAPTURE_LAST_CUDA_ERROR(errors);
   }

   cusolverDnHandle_t cusolver_handle;
   ArenaAllocator h_arena;
   DArenaAllocator d_arena;

   ErrorStream* const errors;
};

template<class T>
void cuda_set(T* dest, const T* src, size_t count, ErrorStream* errors = 0)
{
   CAPTURE_CUDA_ERROR(cudaMemcpy(dest, src, sizeof(T)*count, cudaMemcpyHostToDevice), errors);
}

template<class T>
T cuda_get(T* src)
{
   T result;
   cudaMemcpy(&result, src, sizeof(T), cudaMemcpyDeviceToHost);
   return result;
}

HubbardComputeDevice::HubbardComputeDevice(size_t host_workspace_init_size, size_t device_workspace_init_size, ErrorStream* errors) : errors(errors)
{
   bool init_ok = (gpuDeviceInit(gpuGetMaxGflopsDeviceId()) >= 0);
   assert(init_ok);

   ctx = std::make_unique<ComputeContext>(device_workspace_init_size, host_workspace_init_size, errors);
}

HubbardComputeDevice::~HubbardComputeDevice()
{
   //cudaDeviceReset();
}

ComputeMemoryReqs DnXsyevdx_memory_requirements(cusolverDnHandle_t cusolver_handle, HubbardSizes sz)
{
   cusolverEigMode_t  jobz  = CUSOLVER_EIG_MODE_NOVECTOR;
   cusolverEigRange_t range = CUSOLVER_EIG_RANGE_I;
   cublasFillMode_t   uplo  = CUBLAS_FILL_MODE_LOWER;

   real* d_elements = 0;
   real* W = 0;

   int64_t dim = sz.max_KS_dim;
   int64_t found_count;
   int64_t lda = dim;
   int64_t il = 1;
   int64_t iu = 1;

   size_t required_device_workspace_size;
   size_t required_host_workspace_size;

   cusolverStatus_t buffer_info_status = cusolverDnXsyevdx_bufferSize(
       cusolver_handle,                 // cusolverDnHandle_t handle,
       NULL,                            // cusolverDnParams_t params,
       jobz,                            // cusolverEigMode_t jobz,
       range,                           // cusolverEigRange_t range,
       uplo,                            // cublasFillMode_t uplo,
       dim,                             // int64_t n,
       CUDA_REAL,                       // cudaDataType dataTypeA
       d_elements,                      // void *A,
       lda,                             // int64_t lda,
       0,                               // void * vl,
       0,                               // void * vu,
       il,                              // int64_t il,
       iu,                              // int64_t iu,
       &found_count,                    // int64_t *meig64,
       CUDA_REAL,                       // cudaDataType dataTypeW,
       W,                               // void *W,
       CUDA_COMP_TYPE,                  // cudaDataType computeType,
       &required_device_workspace_size, // size_t *workspaceInBytesOnDevice,
       &required_host_workspace_size    // size_t *workspaceInBytesOnHost
   );

   ComputeMemoryReqs result{};
   result.total_device_memory_sz = required_device_workspace_size + 2*sizeof(real);
   result.total_host_memory_sz = required_host_workspace_size + 2*sizeof(real);
   return result;
}

ComputeMemoryReqs HubbardComputeDevice::get_memory_requirements(HubbardSizes sz)
{
   ComputeMemoryReqs result{};

   size_t max_det_count = size_t(sz.max_dets_in_config);
   size_t max_dim = size_t(sz.max_KS_dim);
   size_t max_elem_count = size_t(max_dim*max_dim);

   ComputeMemoryReqs DnXsyevdx_reqs = DnXsyevdx_memory_requirements(ctx->cusolver_handle, sz);
  
   result.total_host_memory_sz   = DnXsyevdx_reqs.total_host_memory_sz;
   result.total_device_memory_sz = std::max(
      2*(max_det_count + 1)*(sizeof(Det) + sizeof(real)) + 2*sizeof(real),
      (max_elem_count + 1)*sizeof(real) + (max_dim + 1)*sizeof(real) + 2*sizeof(int) + DnXsyevdx_reqs.total_device_memory_sz
      );

   return result;
}

bool HubbardComputeDevice::prepare(HubbardSizes hsz)
{
   ComputeMemoryReqs csz = get_memory_requirements(hsz);
   ctx->h_arena.reserve(csz.total_host_memory_sz, true);
   ctx->d_arena.reserve(csz.total_device_memory_sz, true);

   return (ctx->h_arena.max_size() >= csz.total_host_memory_sz) &&
          (ctx->d_arena.max_size() >= csz.total_device_memory_sz);
}

/*
void HubbardComputeDevice::reset()
{ 
   cudaDeviceReset();

   if(errors) { errors->reset(); }

   size_t device_workspace_init_size = 100*1024*1024;
   size_t host_workspace_init_size = 100*1024*1024;
   ctx.reset(new ComputeContext(device_workspace_init_size, host_workspace_init_size, errors));

   CHECK_NO_CUDA_ERRORS;
}
*/

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

real HubbardComputeDevice::H_int_element(const Det* const bra_dets, const real* const bra_coeffs, int bra_count, 
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

      DArenaCheckpoint d_cpt(ctx->d_arena);

      d_bra_dets   = ctx->d_arena.allocate<Det>(bra_count);
      d_bra_coeffs = ctx->d_arena.allocate<real>(bra_count);
      d_ket_dets   = ctx->d_arena.allocate<Det>(ket_count);
      d_ket_coeffs = ctx->d_arena.allocate<real>(ket_count);

      cuda_set(d_bra_dets,   bra_dets,   bra_count);
      cuda_set(d_bra_coeffs, bra_coeffs, bra_count);
      cuda_set(d_ket_dets,   ket_dets,   ket_count);
      cuda_set(d_ket_coeffs, ket_coeffs, ket_count);
      CHECK_NO_CUDA_ERRORS;

      real* d_result = ctx->d_arena.allocate<real>(1);
      cudaMemset(d_result, 0, sizeof(real));
      H_int_element_term<<<block_count, threads_per_block>>>(d_bra_dets, d_bra_coeffs,
                                                             d_ket_dets, d_ket_coeffs,
                                                             bra_count, params, d_result);
      result = cuda_get(d_result);
      result *= params.U/params.Ns;

      CHECK_NO_CUDA_ERRORS;
   }

   return result;
}

real HubbardComputeDevice::sym_eigs_smallest(real* elements, int dim)
{
   DArenaCheckpoint d_cpt(ctx->d_arena);

   int64_t found_count;
   size_t elem_count = dim*dim;
   real* d_elements = ctx->d_arena.allocate<real>(elem_count);
   cuda_set(d_elements, elements, elem_count);

   real* W = ctx->d_arena.allocate<real>(dim);
   int* d_info = ctx->d_arena.allocate<int>(1);

   cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_NOVECTOR;
   cusolverEigRange_t range = CUSOLVER_EIG_RANGE_I;
   cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;
   int64_t lda = dim;
   int64_t il = 1;
   int64_t iu = 1;

   size_t required_device_workspace_size;
   size_t required_host_workspace_size;
   cusolverStatus_t buffer_info_status = cusolverDnXsyevdx_bufferSize(
       ctx->cusolver_handle,            // cusolverDnHandle_t handle,
       NULL,                            // cusolverDnParams_t params,
       jobz,                            // cusolverEigMode_t jobz,
       range,                           // cusolverEigRange_t range,
       uplo,                            // cublasFillMode_t uplo,
       dim,                             // int64_t n,
       CUDA_REAL,                       // cudaDataType dataTypeA
       d_elements,                      // void *A,
       lda,                             // int64_t lda,
       0,                               // void * vl,
       0,                               // void * vu,
       il,                              // int64_t il,
       iu,                              // int64_t iu,
       &found_count,                    // int64_t *meig64,
       CUDA_REAL,                       // cudaDataType dataTypeW,
       W,                               // void *W,
       CUDA_COMP_TYPE,                  // cudaDataType computeType,
       &required_device_workspace_size, // size_t *workspaceInBytesOnDevice,
       &required_host_workspace_size    // size_t *workspaceInBytesOnHost
   );

   void* h_workspace = (void*)ctx->h_arena.allocate<u8>(required_host_workspace_size);
   void* d_workspace = (void*)ctx->d_arena.allocate<u8>(required_device_workspace_size);

   cusolverStatus_t status = cusolverDnXsyevdx(
       ctx->cusolver_handle,          // cusolverDnHandle_t handle,
       NULL,                          // cusolverDnParams_t params,
       jobz,                          // cusolverEigMode_t jobz,
       range,                         // cusolverEigRange_t range,
       uplo,                          // cublasFillMode_t uplo,
       dim,                           // int64_t n,
       CUDA_REAL,                     // cudaDataType dataTypeA
       d_elements,                    // void *A,
       lda,                           // int64_t lda,
       0,                             // void * vl,
       0,                             // void * vu,
       il,                            // int64_t il,
       iu,                            // int64_t iu,
       &found_count,                  // int64_t *meig64,
       CUDA_REAL,                     // cudaDataType dataTypeW,
       (void*)W,                      // void *W,
       CUDA_COMP_TYPE,                // cudaDataType computeType,
       d_workspace,                   // void *bufferOnDevice,
       required_device_workspace_size,// size_t workspaceInBytesOnDevice,
       h_workspace,                   // void *bufferOnHost,
       required_host_workspace_size,  // size_t workspaceInBytesOnHost,
       d_info                         // int *info
   );

   int info = cuda_get(d_info);
   assert(status == CUSOLVER_STATUS_SUCCESS);
   assert(info == 0);

   real result = cuda_get(W);

   return result;
}
