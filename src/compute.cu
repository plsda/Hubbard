#include <utility>
#include <helper_cuda.h>
#include <cuda/atomic>
#include <cusolver_common.h>
#include <cusolverDn.h>
#include <cuda.h>

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

void* cuda_malloc_managed(size_t size)
{
   void* result;
   cudaError_t error = cudaMallocManaged(&result, size);
   if(error != cudaSuccess)
   {
      result = 0;
      assert(!"cuda_malloc_managed failed!");
   }

   return result;
}

void cuda_free(void* ptr)
{
   cudaError_t error = cudaFree(ptr);
   assert("cuda_free failed!" && (error == cudaSuccess));
}

using CUArenaAllocator = allocation::ArenaAllocator<cuda_malloc_managed, cuda_free, allocation::forbidden_realloc>;
using CUArenaCheckpoint = typename CUArenaAllocator::Checkpoint;

struct HubbardComputeDevice::ComputeContext
{
   ComputeContext(size_t device_workspace_init_size, int _dev_ID = 0, ErrorStream* const _errors = 0) :
      cu_arena(device_workspace_init_size), dev_ID(_dev_ID), errors(_errors)
   {
      cusolverStatus_t status = cusolverDnCreate(&cusolver_handle);
      assert(status == CUSOLVER_STATUS_SUCCESS);
      CAPTURE_LAST_CUDA_ERROR(errors);

      cuCtxGetCurrent(&cu_ctx);
      assert(cu_ctx);

      for(cudaStream_t& s : cu_streams)
      {
         CAPTURE_CUDA_ERROR(cudaStreamCreate(&s), errors);
      }

      CAPTURE_CUDA_ERROR(cudaDeviceSetLimit(cudaLimitDevRuntimePendingLaunchCount, 65536), errors);

      cudaDeviceGetAttribute(&has_concurrent_managed_access, cudaDevAttrConcurrentManagedAccess, dev_ID);
      cudaDeviceGetAttribute(&has_pageable_memory_access, cudaDevAttrPageableMemoryAccess, dev_ID);
   }

   ~ComputeContext()
   {
      cusolverStatus_t status = cusolverDnDestroy(cusolver_handle);
      assert(status == CUSOLVER_STATUS_SUCCESS);

      for(cudaStream_t& s : cu_streams)
      {
         CAPTURE_CUDA_ERROR(cudaStreamDestroy(s), errors);
      }
   }

   ErrorStream* const errors;
   cusolverDnHandle_t cusolver_handle;
   CUArenaAllocator cu_arena;
   int dev_ID;
   CUcontext cu_ctx;
   std::mutex ctx_mutex;
   cudaStream_t cu_streams[2];
   int has_concurrent_managed_access = -1;
   int has_pageable_memory_access = -1;

   // NOTE: Unified memory and concurrent page migration limitations on Windows:
   //          https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#um-requirements
   //          https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#um-data-migration


   template<class T>
   void advise_location(T* ptr, size_t count, int dev_ID)
   {
      if((has_concurrent_managed_access && has_pageable_memory_access) || (dev_ID == cudaCpuDeviceId))
      {
         CAPTURE_CUDA_ERROR(cudaMemAdvise((void*)ptr, sizeof(T)*count, cudaMemAdviseSetPreferredLocation, dev_ID), errors);
      }
   }
   template<class T>
   void advise_location(T* ptr, size_t count) { advise_location(ptr, count, dev_ID); }
   
   template<class T>
   void advise_accessor(T* ptr, size_t count, int dev_ID)
   {
      if((has_concurrent_managed_access && has_pageable_memory_access) || (dev_ID == cudaCpuDeviceId))
      {
         CAPTURE_CUDA_ERROR(cudaMemAdvise((void*)ptr, sizeof(T)*count, cudaMemAdviseSetAccessedBy, dev_ID), errors);
      }
   }
   template<class T>
   void advise_accessor(T* ptr, size_t count) { advise_accessor(ptr, count, dev_ID); }
   
   template<class T>
   void advise_read_mostly(T* ptr, size_t count)
   {
      //if(has_pageable_memory_access)
      //{
         CAPTURE_CUDA_ERROR(cudaMemAdvise((void*)ptr, sizeof(T)*count, cudaMemAdviseSetReadMostly, dev_ID), errors);
      //}
   }
   
   template<class T>
   void advise_unset_location(T* ptr, size_t count)
   {
      CAPTURE_CUDA_ERROR(cudaMemAdvise((void*)ptr, sizeof(T)*count, cudaMemAdviseUnsetPreferredLocation, dev_ID), errors);
   }
   
   template<class T>
   void advise_unset_accessor(T* ptr, size_t count, int dev_ID)
   {
      if(has_pageable_memory_access || (dev_ID == cudaCpuDeviceId))
      {
         CAPTURE_CUDA_ERROR(cudaMemAdvise((void*)ptr, sizeof(T)*count, cudaMemAdviseUnsetAccessedBy, dev_ID), errors);
      }
   }
   template<class T>
   void advise_unset_accessor(T* ptr, size_t count) { advise_unset_accessor(ptr, count, dev_ID); }
   
   template<class T>
   void advise_unset_read_mostly(T* ptr, size_t count)
   {
      //if(has_pageable_memory_access)
      //{
         CAPTURE_CUDA_ERROR(cudaMemAdvise((void*)ptr, sizeof(T)*count, cudaMemAdviseUnsetReadMostly, dev_ID), errors);
      //}
   }
   
   template<class T>
   void advise_all(T* ptr, size_t count, int dev_ID)
   {
      advise_location(ptr, count, dev_ID);
      advise_accessor(ptr, count, dev_ID);
      advise_read_mostly(ptr, count);
   }
   template<class T>
   void advise_all(T* ptr, size_t count) { advise_all(ptr, count, dev_ID); }
   
   template<class T>
   void advise_unset_all(T* ptr, size_t count, int dev_ID)
   {
      advise_unset_location(ptr, count);
      advise_unset_accessor(ptr, count, dev_ID);
      advise_unset_read_mostly(ptr, count);
   }
   template<class T>
   void advise_unset_all(T* ptr, size_t count) { advise_unset_all(ptr, count, dev_ID); }

   template<class T>
   void prefetch(T* ptr, size_t count, int dev_ID, cudaStream_t stream = 0)
   {
   #ifndef _WIN32
      if(has_concurrent_managed_access || (dev_ID == cudaCpuDeviceId))
      {
         CAPTURE_CUDA_ERROR(cudaMemPrefetchAsync((void*)ptr, sizeof(T)*count, dev_ID, stream), errors);
      }
   #endif
   }
   template<class T>
   void prefetch(T* ptr, size_t count, cudaStream_t stream = 0) { prefetch(ptr, count, dev_ID, stream); }

};

template<class T>
void cuda_set(T* dest, const T* src, size_t count = 1, ErrorStream* errors = 0)
{
   CAPTURE_CUDA_ERROR(cudaMemcpy(dest, src, sizeof(T)*count, cudaMemcpyHostToDevice), errors);
}

template<class T>
T cuda_get(T* src, ErrorStream* errors = 0)
{
   T result;
   CAPTURE_CUDA_ERROR(cudaMemcpy(&result, src, sizeof(T), cudaMemcpyDeviceToHost), errors);
   return result;
}

HubbardComputeDevice::HubbardComputeDevice(size_t device_workspace_init_size, ArenaAllocator& alloc, ErrorStream* errors) : errors(errors), h_arena(alloc)
{
   int dev_ID = gpuGetMaxGflopsDeviceId();
   bool init_ok = (gpuDeviceInit(dev_ID) >= 0);
   assert(init_ok);

   ctx = std::make_unique<ComputeContext>(device_workspace_init_size, dev_ID, errors);
}

HubbardComputeDevice::~HubbardComputeDevice()
{
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

   size_t max_dim = size_t(sz.max_KS_dim);
   size_t max_elem_count = size_t(max_dim*max_dim);

   ComputeMemoryReqs DnXsyevdx_reqs = DnXsyevdx_memory_requirements(ctx->cusolver_handle, sz);
  
   result.total_host_memory_sz   = DnXsyevdx_reqs.total_host_memory_sz;
   result.total_device_memory_sz = std::max(
      sizeof(WeightedDet)*(sz.CSF_coeff_count_upper_bound + 1) + sizeof(int)*(sz.max_KS_dim + 2) + sizeof(real)*2*(sz.max_KS_dim + 1),
      sizeof(real)*(max_elem_count + 1) + sizeof(real)*(max_dim + 1) + sizeof(int)*2 + DnXsyevdx_reqs.total_device_memory_sz
      );

   return result;
}

bool HubbardComputeDevice::prepare(HubbardSizes hsz)
{
   ComputeMemoryReqs csz = get_memory_requirements(hsz);
   assert(h_arena.unused_size() >= csz.total_host_memory_sz);
   ctx->cu_arena.reserve(csz.total_device_memory_sz, true);

   return (h_arena.unused_size() >= csz.total_host_memory_sz) &&
          (ctx->cu_arena.unused_size() >= csz.total_device_memory_sz);
}

bool HubbardComputeDevice::begin_compute()
{
   bool result = false;
   if(ctx->ctx_mutex.try_lock())
   {
      CUresult error = cuCtxSetCurrent(ctx->cu_ctx);
      result = (error == CUDA_SUCCESS);
      assert(result);
   }
   else
   {
      assert(!"CUDA context is already in use in another thread!");
   }

   return result;
}

bool HubbardComputeDevice::end_compute()
{
   CUresult error = cuCtxSetCurrent(0);
   bool result = (error == CUDA_SUCCESS);
   assert(result);

   ctx->ctx_mutex.unlock();
   return result;
}

void HubbardComputeDevice::begin_device_memory(ArenaCheckpoint*& result)
{
   size_t used_before = ctx->cu_arena.used_size();
   CUArenaCheckpoint* cpt = ctx->cu_arena.allocate<CUArenaCheckpoint>(1);
   new(cpt) CUArenaCheckpoint(ctx->cu_arena);
   cpt->used_size = used_before;
   result = reinterpret_cast<ArenaCheckpoint*>(cpt);
}

void HubbardComputeDevice::end_device_memory(ArenaCheckpoint* cpt)
{
   allocation::clear_checkpoint(reinterpret_cast<CUArenaCheckpoint*>(cpt));
}

template<class T>
T* HubbardComputeDevice::dev_allocate(size_t count)
{
   T* result = ctx->cu_arena.allocate<T>(count);

   //ctx->advise_location(result, count, cudaCpuDeviceId);
   //ctx->advise_accessor(result, count, cudaCpuDeviceId);
   //ctx->prefetch(result, count, cudaCpuDeviceId);

   return result;
}

template WeightedDet* HubbardComputeDevice::dev_allocate<WeightedDet>(size_t);
template real* HubbardComputeDevice::dev_allocate<real>(size_t);
template int* HubbardComputeDevice::dev_allocate<int>(size_t);

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
void cu_H_int_element(const WeightedDet* const row_csf, const WeightedDet* const col_csf, int col_csf_count,
                      HubbardParams params, real* const result)
{
   int row_csf_det_idx = blockIdx.x;
   int q  = threadIdx.x;
   int k1 = blockIdx.y*blockDim.y + threadIdx.y;
   int k2 = blockIdx.z*blockDim.z + threadIdx.z;

   __shared__ real block_res;
   if(threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0)
   {
      block_res = 0;
   }
   __syncthreads();

   if((k1 < params.Ns) && (k2 < params.Ns))
   {
      WeightedDet row_det = row_csf[row_csf_det_idx];
      auto [new_ket_det, new_ket_det_sign] = sadd(mod(k1 + q, params.Ns),
                                                  ssub(k1,
                                                       sadd(mod(k2 - q, params.Ns) + params.Ns,
                                                            ssub(k2 + params.Ns, 
                                                                 row_det.det))));

      if(new_ket_det_sign != 0 && cmp_det_config(new_ket_det, col_csf->det, params))
      {
         real cur_res = 0;
         for(int col_det_idx = 0; col_det_idx < col_csf_count; col_det_idx++)
         {
            WeightedDet col_det = col_csf[col_det_idx];
            cur_res += new_ket_det_sign*row_det.coeff*col_det.coeff*(new_ket_det == col_det.det);
         }

         atomicAdd(&block_res, cur_res);
      }
   }

   __syncthreads();
   if(threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0)
   {
      atomicAdd(result, block_res*params.U/params.Ns);
   }
}

__global__
void cu_H_int_col(int col_idx, const WeightedDet* const basis, const int* const csf_indices, int H_dim, HubbardParams params, real* const result)
{
   const WeightedDet* const col_csf = basis + csf_indices[col_idx];
   const int col_csf_count = csf_indices[col_idx + 1] - csf_indices[col_idx];

   int result_idx = blockIdx.x*blockDim.x + threadIdx.x;
   int row_idx = result_idx + col_idx;

   if(row_idx < H_dim)
   {
      const WeightedDet* row_csf = basis + csf_indices[row_idx];
      int row_csf_count = csf_indices[row_idx + 1] - csf_indices[row_idx];

      Det row_ref_det = row_csf->det;
      Det col_ref_det = col_csf->det;
      auto [bup, bdn] = get_det_up_down(row_ref_det, params);
      auto [kup, kdn] = get_det_up_down(col_ref_det, params);
      int spinless_diff = get_pop((bup | bdn) ^ (kup | kdn)) +
                          get_pop((bup & bdn) ^ (kup & kdn));

      result[result_idx] = 0;
      if(spinless_diff <= 4)
      {
         int yz_block_n = 2;
         int yz_block_dim = std::ceil(float(params.Ns)/float(yz_block_n));
         dim3 block_count = {unsigned int(row_csf_count),
                             unsigned int(yz_block_n),
                             unsigned int(yz_block_n)};
         dim3 threads_per_block = {unsigned int(params.Ns),
                                   unsigned int(yz_block_dim),
                                   unsigned int(yz_block_dim)};

         cu_H_int_element<<<block_count, threads_per_block>>>(row_csf, col_csf, col_csf_count, params, result + result_idx);
      }
   }
}

void HubbardComputeDevice::H_int(real* result, int KS_dim, const WeightedDet* const basis,
                                 int det_count, const int* const csf_indices,
                                 const HubbardParams& params)
{
   CUArenaCheckpoint dcpt(ctx->cu_arena);

   ctx->advise_location(basis, det_count, cudaCpuDeviceId);
   ctx->advise_location(csf_indices, KS_dim + 1, cudaCpuDeviceId);
   ctx->advise_accessor(basis, det_count);
   ctx->advise_accessor(csf_indices, KS_dim + 1);

   ctx->prefetch(basis, det_count);
   ctx->prefetch(csf_indices, KS_dim + 1);
   ctx->advise_location(basis, det_count);
   ctx->advise_location(csf_indices, KS_dim + 1);

   ctx->advise_read_mostly(basis, det_count);
   ctx->advise_read_mostly(csf_indices, KS_dim + 1);

   real* result_buffer1 = ctx->cu_arena.allocate<real>(KS_dim);
   size_t buf2_effective_sz = ctx->cu_arena.get_aligned_size<real>(KS_dim);
   real* result_buffer2 = ctx->cu_arena.allocate<real>(KS_dim);
   ctx->prefetch(result_buffer1, KS_dim + buf2_effective_sz);

   const int threads_per_block = 256;
   for(int col = 0; col < KS_dim; col++)
   {
      int row_count = KS_dim - col;
      if(col > 0)
      {
         int prev_col = col - 1;
         CAPTURE_CUDA_ERROR(cudaMemcpyAsync(result + prev_col*KS_dim + prev_col, result_buffer2, sizeof(real)*(row_count + 1), cudaMemcpyDeviceToHost, ctx->cu_streams[1]), errors);
      }

      // NOTE: Only store the lower triangular part
      if(row_count < threads_per_block)
      {
         cu_H_int_col<<<1, row_count, 0, ctx->cu_streams[0]>>>(col, basis, csf_indices, KS_dim, params, result_buffer1); 
      }
      else
      {
         int block_count = std::ceil(float(row_count)/float(threads_per_block));
         cu_H_int_col<<<block_count, threads_per_block, 0, ctx->cu_streams[0]>>>(col, basis, csf_indices, KS_dim, params, result_buffer1); 
      }
      CHECK_NO_CUDA_ERRORS;
      CAPTURE_CUDA_ERROR(cudaDeviceSynchronize(), errors);

      std::swap(result_buffer1, result_buffer2);
   }
   result[KS_dim*KS_dim - 1] = cuda_get(result_buffer2, errors);

   ctx->advise_unset_all(basis, det_count);
   ctx->advise_unset_all(csf_indices, KS_dim + 1);
}

real HubbardComputeDevice::H_int_element(const Det* const bra_dets, const real* const bra_coeffs, int bra_count, 
                                         const Det* const ket_dets, const real* const ket_coeffs, int ket_count,
                                         const HubbardParams& params)
{
   assert(!"Not implemented.");
   return 0;
}

real HubbardComputeDevice::sym_eigs_smallest(real* elements, int dim)
{
   CUArenaCheckpoint d_cpt(ctx->cu_arena);

   int64_t found_count;
   size_t elem_count = dim*dim;
   real* d_elements = ctx->cu_arena.allocate<real>(elem_count);
   real* W          = ctx->cu_arena.allocate<real>(dim);
   int* d_info      = ctx->cu_arena.allocate<int>(1);
   cuda_set(d_elements, elements, elem_count);

   ctx->advise_accessor(d_elements, elem_count + dim + 1);
   ctx->prefetch(d_elements, elem_count + dim + 1);
   ctx->advise_location(d_elements, elem_count + dim + 1);

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

   void* h_workspace = (void*)h_arena.allocate<u8>(required_host_workspace_size);
   void* d_workspace = (void*)ctx->cu_arena.allocate<u8>(required_device_workspace_size);

   ctx->advise_accessor((u8*)d_workspace, required_device_workspace_size);
   ctx->prefetch((u8*)d_workspace, required_device_workspace_size);
   ctx->advise_location((u8*)d_workspace, required_device_workspace_size);

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

   ctx->advise_unset_all(d_elements, elem_count + dim + 1);
   ctx->advise_unset_all((u8*)d_workspace, required_device_workspace_size);

   return result;
}
