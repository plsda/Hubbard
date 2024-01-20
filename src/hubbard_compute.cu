#include <utility>
#include <helper_cuda.h>
#include <cuda/atomic>
#include <cusolver_common.h>
#include <cusolverDn.h>
#include "hubbard_compute.h"

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

class HubbardComputeDevice::ComputeContext
{
public:
   ComputeContext(size_t device_workspace_init_size, size_t host_workspace_init_size, ErrorStream* const errors = 0) : errors(errors)
   {
      cusolverStatus_t status = cusolverDnCreate(&cusolver_handle);
      assert(status == CUSOLVER_STATUS_SUCCESS);

      // TODO: Pinned memory
      CAPTURE_CUDA_ERROR(cudaMalloc(&d_memory, device_workspace_init_size + sizeof(real) + sizeof(int)), errors);

      // TODO: Alignment?
      d_real_result = (real*)d_memory;
      d_info        = (int* )(((u8*)d_memory) + sizeof(real));
      d_workspace   = (void*)(((u8*)d_memory) + sizeof(real) + sizeof(int));

      h_workspace = std::make_unique<u8[]>(host_workspace_init_size);
      //h_workspace = std::make_shared<u8[]>(host_workspace_init_size);

      device_workspace_size = device_workspace_init_size;
      host_workspace_size = host_workspace_init_size;
   }

   ~ComputeContext()
   {
      CAPTURE_CUDA_ERROR(cudaFree(d_memory), errors);
   }

   // NOTE: Workspace contents are not copied over
   void resize_device_workspace(size_t new_size)
   {
      CAPTURE_CUDA_ERROR(cudaFree(d_memory), errors);

      CAPTURE_CUDA_ERROR(cudaMalloc(&d_memory, new_size + sizeof(real) + sizeof(int)), errors);
      device_workspace_size = new_size;

      d_real_result = (real*)d_memory;
      d_info        = (int* )(((u8*)d_memory) + sizeof(real));
      d_workspace   = (void*)(((u8*)d_memory) + sizeof(real) + sizeof(int));
   }

   void resize_host_workspace(size_t new_size)
   {
      h_workspace.reset(new u8[new_size]);
      host_workspace_size = new_size;
   }

   real get_real_result()
   {
      real result;
      cuda_memcpy_to_host(&result, d_real_result, sizeof(real));
      return result;
   }

   int get_info()
   {
      int result;
      cuda_memcpy_to_host(&result, d_info, sizeof(int));
      return result;
   }

   cusolverDnHandle_t cusolver_handle;

   void* d_workspace = 0;
   real* d_real_result = 0;
   int* d_info = 0;
   std::unique_ptr<u8[]> h_workspace;
   size_t device_workspace_size = 0; // In bytes
   size_t host_workspace_size = 0;   // In bytes

private:
   ErrorStream* const errors;
   void* d_memory = 0;
};

HubbardComputeDevice::HubbardComputeDevice(ErrorStream* errors) : errors(errors)
{
   bool init_ok = (gpuDeviceInit(gpuGetMaxGflopsDeviceId()) >= 0);
   assert(init_ok);

   size_t device_workspace_init_size = 100*1024*1024;
   size_t host_workspace_init_size = 100*1024*1024;
   ctx = std::make_unique<ComputeContext>(device_workspace_init_size, host_workspace_init_size, errors);
}

HubbardComputeDevice::~HubbardComputeDevice()
{

}

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

      cudaMalloc(&d_bra_dets,   sizeof(Det)*bra_count);
      cudaMalloc(&d_bra_coeffs, sizeof(real)*bra_count);
      cudaMalloc(&d_ket_dets,   sizeof(Det)*ket_count);
      cudaMalloc(&d_ket_coeffs, sizeof(real)*ket_count);

      cuda_memcpy_to_device(d_bra_dets,   bra_dets,    sizeof(Det)*bra_count);
      cuda_memcpy_to_device(d_bra_coeffs, bra_coeffs,  sizeof(real)*bra_count);
      cuda_memcpy_to_device(d_ket_dets,   ket_dets,    sizeof(Det)*ket_count);
      cuda_memcpy_to_device(d_ket_coeffs, ket_coeffs,  sizeof(real)*ket_count);

      CHECK_NO_CUDA_ERRORS;

      real* d_result;
      cudaMalloc(&d_result, sizeof(real));
      cudaMemset(d_result, 0, sizeof(real));
      H_int_element_term<<<block_count, threads_per_block>>>(d_bra_dets, d_bra_coeffs,
                                                             d_ket_dets, d_ket_coeffs,
                                                             bra_count, params, d_result);
      cuda_memcpy_to_host(&result, d_result, sizeof(result));
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

real HubbardComputeDevice::sym_eigs_smallest(real* elements, int dim)
{
   int64_t found_count;
   real* d_elements;
   size_t elements_sz = dim*dim*sizeof(real);
   cudaMalloc(&d_elements, elements_sz); // TODO: Preallocate
   cuda_memcpy_to_device(d_elements, elements, elements_sz);

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
       ctx->d_real_result,              // void *W,
       CUDA_COMP_TYPE,                  // cudaDataType computeType,
       &required_device_workspace_size, // size_t *workspaceInBytesOnDevice,
       &required_host_workspace_size    // size_t *workspaceInBytesOnHost
   );

   if(ctx->device_workspace_size < required_device_workspace_size)
   {
      ctx->resize_device_workspace(required_device_workspace_size);
   }

   if(ctx->host_workspace_size < required_host_workspace_size)
   {
      ctx->resize_host_workspace(required_host_workspace_size);
   }
       
   void* W;
   cudaMalloc(&W, dim*sizeof(real)); // TODO: Preallocate
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
       W,                             // void *W,
       CUDA_COMP_TYPE,                // cudaDataType computeType,
       ctx->d_workspace,              // void *bufferOnDevice,
       ctx->device_workspace_size,    // size_t workspaceInBytesOnDevice,
       ctx->h_workspace.get(),        // void *bufferOnHost,
       ctx->host_workspace_size,      // size_t workspaceInBytesOnHost,
       ctx->d_info                    // int *info
   );
   cudaFree(d_elements);

   int info = ctx->get_info();
   assert(status == CUSOLVER_STATUS_SUCCESS);
   assert(info == 0);

   real result;
   cuda_memcpy_to_host(&result, W, sizeof(result));
   cudaFree(W);

   return result;
}
