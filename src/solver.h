#ifndef SOLVER_H

#include <numbers>
#define PI std::numbers::pi_v<real>
#define LOG2 std::numbers::log2e_v<real>

#include <Eigen/Dense>
using MatR = Eigen::Matrix<real, Eigen::Dynamic, Eigen::Dynamic>;
using Arr2R = Eigen::Array<real, Eigen::Dynamic, Eigen::Dynamic>;
const size_t REAL_EIGEN_ALIGNMENT = []() constexpr -> size_t 
{  
   const size_t a = std::alignment_of_v<real>;
   if constexpr (a <= 16u)  { return 16u; }
   else if      (a <= 32u)  { return 32u; }
   else if      (a <= 64u)  { return 64u; }
   else if      (a <= 128u) { return 128u; }

   return a;
}();
const Eigen::AlignmentType REAL_EIGEN_ALIGNMENT_TYPE = []() constexpr
{  
   switch(REAL_EIGEN_ALIGNMENT)
   {
      case 8:   { return Eigen::AlignmentType::Aligned8;   } break;
      case 16:  { return Eigen::AlignmentType::Aligned16;  } break;
      case 32:  { return Eigen::AlignmentType::Aligned32;  } break;
      case 64:  { return Eigen::AlignmentType::Aligned64;  } break;
      case 128: { return Eigen::AlignmentType::Aligned128; } break;
   }

   assert(!"Unalignable real type.");
   return Eigen::AlignmentType::Unaligned;
}();
using MMatR = Eigen::Map<MatR, REAL_EIGEN_ALIGNMENT_TYPE>;
using MArr2R = Eigen::Map<Arr2R, REAL_EIGEN_ALIGNMENT_TYPE>;

enum class BCS
{
   OPEN = 1,
   PERIODIC = 2,
};

struct HubbardSizes;
class StrideItr;
class KSBlockIterator;
class CSFItr;
class HubbardModel;

HubbardSizes operator*(const HubbardSizes& sz, int n);
HubbardSizes operator*(int n, const HubbardSizes& sz);

real noninteracting_E(int n, real T, int Ns, BCS bcs);
real noninteracting_E0(const HubbardParams& params, BCS bcs);
real dimer_E0(const HubbardParams& params, BCS bcs);
real atomic_E0(const HubbardParams& params);
real halffilled_E_per_N(real T, real U, IntArgs int_args);
HubbardSizes hubbard_memory_requirements(HubbardParams params);
Det get_config_ref_det(const std::span<Det>& config) { return config.front(); }

struct HubbardSizes
{
   int basis_size;
   int min_singles;
   int max_singles;
   int config_count;
   int K_block_config_count_upper_bound;
   int KS_block_config_count_upper_bound;
   int max_KS_dim;
   int max_dets_in_config;
   int max_S_paths;
   int CSF_coeff_count_upper_bound;
   // In bytes:
   size_t alloc_pad; 
   size_t unaligned_workspace_size;
   size_t workspace_size;
};

class StrideItr
{
public:
   using iterator_category = std::forward_iterator_tag;
   using difference_type   = std::ptrdiff_t;
   using value_type        = Det;
   using pointer           = Det*;
   using reference         = Det&;

   StrideItr(int _idx, int _strided_idx) : idx(_idx), strided_idx(_strided_idx), strides(0) { }
   StrideItr(std::span<Det> _data, std::vector<int, IntArena>* _strides, int _idx = 0, int _strided_idx = 0) :
      data(_data), strides(_strides), idx(_idx), strided_idx(_strided_idx) { }

   std::span<Det> operator*() const;
   StrideItr& operator++();
   StrideItr operator++(int);

   StrideItr begin() { return *this; }
   //StrideItr begin() { return StrideItr(data, strides, idx, strided_idx); }
   StrideItr end()   { return StrideItr(-1, data.size()); }

   friend bool operator== (const StrideItr& a, const StrideItr& b) { return a.strided_idx == b.strided_idx; };
   friend bool operator!= (const StrideItr& a, const StrideItr& b) { return a.strided_idx != b.strided_idx; };  

   union
   {
      int idx;
      int config_idx;
   };
   union
   {
      int strided_idx;
      int det_idx;
   };

private:
   std::vector<int, IntArena>* strides;
   std::span<Det> data;
};

class CSFItr
{
public:
   using iterator_category = std::forward_iterator_tag;
   using difference_type   = std::ptrdiff_t;
   using value_type        = CSFItr;
   using pointer           = CSFItr*;
   using reference         = CSFItr&;

   CSFItr(int cidx, int pidx, int idx, int first_coeff_idx, const KSBlockIterator& itr);

   reference operator*() { return *this; }
   CSFItr& operator++();
   CSFItr operator++(int);

   friend bool operator== (const CSFItr& a, const CSFItr& b) { return a.idx == b.idx; };
   friend bool operator!= (const CSFItr& a, const CSFItr& b) { return a.idx != b.idx; };  

   int config_idx;
   int S_path_idx;
   int idx;
   int first_coeff_idx;

private:
   const KSBlockIterator& block_itr;
};

class KSBlockIterator
{
   ArenaAllocator* allocator;
   ArenaCheckpoint cpt;

public:
   KSBlockIterator(ArenaAllocator& _allocator);
   KSBlockIterator(HubbardParams _params, ArenaAllocator& _allocator, HubbardSizes sz);

   KSBlockIterator& operator++();
   operator bool() const { return has_blocks_left; }
   void reset();
   bool next_K_block();
   bool next_S_block();
   void init_K_block(int idx);
   void init_S_block(int idx);

   StrideItr K_configs();

   class CSFs
   {
   public:
      int start_cidx = 0;
      int start_pidx = 0;
      int start_idx = 0;
      int start_first_coeff_idx = 0;
      const KSBlockIterator& block_itr;
      int KS_dim;

      CSFItr begin() { return CSFItr(start_cidx, start_pidx, start_idx, start_first_coeff_idx, block_itr); }
      CSFItr end() { return CSFItr(-1, -1, KS_dim, start_first_coeff_idx, block_itr); }
   };

   CSFs KS_basis(CSFItr first);
   CSFs KS_basis();

   MArr2R& KS_H()
   {
      return _KS_H;
   }
   real& KS_H(CSFItr csf1, CSFItr csf2)
   {
      return _KS_H(csf1.idx, csf2.idx);
   }
   Det* CSF_dets(CSFItr itr)
   {
      return KS_configs[itr.config_idx].data();
   }
   std::span<Det>& CSF_dets_sp(CSFItr itr)
   {
      return KS_configs[itr.config_idx];
   }
   real* CSF_coeffs(CSFItr itr)
   {
      return KS_CSF_coeffs.data() + itr.first_coeff_idx;
   }
   std::span<real> CSF_coeffs_sp(CSFItr itr)
   {
      return std::span<real>(KS_CSF_coeffs.begin() + itr.first_coeff_idx, CSF_size(itr));
   }
   int CSF_size(CSFItr itr) const
   {
      return KS_configs[itr.config_idx].size();
   }

#ifdef HUBBARD_TEST
   void clear_block_data()
   {
      K_single_counts.resize(0);
      KS_configs.resize(0);
      KS_S_path_counts.resize(0);
      KS_single_counts.resize(0);
      KS_CSF_coeffs.resize(0);
      KS_spins.resize(0);
      KS_dim = 0;
   }
   void append_block_data(const KSBlockIterator& other)
   {
      KS_configs.insert(KS_configs.end(), other.KS_configs.cbegin(), other.KS_configs.cend()); // NOTE: This needs the basis array from the other KSBlockIterator
      KS_S_path_counts.insert(KS_S_path_counts.end(), other.KS_S_path_counts.cbegin(), other.KS_S_path_counts.cend());
      KS_single_counts.insert(KS_single_counts.end(), other.KS_single_counts.cbegin(), other.KS_single_counts.cend());
      KS_CSF_coeffs.insert(KS_CSF_coeffs.end(), other.KS_CSF_coeffs.cbegin(), other.KS_CSF_coeffs.cend());
      KS_spins.insert(KS_spins.end(), other.KS_spins.cbegin(), other.KS_spins.cend());

      KS_dim += other.KS_dim;
   }
   void move_basis(KSBlockIterator& other)
   {
      basis = std::move(other.basis);
   }
   std::vector<real, RealArena> KS_spins;
#endif

   HubbardParams params;
   int K_count;
   int S_count;

   // NOTE: KS_* quantities are valid only for the current KS-block
   //       Certain K_* quantities are valid only for the current K-block
   // NOTE: Here 'config' is short for 'orbital configuration' and an orbital configuration here is an array of dets with the same spatial part (orbital configuration) but different spin configuration
   std::vector<std::span<Det>, SpanArena> KS_configs;
   std::vector<int, IntArena>             KS_single_counts;
   std::vector<int, IntArena>             KS_S_path_counts;
   std::vector<real, RealArena>           KS_CSF_coeffs;
   int KS_dim;

private:
   void form_KS_subbasis();
   void form_CSFs();

   real m;
   real S_min;
   real S_max;

   int total_CSF_count;
   int K_block_CSF_count;

   int K_block_idx; 
   int S_block_idx;
   int K_KS_basis_begin_idx;
   bool has_blocks_left;
   real S; // Total spin of the current block

   // NOTE: 'basis' must not be resized/reallocated since KS_configs stores spans referring to it!
   std::vector<Det, DetArena> basis;                 // The determinantal basis sorted according to total momentum and orbital configuration
   std::vector<int, IntArena> K_block_sizes;         // Total number of dets in a K-block
   std::vector<int, IntArena> K_block_begin_indices; // Cumulative K-block sizes

   std::vector<int, IntArena> K_single_counts;
   std::vector<int, IntArena> K_dets_per_config;
   std::vector<Det, DetArena> S_paths;

   int KS_max_path_count; // NOTE: For debug only

   MArr2R _KS_H;
   real* KS_H_data;
};

class HubbardModel
{
public:
   HubbardModel(HubbardComputeDevice& _cdev, ArenaAllocator& _allocator) :
      sz({}), allocator(_allocator), cdev(_cdev), itr(allocator), recompute_E(true), recompute_basis(false) { }

   HubbardModel(const HubbardParams& _params, HubbardComputeDevice& _cdev, ArenaAllocator& _allocator) :
      params(_params), sz(hubbard_memory_requirements(_params)), cdev(_cdev), allocator(_allocator),
      itr(params, allocator, sz), recompute_E(true), recompute_basis(false) { }

   void U(real new_U)
   { 
      recompute_E = true;
      params.U = new_U; 
   }
   void T(real new_T) 
   { 
      recompute_E = true;
      params.T = new_T;
   }
   // TODO: Remove these and always create a new HubbardModel instance when need to change these params? (need to form the basis etc. from scratch anyways)
   /*
   void Ns(int new_Ns) 
   { 
      recompute_E = true;
      recompute_basis = true;
      params.Ns = new_Ns;
   }
   void N_up(int new_N_up) 
   { 
      recompute_E = true;
      recompute_basis = true;
      params.N_up = new_N_up;
      params.N = params.N_up + params.N_down;
   }
   void N_dn(int new_N_down)
   { 
      recompute_E = true;
      recompute_basis = true;
      params.N_down = new_N_down;
      params.N = params.N_up + params.N_down;
   }
   HubbardModel& set_params(const HubbardParams& new_params)
   { 
      recompute_E = true;
      recompute_basis = true;
      params = new_params;
      return *this;
   }
   void update()
   {
      if(recompute_basis)
      {
         HubbardSizes new_sz = hubbard_memory_requirements(params);
         //if(sz.unaligned_workspace_size < new_sz.unaligned_workspace_size)
         //{
         //   allocator = ArenaAllocator(new_sz.unaligned_workspace_size); // TODO: Sus. Add padding to unaligned size + think if reassignment is a good idea. 
         //                                                                //       Check if old arena has enough space. Prob shouldn't be creating new arenas here
         //   sz = new_sz;
         //}
         sz = new_sz;
         assert(allocator.unused_size() >= new_sz.workspace_size);

         //KSBlockIterator asd(params, allocator, new_sz);
         //itr = asd;
         itr = KSBlockIterator(params, allocator, new_sz);
         recompute_basis = false;
      }
   }
   */

   real H_int(const CSFItr& csf1, const CSFItr& csf2);
   real H_0(Det det);
   real E0();

private:
   HubbardParams params;
   HubbardSizes sz;
   ArenaAllocator& allocator;
   HubbardComputeDevice& cdev;
   KSBlockIterator itr;

   real _E0;
   bool recompute_E;
   bool recompute_basis;
};

#define SOLVER_H
#endif
