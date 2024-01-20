#ifndef SOLVER_H

#include <numbers>
#define PI std::numbers::pi_v<real>
#define LOG2 std::numbers::log2e_v<real>

#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
using MatR = Eigen::Matrix<real, Eigen::Dynamic, Eigen::Dynamic>;
using Arr2R = Eigen::Array<real, Eigen::Dynamic, Eigen::Dynamic>;
using Arr3R = Eigen::Tensor<real, 3>;
using MMatR = Eigen::Map<MatR, Eigen::AlignmentType::Aligned>;
using MArr2R = Eigen::Map<Arr2R, Eigen::AlignmentType::Aligned>;

enum class BCS
{
   OPEN = 1,
   PERIODIC = 2,
};

real noninteracting_E(int n, real T, int Ns, BCS bcs);
real noninteracting_E0(const HubbardParams& params, BCS bcs);
real dimer_E0(const HubbardParams& params, BCS bcs);
real atomic_E0(const HubbardParams& params);
real halffilled_E_per_N(real T, real U, IntArgs int_args);
real KSM_basis_compute_E0(HubbardComputeDevice& cdev, ArenaAllocator allocator, const HubbardParams& params);

Det get_config_ref_det(const std::span<Det>& config) { return config.front(); }

class StrideItr
{
public:
   using iterator_category = std::forward_iterator_tag;
   using difference_type   = std::ptrdiff_t;
   using value_type        = Det;
   using pointer           = Det*;
   using reference         = Det&;

   StrideItr(int _idx, int _strided_idx) : idx(_idx), strided_idx(_strided_idx), strides(0) { }
   StrideItr(std::span<Det> _data, std::vector<int>* _strides, int _idx = 0, int _strided_idx = 0) :
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
   std::vector<int>* strides;
   std::span<Det> data;
};

class KSBlockIterator;
class SCFItr
{
public:
   using iterator_category = std::forward_iterator_tag;
   using difference_type   = std::ptrdiff_t;
   using value_type        = SCFItr;
   using pointer           = SCFItr*;
   using reference         = SCFItr&;

   SCFItr(int cidx, int pidx, int idx, int first_coeff_idx, const KSBlockIterator& itr);

   reference operator*() { return *this; }
   SCFItr& operator++();
   SCFItr operator++(int);

   friend bool operator== (const SCFItr& a, const SCFItr& b) { return a.idx == b.idx; };
   friend bool operator!= (const SCFItr& a, const SCFItr& b) { return a.idx != b.idx; };  

   int config_idx;
   int S_path_idx;
   int idx;
   int first_coeff_idx;

private:
   const KSBlockIterator& block_itr;
};

class KSBlockIterator
{
public:
   KSBlockIterator(HubbardParams _params, size_t memory_size) : params(_params), allocator(ArenaAllocator(memory_size)), K_count(params.Ns), _KS_H(MArr2R(NULL, 0, 0)) { init(); }
   KSBlockIterator(HubbardParams _params, ArenaAllocator _allocator) : params(_params), allocator(_allocator), K_count(params.Ns), _KS_H(MArr2R(NULL, 0, 0)) { init(); }

   KSBlockIterator& operator++();
   operator bool() const { return has_blocks_left; }
   void reset();
   bool next_K_block();
   bool next_S_block();
   void init_K_block(int idx);
   void init_S_block(int idx);

   StrideItr K_configs();

   class SCFs
   {
   public:
      int start_cidx = 0;
      int start_pidx = 0;
      int start_idx = 0;
      int start_first_coeff_idx = 0;
      const KSBlockIterator& block_itr;
      int KS_dim;

      SCFItr begin() { return SCFItr(start_cidx, start_pidx, start_idx, start_first_coeff_idx, block_itr); }
      SCFItr end() { return SCFItr(-1, -1, KS_dim, start_first_coeff_idx, block_itr); }
   };

   SCFs KS_basis(SCFItr first);
   SCFs KS_basis();

   MArr2R& KS_H()
   {
      return _KS_H;
   }
   real& KS_H(SCFItr scf1, SCFItr scf2)
   {
      return _KS_H(scf1.idx, scf2.idx);
   }
   Det* SCF_dets(SCFItr itr)
   {
      return KS_configs[itr.config_idx].data();
   }
   std::span<Det>& SCF_dets_sp(SCFItr itr)
   {
      return KS_configs[itr.config_idx];
   }
   real* SCF_coeffs(SCFItr itr)
   {
      return KS_SCF_coeffs.data() + itr.first_coeff_idx;
   }
   std::span<real> SCF_coeffs_sp(SCFItr itr)
   {
      return std::span<real>(KS_SCF_coeffs.begin() + itr.first_coeff_idx, SCF_size(itr));
   }
   int SCF_size(SCFItr itr) const
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
      KS_SCF_coeffs.resize(0);
      KS_spins.resize(0);
      KS_dim = 0;
   }
   void append_block_data(const KSBlockIterator& other)
   {
      KS_configs.insert(KS_configs.end(), other.KS_configs.cbegin(), other.KS_configs.cend()); // NOTE: This needs the basis array from the other KSBlockIterator

      KS_S_path_counts.insert(KS_S_path_counts.end(), other.KS_S_path_counts.cbegin(), other.KS_S_path_counts.cend());
      KS_single_counts.insert(KS_single_counts.end(), other.KS_single_counts.cbegin(), other.KS_single_counts.cend());
      KS_SCF_coeffs.insert(KS_SCF_coeffs.end(), other.KS_SCF_coeffs.cbegin(), other.KS_SCF_coeffs.cend());
      KS_spins.insert(KS_spins.end(), other.KS_spins.cbegin(), other.KS_spins.cend());

      KS_dim += other.KS_dim;
   }
   void move_basis(KSBlockIterator& other)
   {
      basis = std::move(other.basis);
   }
   std::vector<real> KS_spins;
#endif

   HubbardParams params;
   int K_count;
   int S_count;

   // NOTE: KS_* quantities are valid only for the current KS-block
   //       Certain K_* quantities are valid only for the current K-block
   // NOTE: Here 'config' is short for 'orbital configuration' and an orbital configuration here is an array of dets with the same spatial part (orbital configuration) but different spin configuration
   std::vector<std::span<Det>> KS_configs;
   std::vector<int>  KS_single_counts;
   std::vector<int>  KS_S_path_counts;
   std::vector<real> KS_SCF_coeffs;
   int KS_dim;

private:
   void init();
   void form_KS_subbasis();
   void form_SCFs();

   real m;
   real S_min;
   real S_max;

   int total_SCF_count;
   int K_block_SCF_count;

   int K_block_idx; 
   int S_block_idx;
   int K_KS_basis_begin_idx;
   bool has_blocks_left;
   real S; // Total spin of the current block

   // NOTE: 'basis' must not be resized/reallocated since KS_configs stores spans referring to it!
   std::vector<Det> basis;                 // The determinantal basis sorted according to total momentum and orbital configuration
   std::vector<int> K_block_sizes;         // Total number of dets in a K-block
   std::vector<int> K_block_begin_indices; // Cumulative K-block sizes

   std::vector<int> K_single_counts;
   std::vector<int> K_dets_per_config;
   std::vector<Det> S_paths;

   int KS_max_path_count;

   MArr2R _KS_H;
   real* KS_H_data;

   ArenaAllocator allocator;
};

#define SOLVER_H
#endif
