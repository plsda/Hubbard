#ifndef BASIS_H

struct HubbardParams
{
   /*
      Stores the parameters defining a Hubbard model and provides related utilities.

      Parameters
      ----------
      real T     : Hopping energy
      real U     : Interaction strength
      int Ns     : Site count
      int N      : Particle count
      int N_up   : Number of spin-up particles
      int N_down : Number of spin-down particles
   */

   real T{};
   real U{};
   int Ns{};
   int N{};
   int N_up{};
   int N_down{};

   HubbardParams() = default;

   explicit HubbardParams(real T, real U, int Ns, int N_up, int N_down) :
      T(T), U(U), Ns(Ns), N(N_up + N_down), N_up(N_up), N_down(N_down)
   {
      //assert(Ns > 0 && N > 0 && N_up >= 0 && N_down >= 0 &&
      //       N_up <= Ns && N_down <= Ns);

      assert(Ns > 0);
      assert(N > 0);
      assert(N_up >= 0);
      assert(N_down >= 0);
      assert(N_up <= Ns);
      assert(N_down <= Ns);
   }

   constexpr explicit HubbardParams(real T, real U, int Ns, int N_up, int N_down, int) :
      T(T), U(U), Ns(Ns), N(N_up + N_down), N_up(N_up), N_down(N_down) {}

   int basis_size() const;
   void set_half_filling(int new_N_up);

};

struct StructuralHubbardParams : public HubbardParams
{
   constexpr explicit StructuralHubbardParams() {}
   constexpr explicit StructuralHubbardParams(real T, real U, int Ns, int N_up, int N_down) : HubbardParams(T, U, Ns, N_up, N_down, 1) {}
};

struct SDet 
{
   Det det;
   int sign = 1;
};

struct WeightedDet
{
   Det det;
   real coeff;

   real operator*(WeightedDet other)
   {
      return coeff*other.coeff*(det == other.det);
   }
};

struct KBasis
{
   std::vector<Det> basis;
   std::vector<int> momenta;
   std::vector<int> block_sizes;
};

struct KConfigs
{
   std::vector<std::vector<std::shared_ptr<std::vector<Det>>>> configs;
   std::vector<int> block_sizes;
};

struct Det_xor_and
{
   Det det_xor;
   Det det_and;
};

u32 count_state(Det det, u32 state_idx);
u32 get_pop(Det det);
u32 count_higher(u32 state_idx, Det det);
u32 count_lower(u32 state_idx, Det det);
u32 get_up_pop(Det det, u32 Ns);
u32 get_down_pop(Det det, u32 Ns);
SDet sadd(u32 state_idx, Det det, int sign = 1);
SDet sadd(u32 state_idx, SDet det);
SDet ssub(u32 state_idx, Det det, int sign = 1);
SDet ssub(u32 state_idx, SDet det);
std::pair<Det, Det> get_det_up_down(Det det, HubbardParams params);
int count_singles(Det det, const HubbardParams& params);
real compute_interaction_element(Det ket, const HubbardParams& params);
template <class T> Det statelist2det(const T& statelist);
void list_spinless_determinants(std::vector<Det>& result, u32 N_particles, u32 N_states);
void list_determinants(std::vector<Det>& result, const HubbardParams& params);
int state_momentum(Det det, const HubbardParams& params);
void det2spinless_statelist(Det det, const HubbardParams& params, std::vector<int>& result);
int CSV_dim(real spin, int single_count);
Det det_config_ID(Det det, const HubbardParams& params);
Det_xor_and get_det_xor_and(Det det, const HubbardParams& params);
bool cmp_det_config(Det det1, Det det2, HubbardParams params);
real kidx2k(int k, int Ns) ;
KBasis form_K_basis(const HubbardParams& params);
void form_S_paths(Det path, int cur_s, real cur_f, int s, real f, std::vector<Det>& result);
SDet det2path(Det det, const HubbardParams& params);
int get_path_edge(Det path, size_t idx);
real compute_SCF_overlap(Det S_path, Det M_path, int edge_count, real f, real m);
KConfigs get_k_orbitals(const HubbardParams& params);
void form_KS_subbasis(real f, real m,
                      const std::vector<std::shared_ptr<std::vector<Det>>>& K_configs,
                      const std::vector<int>& single_counts,
                      std::vector<std::shared_ptr<std::vector<Det>>>& Kf_basis,
                      std::vector<int>& Kf_single_counts,
                      std::vector<int>& Kf_counts,
                      std::vector<real>& Kf_spins,
                      int& max_config_count,
                      int& max_path_count);
void form_KS_subbasis(real f, real m,
                      const std::vector<std::shared_ptr<std::vector<Det>>>& K_configs,
                      const std::vector<int>& single_counts,
                      std::vector<std::shared_ptr<std::vector<Det>>>& Kf_basis,
                      std::vector<int>& Kf_single_counts,
                      std::vector<int>& Kf_counts,
                      int& max_config_count,
                      int& max_path_count);
/*
void form_SCFs(real f, real m,
               std::span<std::shared_ptr<std::vector<Det>>> Kf_basis,
               std::span<int> single_counts,
               std::span<int> S_path_counts,
               std::vector<Det>& S_paths,
               const HubbardParams& params,
               std::vector<real>& result);
real SCF_spin(const std::vector<Det>& dets, std::span<real> coeffs, const HubbardParams& params);
real SCF_inner(const std::vector<Det>& bra_dets, std::span<real> bra_coeffs, 
               const std::vector<Det>& ket_dets, std::span<real> ket_coeffs);
*/




#define BASIS_H
#endif
