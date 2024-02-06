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
   real m{};

   HubbardParams() = default;

   HubbardParams(real T, real U, int Ns, int N_up, int N_down) :
   //explicit HubbardParams(real T, real U, int Ns, int N_up, int N_down) :
      T(T), U(U), Ns(Ns), N(N_up + N_down), N_up(N_up), N_down(N_down), m(0.5*(N_up - N_down))
   {
      assert(Ns > 0 && N > 0 && N_up >= 0 && N_down >= 0 &&
             N_up <= Ns && N_down <= Ns);
   }

   constexpr explicit HubbardParams(real T, real U, int Ns, int N_up, int N_down, int) :
      T(T), U(U), Ns(Ns), N(N_up + N_down), N_up(N_up), N_down(N_down), m(0.5*(N_up - N_down)) {}

   int basis_size() const;
   void set_half_filling(int new_N_up);

   real S_min() const { return std::abs(m); }
   real S_max() const { return 0.5*N; }
   int S_count() const { return (S_max() - S_min()) + 1; }
   int K_count() const { return Ns; }
   int KS_block_count() const { return S_count()*K_count(); }

   bool operator==(const HubbardParams& other)
   { 
      return 
         (T     == other.T) &&
         (U     == other.U) &&
         (Ns    == other.Ns) &&
         (N     == other.N) &&
         (N_up  == other.N_up) &&
         (N_down== other.N_down) &&
         (m     == other.m);
   }
   bool operator!=(const HubbardParams& other) { return !(*this == other); }
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

bool is_halffilling(const HubbardParams& params);
std::pair<Det, Det> get_det_up_down(Det det, HubbardParams params);
int count_singles(Det det, const HubbardParams& params);
template <class T> Det statelist2det(const T& statelist);
void list_spinless_determinants(std::vector<Det>& result, u32 N_particles, u32 N_states);
template<class A = std::allocator<Det>> void list_determinants(std::vector<Det, A>& result, const HubbardParams& params);
int state_momentum(Det det, const HubbardParams& params);
void det2spinless_statelist(Det det, const HubbardParams& params, std::vector<int>& result);

int CSV_dim(real spin, int single_count);
int SM_space_dim(int N, int Ns, real S);
int dets_per_orbital_config(int config_single_count, const HubbardParams& params);
int config_count(const HubbardParams& params);

Det det_config_ID(Det det, const HubbardParams& params);
Det_xor_and get_det_xor_and(Det det, const HubbardParams& params);
bool cmp_det_config(Det det1, Det det2, HubbardParams params);
real kidx2k(int k, int Ns) ;

KBasis form_K_basis(const HubbardParams& params);
template<class A1, class A2>
void form_K_basis(std::vector<Det, A1>& basis, std::vector<int, A2>& block_sizes,
                  const HubbardParams& params);
template<class A1 = std::allocator<Det>, class A2 = std::allocator<int>>
void form_K_basis(std::vector<Det, A1>& basis, std::vector<int>& momenta, std::vector<int, A2>& block_sizes,
                  const HubbardParams& params);
int dets_per_orbital_config(int config_single_count, const HubbardParams& params);
void sort_K_basis(KBasis& kbasis, const HubbardParams& params);
template<class A1 = std::allocator<Det>, class A2 = std::allocator<int>>
void sort_K_basis(std::vector<Det, A1>& basis, const std::vector<int, A2>& block_sizes, const HubbardParams& params);

void form_S_paths(Det path, int cur_s, real cur_f, int s, real f, std::vector<Det, DetArena>& result);
SDet det2path(Det det, const HubbardParams& params);
int get_path_edge(Det path, size_t idx);
real compute_CSF_overlap(Det S_path, Det M_path, int edge_count, real f, real m);

#define BASIS_H
#endif
