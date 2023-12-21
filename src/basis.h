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
      //T(T), U(U), Ns(Ns), N_up(N_up), N_down(N_down), N(N_up + N_down)
      T(T), U(U), Ns(Ns), N(N_up + N_down), N_up(N_up), N_down(N_down)
   {
      assert(Ns > 0 && N > 0 && N_up >= 0 && N_down >= 0);
   }

   constexpr explicit HubbardParams(real T, real U, int Ns, int N_up, int N_down, int) :
      //T(T), U(U), Ns(Ns), N_up(N_up), N_down(N_down), N(N_up + N_down) 
      T(T), U(U), Ns(Ns), N(N_up + N_down), N_up(N_up), N_down(N_down)
   {}

   // NOTE: Result may overflow for systems larger than 16 sites (but such systems are currently not supported)
   int basis_size() const
   {
      return choose(Ns, N_up)*choose(Ns, N_down);
   }

   void set_half_filling(int new_N_up)
   {
      assert(new_N_up > 0);

      Ns = 2*new_N_up;
      N = 2*new_N_up;
      N_up = new_N_up;
      N_down = new_N_up;
   }

};

struct StructuralHubbardParams : public HubbardParams
{
   constexpr explicit StructuralHubbardParams() {}
   constexpr explicit StructuralHubbardParams(real T, real U, int Ns, int N_up, int N_down) : HubbardParams(T, U, Ns, N_up, N_down, 1) {}
};

std::ostream& operator<<(std::ostream& os, const HubbardParams& params)
{
   os << "{T=" << params.T
      << ", U=" << params.U
      << ", Ns=" << params.Ns
      << ", N_up=" << params.N_up
      << ", N_dn=" << params.N_down << "}";

   return os;
}

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
   std::vector<std::vector<std::vector<Det>>> configs;
   std::vector<int> block_sizes;
};

struct Det_xor_and
{
   Det det_xor;
   Det det_and;
};

void det2spinless_statelist(Det det, const HubbardParams& params, std::vector<int>& result);
real SCF_spin(const std::vector<Det>& dets, std::span<real> coeffs, const HubbardParams& params);
real SCF_inner(const std::vector<Det>& bra_dets, std::span<real> bra_coeffs, 
               const std::vector<Det>& ket_dets, std::span<real> ket_coeffs);

#define BASIS_H
#endif
