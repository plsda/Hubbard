#include "hubbard.h"
#include "utils.cpp"
#include "basis.cpp"

real noninteracting_E(int n, real T, int Ns, BCS bcs) // n = 1, 2, ..., Ns
{
   if(bcs == BCS::OPEN)
   {
      return -2.0*T*std::cos(PI*n/(Ns + 1.0));
   }
   else if(bcs == BCS::PERIODIC)
   {
      return -2.0*T*std::cos(2.0*PI*n/Ns);
   }

   assert(!"Supplied BCS not supported (noninteracting_E)!");
   return 0;
}
    
real noninteracting_wf(int s, int k, int Ns)
{
   return std::sqrt(2.0/(Ns + 1.0))*std::sin(PI*k*s/(Ns + 1.0));
}

// NOTE: This is an asymptotic result
real halffilled_E_per_N(real T, real U, IntArgs int_args)
{
   real y = -U/(4.0*T);

   auto integrand = [y](real x) 
   {
      real J0 = std::cyl_bessel_j(0, x);
      real J1 = std::cyl_bessel_j(1, x);

      real exp_arg = -2.0*y*x;
      int intpart = int(exp_arg);
      real ipow = (intpart >= 0) ? 1.0/(1 << intpart) : (1 << std::abs(intpart));
      // TODO: Could take log, break the integral into two parts so that in one part can approximate log(1 + exp(x))
      //       as a linear function (when x is large), and in the other part can compute as is since x is small.
      //       Finally, take exp and integrate.

      return J0*J1*ipow/(x*(ipow + std::exp(exp_arg - intpart*LOG2)));
   };

   real integ = quad(integrand, int_args);

   return -4.0*T*integ;
}


real kfm_basis_compute_E0(const KConfigs& configs, const HubbardParams& params)
{
   u32 k_count = params.Ns;
   real m = 0.5*(params.N_up - params.N_down);
   real f_min = std::abs(m);
   real f_max = 0.5*params.N;
   int f_count = (f_max - f_min) + 1;

   int total_SCF_count = 0; // NOTE: For debugging
   real min_E = 0;

   size_t configs_max_state_count = 0;
   for(size_t i = 0; i < k_count; i++)
   {
      size_t cur_state_count = configs.configs[i].size();
      if(configs_max_state_count < cur_state_count)
      {
         configs_max_state_count = cur_state_count;
      }
   }
   std::vector<int> single_counts(configs_max_state_count);

   for(size_t K_block_idx = 0; K_block_idx < k_count; K_block_idx++)
   {
      const auto& K_configs = configs.configs[K_block_idx];

      std::for_each(STD_POLICY_UNSEQ RangeItr(size_t(0)), RangeItr(K_configs.size()),
                    [&, K_configs, params](size_t i){single_counts[i] = count_singles(K_configs[i][0], params);});

      int K_block_SCF_count = 0;

      for(int fidx = 0; fidx < f_count; fidx++)
      {
         real f = f_min + fidx;

         // TODO: Preallocate
         std::vector<std::vector<Det>*> Kf_basis;
         std::vector<int> Kf_counts;
         std::vector<int> Kf_single_counts;
         int max_config_count = 0;
         int max_path_count = 0;

         form_KS_subbasis(f, m, K_configs, single_counts,
                          Kf_basis, Kf_single_counts, Kf_counts,
                          max_config_count, max_path_count);


         // TODO: Preallocate (compute max path count over all f's) 
         std::vector<Det> S_paths;
         S_paths.reserve(max_path_count);
         int Kf_dim = std::reduce(Kf_counts.begin(), Kf_counts.end());
         if(Kf_dim > 0)
         {
            // TODO: Preallocate
            Arr2R H_Kf = Arr2R::Zero(Kf_dim, Kf_dim);
            Arr3R SCFs(Kf_basis.size(), max_path_count, max_config_count);
            SCFs.setZero();

            form_SCFs(f, m,
                      std::span<std::vector<Det>*>(Kf_basis), std::span<int>(Kf_single_counts), std::span<int>(Kf_counts),
                      S_paths, params, SCFs);

            // H_0
            int col_idx = 0;
            for(int kidx = 0; kidx < Kf_basis.size(); kidx++)
            {
               Det ket = Kf_basis[kidx]->front();
               int ket_count = Kf_counts[kidx];

               real cur_H_0 = 0;
               for(int k = 0; k < params.Ns; k++)
               {
                  cur_H_0 += noninteracting_E(k + 1, params.T, params.Ns, BCS::PERIODIC)*(count_state(ket, k) + count_state(ket, params.Ns + k));
               }

               H_Kf.matrix().diagonal()(Eigen::seqN(col_idx, ket_count)).array() += cur_H_0;

               col_idx += ket_count;
            }

            // H_int
            col_idx = 0;
            for(int ket_k_idx = 0; ket_k_idx < Kf_basis.size(); ket_k_idx++)
            {
               const std::vector<Det>& ket_dets = *Kf_basis[ket_k_idx];
               for(int ket_path_idx = 0; ket_path_idx < Kf_counts[ket_k_idx]; ket_path_idx++)
               {
                  Eigen::array<Eigen::Index, 3> offsets = {ket_k_idx, ket_path_idx, 0};
                  Eigen::array<Eigen::Index, 3> extents = {1, 1, max_config_count};
                  Eigen::TensorRef<Arr3R> ket_coeffs = SCFs.slice(offsets, extents);

                  // Diagonal
                  real cur_elem = compute_H_int_element(ket_dets, ket_coeffs, ket_dets, ket_coeffs, params);
                  H_Kf(col_idx, col_idx) += cur_elem;

                  // Off-diagonals
                  int row_idx = col_idx + 1;
                  {
                     int bra_k_idx = ket_k_idx;
                     const std::vector<Det>& bra_dets = *Kf_basis[bra_k_idx];

                     for(int bra_path_idx = ket_path_idx + 1; bra_path_idx < Kf_counts[bra_k_idx]; bra_path_idx++)
                     {
                        offsets = {bra_k_idx, bra_path_idx, 0};
                        extents = {1, 1, max_config_count};
                        Eigen::TensorRef<Arr3R> bra_coeffs = SCFs.slice(offsets, extents);

                        cur_elem = compute_H_int_element(bra_dets, bra_coeffs, ket_dets, ket_coeffs, params);
                        H_Kf(row_idx, col_idx) += cur_elem;
                        H_Kf(col_idx, row_idx) += cur_elem;

                        row_idx += 1;
                     }
                  }

                  for(int bra_k_idx = ket_k_idx + 1; bra_k_idx < Kf_basis.size(); bra_k_idx++)
                  {
                     const std::vector<Det>& bra_dets = *Kf_basis[bra_k_idx];
                     for(int bra_path_idx = 0; bra_path_idx < Kf_counts[bra_k_idx]; bra_path_idx++)
                     {
                        offsets = {bra_k_idx, bra_path_idx, 0};
                        extents = {1, 1, max_config_count};
                        Eigen::TensorRef<Arr3R> bra_coeffs = SCFs.slice(offsets, extents);

                        cur_elem = compute_H_int_element(bra_dets, bra_coeffs, ket_dets, ket_coeffs, params);
                        H_Kf(row_idx, col_idx) += cur_elem;
                        H_Kf(col_idx, row_idx) += cur_elem;

                        row_idx += 1;
                     }
                  }


                  col_idx += 1;
               }
            }

            assert(H_Kf.matrix().isApprox(H_Kf.matrix().transpose())); // NOTE: H_Kf is a real matrix

            real E0;
            if(Kf_dim == 1)
            {
               E0 = H_Kf(0, 0);
            }
            else
            {
               // TODO: Use Spectra
               Eigen::SelfAdjointEigenSolver<MatR> eigensolver(H_Kf.matrix());
               E0 = eigensolver.eigenvalues()[0];
            }

            if(E0 < min_E)
            {
               min_E = E0;
            }

            K_block_SCF_count += Kf_dim;
            total_SCF_count += Kf_dim;
         }

      }

      assert(K_block_SCF_count == configs.block_sizes[K_block_idx]);

   }

   assert(total_SCF_count == params.basis_size());

   return min_E;
}

real kfm_basis_compute_E0(const HubbardParams& params)
{
   KConfigs configs = get_k_orbitals(params);
   return kfm_basis_compute_E0(configs, params);
}


int main()
{

   real T = 1;
   real U = real(2.7);
   u32 Ns = 5;
   u32 N_up = 2;
   u32 N_down = 2;

   HubbardParams params(T, U, Ns, N_up, N_down);

   KConfigs configs = get_k_orbitals(params);

   real E0 = kfm_basis_compute_E0(configs, params);

   std::cout << "N = " << N_up + N_down
             << ", Ns = " << Ns
             << ", T = " << T
             << ", U = " << U
             << ", N_up = " << N_up
             << ", N_down = " << N_down << std::endl;
   std::cout << "E0/Ns = " << E0/Ns << std::endl;

   return 0;
}
