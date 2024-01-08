
// Per-particle energy
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

real noninteracting_E0(const HubbardParams& params, BCS bcs)
{
   int T = params.T;
   int Ns = params.Ns;
   int N_up = params.N_up;
   int N_down = params.N_down;

   std::vector<real> E_U0_pp(Ns);
   for(int i = 0; i < Ns; i++)
   {
      E_U0_pp[i] = noninteracting_E(i + 1, T, Ns, bcs);
   }

   std::sort(E_U0_pp.begin(), E_U0_pp.end());
   int half_Ns = Ns/2;
   int min_N = std::min(N_up, N_down);
   real result = 2.0*std::reduce(E_U0_pp.begin(), E_U0_pp.begin() + min_N) + 
                     std::reduce(E_U0_pp.begin() + min_N, E_U0_pp.begin() + min_N + std::abs(N_up - N_down));

   return result;
}
   
real dimer_E0(const HubbardParams& params, BCS bcs)
{
   real result = 0;
   real T = params.T;
   real U = params.U;

   if(bcs == BCS::PERIODIC)
   {
      T *= 2.0;
   }
   else if(bcs != BCS::OPEN)
   {
      assert(!"Supplied BCS not supported (dimer_E0)!");
   }

   switch(params.N)
   {
      case 0: { result = 0; } break;
      case 1: { result = std::min(-T, T); } break;
      case 2: 
      { 
         if(params.N_up == 0 || params.N_down == 0)
         {
            result = 0;
         }
         else
         {
            result = std::min({real(0), U, real(0.5*(U - std::sqrt(16.0*T*T + U*U)))}); 
         }
      } break;
      case 3: { result = std::min(U - T, U + T); } break;
      case 4: { result = 2.0*U; } break;
      default: { assert(!"N > 2*Ns"); }
   }

   return result;
}
    
// Ground state energy in the atomic limit (T = 0)
real atomic_E0(const HubbardParams& params)
{
   real result = 0;
   real U = params.U;

   if(U < 0)
   {
      result = std::min(params.N_up, params.N_down);
   }
   else if(U > 0)
   {
      result = std::max(0, params.N - params.Ns);
   }

   result *= params.U;

   return result;
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

real kfm_basis_compute_E0(Hubbard_compute_device& cdev, const KConfigs& configs, const HubbardParams& params)
{
   u32 k_count = params.Ns;
   real m = 0.5*(params.N_up - params.N_down);
   real f_min = std::abs(m);
   real f_max = 0.5*params.N;
   int f_count = (f_max - f_min) + 1;

   int total_SCF_count = 0; // NOTE: For debugging
   real min_E = std::numeric_limits<real>::max();

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

      if(K_configs.size() == 0) 
      {
         continue;
      }

      std::for_each(STD_POLICY_UNSEQ RangeItr(size_t(0)), RangeItr(K_configs.size()),
                    [K_configs, &single_counts, &params](size_t i){single_counts[i] = count_singles(K_configs[i]->front(), params);});

      int K_block_SCF_count = 0;

      for(int fidx = 0; fidx < f_count; fidx++)
      {
         real f = f_min + fidx;

         // TODO: Preallocate
         std::vector<std::shared_ptr<std::vector<Det>>> Kf_basis;
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
                      std::span<std::shared_ptr<std::vector<Det>>>(Kf_basis), std::span<int>(Kf_single_counts), std::span<int>(Kf_counts),
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
                  Arr3R ket_coeffs = SCFs.slice(offsets, extents);

                  const real* const ket_coeffs_data = ket_coeffs.data();
                  // Diagonal
                  real cur_elem = cdev.H_int_element(ket_dets.data(), ket_coeffs_data, ket_dets.size(), 
                                                     ket_dets.data(), ket_coeffs_data, ket_dets.size(), 
                                                     params);
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
                        Arr3R bra_coeffs = SCFs.slice(offsets, extents);

                        const real* const bra_coeffs_data = bra_coeffs.data();
                        cur_elem = cdev.H_int_element(bra_dets.data(), bra_coeffs_data, bra_dets.size(), 
                                                      ket_dets.data(), ket_coeffs_data, ket_dets.size(), 
                                                      params);
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
                        Arr3R bra_coeffs = SCFs.slice(offsets, extents);

                        const real* const bra_coeffs_data = bra_coeffs.data();
                        cur_elem = cdev.H_int_element(bra_dets.data(), bra_coeffs_data, bra_dets.size(), 
                                                      ket_dets.data(), ket_coeffs_data, ket_dets.size(), 
                                                      params);
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
               E0 = cdev.sym_eigs_smallest(H_Kf.data(), Kf_dim);
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

real kfm_basis_compute_E0(Hubbard_compute_device& cdev, const HubbardParams& params)
{
   KConfigs configs = get_k_orbitals(params);
   return kfm_basis_compute_E0(cdev, configs, params);
}

