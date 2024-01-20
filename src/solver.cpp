
std::span<Det> StrideItr::operator*() const 
{ 
   return std::span<Det>(data.begin() + strided_idx, (*strides)[idx]); 
}

StrideItr& StrideItr::operator++()
{ 
   strided_idx += (*strides)[idx];
   idx++;
   return *this; 
} 

StrideItr StrideItr::operator++(int)
{ 
   StrideItr temp = *this; 
   ++(*this);
   return temp; 
}


SCFItr::SCFItr(int cidx, int pidx, int idx, int first_coeff_idx, const KSBlockIterator& itr) : 
      config_idx(cidx), S_path_idx(pidx), idx(idx), first_coeff_idx(first_coeff_idx), block_itr(itr) 
{
}

SCFItr& SCFItr::operator++()
{ 
   first_coeff_idx += block_itr.SCF_size(*this); 
   if(++S_path_idx >= block_itr.KS_S_path_counts[config_idx])
   {
      config_idx++;
      S_path_idx = 0;
   }
   idx++; 

   return *this; 
} 

SCFItr SCFItr::operator++(int)
{ 
   SCFItr temp = *this; 
   ++(*this);
   return temp; 
}


void KSBlockIterator::init()
{
   KBasis kbasis = form_K_basis(params);
   sort_K_basis(kbasis, params);
   basis = std::move(kbasis.basis);
   K_block_sizes = std::move(kbasis.block_sizes);
   K_block_begin_indices = std::vector<int>(K_block_sizes.size() + 1);
   K_block_begin_indices[0] = 0;
   std::partial_sum(K_block_sizes.cbegin(), K_block_sizes.cend(), K_block_begin_indices.begin() + 1);

   m = params.m;
   S_min = std::abs(m);
   S_max = 0.5*params.N;
   S_count = (S_max - S_min) + 1;

   int max_KS_dim = 0;
   for(int i = 0; i < S_count; i++)
   {
      real cur_S = S_min + i;
      int cur_dim = SM_space_dim(params.N, params.Ns, cur_S);
      if(max_KS_dim < cur_dim) { max_KS_dim = cur_dim; }
   }
   max_KS_dim /= params.Ns;
   KS_H_data = allocator.allocate<real>(max_KS_dim);

   int max_configs_per_K_block = 0;
   int block_begin = 0;
   for(int Kidx = 0; Kidx < K_count; Kidx++)
   {
      int cur_count = 0;
      int block_end = block_begin + K_block_sizes[Kidx];
      for(int i = block_begin; i < block_end; i += dets_per_orbital_config(count_singles(basis[i], params), params), cur_count++) {}
      if(max_configs_per_K_block < cur_count) {max_configs_per_K_block = cur_count;}

      block_begin = block_end;
   }
   // TODO: Allocate on memory arena
   K_single_counts = std::vector<int>(max_configs_per_K_block);
   K_dets_per_config = std::vector<int>(max_configs_per_K_block);

   reset();
}

void KSBlockIterator::form_KS_subbasis()
{
   KS_max_path_count = 0;

   int cidx = 0;
   for(auto config : K_configs()) 
   {
      int s_k = K_single_counts[cidx];
      if(s_k > 0 && S <= 0.5*s_k)
      {
         int S_state_count = CSV_dim(S, s_k);
         if(S_state_count > 0)
         {
            KS_configs.push_back(config);
            KS_single_counts.push_back(s_k);
            KS_S_path_counts.push_back(S_state_count);
#ifdef HUBBARD_TEST
            KS_spins.push_back(S);
#endif

            if(S_state_count > KS_max_path_count)
            {
               KS_max_path_count = S_state_count;
            }
         }
      }
      else if(S == 0)
      {
         // Handle singlets/closed-shell configs
         assert(m == 0);
         KS_configs.push_back(config);
         KS_single_counts.push_back(0);
         KS_S_path_counts.push_back(1);
#ifdef HUBBARD_TEST
         KS_spins.push_back(S);
#endif

         if(KS_max_path_count < 1)
         {
            KS_max_path_count = 1;
         }
      }

      cidx++;
   }
}

void KSBlockIterator::form_SCFs()
{
   int cidx = 0;
   int prev_s_k = -1;
   for(auto config : KS_configs)
   {
      int s_k = KS_single_counts[cidx];
      if(s_k > 0)
      {
         if(prev_s_k != s_k)
         {
            // TODO: Allocate on memory arena (temporary)
            S_paths.resize(0);
            form_S_paths(1, 1, 0.5, s_k, S, S_paths);
            assert(S_paths.size() == KS_S_path_counts[cidx]);

            prev_s_k = s_k;
         }

         for(auto S_path : S_paths)
         {
            for(auto det : config)
            {
               auto [M_path, M_path_sign] = det2path(det, params);
               KS_SCF_coeffs.push_back(M_path_sign*compute_SCF_overlap(S_path, M_path, s_k, S, m));
            }
         }
      }
      else
      {
         assert(KS_configs[cidx].size() == 1);
         KS_SCF_coeffs.push_back(1);
      }
      cidx++;
   }
}

KSBlockIterator& KSBlockIterator::operator++()
{
   has_blocks_left = next_S_block();
   if(!has_blocks_left)
   {
      has_blocks_left = next_K_block();
      //if(has_blocks_left) { init_S_block(0); }
      init_S_block(0);

      assert(has_blocks_left || (total_SCF_count == basis.size()));
   }

   return *this;
}

void KSBlockIterator::reset()
{
   total_SCF_count = 0;   // NOTE: For debug only
   K_block_SCF_count = 0; // NOTE: For debug only
   KS_dim = 0;
   has_blocks_left = true;
   init_K_block(0);
   init_S_block(0);
}

bool KSBlockIterator::next_K_block()
{
   assert(K_block_SCF_count == K_block_sizes[K_block_idx]);
   total_SCF_count += K_block_SCF_count; // NOTE: For debug only

   int new_idx = K_block_idx + 1;
   if(new_idx < K_count)
   {
      init_K_block(new_idx);
      return true;
   }

   return false;
}

void KSBlockIterator::init_K_block(int idx)
{
   K_block_SCF_count = 0; // NOTE: For debug only
   K_block_idx = idx;
   K_KS_basis_begin_idx = K_block_begin_indices[K_block_idx];

   int det_idx = K_KS_basis_begin_idx;
   for(int cidx = 0; det_idx < (K_KS_basis_begin_idx + K_block_sizes[K_block_idx]); cidx++)
   {
      Det det = basis[det_idx];
      int single_count = count_singles(det, params);
      int det_count = dets_per_orbital_config(single_count, params);
      K_single_counts[cidx] = single_count;
      K_dets_per_config[cidx] = det_count;
      det_idx += det_count;
   }
}

bool KSBlockIterator::next_S_block()
{
   int new_idx = S_block_idx + 1;
   if(new_idx < S_count)
   {
      init_S_block(new_idx);
      return true;
   }

   return false;
}

void KSBlockIterator::init_S_block(int idx)
{
   S_block_idx = idx;
   S = S_min + S_block_idx;

   KS_configs.resize(0);
   KS_S_path_counts.resize(0);
   KS_single_counts.resize(0);
#ifdef HUBBARD_TEST
   KS_spins.resize(0);
#endif

   KS_max_path_count = 0;

   form_KS_subbasis();
   sort_multiple(KS_single_counts, KS_configs, KS_S_path_counts
#ifdef HUBBARD_TEST
                 , KS_spins
#endif
                );

   S_paths.reserve(KS_max_path_count);
   KS_dim = std::reduce(KS_S_path_counts.begin(), KS_S_path_counts.end());
   assert(KS_dim >= 0);
   K_block_SCF_count += KS_dim; // NOTE: For debug only

   new (&_KS_H)  MArr2R(KS_H_data, KS_dim, KS_dim);
   _KS_H.setZero();

   KS_SCF_coeffs.resize(0);
   form_SCFs();
}

StrideItr KSBlockIterator::K_configs()
{
   return StrideItr(std::span<Det>(basis.begin() + K_KS_basis_begin_idx, K_block_sizes[K_block_idx]), &K_dets_per_config);
}

KSBlockIterator::SCFs KSBlockIterator::KS_basis(SCFItr first)
{
   return SCFs{.start_cidx = first.config_idx, .start_pidx = first.S_path_idx, .start_idx = first.idx,
               .start_first_coeff_idx = first.first_coeff_idx, .block_itr = *this, .KS_dim = KS_dim};
}

KSBlockIterator::SCFs KSBlockIterator::KS_basis()
{
   return SCFs{.start_cidx = 0, .start_pidx = 0, .start_idx = 0, .start_first_coeff_idx = 0,
               .block_itr = *this, .KS_dim = KS_dim};
}


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

real KSM_basis_compute_E0(HubbardComputeDevice& cdev, KSBlockIterator& itr)
{
   real min_E = std::numeric_limits<real>::max();

   for(itr.reset(); itr; ++itr) 
   {
      if(itr.KS_dim == 0) { continue; }

      // H_int
      for(auto scf1 : itr.KS_basis())
      {
         // NOTE: The diaognal of H_KS has to be zero before starting to compute H_int since the diagonal is multiplied by 0.5 due to double counting in this loop!
         for(auto scf2 : itr.KS_basis(scf1))
         {
            real cur_elem = cdev.H_int_element(itr.SCF_dets(scf1), itr.SCF_coeffs(scf1), itr.SCF_size(scf1), 
                                               itr.SCF_dets(scf2), itr.SCF_coeffs(scf2), itr.SCF_size(scf2),
                                               itr.params);
            itr.KS_H(scf1, scf2) += cur_elem;
            itr.KS_H(scf2, scf1) += cur_elem;
         }
         itr.KS_H(scf1, scf1) *= real(0.5);
      }

      // H_0
      int col_idx = 0;
      int cidx = 0;
      for(auto config : itr.KS_configs)
      {
         Det det = get_config_ref_det(config);
         int state_count = itr.KS_S_path_counts[cidx++];

         real cur_H_0 = 0;
         for(int k = 0; k < itr.K_count; k++)
         {
            cur_H_0 += noninteracting_E(k + 1, itr.params.T, itr.params.Ns, BCS::PERIODIC)*(count_state(det, k) + count_state(det, itr.params.Ns + k));
         }

         itr.KS_H().matrix().diagonal()(Eigen::seqN(col_idx, state_count)).array() += cur_H_0;
         col_idx += state_count;
      }

      real E0;
      if(itr.KS_dim == 1)
      {
         E0 = itr.KS_H()(0, 0);
      }
      else
      {
         E0 = cdev.sym_eigs_smallest(itr.KS_H().data(), itr.KS_dim);
      }

      if(E0 < min_E)
      {
         min_E = E0;
      }
   }

   return min_E;
}

real KSM_basis_compute_E0(HubbardComputeDevice& cdev, ArenaAllocator allocator, const HubbardParams& params)
{
   KSBlockIterator block_itr(params, allocator);
   return KSM_basis_compute_E0(cdev, block_itr);
}
