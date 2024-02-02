
HubbardSizes operator*(const HubbardSizes& sz, int n)
{
   return HubbardSizes{
      .basis_size                        = n*sz.basis_size,
      .min_singles                       = n*sz.min_singles,
      .max_singles                       = n*sz.max_singles,
      .config_count                      = n*sz.config_count,
      .K_block_config_count_upper_bound  = n*sz.K_block_config_count_upper_bound,
      .KS_block_config_count_upper_bound = n*sz.KS_block_config_count_upper_bound,
      .max_KS_dim                        = n*sz.max_KS_dim,
      .max_dets_in_config                = n*sz.max_dets_in_config,
      .max_S_paths                       = n*sz.max_S_paths,
      .CSF_coeff_count_upper_bound       = n*sz.CSF_coeff_count_upper_bound,
      .alloc_pad                         =   sz.alloc_pad,
      .unaligned_workspace_size          = n*sz.unaligned_workspace_size,
      .workspace_size                    = n*sz.workspace_size
   };
}
HubbardSizes operator*(int n, const HubbardSizes& sz) { return sz*n; }

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


CSFItr::CSFItr(int cidx, int pidx, int idx, int first_coeff_idx, const KSBlockIterator& itr) : 
      config_idx(cidx), S_path_idx(pidx), idx(idx), first_coeff_idx(first_coeff_idx), block_itr(itr) 
{
}

CSFItr& CSFItr::operator++()
{ 
   first_coeff_idx += block_itr.CSF_size(*this); 
   if(++S_path_idx >= block_itr.KS_S_path_counts[config_idx])
   {
      config_idx++;
      S_path_idx = 0;
   }
   idx++; 

   return *this; 
} 

CSFItr CSFItr::operator++(int)
{ 
   CSFItr temp = *this; 
   ++(*this);
   return temp; 
}

KSBlockIterator::KSBlockIterator(ArenaAllocator& _allocator) : 
   allocator(&_allocator), 
cpt(_allocator.begin_provision()),
   params({}),
   K_count(0),
   S_count(0),
   KS_dim(0),
   m(0),
   S_min(0),
   S_max(0),
   total_CSF_count(0),
   K_block_CSF_count(0),
   K_block_idx(0), 
   S_block_idx(0),
   K_KS_basis_begin_idx(0),
   has_blocks_left(false),
   S(0),
   KS_max_path_count(0),
   KS_H_data(0),
   basis(                0, DetArena(allocator)),
   K_block_sizes(        0, IntArena(allocator)),
   K_block_begin_indices(0, IntArena(allocator)),
   K_single_counts(      0, IntArena(allocator)),
   K_dets_per_config(    0, IntArena(allocator)),
   S_paths(              0, DetArena(allocator)),
   KS_configs(           0, SpanArena(allocator)),
   KS_single_counts(     0, IntArena(allocator)),
   KS_S_path_counts(     0, IntArena(allocator)),
   KS_CSF_coeffs(        0, RealArena(allocator)),
#ifdef HUBBARD_TEST
   KS_spins(             0, RealArena(allocator)),
#endif
   _KS_H(MArr2R(NULL, 0, 0))
{ 
   allocator->end_provision(cpt);
}

KSBlockIterator::KSBlockIterator(HubbardParams _params, ArenaAllocator& _allocator, HubbardSizes sz) :
   allocator(&_allocator), 
cpt(_allocator, sz.workspace_size),
   params(_params),
   K_count(_params.Ns), 
   basis(                sz.basis_size,                        DetArena(allocator,  sz.basis_size,                        sz.alloc_pad)),
   K_block_sizes(        _params.Ns,                           IntArena(allocator,  _params.Ns,                           sz.alloc_pad)),
   K_block_begin_indices(_params.Ns + 1,                       IntArena(allocator,  _params.Ns + 1,                       sz.alloc_pad)),
   K_single_counts(      sz.K_block_config_count_upper_bound,  IntArena(allocator,  sz.K_block_config_count_upper_bound,  sz.alloc_pad)),
   K_dets_per_config(    sz.K_block_config_count_upper_bound,  IntArena(allocator,  sz.K_block_config_count_upper_bound,  sz.alloc_pad)),
   S_paths(              sz.max_S_paths,                       DetArena(allocator,  sz.max_S_paths,                       sz.alloc_pad)),
   KS_configs(           sz.KS_block_config_count_upper_bound, SpanArena(allocator, sz.KS_block_config_count_upper_bound, sz.alloc_pad)),
   KS_single_counts(     sz.KS_block_config_count_upper_bound, IntArena(allocator,  sz.KS_block_config_count_upper_bound, sz.alloc_pad)),
   KS_S_path_counts(     sz.KS_block_config_count_upper_bound, IntArena(allocator,  sz.KS_block_config_count_upper_bound, sz.alloc_pad)),
   KS_CSF_coeffs(        sz.CSF_coeff_count_upper_bound,       RealArena(allocator, sz.CSF_coeff_count_upper_bound,       sz.alloc_pad)), 
#ifdef HUBBARD_TEST
   KS_spins(             sz.KS_block_config_count_upper_bound, RealArena(allocator, sz.KS_block_config_count_upper_bound, sz.alloc_pad)),
#endif
   _KS_H(MArr2R(NULL, 0, 0))
{ 
   KS_H_data = allocator->allocate<real>(sz.max_KS_dim, REAL_EIGEN_ALIGNMENT);

   form_K_basis(basis, K_block_sizes, params);
   sort_K_basis(basis, K_block_sizes, params);

   K_block_begin_indices[0] = 0;
   std::partial_sum(K_block_sizes.cbegin(), K_block_sizes.cend(), K_block_begin_indices.begin() + 1);

   m = params.m;
   S_min = params.S_min();
   S_max = params.S_max();
   S_count = params.S_count();

   reset();
}

void KSBlockIterator::form_KS_subbasis()
{
   TIME_SCOPE("form_KS_subbasis");

   KS_max_path_count = 0;

   int cidx = 0;
   int S_state_count;
   int prev_s_k = -1;
   for(auto config : K_configs()) 
   {
      int s_k = K_single_counts[cidx];
      if(s_k > 0 && S <= 0.5*s_k)
      {
         if(prev_s_k != s_k)
         {
            S_state_count = CSV_dim(S, s_k);
            prev_s_k = s_k;
         }
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
   assert(KS_max_path_count <= S_paths.capacity());
}

void KSBlockIterator::form_CSFs()
{
   TIME_SCOPE("form_CSFs");

   int cidx = 0;
   int prev_s_k = -1;
   for(auto config : KS_configs)
   {
      int s_k = KS_single_counts[cidx];
      if(s_k > 0)
      {
         if(prev_s_k != s_k)
         {
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
               KS_CSF_coeffs.push_back(M_path_sign*compute_CSF_overlap(S_path, M_path, s_k, S, m));
            }
         }
      }
      else
      {
         assert(KS_configs[cidx].size() == 1);
         KS_CSF_coeffs.push_back(1);
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

      assert(has_blocks_left || (total_CSF_count == basis.size()));
   }

   return *this;
}

void KSBlockIterator::reset()
{
   total_CSF_count = 0;   // NOTE: For debug only
   K_block_CSF_count = 0; // NOTE: For debug only
   KS_dim = 0;
   has_blocks_left = true;
   init_K_block(0);
   init_S_block(0);
}

bool KSBlockIterator::next_K_block()
{
   assert(K_block_CSF_count == K_block_sizes[K_block_idx]);
   total_CSF_count += K_block_CSF_count; // NOTE: For debug only

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
   K_block_CSF_count = 0; // NOTE: For debug only
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

   KS_dim = std::reduce(KS_S_path_counts.begin(), KS_S_path_counts.end());
   assert(KS_dim >= 0);
   K_block_CSF_count += KS_dim; // NOTE: For debug only

   new (&_KS_H)  MArr2R(KS_H_data, KS_dim, KS_dim);
   _KS_H.setZero();

   KS_CSF_coeffs.resize(0);
   form_CSFs();
}

StrideItr KSBlockIterator::K_configs()
{
   return StrideItr(std::span<Det>(basis.begin() + K_KS_basis_begin_idx, K_block_sizes[K_block_idx]), &K_dets_per_config);
}

KSBlockIterator::CSFs KSBlockIterator::KS_basis(CSFItr first)
{
   return CSFs{.start_cidx = first.config_idx, .start_pidx = first.S_path_idx, .start_idx = first.idx,
               .start_first_coeff_idx = first.first_coeff_idx, .block_itr = *this, .KS_dim = KS_dim};
}

KSBlockIterator::CSFs KSBlockIterator::KS_basis()
{
   return CSFs{.start_cidx = 0, .start_pidx = 0, .start_idx = 0, .start_first_coeff_idx = 0,
               .block_itr = *this, .KS_dim = KS_dim};
}


// Per-particle energy
real noninteracting_E(int n, real T, int Ns, BCS bcs) // n = 1, 2, ..., Ns
{
   if(bcs == BCS::OPEN)
   {
      return -2.0*T*std::cos(PI*real(n)/real(Ns + 1.0));
   }
   else if(bcs == BCS::PERIODIC)
   {
      return -2.0*T*std::cos(2.0*PI*real(n)/real(Ns));
   }

   assert(!"Supplied BCS not supported (noninteracting_E)!");
   return 0;
}

real noninteracting_E0(const HubbardParams& params, BCS bcs)
{
   real T = params.T;
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

real HubbardModel::H_int(const CSFItr& csf1, const CSFItr& csf2)
{
   return cdev.H_int_element(itr.CSF_dets(csf1), itr.CSF_coeffs(csf1), itr.CSF_size(csf1), 
                             itr.CSF_dets(csf2), itr.CSF_coeffs(csf2), itr.CSF_size(csf2),
                             params);
}

real HubbardModel::H_0(Det det)
{
   real result = 0;
   for(int k = 0; k < itr.K_count; k++)
   {
      result += noninteracting_E(k + 1, itr.params.T, itr.params.Ns, BCS::PERIODIC)*(count_state(det, k) + count_state(det, itr.params.Ns + k));
   }

   return result;
}

real HubbardModel::E0()
{
   if(!recompute_E) { return _E0; }
   update();

   real min_E = std::numeric_limits<real>::max();

   for(itr.reset(); itr; ++itr) 
   {
      if(itr.KS_dim == 0) { continue; }

      {TIME_SCOPE("Matrix");
      // H_int
      for(auto csf1 : itr.KS_basis())
      {
         for(auto csf2 : itr.KS_basis(csf1))
         {
            real cur_elem = H_int(csf1, csf2);
            itr.KS_H(csf1, csf2) += cur_elem;
            itr.KS_H(csf2, csf1) += cur_elem; // can comment out this and
         }
         itr.KS_H(csf1, csf1) *= real(0.5); // this if can safely assume that HubbardComputeDevice expects either only the upper (or lower) triangular part of KS_H to be stored
      }

      // H_0
      int col_idx = 0;
      int cidx = 0;
      for(auto config : itr.KS_configs)
      {
         Det det = get_config_ref_det(config);
         int state_count = itr.KS_S_path_counts[cidx++];

         itr.KS_H().matrix().diagonal()(Eigen::seqN(col_idx, state_count)).array() += H_0(det);
         col_idx += state_count;
      }
      }

      real E0;
      if(itr.KS_dim == 1)
      {
         E0 = itr.KS_H()(0, 0);
      }
      else
      {
         TIME_SCOPE("Diag")
         E0 = cdev.sym_eigs_smallest(itr.KS_H().data(), itr.KS_dim);
      }

      if(E0 < min_E)
      {
         min_E = E0;
      }
   }

   recompute_E = false;
   _E0 = min_E;
   return min_E;
}

HubbardSizes hubbard_memory_requirements(HubbardParams params)
{
   real S_min = std::abs(params.m);
   real S_max = 0.5*params.N;
   int S_count = (S_max - S_min) + 1;
   int max_KS_dim = 0;
   for(int i = 0; i < S_count; i++)
   {
      real cur_S = S_min + i;
      int cur_dim = SM_space_dim(params.N, params.Ns, cur_S);
      if(max_KS_dim < cur_dim) { max_KS_dim = cur_dim; }
   }
   if(max_KS_dim > 2) { max_KS_dim /= params.Ns; }

   int min_singles = int(2.0*std::abs(params.m));
   int max_singles = (params.N <= params.Ns) ? params.N : (2*params.Ns - params.N);

   real max_csv_S = S_min;
   int max_S_paths = ((max_csv_S == 0) && (max_singles == 0)) ? 1 : CSV_dim(max_csv_S, max_singles);
   while(++max_csv_S <= S_max && max_S_paths < CSV_dim(max_csv_S, max_singles)) 
   {
      max_S_paths = CSV_dim(max_csv_S, max_singles);
   }

   int max_dets_in_config = 0;
   for(int single_count = min_singles; single_count <= max_singles; single_count += 2)
   {
      int cur_count = dets_per_orbital_config(single_count, params);
      if(max_dets_in_config < cur_count) { max_dets_in_config = cur_count; }
   }

   // NOTE: CSF_coeff_count_upper_bound might be much larger than actually needed
   HubbardSizes result = {
      .basis_size                        = params.basis_size(),
      .min_singles                       = min_singles,
      .max_singles                       = max_singles,
      .config_count                      = config_count(params),
      .K_block_config_count_upper_bound  = ((params.Ns > 1) ? int(std::ceil(float(result.config_count)/float(params.Ns - 1))) : result.config_count),
      .KS_block_config_count_upper_bound = result.K_block_config_count_upper_bound,
      .max_KS_dim                        = max_KS_dim,
      .max_dets_in_config                = max_dets_in_config,
      .max_S_paths                       = max_S_paths,
      .CSF_coeff_count_upper_bound       = result.KS_block_config_count_upper_bound*result.max_S_paths*result.max_dets_in_config,
      .alloc_pad = 16,
      // TODO: Not all of these need to be available/allocated simultaneously and true required size might be less
      .unaligned_workspace_size          = 
         sizeof(Det)*result.basis_size +                                   // basis
         sizeof(int)*params.Ns +                                           // K_block_sizes
         sizeof(int)*(params.Ns + 1) +                                     // K_block_begin_indices
         sizeof(int)*result.K_block_config_count_upper_bound +             // K_single_counts
         sizeof(int)*result.K_block_config_count_upper_bound +             // K_dets_per_config
         sizeof(Det)*result.max_S_paths +                                  // S_paths
         (sizeof(real)*result.max_KS_dim*result.max_KS_dim + REAL_EIGEN_ALIGNMENT) + // KS_H_data
         sizeof(std::span<Det>)*result.KS_block_config_count_upper_bound + // KS_configs
         sizeof(int)*result.KS_block_config_count_upper_bound +            // KS_single_counts
         sizeof(int)*result.KS_block_config_count_upper_bound +            // KS_S_path_counts
         sizeof(real)*result.CSF_coeff_count_upper_bound                   // KS_CSF_coeffs
#ifdef HUBBARD_TEST
         + sizeof(real)*result.KS_block_config_count_upper_bound           // KS_spins
#endif
      ,.workspace_size                   = result.unaligned_workspace_size + 16*result.alloc_pad 
   };

   return result;
}

Det get_config_ref_det(const std::span<Det>& config)
{ 
   return config.front();
}
