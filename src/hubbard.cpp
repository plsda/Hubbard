#include "hubbard.h"

template <class... Arrays>
void sort_multiple(Arrays&... arrays)
{
   const auto& sort_array = std::get<0>(std::forward_as_tuple(arrays...)); // Use the first array as a reference

   size_t len = sort_array.size();
   std::vector<size_t> sort_indices(len); // sort_indices[i] = index of the i'th element of the sorted array in the original array
   std::iota(sort_indices.begin(), sort_indices.end(), 0);
   std::sort(sort_indices.begin(), sort_indices.end(), [&sort_array](size_t i, size_t j){return sort_array[i] < sort_array[j];});

   for(int dest_idx = 0; dest_idx < len; dest_idx++)
   {
      size_t src_idx = sort_indices[dest_idx];

      int update_idx = dest_idx;
      for(; sort_indices[update_idx] != dest_idx; update_idx++) {}

      sort_indices[dest_idx] = dest_idx;
      sort_indices[update_idx] = src_idx;

      (std::swap(arrays[dest_idx], arrays[src_idx]),...);
   }
}

template <class T>
T pop_vec(std::vector<T>& vec, int i)
{
   T result = vec[i];
   std::swap(vec[i], vec.back());
   vec.pop_back();

   return result;
}

template <class T>
T pop_vec(std::vector<T>& vec, int i, int& one_past_last)
{
   T result = vec[i];
   std::swap(vec[i], vec[one_past_last-1]);
   one_past_last--;

   return result;
}

std::pair<Det, Det> get_det_up_down(Det det, HubbardParams params)
{
   Det det_up = det & ((2 << (params.Ns - 1)) - 1);
   Det det_down = det >> params.Ns;
   return {det_up, det_down};
}

int count_singles(Det det, const HubbardParams& params)
{
   auto [det_up, det_down] = get_det_up_down(det, params);
   return bitcount_lookup[det_up ^ det_down];
}

// Dimension of the configuration state vector for an orbital configuration with 'singles_count' singles and total spin 'spin'
int CSV_dim(real spin, int single_count)
{
   int result = 0;

   if((single_count > 0) &&
      ((((single_count & 1) == 1) && !is_close(int(spin), spin)) || (((single_count & 1) == 0) && is_close(int(spin), spin))) &&
      (single_count >= 2*spin))
   {
      result = (2.0*spin + 1.0)/(single_count + 1.0)*choose(single_count + 1, 0.5*single_count - spin);

      assert(is_close(int(result), result));
   }

   return result;
}

Det det_config_ID(Det det, const HubbardParams& params)
{
   auto [det_up, det_down] = get_det_up_down(det, params);
   Det det_xor = det_up ^ det_down;
   Det det_and = det_up & det_down;

   Det result = (det_xor << params.Ns) | det_and;

   return result;
}

struct Det_xor_and
{
   Det det_xor;
   Det det_and;
};
Det_xor_and get_det_xor_and(Det det, const HubbardParams& params)
{
   auto [det_up, det_down] = get_det_up_down(det, params);
   Det det_xor = det_up ^ det_down;
   Det det_and = det_up & det_down;

   return {det_xor, det_and};
}

bool cmp_det_config(Det det1, Det det2, HubbardParams params)
{
   return (det_config_ID(det1, params) == det_config_ID(det2, params));
}

// Convert a (momentum) state index into momentum value in the first Brillouin zone
real kidx2k(int k, int Ns) // k = 1, 2, ..., Ns
{
   if((Ns & 1) == 0)
   {
      k = k - int(Ns/2);
   }
   else
   {
      k = k - int(Ns/2) - 1;
   }

   return 2.0*PI*k/Ns;
}

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

// Particle number operator
u32 count_state(Det det, u32 state_idx) // state_idx = 0, 1, ...
{
   return (det >> state_idx) & 1;
}

u32 get_pop(Det det)
{
   // 16-bit lookup
   u32 result = bitcount_lookup[det & 0xFFFF] + bitcount_lookup[det >> 16];
   return result;
}

// Count states with higher index (i.e. nothing to do with level of excitation necessarily)
u32 count_higher(u32 state_idx, Det det)
{
   u32 mask = u32(0xFFFFFFFF << (state_idx + 1));
   return get_pop(det & mask);
}

u32 count_lower(u32 state_idx, Det det)
{
   u32 mask = ~u32(0xFFFFFFFF << state_idx);
   return get_pop(det & mask);
}

u32 get_up_pop(Det det, u32 Ns)
{
   return count_lower(Ns, det);
}

u32 get_down_pop(Det det, u32 Ns)
{
   return count_higher(Ns - 1, det);
}

// Fermionic creation operator
// NOTE: Null ket is denoted by setting sign = 0 and vacuum by det = 0
SDet sadd(u32 state_idx, Det det, int sign = 1)
{
   int neg_state_occ = 1 - count_state(det, state_idx);
   Det result = (det | (1 << state_idx))*neg_state_occ;

   sign *= neg_state_occ;
   sign *= 1 - 2*(count_higher(state_idx, det) & 1);

   return {result, sign};
}

SDet sadd(u32 state_idx, SDet det)
{
   return sadd(state_idx, det.det, det.sign);
}

// Fermionic annihilation operator
SDet ssub(u32 state_idx, Det det, int sign = 1)
{
   int state_occ = count_state(det, state_idx);
   Det result = (det & (~(1 << state_idx)))*state_occ;

   sign *= state_occ;
   sign *= 1 - 2*(count_higher(state_idx, det) & 1);

   return {result, sign};
}

SDet ssub(u32 state_idx, SDet det)
{
   return ssub(state_idx, det.det, det.sign);
}

real compute_interaction_element(Det ket, const HubbardParams& params)
{
   real result = 0;
   for(int s = 0; s < params.Ns; s++)
   {
      result += params.U*count_state(ket, s)*count_state(ket, params.Ns + s);
   }

   return result;
}

template <class T>
Det statelist2det(const T& statelist)
{
    Det result = 0;
    for(auto b : statelist)
    {
       result |= (1 << b);
    }

    return result;
}

void list_spinless_determinants(std::vector<Det>& result, u32 N_particles, u32 N_states)
{
   if(N_particles > 0)
   {

      std::vector<u32> det(N_particles);
      std::iota(det.begin(), det.end(), 0);
      result[0] = Det(-1) >> (8*sizeof(Det) - N_particles);
      u32 det_idx = 1;

      // To form the next state:
      // 1. Check if the last state(i.e. highest state in the bit string) can be incremented
      //    2. If yes, done
      //       If not, check if the state of the preceding particle can be incremented (depends on idx of the
      //       particle and max state idx)
      //       3.  If yes, increment the state of the preceding particle and set the state of 
      //           all the successive particles in an increasing sequence. Go to to 1.
      //           If not, check the preceding state until you find particle whose state that can be incremented or
      //           return if none exist
      //

#ifdef HUBBARD_DEBUG
      std::set<Det> states;
#endif

      bool done = false;
      while(!done)
      {
         // Loop until a suitable state to increment is found
         int particle_idx = N_particles - 1;
         while((particle_idx >= 0) && !(det[particle_idx] < (N_states - N_particles + particle_idx))) 
         {
            particle_idx--;
         }

         if(particle_idx >= 0)
         {
            det[particle_idx]++;
            
            for(u32 i = 1; i < N_particles - particle_idx; i++)
            {
               det[particle_idx + i] = det[particle_idx] + i;
            }

            Det new_det = statelist2det(det);
            result[det_idx] = new_det;

#ifdef HUBBARD_DEBUG
            assert(!states.contains(new_det));
            states.insert(new_det);
#endif
            det_idx++;
         }
         else
         {
            done = true;
         }
      }

      assert(det_idx == int(choose(N_states, N_particles)));
   }
   else
   {
      result[0] = 0;
   }
}

void list_determinants(std::vector<Det>& result, const HubbardParams& params)
{
    std::vector<Det> up_dets(choose(params.Ns, params.N_up));
    std::vector<Det> down_dets(choose(params.Ns, params.N_down));

    list_spinless_determinants(up_dets, params.N_up, params.Ns);
    list_spinless_determinants(down_dets, params.N_down, params.Ns);

    u32 det_idx = 0;
    for(Det up : up_dets)
    {
       for(Det down : down_dets)
       {
          result[det_idx] = (down << params.Ns) | up;
          det_idx++;
       }
    }
}

int state_momentum(Det det, const HubbardParams& params)
{

    auto [up_det, down_det] = get_det_up_down(det, params);

    // NOTE: Here, we take the single-particle momenta to be in the range 0, 1, ..., Ns - 1
    int result = 0;
    for(int sidx = 0; sidx < int(params.Ns); sidx++)
    {
       // NOTE: Can this overflow? (currently, modulo is taken after the loop)
       // Max value for a single-particle momentum, k_i, is Ns
       // Upper bound for the total momentum, K, of a state is 
       //   K < Ns*(Ns*2) = 2*Ns^2
       // --> Unlikely to overflow when using int
       result += sidx*(count_state(up_det, sidx) + count_state(down_det, sidx));
    }

    result = result % params.Ns;

    return result;
}

KBasis form_K_basis(const HubbardParams& params)
{
    int basis_size = params.basis_size();
    std::vector<Det> basis(basis_size);
    std::vector<int> momenta(basis_size);
    std::vector<int> block_sizes(params.Ns);

    list_determinants(basis, params);

    for(int basis_idx = 0; basis_idx < basis_size; basis_idx++) 
    {
        int K = state_momentum(basis[basis_idx], params);
        momenta[basis_idx] = K;
        block_sizes[K]++;
    }

    sort_multiple(momenta, basis);

    return {basis, momenta, block_sizes};
}

// Naive path search
void form_S_paths(Det path, int cur_s, real cur_f, int s, real f, std::vector<Det>& result)
{
   assert((cur_s >= 0) && (s >= 0));

   if((cur_f != f) || (cur_s != s))
   {
      if((cur_s < s) && (2*std::abs(f - cur_f) <= (s - cur_s)))
      {
         // Up edge
         Det new_path = path;
         new_path <<= 1;
         new_path |= 1;
         form_S_paths(new_path, cur_s + 1, cur_f + 0.5, s, f, result);

         if(cur_f > 0)
         {
            // Down edge
            new_path = path;
            new_path <<= 1;
            form_S_paths(new_path, cur_s + 1, cur_f - 0.5, s, f, result);
         }
      }
   }
   else
   {
      result.push_back(path);
   }
}

template <class T>
T pop_bit(T& a)
{
   T result = a & 1;
   a >>= 1;
   return result;
}

template <class T>
T& push_bit(T& a, u8 b = 1)
{
   a <<= 1;
   a |= b;
   return a;
}

SDet det2path(Det det, const HubbardParams& params)
{
   auto [det_up, det_down] = get_det_up_down(det, params);

   Det double_mask = det_up ^ det_down;
   double_mask = (double_mask << params.Ns) | double_mask;

   Det doubleless_det = det & double_mask;
   auto [doubleless_up, doubleless_dn] = get_det_up_down(doubleless_det, params);

   size_t dn_count = 0;
   int sign = 1;

   Det path = 0;
   for(int i = 0; i < params.Ns; i++)
   {
      u32 up_occupancy = pop_bit(det_up);
      u32 down_occupancy = pop_bit(det_down);

      if(up_occupancy ^ down_occupancy)
      {
         push_bit(path, up_occupancy);
      }

      u32 remaining_N_up = count_higher(i, doubleless_up);
      u32 remaining_N_dn = count_higher(0, doubleless_dn);
      bool doubless_dn_occ = pop_bit(doubleless_dn);

      if(doubless_dn_occ)
      {
         sign *= ((dn_count + remaining_N_up + remaining_N_dn) & 1) ? -1 : 1;
         dn_count += 1;
      }
   }

   return {path, sign};
}

int get_path_edge(Det path, size_t idx)
{
   int edge = (path & (1 << idx)) >> idx;
   return edge;
}

real compute_SCF_overlap(Det S_path, Det M_path, int edge_count, real f, real m)
{
   assert(edge_count >= 0);

   real cur_f = 0;
   real cur_m = 0;
   real overlap = 1;
   int sign = 1;

   for(int i = 0; i < edge_count; i++)
   {
      int edge_idx = edge_count - i - 1;
      int S_edge = get_path_edge(S_path, edge_idx);
      int M_edge = get_path_edge(M_path, edge_idx);

      cur_f += (S_edge - 0.5);
      cur_m += (M_edge - 0.5);

      if(std::abs(cur_m) > cur_f)
      {
         return 0;
      }

      real C;

      if (S_edge == 1 && M_edge == 1)
      {
         C = (cur_f + cur_m)/(2*cur_f);
      }
      else if(S_edge == 0 && M_edge == 1)
      {
         C = (cur_f - cur_m + 1)/(2*cur_f + 2);
         sign *= -1;
      }
      else if(S_edge == 1 && M_edge == 0)
      {
         C = (cur_f - cur_m)/(2*cur_f);
      }
      else
      {
         C = (cur_f + cur_m + 1)/(2*cur_f + 2);
      }

      overlap *= C;
   }

   assert(cur_f == f);
   assert(cur_m == m);

   overlap = sign*std::sqrt(overlap);
   return overlap;
}

template <class T>
real compute_H_int_element(const std::vector<Det>& bra_dets, const Eigen::TensorRef<T>& bra_coeffs,
                           const std::vector<Det>& ket_dets, const Eigen::TensorRef<T>& ket_coeffs,
                           const HubbardParams& params)
{
   real result = 0;

   Det bra_ref_det = bra_dets[0];
   Det ket_ref_det = ket_dets[0];
   auto [bup, bdn] = get_det_up_down(bra_ref_det, params);
   auto [kup, kdn] = get_det_up_down(ket_ref_det, params);
   int spinless_diff = bitcount_lookup[(bup | bdn) ^ (kup | kdn)] +
                       bitcount_lookup[(bup & bdn) ^ (kup & kdn)];
   
   if(spinless_diff <= 4)
   {
      for(int k1 = 0; k1 < int(params.Ns); k1++)
      {
         for(int k2 = 0; k2 < int(params.Ns); k2++)
         {
            for(int q = 0; q < int(params.Ns); q++)
            {
               for(int ket_det_idx = 0; ket_det_idx < int(ket_dets.size()); ket_det_idx++)
               {
                  Det ket_det = ket_dets[ket_det_idx];
                  auto [new_ket_det, new_ket_det_sign] = sadd(mod(k1 + q, params.Ns),
                                                              ssub(k1,
                                                                   sadd(mod(k2 - q, params.Ns) + params.Ns,
                                                                        ssub(k2 + params.Ns, 
                                                                             ket_det))));

                  if(new_ket_det_sign != 0 && cmp_det_config(bra_dets[0], new_ket_det, params))
                  {
                     for(int bra_det_idx = 0; bra_det_idx < bra_dets.size(); bra_det_idx++)
                     {
                        Det bra_det = bra_dets[bra_det_idx];
                        assert(bra_det != 0);

                        result += new_ket_det_sign*bra_coeffs(bra_det_idx)*ket_coeffs(ket_det_idx)*(bra_det == new_ket_det);
                     }
                  }

               }
            }
         }
      }

      result *= params.U/params.Ns;
   }

   return result;
}

KConfigs get_k_orbitals(const HubbardParams& params)
{
   auto [basis, K, block_sizes] = form_K_basis(params);

   int i0 = 0;
   std::vector<std::vector<std::vector<Det>>> orbitals;

   for(int block_size : block_sizes)
   {
      int i1 = i0 + block_size;

      std::vector<std::vector<Det>> K_orbitals;
      while(i1 > i0)
      {
         Det ref_det = basis[i1 - 1];
         i1--;

         std::vector<Det> ref_K_orbitals = {ref_det};

         int didx = i1 - 1;
         while(didx >= i0)
         {
            Det det = basis[didx];

            if(cmp_det_config(ref_det, det, params))
            {
               ref_K_orbitals.push_back(det);
               pop_vec(basis, didx, i1);
            }

            didx--;
         }

         K_orbitals.push_back(ref_K_orbitals);
      }

      orbitals.push_back(K_orbitals);
      i0 += block_size;
   }

   return {orbitals, block_sizes};
}

void form_KS_subbasis(real f, real m,
                      const std::vector<std::vector<Det>>& K_configs,
                      const std::vector<int>& single_counts,
                      std::vector<std::vector<Det>*>& Kf_basis,
                      std::vector<int>& Kf_single_counts,
                      std::vector<int>& Kf_counts,
                      std::vector<real>& Kf_spins,
                      int& max_config_count,
                      int& max_path_count)
{
   max_config_count = 0;
   max_path_count = 0;

   for(int config_idx = 0; config_idx < K_configs.size(); config_idx++)
   {
      const std::vector<Det>& config = K_configs[config_idx];
      int config_single_count = single_counts[config_idx];

      if(config_single_count > 0 && f <= 0.5*config_single_count)
      {
         int fstate_count = CSV_dim(f, config_single_count);
         if(fstate_count > 0)
         {
            Kf_basis.push_back(const_cast<std::vector<Det>*>(&config));
            Kf_single_counts.push_back(config_single_count);
            Kf_counts.push_back(fstate_count);
            Kf_spins.push_back(f);

            if(fstate_count > max_path_count)
            {
               max_path_count = fstate_count;
            }
         }
      }
      else if(f == 0)
      {
         // Handle singlets/closed-shell configs
         assert(m == 0);
         Kf_basis.push_back(const_cast<std::vector<Det>*>(&config));
         Kf_single_counts.push_back(0);
         Kf_counts.push_back(1);
         Kf_spins.push_back(f);

         if(max_path_count < 1)
         {
            max_path_count = 1;
         }
      }

      int cur_config_count = int(config.size());
      if(cur_config_count > max_config_count)
      {
         max_config_count = cur_config_count;
      }
   }

}

void form_KS_subbasis(real f, real m,
                      const std::vector<std::vector<Det>>& K_configs,
                      const std::vector<int>& single_counts,
                      std::vector<std::vector<Det>*>& Kf_basis,
                      std::vector<int>& Kf_single_counts,
                      std::vector<int>& Kf_counts,
                      int& max_config_count,
                      int& max_path_count)
{
   max_config_count = 0;
   max_path_count = 0;

   for(int config_idx = 0; config_idx < K_configs.size(); config_idx++)
   {
      const std::vector<Det>& config = K_configs[config_idx];
      int config_single_count = single_counts[config_idx];

      if(config_single_count > 0 && f <= 0.5*config_single_count)
      {
         int fstate_count = CSV_dim(f, config_single_count);
         if(fstate_count > 0)
         {
            Kf_basis.push_back(const_cast<std::vector<Det>*>(&config));
            Kf_single_counts.push_back(config_single_count);
            Kf_counts.push_back(fstate_count);

            if(fstate_count > max_path_count)
            {
               max_path_count = fstate_count;
            }
         }
      }
      else if(f == 0)
      {
         // Handle singlets/closed-shell configs
         assert(m == 0);
         Kf_basis.push_back(const_cast<std::vector<Det>*>(&config));
         Kf_single_counts.push_back(0);
         Kf_counts.push_back(1);

         if(max_path_count < 1)
         {
            max_path_count = 1;
         }
      }

      int cur_config_count = int(config.size());
      if(cur_config_count > max_config_count)
      {
         max_config_count = cur_config_count;
      }
   }

}

void form_SCFs(real f, real m,
               std::span<std::vector<Det>*> Kf_basis,
               std::span<int> single_counts,
               std::span<int> S_path_counts,
               std::vector<Det>& S_paths,
               const HubbardParams& params,
               std::vector<real>& result)
{
   for(int k_idx = 0; k_idx < single_counts.size(); k_idx++)
   {
      int s_k = single_counts[k_idx];
      if(s_k > 0)
      {
         S_paths.resize(0);
         form_S_paths(1, 1, 0.5, s_k, f, S_paths);
         assert(S_paths.size() == S_path_counts[k_idx]);

         for(int s_path_idx = 0; s_path_idx < S_paths.size(); s_path_idx++)
         {
            Det S_path = S_paths[s_path_idx];
            const std::vector<Det>* const cur_dets = Kf_basis[k_idx];

            for(int det_idx = 0; det_idx < cur_dets->size(); det_idx++)
            {
               Det det = cur_dets->at(det_idx);
               auto [M_path, M_path_sign] = det2path(det, params);

               result.push_back(M_path_sign*compute_SCF_overlap(S_path, M_path, s_k, f, m));
            }
         }
      }
      else
      {
         assert(Kf_basis[k_idx]->size() == 1);
         result.push_back(1);
      }
   }
}

void form_SCFs(real f, real m,
               std::span<std::vector<Det>*> Kf_basis,
               std::span<int> single_counts,
               std::span<int> S_path_counts,
               std::vector<Det>& S_paths,
               const HubbardParams& params,
               Arr3R& result)
{
   for(int k_idx = 0; k_idx < single_counts.size(); k_idx++)
   {
      int s_k = single_counts[k_idx];
      if(s_k > 0)
      {
         S_paths.resize(0);
         form_S_paths(1, 1, 0.5, s_k, f, S_paths);
         assert(S_paths.size() == S_path_counts[k_idx]);

         for(int s_path_idx = 0; s_path_idx < S_paths.size(); s_path_idx++)
         {
            Det S_path = S_paths[s_path_idx];
            const std::vector<Det>* const cur_dets = Kf_basis[k_idx];

            for(int det_idx = 0; det_idx < cur_dets->size(); det_idx++)
            {
               Det det = cur_dets->at(det_idx);
               auto [M_path, M_path_sign] = det2path(det, params);

               result(k_idx, s_path_idx, det_idx) = M_path_sign*compute_SCF_overlap(S_path, M_path, s_k, f, m);
            }

         }
      }
      else
      {
         assert(Kf_basis[k_idx]->size() == 1);
         result(k_idx, 0, 0) = 1;
      }
   }
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


void det2spinless_statelist(Det det, const HubbardParams& params, std::vector<int>& result)
{
   auto [det_up, det_dn] = get_det_up_down(det, params);

   for(int i = 0; i < params.Ns; i++)
   {
      if(pop_bit(det_up))
      {
         result.push_back(i);
      }

      if(pop_bit(det_dn))
      {
         result.push_back(i);
      }
   }
}

real SCF_spin(const std::vector<Det>& dets, std::span<real> coeffs, const HubbardParams& params)
{
   int Ns = params.Ns;
   real m = 0.5*(params.N_up - params.N_down);
   real m2 = m*m;
   real result = 0;

   for(int bra_idx = 0; bra_idx < dets.size(); bra_idx++)
   {
      Det bra = dets[bra_idx];
      real bra_coeff = coeffs[bra_idx];

      for(int ket_idx = 0; ket_idx < dets.size(); ket_idx++)
      {
         Det ket = dets[ket_idx];
         real ket_coeff = coeffs[ket_idx];

         for(int k1 = 0; k1 < Ns; k1++)
         {
            for(int k2 = 0; k2 < Ns; k2++)
            {
               auto [new_ket, sign] = sadd(k1 + Ns,
                                           ssub(k1,
                                                sadd(k2,
                                                     ssub(k2 + Ns, ket))));

               result += sign*bra_coeff*ket_coeff*(bra == new_ket);
            }
         }

         result += bra_coeff*ket_coeff*(m + m2)*(bra == ket);
      }
   }

   result = 0.5*(std::sqrt(4.0*result + 1.0) - 1.0);
   assert((result >= 0.0) || is_close(result, 0.0, real(1e-6)));

   return result;
}

real SCF_inner(const std::vector<Det>& bra_dets, std::span<real> bra_coeffs, 
               const std::vector<Det>& ket_dets, std::span<real> ket_coeffs)
{
   real result = 0;
   for(int bra_det_idx = 0; bra_det_idx < bra_dets.size(); bra_det_idx++)
   {
      Det bra_det = bra_dets[bra_det_idx];

      for(int ket_det_idx = 0; ket_det_idx < ket_dets.size(); ket_det_idx++)
      {
         Det ket_det = ket_dets[ket_det_idx];
         result += bra_coeffs[bra_det_idx]*ket_coeffs[ket_det_idx]*(bra_det == ket_det);
      }
   }

   return result;
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
