#include "hubbard.h"

class HubbardComputeDevice::ComputeContext {};

HubbardComputeDevice::HubbardComputeDevice(std::stringstream* const errors = 0) : errors(errors) {}

HubbardComputeDevice::~HubbardComputeDevice() {}

real HubbardComputeDevice::H_int_element(const Det* const bra_dets, const real* const bra_coeffs, int bra_count, 
                                           const Det* const ket_dets, const real* const ket_coeffs, int ket_count,
                                           const HubbardParams& params)
{
   assert(bra_count > 0 && ket_count > 0);

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
               for(int ket_det_idx = 0; ket_det_idx < ket_count; ket_det_idx++)
               {
                  Det ket_det = ket_dets[ket_det_idx];
                  auto [new_ket_det, new_ket_det_sign] = sadd(mod(k1 + q, params.Ns),
                                                              ssub(k1,
                                                                   sadd(mod(k2 - q, params.Ns) + params.Ns,
                                                                        ssub(k2 + params.Ns, 
                                                                             ket_det))));

                  if(new_ket_det_sign != 0 && cmp_det_config(bra_dets[0], new_ket_det, params))
                  {
                     for(int bra_det_idx = 0; bra_det_idx < bra_count; bra_det_idx++)
                     {
                        Det bra_det = bra_dets[bra_det_idx];
                        assert(bra_det != 0);

                        real cur_res = new_ket_det_sign*bra_coeffs[bra_det_idx]*ket_coeffs[ket_det_idx]*(bra_det == new_ket_det);

                        result += cur_res;
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

real HubbardComputeDevice::sym_eigs_smallest(real* elements, int dim)
{
   Eigen::Map<MatR> m(elements, dim, dim);
   Eigen::SelfAdjointEigenSolver<MatR> eigensolver(m);
   real result = eigensolver.eigenvalues()[0];

   return result;
}
