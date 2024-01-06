#ifndef HUBBARD_COMPUTE_H

#include "hubbard_common.h"
#include "utils.h"
#include "basis.h"


bool init_compute(std::stringstream* const errors);

real compute_H_int_element(const Det* const bra_dets, const real* const bra_coeffs, int bra_count, 
                           const Det* const ket_dets, const real* const ket_coeffs, int ket_count,
                           const HubbardParams& params);

// TODO: real sym_eigs();

#define HUBBARD_COMPUTE_H
#endif
