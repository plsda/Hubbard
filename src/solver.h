#ifndef SOLVER_H

enum class BCS
{
   OPEN = 1,
   PERIODIC = 2,
};

real noninteracting_E(int n, real T, int Ns, BCS bcs);
real noninteracting_E0(const HubbardParams& params, BCS bcs);
real dimer_E0(const HubbardParams& params, BCS bcs);
real atomic_E0(const HubbardParams& params);
real halffilled_E_per_N(real T, real U, IntArgs int_args);
real kfm_basis_compute_E0(const KConfigs& configs, const HubbardParams& params);
real kfm_basis_compute_E0(const HubbardParams& params);

#define SOLVER_H
#endif
