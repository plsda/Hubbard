#ifndef SOLVER_H

#include <numbers>
#define PI std::numbers::pi_v<real>
#define LOG2 std::numbers::log2e_v<real>

#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
using MatR = Eigen::Matrix<real, Eigen::Dynamic, Eigen::Dynamic>;
using Arr2R = Eigen::Array<real, Eigen::Dynamic, Eigen::Dynamic>;
using Arr3R = Eigen::Tensor<real, 3>;

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
real kfm_basis_compute_E0(HubbardComputeDevice& cdev, const KConfigs& configs, const HubbardParams& params);
real kfm_basis_compute_E0(HubbardComputeDevice& cdev, const HubbardParams& params);

#define SOLVER_H
#endif
