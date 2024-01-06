#ifndef HUBBARD_H

#include <iostream>
#include <span>

#include "hubbard_compute.h"
#include "solver.h"

#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
using MatR = Eigen::Matrix<real, Eigen::Dynamic, Eigen::Dynamic>;
using Arr2R = Eigen::Array<real, Eigen::Dynamic, Eigen::Dynamic>;
using Arr3R = Eigen::Tensor<real, 3>;

#include <numbers>
#define PI std::numbers::pi_v<real>
#define LOG2 std::numbers::log2e_v<real>

#define HUBBARD_H
#endif
