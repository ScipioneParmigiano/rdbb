#ifndef OPTIMIZER_HPP
#define OPTIMIZER_HPP

#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include "fusion.h"

// std::shared_ptr<monty::ndarray<double, 1>> omegaLMask(const Eigen::VectorXd& positions, int n);

template<typename T>
std::shared_ptr<monty::ndarray<T, 1>> eigenToStdVector(const Eigen::Matrix<T, Eigen::Dynamic, 1>& eigenVec);

template<typename T>
std::shared_ptr<monty::ndarray<T, 2>> eigenToStdMatrix(const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& eigenMat);

// std::vector<double>
std::array<std::vector<double>, 2> performOptimization(int n, double alpha, double lambda,
                                        const Eigen::VectorXd& omega_l_eigen,
                                        const Eigen::VectorXd& sp_eigen,
                                        const Eigen::VectorXd& strike_eigen,
                                        const Eigen::VectorXd& bid_eigen,
                                        const Eigen::VectorXd& ask_eigen,
                                        const Eigen::Matrix<bool, Eigen::Dynamic, 1>& pFlag_eigen);

#endif