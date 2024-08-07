#include "optimizer.hpp"
// #include "utils.hpp"
#include <Eigen/Dense>
#include <vector>
#include "fusion.h"
#include <RcppEigen.h>
 
// Function to perform optimization
// [[Rcpp::depends(RcppEigen)]]

// Function to compute out-of-the-money (OTM) payoff
double otm_payoff_(double spot, double strike, bool isPut) {
  if (isPut) {
    return std::max(strike - spot, 0.0);
  } else {
    return std::max(spot - strike, 0.0);
  }
}

// Function to compute omega L Mask
std::shared_ptr<monty::ndarray<double, 1>> omegaLMask_(const Eigen::VectorXd& positions, int n) {
  // Initialize an n-long array with all elements set to 0.0
  std::vector<double> x(n);
  auto result = monty::new_array_ptr<double>(x);
  std::fill(result->begin(), result->end(), 0.0);
  
  // Set specified positions to 1.0
  for (int i = 0; i < positions.size(); ++i) {
    int pos = positions(i);
    if (pos < n && pos >= 0) {  // Ensure the position is within bounds
      (*result)(pos) = 1.0;
    }
  }
  
  return result;
}

template<typename T>
// Create a shared_ptr for vectors of the appropriate type
std::shared_ptr<monty::ndarray<T, 1>> eigenToStdVector(const Eigen::VectorXd& eigenVec) {
  auto stdVec = std::make_shared<monty::ndarray<T, 1>>(eigenVec.data(), eigenVec.size());
  
  return stdVec;
}

template<typename T>
// Create a shared_ptr for matrices of the appropriate type
std::shared_ptr<monty::ndarray<T, 2>> eigenToStdMatrix(const Eigen::MatrixXd& eigenMat) {
  auto stdMat = std::make_shared<monty::ndarray<T, 2>>(eigenMat.data(), monty::shape(eigenMat.cols(), eigenMat.rows()));
  return stdMat;
}

// Function to compute gross returns from a payoff matrix
std::vector<double> computeGrossReturns_(const Eigen::MatrixXd& payoff_matrix) {
  // Compute gross returns using Eigen's vectorized operations
  Eigen::VectorXd gross_returns = payoff_matrix.rowwise().sum();
  
  // Convert Eigen::VectorXd to std::vector<double>
  return std::vector<double>(gross_returns.data(), gross_returns.data() + gross_returns.size());
}

// [[Rcpp::export]]
std::array<std::vector<double>, 2> performOptimization(int n, double alpha, double lambda,
                         const Eigen::VectorXd& omega_l_eigen,
                         const Eigen::VectorXd& sp_eigen,
                         const Eigen::VectorXd& strike_eigen,
                         const Eigen::VectorXd& bid_eigen,
                         const Eigen::VectorXd& ask_eigen,
                         const Eigen::Matrix<bool, Eigen::Dynamic, 1>& pFlag_eigen
                           ) {
  // Initialize payoff matrix and compute payoffs
  size_t spLen = sp_eigen.rows();
  size_t optLen = bid_eigen.rows();
  Eigen::MatrixXd payoff_matrix(optLen, spLen);

  // Fill the payoff matrix
  for (size_t i = 0; i < optLen; ++i) {
    for (size_t j = 0; j < spLen; ++j) {
      // Compute OTM payoff based on spot, strike, and option type
      payoff_matrix(i, j) = otm_payoff_(sp_eigen(j), strike_eigen(i), pFlag_eigen(i)) / (0.5 * (bid_eigen[i] + ask_eigen[i]));
      // std::cout << payoff_matrix(i, j) << std::endl;
    }
  }
  
  // Initialize the MOSEK Fusion model
  mosek::fusion::Model::t M = new mosek::fusion::Model("main");
  auto _M = monty::finally([&]() { M->dispose(); });

  // Define variables P and Q
  mosek::fusion::Variable::t p = M->variable("P", n, mosek::fusion::Domain::greaterThan(0.0));  // mosek::fusion::Variable P
  mosek::fusion::Variable::t q = M->variable("Q", n, mosek::fusion::Domain::greaterThan(0.0));  // mosek::fusion::Variable Q

  // Add constraints (make the P and Q congruent distributions)
  M->constraint(mosek::fusion::Expr::sum(p), mosek::fusion::Domain::equalsTo(1.0));  // Sum of p elements equals 1
  M->constraint(mosek::fusion::Expr::sum(q), mosek::fusion::Domain::equalsTo(1.0));  // Sum of q elements equals 1

  // Add constraints involving payoff_matrix and q
  Eigen::VectorXd result_bid = bid_eigen.col(0).array() / (0.5 * (bid_eigen.col(0).array() + ask_eigen.col(0).array()));
  Eigen::VectorXd result_ask = ask_eigen.col(0).array() / (0.5 * (bid_eigen.col(0).array() + ask_eigen.col(0).array()));
  mosek::fusion::Matrix::t payoff_monty_matr = mosek::fusion::Matrix::dense(eigenToStdMatrix<double>(payoff_matrix));

  mosek::fusion::Expression::t product = mosek::fusion::Expr::mul(q, payoff_monty_matr);
  M->constraint("bid_", product, mosek::fusion::Domain::greaterThan(eigenToStdVector<double>(result_bid)));
  M->constraint("ask_", product, mosek::fusion::Domain::lessThan(eigenToStdVector<double>(result_ask)));

  // Constraints for second moment of pricing kernel
  mosek::fusion::Variable::t q_square = M->variable(n, mosek::fusion::Domain::greaterThan(0.0));
  mosek::fusion::Variable::t p_square = M->variable(n, mosek::fusion::Domain::greaterThan(0.0));
  mosek::fusion::Variable::t one = M->variable(1, mosek::fusion::Domain::equalsTo(1.0));
  
  Eigen::VectorXd ones_vector = Eigen::VectorXd::Ones(n);
  auto ones_ptr = std::make_shared<monty::ndarray<double, 1>>(ones_vector.data(), ones_vector.size());
  mosek::fusion::Variable::t ones = M->variable(n, mosek::fusion::Domain::equalsTo(ones_ptr));
  M->constraint("q_square", mosek::fusion::Expr::hstack(mosek::fusion::Expr::mul(q_square, 0.5), ones, q), mosek::fusion::Domain::inRotatedQCone());
  M->constraint("p_square", mosek::fusion::Expr::hstack(mosek::fusion::Expr::mul(p_square, 0.5), ones, p), mosek::fusion::Domain::inRotatedQCone());

  mosek::fusion::Variable::t u = M->variable(n);
  M->constraint(mosek::fusion::Expr::hstack(mosek::fusion::Expr::mul(u, 0.5), p, q), mosek::fusion::Domain::inRotatedQCone());
  M->constraint(mosek::fusion::Expr::sum(u), mosek::fusion::Domain::lessThan(alpha));

  // Variance constraint using dot product
  std::vector<double> gross_returns = computeGrossReturns_(payoff_matrix);
  std::shared_ptr<monty::ndarray<double, 1>> payoff(new monty::ndarray<double, 1>(n));
  for (int i = 0; i < n; ++i) {
    (*payoff)[i] = gross_returns[i] - log(gross_returns[i]) - 1;
  }

  mosek::fusion::Expression::t p_var = mosek::fusion::Expr::dot(payoff, p);
  mosek::fusion::Expression::t q_var = mosek::fusion::Expr::dot(payoff, q);

  M->constraint(mosek::fusion::Expr::sub(p_var, q_var), mosek::fusion::Domain::lessThan(0.0));

  mosek::fusion::Variable::t p_vari = M->variable(1, mosek::fusion::Domain::greaterThan(0.0));
  mosek::fusion::Variable::t q_vari = M->variable(1, mosek::fusion::Domain::greaterThan(0.0));
  M->constraint(mosek::fusion::Expr::sub(p_var, p_vari), mosek::fusion::Domain::equalsTo(0.0));
  M->constraint(mosek::fusion::Expr::sub(q_var, q_vari), mosek::fusion::Domain::equalsTo(0.0));

  // Define objective function using mask omega_l
  std::shared_ptr<monty::ndarray<double, 1>> mask = omegaLMask_(omega_l_eigen, n);

  mosek::fusion::Expression::t obj_expr = mosek::fusion::Expr::add(mosek::fusion::Expr::dot(mask, p), mosek::fusion::Expr::dot(mask, q));
  mosek::fusion::Expression::t regularization = mosek::fusion::Expr::add(mosek::fusion::Expr::sum(p_square), mosek::fusion::Expr::sum(q_square));
  mosek::fusion::Expression::t obj_expr_reg = mosek::fusion::Expr::sub(obj_expr, mosek::fusion::Expr::mul(lambda, regularization));
  
  M->objective("obj", mosek::fusion::ObjectiveSense::Maximize, obj_expr_reg);

  // Solve the problem
  M->solve();
  
  // Retrieve and convert solution
  auto p_ptr = p->level();
  auto q_ptr = q->level();

  std::vector<double> p_vec(p_ptr->begin(), p_ptr->end());
  std::vector<double> q_vec(q_ptr->begin(), q_ptr->end());
  
  // Create an array to hold the two vectors
  std::array<std::vector<double>, 2> result = { p_vec, q_vec };
  
  return result;
}
