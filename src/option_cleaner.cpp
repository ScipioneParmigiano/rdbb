// #include "utils.hpp"
#include "option_cleaner.hpp"
#include <vector>

// [[Rcpp::depends(RcppEigen)]]

// Function to compute out-of-the-money (OTM) payoff
double otm_payoff(double spot, double strike, bool isPut) {
  if (isPut) {
    return std::max(strike - spot, 0.0);
  } else {
    return std::max(spot - strike, 0.0);
  }
}

// Function to compute omega L Mask
std::shared_ptr<monty::ndarray<double, 1>> omegaLMask(const Eigen::VectorXd& positions, int n) {
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
std::vector<double> computeGrossReturns(const Eigen::MatrixXd& payoff_matrix) {
  // Compute gross returns using Eigen's vectorized operations
  Eigen::VectorXd gross_returns = payoff_matrix.rowwise().sum();
  
  // Convert Eigen::VectorXd to std::vector<double>
  return std::vector<double>(gross_returns.data(), gross_returns.data() + gross_returns.size());
}

// [[Rcpp::export]]
Eigen::Matrix<bool, Eigen::Dynamic, 1>  getFeasibleOptionFlags(const Eigen::VectorXd& sp,
                         const Eigen::VectorXd& bid,
                         const Eigen::VectorXd& ask,
                         const Eigen::VectorXd& strike,
                         const Eigen::Matrix<bool, Eigen::Dynamic, 1>& pFlag,
                         double spotsP,
                         double spbid,
                         double spask) {

    M_Model M = new mosek::fusion::Model("FeasibleOptionFlags");
    auto _M = monty::finally([&]() {
      M->dispose();
    });

    const double SCALER(sp.size());

    Eigen::VectorXd lb = bid.cwiseQuotient(0.5 * (bid + ask) * spotsP) * SCALER;
    Eigen::VectorXd ub = ask.cwiseQuotient(0.5 * (bid + ask) * spotsP) * SCALER;

    unsigned int OPTLEN(strike.size());
    unsigned int LEN(sp.size());

    Eigen::VectorXd payoffMat(OPTLEN * LEN);

    // Fill the payoff matrix
    for (size_t i = 0; i < OPTLEN; ++i) {
      for (size_t j = 0; j < LEN; ++j) {
        payoffMat[i * LEN + j] = otm_payoff(sp[j], strike[i], pFlag[i]) / (0.5 * (bid[i] + ask[i]) * spotsP);
      }
    }

    const M_Matrix::t payoff_wrap = M_Matrix::dense(std::shared_ptr<M_ndarray_2>(new M_ndarray_2(payoffMat.data(), monty::shape(OPTLEN, LEN))));

    // Q_vars are q probabilities
    M_Variable::t q_vars = M->variable("q_vars", LEN, M_Domain::inRange(0.0, SCALER));
    // opt_vars is the desired output (bool)
    M_Variable::t optVars = M->variable("optVars", OPTLEN, M_Domain::binary());

    // Upper and lower bounds
    auto lb_wrap = std::shared_ptr<M_ndarray_1>(new M_ndarray_1(lb.data(), monty::shape(OPTLEN)));
    auto ub_wrap = std::shared_ptr<M_ndarray_1>(new M_ndarray_1(ub.data(), monty::shape(OPTLEN)));

    M->constraint(M_Expr::sub(M_Expr::mul(payoff_wrap, q_vars), M_Expr::mulElm(lb_wrap, optVars)), M_Domain::greaterThan(0.0));
    M->constraint(M_Expr::sub(M_Expr::mul(payoff_wrap, q_vars), M_Expr::add(M_Expr::mulElm(ub_wrap, optVars), M_Expr::mul(optVars, -SCALER * spotsP))), M_Domain::lessThan(SCALER * spotsP));

    M->constraint(M_Expr::sum(q_vars), M_Domain::equalsTo(SCALER));
    auto sp_wrap = std::shared_ptr<M_ndarray_1>(new M_ndarray_1(LEN, std::function<double(ptrdiff_t)>([&](ptrdiff_t i) { return sp(i); })));

    // Forward pricing constraints
    M->constraint(M_Expr::mul(1.0 / spotsP, M_Expr::dot(sp_wrap, q_vars)), M_Domain::inRange(SCALER * spbid / spotsP, SCALER * spask / spotsP));

    M->objective(mosek::fusion::ObjectiveSense::Maximize, M_Expr::sum(optVars));

    M->solve();
    Eigen::Matrix<bool, Eigen::Dynamic, 1> outOpt;
    if (M->getPrimalSolutionStatus() == mosek::fusion::SolutionStatus::Optimal) {
      auto sol = optVars->level();
      const Eigen::Map<Eigen::VectorXd> solWrap(sol->raw(), OPTLEN);
      outOpt = solWrap.unaryExpr([](double arg) { return arg >= 1.0 ? true : false; });
    } else {
      std::cout << "infeasible " << std::endl;
      exit(0);
    }

    return outOpt;
  }

// [[Rcpp::export]]
Eigen::VectorXd getMidPriceQ(const Eigen::VectorXd& sp,
                             const Eigen::VectorXd& bid,
                             const Eigen::VectorXd& ask,
                             const Eigen::VectorXd& strike,
                             const Eigen::Matrix<bool, Eigen::Dynamic, 1>& pFlag,
                             double spotsP,
                             double spbid,
                             double spask) {

  M_Model M = new mosek::fusion::Model("MidPriceQ");
  auto _M = monty::finally([&]() {
    M->dispose();
  });

  const double SCALER(sp.size());

  Eigen::VectorXd lb = bid.cwiseQuotient(0.5 * (bid + ask) * spotsP) * SCALER;
  Eigen::VectorXd ub = ask.cwiseQuotient(0.5 * (bid + ask) * spotsP) * SCALER;

  unsigned int OPTLEN(strike.size());
  unsigned int LEN(sp.size());

  Eigen::VectorXd payoffMat(OPTLEN * LEN);

  // Fill the payoff matrix
  for (size_t i = 0; i < OPTLEN; ++i) {
    for (size_t j = 0; j < LEN; ++j) {
      payoffMat[i * LEN + j] = otm_payoff(sp[j], strike[i], pFlag[i]) / (0.5 * (bid[i] + ask[i]) * spotsP);
    }
  }

  const M_Matrix::t payoff_wrap = M_Matrix::dense(std::shared_ptr<M_ndarray_2>(new M_ndarray_2(payoffMat.data(), monty::shape(OPTLEN, LEN))));

  M_Variable::t q_vars = M->variable("q_vars", LEN, M_Domain::inRange(0.0, SCALER));
  auto lb_wrap = std::shared_ptr<M_ndarray_1>(new M_ndarray_1(lb.data(), monty::shape(OPTLEN)));
  auto ub_wrap = std::shared_ptr<M_ndarray_1>(new M_ndarray_1(ub.data(), monty::shape(OPTLEN)));
  M_Variable::t options = M->variable("optVars", OPTLEN, M_Domain::inRange(lb_wrap, ub_wrap));
  M->constraint(M_Expr::sub(M_Expr::mul(payoff_wrap, q_vars), options), M_Domain::equalsTo(0.0));

  M_Variable::t uu1 = M->variable(M_Domain::greaterThan(0.0));
  M_Variable::t uu2 = M->variable(M_Domain::greaterThan(0.0));

  // Distance between lb and option (squared), then minimized
  M->constraint("uu1", M_Expr::vstack(0.5, uu1, M_Expr::sub(options, lb_wrap)), M_Domain::inRotatedQCone()); // quadratic cone for objective function
  // Distance between ub and option (squared), then minimized
  M->constraint("uu2", M_Expr::vstack(0.5, uu2, M_Expr::sub(options, ub_wrap)), M_Domain::inRotatedQCone()); // quadratic cone for objective function

  M->constraint(M_Expr::sum(q_vars), M_Domain::equalsTo(SCALER));
  auto sp_wrap = std::shared_ptr<M_ndarray_1>(new M_ndarray_1(LEN, std::function<double(ptrdiff_t)>([&](ptrdiff_t i) { return sp(i); })));

  // Forward pricing constraints
  M->constraint(M_Expr::mul(1.0 / spotsP, M_Expr::dot(sp_wrap, q_vars)), M_Domain::inRange(SCALER * spbid / spotsP, SCALER * spask / spotsP));

  M->objective(mosek::fusion::ObjectiveSense::Minimize, M_Expr::add(uu1, uu2));
  M->solve();
  Eigen::VectorXd outOpt;
  if (M->getPrimalSolutionStatus() == mosek::fusion::SolutionStatus::Optimal) {
    auto sol = q_vars->level();
    const Eigen::Map<Eigen::VectorXd> solWrap(sol->raw(), LEN);
    outOpt = solWrap / SCALER;
  } else {
    std::cout << "infeasible " << std::endl;
    exit(0);
  }
  return outOpt;
}

// [[Rcpp::export]]
Eigen::VectorXd getMidPriceQReg(const Eigen::VectorXd& sp,
                                const Eigen::VectorXd& bid,
                                const Eigen::VectorXd& ask,
                                const Eigen::VectorXd& strike,
                                const Eigen::Matrix<bool, Eigen::Dynamic, 1>& pFlag,
                                double spotsP,
                                double spbid,
                                double spask) {

  M_Model M = new mosek::fusion::Model("MidPriceQReg");
  auto _M = monty::finally([&]() {
    M->dispose();
  });

  const double SCALER(sp.size());

  Eigen::VectorXd lb = bid.cwiseQuotient(0.5 * (bid + ask) * spotsP) * SCALER;
  Eigen::VectorXd ub = ask.cwiseQuotient(0.5 * (bid + ask) * spotsP) * SCALER;

  unsigned int OPTLEN(strike.size());
  unsigned int LEN(sp.size());

  Eigen::VectorXd payoffMat(OPTLEN * LEN);

  // Fill the payoff matrix
  for (size_t i = 0; i < OPTLEN; ++i) {
    for (size_t j = 0; j < LEN; ++j) {
      payoffMat[i * LEN + j] = otm_payoff(sp[j], strike[i], pFlag[i]) / (0.5 * (bid[i] + ask[i]) * spotsP);
    }
  }

  const M_Matrix::t payoff_wrap = M_Matrix::dense(std::shared_ptr<M_ndarray_2>(new M_ndarray_2(payoffMat.data(), monty::shape(OPTLEN, LEN))));

  M_Variable::t q_vars = M->variable("q_vars", LEN, M_Domain::inRange(0.0, SCALER));
  auto lb_wrap = std::shared_ptr<M_ndarray_1>(new M_ndarray_1(lb.data(), monty::shape(OPTLEN)));
  auto ub_wrap = std::shared_ptr<M_ndarray_1>(new M_ndarray_1(ub.data(), monty::shape(OPTLEN)));
  M_Variable::t options = M->variable("optVars", OPTLEN, M_Domain::inRange(lb_wrap, ub_wrap));
  M->constraint(M_Expr::sub(M_Expr::mul(payoff_wrap, q_vars), options), M_Domain::equalsTo(0.0));

  M_Variable::t uu1 = M->variable(M_Domain::greaterThan(0.0));

  // Regularization term
  M->constraint("uu1", M_Expr::vstack(0.5, uu1, M_Expr::sub(options, lb_wrap)), M_Domain::inRotatedQCone()); // quadratic cone for objective function
  M->constraint("uu2", M_Expr::vstack(0.5, uu1, M_Expr::sub(options, ub_wrap)), M_Domain::inRotatedQCone()); // quadratic cone for objective function

  M->constraint(M_Expr::sum(q_vars), M_Domain::equalsTo(SCALER));
  auto sp_wrap = std::shared_ptr<M_ndarray_1>(new M_ndarray_1(LEN, std::function<double(ptrdiff_t)>([&](ptrdiff_t i) { return sp(i); })));

  // Forward pricing constraints
  M->constraint(M_Expr::mul(1.0 / spotsP, M_Expr::dot(sp_wrap, q_vars)), M_Domain::inRange(SCALER * spbid / spotsP, SCALER * spask / spotsP));

  M->objective(mosek::fusion::ObjectiveSense::Minimize, M_Expr::sum(uu1));
  M->solve();
  Eigen::VectorXd outOpt;
  if (M->getPrimalSolutionStatus() == mosek::fusion::SolutionStatus::Optimal) {
    auto sol = q_vars->level();
    const Eigen::Map<Eigen::VectorXd> solWrap(sol->raw(), LEN);
    outOpt = solWrap / SCALER;
  } else {
    std::cout << "infeasible " << std::endl;
    exit(0);
  }
  return outOpt;
}

// [[Rcpp::export]]
Eigen::VectorXd getQReg(const Eigen::VectorXd& sp,
                        const Eigen::VectorXd& bid,
                        const Eigen::VectorXd& ask,
                        const Eigen::VectorXd& strike,
                        const Eigen::Matrix<bool, Eigen::Dynamic, 1>& pFlag,
                        double spotsP,
                        double spbid,
                        double spask) {

  M_Model M = new mosek::fusion::Model("QReg");
  auto _M = monty::finally([&]() {
    M->dispose();
  });

  const double SCALER(sp.size());

  Eigen::VectorXd lb = bid.cwiseQuotient(0.5 * (bid + ask) * spotsP) * SCALER;
  Eigen::VectorXd ub = ask.cwiseQuotient(0.5 * (bid + ask) * spotsP) * SCALER;

  unsigned int OPTLEN(strike.size());
  unsigned int LEN(sp.size());

  Eigen::VectorXd payoffMat(OPTLEN * LEN);

  // Fill the payoff matrix
  for (size_t i = 0; i < OPTLEN; ++i) {
    for (size_t j = 0; j < LEN; ++j) {
      payoffMat[i * LEN + j] = otm_payoff(sp[j], strike[i], pFlag[i]) / (0.5 * (bid[i] + ask[i]) * spotsP);
    }
  }

  const M_Matrix::t payoff_wrap = M_Matrix::dense(std::shared_ptr<M_ndarray_2>(new M_ndarray_2(payoffMat.data(), monty::shape(OPTLEN, LEN))));

  M_Variable::t q_vars = M->variable("q_vars", LEN, M_Domain::inRange(0.0, SCALER));
  auto lb_wrap = std::shared_ptr<M_ndarray_1>(new M_ndarray_1(lb.data(), monty::shape(OPTLEN)));
  auto ub_wrap = std::shared_ptr<M_ndarray_1>(new M_ndarray_1(ub.data(), monty::shape(OPTLEN)));
  M_Variable::t options = M->variable("optVars", OPTLEN, M_Domain::inRange(lb_wrap, ub_wrap));
  M->constraint(M_Expr::sub(M_Expr::mul(payoff_wrap, q_vars), options), M_Domain::equalsTo(0.0));

  M_Variable::t uu = M->variable(M_Domain::greaterThan(0.0));

  // Regularization term
  M->constraint("uu", M_Expr::vstack(0.5, uu, M_Expr::sub(options, lb_wrap)), M_Domain::inRotatedQCone());

  M->constraint(M_Expr::sum(q_vars), M_Domain::equalsTo(SCALER));
  auto sp_wrap = std::shared_ptr<M_ndarray_1>(new M_ndarray_1(LEN, std::function<double(ptrdiff_t)>([&](ptrdiff_t i) { return sp(i); })));

  // Forward pricing constraints
  M->constraint(M_Expr::mul(1.0 / spotsP, M_Expr::dot(sp_wrap, q_vars)), M_Domain::inRange(SCALER * spbid / spotsP, SCALER * spask / spotsP));

  M->objective(mosek::fusion::ObjectiveSense::Minimize, uu);
  M->solve();
  Eigen::VectorXd outOpt;
  if (M->getPrimalSolutionStatus() == mosek::fusion::SolutionStatus::Optimal) {
    auto sol = q_vars->level();
    const Eigen::Map<Eigen::VectorXd> solWrap(sol->raw(), LEN);
    outOpt = solWrap / SCALER;
  } else {
    std::cout << "infeasible " << std::endl;
    exit(0);
  }
  return outOpt;
}
