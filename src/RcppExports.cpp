// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <RcppEigen.h>
#include <Rcpp.h>

using namespace Rcpp;

#ifdef RCPP_USE_GLOBAL_ROSTREAM
Rcpp::Rostream<true>&  Rcpp::Rcout = Rcpp::Rcpp_cout_get();
Rcpp::Rostream<false>& Rcpp::Rcerr = Rcpp::Rcpp_cerr_get();
#endif

// dot_product
double dot_product(const Eigen::VectorXd& x, const Eigen::VectorXd& y);
RcppExport SEXP _rdbb_dot_product(SEXP xSEXP, SEXP ySEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const Eigen::VectorXd& >::type x(xSEXP);
    Rcpp::traits::input_parameter< const Eigen::VectorXd& >::type y(ySEXP);
    rcpp_result_gen = Rcpp::wrap(dot_product(x, y));
    return rcpp_result_gen;
END_RCPP
}
// performOptimization
std::array<std::vector<double>, 2> performOptimization(int n, double alpha, double lambda, const Eigen::VectorXd& omega_l_eigen, const Eigen::VectorXd& sp_eigen, const Eigen::VectorXd& strike_eigen, const Eigen::VectorXd& bid_eigen, const Eigen::VectorXd& ask_eigen, const Eigen::Matrix<bool, Eigen::Dynamic, 1>& pFlag_eigen);
RcppExport SEXP _rdbb_performOptimization(SEXP nSEXP, SEXP alphaSEXP, SEXP lambdaSEXP, SEXP omega_l_eigenSEXP, SEXP sp_eigenSEXP, SEXP strike_eigenSEXP, SEXP bid_eigenSEXP, SEXP ask_eigenSEXP, SEXP pFlag_eigenSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< int >::type n(nSEXP);
    Rcpp::traits::input_parameter< double >::type alpha(alphaSEXP);
    Rcpp::traits::input_parameter< double >::type lambda(lambdaSEXP);
    Rcpp::traits::input_parameter< const Eigen::VectorXd& >::type omega_l_eigen(omega_l_eigenSEXP);
    Rcpp::traits::input_parameter< const Eigen::VectorXd& >::type sp_eigen(sp_eigenSEXP);
    Rcpp::traits::input_parameter< const Eigen::VectorXd& >::type strike_eigen(strike_eigenSEXP);
    Rcpp::traits::input_parameter< const Eigen::VectorXd& >::type bid_eigen(bid_eigenSEXP);
    Rcpp::traits::input_parameter< const Eigen::VectorXd& >::type ask_eigen(ask_eigenSEXP);
    Rcpp::traits::input_parameter< const Eigen::Matrix<bool, Eigen::Dynamic, 1>& >::type pFlag_eigen(pFlag_eigenSEXP);
    rcpp_result_gen = Rcpp::wrap(performOptimization(n, alpha, lambda, omega_l_eigen, sp_eigen, strike_eigen, bid_eigen, ask_eigen, pFlag_eigen));
    return rcpp_result_gen;
END_RCPP
}
// getFeasibleOptionFlags
Eigen::Matrix<bool, Eigen::Dynamic, 1> getFeasibleOptionFlags(const Eigen::VectorXd& sp, const Eigen::VectorXd& bid, const Eigen::VectorXd& ask, const Eigen::VectorXd& strike, const Eigen::Matrix<bool, Eigen::Dynamic, 1>& pFlag, double spotsP, double spbid, double spask);
RcppExport SEXP _rdbb_getFeasibleOptionFlags(SEXP spSEXP, SEXP bidSEXP, SEXP askSEXP, SEXP strikeSEXP, SEXP pFlagSEXP, SEXP spotsPSEXP, SEXP spbidSEXP, SEXP spaskSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const Eigen::VectorXd& >::type sp(spSEXP);
    Rcpp::traits::input_parameter< const Eigen::VectorXd& >::type bid(bidSEXP);
    Rcpp::traits::input_parameter< const Eigen::VectorXd& >::type ask(askSEXP);
    Rcpp::traits::input_parameter< const Eigen::VectorXd& >::type strike(strikeSEXP);
    Rcpp::traits::input_parameter< const Eigen::Matrix<bool, Eigen::Dynamic, 1>& >::type pFlag(pFlagSEXP);
    Rcpp::traits::input_parameter< double >::type spotsP(spotsPSEXP);
    Rcpp::traits::input_parameter< double >::type spbid(spbidSEXP);
    Rcpp::traits::input_parameter< double >::type spask(spaskSEXP);
    rcpp_result_gen = Rcpp::wrap(getFeasibleOptionFlags(sp, bid, ask, strike, pFlag, spotsP, spbid, spask));
    return rcpp_result_gen;
END_RCPP
}
// getMidPriceQ
Eigen::VectorXd getMidPriceQ(const Eigen::VectorXd& sp, const Eigen::VectorXd& bid, const Eigen::VectorXd& ask, const Eigen::VectorXd& strike, const Eigen::Matrix<bool, Eigen::Dynamic, 1>& pFlag, double spotsP, double spbid, double spask);
RcppExport SEXP _rdbb_getMidPriceQ(SEXP spSEXP, SEXP bidSEXP, SEXP askSEXP, SEXP strikeSEXP, SEXP pFlagSEXP, SEXP spotsPSEXP, SEXP spbidSEXP, SEXP spaskSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const Eigen::VectorXd& >::type sp(spSEXP);
    Rcpp::traits::input_parameter< const Eigen::VectorXd& >::type bid(bidSEXP);
    Rcpp::traits::input_parameter< const Eigen::VectorXd& >::type ask(askSEXP);
    Rcpp::traits::input_parameter< const Eigen::VectorXd& >::type strike(strikeSEXP);
    Rcpp::traits::input_parameter< const Eigen::Matrix<bool, Eigen::Dynamic, 1>& >::type pFlag(pFlagSEXP);
    Rcpp::traits::input_parameter< double >::type spotsP(spotsPSEXP);
    Rcpp::traits::input_parameter< double >::type spbid(spbidSEXP);
    Rcpp::traits::input_parameter< double >::type spask(spaskSEXP);
    rcpp_result_gen = Rcpp::wrap(getMidPriceQ(sp, bid, ask, strike, pFlag, spotsP, spbid, spask));
    return rcpp_result_gen;
END_RCPP
}
// getMidPriceQReg
Eigen::VectorXd getMidPriceQReg(const Eigen::VectorXd& sp, const Eigen::VectorXd& bid, const Eigen::VectorXd& ask, const Eigen::VectorXd& strike, const Eigen::Matrix<bool, Eigen::Dynamic, 1>& pFlag, double spotsP, double spbid, double spask);
RcppExport SEXP _rdbb_getMidPriceQReg(SEXP spSEXP, SEXP bidSEXP, SEXP askSEXP, SEXP strikeSEXP, SEXP pFlagSEXP, SEXP spotsPSEXP, SEXP spbidSEXP, SEXP spaskSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const Eigen::VectorXd& >::type sp(spSEXP);
    Rcpp::traits::input_parameter< const Eigen::VectorXd& >::type bid(bidSEXP);
    Rcpp::traits::input_parameter< const Eigen::VectorXd& >::type ask(askSEXP);
    Rcpp::traits::input_parameter< const Eigen::VectorXd& >::type strike(strikeSEXP);
    Rcpp::traits::input_parameter< const Eigen::Matrix<bool, Eigen::Dynamic, 1>& >::type pFlag(pFlagSEXP);
    Rcpp::traits::input_parameter< double >::type spotsP(spotsPSEXP);
    Rcpp::traits::input_parameter< double >::type spbid(spbidSEXP);
    Rcpp::traits::input_parameter< double >::type spask(spaskSEXP);
    rcpp_result_gen = Rcpp::wrap(getMidPriceQReg(sp, bid, ask, strike, pFlag, spotsP, spbid, spask));
    return rcpp_result_gen;
END_RCPP
}
// getQReg
Eigen::VectorXd getQReg(const Eigen::VectorXd& sp, const Eigen::VectorXd& bid, const Eigen::VectorXd& ask, const Eigen::VectorXd& strike, const Eigen::Matrix<bool, Eigen::Dynamic, 1>& pFlag, double spotsP, double spbid, double spask);
RcppExport SEXP _rdbb_getQReg(SEXP spSEXP, SEXP bidSEXP, SEXP askSEXP, SEXP strikeSEXP, SEXP pFlagSEXP, SEXP spotsPSEXP, SEXP spbidSEXP, SEXP spaskSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const Eigen::VectorXd& >::type sp(spSEXP);
    Rcpp::traits::input_parameter< const Eigen::VectorXd& >::type bid(bidSEXP);
    Rcpp::traits::input_parameter< const Eigen::VectorXd& >::type ask(askSEXP);
    Rcpp::traits::input_parameter< const Eigen::VectorXd& >::type strike(strikeSEXP);
    Rcpp::traits::input_parameter< const Eigen::Matrix<bool, Eigen::Dynamic, 1>& >::type pFlag(pFlagSEXP);
    Rcpp::traits::input_parameter< double >::type spotsP(spotsPSEXP);
    Rcpp::traits::input_parameter< double >::type spbid(spbidSEXP);
    Rcpp::traits::input_parameter< double >::type spask(spaskSEXP);
    rcpp_result_gen = Rcpp::wrap(getQReg(sp, bid, ask, strike, pFlag, spotsP, spbid, spask));
    return rcpp_result_gen;
END_RCPP
}
// rcppeigen_hello_world
Eigen::MatrixXd rcppeigen_hello_world();
RcppExport SEXP _rdbb_rcppeigen_hello_world() {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    rcpp_result_gen = Rcpp::wrap(rcppeigen_hello_world());
    return rcpp_result_gen;
END_RCPP
}
// rcppeigen_outerproduct
Eigen::MatrixXd rcppeigen_outerproduct(const Eigen::VectorXd& x);
RcppExport SEXP _rdbb_rcppeigen_outerproduct(SEXP xSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const Eigen::VectorXd& >::type x(xSEXP);
    rcpp_result_gen = Rcpp::wrap(rcppeigen_outerproduct(x));
    return rcpp_result_gen;
END_RCPP
}
// rcppeigen_innerproduct
double rcppeigen_innerproduct(const Eigen::VectorXd& x);
RcppExport SEXP _rdbb_rcppeigen_innerproduct(SEXP xSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const Eigen::VectorXd& >::type x(xSEXP);
    rcpp_result_gen = Rcpp::wrap(rcppeigen_innerproduct(x));
    return rcpp_result_gen;
END_RCPP
}
// rcppeigen_bothproducts
Rcpp::List rcppeigen_bothproducts(const Eigen::VectorXd& x);
RcppExport SEXP _rdbb_rcppeigen_bothproducts(SEXP xSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const Eigen::VectorXd& >::type x(xSEXP);
    rcpp_result_gen = Rcpp::wrap(rcppeigen_bothproducts(x));
    return rcpp_result_gen;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_rdbb_dot_product", (DL_FUNC) &_rdbb_dot_product, 2},
    {"_rdbb_performOptimization", (DL_FUNC) &_rdbb_performOptimization, 9},
    {"_rdbb_getFeasibleOptionFlags", (DL_FUNC) &_rdbb_getFeasibleOptionFlags, 8},
    {"_rdbb_getMidPriceQ", (DL_FUNC) &_rdbb_getMidPriceQ, 8},
    {"_rdbb_getMidPriceQReg", (DL_FUNC) &_rdbb_getMidPriceQReg, 8},
    {"_rdbb_getQReg", (DL_FUNC) &_rdbb_getQReg, 8},
    {"_rdbb_rcppeigen_hello_world", (DL_FUNC) &_rdbb_rcppeigen_hello_world, 0},
    {"_rdbb_rcppeigen_outerproduct", (DL_FUNC) &_rdbb_rcppeigen_outerproduct, 1},
    {"_rdbb_rcppeigen_innerproduct", (DL_FUNC) &_rdbb_rcppeigen_innerproduct, 1},
    {"_rdbb_rcppeigen_bothproducts", (DL_FUNC) &_rdbb_rcppeigen_bothproducts, 1},
    {NULL, NULL, 0}
};

RcppExport void R_init_rdbb(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}