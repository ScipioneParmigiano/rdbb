#include <RcppEigen.h>
#include "fusion.h" 
// [[Rcpp::depends(RcppEigen)]]
// [[Rcpp::export]]
double dot_product(const Eigen::VectorXd &x, const Eigen::VectorXd &y) {
  if (x.size() != y.size()) {
    Rcpp::stop("Vectors must be of the same length.");
  }
  return x.dot(y);
}
