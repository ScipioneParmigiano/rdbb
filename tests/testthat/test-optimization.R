library(testthat)
library(rdbb)

test_that("performOptimization", {
  n <- 4
  alpha <- 1.3
  lambda <- 1.1
  omega_l <- c(0, 3) # zero-indexed
  sp <- c(1200, 1250, 1300, 1350)
  strike <- c(1290, 1295, 1295, 1300)
  bid <- c(27.7, 27.4, 29.4, 25.0)
  ask <- c(29.3, 29.7, 31.4, 26.9)
  pFlag <- c(TRUE, FALSE, TRUE, FALSE)
  
  result <- performOptimization(n, alpha, lambda, omega_l, sp, strike, bid, ask, pFlag)
  p <- result[[1]]
  q <- result[[2]]
  
  exp_p <- c(0.429653, 0.0304595, 0.0442842, 0.495604)
  exp_q <- c(0.325555, 2.34516e-07, 0.147889, 0.526556)
  
  for (i in seq_along(p)) {
    expect_equal(p[i], exp_p[i], tolerance = 1e-5)
    expect_equal(q[i], exp_q[i], tolerance = 1e-5)
  }
})