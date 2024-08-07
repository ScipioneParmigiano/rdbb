library(testthat)
library(rdbb)
library(data.table)

test_that("getFeasibleOptionFlags", {
  
  fw_data <- fread("forward1.txt", header = FALSE)
  data <- fread("spopt1.txt", header = FALSE)
  
  strike_list <- data$V1
  pFlag_list <- data$V2
  bid_list <- data$V3
  ask_list <- data$V4
  
  min_strike <- min(strike_list)
  max_strike <- max(strike_list)
  
  sp <- c(seq(0, min_strike, by = 10), seq(min_strike, max_strike, by = 5), seq(max_strike, 4440, by = 10))
  
  sp_np <- as.numeric(sp)
  bid_np <- as.numeric(bid_list)
  ask_np <- as.numeric(ask_list)
  strike_np <- as.numeric(strike_list)
  pFlag_np <- as.logical(pFlag_list)
  fw <- fw_data$V1[1]
  
  feasible_options <- getFeasibleOptionFlags(sp_np, bid_np, ask_np, strike_np, pFlag_np, fw, fw - 0.02, fw + 0.02)
  false_idx <- which(feasible_options == FALSE)
  
  expect_equal(false_idx, c(100, 125, 144, 146))
})

test_that("getMidPriceQ", {
  
  fw_data <- fread("forward1.txt", header = FALSE)
  data <- fread("spopt1.txt", header = FALSE)
  
  strike_list <- data$V1
  pFlag_list <- data$V2
  bid_list <- data$V3
  ask_list <- data$V4
  
  min_strike <- min(strike_list)
  max_strike <- max(strike_list)
  
  sp <- c(seq(0, min_strike, by = 10), seq(min_strike, max_strike, by = 5), seq(max_strike, 4440, by = 10))
  
  sp_np <- as.numeric(sp)
  bid_np <- as.numeric(bid_list)
  ask_np <- as.numeric(ask_list)
  strike_np <- as.numeric(strike_list)
  pFlag_np <- as.logical(pFlag_list)
  fw <- fw_data$V1[1]
  
  feasible_options <- getFeasibleOptionFlags(sp_np, bid_np, ask_np, strike_np, pFlag_np, fw, fw - 0.02, fw + 0.02)
  q <- getMidPriceQ(sp_np, bid_np[feasible_options], ask_np[feasible_options], strike_np[feasible_options], pFlag_np[feasible_options], fw, fw - 0.02, fw + 0.02)
  
  expect_equal(sum(q), 1)
  expect_lt(abs(sum(q * sp_np) - fw), 0.02)
})

test_that("getMidPriceQReg", {
  
  fw_data <- fread("forward1.txt", header = FALSE)
  data <- fread("spopt1.txt", header = FALSE)
  
  strike_list <- data$V1
  pFlag_list <- data$V2
  bid_list <- data$V3
  ask_list <- data$V4
  
  min_strike <- min(strike_list)
  max_strike <- max(strike_list)
  
  sp <- c(seq(0, min_strike, by = 10), seq(min_strike, max_strike, by = 5), seq(max_strike, 4440, by = 10))
  
  sp_np <- as.numeric(sp)
  bid_np <- as.numeric(bid_list)
  ask_np <- as.numeric(ask_list)
  strike_np <- as.numeric(strike_list)
  pFlag_np <- as.logical(pFlag_list)
  fw <- fw_data$V1[1]
  
  feasible_options <- getFeasibleOptionFlags(sp_np, bid_np, ask_np, strike_np, pFlag_np, fw, fw - 0.02, fw + 0.02)
  q <- getMidPriceQReg(sp_np, bid_np[feasible_options], ask_np[feasible_options], strike_np[feasible_options], pFlag_np[feasible_options], fw, fw - 0.02, fw + 0.02)
  
  expect_equal(sum(q), 1)
  expect_lt(abs(sum(q * sp_np) - fw), 0.02)
})

test_that("getQReg", {
  
  fw_data <- fread("forward1.txt", header = FALSE)
  data <- fread("spopt1.txt", header = FALSE)
  
  strike_list <- data$V1
  pFlag_list <- data$V2
  bid_list <- data$V3
  ask_list <- data$V4
  
  min_strike <- min(strike_list)
  max_strike <- max(strike_list)
  
  sp <- c(seq(0, min_strike, by = 10), seq(min_strike, max_strike, by = 5), seq(max_strike, 4440, by = 10))
  
  sp_np <- as.numeric(sp)
  bid_np <- as.numeric(bid_list)
  ask_np <- as.numeric(ask_list)
  strike_np <- as.numeric(strike_list)
  pFlag_np <- as.logical(pFlag_list)
  fw <- fw_data$V1[1]
  
  feasible_options <- getFeasibleOptionFlags(sp_np, bid_np, ask_np, strike_np, pFlag_np, fw, fw - 0.02, fw + 0.02)
  q <- getQReg(sp_np, bid_np[feasible_options], ask_np[feasible_options], strike_np[feasible_options], pFlag_np[feasible_options], fw, fw - 0.02, fw + 0.02)
  
  expect_equal(sum(q), 1)
  expect_lt(abs(sum(q * sp_np) - fw), 0.02)
})
