library(testthat)
context("NVC")

set.seed(8403)

test_that("NVC Correctness on Absolutely", {
  N <- 1000
  I <- diag(N)
  U1 <- diag(N)
  U2 <-  rWishart(1, N+10, diag(N))[,,1]
  ell <- c(0.0001, 3)
  Sigma <- exp(ell[1])*U1 + exp(ell[2])*U2
  L <- t(chol(Sigma))
  mu <- rep(4, N)
  y <- mu + L %*% rnorm(N) 
  
  fit <- optimNVC(y, mu, rbind(U1, U2), c(0,0))
  expect_equal(fit$ell, ell, tolerance=.1)
})


test_that("NVC Correctness on Rejecting LowRank", {
  N <- 1000
  I <- diag(N)
  U1 <- diag(N)
  U4 <- tcrossprod(rnorm(N))
  U5 <- tcrossprod(rnorm(N))
  ell <- c(1, 1)
  Sigma <- exp(ell[1])*U1 + exp(ell[2])*U4
  L <- t(chol(Sigma))
  mu <- rep(4, N)
  y <- mu + L %*% rnorm(N) 
  
  fit <- optimNVC(y, mu, rbind(U1, U5), c(0,0))
  expect_true(fit$ell[1] > 1)
  expect_true(fit$ell[2] < 0)
})

test_that("NVC Correctness on LowRank", {
  N <- 100
  I <- diag(N)
  U1 <- diag(N)
  U4 <- tcrossprod(rnorm(N))
  U5 <- tcrossprod(rnorm(N))
  ell <- c(1, 1)
  Sigma <- exp(ell[1])*U1 + exp(ell[2])*U4
  L <- t(chol(Sigma))
  mu <- rep(4, N)
  y <- mu + L %*% rnorm(N)

  fit <- optimNVC(y, mu, rbind(U1, U4), c(0,0));
  expect_equal(exp(fit$ell), exp(ell), .1)
})

test_that("NVC Correctness on Trivial", {
  N <- 1000
  I <- diag(N)
  U1 <- diag(N)
  U4 <- tcrossprod(rnorm(N))
  U5 <- tcrossprod(rnorm(N))
  ell <- c(1)
  Sigma <- exp(ell[1])*U1
  L <- t(chol(Sigma))
  mu <- rep(4, N)
  y <- mu + L %*% rnorm(N)
  
  fit <- optimNVC(y, mu, rbind(U1), c(0));
  expect_equal(exp(fit$ell), exp(ell), .1)
})

test_that("NVC Correctness on Trivial 2", {
  N <- 1000
  I <- diag(N)
  U1 <- diag(N)
  U2 <-  rWishart(1, N+10, diag(N))[,,1]
  ell <- c(3)
  Sigma <- exp(ell[1])*U2 
  L <- t(chol(Sigma))
  mu <- rep(4, N)
  y <- mu + L %*% rnorm(N) 
  
  fit <- optimNVC(y, mu, rbind(U2), c(0))
  expect_equal(fit$ell, ell, tolerance=.1)
})




