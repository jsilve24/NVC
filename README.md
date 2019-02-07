# NVC
NVC: Normal Variance Components

Given `y`, `mu`, `U_1`, ..., `U_p` fits the following model to estimate `ell_1`,...,`ell_p`. 

`y ~ N(mu, e^{ell_1} U_1 + ... + e^{ell_p} U_p )`

* Key Functionality implemented in C++ using RcppEigen for scalability. 
* Optimization Performed using L-BFGS implementaion wrapped by RcppNumerical. 

# Example Use
```
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
  
  > fit
  $LogLik
  [1] -4971.439

  $ell
  [1] 0.01796315 2.97466161
```
