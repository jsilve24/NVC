#include <NVCModel.h>

// [[Rcpp::depends(RcppNumerical)]]

using namespace Rcpp;
using Eigen::Map;
using Eigen::MatrixXd;
using Eigen::ArrayXXd;
using Eigen::VectorXd;

//' Function to optimize the NVCModel 
//' @param y N vector of observations
//' @param mu mean vector (N dimensional)
//' @param U PQ x Q matrix of P Variance Components
//' @param ellinit P vector of initialization values for ell
//' @param max_iter maximum iterations for optimization
//' @param eps_f optimization stopping threshold for log-likelihood improvement 
//' @param eps_g optimization stopping threshold for gradient 
//' @details Fits the following model (for unknown ell)
//'   \deqn{y ~ N(mu, e^{ell_1}U_1 + ...+ e^{ell_P}U_1)}
//' using L-BFGS optimization implemented in optimized C++ code. 
//' @return List: LogLik is log likelihood at optimima. Pars argument is ells. 
//' @export
//' @md
//' @examples
//'   N <- 1000
//'   I <- diag(N)
//'   U1 <- diag(N)
//'   U2 <-  rWishart(1, N+10, diag(N))[,,1]
//'   ell <- c(0.0001, 3)
//'   Sigma <- exp(ell[1])*U1 + exp(ell[2])*U2
//'   L <- t(chol(Sigma))
//'   mu <- rep(4, N)
//'   y <- mu + L %*% rnorm(N) 
//'   fit <- optimNVC(y, mu, rbind(U1, U2), c(0,0))
// [[Rcpp::export]]
List optimNVC(const Eigen::VectorXd y, const Eigen::VectorXd mu, 
              const Eigen::MatrixXd U, Eigen::VectorXd ellinit, 
              const int max_iter=10000, const double eps_f = 1e-9, const double eps_g = 1e-6){
  int N = y.rows();
  int Q = U.cols();
  int P = U.rows()/Q;
  
  NCV::NVCModel mod(y, mu, U);
  VectorXd ell = ellinit;
  
  double nllopt; 
  int status = Numer::optim_lbfgs(mod, ell, nllopt, max_iter, eps_f, eps_g);
  
  List out(2);
  out.names() = CharacterVector::create("LogLik", "ell");
  if (status<0)
    Rcpp::warning("Max Iterations Hit, May not be at optima");
  
  out[0] = -(P*0.5*0.79817986+ nllopt);
  out[1] = ell;
  return out;
}
