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
//' @details Fits the following model (for unknown ell)
//'   \deqn{y ~ N(mu, e^{ell_1}U_1 + ...+ e^{ell_P}U_1)}
//' using L-BFGS optimization implemented in optimized C++ code. 
//' @return List: LogLik is log likelihood at optimima. Pars argument is ells. 
//' @export
//' @md
// [[Rcpp::export]]
List optimNVC(const Eigen::VectorXd y, const Eigen::VectorXd mu, 
              const Eigen::MatrixXd U, Eigen::VectorXd ellinit){
  int N = y.rows();
  int Q = U.cols();
  int P = U.rows()/Q;
  
  NCV::NVCModel mod(y, mu, U);
  VectorXd ell = ellinit;
  
  double nllopt; 
  int status = Numer::optim_lbfgs(mod, ell, nllopt);
  
  List out(2);
  out.names() = CharacterVector::create("LogLik", "ell");
  if (status<0)
    Rcpp::warning("Max Iterations Hit, May not be at optima");
  
  out[0] = -(P*0.5*0.79817986+ nllopt);
  out[1] = ell;
  return out;
}
