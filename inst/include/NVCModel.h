#ifndef NVC_MODEL_H
#define NVC_MODEL_H

#define EIGEN_NO_AUTOMATIC_RESIZING

#include <RcppNumerical.h>

using namespace Rcpp;
using Eigen::Map;
using Eigen::MatrixXd;
using Eigen::ArrayXXd;
using Eigen::VectorXd;
using Eigen::Ref;

namespace NCV {

class NVCModel : public Numer::MFuncGrad
{
private: 
  const VectorXd y; // N vector
  const VectorXd mu;
  const MatrixXd U; // PQ x Q matrix of concatenated VCs
  int N;
  int Q;
  int P;
  VectorXd e;
  
public:
  NVCModel(const VectorXd y_,  const VectorXd mu_, const MatrixXd U_) : 
  y(y_), U(U_), mu(mu_)
  {
    N = y.rows();
    Q = U.cols();
    P = U.rows()/Q;
    e = y-mu;
  }
  ~NVCModel(){}
  
  virtual double f_grad(Numer::Constvec& pars, Numer::Refvec grad){
    MatrixXd Sigma = MatrixXd::Zero(Q, Q);
    for (int i=0; i<P; i++){
      Sigma += exp(pars(i))*U.middleRows(i*Q, Q);
    }
    Eigen::LLT<MatrixXd> L(Sigma);
    MatrixXd SigmaInv = L.solve(MatrixXd::Identity(Q,Q));
    double logdetSigma = 2.0*L.matrixLLT().diagonal().array().log().sum();
    
    // Gradient
    for (int i=0; i<P; i++){
      grad(i) = (SigmaInv.array()*U.middleRows(i*Q, Q).array()).sum();
      grad(i) -= e.transpose()*SigmaInv*U.middleRows(i*Q, Q)*SigmaInv*e;
      grad(i) *= 0.5*exp(pars(i));
    }
    // Rcout << "ell: " << pars << std::endl;
    // Rcout << "grad: " << grad << std::endl;
    // Rcout << "loglik: " << 0.5*(logdetSigma + e.transpose()*SigmaInv*e) << std::endl;
    // Rcout << std::endl;
    
    return 0.5*(logdetSigma + e.transpose()*SigmaInv*e);
  }
  
};

}




#endif 
