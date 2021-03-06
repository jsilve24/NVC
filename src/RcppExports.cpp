// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <RcppEigen.h>
#include <Rcpp.h>

using namespace Rcpp;

// optimNVC
List optimNVC(const Eigen::VectorXd y, const Eigen::VectorXd mu, const Eigen::MatrixXd U, Eigen::VectorXd ellinit, const int max_iter, const double eps_f, const double eps_g);
RcppExport SEXP _NVC_optimNVC(SEXP ySEXP, SEXP muSEXP, SEXP USEXP, SEXP ellinitSEXP, SEXP max_iterSEXP, SEXP eps_fSEXP, SEXP eps_gSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const Eigen::VectorXd >::type y(ySEXP);
    Rcpp::traits::input_parameter< const Eigen::VectorXd >::type mu(muSEXP);
    Rcpp::traits::input_parameter< const Eigen::MatrixXd >::type U(USEXP);
    Rcpp::traits::input_parameter< Eigen::VectorXd >::type ellinit(ellinitSEXP);
    Rcpp::traits::input_parameter< const int >::type max_iter(max_iterSEXP);
    Rcpp::traits::input_parameter< const double >::type eps_f(eps_fSEXP);
    Rcpp::traits::input_parameter< const double >::type eps_g(eps_gSEXP);
    rcpp_result_gen = Rcpp::wrap(optimNVC(y, mu, U, ellinit, max_iter, eps_f, eps_g));
    return rcpp_result_gen;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_NVC_optimNVC", (DL_FUNC) &_NVC_optimNVC, 7},
    {NULL, NULL, 0}
};

RcppExport void R_init_NVC(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
