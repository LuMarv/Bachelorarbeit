#ifndef SECULAR_EVAL_HH
#define SECULAR_EVAL_HH

#include <vector>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <random>
#include "Subvector.h"

namespace sec {
  // just your normal random number genrator
  double random_double(const double& random_range);

  // calculates the sum a*b - c*d in a numerically stable way using
  // fma (fused multiply add) in Kahan's Algorithm
  double prod_diff(double a, double b, double c, double d);

  // computes the two roots of a quardatic equation using Kahan's alg.
  void qf(const double a, const double b, const double c, double& root1, double& root2);

  // constructs an initial guess for the modified newton method from
  // b and delta
  double initial_guess(const std::vector<double>& b, const std::vector<double>& delta, int root_ind);

  // evaluates the shifted secular equation (f_i in the notes) at mu
  double sec_eval(const std::vector<double>& b, const std::vector<double>& delta, double mu);

  // evaluates the derivative of the shifted secular equation (f_i in the notes) at mu
  double sec_der_eval(const std::vector<double>& b, const std::vector<double>& delta, double mu);

  // implements the newton method from the starting point "init" for the function
  // f_i(1/gamma) (notes)
  // returnes the "converged" approximation of the closest root to init
  double newton_sec(const std::vector<double>& b, const std::vector<double>& delta, double init);

  //----------------------------------------------------------------------------
  double eval_F(const std::vector<double>& b, const std::vector<double>& delta, double gamma, int rootnr);

  double eval_der_F(const std::vector<double>& b, const std::vector<double>& delta, double gamma, int rootnr);

  double newton_F(const std::vector<double>& b, const std::vector<double>& delta, double init, double rootnr);
  //----------------------------------------------------------------------------

  // calculates the i-th root of the secular equation
  // requires the d's to be sorted
  double cal_sec_root_i(const std::vector<double>& b, const Subvector& d, double rho, int rootnr, std::vector<double>& unst_diff);

  // Calculates all the roots. Takes care of the case where rho is negative
  void calc_roots(std::vector<double>& b, Subvector& d, std::vector<double>& roots, double rho, std::vector<double>& unst_diff);
}

#endif //SECULAR_EVAL_HH
