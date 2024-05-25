#ifndef SPLINALG_HH
#define SPLINALG_HH

#include <iostream>
#include <cmath>
#include <vector>
#include "Subvector.h"

// stands for sparse linalg
namespace splinalg {
  // computes the givensrotation for two input values
  // returnes a triple of cos(alpha) c, sin(alpha) s, radius r
  // doesn't modify input
  std::vector<double> COMPUTE_GIVENS(const double& a, const double& b);

  // applies a givens rotation to two double values that are interpreted as
  // a 2-dim vector
  // input is c (cos), s (sin), a (value 1), b (value 2)
  // changes the doubles that the rotation has to be applied to :)
  void APPLY_GIVENS(const std::vector<double>& cs, double& a, double& b);

  // function that introduces the buldge element and returnes it
  // also works with a shift parameter for implicit method
  double apply_giv_first_row(Subvector& diag, Subvector& subdiag, double shift);

  void apply_giv_last_row(Subvector& diag, Subvector& subdiag, double& buldge);

  // applies a givens rotation to the index-th row and column and below that
  // applies only if row_ind is not the first or second last index of diag
  // takes the diag and subdiag vectors of the tridiag matrix
  // and a givens rotation (c,s,r) as a vector (returned by COMPUTE_GIVENS)
  // Applies only if matrix dimension > 2 (otherwise we can easily compute evs)
  // modifies diag and subdiag only
  // modifies the buldge element for the buldge chase
  void apply_giv_row(Subvector& diag, Subvector& subdiag,
                         const int row_ind, double& buldge);

  // Shifts---------------------------------------------------------------------
  // modified sign function that returns 1 if the entry is 0
  int signum(const double number);

  // computes the Wilkinson shift of the current matrix
  double WILKINSON_SHIFT(Subvector& diag, Subvector& subdiag);

  // computes an exceptional shift

  // computes the eigenvalues of a 2x2 matrix and writes them into the diagonal
  // makes the subdiagonal enries 0
  // this is only called on 2x2 leftover matrices
  void EIGEN_DIM_2(Subvector& diag, Subvector& subdiag);

  // QR-Method------------------------------------------------------------------
  // performes one symmetric tridiagonal qr-step with implicit shift
  // Algorithm 2.5.8 in the lecture
  // operates on diag and subdiag
  void TRIDIAG_QR_STEP(Subvector& diag, Subvector& subdiag);

  // QR-Algorithm with deflation
  void TRIDIAG_QR_ITERATION(std::vector<double>& diagonal, std::vector<double>& subdiagonal);
}

#endif
