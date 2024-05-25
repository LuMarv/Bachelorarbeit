#ifndef PARTITION
#define PARTITION

#include <vector>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <armadillo>
#include <thread>
#include <future>
#include "lib/Subvector.h"
#include "lib/splinalg.h"
#include "lib/mod_qsort.h"
#include "lib/secular_eval.h"

// checks, if a and b are numerically equal with respect to the biggest entry
bool num_equal(const double& a, const double& b, const double& refsize);

// obtains the parameter refsize for above function
// biggest absolute valued entry of d
double ref(const Subvector& d);

double ref(const arma::vec& b);

// returns alpha and theta for the partition of the matrix
// and modifies the corresponding entries of d and subd
// if the subdiagonal entry is numerically zero returns {subd[middle], 0}
// so we set theta to 0
std::vector<double> partition(Subvector& subd, Subvector& d);

// creates tridiagonal symm matrix from two Subvectors
arma::mat make_dense(Subvector& subd, Subvector& d);

// sorts the elements of the vector d and creates Q = diag(Q1,Q2)
// where the columns are switched in an according way to d
// sorts b according to d as well
arma::mat merge_system(arma::vec& b, arma::mat& Q1, arma::mat& Q2, Subvector& d,
                                                const std::vector<double>& ats);

// uses givens rotations to zero elements in the vector b
// transforms the columns of the matrix Q accordingly
void zero_equals(arma::vec& b, arma::mat& Q, Subvector& d, double& refsize);

// computes the frobenius norm of a matrix
// required for deflation
double frobeniusNorm(const Subvector& d, const Subvector& subd);

// defaltes the system and returns a permutation vector of the reorderd terms
// when we already have created all zeros possible
arma::uvec deflate(Subvector& d, arma::vec& b, int& last_filled, double& refval);

// recomputes abs-val of the i-th component of the non-zero part of b with respect
// to the computed eigenvalues for numerical stability
double recomp_abs_b_i(const std::vector<double>& evals, const Subvector& d_defl,
          const double& rho, const int& index, const std::vector<double>& unst_diff);

// compztes the complete embedded eigenvector matrix for
// the ROM-system
arma::mat cmp_eigvecs(const arma::vec& b_hat, const Subvector& d, const std::vector<double>& eigvals,
                                          const std::vector<double>& unst_diff);

// main hub and calling sequence can be found here
arma::mat dc_rec(Subvector& subd, Subvector& d, const int rec_depth, double& refval);

// the main Divide and Conquer algorithm function
// everything begins here
// takes tridiag. symm. matrix in from of two vectors subd and d.
// modifies the d vector (eigenvalues)
// returns eigenvector-matrix Q
arma::mat dc_algo(std::vector<double>& subd, std::vector<double>& d);
#endif
