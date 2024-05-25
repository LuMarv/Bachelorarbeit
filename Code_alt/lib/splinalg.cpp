#include "splinalg.h"

// stable version with hypot
std::vector<double> splinalg::COMPUTE_GIVENS(const double& a, const double& b) {
  double c;
  double s;
  double r;
  if (b == 0) {
    c = 1.0;
    s = 0.0;
  } else {
    if (std::abs(b) > std::abs(a)) {
      double tao = a / b;
      double root = std::sqrt(1.0 + tao * tao);
      s = 1.0 / root;
      c = -1.0 * tao * s;
      r = b * root;
    } else {
      double tao = b / a;
      double root = std::sqrt(1.0 + tao * tao);
      c = 1.0 / root;
      s = -1.0 * tao * c;
      r = a * root;
    }
  }
  std::vector<double> v = {c, s, r};
  return v;
}

void splinalg::APPLY_GIVENS(const std::vector<double>& cs, double& a, double& b) {
  double a_old = a; double b_old = b;
  // applying the rotation to a and b
  // cs[0] == c and cs[1] == s
  // cs[2] == r (not used here)
  a = cs[0] * a_old - cs[1] * b_old;
  b = cs[1] * a_old + cs[0] * b_old;
  return;
}

double splinalg::apply_giv_first_row(Subvector& diag, Subvector& subdiag, double shift) {
  // check if dim > 2
  if (diag.size() < 3) {
    std::cerr << "rotation not recommended" << '\n';
  }
  double buldge = 0.0;
  // we need one temporary extra variables for the superdiagonal element
  double superdiag_0 = subdiag[0];
  // compute the rotation givens = {c, s, r}
  std::vector<double> givens = splinalg::COMPUTE_GIVENS(diag[0] - shift, subdiag[0]);
  // application of the rotation
  //diag[0] = givens[2]; subdiag[0] = 0.0;
  splinalg::APPLY_GIVENS(givens, diag[0], subdiag[0]);
  splinalg::APPLY_GIVENS(givens, superdiag_0, diag[1]);
  splinalg::APPLY_GIVENS(givens, diag[0], superdiag_0);
  splinalg::APPLY_GIVENS(givens, subdiag[0], diag[1]);
  splinalg::APPLY_GIVENS(givens, buldge, subdiag[1]);
  //std::cout << buldge << '\n';
  return buldge;
}

void splinalg::apply_giv_last_row(Subvector& diag, Subvector& subdiag, double& buldge) {
  //comment
  if (diag.size() < 3) {
    std::cerr << "rotation not recommended" << '\n';
  }
  int dim = diag.size();
  double superdiag_last = subdiag[subdiag.size() - 1];
  // compute the rotation givens = {c, s, r}
  std::vector<double> givens = splinalg::COMPUTE_GIVENS(subdiag[dim - 3], buldge);
  // application of the rotation
  subdiag[dim - 3] = givens[2]; buldge = 0;
  splinalg::APPLY_GIVENS(givens, diag[dim - 2], subdiag[dim - 2]);
  splinalg::APPLY_GIVENS(givens, superdiag_last, diag[dim - 1]);
  splinalg::APPLY_GIVENS(givens, diag[dim - 2], superdiag_last);
  splinalg::APPLY_GIVENS(givens, subdiag[dim - 2], diag[dim - 1]);
}

void splinalg::apply_giv_row(Subvector& diag, Subvector& subdiag,
                              const int row_ind, double& buldge) {
  // check if dim > 2 (otherwise we can easily compute evs)
  if (diag.size() < 3) {
    std::cerr << "APPLY_GIV_TRIDIAG: dim too small for givens to make sense :)" << '\n';
  }
  // get the dimension
  int dim = diag.size();
  // check if row_ind is in the right interval
  // dim - 3 is the highest index before we use apply_giv_last_row() function
  // 1 is the lowest index where we use this function
  if (row_ind > dim - 3 || row_ind < 1) {
    std::cerr << "APPLY_GIV_TRIDIAG: index out of bound for rotation" << '\n';
  }
  // Now to the actual alg.-----------------------------------------------------
  // extra memory for the superdiagonal element
  double superdiag_i = subdiag[row_ind];
  // compute the rotation givens = {c, s, r}
  std::vector<double> givens = COMPUTE_GIVENS(subdiag[row_ind - 1], buldge);
  // apply the rotation to the entries
  subdiag[row_ind - 1] = givens[2]; buldge = 0.0;
  // here the buldge element slides down one element of the subdiag vector
  splinalg::APPLY_GIVENS(givens, buldge, subdiag[row_ind + 1]);                 // buldge has been modified
  // Now the rest of the rotations
  splinalg::APPLY_GIVENS(givens, diag[row_ind], subdiag[row_ind]);
  splinalg::APPLY_GIVENS(givens, superdiag_i, diag[row_ind + 1]);
  splinalg::APPLY_GIVENS(givens, diag[row_ind], superdiag_i);
  splinalg::APPLY_GIVENS(givens, subdiag[row_ind], diag[row_ind + 1]);
  return;
}

int splinalg::signum(const double number) {
  return -1.0 * (int)(number < 0.0) + 1.0 * (int)(number >= 0.0);
}

double splinalg::WILKINSON_SHIFT(Subvector& diag, Subvector& subdiag) {
  // Here the notation of Lemma 2.5.7 is used
  // n is the last index of diag
  int n = diag.size() - 1;
  double d = (diag[n - 1] - diag[n]) / 2.0;
  return diag[n] + d - (double)splinalg::signum(d) * std::sqrt(d*d + subdiag[n-1]*subdiag[n-1]);
}

void splinalg::EIGEN_DIM_2(Subvector& diag, Subvector& subdiag) {
  double trace = diag[0] + diag[1];
  double det = diag[0] * diag[1] - subdiag[0] * subdiag[0];
  // from here on it's just the pq-formula
  double sigma = std::sqrt( (trace / 2.0) * (trace / 2.0) - det );
  // writing the eigenvalues in the diagonal
  diag[0] = (trace / 2.0) + sigma;
  diag[1] = (trace / 2.0) - sigma;
  // making the sub and superdiagonal entries 0
  subdiag[0] = 0.0;
  return;
}

void splinalg::TRIDIAG_QR_STEP(Subvector& diag, Subvector& subdiag) {
  // initializing
  int dim = diag.size();
  if (dim < 3) {
    std::cerr << "TRIDIAG_QR_STEP: dimension error" << '\n';
  }
  // computing shift
  double sigma = WILKINSON_SHIFT(diag, subdiag);
  // applying to first row with implicit shift sigma
  double buldge = apply_giv_first_row(diag, subdiag, sigma);
  // buldge chasing from second row onward
  // dim - 3 is the last row index in which the middle apply function is used
  for (int i = 1; i <= dim - 3; i++) {
    apply_giv_row(diag, subdiag, i, buldge);
  }
  // throwing the buldge out of the matrix
  apply_giv_last_row(diag, subdiag, buldge);
  return;
}

void splinalg::TRIDIAG_QR_ITERATION(std::vector<double>& diagonal,
                                    std::vector<double>& subdiagonal) {
  // initializing subvectors for windowing
  const int dim = diagonal.size();
  Subvector diag(diagonal, 0, diagonal.size() - 1);
  Subvector subdiag(subdiagonal, 0, subdiagonal.size() - 1);
  // here we keep track of the area on wich we are currently operating (deflation)
  // in the beginning this is the entire range
  int start = 0; int stop = diagonal.size() - 1;
  bool converged = false;
  // start of the alg-----------------------------------------------------------
  while (!converged) {
    // look for a suited range for the QR-Step
    // iterate over the subdiagonal
    for (int i = start; i < dim; i++) {
      if (i == dim - 1) {
        start = dim - 1;
        converged = true;
        break;
      } else if (std::abs(subdiagonal[i]) > 1e-10 * (std::abs(diagonal[i]) + std::abs(diagonal[i+1]))) {
        start = i;
        break;
      }
    }
    if (start == dim - 1) {
      break;
    }
    // now we need to set stop
    for (int i = start + 1; i < dim; i++) {
      if (i == dim - 1) {
        // end of the matrix
        stop = dim - 1;
        break;
      } else if (std::abs(subdiagonal[i]) <= 1e-10 * (std::abs(diagonal[i]) + std::abs(diagonal[i+1]))) {
        stop = i;
        break;
      }
    }
    //std::cout << "start = " << start << " stop = " << stop << '\n';
    // now we have set the range
    // set the window for the subvectors
    diag.set_frame(start, stop);
    subdiag.set_frame(start, stop - 1);
    // if we have a 2x2 matrix, we can compute the eigenvalues directly
    if (stop - start + 1 == 2) {
      //std::cout << "EIGEN_DIM_2" << '\n';
      EIGEN_DIM_2(diag, subdiag);
    } else if (stop - start + 1 > 2) {
      //std::cout << "TRIDIAG_QR_STEP" << '\n';
      TRIDIAG_QR_STEP(diag, subdiag);
    }
  }
  // algorithm converged
  //std::cout << "The Eigenvalues have been computed!!!" << '\n';
  return;
}
