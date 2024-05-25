#include "dc_alg.h"

template<typename T>
void print_vec2(T v) {
  std::cout << "print_vec = [ ";
  for (int i = 0; i < v.size(); i++) {
    std::cout << v[i] << " ";
  }
  std::cout << "]" << std::endl;
}

bool num_equal(const double& a, const double& b, const double& refsize) {
  return (std::abs(a - b) <= 1e-10 * refsize);
}

double ref(const Subvector& d) {
  double refsize = 0.0;
  for (int i = 0; i < d.size(); i++) {
    refsize = (double)(refsize < std::abs(d[i])) * std::abs(d[i]) + (double)(refsize >= std::abs(d[i])) * refsize;
  }
  return refsize;
}

double ref(const arma::vec& b) {
  double refsize = 0.0;
  for (int i = 0; i < b.n_elem; i++) {
    refsize = (double)(refsize < std::abs(b(i))) * std::abs(b(i)) + (double)(refsize >= std::abs(b(i))) * refsize;
  }
  return refsize;
}

std::vector<double> partition(Subvector& subd, Subvector& d) {
  // get the "middle" Index of the subdiagonal part of the matrix
  // if no middle index, then the split will be rounded to the first half
  int middle = (subd.size() - 1) / 2;
  std::vector<double> alpha_theta_spl(3);
  alpha_theta_spl[0] = subd[middle];
  alpha_theta_spl[2] = (double)middle;

  // special case: matrix is reduced
  if (std::abs(subd[middle]) < 1e-14 * (std::abs(d[middle]) + std::abs(d[middle + 1]))) {
    // subdiagonal entry is numerically zero
    alpha_theta_spl[1] = 0.0;
    return alpha_theta_spl;
  }

  // choice of theta
  if (std::copysign(1, d[middle]) == std::copysign(1, d[middle + 1])) {
    // - theta * alpha has to have the same sign as d[middle]
    alpha_theta_spl[1] = std::copysign(1, (-1) * d[middle] / alpha_theta_spl[0]);
  } else {
    // detect possible cases of cancellation
    if (std::abs(alpha_theta_spl[0]) <= 0.8 * std::abs(d[middle + 1])
                || std::abs(alpha_theta_spl[0]) >= 1.2 * std::abs(d[middle + 1])) {
      // no cancellation anyways
      alpha_theta_spl[1] = std::copysign(1, (-1) * d[middle] / alpha_theta_spl[0]);
    } else {
      // might have cancellation
      double sign = std::copysign(1, (-1) * d[middle] / alpha_theta_spl[0]);
      double magnitude_inv = 1.5 * std::abs(d[middle + 1] / alpha_theta_spl[0]);
      alpha_theta_spl[1] = std::copysign(1.0 / magnitude_inv, sign);
    }
  }
  // modify the entries d[n], d[n+1], subd[n] (, superd[n])
  d[middle] -= alpha_theta_spl[0] * alpha_theta_spl[1];
  d[middle + 1] -= alpha_theta_spl[0] * (1.0 / alpha_theta_spl[1]);
  subd[middle] = 0;
  return alpha_theta_spl;
}

arma::mat make_tridiag(Subvector& subd, Subvector& d) {
  arma::mat T(d.size(), d.size());
  for (int i = 0; i < d.size(); i++) {
    T(i, i) = d[i];
  }
  for (int i = 0; i < subd.size(); i++) {
    T(i+1, i) = subd[i];
    T(i, i+1) = subd[i];
  }
  return T;
}

arma::mat merge_system(arma::vec& b, arma::mat& Q1, arma::mat& Q2, Subvector& d,
                                              const std::vector<double>& ats) {
  // build vectors of according sizes
  std::vector<double> merge1(Q1.n_cols); std::vector<double> merge2(Q2.n_cols);
  // fill them
  int l = 0;
  while (l < merge1.size()) {
    merge1[l] = d[l];
    l++;
  }
  while (l < d.size()) {
    merge2[l - merge1.size()] = d[l];
    l++;
  }
  // declar matrix Q
  arma::mat Q(Q1.n_rows + Q2.n_rows, Q1.n_cols + Q2.n_cols);
  // initialize indices for merge
  int i = 0; int j = 0; int k = 0;
  int n1 = Q1.n_cols;
  int n2 = Q2.n_cols;

  // begin merging
  while (i < n1 && j < n2) {
    if (merge1[i] <= merge2[j]) {
      d[k] = merge1[i];
      b(k) = Q1(Q1.n_rows - 1, i);
      Q.submat(0, k, Q1.n_rows - 1, k) = Q1.col(i);
      k++; i++;
    } else {
      d[k] = merge2[j];
      b(k) = (1 / ats[1]) * Q2(0, j);
      Q.submat(Q1.n_rows, k, Q.n_rows - 1, k) = Q2.col(j);
      k++; j++;
    }
  }

  // treat remaining elements
  while (i < n1) {
    d[k] = merge1[i];
    b(k) = Q1(Q1.n_rows - 1, i);
    Q.submat(0, k, Q1.n_rows - 1, k) = Q1.col(i);
    k++; i++;
  }

  while (j < n2) {
    d[k] = merge2[j];
    b(k) = (1 / ats[1]) * Q2(0, j);
    Q.submat(Q1.n_rows, k, Q.n_rows - 1, k) = Q2.col(j);
    k++; j++;
  }
  return Q;
}

void zero_equals(arma::vec& b, arma::mat& Q, Subvector& d, double& refsize) {
  // obtain the magnitude of the biggest entry
  //double refsize = ref(d);
  int j = 0; int i = 0;
  while (i < d.size() - 1) {
    j = i + 1;
    double refs = ref(d);
    while (num_equal(d[i], d[j], refs)) {
      std::vector<double> giv = splinalg::COMPUTE_GIVENS(b(i), b(j));
      // set the values of b according to the computation
      splinalg::APPLY_GIVENS(giv, b(i), b(j));
      // modify the corresponding two columns of Q
      for (int l = 0; l < d.size(); l++) {
        splinalg::APPLY_GIVENS(giv, Q(l, i), Q(l, j));
      }
      j++;
      if (j >= d.size()) { break; }
    }
    i = j;
  }
  return;
}

double frobeniusNorm(const Subvector& d, const Subvector& subd) {
  double norm = 0.0;

  // Step 1: Compute the sum of squares of main diagonal elements
  for (int i = 0; i < d.size(); i++) {
    norm += d[i] * d[i];
  }

  // Step 2: Compute the sum of squares of subdiagonal elements
  for (int i = 0; i < subd.size(); i++) {
    norm += 2 * subd[i] * subd[i]; // Subdiagonal elements appear twice in the Frobenius norm calculation
  }

  // Step 3: Take the square root of the sum
  norm = std::sqrt(norm);
  //std::cout << norm << '\n';
  return norm;
}

arma::uvec deflate(Subvector& d, arma::vec& b, int& last_filled, double& refval) {
  // initialize permutation vector
  arma::uvec perm(d.size());
  for (int i = 0; i < perm.n_elem; i++) {
    perm(i) = i;
  }
  //std::cout << "ref: " << refval << '\n';
  // switch all zero entries to the back
  int ind1 = 0; int ind2 = d.size() - 1;
  while (ind1 < ind2) {
    while (!num_equal(0.0, b(ind1), refval) && ind1 < ind2) {
      ind1++;
    }
    while (num_equal(0.0, b(ind2), refval) && ind1 < ind2) {
      ind2--;
    }
    if (ind1 >= ind2) {
      break;
    }
    // if get here we have found elements to switch
    std::swap(b(ind1), b(ind2));
    std::swap(d[ind1], d[ind2]);
    std::swap(perm(ind1), perm(ind2));
    ind1++; ind2--;
  }
  // all zeros are at the bottom now
  // compute the last index without zero element (either ind1 or ind1 - 1)
  if (!num_equal(0.0, b(ind1), refval)) {
    last_filled = ind1;
  } else {
    last_filled = ind1 - 1;
  }
  // now sort the top part: void modquicksort(Svec, low, high, b, perm)
  // sorts b and perm accordingly
  modquicksort(d, 0, last_filled, b, perm);
  if (last_filled < d.size() - 1) {
    modquicksort(d, last_filled + 1, d.size() - 1, b, perm);
  }
  return perm;
}

double recomp_abs_b_i(const std::vector<double>& evals, const Subvector& d_defl,
              const double& rho, const int& index, const std::vector<double>& unst_diff) {
  double b_i = 1.0;
  int ind_it = 0;
  // first product
  while (ind_it < index) {
    b_i *= (evals[ind_it] - d_defl[index]) / (d_defl[ind_it] - d_defl[index]);
    ind_it++;
  }
  // skip ind_it == index
  ++ind_it;
  while (ind_it < d_defl.size()) {
    b_i *= (evals[ind_it] - d_defl[index]) / (d_defl[ind_it] - d_defl[index]);
    ind_it++;
  }
  // only (\hat{\lambda}_i - d[i]) is missing as a factor
  // this is potentially unstable. take the value of unst_diff
  //b_i *= (evals[index] - d_defl[index]);
  b_i *= unst_diff[index];
  b_i *= 1.0 / rho;
  return std::sqrt(std::abs(b_i));
}

arma::mat cmp_eigvecs(const arma::vec& b_hat, const Subvector& d, const std::vector<double>& eigvals,
                                                      const std::vector<double>& unst_diff) {
  // create a matrix for the eigenvectors
  arma::mat Q_prime(d.size(), d.size());
  // compute the non-trivial eigenvectors
  // as well as the norm. take (d_i - lambda_i) as -unst_diff[i]
  #pragma omp parallel for
  for (int i = 0; i < b_hat.n_elem; i++) {
    double norm = 0.0;
    arma::vec temp(b_hat.n_elem);
    for (int j = 0; j < b_hat.n_elem; j++) {
      double entry;
      // handle the unstable case here
      if (j != i) {
        entry = b_hat(j) / (d[j] - eigvals[i]);
      } else {
        entry = b_hat(j) / -unst_diff[i];
      }
      norm += entry * entry;
      temp(j) = entry;
    }
    norm = std::sqrt(norm);
    // divide every entry of temp by the norm to nomalize
    for (int j = 0; j < b_hat.n_elem; j++) {
      temp(j) /= norm;
    }
    //std::cout << temp.t() << '\n';
    //std::cout << Q_prime.submat(0, i, b_hat.n_elem, i).t() << '\n';
    Q_prime.submat(0, i, b_hat.n_elem - 1, i) = temp;
  }
  // in case of t´defaltion there are leftover-elements
  for (int i = b_hat.n_elem; i < d.size(); i++) {
    Q_prime(i, i) = 1.0;
  }
  // return eigenvector-matrix
  //std::cout << Q_prime << '\n';
  return Q_prime;
}

arma::mat dc_rec(Subvector& subd, Subvector& d, const int rec_depth, double& refval) {
  //std::cout << rec_depth << " " << std::flush;
  // initializing some first things
  arma::mat Q; arma::vec vals;
  // if size is small we switch to armadillos stdandard QR algorithm (building block)
  // base case
  if (d.size() <= 30) {
    arma::mat T = make_tridiag(subd, d);
    // armadillo QR-algorithm
    arma::eig_sym(vals, Q, T, "std");
    // overwriting d with the eigenvalues
    d.overwrite(vals);
    return Q;
  }
  // initializing for splitting
  arma::mat Q1; arma::mat Q2;
  // partition the matrix and create new Subvector windows
  std::vector<double> ats = partition(subd, d);
  Subvector subd1 = subd.subvector(0, (int)ats[2] - 1);
  Subvector d1 = d.subvector(0, (int)ats[2]);
  Subvector subd2 = subd.subvector((int)ats[2] + 1, subd.size() - 1);
  Subvector d2 = d.subvector((int)ats[2] + 1, d.size() - 1);
  // call the recursive algorithm with the new windows
  if (rec_depth > -1) {
    // single threaded version
    Q1 = dc_rec(subd1, d1, rec_depth + 1, refval);
    Q2 = dc_rec(subd2, d2, rec_depth + 1, refval);
  } else {
    // multithreaded version
    auto future1 = std::async(std::launch::async, dc_rec, std::ref(subd1), std::ref(d1), rec_depth + 1, std::ref(refval));
    auto future2 = std::async(std::launch::async, dc_rec, std::ref(subd2), std::ref(d2), rec_depth + 1, std::ref(refval));
    Q1 = future1.get();
    Q2 = future2.get();
  }
  /*
  */
  // create the vector b = {Q1(last), theta^-1 Q2(first)}
  arma::vec b(Q1.n_cols + Q2.n_cols);
  // merge Q1 and Q2 into a big matrix following the merging of the Eigenvalues
  // and merge the elements into b according to d
  Q = merge_system(b, Q1, Q2, d, ats);
  // free the memory of Q1 and Q2
  Q1.reset(); Q2.reset();
  // if the system was block-diagonal at the splitting point from the beginning, we can return now
  // this was indicated by theta = 0
  if (ats[1] == 0.0) {
    return Q;
  }
  // deflation of the system!!! We first look for equal diagonal entries
  zero_equals(b, Q, d, refval);
  // then reorder the entries such that the first ones are non-reduced
  // collect the permutation vector (last_filled is the deflation border index)
  // the index of the last element, that couldn't be deflated
  int last_filled;
  arma::uvec perm = deflate(d, b, last_filled, refval);
  //std::cout << "permutation" << perm.t() << '\n';

  // meat and potatoes----------------------------------------------------------
  // portait the reduced system
  std::vector<double> b_cp(last_filled + 1);
  for (int i = 0; i < last_filled + 1; i++) {
    b_cp[i] = b(i);
  }
  std::vector<double> evals(last_filled + 1);
  Subvector d_defl = d.subvector(0, last_filled);
  std::vector<double> unst_diff(last_filled + 1);
  // time to find the eigenvalues and eigenvectors
  //--------------------------------------------------------------
  /*
  std::cout << "b" << '\n';
  print_vec2(b_cp);
  std::cout << "d" << '\n';
  print_vec2(d_defl);
  std::cout << "rho: " << ats[0] * ats[1] << ", rec_depth: " << rec_depth << '\n';
  */
  //--------------------------------------------------------------
  sec::calc_roots(b_cp, d_defl, evals, ats[0] * ats[1], unst_diff);
  //std::cout << "calc_roots rec: " << rec_depth << '\n';
  //print_vec2(evals);
  // roots (eigenvalues) are now computed
  // for stability of the eigenvectors we recompute the non-zero
  // components of the vector b with respect to our computed eigenvalues
  arma::vec b_hat(b_cp.size());
  for (int i = 0; i < b_cp.size(); i++) {
    b_hat(i) = std::copysign(recomp_abs_b_i(evals, d_defl, ats[0]*ats[1], i, unst_diff), b_cp[i]);
  }
  // now the b_i fit the eigenvals correctly
  // compute the eigenvectors
  arma::mat Q_prime = cmp_eigvecs(b_hat, d, evals, unst_diff);
  // write the roots back into the first part of d
  for (int i = 0; i < evals.size(); i++) {
    d[i] = evals[i];
  }
  // free the memory of evals
  evals.resize(0);
  // sort the eigenvalues
  arma::uvec perm2 = arma::regspace<arma::uvec>(0, d.size() - 1);
  // b is just a placeholder here
  modquicksort(d, b, perm2);
  // all that is left is ot permute the rows of Q_prime according to perm2
  // and the columns of Q according to Q and then return the product Q*Q_prime
  Q = Q.cols(perm);
  //Q_prime = Q_prime.cols(perm2);
  //std::cout << Q << '\n';
  //std::cout << Q_prime << '\n';
  // this is the most expensive step in the algorithm and sadly can't be avoided
  //std::cout << "hello" << '\n';
  Q.submat(0, 0, Q.n_rows - 1, last_filled) = Q.submat(0, 0, Q.n_rows - 1, last_filled) * Q_prime.submat(0, 0, last_filled, last_filled);
  //Q = Q.cols(perm2);
  return Q.cols(perm2);
}

arma::mat dc_algo(std::vector<double>& subd, std::vector<double>& d) {
  // transforming input to be dynamically addressable
  Subvector subdiag(subd, 0, subd.size()-1);
  Subvector diag(d, 0, d.size()-1);

  // for size-deflation checks, we compute the frobenius norm
  double refval = 10000 * frobeniusNorm(diag, subdiag);

  // calling recursive alg and returning Q
  return dc_rec(subdiag, diag, 0, refval);
}
