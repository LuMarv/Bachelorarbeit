#include "dc_alg.h"
#include "lib/Subvector.h"
#include <cstdlib>
#include <random>
//#include "secular_eval.cpp"

double rng() {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(0.0, 1.0);
  double random_number = dis(gen);
  return random_number;
}

template<typename T>
void print_vec(T v) {
  std::cout << "print_vec = [ ";
  for (int i = 0; i < v.size(); i++) {
    std::cout << v[i] << " ";
  }
  std::cout << "]" << std::endl;
}

int main(int argc, char const *argv[]) {
  int i = 40;
  if (argv[1] != nullptr) {
    i = std::atoi(argv[1]);
  }
  for (int n = 500; n < i; n+=500) {
    //std::cout << n << '\n';
    std::vector<double> diag(n, 2);
    std::vector<double> subdiag(n - 1, 1);
    // random Matrix
    for (int i = 0; i < diag.size() - 1; i++) {
      diag[i] =  rng();
      subdiag[i] = rng();
    }
    diag[diag.size() - 1] =  rng();
    // wilkinson matrix
    /*
    for (int i = 0; i < n/2; i++) {
      diag[i] = (double)(n - 2*i - 1) / 2.0;
      diag[n - i - 1] = (double)(n - 2*i - 1) / 2.0;
    }
    if (diag.size() % 2 != 0) {
      diag[diag.size() / 2] = 0;
    }
    */
    arma::mat M(n, n);
    for (int i = 0; i < n; i++) {
      M(i, i) = diag[i];
    }
    for (int i = 0; i < n - 1; i++) {
      M(i+1, i) = subdiag[i];
      M(i, i+1) = subdiag[i];
    }
    //std::cout << M << '\n';

    arma::vec eigv = arma::eig_sym(M);
    //std::cout << eigv.t() << '\n';

    //subdiag[(subdiag.size() - 1) / 2] = 0.0;

    //print_vec(diag);
    //print_vec(subdiag);

    arma::mat Q = dc_algo(subdiag, diag);
    std::cout << '\n';
    //std::cout << Q * Q.t() << '\n';

    //std::cout << Q.t() * M * Q << '\n';

    double error = 0.0;
    //print_vec(diag);
    //print_vec(eigv);
    std::cout << '\n';
    //std::cout << Q.t() * M * Q << '\n';

    //print_vec(eigv);
    double ersum = 0.0;
    for (int i = 0; i < diag.size(); i++) {
      ersum += std::abs(diag[i] - eigv(i));
      if (std::abs(diag[i] - eigv(i)) > error) {
        error = std::abs(diag[i] - eigv(i));
      }
    }
    std::cout << "size: " << diag.size() << '\n';
    std::cout << "avg error: " << ersum / diag.size() << '\n';
    std::cout << "Largest error is: " << error << '\n';
  }
  /*
  */
  return 0;
}
