#include "secular_eval.h"


double sec::random_double(const double& random_range) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> distribution(-1.0 * random_range, random_range);
  return distribution(gen);
}

double sec::prod_diff(double a, double b, double c, double d) {
  double cd = c * d;
  double error = std::fma(-c, d, cd);
  double result = std::fma(a, b, -cd);
  return result + error;
}

void sec::qf(const double a, const double b, const double c, double& root1, double& root2) {
  double q = -0.5 * (b + std::copysign(std::sqrt(prod_diff(b, b, 4.0 * a, c)), b));
  root1 = q / a;
  root2 = c / q;
  return;
}

double sec::initial_guess(const std::vector<double>& b, const std::vector<double>& delta, int root_ind) {
  // The starting value of the last root can be 0 (A.Melman)
  // The way the calculation of F(gamma) is done ( f(1/gamma) ) we get a problem
  // if gamma is 0 though. This discontinuity is removable though and in this case
  // F(0) = 1 and F'(0) is the negative sum of all the b_i^2
  // To get a viable starting value for this case, one newton step is calculated
  // in advance using the above knowledge
  if (root_ind == b.size() - 1) {
    double sum = 0.0;
    for (int i = 0; i < b.size(); i++) {
      sum += std::pow(b[i], 2);
    }
    // computing a viable first iterate
    return 1.0 / sum;
  }
  // calculating temporary constants
  double c2 = std::pow(b[root_ind + 1] / delta[root_ind + 1], 2);
  // using fused multiply add for summation
  // summation in two separate for loops reduces the need for internal if else expressions
  double c1 = 1.0;
  for (int j = 0; j < root_ind; j++) {
    c1 = std::fma(std::pow(b[j], 2), 1.0 / delta[j], c1);
  }
  for (int j = root_ind + 1; j < b.size(); j++) {
    c1 = std::fma(std::pow(b[j], 2), 1.0 / delta[j], c1);
  }
  // setting up the quadratic equation
  double root1, root2;
  double arg1 = -1.0 * std::pow(b[root_ind], 2);
  double arg2 = std::fma(-arg1, 1.0 / delta[root_ind + 1], c1);
  double arg3 = std::fma(-c1, 1.0 / delta[root_ind + 1], c2);
  // computing the roots of which only one is important
  qf(arg1, arg2, arg3, root1, root2);
  //std::cout << "roots: " << root1 << ", " << root2 << '\n';
  if (root1 < 1.0 / delta[root_ind + 1]) {
    return root2;
  } else if (root2 < 1.0 / delta[root_ind + 1]) {
    return root1;
  }
  return std::min(root1, root2);
}

double sec::sec_eval(const std::vector<double>& b, const std::vector<double>& delta, double mu) {
  double sum = 1.0;
  for (int j = 0; j < b.size(); j++) {
    sum += b[j] * b[j] / (delta[j] - mu);
  }
  return sum;
}

double sec::sec_der_eval(const std::vector<double>& b, const std::vector<double>& delta, double mu) {
  double sum = 0;
  for (int j = 0; j < b.size(); j++) {
    double temp = b[j] / (delta[j] - mu);
    sum += temp * temp;
  }
  return sum;
}

/*
double sec::eval_F(const std::vector<double>& b, const std::vector<double>& delta, double gamma, int rootnr) {
  double term1 = 1.0;
  double term2 = 0.0;
  for (int j = 0; j < b.size(); j++) {
    if (j != rootnr) {
      term2 += (b[j] * b[j]) / delta[j];
    }
  }
  double term3 = (-1.0) * b[rootnr] * b[rootnr] * gamma;
  double term4 = 0.0;
  for (int j = 0; j < b.size(); j++) {
    if (j != rootnr) {
      term4 += std::pow(b[j] / delta[j], 2) / ( gamma - (1.0 / delta[j]) );
    }
  }
  return term1 + term2 + term3 + term4;
}

double sec::eval_der_F(const std::vector<double>& b, const std::vector<double>& delta, double gamma, int rootnr) {
  double sum = 0.0;
  for (int j = 0; j < b.size(); j++) {
    if (j != rootnr) {
      sum += std::pow(b[j] / delta[j], 2) / std::pow( gamma - (1.0 / delta[j]), 2 );
    }
  }
  sum += b[rootnr];
  return (-1.0) * sum;
}
*/

double sec::eval_F(const std::vector<double>& b, const std::vector<double>& delta, double gamma, int rootnr) {
    double sum1 = 0.0;
    double sum2 = 0.0;

    for (int j = 0; j < b.size(); ++j) {
        if (j != rootnr) {
            sum1 += b[j] * b[j] / delta[j];
        }
    }

    for (int j = 0; j < b.size(); ++j) {
        if (j != rootnr) {
            double denom = gamma - 1.0 / delta[j];
            if (std::abs(denom) < 1e-14) {
                denom = (denom >= 0) ? 1e-14 : -1e-14;  // Adjust denominator slightly
            }
            sum2 += (b[j] / delta[j]) * (b[j] / delta[j]) / denom;
        }
    }

    double result = 1.0 + sum1 - b[rootnr] * b[rootnr] * gamma + sum2;
    return result;
}

double sec::eval_der_F(const std::vector<double>& b, const std::vector<double>& delta, double gamma, int rootnr) {
    double sum = 0.0;

    for (int j = 0; j < b.size(); ++j) {
        if (j != rootnr) {
            double denom = gamma - 1.0 / delta[j];
            if (std::abs(denom) < 1e-14) {
                denom = (denom >= 0) ? 1e-14 : -1e-14;  // Adjust denominator slightly
            }
            sum += (b[j] / delta[j]) * (b[j] / delta[j]) / (denom * denom);
        }
    }

    double result = -b[rootnr] * b[rootnr] - sum;
    return result;
}

double sec::newton_sec(const std::vector<double>& b, const std::vector<double>& delta, double init) {
  double it1, it2;
  it2 = init;
  it1 = it2 + 1.0;
  while (std::abs(it1 - it2) > 1e-10 * std::abs(it1)) {
    //std::cout << "/ message /" << '\n';
    it1 = it2;
    double eval = sec_eval(b, delta, 1.0 / it1);
    // evaluating the darivative of F_i requires dividing by -it1^2
    double eval_der = (-1.0) * sec_der_eval(b, delta, 1.0 / it1) / (it1 * it1);
    it2 = it1 - ( eval / eval_der );
  }
  return it2;
}

double sec::newton_F(const std::vector<double>& b, const std::vector<double>& delta, double init, double rootnr) {
  double it1, it2;
  it2 = init;
  it1 = it2 + 1.0;
  while (std::abs(it1 - it2) > 1e-10 * std::abs(it1)) {
    it1 = it2;
    double eval = eval_F(b, delta, it1, rootnr);
    // evaluating the darivative of F_i
    double eval_der = eval_der_F(b, delta, it1, rootnr);
    it2 = it1 - ( eval / eval_der );
  }
  return it2;
}

double sec::cal_sec_root_i(const std::vector<double>& b, const Subvector& d, double rho, int rootnr, std::vector<double>& unst_diff) {
  // computing delta vector for i-th root: delta[j] = (d[j] - d[i]) / rho
  std::vector<double> delta(b.size());
  for (int j = 0; j < b.size(); j++) {
    delta[j] = (d[j] - d[rootnr]) / rho;
  }
  double init = initial_guess(b, delta, rootnr);
  //std::cout << init << '\n';

  //double gamma = newton_sec(b, delta, init);
  double gamma = newton_F(b, delta, init, rootnr);
  // resubstituting the transformations x = 1/gamma
  // x is root of f_i(x)
  gamma = 1.0 / gamma;
  if (gamma == 0.0) {
    std::cout << "yohoho" << '\n';
    unst_diff[rootnr] = 1e-14;
  }
  unst_diff[rootnr] = gamma * rho;
  if (std::abs(std::fma(rho, gamma, d[rootnr])) == std::abs(d[rootnr])) {
    return std::fma(rho, gamma, d[rootnr] + 1e-14 * d[rootnr]);
  }
  // lambda = rho * x + d[i] (backsubstitution)
  return std::fma(rho, gamma, d[rootnr]);
}

void sec::calc_roots(std::vector<double>& b, Subvector& d, std::vector<double>& roots, double rho, std::vector<double>& unst_diff) {
  if (rho < 0.0) {
    //std::cout << "rho < 0" << '\n';
    std::reverse(b.begin(), b.end());
    std::reverse(d.begin(), d.end());
    for (int i = 0; i < b.size(); i++) {
      d[i] *= -1.0;
    }
    // computation
    #pragma omp parallel for
    for (int i = 0; i < b.size(); i++) {
      roots[roots.size() - 1 - i] = -cal_sec_root_i(b, d, -rho, i, unst_diff);
    }
    // comutation done
    std::reverse(b.begin(), b.end());
    std::reverse(d.begin(), d.end());
    std::reverse(unst_diff.begin(), unst_diff.end());
    for (int i = 0; i < b.size(); i++) {
      d[i] *= -1.0;
      unst_diff[i] *= -1.0;
    }
    return;
  }
  #pragma omp parallel for
  for (int i = 0; i < b.size(); i++) {
    roots[i] = cal_sec_root_i(b, d, rho, i, unst_diff);
  }
  return;
}
