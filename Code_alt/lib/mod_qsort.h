#ifndef MOD_QSORT
#define MOD_QSORT

#include <iostream>
#include <vector>
#include <random>
#include <ctime>
#include <armadillo>
#include "Subvector.h"

// everything one needs for a quicksort, that saves permutations
// also and permutes an arma::vec accordingly

int randomPivot(int low, int high);

int partition(Subvector& arr, int low, int high, arma::vec& b, arma::uvec& perm);

void modquicksort(Subvector& arr, int low, int high, arma::vec& b, arma::uvec& perm);

// for sorting the entire vector
void modquicksort(Subvector& arr, arma::vec& b, arma::uvec& perm);

#endif //MOD_QSORT
