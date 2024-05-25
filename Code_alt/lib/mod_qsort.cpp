#include "mod_qsort.h"

// Function to generate a random pivot index between low and high (inclusive)
int randomPivot(int low, int high) {
  std::mt19937 gen(std::time(0));
  std::uniform_int_distribution<int> dist(low, high);
  return dist(gen);
}

// Partition function for the Quick Sort algorithm with random pivot
int partition(Subvector& arr, int low, int high, arma::vec& b, arma::uvec& perm) {
  // Choose a random pivot and swap it with the last element
  int randomIndex = randomPivot(low, high);
  std::swap(arr[randomIndex], arr[high]);
  std::swap(b(randomIndex), b(high));
  std::swap(perm(randomIndex), perm(high));
  double pivot = arr[high]; // Pivot is now the last element
  int i = low - 1; // Index of smaller element

  for (int j = low; j < high; j++) {
    // If current element is smaller than or equal to pivot
    if (arr[j] <= pivot) {
      i++; // Increment index of smaller element
      std::swap(arr[i], arr[j]);
      std::swap(b(i), b(j));
      std::swap(perm(i), perm(j));
    }
  }
  std::swap(arr[i + 1], arr[high]);
  std::swap(b(i + 1), b(high));
  std::swap(perm(i + 1), perm(high));
  return i + 1;
}

// Quick Sort function
void modquicksort(Subvector& arr, int low, int high, arma::vec& b, arma::uvec& perm) {
  if (low < high) {
    // Partitioning index
    int pi = partition(arr, low, high, b, perm);

    // Separately sort elements before and after partition
    modquicksort(arr, low, pi - 1, b, perm);
    modquicksort(arr, pi + 1, high, b, perm);
  }
}

// Wrapper function for modquicksort
void modquicksort(Subvector& arr, arma::vec& b, arma::uvec& perm) {
  modquicksort(arr, 0, arr.size() - 1, b, perm);
}


/*
int main() {
  std::vector<double> arr1 = {10.0, 7.0, 8.0, 9.0, 1.0, 5.0};
  Subvector arr(arr1, 0, arr1.size() - 1);
  arma::vec b(arr.size(), arma::fill::randu);
  arma::uvec perm = arma::linspace<arma::uvec>(0, 5, 5 + 1);
  std::cout << "perm " << perm.t() << '\n';
  std::cout << "b " << b.t() << '\n';
  std::cout << "Original array: ";
  for (double x : arr1) {
    std::cout << x << " ";
  }
  std::cout << std::endl;

  modquicksort(arr, b, perm);

  std::cout << "Sorted array: ";
  for (double x : arr1) {
    std::cout << x << " ";
  }
  std::cout << std::endl;

  std::cout << "perm " << perm.t() << '\n';
  std::cout << "b " << b.t() << '\n';

  return 0;
}
*/
