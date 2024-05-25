#ifndef SUBVECTOR_HH
#define SUBVECTOR_HH

#include <vector>
#include <iostream>
#include <armadillo>

class Subvector {
private:
  std::vector<double>& vec_;
  int startIndex_;
  int endIndex_;

public:
  // Constructer
  Subvector(std::vector<double>& vec, int startIndex, int endIndex)
    : vec_(vec), startIndex_(startIndex), endIndex_(endIndex) {}

  // Returns a new subvector with the same referenced vector
  Subvector subvector(int subStartIndex, int subEndIndex);

        double& operator[](int index);
  const double& operator[](int index) const;

  // returns the size of the referenced part
  int size() const { return endIndex_ - startIndex_ + 1; }

  // returns the size of the original vector that is being referenced
  int original_size() const { return vec_.size(); }

  // overwrite with arma::vector
  void overwrite(arma::vec& v);

  // iterator to the first element of Subvector
  std::vector<double>::iterator begin();

  // iterator to after the last element of Subvector
  std::vector<double>::iterator end();

  // setters--------------------------------------------------------------------
  void set_frame(const int startIndex, const int endIndex);
};

#endif
