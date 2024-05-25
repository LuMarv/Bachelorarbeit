#include "Subvector.h"

Subvector Subvector::subvector(int subStartIndex, int subEndIndex) {
  if (subStartIndex >= 0 && subEndIndex < size()) {
    return Subvector(vec_, startIndex_ + subStartIndex, startIndex_ + subEndIndex);
  } else {
    std::cerr << "subvector: Index out of bounds" << '\n';
    return Subvector(vec_, startIndex_, endIndex_);
  }
}

double& Subvector::operator[](int index) {
  if (index < size()) {
    return vec_[startIndex_ + index];
  } else {
    std::cerr << "Subvector[]: Index out of bounds " << index << '\n';
    return vec_[startIndex_];
  }
}

const double& Subvector::operator[](int index) const {
  if (index < size()) {
    return vec_[startIndex_ + index];
  } else {
    std::cerr << "Subvector[]: Index out of bounds " << index << '\n';
    return vec_[startIndex_];
  }
}

void Subvector::overwrite(arma::vec& v) {
  for (int i = 0; i < this->size(); i++) {
    this->operator[](i) = v(i);
  }
  return;
}

void Subvector::set_frame(const int startIndex, const int endIndex) {
  if (startIndex < 0 || endIndex >= vec_.size() || startIndex - 1 > endIndex) {
    std::cerr << "set_frame: Invalid frame parameters!!!" << '\n';
    return;
  }
  startIndex_ = startIndex;
  endIndex_ = endIndex;
  return;
}

std::vector<double>::iterator Subvector::begin() {
  return vec_.begin() + startIndex_;
}

std::vector<double>::iterator Subvector::end() {
  return vec_.begin() + endIndex_ + 1;
}
