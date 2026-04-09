#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace openmp_lab {

class ScopeTimer {
 public:
  explicit ScopeTimer(std::string label)
      : label_(std::move(label)),
        start_(std::chrono::steady_clock::now()) {}

  ~ScopeTimer() {
    const auto end = std::chrono::steady_clock::now();
    const auto ms =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start_).count() / 1000.0;
    std::cout << std::fixed << std::setprecision(3) << "[timer] " << label_ << ": " << ms
              << " ms\n";
  }

 private:
  std::string label_;
  std::chrono::steady_clock::time_point start_;
};

inline void print_divider(const std::string& title) {
  std::cout << "\n==== " << title << " ====\n";
}

class Matrix {
 public:
  Matrix(int rows, int cols) : rows_(rows), cols_(cols), data_(static_cast<std::size_t>(rows) * cols) {}

  double& operator()(int r, int c) { return data_[static_cast<std::size_t>(r) * cols_ + c]; }
  double operator()(int r, int c) const {
    return data_[static_cast<std::size_t>(r) * cols_ + c];
  }

  int rows() const { return rows_; }
  int cols() const { return cols_; }
  const std::vector<double>& raw() const { return data_; }

 private:
  int rows_;
  int cols_;
  std::vector<double> data_;
};

inline Matrix make_matrix(int rows, int cols, std::uint32_t seed) {
  Matrix out(rows, cols);
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      out(i, j) = dist(rng);
    }
  }
  return out;
}

inline bool almost_equal(const Matrix& lhs, const Matrix& rhs, double eps = 1e-8) {
  if (lhs.rows() != rhs.rows() || lhs.cols() != rhs.cols()) {
    return false;
  }
  for (int i = 0; i < lhs.rows(); ++i) {
    for (int j = 0; j < lhs.cols(); ++j) {
      if (std::abs(lhs(i, j) - rhs(i, j)) > eps) {
        return false;
      }
    }
  }
  return true;
}

inline double compute_speedup(double baseline_ms, double optimized_ms) {
  return baseline_ms / optimized_ms;
}

}  // namespace openmp_lab
