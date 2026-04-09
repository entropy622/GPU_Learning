#include "openmp_lab/common.hpp"

#include <omp.h>

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <stdexcept>

namespace {

openmp_lab::Matrix matmul_serial(const openmp_lab::Matrix& a, const openmp_lab::Matrix& b) {
  openmp_lab::Matrix c(a.rows(), b.cols());
  for (int i = 0; i < a.rows(); ++i) {
    for (int j = 0; j < b.cols(); ++j) {
      double sum = 0.0;
      for (int k = 0; k < a.cols(); ++k) {
        sum += a(i, k) * b(k, j);
      }
      c(i, j) = sum;
    }
  }
  return c;
}

openmp_lab::Matrix transpose(const openmp_lab::Matrix& in) {
  openmp_lab::Matrix out(in.cols(), in.rows());
  for (int i = 0; i < in.rows(); ++i) {
    for (int j = 0; j < in.cols(); ++j) {
      out(j, i) = in(i, j);
    }
  }
  return out;
}

openmp_lab::Matrix matmul_blocked_exercise(const openmp_lab::Matrix& a,
                                           const openmp_lab::Matrix& b_t,
                                           int block_size) {
  if (a.cols() != b_t.cols()) {
    throw std::invalid_argument("matmul_blocked_exercise expects b to be transposed");
  }

  openmp_lab::Matrix c(a.rows(), b_t.rows());

  // Exercise goals:
  // 1. Add an OpenMP pragma at the right loop level.
  // 2. Keep each thread writing to distinct c(i, j) elements.
  // 3. Do not add atomics unless you prove you need them.
  // 4. Verify against the serial reference before trusting speed.

  // Suggested direction:
  // #pragma omp parallel for collapse(2) schedule(static)
  for (int ii = 0; ii < a.rows(); ii += block_size) {
    for (int jj = 0; jj < b_t.rows(); jj += block_size) {
      for (int i = ii; i < std::min(ii + block_size, a.rows()); ++i) {
        for (int j = jj; j < std::min(jj + block_size, b_t.rows()); ++j) {
          double sum = 0.0;

          // TODO(student):
          // Replace this inner loop with blocked traversal over k:
          // for (int kk = 0; kk < a.cols(); kk += block_size) { ... }
          // and then accumulate over the local tile.
          for (int k = 0; k < a.cols(); ++k) {
            sum += a(i, k) * b_t(j, k);
          }

          c(i, j) = sum;
        }
      }
    }
  }

  return c;
}

template <typename Fn>
double time_run(const char* label, Fn fn, const openmp_lab::Matrix& a, const openmp_lab::Matrix& b,
                openmp_lab::Matrix& out) {
  const auto start = std::chrono::steady_clock::now();
  out = fn(a, b);
  const auto end = std::chrono::steady_clock::now();
  const double ms =
      std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;
  std::cout << label << " time=" << ms << " ms\n";
  return ms;
}

}  // namespace

int main(int argc, char** argv) {
  const int n = (argc > 1) ? std::atoi(argv[1]) : 256;
  const int block_size = (argc > 2) ? std::atoi(argv[2]) : 32;

  std::cout << "Blocked matrix multiply exercise\n";
  std::cout << "n=" << n << ", block_size=" << block_size << ", max_threads=" << omp_get_max_threads()
            << "\n";
  std::cout << "This file intentionally leaves the key optimization step to you.\n";

  auto a = openmp_lab::make_matrix(n, n, 7);
  auto b = openmp_lab::make_matrix(n, n, 9);
  auto b_t = transpose(b);

  openmp_lab::Matrix serial(1, 1);
  openmp_lab::Matrix blocked(1, 1);

  const double serial_ms = time_run("serial_reference", matmul_serial, a, b, serial);
  const auto blocked_start = std::chrono::steady_clock::now();
  blocked = matmul_blocked_exercise(a, b_t, block_size);
  const auto blocked_end = std::chrono::steady_clock::now();
  const double blocked_ms =
      std::chrono::duration_cast<std::chrono::microseconds>(blocked_end - blocked_start).count() /
      1000.0;

  std::cout << "blocked_exercise time=" << blocked_ms << " ms\n";
  std::cout << "correct? " << std::boolalpha << openmp_lab::almost_equal(serial, blocked) << "\n";
  std::cout << "speedup=" << openmp_lab::compute_speedup(serial_ms, blocked_ms) << "\n";

  std::cout << "\nYour tasks:\n"
               "1. Add OpenMP to the correct loop nest.\n"
               "2. Implement real k-blocking in the TODO section.\n"
               "3. Experiment with block size and thread count.\n"
               "4. Keep correctness checks after every optimization step.\n";
  return 0;
}
