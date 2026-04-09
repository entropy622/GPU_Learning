#include "openmp_lab/common.hpp"

#include <omp.h>

#include <cstdlib>
#include <iostream>

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

openmp_lab::Matrix matmul_parallel_outer(const openmp_lab::Matrix& a,
                                         const openmp_lab::Matrix& b) {
  openmp_lab::Matrix c(a.rows(), b.cols());

#pragma omp parallel for schedule(static)
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

openmp_lab::Matrix matmul_parallel_collapse(const openmp_lab::Matrix& a,
                                            const openmp_lab::Matrix& b) {
  openmp_lab::Matrix c(a.rows(), b.cols());

#pragma omp parallel for collapse(2) schedule(static)
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

template <typename Fn>
double time_matmul(const char* label, Fn fn, const openmp_lab::Matrix& a, const openmp_lab::Matrix& b,
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
  std::cout << "Matrix size: " << n << " x " << n << "\n";
  std::cout << "Max threads: " << omp_get_max_threads() << "\n";

  auto a = openmp_lab::make_matrix(n, n, 42);
  auto b = openmp_lab::make_matrix(n, n, 1337);

  openmp_lab::Matrix serial(1, 1);
  openmp_lab::Matrix outer(1, 1);
  openmp_lab::Matrix collapse(1, 1);

  const double serial_ms = time_matmul("serial", matmul_serial, a, b, serial);
  const double outer_ms = time_matmul("parallel_outer", matmul_parallel_outer, a, b, outer);
  const double collapse_ms =
      time_matmul("parallel_collapse2", matmul_parallel_collapse, a, b, collapse);

  std::cout << "parallel_outer correct? " << std::boolalpha
            << openmp_lab::almost_equal(serial, outer) << "\n";
  std::cout << "parallel_collapse2 correct? " << std::boolalpha
            << openmp_lab::almost_equal(serial, collapse) << "\n";
  std::cout << "speedup outer=" << openmp_lab::compute_speedup(serial_ms, outer_ms) << "\n";
  std::cout << "speedup collapse(2)=" << openmp_lab::compute_speedup(serial_ms, collapse_ms)
            << "\n";

  std::cout << "\nQuestion to answer after running:\n"
               "- Why is parallelizing the outer loop usually safe here?\n"
               "- Why would parallelizing the innermost reduction loop be harder?\n";
  return 0;
}
