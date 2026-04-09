#include "openmp_lab/common.hpp"

#include <omp.h>

#include <algorithm>
#include <iostream>
#include <vector>

namespace {

double parallel_sum(std::size_t n, int thread_count) {
  omp_set_num_threads(thread_count);

  double sum = 0.0;
  const auto start = std::chrono::steady_clock::now();

  // TODO(student):
  // Add a parallel for with reduction on sum.
  // This is the core pattern for safe parallel accumulation.
  #pragma omp parallel for reduction(+:sum)
  for (int i = 0; i < static_cast<int>(n); ++i) {
    sum += 1.0 / (1.0 + i);
  }
  const auto end = std::chrono::steady_clock::now();
  const double ms =
      std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;

  std::cout << "threads=" << thread_count << ", sum=" << sum << ", time=" << ms << " ms\n";
  return ms;
}

}  // namespace

int main() {
  openmp_lab::print_divider("thread count sweep");
  constexpr std::size_t n = 40'000'000;

  const int hw = std::max(2, omp_get_max_threads());
  std::vector<int> thread_candidates = {1, 2, 4, 8, hw};
  std::sort(thread_candidates.begin(), thread_candidates.end());
  thread_candidates.erase(std::unique(thread_candidates.begin(), thread_candidates.end()),
                          thread_candidates.end());

  std::cout << "Test problem size n=" << n << "\n";
  std::cout << "Question: more threads are not always better. Measure, do not assume.\n";
  std::cout << "Exercise: fill the OpenMP reduction TODO before trusting these timings.\n";

  double baseline_ms = 0.0;
  for (const int threads : thread_candidates) {
    const double ms = parallel_sum(n, threads);
    if (threads == 1) {
      baseline_ms = ms;
    } else {
      std::cout << "speedup_vs_1_thread=" << openmp_lab::compute_speedup(baseline_ms, ms)
                << "\n";
    }
  }

  std::cout << "\nThink about why scaling may flatten:\n"
               "- memory bandwidth\n"
               "- scheduling overhead\n"
               "- reduction overhead\n"
               "- core count / SMT limits\n";
  return 0;
}
