#include "openmp_lab/common.hpp"

#include <omp.h>

#include <iostream>
#include <vector>

namespace {

void parallel_region_demo() {
  openmp_lab::print_divider("parallel region");
  std::cout << "This demo shows that OpenMP starts multiple threads in one process.\n";

#pragma omp parallel
  {
    const int tid = omp_get_thread_num();
    const int team_size = omp_get_num_threads();
#pragma omp critical
    std::cout << "Hello from thread " << tid << " / " << team_size << "\n";
  }
}

void data_race_demo() {
  openmp_lab::print_divider("data race vs atomic vs reduction");

  constexpr int increments = 200000;
  int wrong_counter = 0;

#pragma omp parallel for
  for (int i = 0; i < increments; ++i) {
    ++wrong_counter;
  }

  std::cout << "wrong_counter=" << wrong_counter
            << " (likely wrong because multiple threads updated one variable without sync)\n";

  int atomic_counter = 0;
#pragma omp parallel for
  for (int i = 0; i < increments; ++i) {
#pragma omp atomic
    ++atomic_counter;
  }

  std::cout << "atomic_counter=" << atomic_counter << " (correct, but may serialize updates)\n";

  int reduction_sum = 0;
#pragma omp parallel for reduction(+ : reduction_sum)
  for (int i = 0; i < increments; ++i) {
    reduction_sum += 1;
  }

  std::cout << "reduction_sum=" << reduction_sum
            << " (correct and usually better for accumulation patterns)\n";
}

void scheduling_demo() {
  openmp_lab::print_divider("where to parallelize");
  std::vector<int> work(16);
  for (int i = 0; i < static_cast<int>(work.size()); ++i) {
    work[static_cast<std::size_t>(i)] = (i % 4 == 0) ? 5000 : 1000;
  }

  std::cout << "Pretend each loop iteration has different work. This is where schedule choice matters.\n";

#pragma omp parallel for schedule(static, 2)
  for (int i = 0; i < static_cast<int>(work.size()); ++i) {
    volatile int sink = 0;
    for (int step = 0; step < work[static_cast<std::size_t>(i)]; ++step) {
      sink += step;
    }
#pragma omp critical
    std::cout << "static schedule handled iteration " << i << " on thread "
              << omp_get_thread_num() << "\n";
  }
}

}  // namespace

int main() {
  std::cout << "OpenMP version: " << _OPENMP << "\n";
  std::cout << "Max available threads: " << omp_get_max_threads() << "\n";
  std::cout << "Learning checklist:\n"
               "1. parallel region = where threads are created\n"
               "2. data race = multiple threads update shared state unsafely\n"
               "3. atomic/reduction = common tools to fix conflicts\n";

  parallel_region_demo();
  data_race_demo();
  scheduling_demo();

  std::cout << "\nNext: run omp_thread_sweep and omp_matrix_mul to study scaling and correctness.\n";
  return 0;
}
