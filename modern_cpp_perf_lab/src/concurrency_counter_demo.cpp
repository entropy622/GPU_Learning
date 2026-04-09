#include "lab/common.hpp"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <iostream>
#include <mutex>
#include <thread>
#include <vector>

namespace {

constexpr int kIterationsPerThread = 500000;

struct PaddedCounter {
  alignas(64) std::uint64_t value = 0;
};

template <typename Fn>
double run_threads(Fn fn, int thread_count) {
  std::vector<std::thread> threads;
  threads.reserve(thread_count);

  const auto start = std::chrono::steady_clock::now();
  for (int tid = 0; tid < thread_count; ++tid) {
    threads.emplace_back(fn, tid);
  }
  for (auto& thread : threads) {
    thread.join();
  }
  const auto end = std::chrono::steady_clock::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;
}

}  // namespace

int main() {
  lab::print_divider("Concurrency: mutex, atomic, and contention");

  const int thread_count =
      std::max(2u, std::thread::hardware_concurrency() == 0 ? 4u : std::thread::hardware_concurrency());
  const std::uint64_t expected =
      static_cast<std::uint64_t>(thread_count) * kIterationsPerThread;

  std::cout << "thread_count=" << thread_count << ", iterations/thread=" << kIterationsPerThread
            << "\n";

  {
    std::uint64_t counter = 0;
    std::mutex mu;
    const double ms = run_threads(
        [&](int) {
          for (int i = 0; i < kIterationsPerThread; ++i) {
            std::lock_guard<std::mutex> lock(mu);
            ++counter;
          }
        },
        thread_count);
    std::cout << "mutex protected counter=" << counter << ", expected=" << expected
              << ", time=" << ms << " ms\n";
  }

  {
    std::atomic<std::uint64_t> counter{0};
    const double ms = run_threads(
        [&](int) {
          for (int i = 0; i < kIterationsPerThread; ++i) {
            counter.fetch_add(1, std::memory_order_relaxed);
          }
        },
        thread_count);
    std::cout << "single hot atomic counter=" << counter.load() << ", expected=" << expected
              << ", time=" << ms << " ms\n";
  }

  {
    std::vector<PaddedCounter> locals(static_cast<std::size_t>(thread_count));
    const double ms = run_threads(
        [&](int tid) {
          auto& local = locals[static_cast<std::size_t>(tid)].value;
          for (int i = 0; i < kIterationsPerThread; ++i) {
            ++local;
          }
        },
        thread_count);

    std::uint64_t total = 0;
    for (const auto& local : locals) {
      total += local.value;
    }
    std::cout << "sharded local counters=" << total << ", expected=" << expected
              << ", time=" << ms << " ms\n";
  }

  std::cout
      << "\nInterview cue: atomic makes the shared counter correct, but a single contended"
         " fetch_add can still serialize throughput.\n";
  return 0;
}
