#include "lab/common.hpp"

#include <atomic>
#include <chrono>
#include <iostream>
#include <thread>
#include <vector>

namespace {

struct Payload {
  int a = 0;
  int b = 0;
  std::vector<int> values;
};

void run_relaxed_counter_demo() {
  lab::print_divider("relaxed for independent statistics");

  std::atomic<int> counter{0};
  std::thread producer([&] {
    for (int i = 0; i < 100000; ++i) {
      counter.fetch_add(1, std::memory_order_relaxed);
    }
  });

  int snapshots = 0;
  std::thread observer([&] {
    while (counter.load(std::memory_order_relaxed) < 100000) {
      ++snapshots;
    }
  });

  producer.join();
  observer.join();
  std::cout << "final counter=" << counter.load(std::memory_order_relaxed)
            << ", observer snapshots=" << snapshots << "\n";
}

void run_release_acquire_demo() {
  lab::print_divider("release/acquire for publication");

  Payload payload;
  std::atomic<bool> ready{false};

  std::thread publisher([&] {
    payload.a = 7;
    payload.b = 35;
    payload.values = {1, 2, 3, 5, 8};
    ready.store(true, std::memory_order_release);
  });

  std::thread consumer([&] {
    while (!ready.load(std::memory_order_acquire)) {
      std::this_thread::yield();
    }
    std::cout << "payload.a=" << payload.a << ", payload.b=" << payload.b
              << ", payload.values.size()=" << payload.values.size() << "\n";
  });

  publisher.join();
  consumer.join();
}

void run_seq_cst_demo() {
  lab::print_divider("seq_cst as a correctness-first default");
  std::atomic<int> x{0};
  std::atomic<int> y{0};

  std::thread t1([&] {
    x.store(1, std::memory_order_seq_cst);
    std::cout << "t1 observes y=" << y.load(std::memory_order_seq_cst) << "\n";
  });

  std::thread t2([&] {
    y.store(1, std::memory_order_seq_cst);
    std::cout << "t2 observes x=" << x.load(std::memory_order_seq_cst) << "\n";
  });

  t1.join();
  t2.join();
  std::cout << "final x=" << x.load() << ", final y=" << y.load() << "\n";
}

}  // namespace

int main() {
  run_relaxed_counter_demo();
  run_release_acquire_demo();
  run_seq_cst_demo();

  std::cout << "\nInterview template: count with relaxed; publish constructed data with"
               " release/store + acquire/load; use seq_cst first when proving correctness.\n";
  return 0;
}
