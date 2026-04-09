#pragma once

#include <chrono>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <string>

namespace lab {

class ScopeTimer {
 public:
  explicit ScopeTimer(std::string label)
      : label_(std::move(label)),
        start_(std::chrono::steady_clock::now()) {}

  ~ScopeTimer() {
    const auto end = std::chrono::steady_clock::now();
    const auto ms =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start_).count() /
        1000.0;
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

inline std::size_t bytes_to_mib(std::size_t bytes) {
  return bytes / (1024 * 1024);
}

}  // namespace lab
