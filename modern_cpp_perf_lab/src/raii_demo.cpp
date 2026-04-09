#include "lab/common.hpp"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

class RawBuffer {
 public:
  explicit RawBuffer(std::size_t size) : size_(size), data_(new int[size]) {
    std::cout << "RawBuffer acquired " << size_ << " ints\n";
  }

  ~RawBuffer() {
    delete[] data_;
    std::cout << "RawBuffer released\n";
  }

  RawBuffer(const RawBuffer&) = delete;
  RawBuffer& operator=(const RawBuffer&) = delete;

  int* data() { return data_; }
  std::size_t size() const { return size_; }

 private:
  std::size_t size_;
  int* data_;
};

class FileGuard {
 public:
  explicit FileGuard(const std::string& path) : file_(path, std::ios::app) {
    if (!file_) {
      throw std::runtime_error("failed to open file: " + path);
    }
    std::cout << "Opened file: " << path << "\n";
  }

  ~FileGuard() { std::cout << "Closing file automatically\n"; }

  std::ofstream& stream() { return file_; }

 private:
  std::ofstream file_;
};

int sum_with_exception(bool throw_midway) {
  RawBuffer buffer(1 << 20);
  std::iota(buffer.data(), buffer.data() + static_cast<std::ptrdiff_t>(buffer.size()), 1);

  FileGuard log("modern_cpp_perf_lab/runtime.log");
  log.stream() << "sum_with_exception begin throw_midway=" << throw_midway << '\n';

  if (throw_midway) {
    throw std::runtime_error("simulated failure after resources were acquired");
  }

  return std::accumulate(buffer.data(),
                         buffer.data() + static_cast<std::ptrdiff_t>(buffer.size()),
                         0LL) %
         100000;
}

void vector_demo() {
  std::vector<int> data(1 << 20, 7);
  std::cout << "std::vector owns " << data.size() << " ints in contiguous storage ("
            << lab::bytes_to_mib(data.size() * sizeof(int)) << " MiB)\n";
  std::cout << "front addr=" << static_cast<const void*>(data.data())
            << ", back addr="
            << static_cast<const void*>(data.data() + static_cast<std::ptrdiff_t>(data.size() - 1))
            << "\n";
}

}  // namespace

int main() {
  lab::print_divider("RAII and deterministic cleanup");

  try {
    {
      lab::ScopeTimer timer("successful RAII scope");
      const int result = sum_with_exception(false);
      std::cout << "result=" << result << "\n";
    }

    {
      lab::ScopeTimer timer("exception path");
      try {
        (void)sum_with_exception(true);
      } catch (const std::exception& ex) {
        std::cout << "caught exception: " << ex.what() << "\n";
      }
    }

    vector_demo();
  } catch (const std::exception& ex) {
    std::cerr << "fatal: " << ex.what() << "\n";
    return 1;
  }

  std::cout << "\nInterview cue: RAII is not just smart pointers. It is any object whose lifetime"
               " owns a resource and releases it in the destructor.\n";
  return 0;
}
