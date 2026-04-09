#include "lab/common.hpp"

#include <cstddef>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

namespace {

struct Counters {
  int copies = 0;
  int moves = 0;
};

class TrackedBuffer {
 public:
  explicit TrackedBuffer(std::size_t size = 0, Counters* counters = nullptr)
      : counters_(counters), payload_(size, 1) {}

  TrackedBuffer(const TrackedBuffer& other)
      : counters_(other.counters_), payload_(other.payload_) {
    if (counters_ != nullptr) {
      ++counters_->copies;
    }
  }

  TrackedBuffer(TrackedBuffer&& other) noexcept
      : counters_(std::exchange(other.counters_, nullptr)),
        payload_(std::move(other.payload_)) {
    if (counters_ != nullptr) {
      ++counters_->moves;
    }
  }

  TrackedBuffer& operator=(const TrackedBuffer& other) {
    if (this == &other) {
      return *this;
    }
    counters_ = other.counters_;
    payload_ = other.payload_;
    if (counters_ != nullptr) {
      ++counters_->copies;
    }
    return *this;
  }

  TrackedBuffer& operator=(TrackedBuffer&& other) noexcept {
    if (this == &other) {
      return *this;
    }
    counters_ = std::exchange(other.counters_, nullptr);
    payload_ = std::move(other.payload_);
    if (counters_ != nullptr) {
      ++counters_->moves;
    }
    return *this;
  }

  std::size_t size() const { return payload_.size(); }

 private:
  Counters* counters_;
  std::vector<int> payload_;
};

TrackedBuffer build_buffer(std::size_t size, Counters& counters) {
  TrackedBuffer local(size, &counters);
  return local;
}

void print_counters(const std::string& label, const Counters& counters) {
  std::cout << label << " -> copies=" << counters.copies << ", moves=" << counters.moves
            << "\n";
}

}  // namespace

int main() {
  lab::print_divider("Move semantics and avoiding unnecessary copies");

  {
    Counters counters;
    std::vector<TrackedBuffer> values;
    values.reserve(3);

    TrackedBuffer heavy(1 << 20, &counters);
    values.push_back(heavy);
    values.push_back(std::move(heavy));
    values.emplace_back(1 << 20, &counters);

    print_counters("push_back(lvalue) vs push_back(move) vs emplace_back", counters);
  }

  {
    Counters counters;
    auto produced = build_buffer(1 << 20, counters);
    std::cout << "returned buffer size=" << produced.size() << "\n";
    print_counters("return local object", counters);
  }

  {
    Counters counters;
    std::vector<TrackedBuffer> values;
    for (int i = 0; i < 8; ++i) {
      values.emplace_back(1 << 18, &counters);
    }
    print_counters("vector growth without reserve", counters);
  }

  {
    Counters counters;
    std::vector<TrackedBuffer> values;
    values.reserve(8);
    for (int i = 0; i < 8; ++i) {
      values.emplace_back(1 << 18, &counters);
    }
    print_counters("vector growth with reserve", counters);
  }

  std::cout << "\nInterview cue: move does not mean zero cost. It means resource ownership is"
               " transferred instead of deep-copied when the type supports it.\n";
  return 0;
}
