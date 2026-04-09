#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mutex>
#include <numeric>
#include <random>
#include <span>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

namespace psort {

struct SortProfile {
    double total_ms = 0.0;
    double small_sort_ms = 0.0;
    double buffer_copy_in_ms = 0.0;
    double buffer_copy_out_ms = 0.0;
    double histogram_ms = 0.0;
    double histogram_worker_ms = 0.0;
    double prefix_sum_ms = 0.0;
    double scatter_ms = 0.0;
    double scatter_worker_ms = 0.0;
    double parallel_region_ms = 0.0;
    std::size_t passes = 0;
    std::size_t threads = 0;
    std::size_t elements = 0;
};

inline double elapsed_ms(std::chrono::steady_clock::time_point begin,
                         std::chrono::steady_clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - begin).count();
}

inline void accumulate_profile(SortProfile& dst, const SortProfile& src) {
    dst.total_ms += src.total_ms;
    dst.small_sort_ms += src.small_sort_ms;
    dst.buffer_copy_in_ms += src.buffer_copy_in_ms;
    dst.buffer_copy_out_ms += src.buffer_copy_out_ms;
    dst.histogram_ms += src.histogram_ms;
    dst.histogram_worker_ms += src.histogram_worker_ms;
    dst.prefix_sum_ms += src.prefix_sum_ms;
    dst.scatter_ms += src.scatter_ms;
    dst.scatter_worker_ms += src.scatter_worker_ms;
    dst.parallel_region_ms += src.parallel_region_ms;
    dst.passes += src.passes;
    dst.threads = src.threads;
    dst.elements = src.elements;
}

inline void divide_profile(SortProfile& profile, double divisor) {
    if (divisor <= 0.0) {
        return;
    }
    profile.total_ms /= divisor;
    profile.small_sort_ms /= divisor;
    profile.buffer_copy_in_ms /= divisor;
    profile.buffer_copy_out_ms /= divisor;
    profile.histogram_ms /= divisor;
    profile.histogram_worker_ms /= divisor;
    profile.prefix_sum_ms /= divisor;
    profile.scatter_ms /= divisor;
    profile.scatter_worker_ms /= divisor;
    profile.parallel_region_ms /= divisor;
    profile.passes = static_cast<std::size_t>(profile.passes / divisor);
}

inline void print_profile(const SortProfile& profile, const std::string& label) {
    const double total = profile.total_ms > 0.0 ? profile.total_ms : 1.0;
    auto pct = [&](double ms) { return 100.0 * ms / total; };

    std::cout << label << " profile\n";
    std::cout << "  total:           " << profile.total_ms << " ms\n";
    std::cout << "  copy_in:         " << profile.buffer_copy_in_ms << " ms (" << pct(profile.buffer_copy_in_ms) << "%)\n";
    std::cout << "  histogram:       " << profile.histogram_ms << " ms (" << pct(profile.histogram_ms) << "%)\n";
    std::cout << "  histogram_work:  " << profile.histogram_worker_ms << " ms (thread-sum)\n";
    std::cout << "  prefix_sum:      " << profile.prefix_sum_ms << " ms (" << pct(profile.prefix_sum_ms) << "%)\n";
    std::cout << "  scatter:         " << profile.scatter_ms << " ms (" << pct(profile.scatter_ms) << "%)\n";
    std::cout << "  scatter_work:    " << profile.scatter_worker_ms << " ms (thread-sum)\n";
    std::cout << "  copy_out:        " << profile.buffer_copy_out_ms << " ms (" << pct(profile.buffer_copy_out_ms) << "%)\n";
    std::cout << "  parallel_region: " << profile.parallel_region_ms << " ms (" << pct(profile.parallel_region_ms) << "%)\n";
    if (profile.small_sort_ms > 0.0) {
        std::cout << "  small_sort:      " << profile.small_sort_ms << " ms (" << pct(profile.small_sort_ms) << "%)\n";
    }
    std::cout << "  passes:          " << profile.passes << "\n";
    std::cout << "  threads:         " << profile.threads << "\n";
    std::cout << "  elements:        " << profile.elements << "\n";
}

class ChunkThreadPool {
public:
    explicit ChunkThreadPool(std::size_t worker_count) : worker_count_(worker_count) {
        workers_.reserve(worker_count_);
        for (std::size_t worker_index = 0; worker_index < worker_count_; ++worker_index) {
            workers_.emplace_back([this, worker_index] { worker_loop(worker_index + 1); });
        }
    }

    ~ChunkThreadPool() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stop_ = true;
            ++generation_;
        }
        cv_work_.notify_all();
        for (auto& worker : workers_) {
            worker.join();
        }
    }

    ChunkThreadPool(const ChunkThreadPool&) = delete;
    ChunkThreadPool& operator=(const ChunkThreadPool&) = delete;

    std::size_t capacity() const noexcept {
        return worker_count_ + 1;
    }

    template <typename Fn>
    double parallel_for(std::size_t count, std::size_t threads, Fn&& fn) {
        if (count == 0) {
            return 0.0;
        }

        threads = std::max<std::size_t>(1, std::min<std::size_t>(threads, count));
        const std::size_t chunk = (count + threads - 1) / threads;
        const auto region_begin = std::chrono::steady_clock::now();

        if (threads == 1 || worker_count_ == 0) {
            fn(0, 0, count);
            return elapsed_ms(region_begin, std::chrono::steady_clock::now());
        }

        {
            std::lock_guard<std::mutex> lock(mutex_);
            active_threads_ = threads;
            chunk_size_ = chunk;
            count_ = count;
            pending_workers_ = threads - 1;
            current_fn_ = std::forward<Fn>(fn);
            ++generation_;
        }
        cv_work_.notify_all();

        fn(0, 0, std::min(count, chunk));

        std::unique_lock<std::mutex> lock(mutex_);
        cv_done_.wait(lock, [&] { return pending_workers_ == 0; });
        current_fn_ = {};
        return elapsed_ms(region_begin, std::chrono::steady_clock::now());
    }

private:
    void worker_loop(std::size_t tid) {
        std::size_t seen_generation = 0;

        while (true) {
            std::function<void(std::size_t, std::size_t, std::size_t)> fn;
            std::size_t begin = 0;
            std::size_t end = 0;

            {
                std::unique_lock<std::mutex> lock(mutex_);
                cv_work_.wait(lock, [&] { return stop_ || generation_ != seen_generation; });
                if (stop_) {
                    return;
                }

                seen_generation = generation_;
                if (tid >= active_threads_) {
                    continue;
                }

                const std::size_t chunk_index = tid;
                begin = chunk_index * chunk_size_;
                end = std::min(count_, begin + chunk_size_);
                fn = current_fn_;
            }

            if (begin < end) {
                fn(tid, begin, end);
            }

            {
                std::lock_guard<std::mutex> lock(mutex_);
                if (--pending_workers_ == 0) {
                    cv_done_.notify_one();
                }
            }
        }
    }

    std::size_t worker_count_ = 0;
    std::vector<std::thread> workers_;
    std::mutex mutex_;
    std::condition_variable cv_work_;
    std::condition_variable cv_done_;
    bool stop_ = false;
    std::size_t generation_ = 0;
    std::size_t active_threads_ = 1;
    std::size_t pending_workers_ = 0;
    std::size_t chunk_size_ = 0;
    std::size_t count_ = 0;
    std::function<void(std::size_t, std::size_t, std::size_t)> current_fn_;
};

inline ChunkThreadPool& global_pool() {
    static ChunkThreadPool pool(std::max<std::size_t>(1, std::thread::hardware_concurrency()) - 1);
    return pool;
}

inline std::size_t choose_threads(std::size_t n, std::size_t requested_threads = std::thread::hardware_concurrency()) {
    constexpr std::size_t kMinElementsPerThread = 1u << 16;
    const std::size_t hw_threads = std::max<std::size_t>(1, global_pool().capacity());
    const std::size_t limit = requested_threads == 0 ? hw_threads : std::min(requested_threads, hw_threads);
    const std::size_t data_limited = std::max<std::size_t>(1, (n + kMinElementsPerThread - 1) / kMinElementsPerThread);
    return std::max<std::size_t>(1, std::min(limit, data_limited));
}

inline void warmup_thread_pool() {
    global_pool();
}

template <typename T>
struct KeyTransform;

template <>
struct KeyTransform<std::uint32_t> {
    using key_type = std::uint32_t;
    static key_type encode(std::uint32_t value) noexcept { return value; }
};

template <>
struct KeyTransform<std::uint64_t> {
    using key_type = std::uint64_t;
    static key_type encode(std::uint64_t value) noexcept { return value; }
};

template <>
struct KeyTransform<std::int32_t> {
    using key_type = std::uint32_t;
    static key_type encode(std::int32_t value) noexcept {
        return static_cast<key_type>(value) ^ 0x80000000u;
    }
};

template <>
struct KeyTransform<std::int64_t> {
    using key_type = std::uint64_t;
    static key_type encode(std::int64_t value) noexcept {
        return static_cast<key_type>(value) ^ 0x8000000000000000ull;
    }
};

template <>
struct KeyTransform<float> {
    using key_type = std::uint32_t;
    static key_type encode(float value) noexcept {
        const key_type bits = std::bit_cast<key_type>(value);
        return (bits & 0x80000000u) ? ~bits : (bits ^ 0x80000000u);
    }
};

template <>
struct KeyTransform<double> {
    using key_type = std::uint64_t;
    static key_type encode(double value) noexcept {
        const key_type bits = std::bit_cast<key_type>(value);
        return (bits & 0x8000000000000000ull) ? ~bits : (bits ^ 0x8000000000000000ull);
    }
};

template <typename T>
inline constexpr bool kSupportedType =
    std::is_same_v<T, std::uint32_t> || std::is_same_v<T, std::uint64_t> ||
    std::is_same_v<T, std::int32_t> || std::is_same_v<T, std::int64_t> ||
    std::is_same_v<T, float> || std::is_same_v<T, double>;

template <typename T>
void parallel_radix_sort(std::span<T> data,
                         std::size_t requested_threads = std::thread::hardware_concurrency(),
                         SortProfile* profile = nullptr) {
    static_assert(kSupportedType<T>, "parallel_radix_sort only supports uint32_t, uint64_t, int32_t, int64_t, float, double");

    constexpr std::size_t kBins = 256;
    constexpr std::size_t kRadixBits = 8;
    constexpr std::size_t kSmallThreshold = 1u << 15;

    const bool profiling_enabled = profile != nullptr;
    SortProfile local_profile {};
    local_profile.elements = data.size();
    local_profile.threads = requested_threads == 0 ? 1 : requested_threads;
    const auto total_begin = profiling_enabled ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point {};

    const std::size_t n = data.size();
    if (n <= 1) {
        if (profiling_enabled) {
            local_profile.total_ms = elapsed_ms(total_begin, std::chrono::steady_clock::now());
        }
        if (profile) {
            *profile = local_profile;
        }
        return;
    }

    if (n < kSmallThreshold) {
        const auto begin = profiling_enabled ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point {};
        std::sort(data.begin(), data.end());
        if (profiling_enabled) {
            local_profile.small_sort_ms = elapsed_ms(begin, std::chrono::steady_clock::now());
            local_profile.total_ms = elapsed_ms(total_begin, std::chrono::steady_clock::now());
        }
        if (profile) {
            *profile = local_profile;
        }
        return;
    }

    const std::size_t threads = choose_threads(n, requested_threads);
    local_profile.threads = threads;

    using key_type = typename KeyTransform<T>::key_type;
    constexpr std::size_t kPasses = sizeof(key_type);
    local_profile.passes = kPasses;

    const auto copy_in_begin = profiling_enabled ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point {};
    std::vector<T> scratch(n);
    std::vector<T>* src = nullptr;
    std::vector<T>* dst = nullptr;

    std::vector<T> source(data.begin(), data.end());
    if (profiling_enabled) {
        local_profile.buffer_copy_in_ms = elapsed_ms(copy_in_begin, std::chrono::steady_clock::now());
    }
    src = &source;
    dst = &scratch;

    std::vector<std::size_t> histogram(threads * kBins);
    std::vector<std::size_t> thread_offsets(threads * kBins);
    std::array<std::size_t, kBins> global_offsets {};

    for (std::size_t pass = 0; pass < kPasses; ++pass) {
        const std::size_t shift = pass * kRadixBits;
        std::fill(histogram.begin(), histogram.end(), 0);

        const double histogram_region_ms = global_pool().parallel_for(n, threads, [&](std::size_t tid, std::size_t begin, std::size_t end) {
            auto* local_hist = histogram.data() + tid * kBins;
            const auto phase_begin = profiling_enabled ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point {};
            for (std::size_t i = begin; i < end; ++i) {
                const auto key = KeyTransform<T>::encode((*src)[i]);
                const std::size_t bucket = static_cast<std::size_t>((key >> shift) & 0xFFu);
                ++local_hist[bucket];
            }
            if (profiling_enabled) {
                const double ms = elapsed_ms(phase_begin, std::chrono::steady_clock::now());
                static std::mutex histogram_lock;
                std::lock_guard<std::mutex> lock(histogram_lock);
                local_profile.histogram_worker_ms += ms;
            }
        });
        if (profiling_enabled) {
            local_profile.histogram_ms += histogram_region_ms;
            local_profile.parallel_region_ms += histogram_region_ms;
        }

        const auto prefix_begin = profiling_enabled ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point {};
        std::size_t prefix = 0;
        for (std::size_t bucket = 0; bucket < kBins; ++bucket) {
            global_offsets[bucket] = prefix;
            for (std::size_t tid = 0; tid < threads; ++tid) {
                const std::size_t index = tid * kBins + bucket;
                thread_offsets[index] = prefix;
                prefix += histogram[index];
            }
        }
        if (profiling_enabled) {
            local_profile.prefix_sum_ms += elapsed_ms(prefix_begin, std::chrono::steady_clock::now());
        }

        const double scatter_region_ms = global_pool().parallel_for(n, threads, [&](std::size_t tid, std::size_t begin, std::size_t end) {
            std::array<std::size_t, kBins> local_offsets {};
            const auto base = tid * kBins;
            const auto phase_begin = profiling_enabled ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point {};
            for (std::size_t bucket = 0; bucket < kBins; ++bucket) {
                local_offsets[bucket] = thread_offsets[base + bucket];
            }

            for (std::size_t i = begin; i < end; ++i) {
                const auto value = (*src)[i];
                const auto key = KeyTransform<T>::encode(value);
                const std::size_t bucket = static_cast<std::size_t>((key >> shift) & 0xFFu);
                (*dst)[local_offsets[bucket]++] = value;
            }
            if (profiling_enabled) {
                const double ms = elapsed_ms(phase_begin, std::chrono::steady_clock::now());
                static std::mutex scatter_lock;
                std::lock_guard<std::mutex> lock(scatter_lock);
                local_profile.scatter_worker_ms += ms;
            }
        });
        if (profiling_enabled) {
            local_profile.scatter_ms += scatter_region_ms;
            local_profile.parallel_region_ms += scatter_region_ms;
        }

        std::swap(src, dst);
    }

    const auto copy_out_begin = profiling_enabled ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point {};
    std::copy(src->begin(), src->end(), data.begin());
    if (profiling_enabled) {
        local_profile.buffer_copy_out_ms = elapsed_ms(copy_out_begin, std::chrono::steady_clock::now());
        local_profile.total_ms = elapsed_ms(total_begin, std::chrono::steady_clock::now());
    }
    if (profile) {
        *profile = local_profile;
    }
}

template <typename T>
bool same_as_std_sorted(const std::vector<T>& input, std::size_t threads) {
    std::vector<T> expected = input;
    std::vector<T> actual = input;
    std::sort(expected.begin(), expected.end());
    parallel_radix_sort<T>(actual, threads);
    return expected == actual;
}

template <typename T, typename Generator>
std::vector<T> generate_data(std::size_t n, Generator&& gen) {
    std::vector<T> values(n);
    for (auto& value : values) {
        value = gen();
    }
    return values;
}

template <typename T>
double benchmark_ms(const std::vector<T>& input, std::size_t rounds, const std::function<void(std::vector<T>&)>& sorter) {
    double best_ms = std::numeric_limits<double>::max();

    for (std::size_t i = 0; i < rounds; ++i) {
        std::vector<T> values = input;
        const auto start = std::chrono::steady_clock::now();
        sorter(values);
        const auto end = std::chrono::steady_clock::now();
        const double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
        best_ms = std::min(best_ms, elapsed_ms);
    }

    return best_ms;
}

template <typename T>
void run_case(std::size_t n, std::size_t threads, std::size_t rounds) {
    std::mt19937_64 rng(123456789ull);

    std::vector<T> input;
    if constexpr (std::is_same_v<T, std::uint32_t>) {
        input = generate_data<T>(n, [&] { return static_cast<T>(rng()); });
    } else if constexpr (std::is_same_v<T, std::uint64_t>) {
        input = generate_data<T>(n, [&] { return static_cast<T>(rng()); });
    } else if constexpr (std::is_same_v<T, std::int32_t>) {
        std::uniform_int_distribution<std::int32_t> dist(std::numeric_limits<std::int32_t>::min(),
                                                         std::numeric_limits<std::int32_t>::max());
        input = generate_data<T>(n, [&] { return dist(rng); });
    } else if constexpr (std::is_same_v<T, std::int64_t>) {
        std::uniform_int_distribution<std::int64_t> dist(std::numeric_limits<std::int64_t>::min(),
                                                         std::numeric_limits<std::int64_t>::max());
        input = generate_data<T>(n, [&] { return dist(rng); });
    } else if constexpr (std::is_same_v<T, float>) {
        std::uniform_real_distribution<float> dist(-1.0e9f, 1.0e9f);
        input = generate_data<T>(n, [&] { return dist(rng); });
    } else if constexpr (std::is_same_v<T, double>) {
        std::uniform_real_distribution<double> dist(-1.0e12, 1.0e12);
        input = generate_data<T>(n, [&] { return dist(rng); });
    }

    const bool ok = same_as_std_sorted<T>(input, threads);
    if (!ok) {
        std::cerr << "verification failed\n";
        std::exit(2);
    }

    const double std_ms = benchmark_ms<T>(input, rounds, [](std::vector<T>& values) {
        std::sort(values.begin(), values.end());
    });

    const double psort_ms = benchmark_ms<T>(input, rounds, [&](std::vector<T>& values) {
        parallel_radix_sort<T>(values, threads);
    });

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "n=" << n << ", threads=" << threads << ", rounds=" << rounds << "\n";
    std::cout << "std::sort(best):     " << std_ms << " ms\n";
    std::cout << "parallel_radix(best): " << psort_ms << " ms\n";
    std::cout << "speedup:             " << (std_ms / psort_ms) << "x\n";
}

}  // namespace psort

#ifndef PSORT_NO_MAIN
namespace {

class CPUSort {
public:
    explicit CPUSort(std::size_t threads) : requested_threads_(threads == 0 ? 1 : threads) {}

    void Warmup() const {
        psort::warmup_thread_pool();
    }

    void Sort(int* arr, int len, psort::SortProfile* profile = nullptr) const {
        psort::parallel_radix_sort<int>(std::span<int>(arr, static_cast<std::size_t>(len)), requested_threads_, profile);
    }

    std::size_t RequestedThreads() const {
        return requested_threads_;
    }

    std::size_t ActiveThreads(int len) const {
        return psort::choose_threads(static_cast<std::size_t>(len), requested_threads_);
    }

private:
    std::size_t requested_threads_;
};

void ShuffleArr(int* arr, int len, std::mt19937_64& rng) {
    std::shuffle(arr, arr + len, rng);
}

double benchmark_cpu_sort(std::vector<int>& arr, int len, int test_cnt, const CPUSort& sorter, std::mt19937_64& rng) {
    double best = std::numeric_limits<double>::max();

    for (int i = 0; i < test_cnt; ++i) {
        auto stt = std::chrono::high_resolution_clock::now();
        sorter.Sort(arr.data(), len);
        auto end = std::chrono::high_resolution_clock::now();
        const double seconds = std::chrono::duration<double>(end - stt).count();
        best = std::min(best, seconds);
        ShuffleArr(arr.data(), len, rng);
    }

    return best;
}

psort::SortProfile benchmark_cpu_sort_profile(std::vector<int>& arr, int len, int test_cnt, const CPUSort& sorter, std::mt19937_64& rng) {
    psort::SortProfile total_profile {};

    for (int i = 0; i < test_cnt; ++i) {
        psort::SortProfile run_profile {};
        sorter.Sort(arr.data(), len, &run_profile);
        psort::accumulate_profile(total_profile, run_profile);
        ShuffleArr(arr.data(), len, rng);
    }

    psort::divide_profile(total_profile, static_cast<double>(test_cnt));
    return total_profile;
}

double benchmark_std_sort(std::vector<int>& arr, int len, int test_cnt, std::mt19937_64& rng) {
    double best = std::numeric_limits<double>::max();

    for (int i = 0; i < test_cnt; ++i) {
        auto stt = std::chrono::high_resolution_clock::now();
        std::sort(arr.data(), arr.data() + len);
        auto end = std::chrono::high_resolution_clock::now();
        const double seconds = std::chrono::duration<double>(end - stt).count();
        best = std::min(best, seconds);
        ShuffleArr(arr.data(), len, rng);
    }

    return best;
}

bool verify_cpu_sort(std::vector<int> input, std::size_t threads) {
    auto expected = input;
    std::sort(expected.begin(), expected.end());
    CPUSort sorter(threads);
    sorter.Warmup();
    sorter.Sort(input.data(), static_cast<int>(input.size()));
    return input == expected;
}

}  // namespace

int main(int argc, char** argv) {
    const int initial_len = argc > 1 ? std::stoi(argv[1]) : 25'000'000;
    const std::size_t threads = argc > 2 ? static_cast<std::size_t>(std::stoull(argv[2])) : std::thread::hardware_concurrency();
    const int initial_test_cnt = argc > 3 ? std::stoi(argv[3]) : 2;

    if (initial_len <= 0 || initial_test_cnt <= 0) {
        std::cerr << "usage: parallel_sort.exe [initial_len] [threads] [initial_test_cnt]\n";
        return 1;
    }

    std::mt19937_64 rng(123456789ull);
    std::uniform_int_distribution<int> dist(0, 100000000);

    std::vector<int> arr(static_cast<std::size_t>(initial_len));
    for (auto& v : arr) {
        v = dist(rng);
    }

    if (!verify_cpu_sort(std::vector<int>(arr.begin(), arr.begin() + std::min<int>(initial_len, 1'000'000)), threads)) {
        std::cerr << "verification failed\n";
        return 2;
    }

    int len = initial_len;
    int test_cnt = initial_test_cnt;
    CPUSort sorter(threads);
    sorter.Warmup();

    std::cout << std::fixed << std::setprecision(6);
    while (len > 10000) {
        const std::size_t active_threads = sorter.ActiveThreads(len);
        const double cpu_seconds = benchmark_cpu_sort(arr, len, test_cnt, sorter, rng);
        std::cout << "CPUSort  time: " << cpu_seconds << " seconds, len: " << len
                  << ", tests: " << test_cnt << ", requested_threads: " << sorter.RequestedThreads()
                  << ", active_threads: " << active_threads << "\n";

        auto profiled_arr = arr;
        psort::SortProfile avg_profile = benchmark_cpu_sort_profile(profiled_arr, len, test_cnt, sorter, rng);
        psort::print_profile(avg_profile, "CPUSort");

        const double std_seconds = benchmark_std_sort(arr, len, test_cnt, rng);
        std::cout << "std::sort time: " << std_seconds << " seconds, len: " << len
                  << ", tests: " << test_cnt << "\n";

        if (cpu_seconds > 0.0) {
            std::cout << "speedup: " << (std_seconds / cpu_seconds) << "x\n";
        }
        std::cout << "\n";

        len /= 10;
        test_cnt *= 5;
    }

    return 0;
}
#endif
