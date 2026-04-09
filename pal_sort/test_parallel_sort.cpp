#define PSORT_NO_MAIN
#include "parallel_sort.cpp"

#include <exception>

namespace {

template <typename T>
void expect_sorted_case(const std::string& name, std::vector<T> input, std::size_t threads) {
    std::vector<T> expected = input;
    std::sort(expected.begin(), expected.end());

    psort::parallel_radix_sort<T>(input, threads);

    if (input != expected) {
        std::cerr << "[FAIL] " << name << "\n";
        std::exit(1);
    }

    std::cout << "[PASS] " << name << "\n";
}

template <typename T>
void expect_random_case(const std::string& name, std::size_t n, std::size_t threads, std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::vector<T> input;

    if constexpr (std::is_same_v<T, std::uint32_t>) {
        input = psort::generate_data<T>(n, [&] { return static_cast<T>(rng()); });
    } else if constexpr (std::is_same_v<T, std::uint64_t>) {
        input = psort::generate_data<T>(n, [&] { return static_cast<T>(rng()); });
    } else if constexpr (std::is_same_v<T, std::int32_t>) {
        std::uniform_int_distribution<std::int32_t> dist(std::numeric_limits<std::int32_t>::min(),
                                                         std::numeric_limits<std::int32_t>::max());
        input = psort::generate_data<T>(n, [&] { return dist(rng); });
    } else if constexpr (std::is_same_v<T, std::int64_t>) {
        std::uniform_int_distribution<std::int64_t> dist(std::numeric_limits<std::int64_t>::min(),
                                                         std::numeric_limits<std::int64_t>::max());
        input = psort::generate_data<T>(n, [&] { return dist(rng); });
    } else if constexpr (std::is_same_v<T, float>) {
        std::uniform_real_distribution<float> dist(-1.0e6f, 1.0e6f);
        input = psort::generate_data<T>(n, [&] { return dist(rng); });
    } else if constexpr (std::is_same_v<T, double>) {
        std::uniform_real_distribution<double> dist(-1.0e12, 1.0e12);
        input = psort::generate_data<T>(n, [&] { return dist(rng); });
    }

    expect_sorted_case<T>(name, std::move(input), threads);
}

void run_correctness_suite(std::size_t threads) {
    expect_sorted_case<std::uint64_t>("u64 empty", {}, threads);
    expect_sorted_case<std::uint64_t>("u64 single", {42}, threads);
    expect_sorted_case<std::uint64_t>("u64 repeated", {7, 7, 7, 7, 7}, threads);
    expect_sorted_case<std::int32_t>("i32 mixed", {5, -1, 3, 0, -9, 3, 8, -2}, threads);
    expect_sorted_case<float>("f32 mixed", {3.5f, -0.0f, 7.25f, -19.0f, 3.5f, 2.0f}, threads);
    expect_random_case<std::uint32_t>("u32 random 1M", 1'000'000, threads, 1);
    expect_random_case<std::uint64_t>("u64 random 1M", 1'000'000, threads, 2);
    expect_random_case<std::int64_t>("i64 random 1M", 1'000'000, threads, 3);
    expect_random_case<double>("f64 random 1M", 1'000'000, threads, 4);
}

void run_benchmark(std::size_t n, std::size_t threads, std::size_t rounds) {
    std::mt19937_64 rng(123456789ull);
    auto input = psort::generate_data<std::uint64_t>(n, [&] { return static_cast<std::uint64_t>(rng()); });

    const double std_ms = psort::benchmark_ms<std::uint64_t>(input, rounds, [](std::vector<std::uint64_t>& values) {
        std::sort(values.begin(), values.end());
    });

    const double psort_ms = psort::benchmark_ms<std::uint64_t>(input, rounds, [&](std::vector<std::uint64_t>& values) {
        psort::parallel_radix_sort<std::uint64_t>(values, threads);
    });

    std::cout << "\nBenchmark\n";
    std::cout << "n=" << n << ", threads=" << threads << ", rounds=" << rounds << "\n";
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "std::sort(best):      " << std_ms << " ms\n";
    std::cout << "parallel_radix(best): " << psort_ms << " ms\n";
    std::cout << "speedup:              " << (std_ms / psort_ms) << "x\n";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const std::size_t threads = argc > 1 ? static_cast<std::size_t>(std::stoull(argv[1])) : std::thread::hardware_concurrency();
        const std::size_t n = argc > 2 ? static_cast<std::size_t>(std::stoull(argv[2])) : 5'000'000ull;
        const std::size_t rounds = argc > 3 ? static_cast<std::size_t>(std::stoull(argv[3])) : 3ull;

        run_correctness_suite(threads == 0 ? 1 : threads);
        run_benchmark(n, threads == 0 ? 1 : threads, rounds);
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "error: " << ex.what() << "\n";
        std::cerr << "usage: test_parallel_sort.exe [threads] [n] [rounds]\n";
        return 1;
    }
}
