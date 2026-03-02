#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <exception>
#include <future>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#include <boost/multiprecision/cpp_int.hpp>
#include <sched.h>

namespace runtime {

unsigned detect_available_cores() {
    cpu_set_t set;
    CPU_ZERO(&set);
    if (sched_getaffinity(0, sizeof(set), &set) == 0) {
        const int count = CPU_COUNT(&set);
        if (count > 0) return static_cast<unsigned>(count);
    }
    const unsigned hc = std::thread::hardware_concurrency();
    return hc == 0u ? 1u : hc;
}

}  // namespace runtime

namespace mersenne {
using boost::multiprecision::cpp_int;

bool is_prime_exponent(uint32_t n) {
    if (n < 2u) return false;
    if (n == 2u) return true;
    if ((n & 1u) == 0u) return false;
    if (n % 5u == 0u && n != 5u) return false;
    for (uint32_t i = 3u; i <= (n / i); i += 2u) {
        if (n % i == 0u) return false;
    }
    return true;
}

cpp_int mersenne_mod(cpp_int x, uint32_t p, const cpp_int& mask) {
    while (x > mask) {
        x = (x & mask) + (x >> p);
    }
    if (x == mask) return 0;
    return x;
}

bool lucas_lehmer(uint32_t p, bool progress) {
    if (p < 2u) return false;
    if (p == 2u) return true;
    if ((p & 1u) == 0u) return false;
    if (!is_prime_exponent(p)) return false;

    const cpp_int mask = (cpp_int(1) << p) - 1;
    cpp_int s = 4;
    const uint32_t iters = p - 2u;

    const std::time_t start = std::time(nullptr);
    for (uint32_t i = 0; i < iters; ++i) {
        s = mersenne_mod(s * s - 2, p, mask);
        if (progress && ((i + 1u) % 10000u == 0u)) {
            const std::time_t now = std::time(nullptr);
            const double elapsed = std::difftime(now, start);
            const double avg = elapsed / static_cast<double>(i + 1u);
            const double rem = avg * static_cast<double>(iters - (i + 1u));
            std::printf("  Iteration %u/%u, elapsed: %.1f s, est. remaining: %.1f s\n", i + 1u, iters, elapsed, rem);
        }
    }
    return s == 0;
}

const std::vector<uint32_t>& known_mersenne_prime_exponents() {
    static const std::vector<uint32_t> exponents = {
        2, 3, 5, 7, 13, 17, 19, 31, 61, 89, 107, 127, 521, 607, 1279, 2203,
        2281, 3217, 4253, 4423, 9689, 9941, 11213, 19937, 21701, 23209, 44497,
        86243, 110503, 132049, 216091, 756839, 859433, 1257787, 1398269,
        2976221, 3021377, 6972593, 13466917, 20996011, 24036583, 25964951,
        30402457, 32582657, 37156667, 42643801, 43112609, 57885161, 74207281,
        77232917, 82589933, 136279841,
    };
    return exponents;
}

}  // namespace mersenne

#ifndef BIGNUM_NO_MAIN
int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);

    const unsigned maxCores = runtime::detect_available_cores();

    size_t startIndex = 0u;
    unsigned threads = maxCores;
    bool progress = false;

    if (argc >= 2 && argv[1] && argv[1][0] != '\0') {
        startIndex = static_cast<size_t>(std::strtoull(argv[1], nullptr, 10));
    }
    if (argc >= 3 && argv[2] && argv[2][0] != '\0') {
        const unsigned requested = static_cast<unsigned>(std::strtoull(argv[2], nullptr, 10));
        threads = requested == 0u ? maxCores : std::min(requested, maxCores);
    }
    if (argc >= 4) {
        progress = true;
    }

    const bool stopAfterOne = []() -> bool {
        const char* s = std::getenv("LL_STOP_AFTER_ONE");
        if (!s || *s == '\0') return false;
        return std::strtoull(s, nullptr, 10) != 0ull;
    }();

    const auto& exponents = mersenne::known_mersenne_prime_exponents();
    if (startIndex >= exponents.size()) {
        std::fprintf(stderr, "Invalid start index: %zu (max %zu)\n", startIndex, exponents.size() - 1u);
        return 1;
    }

    std::printf("Using %u worker(s) out of %u available core(s).\n", threads, maxCores);

    if (stopAfterOne || threads == 1u) {
        const uint32_t p = exponents[startIndex];
        std::printf("Testing M_%u ...\n", p);
        const auto t0 = std::chrono::steady_clock::now();
        const bool isPrime = mersenne::lucas_lehmer(p, progress);
        const auto t1 = std::chrono::steady_clock::now();
        const std::chrono::duration<double> elapsed = t1 - t0;
        std::printf("M_%u is %s. Time: %.3f s\n", p, isPrime ? "prime" : "composite", elapsed.count());
        return isPrime ? 0 : 2;
    }

    std::atomic<size_t> next{startIndex};
    std::mutex printMutex;
    std::vector<std::thread> workers;
    workers.reserve(threads);

    for (unsigned t = 0; t < threads; ++t) {
        workers.emplace_back([&]() {
            for (;;) {
                const size_t idx = next.fetch_add(1u, std::memory_order_relaxed);
                if (idx >= exponents.size()) break;
                const uint32_t p = exponents[idx];
                const auto t0 = std::chrono::steady_clock::now();
                const bool isPrime = mersenne::lucas_lehmer(p, true);
                const auto t1 = std::chrono::steady_clock::now();
                const std::chrono::duration<double> elapsed = t1 - t0;
                std::lock_guard<std::mutex> lock(printMutex);
                std::printf("Testing M_%u ...\n", p);
                std::printf("M_%u is %s. Time: %.3f s\n", p, isPrime ? "prime" : "composite", elapsed.count());
            }
        });
    }

    for (auto& th : workers) th.join();
    return 0;
}
#endif
