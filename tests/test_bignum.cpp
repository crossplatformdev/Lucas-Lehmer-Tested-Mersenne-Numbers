#include <algorithm>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <vector>

#define BIGNUM_NO_MAIN
#include "../src/BigNum.cpp"

// Verify that precharge_work_matrix distributes all items across threads.
static void test_precharge_distribution() {
    const std::vector<uint32_t> exponents = {2, 3, 5, 7, 13, 17, 19, 31};
    const unsigned threads = 3u;
    const size_t startIndex = 1u; // skip first element

    const auto work_matrix = mersenne::precharge_work_matrix(exponents, startIndex, threads);

    // Total items distributed must equal exponents.size() - startIndex
    size_t total = 0;
    for (const auto& lane : work_matrix) total += lane.size();
    assert(total == exponents.size() - startIndex);

    // Every item from startIndex onwards must appear exactly once
    std::vector<uint32_t> collected;
    collected.reserve(total);
    for (const auto& lane : work_matrix)
        collected.insert(collected.end(), lane.begin(), lane.end());
    std::sort(collected.begin(), collected.end());
    const std::vector<uint32_t> expected(exponents.begin() + startIndex, exponents.end());
    assert(collected == expected);
}

int main() {
    using namespace mersenne;

    // --- is_prime_exponent ---
    assert(is_prime_exponent(2));
    assert(is_prime_exponent(3));
    assert(!is_prime_exponent(1));
    assert(!is_prime_exponent(9));

    // --- lucas_lehmer: small cases via GenericBackend (p < 128) ---
    assert(lucas_lehmer(2, false));
    assert(lucas_lehmer(3, false));
    assert(lucas_lehmer(5, false));
    assert(lucas_lehmer(11, false) == false);  // M_11 is composite

    // --- lucas_lehmer: medium cases via FftMersenneBackend ---
    assert(lucas_lehmer(521,  false, /*benchmark_mode=*/true));
    assert(lucas_lehmer(607,  false, /*benchmark_mode=*/true));
    assert(lucas_lehmer(1279, false, /*benchmark_mode=*/true));

    // --- lucas_lehmer: large case via FftMersenneBackend (p >= kLimbFftCrossover = 4000) ---
    // Ensures the updated FFT carry path is exercised in CI.
    assert(lucas_lehmer(4253, false, /*benchmark_mode=*/true));

    // --- lucas_lehmer: fast rejection for composite exponent p >= 128 ---
    assert(!lucas_lehmer(129, false));

    // --- runtime ---
    const unsigned cores = runtime::detect_available_cores();
    assert(cores >= 1u);

    // --- backend::choose_fft_length sanity ---
    // For p=9689, the minimum safe FFT length must be >= ceil(9689 / b_hi_max).
    {
        const size_t n = backend::choose_fft_length(9689);
        assert(n >= 1024u && (n & (n - 1)) == 0u);  // power-of-two, ≥ 1024
    }
    test_precharge_distribution();

    std::cout << "All tests passed\n";
    return 0;
}
