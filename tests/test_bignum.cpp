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

// ---- sweep::generate_natural tests ----
static void test_sweep_natural() {
    // Basic range
    const auto v = sweep::generate_natural(5u, 10u);
    assert((v == std::vector<uint32_t>{5, 6, 7, 8, 9, 10}));

    // Single element
    const auto v1 = sweep::generate_natural(7u, 7u);
    assert((v1 == std::vector<uint32_t>{7}));

    // Empty (min > max)
    const auto v2 = sweep::generate_natural(10u, 5u);
    assert(v2.empty());
}

// ---- sweep::generate_prime tests ----
static void test_sweep_prime() {
    // Primes in [5, 20]
    const auto v = sweep::generate_prime(5u, 20u);
    assert((v == std::vector<uint32_t>{5, 7, 11, 13, 17, 19}));

    // Range starting below 2 – should produce primes from 2
    const auto v2 = sweep::generate_prime(0u, 10u);
    assert((v2 == std::vector<uint32_t>{2, 3, 5, 7}));

    // Empty range
    const auto v3 = sweep::generate_prime(14u, 16u);
    assert(v3.empty());  // 15 is 3*5, no primes in [14,16]? 14=2*7, 15=3*5, 16=2^4 → empty
}

// ---- sweep::generate_mersenne_first tests ----
static void test_sweep_mersenne_first() {
    // Range [2, 30]: known Mersenne primes = {2,3,5,7,13,17,19} (31 > 30)
    // Remaining primes in [2,30] not in {2,3,5,7,13,17,19}: {11,23,29}
    const auto v = sweep::generate_mersenne_first(2u, 30u);
    assert(v.size() == 10u);
    // First 7 must be the Mersenne primes
    assert(v[0] == 2u && v[1] == 3u && v[2] == 5u && v[3] == 7u);
    assert(v[4] == 13u && v[5] == 17u && v[6] == 19u);
    // Remaining three are the non-Mersenne primes in ascending order
    const std::vector<uint32_t> tail(v.begin() + 7, v.end());
    assert((tail == std::vector<uint32_t>{11, 23, 29}));

    // All returned values are prime
    for (uint32_t p : v)
        assert(mersenne::is_prime_exponent(p));

    // No duplicates
    std::vector<uint32_t> sorted_v = v;
    std::sort(sorted_v.begin(), sorted_v.end());
    for (size_t i = 1; i < sorted_v.size(); ++i)
        assert(sorted_v[i] != sorted_v[i - 1]);
}

// ---- sweep::apply_shard tests ----
static void test_sweep_apply_shard() {
    const std::vector<uint32_t> items = {10, 20, 30, 40, 50, 60};

    // shard_count == 1 → identity
    assert(sweep::apply_shard(items, 0u, 1u) == items);

    // 2 shards
    const auto s0 = sweep::apply_shard(items, 0u, 2u);
    const auto s1 = sweep::apply_shard(items, 1u, 2u);
    assert((s0 == std::vector<uint32_t>{10, 30, 50}));
    assert((s1 == std::vector<uint32_t>{20, 40, 60}));

    // 3 shards
    const auto t0 = sweep::apply_shard(items, 0u, 3u);
    const auto t1 = sweep::apply_shard(items, 1u, 3u);
    const auto t2 = sweep::apply_shard(items, 2u, 3u);
    assert((t0 == std::vector<uint32_t>{10, 40}));
    assert((t1 == std::vector<uint32_t>{20, 50}));
    assert((t2 == std::vector<uint32_t>{30, 60}));

    // Union of all shards = original set
    std::vector<uint32_t> union3;
    for (const auto& s : {t0, t1, t2})
        union3.insert(union3.end(), s.begin(), s.end());
    std::sort(union3.begin(), union3.end());
    assert(union3 == items);

    // Empty input
    assert(sweep::apply_shard({}, 0u, 3u).empty());
}

// ---- sweep determinism: mersenne-first ordering is stable ----
static void test_sweep_mersenne_first_determinism() {
    // Two calls with the same parameters must return identical results.
    const auto v1 = sweep::generate_mersenne_first(2u, 1000u);
    const auto v2 = sweep::generate_mersenne_first(2u, 1000u);
    assert(v1 == v2);
    assert(!v1.empty());
    // First element must be the first known Mersenne prime in range (2).
    assert(v1[0] == 2u);
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
    assert(lucas_lehmer(7, false));
    assert(lucas_lehmer(13, false));
    assert(lucas_lehmer(17, false));
    assert(lucas_lehmer(19, false));
    assert(lucas_lehmer(31, false));
    assert(lucas_lehmer(61, false));
    assert(lucas_lehmer(89, false));
    assert(lucas_lehmer(107, false));
    assert(lucas_lehmer(127, false));
    assert(lucas_lehmer(11, false) == false);   // M_11 is composite
    assert(lucas_lehmer(23, false) == false);   // M_23 is composite

    // --- lucas_lehmer: medium cases via LimbBackend (128 <= p < ~4000) ---
    assert(lucas_lehmer(521,  false, /*benchmark_mode=*/true));
    assert(lucas_lehmer(607,  false, /*benchmark_mode=*/true));
    assert(lucas_lehmer(1279, false, /*benchmark_mode=*/true));
    assert(lucas_lehmer(2203, false, /*benchmark_mode=*/true));
    assert(lucas_lehmer(2281, false, /*benchmark_mode=*/true));
    assert(lucas_lehmer(3217, false, /*benchmark_mode=*/true));

    // --- lucas_lehmer: large cases via FftMersenneBackend (p >= ~4000) ---
    assert(lucas_lehmer(4253, false, /*benchmark_mode=*/true));
    assert(lucas_lehmer(4423, false, /*benchmark_mode=*/true));

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

    // --- Sweep-mode generator tests ---
    test_sweep_natural();
    test_sweep_prime();
    test_sweep_mersenne_first();
    test_sweep_apply_shard();
    test_sweep_mersenne_first_determinism();
    test_precharge_distribution();

    std::cout << "All tests passed\n";
    return 0;
}
