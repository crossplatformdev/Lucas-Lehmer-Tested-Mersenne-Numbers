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

// ---- discover-mode tests ----

static void test_is_known_mersenne_prime() {
    // Known entries must return true.
    assert(mersenne::is_known_mersenne_prime(2u));
    assert(mersenne::is_known_mersenne_prime(3u));
    assert(mersenne::is_known_mersenne_prime(136279841u));
    // Primes not in the list must return false.
    assert(!mersenne::is_known_mersenne_prime(11u));       // M_11 is composite
    assert(!mersenne::is_known_mersenne_prime(136279843u)); // > last known, not in list
    assert(!mersenne::is_known_mersenne_prime(0u));
    assert(!mersenne::is_known_mersenne_prime(1u));
    // Values above uint32 max can never be known.
    assert(!mersenne::is_known_mersenne_prime(UINT64_C(5000000000)));
}

static void test_generate_post_known_exponents() {
    // Use small range for speed: primes in (10, 30] are 11, 13, 17, 19, 23, 29.
    const auto v = mersenne::generate_post_known_exponents(UINT64_C(10), UINT64_C(30));
    const std::vector<uint64_t> expected = {11, 13, 17, 19, 23, 29};
    assert(v == expected);

    // Empty range.
    assert(mersenne::generate_post_known_exponents(UINT64_C(30), UINT64_C(30)).empty());
    assert(mersenne::generate_post_known_exponents(UINT64_C(30), UINT64_C(10)).empty());

    // Single element: primes in (28, 30] = {29}.
    const auto w = mersenne::generate_post_known_exponents(UINT64_C(28), UINT64_C(30));
    assert(w.size() == 1u && w[0] == 29u);
}

static void test_generate_post_known_exponents_u64_range() {
    // Verify generation works for values well above uint32 max.
    // Primes in (4294967311, 4294967331]: 4294967357 is outside, let's find actual primes.
    // We use a small range just above UINT32_MAX: (UINT32_MAX, UINT32_MAX+100].
    const uint64_t base = static_cast<uint64_t>(UINT32_MAX);
    const auto v = mersenne::generate_post_known_exponents(base, base + 100u);
    // All returned values must be > UINT32_MAX and prime.
    for (uint64_t p : v) {
        assert(p > base);
        assert(p <= base + 100u);
        assert(mersenne::is_prime_exponent(p));
    }
    // is_prime_exponent itself must handle uint64_t inputs.
    assert(mersenne::is_prime_exponent(UINT64_C(4294967311)));  // known prime > UINT32_MAX
    assert(!mersenne::is_prime_exponent(UINT64_C(4294967312))); // even
}

static void test_discover_exponent_list_ordering() {
    // single_exp=13 first, then range (10,30] minus 13.
    // Range primes in (10,30]: 11,13,17,19,23,29  → minus 13 → 11,17,19,23,29
    const auto v = mersenne::discover_exponent_list(UINT64_C(13), UINT64_C(10), UINT64_C(30));
    assert(!v.empty() && v[0] == 13u);
    // 13 must appear exactly once.
    assert(std::count(v.begin(), v.end(), UINT64_C(13)) == 1u);
    // Range part (all but first) must be ascending.
    for (size_t i = 2; i < v.size(); ++i)
        assert(v[i] > v[i - 1]);
}

static void test_discover_exponent_list_dedup() {
    // single_exp falls in the range: must appear only once and be first.
    const auto v = mersenne::discover_exponent_list(UINT64_C(17), UINT64_C(10), UINT64_C(30));
    assert(std::count(v.begin(), v.end(), UINT64_C(17)) == 1u);
    assert(v[0] == 17u);
}

static void test_discover_exponent_list_range_limiting() {
    // max_incl = 20: range primes in (10,20] = 11,13,17,19.
    const auto v = mersenne::discover_exponent_list(UINT64_C(0), UINT64_C(10), UINT64_C(20));
    for (uint64_t p : v) {
        assert(p > 10u && p <= 20u);
        assert(mersenne::is_prime_exponent(p));
    }
    // Ensure 23 and 29 are absent.
    assert(std::find(v.begin(), v.end(), UINT64_C(23)) == v.end());
    assert(std::find(v.begin(), v.end(), UINT64_C(29)) == v.end());
}

static void test_discover_exponent_list_sharding() {
    // Sharding partitions the range part (not single_exp) deterministically.
    // Range primes in (10,30]: 11,13,17,19,23,29 (6 items)
    std::vector<uint64_t> all_range_items;
    for (uint32_t si = 0; si < 3u; ++si) {
        const auto s = mersenne::discover_exponent_list(UINT64_C(0), UINT64_C(10), UINT64_C(30),
                                                        /*reverse=*/false,
                                                        /*shard_count=*/3u,
                                                        /*shard_index=*/si);
        all_range_items.insert(all_range_items.end(), s.begin(), s.end());
    }
    // Total items across all shards == full range size.
    const auto full = mersenne::generate_post_known_exponents(UINT64_C(10), UINT64_C(30));
    assert(all_range_items.size() == full.size());
    // Every item appears exactly once.
    std::sort(all_range_items.begin(), all_range_items.end());
    assert(all_range_items == full);
}

static void test_discover_exponent_list_reverse_order() {
    // Range part must be in descending order when reverse_order=true.
    const auto v = mersenne::discover_exponent_list(UINT64_C(0), UINT64_C(10), UINT64_C(30),
                                                    /*reverse=*/true,
                                                    /*shard_count=*/1u,
                                                    /*shard_index=*/0u);
    // v should be descending.
    for (size_t i = 1; i < v.size(); ++i)
        assert(v[i] < v[i - 1]);
}

static void test_discover_exponent_list_non_prime_single_exp() {
    // single_exp that is not prime must be excluded from the list.
    const auto v = mersenne::discover_exponent_list(UINT64_C(15), UINT64_C(10), UINT64_C(30)); // 15 = 3*5
    assert(std::find(v.begin(), v.end(), UINT64_C(15)) == v.end());
}

static void test_discover_exponent_list_post_known_defaults() {
    // Verify that a small range above 136279841 yields only primes > 136279841
    // and <= max_incl.
    const uint64_t min_excl = 136279841u;
    const uint64_t max_incl = 136279950u;
    const auto v = mersenne::discover_exponent_list(UINT64_C(0), min_excl, max_incl);
    for (uint64_t p : v) {
        assert(p > min_excl);
        assert(p <= max_incl);
        assert(mersenne::is_prime_exponent(p));
    }
    // None of the candidates should be in the known list (they're all > last known).
    for (uint64_t p : v)
        assert(!mersenne::is_known_mersenne_prime(p));
}

static void test_discovery_classification() {
    // Known exponent: is_known_mersenne_prime → true, treated as known.
    assert(mersenne::is_known_mersenne_prime(136279841u));
    // Exponent above the known list: not in list → new-discovery candidate.
    // (136279879 is prime and > 136279841, so it's not in the known list.)
    assert(mersenne::is_prime_exponent(UINT64_C(136279879)));
    assert(!mersenne::is_known_mersenne_prime(136279879u));
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

    // --- discover-mode tests ---
    test_is_known_mersenne_prime();
    test_generate_post_known_exponents();
    test_generate_post_known_exponents_u64_range();
    test_discover_exponent_list_ordering();
    test_discover_exponent_list_dedup();
    test_discover_exponent_list_range_limiting();
    test_discover_exponent_list_sharding();
    test_discover_exponent_list_reverse_order();
    test_discover_exponent_list_non_prime_single_exp();
    test_discover_exponent_list_post_known_defaults();
    test_discovery_classification();

    // --- power_bucket tests ---
    // 1. bucket_range boundary generation
    {
        // n=1 normalized to [2, 2]
        const auto r1 = power_bucket::bucket_range(1u);
        assert(r1.lo == 2u && r1.hi == 2u);
        // n=2: [2, 3]
        const auto r2 = power_bucket::bucket_range(2u);
        assert(r2.lo == 2u && r2.hi == 3u);
        // n=3: [4, 7]
        const auto r3 = power_bucket::bucket_range(3u);
        assert(r3.lo == 4u && r3.hi == 7u);
        // n=4: [8, 15]
        const auto r4 = power_bucket::bucket_range(4u);
        assert(r4.lo == 8u && r4.hi == 15u);
        // n=64: [2^63, UINT64_MAX]
        const auto r64 = power_bucket::bucket_range(64u);
        assert(r64.lo == (UINT64_C(1) << 63) && r64.hi == UINT64_MAX);
        // Invalid n=0 and n=65 return {0,0}
        assert(power_bucket::bucket_range(0u).lo == 0u);
        assert(power_bucket::bucket_range(65u).lo == 0u);
    }

    // 2. bucket normalization for n=1
    {
        const auto r = power_bucket::bucket_range(1u);
        assert(r.lo == 2u && r.hi == 2u);
        const auto v = power_bucket::enumerate_bucket_primes(1u);
        assert(v.size() == 1u && v[0] == 2u);
    }

    // 3. no overlap between non-trivial adjacent buckets (n >= 3)
    {
        for (uint32_t n = 3u; n < 32u; ++n) {
            const auto rn  = power_bucket::bucket_range(n);
            const auto rn1 = power_bucket::bucket_range(n + 1u);
            assert(rn.hi + 1u == rn1.lo);  // ranges are contiguous and non-overlapping
        }
    }

    // 4. full coverage: union of all 64 buckets covers [2, UINT64_MAX]
    {
        // B_1=[2,2], B_2=[2,3], B_3=[4,7], ..., B_64=[2^63,UINT64_MAX]
        // Verify B_2..B_64 partition [2, UINT64_MAX] with no gaps or overlaps.
        uint64_t expected_lo = 2u;
        for (uint32_t n = 2u; n <= 64u; ++n) {
            const auto r = power_bucket::bucket_range(n);
            assert(r.lo == expected_lo);
            if (n < 64u)
                expected_lo = r.hi + 1u;
            else
                assert(r.hi == UINT64_MAX);
        }
    }

    // 5. prime exponent enumeration inside each bucket (small buckets only)
    {
        // Bucket 2: [2, 3] → primes 2, 3
        const auto v2 = power_bucket::enumerate_bucket_primes(2u);
        assert((v2 == std::vector<uint64_t>{2, 3}));

        // Bucket 3: [4, 7] → primes 5, 7
        const auto v3 = power_bucket::enumerate_bucket_primes(3u);
        assert((v3 == std::vector<uint64_t>{5, 7}));

        // Bucket 4: [8, 15] → primes 11, 13
        const auto v4 = power_bucket::enumerate_bucket_primes(4u);
        assert((v4 == std::vector<uint64_t>{11, 13}));

        // All returned values must be prime
        for (uint64_t p : v2) assert(mersenne::is_prime_exponent(p));
        for (uint64_t p : v3) assert(mersenne::is_prime_exponent(p));
        for (uint64_t p : v4) assert(mersenne::is_prime_exponent(p));

        // Bucket 1 (normalized [2,2]): only prime is 2
        const auto v1 = power_bucket::enumerate_bucket_primes(1u);
        assert(v1.size() == 1u && v1[0] == 2u);
    }

    // 6. reverse ordering (verify via sorted comparison)
    {
        auto forward = power_bucket::enumerate_bucket_primes(5u);  // [16, 31]
        auto reversed = forward;
        std::reverse(reversed.begin(), reversed.end());
        // All elements equal, just different order
        auto fwd_sorted = forward;
        auto rev_sorted = reversed;
        std::sort(fwd_sorted.begin(), fwd_sorted.end());
        std::sort(rev_sorted.begin(), rev_sorted.end());
        assert(fwd_sorted == rev_sorted);
        // forward is ascending, reversed is descending
        for (size_t i = 1; i < forward.size(); ++i)
            assert(forward[i] > forward[i - 1]);
        for (size_t i = 1; i < reversed.size(); ++i)
            assert(reversed[i] < reversed[i - 1]);
    }

    // 7. dry-run: bucket_range and enumerate_bucket_primes are deterministic
    {
        // Two calls with same n must return identical results.
        const auto a = power_bucket::enumerate_bucket_primes(6u);
        const auto b = power_bucket::enumerate_bucket_primes(6u);
        assert(a == b);
        assert(!a.empty());
    }

    // 8. invalid bucket returns empty prime list
    {
        assert(power_bucket::enumerate_bucket_primes(0u).empty());
        assert(power_bucket::enumerate_bucket_primes(65u).empty());
    }

    // --- ProgressContext and LLResult tests ---

    // 9. ProgressContext default construction.
    {
        ProgressContext ctx;
        assert(ctx.bucket_n == 0u);
        assert(ctx.exp_index == 0u);
        assert(ctx.exp_total == 0u);
        assert(ctx.interval_iters == 10000u);
        assert(ctx.interval_secs == 0.0);
    }

    // 10. LLResult default construction.
    {
        LLResult r;
        assert(!r.is_prime);
        assert(r.final_residue_hex == "0000000000000000");
    }

    // 11. residue_hex format: always 16 lowercase hex characters.
    auto check_residue_fmt = [](const std::string& hex) {
        assert(hex.size() == 16u);
        for (char c : hex)
            assert((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'));
    };

    // 12. GenericBackend residue: prime → zeros, composite → non-zero.
    {
        // p=3 is a known Mersenne prime (benchmark_mode=true to skip primality check).
        {
            LucasLehmerEngine<backend::GenericBackend> eng(3u, true);
            const LLResult res = eng.run_ex(false, ProgressContext{});
            assert(res.is_prime);
            assert(res.final_residue_hex == "0000000000000000");
            check_residue_fmt(res.final_residue_hex);
        }
        // p=11 is composite.
        {
            LucasLehmerEngine<backend::GenericBackend> eng(11u, true);
            const LLResult res = eng.run_ex(false, ProgressContext{});
            assert(!res.is_prime);
            assert(res.final_residue_hex != "0000000000000000");
            check_residue_fmt(res.final_residue_hex);
        }
    }

    // 13. LimbBackend residue: prime → zeros, composite → non-zero.
    {
        // p=607 is a known Mersenne prime; use LimbBackend (p < kLimbFftCrossover).
        {
            LucasLehmerEngine<backend::LimbBackend> eng(607u, true);
            const LLResult res = eng.run_ex(false, ProgressContext{});
            assert(res.is_prime);
            assert(res.final_residue_hex == "0000000000000000");
            check_residue_fmt(res.final_residue_hex);
        }
        // p=601 is a prime exponent but M_601 is composite.
        {
            LucasLehmerEngine<backend::LimbBackend> eng(601u, true);
            const LLResult res = eng.run_ex(false, ProgressContext{});
            assert(!res.is_prime);
            assert(res.final_residue_hex != "0000000000000000");
            check_residue_fmt(res.final_residue_hex);
        }
    }

    // 14. lucas_lehmer_ex: results match lucas_lehmer for known primes and composites.
    {
        const ProgressContext ctx{};
        // p=2 (special case): prime, zero residue.
        {
            const LLResult r = mersenne::lucas_lehmer_ex(2u, false, true, ctx);
            assert(r.is_prime);
            assert(r.final_residue_hex == "0000000000000000");
        }
        // p=5: known Mersenne prime.
        {
            const LLResult r = mersenne::lucas_lehmer_ex(5u, false, true, ctx);
            assert(r.is_prime);
            assert(r.final_residue_hex == "0000000000000000");
            check_residue_fmt(r.final_residue_hex);
        }
        // p=11: composite.
        {
            const bool b_old = mersenne::lucas_lehmer(11u, false, true);
            const LLResult r = mersenne::lucas_lehmer_ex(11u, false, true, ctx);
            assert(r.is_prime == b_old);
            assert(!r.is_prime);
            assert(r.final_residue_hex != "0000000000000000");
            check_residue_fmt(r.final_residue_hex);
        }
        // p=13: known prime; result must agree with lucas_lehmer.
        {
            const bool b_old = mersenne::lucas_lehmer(13u, false, true);
            const LLResult r = mersenne::lucas_lehmer_ex(13u, false, true, ctx);
            assert(r.is_prime == b_old);
            assert(r.is_prime);
        }
    }

    // 15. ProgressContext with bucket context fields set.
    {
        ProgressContext ctx;
        ctx.bucket_n       = 5u;
        ctx.bucket_lo      = 16u;
        ctx.bucket_hi      = 31u;
        ctx.exp_index      = 3u;
        ctx.exp_total      = 8u;
        ctx.interval_iters = 500u;
        ctx.interval_secs  = 30.0;
        assert(ctx.bucket_n == 5u);
        assert(ctx.exp_index == 3u);
        assert(ctx.exp_total == 8u);
        assert(ctx.interval_iters == 500u);
        assert(ctx.interval_secs == 30.0);
        // Run a small exponent with context set (no output since progress=false).
        const LLResult r = mersenne::lucas_lehmer_ex(7u, false, true, ctx);
        assert(r.is_prime);
        assert(r.final_residue_hex == "0000000000000000");
    }

    // 16. Residue consistency: lucas_lehmer_ex agrees with lucas_lehmer on primality.
    {
        const ProgressContext ctx{};
        const std::vector<uint32_t> test_exps = {3, 5, 7, 11, 13, 17, 19, 23, 29, 31};
        for (uint32_t p : test_exps) {
            const bool b_old = mersenne::lucas_lehmer(p, false, true);
            const LLResult r = mersenne::lucas_lehmer_ex(p, false, true, ctx);
            assert(r.is_prime == b_old);
            if (r.is_prime)
                assert(r.final_residue_hex == "0000000000000000");
            else
                assert(r.final_residue_hex != "0000000000000000");
            check_residue_fmt(r.final_residue_hex);
        }
    }

    std::cout << "All tests passed\n";
    return 0;
}
