// BigNum.cpp – Optimized Lucas–Lehmer benchmark for Mersenne numbers.
//
// Backend hierarchy (auto-selected by exponent size):
//   p < 128      : GenericBackend  (boost::multiprecision::cpp_int – reference)
//   128 <= p < kLimbFftCrossover :
//                  LimbBackend (true Comba squaring + Mersenne fold)
//   p >= kLimbFftCrossover :
//                  FftMersenneBackend (Crandall–Bailey DWT/FFT with adaptive
//                  digit width; covers all known Mersenne-prime exponents)
//
// Key design points:
//  • Fused hot op: square_sub2_mod_mersenne()  (one FFT pair per LL iteration)
//  • Precomputed and reused: twiddle table, DWT weights, bit-reversal table,
//    digit-width table – all allocated once per engine lifetime.
//  • No heap allocation inside the hot loop; scratch buffers are pre-allocated.
//  • LimbBackend uses true Comba squaring: n*(n+1)/2 multiplications vs n^2.
//  • limb_shift precomputed in LimbState to avoid hot-loop division.
//  • Worker threads now respect the caller-supplied progress flag (bug fix).
//  • Progress reporting uses steady_clock (replaces std::time/difftime).
//  • is_prime_exponent() is skipped in benchmark mode for the known-prime list.
//  • Separate thread pool for throughput mode (many exponents concurrently).
//
// Performance optimizations (phased):
//  [Phase 1] Carmack fast-inverse-sqrt principle applied to is_prime_exponent():
//    precompute sqrt(n) once before the trial-division loop, replacing a 128-bit
//    i*i multiply in the loop condition with a simple 64-bit comparison.
//  [Phase 2] fft_core() sign branch removed: caller passes pre-negated tw_im_half_inv
//    for the inverse n/2-point pass.  Eliminates a data-dependent branch from the
//    butterfly inner loop, enabling the compiler to auto-vectorize with AVX2/FMA.
//  [Phase 3] inv_mod_pow[] precomputed reciprocals (Carmack principle): carry
//    propagation uses multiply-by-reciprocal (4 cycles) instead of divide (20+).
//    std::nearbyint replaces std::round (single VROUNDSD vs multi-instruction).
//  [Phase 4] Real-FFT optimization in fft_square(): the DWT-weighted input is
//    purely real, so two n-point complex FFTs are replaced by two n/2-point FFTs
//    plus O(n) post/pre-processing, saving ~46% of butterfly work per iteration.
//    (For p=44497, n=8192 → n/2=4096: measured ~40% wall-clock speedup.)
//  [Phase 5] fast_ldexp_neg(): Carmack-style IEEE-754 exponent-field subtraction
//    available as a utility; used for correctness cross-checks and future micro-
//    optimization of the carry path.

#include <algorithm>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cmath>
#include <complex>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <mutex>
#include <new>
#include <queue>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>
#include <boost/multiprecision/cpp_int.hpp>
#include <cerrno>
#include <cinttypes>
#include <ctime>
#include <fstream>
#include <sys/stat.h>
#include <sched.h>

// ============================================================
// namespace runtime
// ============================================================
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

// Simple thread pool for independent-exponent throughput mode.
class ThreadPool {
public:
    explicit ThreadPool(unsigned num_threads) {
        workers_.reserve(num_threads);
        for (unsigned i = 0; i < num_threads; ++i)
            workers_.emplace_back([this] { worker_loop(); });
    }

    ~ThreadPool() {
        {
            std::lock_guard<std::mutex> lk(mu_);
            stop_ = true;
        }
        cv_.notify_all();
        for (auto& t : workers_) t.join();
    }

    template <class F>
    void submit(F&& f) {
        {
            std::lock_guard<std::mutex> lk(mu_);
            tasks_.emplace(std::forward<F>(f));
        }
        cv_.notify_one();
    }

    void wait_all() {
        std::unique_lock<std::mutex> lk(mu_);
        done_cv_.wait(lk, [this] { return tasks_.empty() && busy_ == 0; });
    }

private:
    void worker_loop() {
        for (;;) {
            std::function<void()> task;
            {
                std::unique_lock<std::mutex> lk(mu_);
                cv_.wait(lk, [this] { return stop_ || !tasks_.empty(); });
                if (stop_ && tasks_.empty()) return;
                task = std::move(tasks_.front());
                tasks_.pop();
                ++busy_;
            }
            task();
            {
                std::lock_guard<std::mutex> lk(mu_);
                --busy_;
            }
            done_cv_.notify_all();
        }
    }

    std::mutex mu_;
    std::condition_variable cv_;
    std::condition_variable done_cv_;
    std::vector<std::thread> workers_;
    std::queue<std::function<void()>> tasks_;
    unsigned busy_{0};
    bool stop_{false};
};

}  // namespace runtime

// ============================================================
// namespace mersenne  (exponent list, primality helper)
// ============================================================
namespace mersenne {

// Carmack fast-inverse-sqrt insight applied to trial division:
// instead of computing i*i (128-bit) every loop iteration, precompute the
// loop bound once via hardware sqrt (VSQRTSD on x86-64).  This removes the
// 128-bit multiply from the inner loop entirely, cutting cycle count per
// iteration significantly for large n.
bool is_prime_exponent(uint64_t n) {
    if (n < 2u) return false;
    if (n == 2u) return true;
    if ((n & 1u) == 0u) return false;
    if (n % 3u == 0u) return n == 3u;
    if (n % 5u == 0u) return n == 5u;
    // Precompute floor(sqrt(n)) via hardware VSQRTSD (Carmack: one approximation
    // before the loop, then a cheap integer comparison inside every iteration).
    // For n < 2^53 the double result is exact.  For n >= 2^53 the floating-point
    // sqrt may under-approximate floor(sqrt(n)) by up to 1 ULP; we correct that
    // with at most one post-adjustment increment using an overflow-safe __uint128_t
    // comparison (GCC/Clang extension, required by the build system anyway).
    // The correction runs once outside the hot loop, so the inner loop body
    // remains a plain 64-bit `i <= isqrt` comparison with no 128-bit arithmetic.
    __extension__ typedef unsigned __int128 u128_isqrt;
    uint64_t isqrt = static_cast<uint64_t>(std::sqrt(static_cast<double>(n)));
    // Adjust upward until isqrt == floor(sqrt(n)).  At most one increment.
    while ((u128_isqrt)isqrt * (u128_isqrt)isqrt < (u128_isqrt)n) ++isqrt;
    for (uint64_t i = 7u; i <= isqrt; i += 2u) {
        if (n % i == 0u) return false;
    }
    return true;
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

// Distribute exponent values starting at startIndex across `threads` lanes
// using round-robin, so each thread gets a pre-assigned slice of work.
inline std::vector<std::vector<uint32_t>> precharge_work_matrix(
    const std::vector<uint32_t>& exponents, size_t startIndex, unsigned threads)
{
    if (threads == 0u) threads = 1u;
    std::vector<std::vector<uint32_t>> work_matrix(threads);
    unsigned slot = 0u;
    for (size_t idx = startIndex; idx < exponents.size(); ++idx) {
        work_matrix[slot].push_back(exponents[idx]);
        if (++slot >= threads) slot = 0u;
    }
    return work_matrix;
}

// Returns true if p is in the known Mersenne prime exponent list.
inline bool is_known_mersenne_prime(uint64_t p) {
    const auto& known = known_mersenne_prime_exponents();
    // All known exponents fit in uint32_t; values above uint32 max are never known.
    if (p > UINT32_MAX) return false;
    const uint32_t p32 = static_cast<uint32_t>(p);
    return std::binary_search(known.begin(), known.end(), p32);
}

// Generate all prime exponents p such that min_excl < p <= max_incl.
// Returns results in ascending order.  Uses trial division via
// is_prime_exponent(); iterates only odd candidates for efficiency.
// Bounds are uint64_t so searches above the uint32_t range are supported.
inline std::vector<uint64_t> generate_post_known_exponents(
    uint64_t min_excl, uint64_t max_incl)
{
    std::vector<uint64_t> result;
    if (max_incl <= min_excl) return result;
    uint64_t start = min_excl + 1u;
    // Skip even numbers > 2.
    if (start > 2u && (start & 1u) == 0u) ++start;
    for (uint64_t p = start; p <= max_incl; ) {
        if (is_prime_exponent(p)) result.push_back(p);
        const uint64_t step = (p == 2u ? 1u : 2u);
        const uint64_t next = p + step;
        // Stop if the next increment would overflow or exceed max_incl.
        if (next <= p || next > max_incl) break;
        p = next;
    }
    return result;
}

// Discover-mode exponent list:
//   1. single_exp first (if non-zero and prime),
//   2. all prime p: min_excl < p <= max_incl (single_exp deduplicated),
//   3. optionally reversed (range part only),
//   4. optionally sharded (range part only; single_exp always included).
// All exponent values are uint64_t to support ranges above the uint32 limit.
inline std::vector<uint64_t> discover_exponent_list(
    uint64_t single_exp,
    uint64_t min_excl,
    uint64_t max_incl,
    bool     reverse_order = false,
    uint32_t shard_count   = 1u,
    uint32_t shard_index   = 0u)
{
    if (shard_count == 0u) shard_count = 1u;
    if (shard_index >= shard_count) shard_index = 0u;

    // Generate exploration range.
    std::vector<uint64_t> range = generate_post_known_exponents(min_excl, max_incl);

    // Deduplicate: remove single_exp from range if it falls inside.
    if (single_exp != 0u) {
        range.erase(std::remove(range.begin(), range.end(), single_exp), range.end());
    }

    // Apply shard selection to the range part.
    if (shard_count > 1u) {
        std::vector<uint64_t> shard;
        shard.reserve(range.size() / shard_count + 1u);
        for (size_t i = 0; i < range.size(); ++i) {
            if (static_cast<uint32_t>(i % shard_count) == shard_index)
                shard.push_back(range[i]);
        }
        range = std::move(shard);
    }

    // Optionally reverse the range (largest-first scheduling).
    if (reverse_order) std::reverse(range.begin(), range.end());

    // Build result: explicit exponent first, then range.
    std::vector<uint64_t> result;
    result.reserve((single_exp != 0u ? 1u : 0u) + range.size());
    if (single_exp != 0u && is_prime_exponent(single_exp))
        result.push_back(single_exp);
    result.insert(result.end(), range.begin(), range.end());
    return result;
}

}  // namespace mersenne

// ============================================================
// namespace sweep  (work-list generators for sweep modes n / p / m)
// ============================================================
namespace sweep {

// mode 'n': all natural numbers in [min_exp, max_exp] inclusive.
inline std::vector<uint32_t> generate_natural(uint32_t min_exp, uint32_t max_exp) {
    std::vector<uint32_t> v;
    if (max_exp >= min_exp) {
        v.reserve(static_cast<size_t>(max_exp - min_exp) + 1u);
        for (uint32_t i = min_exp; i <= max_exp; ++i)
            v.push_back(i);
    }
    return v;
}

// mode 'p': all prime numbers in [min_exp, max_exp] inclusive.
// Uses a segmented sieve of Eratosthenes:
//   (1) compute base primes up to floor(sqrt(max_exp)) via a simple sieve,
//   (2) sieve in segments of 1M numbers,
//   (3) collect all primes in [start, max_exp].
inline std::vector<uint32_t> generate_prime(uint32_t min_exp, uint32_t max_exp) {
    std::vector<uint32_t> v;
    const uint32_t start = (min_exp < 2u) ? 2u : min_exp;
    if (start > max_exp) return v;

    // Step 1: base primes up to floor(sqrt(max_exp)).
    // Use floating-point sqrt to get an initial estimate, then increment until we reach the exact floor(sqrt(max_exp)).
    uint32_t sqrt_max = static_cast<uint32_t>(std::sqrt(static_cast<double>(max_exp)));
    while ((static_cast<uint64_t>(sqrt_max) + 1u) * (static_cast<uint64_t>(sqrt_max) + 1u)
           <= static_cast<uint64_t>(max_exp))
        ++sqrt_max;

    std::vector<bool> base_sieve(sqrt_max + 1u, true);
    base_sieve[0] = false;
    if (sqrt_max >= 1u) base_sieve[1] = false;
    for (uint32_t i = 2u; static_cast<uint64_t>(i) * i <= sqrt_max; ++i)
        if (base_sieve[i])
            for (uint32_t j = i * i; j <= sqrt_max; j += i)
                base_sieve[j] = false;

    std::vector<uint32_t> base_primes;
    for (uint32_t i = 2u; i <= sqrt_max; ++i)
        if (base_sieve[i]) base_primes.push_back(i);

    // Step 2: segmented sieve in blocks of 1M numbers.
    // Allocate the sieve buffer once at the maximum segment size and reuse across iterations.
    constexpr uint32_t SEG_SIZE = 1000000u;
    std::vector<bool> sieve(SEG_SIZE, true);
    uint32_t seg_lo = start;
    while (seg_lo <= max_exp) {
        const uint32_t seg_hi = (max_exp - seg_lo < SEG_SIZE) ? max_exp : seg_lo + SEG_SIZE - 1u;
        const uint32_t seg_len = seg_hi - seg_lo + 1u;
        std::fill(sieve.begin(), sieve.begin() + seg_len, true);

        // Cross off composites using base primes.
        for (uint32_t p : base_primes) {
            // First multiple of p in [seg_lo, seg_hi]; skip p itself.
            uint64_t first = ((static_cast<uint64_t>(seg_lo) + p - 1u) / p) * p;
            if (first == static_cast<uint64_t>(p)) first += p;
            for (uint64_t j = first; j <= seg_hi; j += p)
                sieve[static_cast<size_t>(j - seg_lo)] = false;
        }

        // Collect primes from this segment.
        for (uint32_t i = seg_lo; i <= seg_hi; ++i)
            if (sieve[i - seg_lo]) v.push_back(i);

        if (seg_hi == max_exp) break;
        seg_lo = seg_hi + 1u;
    }

    return v;
}

// mode 'm': known Mersenne-prime exponents in [min_exp, max_exp] first
//           (in their natural order), then all remaining prime exponents in
//           [min_exp, max_exp] that were not already listed.
inline std::vector<uint32_t> generate_mersenne_first(uint32_t min_exp, uint32_t max_exp) {
    const auto& known = mersenne::known_mersenne_prime_exponents();
    std::vector<uint32_t> v;

    // Phase 1: known Mersenne primes in range (list is already sorted).
    for (uint32_t p : known)
        if (p >= min_exp && p <= max_exp)
            v.push_back(p);

    // Phase 2: remaining primes in range not already in the known list.
    const uint32_t start = (min_exp < 2u) ? 2u : min_exp;
    for (uint32_t i = start; i <= max_exp; ++i)
        if (mersenne::is_prime_exponent(i) &&
            !std::binary_search(known.begin(), known.end(), i))
            v.push_back(i);

    return v;
}

// Apply shard selection: keep items where (position % shard_count == shard_index).
inline std::vector<uint32_t> apply_shard(const std::vector<uint32_t>& items,
                                          size_t shard_index, size_t shard_count) {
    if (shard_count <= 1u) return items;
    std::vector<uint32_t> v;
    for (size_t i = 0; i < items.size(); ++i)
        if (i % shard_count == shard_index)
            v.push_back(items[i]);
    return v;
}

}  // namespace sweep

// ============================================================
// namespace power_bucket  (power-of-two exponent range partitioning)
// ============================================================
// Divides the exponent space into 64 buckets by powers of two:
//   B_1  = [2, 2]              (normalized: [2, 2^1-1] is empty, so cap at 2)
//   B_n  = [2^(n-1), 2^n - 1] for n in [2, 64]
//   B_64 = [2^63, 2^64-1]     (hi = UINT64_MAX)
// ============================================================
namespace power_bucket {

struct Range { uint64_t lo; uint64_t hi; };

// Return the exponent range for bucket n (1-indexed, 1 <= n <= 64).
// Returns {0, 0} for out-of-range n.
inline Range bucket_range(uint32_t n) {
    if (n == 0u || n > 64u) return {0u, 0u};
    if (n == 1u) return {2u, 2u};  // normalized
    const uint64_t lo = UINT64_C(1) << (n - 1u);
    const uint64_t hi = (n == 64u) ? UINT64_MAX
                                    : ((UINT64_C(1) << n) - UINT64_C(1));
    return {lo, hi};
}

// Enumerate all prime exponents in bucket n, in ascending order.
inline std::vector<uint64_t> enumerate_bucket_primes(uint32_t n) {
    const Range r = bucket_range(n);
    if (r.lo == 0u) return {};
    // generate_post_known_exponents(min_excl, max_incl) → primes > min_excl and <= max_incl.
    const uint64_t min_excl = r.lo - 1u;
    return mersenne::generate_post_known_exponents(min_excl, r.hi);
}

}  // namespace power_bucket

// ============================================================
// ProgressContext: per-exponent progress config for lucas_lehmer_ex().
// LLResult:        result returned by lucas_lehmer_ex().
// ============================================================
struct ProgressContext {
    uint32_t bucket_n{0};         // 0 = no bucket context
    uint64_t bucket_lo{0};
    uint64_t bucket_hi{0};
    size_t   exp_index{0};        // 1-based index within current bucket
    size_t   exp_total{0};        // total exponents in current bucket
    uint32_t interval_iters{10000}; // checkpoint every N iterations
    double   interval_secs{0.0};  // also checkpoint every N seconds (0 = off)
};

struct LLResult {
    bool        is_prime{false};
    std::string final_residue_hex{"0000000000000000"}; // 16-char lowercase hex
};

// ============================================================
// namespace backend
// ============================================================
namespace backend {

// ---- Utilities ----

// Return smallest power of two >= x (x > 0).
static inline size_t next_pow2(size_t x) {
    if (x == 0) return 1;
    --x;
    x |= x >> 1; x |= x >> 2; x |= x >> 4;
    x |= x >> 8; x |= x >> 16;
    if constexpr (sizeof(size_t) > 4) {
        x |= x >> 32;
    }
    return x + 1;
}

// Integer log2 of a power of two.
static inline int ilog2(size_t n) {
    int k = 0;
    while ((size_t(1) << k) < n) ++k;
    return k;
}

// ============================================================
// GenericBackend  (boost::multiprecision::cpp_int)
// Reference / small-exponent path.
// ============================================================
struct GenericState {
    boost::multiprecision::cpp_int s;
    boost::multiprecision::cpp_int mask;
    boost::multiprecision::cpp_int sq;  // reused scratch to reduce allocator pressure
    uint32_t p{0};
    double max_roundoff{0.0};  // unused, kept for uniform interface
};

struct GenericBackend {
    using State = GenericState;

    static State init(uint32_t p) {
        using boost::multiprecision::cpp_int;
        State st;
        st.p    = p;
        st.mask = (cpp_int(1) << p) - 1;
        st.s    = 4;
        return st;
    }

    static void step(State& st) {
        using boost::multiprecision::cpp_int;
        st.sq = st.s * st.s;
        st.sq -= 2;
        // Reduce mod 2^p - 1 (at most 2 steps after squaring).
        if (st.sq > st.mask) {
            st.sq = (st.sq & st.mask) + (st.sq >> st.p);
            if (st.sq > st.mask) st.sq = (st.sq & st.mask) + (st.sq >> st.p);
        }
        if (st.sq == st.mask) st.sq = 0;
        st.s = std::move(st.sq);
    }

    static bool is_zero(const State& st) { return st.s == 0; }
    static double max_roundoff(const State& st) { return st.max_roundoff; }
    static std::string residue_hex(const State& st) {
        if (is_zero(st)) return "0000000000000000";
        using boost::multiprecision::cpp_int;
        const cpp_int mask64 = (cpp_int(1) << 64) - 1;
        const uint64_t low64 = static_cast<uint64_t>(st.s & mask64);
        char buf[17];
        std::snprintf(buf, sizeof(buf), "%016" PRIx64, low64);
        return std::string(buf);
    }
};

// ============================================================
// LimbBackend
//
// Plain 64-bit limb squaring + Mersenne fold for medium exponents.
// Faster than FFT for p below kLimbFftCrossover due to lower overhead
// on small-to-medium limb counts.
//
// Uses schoolbook squaring for nlimbs <= KARA_BASE_LIMBS,
// and recursive Karatsuba for larger sizes.
// ============================================================

static constexpr int KARA_BASE_LIMBS = 12;

// 128-bit accumulator type – GCC/Clang extension; __extension__ silences -Wpedantic.
__extension__ typedef unsigned __int128 u128_t;

// True Comba/schoolbook squaring: sq[0..2n-1] = a[0..n-1]^2.
//
// Uses n*(n+1)/2 multiplications (vs n^2 for generic multiply), exploiting
// the identity  a^2 = diag + 2*cross  where:
//   diag  = sum_{i}   a[i]^2  at position 2*i
//   cross = sum_{i<j} a[i]*a[j]  at position i+j (doubled via a left-shift)
//
// Three-phase algorithm:
//   Phase 1: accumulate cross terms a[i]*a[j] for i < j into sq.
//   Phase 2: double sq by left-shifting one bit (2*cross fits in 2*n limbs).
//   Phase 3: add diagonal terms a[i]^2 at position 2*i.
static void schoolbook_sq(const uint64_t* __restrict__ a,
                           uint64_t* __restrict__ sq, int n) {
    std::fill(sq, sq + 2 * n, uint64_t(0));

    // Phase 1: off-diagonal cross terms (i < j only).
    for (int i = 0; i < n; ++i) {
        uint64_t carry = 0;
        for (int j = i + 1; j < n; ++j) {
            u128_t t = (u128_t)a[i] * a[j] + sq[i + j] + carry;
            sq[i + j] = (uint64_t)t;
            carry      = (uint64_t)(t >> 64);
        }
        for (int k = i + n; carry && k < 2 * n; ++k) {
            u128_t t = (u128_t)sq[k] + carry;
            sq[k]  = (uint64_t)t;
            carry  = (uint64_t)(t >> 64);
        }
        // Invariant: sum of cross terms so far fits in 2*n limbs, so carry
        // must be zero after the cleanup loop.  Catches overflow if n is
        // ever set above KARA_BASE_LIMBS without adjusting the buffer size.
        assert(carry == 0);
    }

    // Phase 2: double the off-diagonal sum (shift left by 1 bit).
    // Guaranteed no carry out: 2*(off-diag) < a^2 < 2^{128*n}.
    uint64_t shl_carry = 0;
    for (int k = 0; k < 2 * n; ++k) {
        const uint64_t top = sq[k] >> 63;
        sq[k] = (sq[k] << 1) | shl_carry;
        shl_carry = top;
    }
    // shl_carry == 0 is guaranteed; assert in debug builds.
    assert(shl_carry == 0);

    // Phase 3: add diagonal terms a[i]^2 at position 2*i.
    for (int i = 0; i < n; ++i) {
        const u128_t d = (u128_t)a[i] * a[i];
        uint64_t carry = 0;
        {
            const u128_t t = (u128_t)sq[2 * i] + (uint64_t)d;
            sq[2 * i] = (uint64_t)t;
            carry = (uint64_t)(t >> 64) + (uint64_t)(d >> 64);
        }
        for (int k = 2 * i + 1; carry && k < 2 * n; ++k) {
            const u128_t t = (u128_t)sq[k] + carry;
            sq[k] = (uint64_t)t;
            carry  = (uint64_t)(t >> 64);
        }
    }
}

// Subtract b[0..nb-1] from a[0..na-1] in-place. na >= nb. Returns final borrow.
static uint64_t limb_sub(uint64_t* a, int na, const uint64_t* b, int nb, uint64_t borrow) {
    for (int i = 0; i < nb; ++i) {
        u128_t t = (u128_t)a[i] - b[i] - borrow;
        a[i]   = (uint64_t)t;
        borrow = (uint64_t)(t >> 64) ? 1u : 0u;
    }
    for (int i = nb; i < na && borrow; ++i) {
        u128_t t = (u128_t)a[i] - borrow;
        a[i]   = (uint64_t)t;
        borrow = (uint64_t)(t >> 64) ? 1u : 0u;
    }
    return borrow;
}

// Forward declaration.
static void karatsuba_sq(const uint64_t* a, int n, uint64_t* out, uint64_t* scratch);

static void karatsuba_sq(const uint64_t* a, int n,
                          uint64_t* out, uint64_t* scratch) {
    if (n <= KARA_BASE_LIMBS) {
        schoolbook_sq(a, out, n);
        return;
    }

    const int lo_n  = n / 2;
    const int hi_n  = n - lo_n;
    const int sum_n = hi_n + 1;

    uint64_t* lo_sq   = scratch;
    uint64_t* hi_sq   = lo_sq + 2 * lo_n;
    uint64_t* sum_buf = hi_sq + 2 * hi_n;
    uint64_t* mid_sq  = sum_buf + sum_n;
    uint64_t* rest    = mid_sq + 2 * sum_n;

    // lo_sq = a[0..lo_n-1]^2
    karatsuba_sq(a, lo_n, lo_sq, rest);

    // hi_sq = a[lo_n..n-1]^2
    karatsuba_sq(a + lo_n, hi_n, hi_sq, rest);

    // sum_buf = a[0..lo_n-1] + a[lo_n..n-1]
    std::copy(a + lo_n, a + n, sum_buf);
    sum_buf[hi_n] = 0;
    {
        uint64_t carry = 0;
        for (int i = 0; i < lo_n; ++i) {
            u128_t t = (u128_t)sum_buf[i] + a[i] + carry;
            sum_buf[i] = (uint64_t)t;
            carry       = (uint64_t)(t >> 64);
        }
        for (int i = lo_n; carry && i < sum_n; ++i) {
            u128_t t = (u128_t)sum_buf[i] + carry;
            sum_buf[i] = (uint64_t)t;
            carry       = (uint64_t)(t >> 64);
        }
    }

    // mid_sq = sum_buf^2
    karatsuba_sq(sum_buf, sum_n, mid_sq, rest);

    // mid_sq -= lo_sq + hi_sq
    limb_sub(mid_sq, 2 * sum_n, lo_sq, 2 * lo_n, 0);
    limb_sub(mid_sq, 2 * sum_n, hi_sq, 2 * hi_n, 0);

    // Assemble: out = lo_sq + mid_sq<<(lo_n*64) + hi_sq<<(n*64)
    std::fill(out, out + 2 * n, uint64_t(0));

    // Add lo_sq
    {
        uint64_t carry = 0;
        for (int i = 0; i < 2 * lo_n; ++i) {
            u128_t t = (u128_t)out[i] + lo_sq[i] + carry;
            out[i] = (uint64_t)t;
            carry   = (uint64_t)(t >> 64);
        }
        for (int i = 2 * lo_n; carry && i < 2 * n; ++i) {
            u128_t t = (u128_t)out[i] + carry;
            out[i] = (uint64_t)t;
            carry   = (uint64_t)(t >> 64);
        }
    }

    // Add mid_sq at offset lo_n
    {
        uint64_t carry = 0;
        const int mid_len = 2 * sum_n;
        for (int i = 0; i < mid_len && lo_n + i < 2 * n; ++i) {
            u128_t t = (u128_t)out[lo_n + i] + mid_sq[i] + carry;
            out[lo_n + i] = (uint64_t)t;
            carry          = (uint64_t)(t >> 64);
        }
        for (int i = lo_n + mid_len; carry && i < 2 * n; ++i) {
            u128_t t = (u128_t)out[i] + carry;
            out[i] = (uint64_t)t;
            carry   = (uint64_t)(t >> 64);
        }
    }

    // Add hi_sq at offset 2*lo_n.
    // Karatsuba decomposition: a = a_hi*B^lo_n + a_lo → a^2 = lo_sq + mid*B^lo_n + hi_sq*B^{2*lo_n}.
    // For odd n, 2*lo_n < n, so this offset differs from n.
    {
        const int hi_off = 2 * lo_n;
        uint64_t carry = 0;
        for (int i = 0; i < 2 * hi_n && hi_off + i < 2 * n; ++i) {
            u128_t t = (u128_t)out[hi_off + i] + hi_sq[i] + carry;
            out[hi_off + i] = (uint64_t)t;
            carry            = (uint64_t)(t >> 64);
        }
        for (int i = hi_off + 2 * hi_n; carry && i < 2 * n; ++i) {
            u128_t t = (u128_t)out[i] + carry;
            out[i] = (uint64_t)t;
            carry   = (uint64_t)(t >> 64);
        }
    }
}

static int karatsuba_scratch_size(int n) {
    if (n <= KARA_BASE_LIMBS) return 0;
    const int hi_n  = n - n / 2;
    const int sum_n = hi_n + 1;
    return 2 * (n / 2) + 2 * hi_n + sum_n + 2 * sum_n + karatsuba_scratch_size(sum_n);
}

struct LimbState {
    uint32_t p{0};
    int      nlimbs{0};       // ceil(p/64)
    int      partial{0};      // p % 64; 0 means limb-aligned
    int      limb_shift{0};   // p / 64, precomputed for fold (avoids hot-loop division)
    uint64_t top_mask{0};     // mask for top limb

    std::vector<uint64_t> s;        // nlimbs: current value
    std::vector<uint64_t> sq;       // 2*nlimbs: squaring result
    std::vector<uint64_t> scratch;  // Karatsuba scratch

    double max_roundoff{0.0};  // unused (exact arithmetic), kept for interface
};

struct LimbBackend {
    using State = LimbState;

    static State init(uint32_t p) {
        LimbState st;
        st.p          = p;
        st.nlimbs     = static_cast<int>((p + 63u) / 64u);
        st.partial    = static_cast<int>(p % 64u);
        st.limb_shift = static_cast<int>(p / 64u);   // precomputed; avoids division in step()
        st.top_mask   = (st.partial == 0)
            ? ~uint64_t(0)
            : ((uint64_t(1) << st.partial) - 1u);

        const int n  = st.nlimbs;
        // kss(n) + 16: exact scratch for the recursive Karatsuba tree, plus 16 guard words.
        // 8*n + 16: a generous O(n) lower bound that dominates for small n and avoids
        //           underestimation when kss recurses only on sum_n (not hi_n).
        const int sc = std::max(karatsuba_scratch_size(n) + 16, 8 * n + 16);
        st.s.assign(n, 0u);
        st.sq.assign(2 * n, 0u);
        st.scratch.assign(static_cast<size_t>(sc), 0u);
        st.s[0] = 4u;
        return st;
    }

    // Fused: s ← (s² − 2) mod (2^p − 1)
    static void step(State& st) {
        const int n = st.nlimbs;

        // 1. Square
        if (n <= KARA_BASE_LIMBS) {
            schoolbook_sq(st.s.data(), st.sq.data(), n);
        } else {
            karatsuba_sq(st.s.data(), n, st.sq.data(), st.scratch.data());
        }

        // 2. Mersenne fold: s = (sq mod 2^p) + (sq >> p)
        {
            const int    limb_shift = st.limb_shift;  // precomputed in init()
            const int    bit_shift  = st.partial;     // p % 64

            if (bit_shift == 0) {
                // p is limb-aligned
                uint64_t carry = 0;
                for (int i = 0; i < n; ++i) {
                    u128_t t =
                        (u128_t)st.sq[i] + st.sq[i + n] + carry;
                    st.s[i] = (uint64_t)t;
                    carry    = (uint64_t)(t >> 64);
                }
                if (carry) {
                    uint64_t c2 = carry;
                    for (int i = 0; c2 && i < n; ++i) {
                        u128_t t = (u128_t)st.s[i] + c2;
                        st.s[i] = (uint64_t)t;
                        c2       = (uint64_t)(t >> 64);
                    }
                }
            } else {
                uint64_t carry = 0;
                for (int i = 0; i < n; ++i) {
                    const uint64_t lo_i = (i < limb_shift)    ? st.sq[i]
                                        : (i == limb_shift) ? (st.sq[i] & st.top_mask)
                                                             : uint64_t(0);
                    const int src = limb_shift + i;
                    const uint64_t hi_lo = (src < 2 * n) ? (st.sq[src] >> bit_shift) : uint64_t(0);
                    const uint64_t hi_hi = (src + 1 < 2 * n)
                        ? (st.sq[src + 1] << (64 - bit_shift))
                        : uint64_t(0);
                    const uint64_t hi_i  = hi_lo | hi_hi;

                    u128_t t = (u128_t)lo_i + hi_i + carry;
                    st.s[i] = (uint64_t)t;
                    carry    = (uint64_t)(t >> 64);
                }
                // For non-limb-aligned p: n*64 - p = 64 - partial > 0, so the
                // n-limb addition result fits in n*64 bits (lo+hi < 2^{p+1} < 2^{n*64}).
                // carry must be 0: lo+hi < 2^p + 2^p = 2^{p+1} ≤ 2^{n*64-1} < 2^{n*64}.
                assert(carry == 0);
                // Extract overflow bits above bit position (partial-1) from the top limb
                // and fold back: 2^p ≡ 1 mod M_p.
                uint64_t overflow = st.s[n - 1] >> bit_shift;
                st.s[n - 1] &= st.top_mask;
                if (overflow) {
                    uint64_t c2 = overflow;
                    for (int i = 0; c2 && i < n; ++i) {
                        u128_t t = (u128_t)st.s[i] + c2;
                        st.s[i] = (uint64_t)t;
                        c2       = (uint64_t)(t >> 64);
                    }
                    // Re-mask after potential carry into top limb.
                    overflow = st.s[n - 1] >> bit_shift;
                    if (overflow) {
                        st.s[n - 1] &= st.top_mask;
                        uint64_t c3 = overflow;
                        for (int i = 0; c3 && i < n; ++i) {
                            u128_t t = (u128_t)st.s[i] + c3;
                            st.s[i] = (uint64_t)t;
                            c3       = (uint64_t)(t >> 64);
                        }
                        st.s[n - 1] &= st.top_mask;
                    }
                }
            }

            // Normalize M_p (= 2^p - 1) to 0.
            bool all_ones = ((st.s[n - 1] & st.top_mask) == st.top_mask);
            if (all_ones) {
                for (int i = 0; i < n - 1 && all_ones; ++i)
                    if (st.s[i] != ~uint64_t(0)) all_ones = false;
                if (all_ones)
                    std::fill(st.s.begin(), st.s.end(), uint64_t(0));
            }
        }

        // 3. Subtract 2 (mod 2^p - 1)
        {
            uint64_t borrow = 0u;
            if (st.s[0] >= 2u) {
                st.s[0] -= 2u;
                borrow = 0u;
            } else {
                // s[0] < 2: compute s[0] - 2 mod 2^64.
                // s[0] + (2^64 - 2) = s[0] - 2 + 2^64; borrow 1 from s[1].
                st.s[0] = st.s[0] + (UINT64_MAX - 1u);  // = s[0] + 2^64 - 2
                borrow = 1u;
            }
            for (int i = 1; i < st.nlimbs && borrow; ++i) {
                if (st.s[i] >= borrow) {
                    st.s[i] -= borrow;
                    borrow = 0u;
                } else {
                    st.s[i] -= borrow;  // wraps to UINT64_MAX
                    borrow = 1u;
                }
            }
            if (borrow) {
                // s was < 2: add M_p = 2^p - 1
                uint64_t carry = 0u;
                for (int i = 0; i < st.nlimbs; ++i) {
                    const uint64_t mp_i = (i + 1 < st.nlimbs) ? ~uint64_t(0) : st.top_mask;
                    u128_t t = (u128_t)st.s[i] + mp_i + carry;
                    st.s[i] = (uint64_t)t;
                    carry    = (uint64_t)(t >> 64);
                }
                st.s[st.nlimbs - 1] &= st.top_mask;
            }
        }
    }

    static bool is_zero(const State& st) {
        for (const auto v : st.s) if (v != 0u) return false;
        return true;
    }

    static double max_roundoff(const State& st) { return st.max_roundoff; }
    static std::string residue_hex(const State& st) {
        if (is_zero(st)) return "0000000000000000";
        const uint64_t low64 = st.s.empty() ? 0u : st.s[0];
        char buf[17];
        std::snprintf(buf, sizeof(buf), "%016" PRIx64, low64);
        return std::string(buf);
    }
};

// ============================================================
// FftMersenneBackend
//
// Crandall–Bailey DWT/FFT squaring mod 2^p − 1.
//
// Representation: x = Σ_{j=0}^{n-1} d_j · 2^{B_j}
//   where B_j = ⌊j·p/n⌋,  b_j = B_{j+1} − B_j ∈ {b_lo, b_hi}.
//
// Irrational-base coefficient: ã_j = d_j / w_j  where w_j = 2^{frac(j·p/n)}.
// This maps to the ring Z[β]/(β^n − 1), β = 2^{p/n}, where squaring
// is exactly the cyclic convolution (since β^n = 2^p ≡ 1 mod M_p).
//
// Algorithm per iteration:
//   Forward DWT:  ỹ_j = d_j / w_j  (divide by weight)
//   Forward FFT:  Y = FFT(ỹ)
//   Pointwise:    Z_k = Y_k²
//   Inverse FFT:  z = IFFT(Z)  (unnormalized, includes factor n)
//   Inverse DWT:  d'_j = z_j · (w_j / n),  round, carry-propagate
//
// Precision bound: 5 · log2(n) · n · (2^b_hi)² · 2^{−53} < 0.45
// ============================================================

struct FftMersenneState {
    uint32_t p{0};
    size_t   n{0};         // transform length (power of 2)
    int      b_lo{0};      // ⌊p/n⌋ – narrow digit width
    int      b_hi{0};      // ⌈p/n⌉ – wide digit width (b_lo or b_lo+1)

    // Per-digit info (length n), precomputed once.
    std::vector<int>      digit_width;  // b_j for each digit
    std::vector<uint64_t> bit_pos;      // B_j = ⌊j·p/n⌋
    std::vector<double>   w_fwd;        // DWT weight: 2^{frac(j·p/n)}
    std::vector<double>   w_fwd_inv;    // forward DWT divisor: 1/w_fwd[j]
    std::vector<double>   w_inv;        // inverse DWT scale:   w_fwd[j] / n

    // FFT twiddle table, length n/2:  twiddle[k] = e^{−2πik/n}.
    std::vector<double>   tw_re;        // cos(−2πk/n)
    std::vector<double>   tw_im;        // sin(−2πk/n)
    // Note: the n-point inverse twiddle table is NOT stored here.
    // The current real-FFT path uses the precomputed half-size tw_im_half_inv
    // table instead, so a separate n-point negated table would be dead weight.

    // Bit-reversal permutation table (length n).
    std::vector<size_t>   bitrv;

    // Precomputed exp2 tables for carry propagation (length n).
    std::vector<double>   mod_pow;      // 2^{digit_width[j]} per digit
    // Carmack fast-inverse-sqrt insight: precompute the reciprocal of each
    // digit modulus so carry reduction is a multiply (4 cycles) instead of
    // a divide (20+ cycles).  Since mod_pow[j] = 2^{b_j} (exact power of 2),
    // the reciprocal is also exact: inv_mod_pow[j] = 2^{-b_j} = ldexp(1,-b_j).
    // fast_ldexp_neg() below performs the same operation via IEEE-754 exponent
    // manipulation (subtract b from the biased exponent field), inspired
    // directly by Carmack's bit-level float hack in Quake III.
    std::vector<double>   inv_mod_pow;  // = 1.0/mod_pow[j] = ldexp(1,-b_j)

    // Real-FFT half-size tables (squaring a real input via an n/2-point FFT).
    // Replaces two n-point FFTs with two n/2-point FFTs + O(n) post/pre steps,
    // saving ~46% of butterfly operations per iteration.
    size_t               half_n{0};        // = n/2
    std::vector<size_t>  bitrv_half;       // bit-reversal for n/2-point FFT
    std::vector<double>  tw_re_half;       // cos twiddles for n/2-point FFT (length n/4)
    std::vector<double>  tw_im_half;       // sin twiddles for n/2-point FFT
    std::vector<double>  tw_im_half_inv;   // = -tw_im_half (for inverse n/2-point FFT)
    std::vector<double>  hbuf_re;          // working buffer, size n/2
    std::vector<double>  hbuf_im;          // working buffer, size n/2
    std::vector<double>  w_inv_half;       // = 2*w_inv[j] = w_fwd[j]/(n/2) for real-FFT unpack

    // Working buffers (length n) – retained for reference / fallback.
    std::vector<double>   buf_re;
    std::vector<double>   buf_im;

    // Current state: digit representation d_j.
    std::vector<double>   digits;

    // Diagnostic.
    double max_roundoff{0.0};
};

// Choose smallest n = 2^k satisfying the error bound.
// Returns 0 if no suitable n found (caller should fall back).
static size_t choose_fft_length(uint32_t p) {
    // Threshold: 5 · k · n · (2^b_hi)^2 · 2^{−53} < 0.45
    // With b_hi = ⌈p/n⌉ and n = 2^k:
    //   5 · k · 2^{k + 2·b_hi} · 2^{−53} < 0.45
    static const double kThresh = 0.45;
    for (int k = 1; k <= 30; ++k) {
        const size_t n    = size_t(1) << k;
        const size_t b_hi = (p + n - 1) / n;   // ⌈p/n⌉
        if (b_hi > 53) continue;                // digit too wide for double
        // Compute error estimate (no overflow because b_hi ≤ 53).
        const double err = 5.0 * k * std::ldexp(static_cast<double>(n),
                                                  static_cast<int>(2 * b_hi) - 53);
        if (err < kThresh) return n;
    }
    return 0;
}

// Build all precomputed tables for the given (p, n).
static void fft_mersenne_init_tables(FftMersenneState& st) {
    const uint32_t p = st.p;
    const size_t   n = st.n;
    const int      k = ilog2(n);

    st.b_lo = static_cast<int>(p / n);
    st.b_hi = static_cast<int>((p + n - 1) / n);

    // Bit-reversal permutation: br[j] = bit-reverse of j in k bits.
    st.bitrv.resize(n);
    for (size_t j = 0; j < n; ++j) {
        size_t r = 0, v = j;
        for (int b = 0; b < k; ++b) { r = (r << 1) | (v & 1); v >>= 1; }
        st.bitrv[j] = r;
    }

    // Twiddle table: twiddle[m] = e^{−2πim/n} for m = 0 .. n/2−1.
    const size_t half = n >> 1;
    st.tw_re.resize(half);
    st.tw_im.resize(half);
    const double pi = std::acos(-1.0);
    for (size_t m = 0; m < half; ++m) {
        const double angle = -2.0 * pi * m / static_cast<double>(n);
        st.tw_re[m] = std::cos(angle);
        st.tw_im[m] = std::sin(angle);
    }
    // Per-digit tables.
    st.digit_width.resize(n);
    st.bit_pos.resize(n);
    st.w_fwd.resize(n);
    st.w_fwd_inv.resize(n);
    st.w_inv.resize(n);

    for (size_t j = 0; j < n; ++j) {
        // bit position B_j = ⌊j·p/n⌋  (no overflow: j < n ≤ 2^30, p < 2^32, computed in uint64_t)
        const uint64_t jp = static_cast<uint64_t>(j) * p;
        const uint64_t Bj = jp / n;
        const uint64_t rem = jp % n;               // n · frac(j·p/n)
        st.bit_pos[j]     = Bj;
        // digit width b_j = B_{j+1} − B_j
        const uint64_t jp1 = static_cast<uint64_t>(j + 1) * p;
        st.digit_width[j] = static_cast<int>(jp1 / n - Bj);
        // w_fwd[j] = 2^{frac(j·p/n)} = 2^{rem/n}
        st.w_fwd[j]     = std::exp2(static_cast<double>(rem) / static_cast<double>(n));
        st.w_fwd_inv[j] = 1.0 / st.w_fwd[j];            // for forward DWT (divide)
        st.w_inv[j]     = st.w_fwd[j] / static_cast<double>(n);  // for inverse DWT (multiply)
    }

    // Working buffers.
    st.buf_re.assign(n, 0.0);
    st.buf_im.assign(n, 0.0);
    st.digits.assign(n, 0.0);

    // Precomputed exp2 tables.
    st.mod_pow.resize(n);
    st.inv_mod_pow.resize(n);
    for (size_t j = 0; j < n; ++j) {
        st.mod_pow[j]     = std::ldexp(1.0,  st.digit_width[j]);
        // Exact reciprocal (2^{-b_j}) – no approximation needed since the
        // moduli are exact powers of two.  Carmack's trick: precompute once,
        // multiply in the hot loop instead of dividing.
        st.inv_mod_pow[j] = std::ldexp(1.0, -st.digit_width[j]);
    }

    // ---- Real-FFT half-size tables ----
    // The DWT-weighted input to the FFT is purely real (imaginary = 0).
    // We exploit this via the "real FFT" trick: pack n real values into
    // n/2 complex values, apply an n/2-point FFT, post-process to recover
    // the full n-point Hermitian spectrum, square, pre-process, then apply
    // an n/2-point IFFT and unpack.  This halves the transform size and
    // saves ~46% of butterfly operations per Lucas-Lehmer iteration.
    st.half_n = n / 2;
    const size_t M  = st.half_n;
    const int    kh = k - 1;   // ilog2(M)

    // Bit-reversal for the n/2-point FFT (kh bits).
    st.bitrv_half.resize(M);
    for (size_t j = 0; j < M; ++j) {
        size_t r = 0, v = j;
        for (int b = 0; b < kh; ++b) { r = (r << 1) | (v & 1); v >>= 1; }
        st.bitrv_half[j] = r;
    }

    // Twiddle tables for the n/2-point FFT.
    // The n/2-point FFT twiddle at index m equals the n-point twiddle at 2m:
    //   e^{-2πim/(n/2)} = e^{-2πi(2m)/n} = twiddle_n[2m]
    const size_t half_M = M / 2;   // = n/4 entries needed
    st.tw_re_half.resize(half_M);
    st.tw_im_half.resize(half_M);
    st.tw_im_half_inv.resize(half_M);
    for (size_t m = 0; m < half_M; ++m) {
        st.tw_re_half[m]     =  st.tw_re[2 * m];
        st.tw_im_half[m]     =  st.tw_im[2 * m];
        st.tw_im_half_inv[m] = -st.tw_im_half[m];
    }

    // Working buffers for the real-FFT path (half size).
    st.hbuf_re.assign(M, 0.0);
    st.hbuf_im.assign(M, 0.0);

    // Scaled inverse DWT weights for the real-FFT path.
    // After an unnormalized n/2-point IFFT the output is (n/2) × the true
    // convolution value, whereas the n-point IFFT would give n × the same
    // value.  Compensate by using 2×w_inv instead of w_inv during unpack.
    st.w_inv_half.resize(n);
    for (size_t j = 0; j < n; ++j)
        st.w_inv_half[j] = 2.0 * st.w_inv[j];   // = w_fwd[j] / (n/2)
}

// In-place iterative Cooley–Tukey DIT FFT of length n (must be power of 2).
// Data is split into re[]/im[] arrays.
// tw_im must be the correctly-signed imaginary twiddles for the desired
// direction: pass tw_im for forward or a pre-negated table for inverse.
// The branch `sign >= 0 ? tw_im[m] : -tw_im[m]`
// has been removed from the inner loop; the caller pre-selects the table.
// This eliminates a branch inside the hottest nested loop and allows the
// compiler to auto-vectorize the butterfly with AVX2/FMA.
// The inverse does NOT divide by n; caller handles the scaling factor.
static void fft_core(double* __restrict__ re, double* __restrict__ im,
                     const size_t* __restrict__ bitrv,
                     const double* __restrict__ tw_re,
                     const double* __restrict__ tw_im,
                     size_t n) {
    // Bit-reversal permutation.
    for (size_t j = 0; j < n; ++j) {
        const size_t r = bitrv[j];
        if (r > j) {
            std::swap(re[j], re[r]);
            std::swap(im[j], im[r]);
        }
    }

    // Butterfly stages.
    const size_t half_n = n >> 1;
    for (size_t len = 1; len < n; len <<= 1) {
        const size_t step = half_n / len;   // twiddle stride
        for (size_t start = 0; start < n; start += len * 2) {
            for (size_t k = 0; k < len; ++k) {
                const size_t m   = k * step;
                const double wr  = tw_re[m];
                const double wi  = tw_im[m];   // sign already baked in by caller
                const size_t u   = start + k;
                const size_t v   = u + len;
                const double tr  = wr * re[v] - wi * im[v];
                const double ti  = wr * im[v] + wi * re[v];
                re[v] = re[u] - tr;
                im[v] = im[u] - ti;
                re[u] = re[u] + tr;
                im[u] = im[u] + ti;
            }
        }
    }
}

// Carmack-inspired IEEE-754 exponent manipulation:
// Computes x * 2^{-b} by directly subtracting b from the biased exponent
// field of the IEEE-754 double, avoiding a floating-point multiply entirely.
// Valid for normalised non-zero x and 1 <= b <= 52.
// Inspired by Carmack's Quake-III fast-inverse-sqrt: treat float bits as an
// integer, manipulate the exponent, reinterpret as float.  Here the operation
// is exact (no Newton-Raphson refinement needed) because we only adjust the
// exponent, never the mantissa.
[[maybe_unused]] static inline double fast_ldexp_neg(double x, int b) noexcept {
    uint64_t bits;
    std::memcpy(&bits, &x, sizeof(bits));
    bits -= static_cast<uint64_t>(b) << 52;   // subtract b from biased exponent
    std::memcpy(&x, &bits, sizeof(x));
    return x;
}

// Square st.digits in-place, mod 2^p − 1, via a real-input DWT/FFT.
//
// Real-FFT optimization (saves ~46% of butterfly work per iteration):
//   Since the DWT-weighted input is purely real, we pack the n real values
//   into n/2 complex values, apply an n/2-point FFT, post-process to recover
//   the full Hermitian spectrum, pointwise-square, pre-process to pack back,
//   apply an n/2-point IFFT, and unpack.
//
// Derivation of post/pre-processing formulas (Hermitian split-radix):
//   Let z[k] = x[2k] + i*x[2k+1], Z = DFT_{n/2}(z), M = n/2.
//   The n-point DFT of x is:
//     X[k] = E[k] + W_k * O[k]
//   where W_k = e^{-2πik/n}, and
//     E[k] = (Z[k mod M] + Z*[(M-k) mod M]) / 2
//     O[k] = (Z[k mod M] - Z*[(M-k) mod M]) / (2i)
//   The identities  E[M-k] = E*[k]  and  W_{M-k} = -W*_k  give:
//     X[k] + X*[M-k] = 2*E[k]    →  Z'[k] = E'[k] + i*O'[k]
//   where E' and O' are computed from Y[k]=X[k]^2 using the same formulas.
//   This fused post + square + pre pass is applied in-place on hbuf.
static void fft_square(FftMersenneState& st) {
    const size_t n = st.n;
    const size_t M = st.half_n;   // = n/2
    double*       hr  = st.hbuf_re.data();
    double*       hi  = st.hbuf_im.data();
    const double* twr = st.tw_re.data();   // n-point twiddles used in post/pre step
    const double* twi = st.tw_im.data();

    // ---- Step 1: DWT + pack ----
    // z[k] = w_fwd_inv[2k]*d[2k]  +  i * w_fwd_inv[2k+1]*d[2k+1]
    {
        const double* wfi = st.w_fwd_inv.data();
        const double* d   = st.digits.data();
        for (size_t k = 0; k < M; ++k) {
            hr[k] = wfi[2 * k]     * d[2 * k];
            hi[k] = wfi[2 * k + 1] * d[2 * k + 1];
        }
    }

    // ---- Step 2: n/2-point forward FFT ----
    fft_core(hr, hi,
             st.bitrv_half.data(),
             st.tw_re_half.data(), st.tw_im_half.data(),
             M);

    // ---- Step 3: Fused postprocess + pointwise square + preprocess ----
    //
    // For k=0 (DC + Nyquist), the algebra simplifies:
    //   X[0] = Z[0].re + Z[0].im   (real)
    //   X[M] = Z[0].re - Z[0].im   (real)
    //   Z'[0].re = (Y[0]+Y[M])/2,  Z'[0].im = (Y[0]-Y[M])/2
    {
        const double a = hr[0], b = hi[0];
        const double x0 = a + b,  xm = a - b;
        const double y0 = x0 * x0, ym = xm * xm;
        hr[0] = (y0 + ym) * 0.5;
        hi[0] = (y0 - ym) * 0.5;
    }

    // For k = 1 .. M/2, process pairs (k, M-k) simultaneously.
    // Twiddle identities: tw_re[M-k] = -tw_re[k],  tw_im[M-k] = tw_im[k].
    // This lets us compute Z'[k] and Z'[M-k] using only tw_re[k]/tw_im[k].
    for (size_t k = 1; k <= M / 2; ++k) {
        const size_t mk = M - k;

        const double zk_re = hr[k],  zk_im = hi[k];
        const double zm_re = hr[mk], zm_im = hi[mk];

        // ---- postprocess: X[k] = E[k] + W_k * O[k] ----
        const double p_re = (zk_re + zm_re) * 0.5;
        const double p_im = (zk_im - zm_im) * 0.5;
        const double q_re = (zk_im + zm_im) * 0.5;
        const double q_im = (zm_re - zk_re) * 0.5;   // = -(zk_re - zm_re)/2

        // n-point twiddle W_k (valid for k <= M/2 = n/4, within the n/2-entry table).
        const double wr = twr[k];
        const double wi = twi[k];

        const double xk_re = p_re + wr * q_re - wi * q_im;
        const double xk_im = p_im + wr * q_im + wi * q_re;

        // ---- square Y[k] = X[k]^2 ----
        const double yk_re = xk_re * xk_re - xk_im * xk_im;
        const double yk_im = xk_re * xk_im * 2.0;

        if (mk == k) {
            // k = M/2: self-paired (only one unique value).
            // At k=M/2: wr = cos(-π/2) = 0, wi = sin(-π/2) = -1.
            // V.re = 0, V.im = 2*yk_im  →  Z'[k].re = yk_re, Z'[k].im = -yk_im.
            hr[k] = yk_re;
            hi[k] = -yk_im;
        } else {
            // ---- postprocess X[M-k], using twiddle identities ----
            const double xm_re = p_re - (wr * q_re - wi * q_im);
            const double xm_im = -p_im + (wr * q_im + wi * q_re);

            // ---- square Y[M-k] = X[M-k]^2 ----
            const double ym_re = xm_re * xm_re - xm_im * xm_im;
            const double ym_im = xm_re * xm_im * 2.0;

            // ---- preprocess Z'[k] and Z'[M-k] ----
            // E'[k] = (Y[k]+Y*[M-k])/2,  V[k] = Y[k]-Y*[M-k]
            // t1 = (wi*V.re - wr*V.im)/2,  t2 = (wr*V.re + wi*V.im)/2
            // Z'[k].re = E'.re + t1,  Z'[k].im = E'.im + t2
            // Z'[M-k].re = E'.re - t1 (tw_re[M-k] = -wr)
            // Z'[M-k].im = -E'.im + t2 (tw_im[M-k] = wi)
            const double ep_re = (yk_re + ym_re) * 0.5;
            const double ep_im = (yk_im - ym_im) * 0.5;
            const double vk_re = yk_re - ym_re;
            const double vk_im = yk_im + ym_im;
            const double t1 = (wi * vk_re - wr * vk_im) * 0.5;
            const double t2 = (wr * vk_re + wi * vk_im) * 0.5;

            hr[k]  = ep_re + t1;
            hi[k]  = ep_im + t2;
            hr[mk] = ep_re - t1;
            hi[mk] = -ep_im + t2;
        }
    }

    // ---- Step 4: n/2-point inverse FFT (unnormalized) ----
    fft_core(hr, hi,
             st.bitrv_half.data(),
             st.tw_re_half.data(), st.tw_im_half_inv.data(),
             M);

    // ---- Step 5: Unpack + inverse DWT ----
    // d[2k]   = hr[k] * w_inv_half[2k]     (= hr[k] * 2 * w_inv[2k])
    // d[2k+1] = hi[k] * w_inv_half[2k+1]
    {
        const double* wih = st.w_inv_half.data();
        double*       d   = st.digits.data();
        for (size_t k = 0; k < M; ++k) {
            d[2 * k]     = hr[k] * wih[2 * k];
            d[2 * k + 1] = hi[k] * wih[2 * k + 1];
        }
    }

    // ---- Step 6: Half-integer correction (unchanged) ----
    for (size_t k = 0; k < n; ++k) {
        const double v    = st.digits[k];
        const double frac = v - std::floor(v);
        if (std::abs(frac - 0.5) < 0.25) {
            st.digits[k] = std::floor(v);
            const size_t kp = (k == 0) ? (n - 1) : (k - 1);
            st.digits[kp] += 0.5 * st.mod_pow[kp];
        }
    }

    // ---- Step 7: Carry propagation ----
    // Uses precomputed inv_mod_pow (Carmack principle: precompute reciprocal,
    // multiply instead of divide in the hot loop) and std::nearbyint (maps to
    // a single VROUNDSD instruction, faster than std::round which handles
    // ±∞/NaN edge cases we never encounter here).
    double carry   = 0.0;
    double max_err = 0.0;
    {
        const double* mp  = st.mod_pow.data();
        const double* imp = st.inv_mod_pow.data();
        double*       d   = st.digits.data();
        for (size_t j = 0; j < n; ++j) {
            const double raw     = d[j] + carry;
            const double rounded = std::nearbyint(raw);
            const double err     = std::abs(raw - rounded);
            if (err > max_err) max_err = err;
            carry = std::floor(rounded * imp[j]);   // multiply beats divide
            d[j]  = rounded - carry * mp[j];
        }
    }

    // Wrap-around: carry * 2^p ≡ carry (mod M_p).
    if (carry != 0.0) {
        const double* mp  = st.mod_pow.data();
        const double* imp = st.inv_mod_pow.data();
        double*       d   = st.digits.data();
        double carry2 = carry;
        for (size_t j = 0; j < n && carry2 != 0.0; ++j) {
            const double raw_j     = d[j] + carry2;
            const double rounded_j = std::nearbyint(raw_j);
            const double err_j     = std::abs(raw_j - rounded_j);
            if (err_j > max_err) max_err = err_j;
            carry2 = std::floor(rounded_j * imp[j]);
            d[j]   = rounded_j - carry2 * mp[j];
        }
    }

    if (max_err > st.max_roundoff) st.max_roundoff = max_err;
}

// Return true iff all digits are zero (s ≡ 0 mod 2^p − 1).
static bool fft_is_zero(const FftMersenneState& st) {
    for (size_t j = 0; j < st.n; ++j)
        if (st.digits[j] != 0.0) return false;
    return true;
}

struct FftMersenneBackend {
    using State = FftMersenneState;

    static State init(uint32_t p) {
        FftMersenneState st;
        st.p = p;
        st.n = choose_fft_length(p);
        if (st.n == 0)
            throw std::runtime_error("FftMersenneBackend: no suitable FFT length for p=" + std::to_string(p));
        fft_mersenne_init_tables(st);
        // s_0 = 4: digit 0 = 4, rest = 0.
        st.digits[0] = 4.0;
        return st;
    }

    // Fused: s ← (s² − 2) mod (2^p − 1).
    static void step(State& st) {
        fft_square(st);
        // Subtract 2 from the representation (add −2 to digit 0).
        st.digits[0] -= 2.0;
    }

    static bool is_zero(const State& st) { return fft_is_zero(st); }
    static double max_roundoff(const State& st) { return st.max_roundoff; }
    static std::string residue_hex(const State& st) {
        if (is_zero(st)) return "0000000000000000";
        // Reconstruct the low 64 bits from the DWT digit representation.
        // bit_pos[j] = floor(j*p/n) is non-decreasing; stop once it reaches 64.
        uint64_t low64 = 0u;
        for (size_t j = 0u; j < st.n; ++j) {
            const uint64_t bp = st.bit_pos[j];
            if (bp >= 64u) break;
            const int64_t d = static_cast<int64_t>(std::round(st.digits[j]));
            low64 += static_cast<uint64_t>(d) << static_cast<unsigned>(bp);
        }
        char buf[17];
        std::snprintf(buf, sizeof(buf), "%016" PRIx64, low64);
        return std::string(buf);
    }
};

}  // namespace backend

// ============================================================
// LucasLehmerEngine – drives the hot loop with a chosen backend.
// ============================================================
template <typename Backend>
class LucasLehmerEngine {
public:
    using State = typename Backend::State;

    // benchmark_mode = true → skip is_prime_exponent() check.
    LucasLehmerEngine(uint32_t p, bool benchmark_mode)
        : p_(p), benchmark_mode_(benchmark_mode), state_(Backend::init(p)) {}

    // Run (p − 2) iterations. Returns true iff the final residue is zero.
    // progress = true → print sparse progress to stdout.
    bool run(bool progress) {
        if (p_ < 2u) return false;
        if (p_ == 2u) return true;
        if ((p_ & 1u) == 0u) return false;
        if (!benchmark_mode_ && !mersenne::is_prime_exponent(p_)) return false;

        const uint32_t iters = p_ - 2u;
        const auto t_start   = std::chrono::steady_clock::now();
        uint32_t countdown   = 10000u;

        for (uint32_t i = 0; i < iters; ++i) {
            Backend::step(state_);

            if (progress && --countdown == 0u) {
                countdown = 10000u;
                const auto now = std::chrono::steady_clock::now();
                const double elapsed =
                    std::chrono::duration<double>(now - t_start).count();
                const double avg = elapsed / static_cast<double>(i + 1u);
                const double rem = avg * static_cast<double>(iters - (i + 1u));
                std::printf("  iter %u/%u  elapsed %.1fs  est.rem %.1fs  "
                            "max_err %.4f\n",
                            i + 1u, iters, elapsed, rem,
                            Backend::max_roundoff(state_));
            }
        }
        return Backend::is_zero(state_);
    }

    // run_ex: like run() but returns LLResult{is_prime, final_residue_hex}
    // and prints richer per-exponent progress using the supplied ProgressContext.
    LLResult run_ex(bool progress, const ProgressContext& ctx) {
        if (p_ < 2u) return {false, "0000000000000000"};
        if (p_ == 2u) return {true,  "0000000000000000"};
        if ((p_ & 1u) == 0u) return {false, "0000000000000000"};
        if (!benchmark_mode_ && !mersenne::is_prime_exponent(p_))
            return {false, "0000000000000000"};

        const uint32_t iters    = p_ - 2u;
        const auto     t_start  = std::chrono::steady_clock::now();
        const uint32_t chk_base = (ctx.interval_iters > 0u) ? ctx.interval_iters : 10000u;
        uint32_t       countdown = chk_base;
        double         ema_avg   = 0.0;  // EMA seconds/iteration for stable ETA

        for (uint32_t i = 0u; i < iters; ++i) {
            Backend::step(state_);

            if (progress && --countdown == 0u) {
                const auto   now     = std::chrono::steady_clock::now();
                const double elapsed = std::chrono::duration<double>(now - t_start).count();
                const double avg_i   = elapsed / static_cast<double>(i + 1u);
                const double rem     = static_cast<double>(iters - (i + 1u));
                // Exponential moving average for stable ETA.
                ema_avg = (ema_avg == 0.0) ? avg_i : (0.3 * avg_i + 0.7 * ema_avg);
                const double eta     = ema_avg * rem;
                const double pct     = 100.0 * static_cast<double>(i + 1u) /
                                               static_cast<double>(iters);
                const double max_err = Backend::max_roundoff(state_);
                if (ctx.bucket_n > 0u) {
                    std::printf("  [B%" PRIu32 " exp %zu/%zu]"
                                " p=%" PRIu32 "  iter %u/%u (%.1f%%)"
                                "  elapsed %.1fs  ETA %.1fs"
                                "  avg %.3fus/iter  max_err %.4f\n",
                                ctx.bucket_n, ctx.exp_index, ctx.exp_total, p_,
                                i + 1u, iters, pct, elapsed, eta,
                                avg_i * 1e6, max_err);
                } else {
                    std::printf("  iter %u/%u (%.1f%%)"
                                "  elapsed %.1fs  ETA %.1fs"
                                "  avg %.3fus/iter  max_err %.4f\n",
                                i + 1u, iters, pct, elapsed, eta,
                                avg_i * 1e6, max_err);
                }
                std::fflush(stdout);
                // Adjust countdown for time-based interval (approximate via EMA).
                if (ctx.interval_secs > 0.0 && ema_avg > 0.0) {
                    double est_d = ctx.interval_secs / ema_avg;
                    if (est_d <= 0.0) {
                        countdown = chk_base;
                    } else {
                        // Clamp to uint32_t range before casting to avoid UB.
                        static constexpr double kMaxU32 = 4294967295.0;
                        if (est_d > kMaxU32) est_d = kMaxU32;
                        const uint32_t est = static_cast<uint32_t>(est_d);
                        countdown = (est < 100u)           ? 100u
                                  : (est > chk_base * 10u) ? chk_base * 10u
                                  : est;
                    }
                } else {
                    countdown = chk_base;
                }
            }
        }
        const bool  is_prime = Backend::is_zero(state_);
        std::string residue  = is_prime ? "0000000000000000"
                                        : Backend::residue_hex(state_);
        return LLResult{is_prime, std::move(residue)};
    }

    double max_roundoff() const { return Backend::max_roundoff(state_); }

private:
    uint32_t p_;
    bool     benchmark_mode_;
    State    state_;
};

// ============================================================
// Public entry point: mersenne::lucas_lehmer()
// Auto-selects backend based on exponent size.
// benchmark_mode skips is_prime_exponent() (for the known-prime list).
// ============================================================
namespace mersenne {

bool lucas_lehmer(uint32_t p, bool progress, bool benchmark_mode = false) {
    if (p < 2u) return false;
    if (p == 2u) return true;
    if ((p & 1u) == 0u) return false;
    if (!benchmark_mode && !is_prime_exponent(p)) return false;

    // Threshold: use GenericBackend for tiny p (ensures correctness of
    // test cases p = 2, 3, 5, 7, 11, etc.).
    if (p < 128u) {
        LucasLehmerEngine<backend::GenericBackend> eng(p, /*benchmark_mode=*/true);
        return eng.run(progress);
    }

    // LimbBackend: schoolbook/Karatsuba exact squaring, faster than FFT for p < threshold.
    // kLimbFftCrossover measured empirically on this machine (-O3 -march=native):
    //   p=3217 (51 limbs): LimbBackend ~10ms vs FFT ~30ms → Limb wins.
    //   p=4253 (67 limbs): LimbBackend ~50ms vs FFT ~32ms → FFT wins.
    //   Crossover observed near p≈4000; tune with `make bench` on the target CPU.
    //   Override at runtime via the LL_LIMB_FFT_CROSSOVER environment variable.
    // Upper bound: no known Mersenne exponent exceeds 100 million digits (p < 10^9),
    // so 1 000 000 is a safe sanity cap for the crossover threshold.
    static constexpr uint32_t kMaxCrossoverThreshold = 1000000u;
    static const uint32_t kLimbFftCrossover = []() -> uint32_t {
        const char* env = std::getenv("LL_LIMB_FFT_CROSSOVER");
        if (env && *env) {
            char* end = nullptr;
            const long v = std::strtol(env, &end, 10);
            if (end != env && *end == '\0' && v > 0 && v < static_cast<long>(kMaxCrossoverThreshold))
                return static_cast<uint32_t>(v);
            std::fprintf(stderr,
                "LL_LIMB_FFT_CROSSOVER='%s' is invalid; using default 4000\n", env);
        }
        return 4000u;
    }();
    if (p < kLimbFftCrossover) {
        LucasLehmerEngine<backend::LimbBackend> eng(p, /*benchmark_mode=*/true);
        return eng.run(progress);
    }

    // FFT/DWT backend for large exponents.
    LucasLehmerEngine<backend::FftMersenneBackend> eng(p, /*benchmark_mode=*/true);
    return eng.run(progress);
}

// lucas_lehmer_ex: like lucas_lehmer() but returns LLResult{is_prime, residue_hex}
// and prints richer per-exponent progress using the supplied ProgressContext.
LLResult lucas_lehmer_ex(uint32_t p, bool progress, bool benchmark_mode,
                          const ProgressContext& ctx) {
    if (p < 2u) return {false, "0000000000000000"};
    if (p == 2u) return {true,  "0000000000000000"};
    if ((p & 1u) == 0u) return {false, "0000000000000000"};
    if (!benchmark_mode && !is_prime_exponent(p)) return {false, "0000000000000000"};

    if (p < 128u) {
        LucasLehmerEngine<backend::GenericBackend> eng(p, /*benchmark_mode=*/true);
        return eng.run_ex(progress, ctx);
    }
    // Reuse the same crossover threshold logic as lucas_lehmer().
    static const uint32_t kCrossoverEx = []() -> uint32_t {
        const char* env = std::getenv("LL_LIMB_FFT_CROSSOVER");
        if (env && *env) {
            char* end = nullptr;
            const long v = std::strtol(env, &end, 10);
            if (end != env && *end == '\0' && v > 0 && v < 1000000L)
                return static_cast<uint32_t>(v);
        }
        return 4000u;
    }();
    if (p < kCrossoverEx) {
        LucasLehmerEngine<backend::LimbBackend> eng(p, /*benchmark_mode=*/true);
        return eng.run_ex(progress, ctx);
    }
    LucasLehmerEngine<backend::FftMersenneBackend> eng(p, /*benchmark_mode=*/true);
    return eng.run_ex(progress, ctx);
}

}  // namespace mersenne

// ============================================================
// main()
// ============================================================
#ifndef BIGNUM_NO_MAIN

// ---- Discover-mode infrastructure ----

// UTF-8 discovery banner prefix (fire+siren emoji: 🚨).
static constexpr const char* kDiscoveryEmoji = "\xF0\x9F\x9A\xA8";

// Format current UTC time as ISO-8601 string into buf[32].
// Returns buf, or "unknown" if gmtime fails.
static const char* iso_timestamp_now(char buf[32]) {
    std::time_t now = std::time(nullptr);
    struct tm* tm_utc = std::gmtime(&now);
    if (!tm_utc) {
        std::snprintf(buf, 32, "unknown");
        return buf;
    }
    std::strftime(buf, 32, "%Y-%m-%dT%H:%M:%SZ", tm_utc);
    return buf;
}

struct DiscoverResult {
    uint64_t    exponent{0};
    bool        is_prime{false};
    bool        is_known{false};
    bool        is_new_discovery{false};
    bool        is_explicit_first{false};
    double      elapsed_sec{0.0};
    unsigned    threads{1};
    uint32_t    shard_index{0};
    uint32_t    shard_count{1};
    std::string backend_name;
};

// Return the backend name that lucas_lehmer() would select for exponent p.
static std::string discover_backend_name(uint64_t p) {
    if (p < 128u) return "GenericBackend";
    const char* env = std::getenv("LL_LIMB_FFT_CROSSOVER");
    uint32_t crossover = 4000u;
    if (env && *env) {
        char* end = nullptr;
        const long v = std::strtol(env, &end, 10);
        if (end != env && *end == '\0' && v > 0 && v < 1000000L)
            crossover = static_cast<uint32_t>(v);
    }
    return p < crossover ? "LimbBackend" : "FftMersenneBackend";
}

static void write_discover_csv(
    const std::string& path,
    const std::vector<DiscoverResult>& results)
{
    std::ofstream f(path);
    if (!f) { std::fprintf(stderr, "Cannot write CSV: %s\n", path.c_str()); return; }
    f << "exponent,result,backend,elapsed_sec,threads,mode,"
         "shard_index,shard_count,is_explicit_first,is_new_discovery\n";
    for (const auto& r : results) {
        char ebuf[32];
        std::snprintf(ebuf, sizeof(ebuf), "%.6f", r.elapsed_sec);
        f << r.exponent << ","
          << (r.is_prime ? "prime" : "composite") << ","
          << r.backend_name << ","
          << ebuf << ","
          << r.threads << ","
          << "discover,"
          << r.shard_index << ","
          << r.shard_count << ","
          << (r.is_explicit_first ? "1" : "0") << ","
          << (r.is_new_discovery ? "1" : "0") << "\n";
    }
}

// Minimal JSON string escaper: handles characters that would break JSON.
static std::string json_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 4);
    for (unsigned char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:
                if (c < 0x20) {
                    char buf[7];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", static_cast<unsigned>(c));
                    out += buf;
                } else {
                    out += static_cast<char>(c);
                }
        }
    }
    return out;
}

static void write_discover_json(
    const std::string& path,
    const std::vector<DiscoverResult>& results,
    uint64_t single_exp, uint64_t min_excl, uint64_t max_incl,
    uint32_t shard_index, uint32_t shard_count,
    const std::string& run_url)
{
    std::ofstream f(path);
    if (!f) { std::fprintf(stderr, "Cannot write JSON: %s\n", path.c_str()); return; }

    char tbuf[32];
    iso_timestamp_now(tbuf);

    const char* sha = std::getenv("GITHUB_SHA");
    if (!sha || !*sha) sha = "";

    f << "{\n"
      << "  \"mode\": \"discover\",\n"
      << "  \"single_exponent\": " << single_exp << ",\n"
      << "  \"min_excl\": " << min_excl << ",\n"
      << "  \"max_incl\": " << max_incl << ",\n"
      << "  \"shard_index\": " << shard_index << ",\n"
      << "  \"shard_count\": " << shard_count << ",\n"
      << "  \"timestamp\": \"" << json_escape(tbuf) << "\",\n"
      << "  \"commit_sha\": \"" << json_escape(sha) << "\",\n"
      << "  \"workflow_run_url\": \"" << json_escape(run_url) << "\",\n"
      << "  \"results\": [\n";

    for (size_t i = 0; i < results.size(); ++i) {
        const auto& r = results[i];
        char ebuf[32];
        std::snprintf(ebuf, sizeof(ebuf), "%.6f", r.elapsed_sec);
        f << "    {\n"
          << "      \"exponent\": " << r.exponent << ",\n"
          << "      \"result\": \"" << (r.is_prime ? "prime" : "composite") << "\",\n"
          << "      \"backend\": \"" << json_escape(r.backend_name) << "\",\n"
          << "      \"elapsed_sec\": " << ebuf << ",\n"
          << "      \"threads\": " << r.threads << ",\n"
          << "      \"shard_index\": " << r.shard_index << ",\n"
          << "      \"shard_count\": " << r.shard_count << ",\n"
          << "      \"is_explicit_first\": " << (r.is_explicit_first ? "true" : "false") << ",\n"
          << "      \"is_known_mersenne_prime\": " << (r.is_known ? "true" : "false") << ",\n"
          << "      \"is_new_discovery\": " << (r.is_new_discovery ? "true" : "false") << "\n"
          << "    }" << (i + 1 < results.size() ? "," : "") << "\n";
    }
    f << "  ]\n}\n";
}

static void write_discovery_event_json(
    const std::string& path,
    uint64_t exponent,
    double elapsed_sec,
    uint32_t shard_index,
    uint32_t shard_count,
    const std::string& run_url)
{
    std::ofstream f(path);
    if (!f) { std::fprintf(stderr, "Cannot write discovery event JSON: %s\n", path.c_str()); return; }

    char tbuf[32];
    iso_timestamp_now(tbuf);

    const char* sha = std::getenv("GITHUB_SHA");
    if (!sha || !*sha) sha = "";

    char ebuf[32];
    std::snprintf(ebuf, sizeof(ebuf), "%.6f", elapsed_sec);

    f << "{\n"
      << "  \"event\": \"NEW_MERSENNE_PRIME_FOUND\",\n"
      << "  \"exponent\": " << exponent << ",\n"
      << "  \"timestamp\": \"" << json_escape(tbuf) << "\",\n"
      << "  \"commit_sha\": \"" << json_escape(sha) << "\",\n"
      << "  \"workflow_run_url\": \"" << json_escape(run_url) << "\",\n"
      << "  \"elapsed_sec\": " << ebuf << ",\n"
      << "  \"shard_index\": " << shard_index << ",\n"
      << "  \"shard_count\": " << shard_count << "\n"
      << "}\n";
}

// Emit a loud, multi-channel discovery notification.
static void emit_discovery_notification(uint64_t p, const std::string& run_url) {
    // GitHub Actions workflow commands produce annotations in the UI.
    std::printf("::notice title=NEW MERSENNE PRIME FOUND::M_%" PRIu64 " is prime! "
                "Exponent %" PRIu64 " not in known list. Run: %s\n",
                p, p, run_url.c_str());
    std::printf("::warning title=NEW MERSENNE PRIME FOUND::M_%" PRIu64 " verified prime "
                "by Lucas-Lehmer. Exponent %" PRIu64 " unknown to the known list.\n",
                p, p);

    // Prominent console banner (width-agnostic: no fixed-width field for exponent).
    std::printf("\n");
    std::printf("***************************************************************\n");
    std::printf("***                                                         ***\n");
    std::printf("***   NEW MERSENNE PRIME CANDIDATE: M_%" PRIu64 "\n", p);
    std::printf("***   Exponent %" PRIu64 " NOT in known Mersenne list!\n", p);
    std::printf("***   VERIFY INDEPENDENTLY BEFORE ANNOUNCING!              ***\n");
    std::printf("***                                                         ***\n");
    std::printf("***************************************************************\n");
    std::printf("\n");

    // Append to GitHub Actions step summary if the env var is set.
    const char* summary_path = std::getenv("GITHUB_STEP_SUMMARY");
    if (summary_path && *summary_path) {
        std::ofstream sf(summary_path, std::ios::app);
        if (sf) {
            sf << "\n## " << kDiscoveryEmoji << " NEW MERSENNE PRIME FOUND: M_" << p << "\n\n"
               << "| Field | Value |\n"
               << "|-------|-------|\n"
               << "| Exponent | " << p << " |\n"
               << "| Status | **NOT** in the known Mersenne prime list |\n"
               << "| Workflow run | " << run_url << " |\n\n"
               << "> \xE2\x9A\xA0\xEF\xB8\x8F Independent verification required before any announcement.\n\n";
        }
    }
    std::fflush(stdout);
}

// Parse an unsigned integer from an environment variable.
// Returns def if the variable is unset/empty/invalid/out-of-range.
static uint32_t env_uint32(const char* name, uint32_t def) {
    const char* s = std::getenv(name);
    if (!s || !*s) return def;
    if (s[0] == '-') {
        std::fprintf(stderr, "Invalid %s='%s' (negative); using default %u\n", name, s, def);
        return def;
    }
    char* end = nullptr;
    errno = 0;
    const unsigned long v = std::strtoul(s, &end, 10);
    if (errno != 0 || end == s || *end != '\0' || v > static_cast<unsigned long>(UINT32_MAX)) {
        std::fprintf(stderr, "Invalid %s='%s'; using default %u\n", name, s, def);
        return def;
    }
    return static_cast<uint32_t>(v);
}

// Parse a 64-bit unsigned integer from an environment variable.
// Returns def if the variable is unset/empty/invalid/out-of-range.
static uint64_t env_uint64(const char* name, uint64_t def) {
    const char* s = std::getenv(name);
    if (!s || !*s) return def;
    if (s[0] == '-') {
        std::fprintf(stderr, "Invalid %s='%s' (negative); using default %" PRIu64 "\n", name, s, def);
        return def;
    }
    char* end = nullptr;
    errno = 0;
    const unsigned long long v = std::strtoull(s, &end, 10);
    if (errno != 0 || end == s || *end != '\0') {
        std::fprintf(stderr, "Invalid %s='%s'; using default %" PRIu64 "\n", name, s, def);
        return def;
    }
    return static_cast<uint64_t>(v);
}

static bool env_bool(const char* name, bool def = false) {
    const char* s = std::getenv(name);
    if (!s || !*s) return def;
    return std::strcmp(s, "1") == 0 || std::strcmp(s, "true") == 0 ||
           std::strcmp(s, "yes") == 0;
}

// Discover-mode entry point (called from main when LL_SWEEP_MODE=discover).
static int run_discover_mode(int argc, char** argv) {
    const unsigned maxCores = runtime::detect_available_cores();

    // --- Parse parameters (env vars, then argv override for threads) ---
    const uint64_t single_exp   = env_uint64("LL_SINGLE_EXPONENT", 0u);
    const uint64_t min_excl     = env_uint64("LL_MIN_EXPONENT",  136279841u);
    const uint64_t max_incl     = env_uint64("LL_MAX_EXPONENT",  200000000u);
    const uint32_t shard_count  = env_uint32("LL_SHARD_COUNT",   1u);
    const uint32_t shard_index  = env_uint32("LL_SHARD_INDEX",   0u);
    const uint64_t stop_after_n = env_uint64("LL_STOP_AFTER_N_CASES", 0u);
    const bool reverse_order    = env_bool("LL_REVERSE_ORDER");
    const bool dry_run          = env_bool("LL_DRY_RUN");
    const bool progress         = env_bool("LL_PROGRESS");

    unsigned threads = maxCores;
    if (argc >= 3 && argv[2] && argv[2][0] != '\0') {
        const char* arg = argv[2];
        char* end = nullptr;
        errno = 0;
        const unsigned long long raw = std::strtoull(arg, &end, 10);
        if (errno != 0 || end == arg || *end != '\0' || arg[0] == '-') {
            std::fprintf(stderr, "Invalid thread count '%s'; using all cores (%u)\n", arg, maxCores);
        } else {
            const unsigned req = (raw > static_cast<unsigned long long>(maxCores))
                                     ? maxCores
                                     : static_cast<unsigned>(raw);
            threads = (req == 0u) ? maxCores : req;
        }
    } else {
        const uint32_t req = env_uint32("LL_THREADS", 0u);
        threads = (req == 0u) ? maxCores : std::min(static_cast<unsigned>(req), maxCores);
    }

    // Construct workflow run URL from GitHub Actions env vars.
    const char* server_url = std::getenv("GITHUB_SERVER_URL");
    const char* repo       = std::getenv("GITHUB_REPOSITORY");
    const char* run_id     = std::getenv("GITHUB_RUN_ID");
    std::string run_url;
    if (server_url && *server_url && repo && *repo && run_id && *run_id)
        run_url = std::string(server_url) + "/" + repo + "/actions/runs/" + run_id;

    // Output directory (defaults to current directory).
    const char* out_dir_env = std::getenv("LL_OUTPUT_DIR");
    std::string out_dir;
    if (out_dir_env && *out_dir_env) {
        out_dir = out_dir_env;
        // Ensure trailing slash.
        if (out_dir.back() != '/') out_dir += '/';
        // Create directory; ignore EEXIST, warn on other errors.
        if (::mkdir(out_dir.c_str(), 0755) != 0 && errno != EEXIST) {
            std::fprintf(stderr,
                "Warning: cannot create output directory '%s': %s\n",
                out_dir.c_str(), std::strerror(errno));
        }
    }

    // --- Generate exponent list ---
    const std::vector<uint64_t> exps = mersenne::discover_exponent_list(
        single_exp, min_excl, max_incl, reverse_order, shard_count, shard_index);

    // --- Print plan ---
    std::printf("=== DISCOVER MODE ===\n");
    std::printf("  single_exponent  : %" PRIu64 "\n", single_exp);
    std::printf("  min_excl         : %" PRIu64 "\n", min_excl);
    std::printf("  max_incl         : %" PRIu64 "\n", max_incl);
    std::printf("  shard            : %u/%u\n", shard_index, shard_count);
    std::printf("  reverse_order    : %s\n", reverse_order ? "yes" : "no");
    std::printf("  stop_after_n     : %" PRIu64 "\n", stop_after_n);
    std::printf("  exponents_in_list: %zu\n", exps.size());
    std::printf("  threads          : %u / %u available\n", threads, maxCores);
    std::printf("  dry_run          : %s\n", dry_run ? "yes" : "no");
    if (!run_url.empty())
        std::printf("  workflow_run     : %s\n", run_url.c_str());
    std::printf("\n");

    if (dry_run) {
        std::printf("DRY RUN: exponent plan (%zu items):\n", exps.size());
        for (size_t i = 0; i < exps.size(); ++i) {
            const bool first = (single_exp != 0u && i == 0u && exps[i] == single_exp);
            std::printf("  [%zu] p=%-20" PRIu64 "  backend=%-20s  %s\n",
                        i, exps[i],
                        discover_backend_name(exps[i]).c_str(),
                        first ? "(explicit first)" : "");
        }
        return 0;
    }

    if (exps.empty()) {
        std::printf("No exponents to test in discover mode. "
                    "Check LL_SINGLE_EXPONENT / LL_MAX_EXPONENT range.\n");
        return 0;
    }

    // --- Run exponents ---
    std::vector<DiscoverResult> results;
    results.reserve(exps.size());

    bool any_new_discovery = false;
    uint64_t tested = 0u;

    for (size_t i = 0; i < exps.size(); ++i) {
        if (stop_after_n != 0u && tested >= stop_after_n) {
            std::printf("Stopping after %" PRIu64 " cases (LL_STOP_AFTER_N_CASES).\n", stop_after_n);
            break;
        }
        const uint64_t p = exps[i];
        const bool is_first = (single_exp != 0u && i == 0u && p == single_exp);

        std::printf("Testing M_%" PRIu64 " ...%s\n", p, is_first ? " [explicit first exponent]" : "");

        // Lucas-Lehmer currently requires p to fit in uint32_t.
        // Exponents > UINT32_MAX cannot be tested by this implementation.
        if (p > UINT32_MAX) {
            std::printf("M_%" PRIu64 ": exponent exceeds uint32 range; skipping LL test.\n", p);
            DiscoverResult r;
            r.exponent         = p;
            r.is_prime         = false;
            r.is_known         = false;
            r.is_new_discovery = false;
            r.is_explicit_first = is_first;
            r.elapsed_sec      = 0.0;
            r.threads          = threads;
            r.shard_index      = shard_index;
            r.shard_count      = shard_count;
            r.backend_name     = "skipped_exceeds_uint32";
            results.push_back(r);
            ++tested;
            continue;
        }
        const uint32_t p32 = static_cast<uint32_t>(p);

        const auto t0     = std::chrono::steady_clock::now();
        const bool isPrime = mersenne::lucas_lehmer(p32, progress, /*benchmark_mode=*/false);
        const auto t1     = std::chrono::steady_clock::now();
        const double elapsed = std::chrono::duration<double>(t1 - t0).count();

        const bool is_known = mersenne::is_known_mersenne_prime(p);
        const bool is_new   = isPrime && !is_known;

        std::printf("M_%" PRIu64 " is %s. Time: %.3f s%s\n",
                    p, isPrime ? "prime" : "composite", elapsed,
                    is_new ? "  *** NEW DISCOVERY ***" : "");

        DiscoverResult r;
        r.exponent         = p;
        r.is_prime         = isPrime;
        r.is_known         = is_known;
        r.is_new_discovery = is_new;
        r.is_explicit_first = is_first;
        r.elapsed_sec      = elapsed;
        r.threads          = threads;
        r.shard_index      = shard_index;
        r.shard_count      = shard_count;
        r.backend_name     = discover_backend_name(p);
        results.push_back(r);

        ++tested;

        if (is_new) {
            any_new_discovery = true;
            emit_discovery_notification(p, run_url);
            write_discovery_event_json(
                out_dir + "discovery_event.json",
                p, elapsed, shard_index, shard_count, run_url);
        }
    }

    // --- Write result files ---
    write_discover_csv(out_dir + "discover_results.csv", results);
    write_discover_json(out_dir + "discover_results.json", results,
                        single_exp, min_excl, max_incl,
                        shard_index, shard_count, run_url);

    // --- Write workflow summary ---
    {
        std::ofstream sf(out_dir + "discover_workflow_summary.md");
        if (sf) {
            sf << "# Discover Mode Summary\n\n"
               << "| Parameter | Value |\n"
               << "|-----------|-------|\n"
               << "| single_exponent | " << single_exp << " |\n"
               << "| min_excl | " << min_excl << " |\n"
               << "| max_incl | " << max_incl << " |\n"
               << "| shard | " << shard_index << " / " << shard_count << " |\n"
               << "| exponents tested | " << tested << " |\n"
               << "| threads | " << threads << " |\n"
               << "| new discoveries | " << (any_new_discovery ? "YES" : "none") << " |\n\n";

            // Per-exponent table (only if not too long).
            if (results.size() <= 200) {
                sf << "## Results\n\n"
                   << "| Exponent | Result | Backend | Elapsed (s) | New? |\n"
                   << "|----------|--------|---------|-------------|------|\n";
                for (const auto& r : results) {
                    char ebuf[32];
                    std::snprintf(ebuf, sizeof(ebuf), "%.3f", r.elapsed_sec);
                    sf << "| " << r.exponent
                       << " | " << (r.is_prime ? "prime" : "composite")
                       << " | " << r.backend_name
                       << " | " << ebuf
                       << " | " << (r.is_new_discovery ? "**YES**" : "no") << " |\n";
                }
                sf << "\n";
            }

            if (any_new_discovery) {
                sf << "## " << kDiscoveryEmoji << " New Discoveries\n\n";
                for (const auto& r : results) {
                    if (r.is_new_discovery)
                        sf << "- **M_" << r.exponent << "** is a new Mersenne prime candidate!\n";
                }
                sf << "\n";
            }

            if (!run_url.empty())
                sf << "**Workflow run:** " << run_url << "\n";
        }

        // Also append to GitHub Actions step summary if available.
        const char* summary_path = std::getenv("GITHUB_STEP_SUMMARY");
        if (summary_path && *summary_path) {
            std::ofstream gsf(summary_path, std::ios::app);
            if (gsf) {
                gsf << "## Discover Mode Results\n\n"
                    << "- Exponents tested: " << tested << "\n"
                    << "- Range: " << min_excl << " < p <= " << max_incl << "\n"
                    << "- Shard: " << shard_index << " / " << shard_count << "\n"
                    << "- New discoveries: " << (any_new_discovery ? "**YES**" : "none") << "\n\n";
                if (!run_url.empty())
                    gsf << "Workflow run: " << run_url << "\n\n";
            }
        }
    }

    std::printf("\nDiscover mode complete. Tested %" PRIu64 " exponent(s).\n", tested);
    if (any_new_discovery) {
        std::printf("NEW DISCOVERIES FOUND — see discovery_event.json and discover_results.json\n");
        return 3;  // distinct exit code for discovery
    }
    return 0;
}

// ---- Power-bucket mode infrastructure ----

struct BucketResult {
    uint64_t    exponent{0};
    uint32_t    bucket_n{0};
    uint64_t    bucket_lo{0};
    uint64_t    bucket_hi{0};
    size_t      exponent_index{0};           // 1-based index within bucket
    size_t      bucket_total_exponents{0};   // total exponents in bucket
    bool        is_prime{false};
    bool        is_known{false};
    bool        is_new_discovery{false};
    double      elapsed_sec{0.0};
    uint32_t    iterations_total{0};         // p - 2
    double      avg_iter_seconds{0.0};       // elapsed_sec / iterations_total
    unsigned    threads{1};
    std::string backend_name;
    std::string final_residue_hex{"0000000000000000"};
};

static void write_bucket_csv(
    const std::string& path,
    const std::vector<BucketResult>& results)
{
    std::ofstream f(path);
    if (!f) { std::fprintf(stderr, "Cannot write CSV: %s\n", path.c_str()); return; }
    f << "exponent,bucket_n,bucket_lo,bucket_hi,exponent_index,bucket_total,"
         "result,backend,elapsed_sec,iterations_total,avg_iter_seconds,"
         "threads,is_known_mersenne,is_new_discovery,final_residue_hex\n";
    for (const auto& r : results) {
        char ebuf[32], abuf[32];
        std::snprintf(ebuf, sizeof(ebuf), "%.6f", r.elapsed_sec);
        std::snprintf(abuf, sizeof(abuf), "%.9f", r.avg_iter_seconds);
        f << r.exponent << ","
          << r.bucket_n << ","
          << r.bucket_lo << ","
          << r.bucket_hi << ","
          << r.exponent_index << ","
          << r.bucket_total_exponents << ","
          << (r.is_prime ? "prime" : "composite") << ","
          << r.backend_name << ","
          << ebuf << ","
          << r.iterations_total << ","
          << abuf << ","
          << r.threads << ","
          << (r.is_known ? "1" : "0") << ","
          << (r.is_new_discovery ? "1" : "0") << ","
          << r.final_residue_hex << "\n";
    }
}

static void write_bucket_json(
    const std::string& path,
    const std::vector<BucketResult>& results,
    uint32_t bucket_n, uint64_t bucket_lo, uint64_t bucket_hi,
    const std::string& run_url)
{
    std::ofstream f(path);
    if (!f) { std::fprintf(stderr, "Cannot write JSON: %s\n", path.c_str()); return; }
    char tbuf[32];
    iso_timestamp_now(tbuf);
    const char* sha = std::getenv("GITHUB_SHA");
    if (!sha || !*sha) sha = "";

    f << "{\n"
      << "  \"mode\": \"power_bucket_primes\",\n"
      << "  \"bucket_n\": " << bucket_n << ",\n"
      << "  \"bucket_lo\": " << bucket_lo << ",\n"
      << "  \"bucket_hi\": " << bucket_hi << ",\n"
      << "  \"timestamp\": \"" << json_escape(tbuf) << "\",\n"
      << "  \"commit_sha\": \"" << json_escape(sha) << "\",\n"
      << "  \"workflow_run_url\": \"" << json_escape(run_url) << "\",\n"
      << "  \"results\": [\n";

    for (size_t i = 0; i < results.size(); ++i) {
        const auto& r = results[i];
        char ebuf[32], abuf[32];
        std::snprintf(ebuf, sizeof(ebuf), "%.6f", r.elapsed_sec);
        std::snprintf(abuf, sizeof(abuf), "%.9f", r.avg_iter_seconds);
        f << "    {\n"
          << "      \"exponent\": " << r.exponent << ",\n"
          << "      \"bucket_n\": " << r.bucket_n << ",\n"
          << "      \"bucket_min_exponent\": " << r.bucket_lo << ",\n"
          << "      \"bucket_max_exponent\": " << r.bucket_hi << ",\n"
          << "      \"bucket_exponent_index\": " << r.exponent_index << ",\n"
          << "      \"bucket_total_exponents\": " << r.bucket_total_exponents << ",\n"
          << "      \"result\": \"" << (r.is_prime ? "prime" : "composite") << "\",\n"
          << "      \"backend\": \"" << json_escape(r.backend_name) << "\",\n"
          << "      \"elapsed_sec\": " << ebuf << ",\n"
          << "      \"iterations_total\": " << r.iterations_total << ",\n"
          << "      \"avg_iter_seconds\": " << abuf << ",\n"
          << "      \"threads\": " << r.threads << ",\n"
          << "      \"is_known_mersenne_prime\": " << (r.is_known ? "true" : "false") << ",\n"
          << "      \"is_new_discovery\": " << (r.is_new_discovery ? "true" : "false") << ",\n"
          << "      \"final_residue_hex\": \"" << r.final_residue_hex << "\"\n"
          << "    }" << (i + 1 < results.size() ? "," : "") << "\n";
    }
    f << "  ]\n}\n";
}

static int run_power_bucket_mode(int argc, char** argv) {
    const unsigned maxCores = runtime::detect_available_cores();

    // --- Parse parameters ---
    const uint32_t bucket_n      = env_uint32("LL_BUCKET_N", 0u);
    const uint32_t bucket_start  = env_uint32("LL_BUCKET_START", 1u);
    const uint32_t bucket_end    = env_uint32("LL_BUCKET_END", 64u);
    const uint64_t max_exps      = env_uint64("LL_MAX_EXPONENTS_PER_JOB", 0u);
    const uint64_t resume_from   = env_uint64("LL_RESUME_FROM_EXPONENT", 0u);
    const bool reverse_order     = env_bool("LL_REVERSE_ORDER");
    const bool dry_run           = env_bool("LL_DRY_RUN");
    const bool progress          = env_bool("LL_PROGRESS");
    const bool benchmark_mode    = env_bool("LL_BENCHMARK_MODE", false);
    const bool stop_first_prime  = env_bool("LL_STOP_AFTER_FIRST_PRIME_RESULT");

    // Configurable checkpoint intervals for per-exponent progress output.
    const uint32_t prog_interval_iters = env_uint32("LL_PROGRESS_INTERVAL_ITERS", 10000u);
    const double prog_interval_secs = [&]() -> double {
        const char* s = std::getenv("LL_PROGRESS_INTERVAL_SECONDS");
        if (!s || !*s) return 0.0;
        char* end = nullptr;
        const double v = std::strtod(s, &end);
        return (end != s && *end == '\0' && v > 0.0) ? v : 0.0;
    }();

    unsigned threads = maxCores;
    if (argc >= 3 && argv[2] && argv[2][0] != '\0') {
        const char* arg = argv[2];
        char* end = nullptr;
        errno = 0;
        const unsigned long long raw = std::strtoull(arg, &end, 10);
        if (errno != 0 || end == arg || *end != '\0' || arg[0] == '-') {
            std::fprintf(stderr, "Invalid thread count '%s'; using all cores (%u)\n",
                         arg, maxCores);
        } else {
            const unsigned req = (raw > static_cast<unsigned long long>(maxCores))
                                     ? maxCores : static_cast<unsigned>(raw);
            threads = (req == 0u) ? maxCores : req;
        }
    } else {
        const uint32_t req = env_uint32("LL_THREADS", 0u);
        threads = (req == 0u) ? maxCores : std::min(static_cast<unsigned>(req), maxCores);
    }

    // Determine bucket range.
    const uint32_t n_lo = (bucket_n != 0u) ? bucket_n
                                            : std::max(1u, std::min(bucket_start, 64u));
    const uint32_t n_hi = (bucket_n != 0u) ? bucket_n
                                            : std::max(n_lo, std::min(bucket_end, 64u));

    // Workflow run URL.
    const char* server_url = std::getenv("GITHUB_SERVER_URL");
    const char* repo       = std::getenv("GITHUB_REPOSITORY");
    const char* run_id     = std::getenv("GITHUB_RUN_ID");
    std::string run_url;
    if (server_url && *server_url && repo && *repo && run_id && *run_id)
        run_url = std::string(server_url) + "/" + repo + "/actions/runs/" + run_id;

    // Output directory.
    const char* out_dir_env = std::getenv("LL_OUTPUT_DIR");
    std::string out_dir;
    if (out_dir_env && *out_dir_env) {
        out_dir = out_dir_env;
        if (out_dir.back() != '/') out_dir += '/';
        if (::mkdir(out_dir.c_str(), 0755) != 0 && errno != EEXIST)
            std::fprintf(stderr, "Warning: cannot create output directory '%s': %s\n",
                         out_dir.c_str(), std::strerror(errno));
    }

    std::printf("=== POWER BUCKET PRIME SWEEP ===\n");
    std::printf("  bucket range     : %u..%u\n", n_lo, n_hi);
    std::printf("  threads          : %u / %u available\n", threads, maxCores);
    std::printf("  dry_run          : %s\n", dry_run ? "yes" : "no");
    std::printf("  reverse_order    : %s\n", reverse_order ? "yes" : "no");
    if (max_exps > 0u)
        std::printf("  max_exps_per_job : %" PRIu64 "\n", max_exps);
    if (resume_from > 0u)
        std::printf("  resume_from      : %" PRIu64 "\n", resume_from);
    if (!run_url.empty())
        std::printf("  workflow_run     : %s\n", run_url.c_str());
    std::printf("\n");

    bool any_new_discovery = false;

    for (uint32_t n = n_lo; n <= n_hi; ++n) {
        const power_bucket::Range br = power_bucket::bucket_range(n);
        std::vector<uint64_t> exps = power_bucket::enumerate_bucket_primes(n);

        // Resume filter.
        if (resume_from > 0u)
            exps.erase(std::remove_if(exps.begin(), exps.end(),
                           [resume_from](uint64_t p) { return p < resume_from; }),
                       exps.end());

        // Reverse ordering.
        if (reverse_order) std::reverse(exps.begin(), exps.end());

        // Per-job cap.
        if (max_exps > 0u && exps.size() > static_cast<size_t>(max_exps))
            exps.resize(static_cast<size_t>(max_exps));

        std::printf("--- Bucket %u: [%" PRIu64 ", %" PRIu64 "] — %zu prime exponent(s) ---\n",
                    n, br.lo, br.hi, exps.size());

        if (dry_run) {
            for (size_t i = 0; i < exps.size(); ++i)
                std::printf("  [%zu] p=%-20" PRIu64 "  backend=%s\n",
                            i, exps[i], discover_backend_name(exps[i]).c_str());
            continue;
        }

        if (exps.empty()) {
            std::printf("  (no prime exponents in bucket %u)\n", n);
            continue;
        }

        // Per-bucket output subdirectory.
        std::string bdir;
        if (!out_dir.empty()) {
            bdir = out_dir + "bucket_" + std::to_string(n) + "/";
            if (::mkdir(bdir.c_str(), 0755) != 0 && errno != EEXIST)
                std::fprintf(stderr, "Warning: cannot create bucket dir '%s': %s\n",
                             bdir.c_str(), std::strerror(errno));
        }

        std::vector<BucketResult> results;
        results.reserve(exps.size());

        // Bucket-level progress tracking.
        const auto bucket_t0 = std::chrono::steady_clock::now();
        size_t bucket_composites     = 0u;
        size_t bucket_known_primes   = 0u;
        size_t bucket_new_disc       = 0u;
        double bucket_elapsed_total  = 0.0;

        for (size_t exp_idx = 0u; exp_idx < exps.size(); ++exp_idx) {
            const uint64_t p = exps[exp_idx];
            if (stop_first_prime && any_new_discovery) break;

            const std::string bname = (p <= UINT32_MAX)
                                      ? discover_backend_name(p)
                                      : "skipped_exceeds_uint32";
            const uint32_t iters_total = (p > 1u && p <= UINT32_MAX)
                                         ? static_cast<uint32_t>(p) - 2u : 0u;

            std::printf("  [B%u exp %zu/%zu] Testing M_%" PRIu64
                        " (%u iters, backend=%s) ...\n",
                        n, exp_idx + 1u, exps.size(), p,
                        iters_total, bname.c_str());
            std::fflush(stdout);

            BucketResult res;
            res.exponent                = p;
            res.bucket_n                = n;
            res.bucket_lo               = br.lo;
            res.bucket_hi               = br.hi;
            res.exponent_index          = exp_idx + 1u;
            res.bucket_total_exponents  = exps.size();
            res.iterations_total        = iters_total;
            res.threads                 = threads;
            res.is_known                = mersenne::is_known_mersenne_prime(p);

            if (p > UINT32_MAX) {
                res.backend_name = "skipped_exceeds_uint32";
                res.is_prime     = false;
                std::printf("  M_%" PRIu64 ": skipped (exceeds uint32_t limit)\n", p);
            } else {
                ProgressContext ctx;
                ctx.bucket_n       = n;
                ctx.bucket_lo      = br.lo;
                ctx.bucket_hi      = br.hi;
                ctx.exp_index      = exp_idx + 1u;
                ctx.exp_total      = exps.size();
                ctx.interval_iters = prog_interval_iters;
                ctx.interval_secs  = prog_interval_secs;

                const auto t0 = std::chrono::steady_clock::now();
                const LLResult llr = mersenne::lucas_lehmer_ex(
                    static_cast<uint32_t>(p), progress, benchmark_mode, ctx);
                const auto t1 = std::chrono::steady_clock::now();
                res.elapsed_sec      = std::chrono::duration<double>(t1 - t0).count();
                res.is_prime         = llr.is_prime;
                res.final_residue_hex = llr.final_residue_hex;
                res.backend_name     = bname;
                if (iters_total > 0u && res.elapsed_sec > 0.0)
                    res.avg_iter_seconds = res.elapsed_sec /
                                          static_cast<double>(iters_total);
                std::printf("  M_%" PRIu64 " is %s. Time: %.3f s"
                            "  residue: %s\n",
                            p, res.is_prime ? "prime" : "composite",
                            res.elapsed_sec, res.final_residue_hex.c_str());
            }

            bucket_elapsed_total += res.elapsed_sec;
            if (res.is_prime) {
                if (res.is_known) ++bucket_known_primes;
                else              ++bucket_new_disc;
            } else {
                ++bucket_composites;
            }

            res.is_new_discovery = res.is_prime && !res.is_known;
            if (res.is_new_discovery) {
                any_new_discovery = true;
                emit_discovery_notification(p, run_url);
                if (!bdir.empty())
                    write_discovery_event_json(
                        bdir + "discovery_event.json",
                        p, res.elapsed_sec, 0u, 1u, run_url);
            }

            // Bucket-level progress line after each exponent.
            {
                const auto now_b = std::chrono::steady_clock::now();
                const double b_elapsed =
                    std::chrono::duration<double>(now_b - bucket_t0).count();
                const size_t done = exp_idx + 1u;
                const size_t remaining = exps.size() - done;
                const double avg_exp = (done > 0u)
                    ? bucket_elapsed_total / static_cast<double>(done) : 0.0;
                const double eta_bucket = avg_exp * static_cast<double>(remaining);
                const double pct_bucket =
                    100.0 * static_cast<double>(done) /
                            static_cast<double>(exps.size());
                std::printf("  [Bucket %u progress] %zu/%zu done (%.1f%%)"
                            "  elapsed %.1fs  ETA %.1fs"
                            "  primes=%zu (known=%zu new=%zu)"
                            "  composites=%zu\n",
                            n, done, exps.size(), pct_bucket,
                            b_elapsed, eta_bucket,
                            bucket_known_primes + bucket_new_disc,
                            bucket_known_primes, bucket_new_disc,
                            bucket_composites);
                std::fflush(stdout);
            }

            results.push_back(std::move(res));
        }

        if (!bdir.empty()) {
            const std::string pfx = bdir + "bucket_" + std::to_string(n);
            write_bucket_csv (pfx + "_results.csv",  results);
            write_bucket_json(pfx + "_results.json", results,
                              n, br.lo, br.hi, run_url);

            std::ofstream sf(pfx + "_summary.md");
            if (sf) {
                // Aggregate stats.
                size_t n_prime = 0u, n_composite = 0u, n_known = 0u, n_new = 0u;
                double total_time = 0.0;
                for (const auto& r : results) {
                    if (r.is_prime) ++n_prime; else ++n_composite;
                    if (r.is_known) ++n_known;
                    if (r.is_new_discovery) ++n_new;
                    total_time += r.elapsed_sec;
                }
                const double avg_time = results.empty() ? 0.0
                    : total_time / static_cast<double>(results.size());
                char ttbuf[32], atbuf[32];
                std::snprintf(ttbuf, sizeof(ttbuf), "%.3f", total_time);
                std::snprintf(atbuf, sizeof(atbuf), "%.3f", avg_time);

                sf << "# Bucket " << n << " Summary\n\n"
                   << "- **Range**: [" << br.lo << ", " << br.hi << "]\n"
                   << "- **Total tested**: " << results.size() << "\n"
                   << "- **Primes**: " << n_prime
                   << " (known: " << n_known << ", new: " << n_new << ")\n"
                   << "- **Composites**: " << n_composite << "\n"
                   << "- **Total time**: " << ttbuf << " s\n"
                   << "- **Avg time**: " << atbuf << " s/exponent\n\n";

                // Verified known Mersenne primes.
                sf << "## Verified Known Mersenne Primes (" << n_known << ")\n\n";
                if (n_known == 0u) {
                    sf << "_None in this bucket._\n\n";
                } else {
                    sf << "| Exponent | Backend | Elapsed (s) | Residue |\n"
                       << "|----------|---------|-------------|--------|\n";
                    for (const auto& r : results) {
                        if (!r.is_known) continue;
                        char ebuf[32];
                        std::snprintf(ebuf, sizeof(ebuf), "%.3f", r.elapsed_sec);
                        sf << "| " << r.exponent << " | " << r.backend_name
                           << " | " << ebuf
                           << " | `" << r.final_residue_hex << "` |\n";
                    }
                    sf << "\n";
                }

                // New discoveries.
                sf << "## New Mersenne Prime Discoveries (" << n_new << ")\n\n";
                if (n_new == 0u) {
                    sf << "_None in this bucket._\n\n";
                } else {
                    sf << "| Exponent | Backend | Elapsed (s) | Residue |\n"
                       << "|----------|---------|-------------|--------|\n";
                    for (const auto& r : results) {
                        if (!r.is_new_discovery) continue;
                        char ebuf[32];
                        std::snprintf(ebuf, sizeof(ebuf), "%.3f", r.elapsed_sec);
                        sf << "| **" << r.exponent << "** | " << r.backend_name
                           << " | " << ebuf
                           << " | `" << r.final_residue_hex << "` |\n";
                    }
                    sf << "\n";
                }

                // All tested exponents.
                sf << "## All Tested Exponents\n\n"
                   << "| Exponent | Result | Backend | Elapsed (s)"
                      " | Residue | Known | New |\n"
                   << "|----------|--------|---------|-------------|"
                      "---------|-------|-----|\n";
                for (const auto& r : results) {
                    char ebuf[32];
                    std::snprintf(ebuf, sizeof(ebuf), "%.3f", r.elapsed_sec);
                    sf << "| " << r.exponent
                       << " | " << (r.is_prime ? "prime" : "composite")
                       << " | " << r.backend_name
                       << " | " << ebuf
                       << " | `" << r.final_residue_hex << "`"
                       << " | " << (r.is_known ? "yes" : "no")
                       << " | " << (r.is_new_discovery ? "**YES**" : "no")
                       << " |\n";
                }

                if (!run_url.empty())
                    sf << "\n**Workflow run:** " << run_url << "\n";
            }
        }
    }

    if (dry_run) return 0;

    // Append to GitHub Actions step summary.
    const char* summary_path = std::getenv("GITHUB_STEP_SUMMARY");
    if (summary_path && *summary_path) {
        std::ofstream gsf(summary_path, std::ios::app);
        if (gsf) {
            gsf << "## Power Bucket Prime Sweep\n\n"
                << "- Bucket range: " << n_lo << ".." << n_hi << "\n"
                << "- New discoveries: " << (any_new_discovery ? "**YES**" : "none") << "\n";
            if (!run_url.empty())
                gsf << "- Workflow run: " << run_url << "\n";
            gsf << "\n";
        }
    }

    std::printf("\nPower bucket sweep complete.\n");
    if (any_new_discovery) {
        std::printf("NEW DISCOVERIES FOUND — see bucket output directories.\n");
        return 3;
    }
    return 0;
}

static bool test_exponent(uint32_t p, bool progress, bool benchmark_mode) {
    std::printf("Testing M_%u ...\n", p);
    const auto t0     = std::chrono::steady_clock::now();
    const bool isPrime = mersenne::lucas_lehmer(p, progress, benchmark_mode);
    const auto t1     = std::chrono::steady_clock::now();
    const std::chrono::duration<double> elapsed = t1 - t0;
    std::printf("M_%u is %s. Time: %.3f s\n",
                p, isPrime ? "prime" : "composite", elapsed.count());
    return isPrime;
}

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);

    // Dispatch to discover mode when LL_SWEEP_MODE=discover.
    {
        const char* sweep_mode = std::getenv("LL_SWEEP_MODE");
        if (sweep_mode && std::strcmp(sweep_mode, "discover") == 0)
            return run_discover_mode(argc, argv);
        if (sweep_mode && std::strcmp(sweep_mode, "power_bucket_primes") == 0)
            return run_power_bucket_mode(argc, argv);
    }

    const unsigned maxCores = runtime::detect_available_cores();

    // --- Parse legacy positional arguments (backward-compatible) ---
    size_t   startIndex = 0u;
    unsigned threads    = maxCores;
    bool     progress   = false;

    if (argc >= 2 && argv[1] && argv[1][0] != '\0')
        startIndex = static_cast<size_t>(std::strtoull(argv[1], nullptr, 10));
    if (argc >= 3 && argv[2] && argv[2][0] != '\0') {
        const unsigned requested =
            static_cast<unsigned>(std::strtoull(argv[2], nullptr, 10));
        threads = (requested == 0u) ? maxCores : std::min(requested, maxCores);
    }
    if (argc >= 4) progress = true;

    // --- Environment-variable helpers ---
    auto read_env_ul = [](const char* name, unsigned long def) -> unsigned long {
        const char* s = std::getenv(name);
        if (!s || !*s) return def;
        char* end = nullptr;
        const unsigned long v = std::strtoul(s, &end, 10);
        if (end == s || *end != '\0') return def;
        return v;
    };
    auto read_env_bool = [](const char* name, bool def) -> bool {
        const char* s = std::getenv(name);
        if (!s || !*s) return def;
        return std::strtoull(s, nullptr, 10) != 0ull;
    };

    // --- New environment variables ---
    const char* sweep_mode_env = std::getenv("LL_SWEEP_MODE");
    const char  sweep_mode     = (sweep_mode_env && *sweep_mode_env)
                                 ? sweep_mode_env[0] : '\0';  // 'n', 'p', 'm', or 0

    // LL_MIN_EXPONENT / LL_MAX_EXPONENT: value bounds for sweep modes.
    const uint32_t min_exp = static_cast<uint32_t>(read_env_ul("LL_MIN_EXPONENT", 2ul));
    // Default max_exp: when sweep mode is active but LL_MAX_EXPONENT is not set,
    // cap at the largest known Mersenne prime exponent to avoid accidental giant sweeps.
    const auto& kmp_list = mersenne::known_mersenne_prime_exponents();
    const unsigned long default_max_exp =
        (sweep_mode != '\0' && !kmp_list.empty())
        ? static_cast<unsigned long>(kmp_list.back())
        : static_cast<unsigned long>(UINT32_MAX);
    const uint32_t max_exp = static_cast<uint32_t>(
        read_env_ul("LL_MAX_EXPONENT", default_max_exp));

    // LL_MAX_CASES: hard cap on number of exponents to process.
    const size_t max_cases = static_cast<size_t>(
        read_env_ul("LL_MAX_CASES", static_cast<unsigned long>(SIZE_MAX)));

    // LL_SHARD_INDEX / LL_SHARD_COUNT: divide work across parallel jobs.
    const size_t shard_index = static_cast<size_t>(read_env_ul("LL_SHARD_INDEX", 0ul));
    const size_t shard_count = static_cast<size_t>(read_env_ul("LL_SHARD_COUNT", 1ul));

    // LL_REVERSE_ORDER: reverse the work list before executing.
    const bool reverse_order = read_env_bool("LL_REVERSE_ORDER", false);

    // LL_LARGEST_FIRST: sort work list largest-p-first for better load balance.
    const bool largest_first = read_env_bool("LL_LARGEST_FIRST", false);

    // LL_THREADS: override thread count (if not already set via argv[2]).
    if (argc < 3) {
        const unsigned long t = read_env_ul("LL_THREADS", 0ul);
        if (t > 0ul)
            threads = std::min(static_cast<unsigned>(t), maxCores);
    }

    // LL_STOP_AFTER_ONE: run exactly one exponent (existing behaviour).
    const bool stopAfterOne = read_env_bool("LL_STOP_AFTER_ONE", false);

    // LL_BENCHMARK_MODE: skip is_prime_exponent() check inside lucas_lehmer().
    // Default: true for Mersenne list / 'p' / 'm' (all known prime); false for 'n'.
    const bool bm_default = (sweep_mode == 'n') ? false : true;
    const bool benchmark_mode = read_env_bool("LL_BENCHMARK_MODE", bm_default);

    // --- Build the work list ---
    std::vector<uint32_t> work;

    if (sweep_mode == 'n') {
        work = sweep::generate_natural(min_exp, max_exp);
    } else if (sweep_mode == 'p') {
        work = sweep::generate_prime(min_exp, max_exp);
    } else if (sweep_mode == 'm') {
        work = sweep::generate_mersenne_first(min_exp, max_exp);
    } else {
        // Default: use the known Mersenne-prime exponent list with index bounds.
        const auto& exponents = mersenne::known_mersenne_prime_exponents();
        if (startIndex >= exponents.size()) {
            std::fprintf(stderr, "Invalid start index: %zu (max %zu)\n",
                         startIndex, exponents.size() - 1u);
            return 1;
        }

        // LL_MAX_EXPONENT_INDEX: backward-compat exclusive upper bound on the list index.
        const size_t maxExponentIndex = [&exponents] {
            const char* s = std::getenv("LL_MAX_EXPONENT_INDEX");
            if (s && *s != '\0') {
                char* end = nullptr;
                const unsigned long v = std::strtoul(s, &end, 10);
                if (end == s || *end != '\0' || v > exponents.size()) {
                    std::fprintf(stderr,
                                 "Ignoring invalid LL_MAX_EXPONENT_INDEX=\"%s\"; "
                                 "must be an integer in [0, %zu].\n",
                                 s, exponents.size());
                    return exponents.size();
                }
                return static_cast<size_t>(v);
            }
            return exponents.size();
        }();

        const size_t endIndex = std::min(exponents.size(), maxExponentIndex);
        if (!stopAfterOne && startIndex >= endIndex) {
            std::fprintf(stderr,
                         "No exponents to test: startIndex=%zu >= endIndex=%zu. "
                         "Check LL_MAX_EXPONENT_INDEX.\n",
                         startIndex, endIndex);
            return 1;
        }
        for (size_t i = startIndex; i < endIndex; ++i)
            work.push_back(exponents[i]);
    }

    // --- Apply shard selection ---
    if (shard_count > 1u) {
        if (shard_index >= shard_count) {
            std::fprintf(stderr,
                         "Invalid shard configuration: LL_SHARD_INDEX=%zu >= "
                         "LL_SHARD_COUNT=%zu\n",
                         shard_index, shard_count);
            return 1;
        }
        work = sweep::apply_shard(work, shard_index, shard_count);
    }

    // --- Apply max_cases cap ---
    if (max_cases < work.size())
        work.resize(max_cases);

    // --- Ordering: largest-first or reverse ---
    if (largest_first)
        std::sort(work.begin(), work.end(), std::greater<uint32_t>());
    else if (reverse_order)
        std::reverse(work.begin(), work.end());

    if (work.empty() && !stopAfterOne) {
        std::fprintf(stderr, "Work list is empty after filtering.\n");
        return 1;
    }

    std::printf("Using %u worker(s) out of %u available core(s).\n",
                threads, maxCores);

    // --- Optional machine-readable CSV output ---
    const char* bench_output_path = std::getenv("LL_BENCH_OUTPUT");
    FILE* bench_fp = nullptr;
    if (bench_output_path && *bench_output_path) {
        bench_fp = std::fopen(bench_output_path, "w");
        if (bench_fp)
            std::fprintf(bench_fp, "p,is_prime,time_sec\n");
        else
            std::fprintf(stderr, "Warning: cannot open LL_BENCH_OUTPUT='%s'\n",
                         bench_output_path);
    }
    std::mutex bench_mu;
    auto record_result = [&](uint32_t p, bool isPrime, double elapsed) {
        if (!bench_fp) return;
        std::lock_guard<std::mutex> lk(bench_mu);
        std::fprintf(bench_fp, "%u,%s,%.6f\n",
                     p, isPrime ? "true" : "false", elapsed);
        std::fflush(bench_fp);
    };

    // --- StopAfterOne: run exactly one exponent ---
    if (stopAfterOne) {
        if (work.empty()) {
            // Fall back to startIndex in the known list (existing behavior).
            const auto& exponents = mersenne::known_mersenne_prime_exponents();
            if (startIndex >= exponents.size()) {
                std::fprintf(stderr, "No exponents available.\n");
                if (bench_fp) std::fclose(bench_fp);
                return 1;
            }
            work.push_back(exponents[startIndex]);
        }
        const uint32_t p = work[0];
        const bool isPrime = test_exponent(p, progress, benchmark_mode);
        record_result(p, isPrime, 0.0);  // timing already printed by test_exponent
        if (bench_fp) std::fclose(bench_fp);
        return isPrime ? 0 : 2;
    }

    // --- Sequential execution ---
    if (threads == 1u) {
        for (uint32_t p : work) {
            const auto t0     = std::chrono::steady_clock::now();
            const bool isPrime = test_exponent(p, progress, benchmark_mode);
            const auto t1     = std::chrono::steady_clock::now();
            record_result(p, isPrime,
                          std::chrono::duration<double>(t1 - t0).count());
        }
        if (bench_fp) std::fclose(bench_fp);
        return 0;
    }

    // --- Thread-pool throughput mode ---
    // Workers pull from the pre-built work list via an atomic index counter.
    std::atomic<size_t> next{0u};
    std::mutex          printMu;
    runtime::ThreadPool pool(threads);

    for (unsigned t = 0; t < threads; ++t) {
        pool.submit([&] {
            for (;;) {
                const size_t idx = next.fetch_add(1u, std::memory_order_relaxed);
                if (idx >= work.size()) break;
                const uint32_t p = work[idx];
                const auto t0    = std::chrono::steady_clock::now();
                const bool isPrime =
                    mersenne::lucas_lehmer(p, progress, benchmark_mode);
                const auto t1 = std::chrono::steady_clock::now();
                const double elapsed =
                    std::chrono::duration<double>(t1 - t0).count();
                {
                    std::lock_guard<std::mutex> lk(printMu);
                    std::printf("Testing M_%u ...\n", p);
                    std::printf("M_%u is %s. Time: %.3f s\n",
                                p, isPrime ? "prime" : "composite", elapsed);
                }
                record_result(p, isPrime, elapsed);
            }
        });
    }

    pool.wait_all();
    if (bench_fp) std::fclose(bench_fp);
    return 0;
}
#endif
