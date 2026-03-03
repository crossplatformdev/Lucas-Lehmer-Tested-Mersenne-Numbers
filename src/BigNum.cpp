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

bool is_prime_exponent(uint32_t n) {
    if (n < 2u) return false;
    if (n == 2u) return true;
    if ((n & 1u) == 0u) return false;
    if (n % 3u == 0u) return n == 3u;
    if (n % 5u == 0u) return n == 5u;
    const uint64_t n64 = n;
    for (uint32_t i = 7u; (uint64_t)i * i <= n64; i += 2u) {
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

}  // namespace mersenne

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

    // Bit-reversal permutation table (length n).
    std::vector<size_t>   bitrv;

    // Precomputed exp2 tables for carry propagation (length n).
    std::vector<double>   mod_pow;      // 2^{digit_width[j]} per digit
    // inv_mod_pow[j] == 1.0/mod_pow[j], half_pow[j] == 0.5*mod_pow[j]: derived inline.

    // Working buffers (length n) – reused across every iteration.
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
    for (size_t j = 0; j < n; ++j)
        st.mod_pow[j] = std::ldexp(1.0, st.digit_width[j]);
}

// In-place iterative Cooley–Tukey DIT FFT of length n (must be power of 2).
// Data is split into re[]/im[] arrays.  sign = +1 → forward, −1 → inverse.
// The inverse does NOT divide by n; caller handles the n·w_inv[j] factor.
static void fft_core(double* re, double* im, const size_t* bitrv,
                     const double* tw_re, const double* tw_im,
                     size_t n, int sign) {
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
                const double wi  = sign >= 0 ? tw_im[m] : -tw_im[m];
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

// Square st.digits in-place, mod 2^p − 1, via DWT/FFT.
// Result is stored back in st.digits.
static void fft_square(FftMersenneState& st) {
    const size_t n = st.n;
    double* re = st.buf_re.data();
    double* im = st.buf_im.data();

    // Forward DWT: ỹ_j = d_j / w_j  (multiply by 1/w_fwd[j]).
    for (size_t j = 0; j < n; ++j) {
        re[j] = st.w_fwd_inv[j] * st.digits[j];
        im[j] = 0.0;
    }

    // Forward FFT.
    fft_core(re, im, st.bitrv.data(), st.tw_re.data(), st.tw_im.data(), n, +1);

    // Pointwise square: Z_k = Y_k · Y_k.
    for (size_t k = 0; k < n; ++k) {
        const double r = re[k], i = im[k];
        re[k] = r * r - i * i;
        im[k] = 2.0 * r * i;
    }

    // Inverse FFT (unnormalized – no 1/n; that factor is absorbed into w_inv).
    fft_core(re, im, st.bitrv.data(), st.tw_re.data(), st.tw_im.data(), n, -1);

    // Inverse DWT: d'_j = z_j · (w_fwd[j] / n).
    for (size_t j = 0; j < n; ++j)
        st.digits[j] = re[j] * st.w_inv[j];

    // Half-integer correction (Mersenne-specific).
    //
    // Background: for diagonal squaring pair (j,j) where carry(j,j)=1
    // (i.e. frac(j·p/n) ≥ 0.5) and d_j is odd, the inverse DWT places
    // the value d_j²/2 at position k = (2j) mod n, giving a half-integer
    // pre-carry digit.
    //
    // Interpretation: 0.5 at bit-position B_k represents 2^{B_k−1}.
    // Since B_k−1 = B_{k−1} + b_{k−1} − 1, adding 2^{b_{k−1}−1} to
    // digit (k−1) mod n (with Mersenne wrap-around for k=0) gives the
    // correct integer representation and avoids rounding ambiguity.
    //
    // Detection: |frac(d'_k) − 0.5| < 0.25  (FFT error is < 0.45 away
    // from the true value, so a genuine half-integer is never ambiguous).
    for (size_t k = 0; k < n; ++k) {
        const double v    = st.digits[k];
        const double frac = v - std::floor(v);
        if (std::abs(frac - 0.5) < 0.25) {
            // Remove the 0.5 from digit k.
            st.digits[k] = std::floor(v);
            // Propagate the half-unit backward (mod n, Mersenne wrap).
            const size_t kp = (k == 0) ? (n - 1) : (k - 1);
            st.digits[kp] += 0.5 * st.mod_pow[kp];
        }
    }

    // Carry propagation: reduce each digit to [0, 2^{b_j}).
    // Carries propagate upward; the final carry wraps around (2^p ≡ 1 mod M_p).
    double carry = 0.0;
    double max_err = 0.0;
    for (size_t j = 0; j < n; ++j) {
        const double raw     = st.digits[j] + carry;
        const double rounded = std::round(raw);
        const double err     = std::abs(raw - rounded);
        if (err > max_err) max_err = err;

        const double modj = st.mod_pow[j];
        carry          = std::floor(rounded / modj);
        st.digits[j]   = rounded - carry * modj;
    }

    // Wrap-around: carry * 2^p ≡ carry (mod M_p); add to digit 0 and re-normalize.
    if (carry != 0.0) {
        double carry2 = carry;
        for (size_t j = 0; j < n && carry2 != 0.0; ++j) {
            const double raw_j     = st.digits[j] + carry2;
            const double rounded_j = std::round(raw_j);
            const double err_j     = std::abs(raw_j - rounded_j);
            if (err_j > max_err) max_err = err_j;
            const double modj = st.mod_pow[j];
            carry2         = std::floor(rounded_j / modj);
            st.digits[j]   = rounded_j - carry2 * modj;
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

}  // namespace mersenne

// ============================================================
// main()
// ============================================================
#ifndef BIGNUM_NO_MAIN

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

    const unsigned maxCores = runtime::detect_available_cores();

    size_t   startIndex    = 0u;
    unsigned threads       = maxCores;
    bool     progress      = false;
    bool     benchmark_mode = true;  // known-prime list → skip primality check

    if (argc >= 2 && argv[1] && argv[1][0] != '\0')
        startIndex = static_cast<size_t>(std::strtoull(argv[1], nullptr, 10));
    if (argc >= 3 && argv[2] && argv[2][0] != '\0') {
        const unsigned requested =
            static_cast<unsigned>(std::strtoull(argv[2], nullptr, 10));
        threads = (requested == 0u) ? maxCores : std::min(requested, maxCores);
    }
    if (argc >= 4) progress = true;

    const bool stopAfterOne = [] {
        const char* s = std::getenv("LL_STOP_AFTER_ONE");
        return s && *s != '\0' && std::strtoull(s, nullptr, 10) != 0ull;
    }();

    const auto& exponents = mersenne::known_mersenne_prime_exponents();
    if (startIndex >= exponents.size()) {
        std::fprintf(stderr, "Invalid start index: %zu (max %zu)\n",
                     startIndex, exponents.size() - 1u);
        return 1;
    }

    std::printf("Using %u worker(s) out of %u available core(s).\n",
                threads, maxCores);

    if (stopAfterOne) {
        const uint32_t p = exponents[startIndex];
        return test_exponent(p, progress, benchmark_mode) ? 0 : 2;
    }

    if (threads == 1u) {
        for (size_t idx = startIndex; idx < exponents.size(); ++idx)
            test_exponent(exponents[idx], progress, benchmark_mode);
        return 0;
    }

    // Thread-pool throughput mode: distribute exponents across workers.
    std::atomic<size_t> next{startIndex};
    std::mutex          printMu;

    runtime::ThreadPool pool(threads);

    for (unsigned t = 0; t < threads; ++t) {
        pool.submit([&] {
            for (;;) {
                const size_t idx =
                    next.fetch_add(1u, std::memory_order_relaxed);
                if (idx >= exponents.size()) break;
                const uint32_t p       = exponents[idx];
                const auto     t0      = std::chrono::steady_clock::now();
                const bool     isPrime =
                    mersenne::lucas_lehmer(p, progress, benchmark_mode);
                const auto t1 = std::chrono::steady_clock::now();
                const std::chrono::duration<double> elapsed = t1 - t0;
                std::lock_guard<std::mutex> lk(printMu);
                std::printf("Testing M_%u ...\n", p);
                std::printf("M_%u is %s. Time: %.3f s\n",
                            p, isPrime ? "prime" : "composite", elapsed.count());
            }
        });
    }

    pool.wait_all();
    return 0;
}
#endif
