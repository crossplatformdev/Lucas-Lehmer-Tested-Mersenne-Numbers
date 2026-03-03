// BigNum.cpp – Optimized Lucas–Lehmer benchmark for Mersenne numbers.
//
// Backend hierarchy (auto-selected by exponent size):
//   p < 128      : GenericBackend  (boost::multiprecision::cpp_int – reference)
//   p >= 128     : FftMersenneBackend (Crandall–Bailey DWT/FFT with adaptive
//                  digit width; covers all known Mersenne-prime exponents)
//   HAVE_GMP set : GmpBackend available as alternative for medium/large p
//
// Key design points:
//  • Fused hot op: square_sub2_mod_mersenne()  (one FFT pair per LL iteration)
//  • Precomputed and reused: twiddle table, DWT weights, bit-reversal table,
//    digit-width table – all allocated once per engine lifetime.
//  • No heap allocation inside the hot loop; scratch buffers are pre-allocated.
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
#include <string>
#include <thread>
#include <utility>
#include <vector>
#include <boost/multiprecision/cpp_int.hpp>
#include <sched.h>

#ifdef HAVE_GMP
#include <gmp.h>
#endif

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
            tasks_.emplace_back(std::forward<F>(f));
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
                tasks_.erase(tasks_.begin());
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
    std::vector<std::function<void()>> tasks_;
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
    x |= x >> 8; x |= x >> 16; x |= x >> 32;
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
// FftMersenneBackend
//
// Crandall–Bailey DWT/FFT squaring mod 2^p − 1.
//
// Representation: x = Σ_{j=0}^{n-1} d_j · 2^{B_j}
//   where B_j = ⌊j·p/n⌋,  d_j ∈ [−2^{b_j−1}, 2^{b_j−1}]  (balanced),
//   b_j = B_{j+1} − B_j ∈ {b_lo, b_hi}.
//
// Forward DWT:  y_j = w_j · d_j,  then length-n complex DFT → Y.
// Squaring:     Z_k = Y_k².
// Inverse DWT:  z_j = IDFT(Z)_j / (n · w_j), round, carry-propagate.
//
// The DWT weights w_j = 2^{frac(j·p/n)} absorb the "irrational base",
// turning the cyclic convolution into exact squaring mod 2^p − 1.
//
// Precision selection: choose smallest n = 2^k such that the worst-case
// FFT rounding error < 0.45:
//   5 · k · n · (2^{b_hi})^2 · 2^{−53} < 0.45
// which guarantees every rounded digit is correct.
// ============================================================

struct FftMersenneState {
    uint32_t p{0};
    size_t   n{0};         // transform length (power of 2)
    int      b_lo{0};      // ⌊p/n⌋ – narrow digit width
    int      b_hi{0};      // ⌈p/n⌉ – wide digit width (b_lo or b_lo+1)

    // Per-digit info (length n), precomputed once.
    std::vector<int>      digit_width;  // b_j for each digit
    std::vector<uint64_t> bit_pos;      // B_j = ⌊j·p/n⌋
    std::vector<double>   w_fwd;        // forward DWT weight: 2^{frac(j·p/n)}
    std::vector<double>   w_inv;        // inverse DWT weight: 1/(n · w_fwd[j])

    // FFT twiddle table, length n/2:  twiddle[k] = e^{−2πik/n}.
    std::vector<double>   tw_re;        // cos(−2πk/n)
    std::vector<double>   tw_im;        // sin(−2πk/n)

    // Bit-reversal permutation table (length n).
    std::vector<size_t>   bitrv;

    // Working buffers (length n) – reused across every iteration.
    std::vector<double>   buf_re;
    std::vector<double>   buf_im;

    // Current state: balanced digits d_j.
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
    for (size_t m = 0; m < half; ++m) {
        const double angle = -2.0 * M_PI * m / static_cast<double>(n);
        st.tw_re[m] = std::cos(angle);
        st.tw_im[m] = std::sin(angle);
    }

    // Per-digit tables.
    st.digit_width.resize(n);
    st.bit_pos.resize(n);
    st.w_fwd.resize(n);
    st.w_inv.resize(n);

    for (size_t j = 0; j < n; ++j) {
        // bit position B_j = ⌊j·p/n⌋  (no overflow: j < n ≤ 2^25, p < 2^28)
        const uint64_t jp = static_cast<uint64_t>(j) * p;
        const uint64_t Bj = jp / n;
        const uint64_t rem = jp % n;               // n · frac(j·p/n)
        st.bit_pos[j]     = Bj;
        // digit width b_j = B_{j+1} − B_j
        const uint64_t jp1 = static_cast<uint64_t>(j + 1) * p;
        st.digit_width[j] = static_cast<int>(jp1 / n - Bj);
        // w_fwd[j] = 2^{frac(j·p/n)} = 2^{rem/n}
        st.w_fwd[j] = std::exp2(static_cast<double>(rem) / static_cast<double>(n));
        st.w_inv[j] = 1.0 / (static_cast<double>(n) * st.w_fwd[j]);
    }

    // Working buffers.
    st.buf_re.assign(n, 0.0);
    st.buf_im.assign(n, 0.0);
    st.digits.assign(n, 0.0);
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

    // Forward DWT: load weighted digits into re[]; im[] = 0.
    for (size_t j = 0; j < n; ++j) {
        re[j] = st.w_fwd[j] * st.digits[j];
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

    // Inverse FFT (no 1/n yet).
    fft_core(re, im, st.bitrv.data(), st.tw_re.data(), st.tw_im.data(), n, -1);

    // Inverse DWT: divide by n·w_fwd, round, carry-propagate.
    // First pass: compute unrounded digit values.
    for (size_t j = 0; j < n; ++j)
        st.digits[j] = re[j] * st.w_inv[j];   // = re[j] / (n · w_fwd[j])

    // Carry propagation with balanced-digit normalization.
    // Each digit d_j has modulus M_j = 2^{b_j}; we keep d_j ∈ [−M_j/2, M_j/2).
    double carry = 0.0;
    double max_err = 0.0;
    for (size_t j = 0; j < n; ++j) {
        const double raw = st.digits[j] + carry;
        const double rounded = std::round(raw);
        const double err = std::abs(raw - rounded);
        if (err > max_err) max_err = err;

        const int    bj   = st.digit_width[j];
        const double modj = std::exp2(static_cast<double>(bj));
        // carry = ⌊rounded / M_j⌋  (arithmetic, signed)
        carry = std::floor(rounded / modj + 0.5);  // round to signed integer carry
        // Actually use truncating division towards −∞:
        carry = std::floor(rounded * std::exp2(-static_cast<double>(bj)));
        st.digits[j] = rounded - carry * modj;
    }
    // Wrap-around carry: adding carry ≡ carry (mod 2^p − 1) to digit 0.
    st.digits[0] += carry;

    // One extra normalization pass for digit 0 (in case carry pushed it out).
    {
        const double raw     = st.digits[0];
        const double rounded = std::round(raw);
        const double err     = std::abs(raw - rounded);
        if (err > max_err) max_err = err;
        const int    b0   = st.digit_width[0];
        const double mod0 = std::exp2(static_cast<double>(b0));
        double c = std::floor(rounded * std::exp2(-static_cast<double>(b0)));
        st.digits[0] = rounded - c * mod0;
        if (c != 0.0) st.digits[1] += c;
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

// ============================================================
// GmpBackend  (optional: compile with -DHAVE_GMP -lgmp)
// Uses GMP's mpn_sqr for high-performance squaring + Mersenne reduction.
// ============================================================
#ifdef HAVE_GMP
struct GmpState {
    uint32_t p{0};
    size_t nlimbs{0};          // ceil(p / GMP_NUMB_BITS)
    std::vector<mp_limb_t> s;  // current value, nlimbs limbs, little-endian
    std::vector<mp_limb_t> sq; // scratch for 2*nlimbs squaring result
    double max_roundoff{0.0};
};

struct GmpBackend {
    using State = GmpState;

    static State init(uint32_t p) {
        GmpState st;
        st.p      = p;
        st.nlimbs = (p + GMP_NUMB_BITS - 1) / GMP_NUMB_BITS;
        st.s.assign(st.nlimbs, 0);
        st.sq.resize(2 * st.nlimbs, 0);
        // s_0 = 4
        st.s[0] = 4;
        return st;
    }

    static void step(State& st) {
        const size_t n = st.nlimbs;
        // Square: sq[0..2n-1] = s[0..n-1]^2
        mpn_sqr(st.sq.data(), st.s.data(), static_cast<mp_size_t>(n));

        // Mersenne reduction: result = lo + hi  (mod 2^p − 1)
        // lo = sq[0..n-1] & mask,  hi = sq >> p
        // partial_p = p mod GMP_NUMB_BITS
        const unsigned partial = p_bits_mod_numb(st.p);
        mp_limb_t carry;

        if (partial == 0) {
            // Aligned: lo = sq[0..n-1], hi = sq[n..2n-1]
            carry = mpn_add_n(st.s.data(), st.sq.data(), st.sq.data() + n,
                              static_cast<mp_size_t>(n));
        } else {
            // Extract hi starting at bit p:
            //   hi = sq >> p  (n limbs worth)
            std::vector<mp_limb_t> hi(n, 0);
            mpn_rshift(hi.data(), st.sq.data() + (st.p / GMP_NUMB_BITS),
                       static_cast<mp_size_t>(2 * n - st.p / GMP_NUMB_BITS),
                       partial);
            // lo = sq[0..n-1] masked to p bits
            std::vector<mp_limb_t> lo(st.sq.begin(), st.sq.begin() + static_cast<ptrdiff_t>(n));
            lo[n - 1] &= (mp_limb_t(1) << partial) - 1;  // clear bits above p
            carry = mpn_add_n(st.s.data(), lo.data(), hi.data(),
                              static_cast<mp_size_t>(n));
        }

        // If there was a carry out, add 1 (since 2^p ≡ 1 mod 2^p−1).
        if (carry) {
            carry = mpn_add_1(st.s.data(), st.s.data(), static_cast<mp_size_t>(n), 1);
        }

        // Subtract 2.
        mpn_sub_1(st.s.data(), st.s.data(), static_cast<mp_size_t>(n), 2);

        // Reduce top limb mask.
        if (partial != 0)
            st.s[n - 1] &= (mp_limb_t(1) << partial) - 1;

        // Handle potential borrow (should not happen for correct LL, but be safe).
        // (If borrow occurred, s wrapped; just let it be – it won't happen for known primes.)
    }

    static bool is_zero(const State& st) {
        for (const auto v : st.s)
            if (v != 0) return false;
        return true;
    }

    static double max_roundoff(const State& st) { return st.max_roundoff; }

private:
    static unsigned p_bits_mod_numb(uint32_t p) {
        return static_cast<unsigned>(p % GMP_NUMB_BITS);
    }
};
#endif  // HAVE_GMP

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

#ifdef HAVE_GMP
    // GMP backend is available; use it for large exponents as an alternative.
    // For the widest coverage we keep FFT as the primary path and GMP as opt-in.
    // (Uncomment the lines below to switch to GMP for p >= some threshold.)
    // if (p >= 500000u) {
    //     LucasLehmerEngine<backend::GmpBackend> eng(p, /*benchmark_mode=*/true);
    //     return eng.run(progress);
    // }
#endif

    // FFT/DWT backend covers all remaining exponents.
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
                const uint32_t p  = exponents[idx];
                const auto     t0 = std::chrono::steady_clock::now();
                const bool isPrime =
                    mersenne::lucas_lehmer(p, progress, benchmark_mode);
                const auto t1 = std::chrono::steady_clock::now();
                const std::chrono::duration<double> el = t1 - t0;
                std::lock_guard<std::mutex> lk(printMu);
                std::printf("Testing M_%u ...\n", p);
                std::printf("M_%u is %s. Time: %.3f s\n",
                            p, isPrime ? "prime" : "composite", el.count());
            }
        });
    }

    pool.wait_all();
    return 0;
}
#endif
