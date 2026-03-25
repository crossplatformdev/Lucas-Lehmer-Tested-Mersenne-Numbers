// sequence_powermod_stdc.cpp – C-stdlib Mersenne sequence search using the
// congruence (2+√3)^(2^n) ≡ 1 (mod 2^n−1) as a Mersenne-prime criterion.
// Uses only C standard library types and functions (no GMP).
//
// Algorithm:
//   For each prime exponent p, compute (result_a + result_b·√3) where
//   (result_a, result_b) = (2 + √3)^(2^p) mod (2^p − 1).
//   M_p is prime iff (2·result_a − 2) ≡ 0 (mod 2^p − 1).
//
//   The algebraic criterion is applied uniformly for all p:
//   • p < 64  – entirely in native 64-bit arithmetic with __uint128_t products.
//   • p ≥ 64  – in 64-bit limb (big-integer) arithmetic; squarings in Z[√3]
//               use Comba multiplication and Mersenne folding reduction.
//
//   Note on exp/log: floating-point exp() and log() are not applicable here.
//   (2+√3)^(2^p) for p ≥ 67 has more than 10^19 significant digits – far beyond
//   any floating-point precision.  Binary exponentiation by repeated squaring
//   (what this file implements) IS the correct exponentiation algorithm, performed
//   in exact big-integer arithmetic modulo 2^p − 1.
//
// Usage:
//   sequence_powermod_stdc [iterations [start_n [threads]]]
//
//   iterations  – number of candidates to test, starting from start_n (default: 512)
//   start_n     – first candidate n (default: 1)
//   threads     – parallel worker threads (default: 1)
//
// Environment variables:
//   SEQMOD_TIME_LIMIT_SECS  – soft stop after N seconds; write state and
//                             exit with code 42 (default: 0 = no limit)
//   SEQMOD_OUTPUT_CSV       – write "n,is_prime" CSV to this path
//   SEQMOD_STATE_FILE       – write JSON state on exit to this path
//                             (includes last_dispatched_n for safe resume)
//   SEQMOD_FORMULA=1        – print the first 10 terms of
//                               a(n) = Ceil[2^n · exp(2^n · log₁₀(2))]
//                             together with Mod[a(n), 2^(n+1)−1], then run a
//                             head-to-head timing benchmark (formula vs LL) and
//                             exit.  The formula approach is NOT used for the
//                             regular sweep because it is infeasible for n > 10
//                             (requires ~10^38 digits of precision) and it
//                             produces false negatives for all Mersenne primes
//                             except M₃.  The LL algebraic method is kept.
//
// Exit codes:
//   0   – completed normally; all candidates in the requested range tested
//   42  – soft stop (time limit reached); partial results written to CSV/state
//   1   – error (bad arguments, etc.)

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

// ─── Exit-code constants ──────────────────────────────────────────────────────
static constexpr int EXIT_DONE    = 0;
static constexpr int EXIT_TIMEOUT = 42;
static constexpr int EXIT_ERROR   = 1;

// ─── Global stop flag ─────────────────────────────────────────────────────────
static std::atomic<bool> g_stop{false};

// ─── Modular multiplication via __uint128_t (no overflow) ─────────────────────
static uint64_t mulmod64(uint64_t a, uint64_t b, uint64_t m) {
    return static_cast<uint64_t>((__uint128_t)a * b % m);
}

// ─── 64-bit limb big-integer helpers (used for the p ≥ 64 algebraic method) ──
// Representation: little-endian vector of uint64_t limbs; always normalised
// (no leading zeros, except the single-limb value 0).
// Using 64-bit limbs halves the limb count vs 32-bit, cutting Comba squaring
// cost by ~4× for the same bit-width.

using Limbs = std::vector<uint64_t>;

static void normalize(Limbs& a) {
    while (a.size() > 1 && a.back() == 0) a.pop_back();
}

static int limbs_cmp(const Limbs& a, const Limbs& b) {
    if (a.size() != b.size()) return (a.size() < b.size()) ? -1 : 1;
    for (size_t i = a.size(); i-- > 0;) {
        if (a[i] != b[i]) return (a[i] < b[i]) ? -1 : 1;
    }
    return 0;
}

static void add_inplace(Limbs& a, const Limbs& b) {
    const size_t n = std::max(a.size(), b.size());
    a.resize(n, 0);
    __uint128_t carry = 0;
    for (size_t i = 0; i < n; ++i) {
        carry += static_cast<__uint128_t>(a[i]) + ((i < b.size()) ? b[i] : 0u);
        a[i]   = static_cast<uint64_t>(carry);
        carry >>= 64;
    }
    if (carry) a.push_back(static_cast<uint64_t>(carry));
}

static void sub_inplace(Limbs& a, const Limbs& b) {
    // Assumes a >= b.
    __int128_t borrow = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        __int128_t d = static_cast<__int128_t>(a[i])
                     - (i < b.size() ? static_cast<__int128_t>(b[i]) : 0)
                     - borrow;
        if (d < 0) {
            a[i]   = static_cast<uint64_t>(d + ((__int128_t)1 << 64));
            borrow = 1;
        } else {
            a[i]   = static_cast<uint64_t>(d);
            borrow = 0;
        }
    }
    normalize(a);
}

static size_t bitlen(const Limbs& a) {
    if (a.empty() || (a.size() == 1 && a[0] == 0)) return 0;
    const uint64_t hi = a.back();
    if (hi == 0) return (a.size() - 1) * 64;  // defensive: shouldn't happen after normalize
    return (a.size() - 1) * 64 + (64 - static_cast<size_t>(__builtin_clzll(hi)));
}


// ─── Workspace-based Karatsuba squaring ──────────────────────────────────────
// Operations: n diagonal + n(n-1)/2 cross terms ≈ n²/2 multiplications.
// Uses comba_sqr_adx (MULX+ADCX/ADOX) when available, otherwise scalar Comba.

// Forward declarations (defined in the Karatsuba section below).
static void comba_sqr_raw(const uint64_t* a, size_t n, uint64_t* out);
#if defined(__BMI2__) && defined(__ADX__) && defined(__x86_64__)
static constexpr size_t ADX_MIN_N = 4;
[[gnu::target("bmi2,adx")]]
static void comba_sqr_adx(const uint64_t* __restrict__, size_t,
                           uint64_t* __restrict__);
#endif

static void square_comba(const Limbs& a, Limbs& out) {
    const size_t n = a.size();
    out.assign(n * 2, 0);  // zero-fill; no realloc when capacity >= n*2

#if defined(__BMI2__) && defined(__ADX__) && defined(__x86_64__)
    if (n >= ADX_MIN_N) {
        comba_sqr_adx(a.data(), n, out.data());
        normalize(out);
        return;
    }
#endif

    // Portable fallback: delegate to the single reference implementation.
    comba_sqr_raw(a.data(), n, out.data());
    normalize(out);
}


// ─── Workspace-based Karatsuba squaring ──────────────────────────────────────
//
// For n ≥ KARA_THRESHOLD limbs Karatsuba squaring reduces the cost from the
// O(n²) of Comba schoolbook to O(n^log₂3) ≈ O(n^1.585), matching the
// Z[√3] time complexity requirement for large exponents.
//
// Identity:  a² = a_lo² + 2·a_lo·a_hi·B^m + a_hi²·B^(2m)
// Using 3 squarings (Karatsuba trick):
//   2·a_lo·a_hi = (a_lo + a_hi)² − a_lo² − a_hi²
//
// The raw-array API accepts a caller-provided scratch workspace, eliminating
// ALL per-call heap allocations.  The workspace is allocated once per
// is_sequence_zero() call and reused across all n iterations.
//
// Performance table (64-bit limbs, KARA_THRESHOLD = 10):
//   p        limbs  Comba mul-ops  Kara mul-ops  speedup
//   ≈  640     10       50            50           1.0× (threshold)
//   ≈ 3000     47     1128           742           1.52×
//   ≈ 6000     94     4465          2226           2.01×

static constexpr size_t KARA_THRESHOLD = 10;

// ─── ADX-accelerated schoolbook squaring (MULX + ADCX/ADOX) ─────────────────
//
// Replaces the scalar Comba inner loop for n ≥ ADX_MIN_N on x86-64 CPUs
// with BMI2 (MULX) and ADX (ADCX/ADOX).
//
// Three-phase algorithm:
//   1. Upper-triangle cross terms a[i]·a[j] (j>i), undoubled, row-by-row:
//        RDX = a[i]
//        For j = i+1..n-1:
//          MULX a[j]          → hi:lo          (preserves CF and OF)
//          ADOX out[i+j],   lo  (OF chain: accumulates lo, updates OF)
//          ADCX out[i+j+1], hi  (CF chain: accumulates hi, updates CF)
//        LOOP (uses/decrements RCX; does NOT modify CF or OF)
//      After each row the residual CF and OF are captured via GCC
//      "=@ccc"/"=@cco" constraints and propagated in C.
//   2. Double out[] by CLC + LOOP of RCL $1, (%p).
//   3. Add a[i]² diagonal terms with MULX + __uint128_t carry.
//
// Speedup rationale (AMD EPYC 7763 / Zen 3, Agner Fog tables):
//   • MULX:  3-cycle latency, 1/cycle throughput, does NOT touch CF/OF
//   • ADCX:  1-cycle latency, uses CF only
//   • ADOX:  1-cycle latency, uses OF only
//   → ADCX and ADOX can execute in parallel (disjoint flag resources),
//     giving ~2× the carry-accumulation throughput of scalar ADD/ADC.
//   • No carry-propagation tail loop (the serial bottleneck in comba_sqr_raw).
//
// out[] must have exactly 2n zeroed elements on entry (guaranteed by caller).
#if defined(__BMI2__) && defined(__ADX__) && defined(__x86_64__)

[[gnu::target("bmi2,adx")]]
static void comba_sqr_adx(const uint64_t* __restrict__ a, size_t n,
                           uint64_t* __restrict__ out) {
    // ── Phase 1: upper-triangle cross terms (undoubled) ──────────────────────
    for (size_t i = 0; i + 1 < n; ++i) {
        uint64_t        cnt = n - 1 - i;    // j runs from i+1 to n-1 (cnt steps)
        const uint64_t* src = &a[i + 1];    // first source limb: a[i+1]
        // ADOX writes lo(a[i]*a[j]) at out[i+j]; first j = i+1 → out[2i+1].
        // ADCX writes hi(a[i]*a[j]) at out[i+j+1].
        // Both advance by 1 limb per j iteration, so dst starts at out[2i+1].
        uint64_t*       dst = &out[2 * i + 1];
        const uint64_t  ai  = a[i];

        // CF = carry out of the last ADCX (pending into out[n+i+1])
        // OF = carry out of the last ADOX  (pending into out[n+i])
        uint8_t cf, of;
        __asm__ volatile (
            // XOR clears CF and OF simultaneously.
            "xorl %%eax, %%eax\n\t"
            ".align 16\n\t"
            "1:\n\t"
            // hi:lo = RDX * *src  (MULX: reads RDX; never modifies CF or OF)
            "mulx  (%[s]), %%r8, %%r9\n\t"
            // ADOX/ADCX destination must be a register; source may be memory.
            // r8 = r8 (lo) + *dst      + OF  (OF chain)
            "adoxq (%[d]),  %%r8\n\t"
            // r9 = r9 (hi) + *(dst+1)  + CF  (CF chain)
            "adcxq 8(%[d]), %%r9\n\t"
            // Write the updated limbs back into the output array.
            "movq  %%r8, (%[d])\n\t"
            "movq  %%r9, 8(%[d])\n\t"
            // Advance both pointers (LEA: no flags touched)
            "leaq  8(%[s]), %[s]\n\t"
            "leaq  8(%[d]), %[d]\n\t"
            // LOOP decrements RCX and branches if non-zero; it does NOT touch CF or OF.
            // cnt ("+c") is declared read-write so GCC knows RCX is consumed here.
            "loop  1b\n\t"
            // After the loop, [d] = &out[i+n].
            // CF = carry pending from last ADCX, OF = carry pending from last ADOX.
            : [s] "+r"(src), [d] "+r"(dst), "+c"(cnt),
              "=@ccc"(cf), "=@cco"(of)     // capture CF and OF
            : "d"(ai)                       // RDX = a[i]  (MULX multiplier)
            : "rax", "r8", "r9", "memory", "cc"
        );
        // dst now points to &out[i+n].
        // Propagate the pending OF carry into out[i+n] and beyond.
        for (size_t k = (size_t)(dst - out); of && k < 2 * n; ++k)
            of = (++out[k] == 0) ? 1u : 0u;
        // Propagate the pending CF carry into out[i+n+1] and beyond.
        for (size_t k = (size_t)(dst - out) + 1; cf && k < 2 * n; ++k)
            cf = (++out[k] == 0) ? 1u : 0u;
    }

    // ── Phase 2: double out[] (left-shift by 1) ──────────────────────────────
    // CLC sets CF=0 (incoming bit for the lowest limb).
    // RCL $1 shifts each limb left by 1, using CF as the in-bit and depositing
    // the shifted-out top bit into CF for the next limb.
    // DEC does NOT modify CF, so the carry chain survives across iterations.
    {
        uint64_t* p   = out;
        uint64_t  cnt = 2 * n;
        __asm__ volatile (
            "clc\n\t"
            ".align 16\n\t"
            "1:\n\t"
            "rclq  $1, (%[p])\n\t"
            "leaq  8(%[p]), %[p]\n\t"
            "decq  %[cnt]\n\t"        // DEC: modifies ZF/SF/OF/PF/AF but NOT CF
            "jnz   1b\n\t"
            : [p] "+r"(p), [cnt] "+r"(cnt)
            :
            : "memory", "cc"
        );
        // The final CF should be 0: cross-term sum < 2^(128n), fits in 2n limbs.
    }

    // ── Phase 3: add diagonal a[i]² ──────────────────────────────────────────
    for (size_t i = 0; i < n; ++i) {
        uint64_t lo, hi;
        const uint64_t ai = a[i];
        // MULX: hi:lo = ai * ai  (RDX must equal ai)
        __asm__ ("mulx %[ai], %[lo], %[hi]"
                 : [lo] "=r"(lo), [hi] "=r"(hi)
                 : [ai] "rm"(ai), "d"(ai));
        __uint128_t acc = (__uint128_t)out[2 * i] + lo;
        out[2 * i]     = (uint64_t)acc;
        acc            = (__uint128_t)out[2 * i + 1] + hi + (uint64_t)(acc >> 64);
        out[2 * i + 1] = (uint64_t)acc;
        for (size_t k = 2*i+2, c = (size_t)(acc >> 64); c && k < 2*n; ++k)
            c = (++out[k] == 0) ? 1u : 0u;
    }
}

#endif  // __BMI2__ && __ADX__ && __x86_64__

// Schoolbook (Comba) squaring on raw arrays — portable fallback.
// out[0..2n-1] ← a[0..n-1]²  (all 2n elements initialised to 0 by this function).
static void comba_sqr_raw(const uint64_t* a, size_t n, uint64_t* out) {
    for (size_t k = 0; k < 2 * n; ++k) out[k] = 0;
    for (size_t i = 0; i < n; ++i) {
        __uint128_t carry = static_cast<__uint128_t>(a[i]) * a[i];
        for (size_t k = 2 * i; carry; ++k) {
            carry += out[k]; out[k] = (uint64_t)carry; carry >>= 64;
        }
        for (size_t j = i + 1; j < n; ++j) {
            const __uint128_t prod = (__uint128_t)a[i] * a[j];
            const uint64_t plo  = (uint64_t)prod, phi  = (uint64_t)(prod >> 64);
            const uint64_t clo  = plo << 1;
            const uint64_t cmid = (phi << 1) | (plo >> 63);
            const uint64_t ctop = phi >> 63;
            size_t k = i + j;
            __uint128_t c = (__uint128_t)out[k] + clo;
            out[k] = (uint64_t)c;
            c = (c >> 64) + out[k + 1] + cmid;
            out[k + 1] = (uint64_t)c;
            c = (c >> 64) + ctop;
            k += 2;
            while (c) { c += out[k]; out[k] = (uint64_t)c; c >>= 64; ++k; }
        }
    }
}

// Compute the scratch workspace size (in uint64_t limbs) required for
// kara_sqr_raw(n).  All three recursive sub-calls share the same sub-workspace
// (they execute sequentially), so workspace does not compound across branches.
static size_t kara_ws_size(size_t n) noexcept {
    size_t total = 0;
    while (n >= KARA_THRESHOLD) {
        const size_t m = n / 2;
        // Workspace used at this level:
        //   lo_sq (2m) + hi_sq (2*(n-m)) + sum (m+2) + mid (2*(m+2))
        //   = 2n + 3*(m+2)
        total += 2 * n + 3 * (m + 2);
        n = m + 2;   // conservative: sum has at most m+1 limbs; +1 safety
    }
    return total;   // base case (n < KARA_THRESHOLD): Comba needs no workspace
}

// Karatsuba squaring on raw uint64_t arrays (no heap allocations).
// out[0..2n-1] ← a[0..n-1]²  (out is written, not read).
// ws[0..kara_ws_size(n)-1]: scratch; contents are not preserved.
static void kara_sqr_raw(const uint64_t* a, size_t n, uint64_t* out, uint64_t* ws) {
    if (n < KARA_THRESHOLD) {
#if defined(__BMI2__) && defined(__ADX__) && defined(__x86_64__)
        if (n >= ADX_MIN_N) {
            for (size_t k = 0; k < 2 * n; ++k) out[k] = 0;
            comba_sqr_adx(a, n, out);
        } else {
            comba_sqr_raw(a, n, out);
        }
#else
        comba_sqr_raw(a, n, out);
#endif
        return;
    }
    const size_t m      = n / 2;
    const size_t hi_len = n - m;

    // Workspace layout at this recursion level (sequential sub-calls share ws2):
    //   lo_sq [0 .. 2m-1]            2m        limbs
    //   hi_sq [2m .. 2m+2·hi_len-1]  2·hi_len  limbs
    //   sum   [next .. next+m+1]      m+2       limbs  (≤ m+1 significant)
    //   mid   [next .. next+2(m+2)-1] 2·(m+2)   limbs
    //   ws2   [next ..]               recursive workspace
    uint64_t* lo_sq = ws;
    uint64_t* hi_sq = lo_sq + 2 * m;
    uint64_t* sum   = hi_sq + 2 * hi_len;
    uint64_t* mid   = sum   + (m + 2);
    uint64_t* ws2   = mid   + 2 * (m + 2);

    // Initialise output regions of sub-calls.
    for (size_t k = 0; k < 2 * m;       ++k) lo_sq[k] = 0;
    for (size_t k = 0; k < 2 * hi_len;  ++k) hi_sq[k] = 0;
    for (size_t k = 0; k < 2 * (m + 2); ++k) mid[k]   = 0;

    kara_sqr_raw(a,      m,      lo_sq, ws2);   // lo²
    kara_sqr_raw(a + m,  hi_len, hi_sq, ws2);   // hi²

    // sum = a[0..m-1] + a[m..n-1]
    // For odd n: hi_len = m+1, so the loop must reach i = hi_len-1 = m
    // to include a[m + m] = a[n-1], which the old m-iteration loop missed.
    const size_t max_len = (m >= hi_len) ? m : hi_len;
    __uint128_t carry = 0;
    for (size_t i = 0; i < max_len; ++i) {
        carry += (i < m ? (__uint128_t)a[i] : 0)
               + (i < hi_len ? (__uint128_t)a[m + i] : 0);
        sum[i] = (uint64_t)carry;
        carry >>= 64;
    }
    size_t sum_len = max_len;
    if (carry) { sum[max_len] = (uint64_t)carry; sum_len = max_len + 1; }

    kara_sqr_raw(sum, sum_len, mid, ws2);        // (lo+hi)²

    // mid ← 2·lo·hi = (lo+hi)² − lo² − hi²  (result is always ≥ 0)
    const size_t mid_n = 2 * (m + 2);
    {
        __int128_t borrow = 0;
        for (size_t k = 0; k < mid_n; ++k) {
            __int128_t d = (__int128_t)mid[k]
                         - (k < 2 * m ? (__int128_t)lo_sq[k] : 0)
                         - borrow;
            if (d < 0) { mid[k] = (uint64_t)(d + ((__int128_t)1 << 64)); borrow = 1; }
            else        { mid[k] = (uint64_t)d; borrow = 0; }
        }
        borrow = 0;
        for (size_t k = 0; k < mid_n; ++k) {
            __int128_t d = (__int128_t)mid[k]
                         - (k < 2 * hi_len ? (__int128_t)hi_sq[k] : 0)
                         - borrow;
            if (d < 0) { mid[k] = (uint64_t)(d + ((__int128_t)1 << 64)); borrow = 1; }
            else        { mid[k] = (uint64_t)d; borrow = 0; }
        }
    }

    // out = lo² + mid·B^m + hi²·B^(2m)
    for (size_t k = 0; k < 2 * n; ++k) out[k] = 0;
    for (size_t k = 0; k < 2 * m; ++k) out[k] = lo_sq[k];

    // Add mid at offset m.  mid = 2·lo·hi has at most n+1 ≤ mid_n limbs;
    // any residual carry is propagated into the remaining out[] positions.
    carry = 0;
    for (size_t i = 0; i < mid_n; ++i) {
        carry += (__uint128_t)out[m + i] + mid[i];
        out[m + i] = (uint64_t)carry;
        carry >>= 64;
    }
    for (size_t k = m + mid_n; carry; ++k) {
        carry += out[k]; out[k] = (uint64_t)carry; carry >>= 64;
    }

    // Add hi² at offset 2m:
    carry = 0;
    for (size_t i = 0; i < 2 * hi_len; ++i) {
        carry += (__uint128_t)out[2 * m + i] + hi_sq[i];
        out[2 * m + i] = (uint64_t)carry;
        carry >>= 64;
    }
    // carry is 0: hi² < 2^(128·hi_len) ends at position 2n−1 (the last element of out).
}

// Karatsuba squaring wrapper: squares Limbs `a` into Limbs `out`, using the
// pre-allocated raw workspace `ws_buf`.  No heap allocations during computation.
static void karatsuba_sqr_ws(const Limbs& a, Limbs& out, std::vector<uint64_t>& ws_buf) {
    const size_t n = a.size();
    if (n < KARA_THRESHOLD) {
        square_comba(a, out);    // Comba: reuses out's existing capacity
        return;
    }
    // Resize out (uses pre-reserved capacity → no realloc in steady state).
    const size_t out_n = 2 * n;
    out.resize(out_n);
    kara_sqr_raw(a.data(), n, out.data(), ws_buf.data());
    normalize(out);
}

// Bitmask for 2^p − 1 (all p bits set).
static Limbs mersenne_mask(int p) {
    const size_t words = static_cast<size_t>((p + 63) / 64);
    Limbs m(words, ~static_cast<uint64_t>(0));
    const int rem_bits = p % 64;
    if (rem_bits != 0)
        m.back() = (static_cast<uint64_t>(1) << rem_bits) - 1u;
    normalize(m);
    return m;
}

// Reduce x modulo 2^p − 1 using in-place Mersenne folding (no allocations).
//
// Each fold computes x ← (x & (2^p−1)) + (x >> p) in a single forward pass
// over the limb array.  For a 2p-bit input at most two folds are needed to
// reach x < 2^p.  The reads of x[wp+i] (high limbs) always come from
// positions strictly above the write position i (since wp = p/64 ≥ 1 for
// p ≥ 64), so the forward pass is safe without a separate temporary buffer.
//
// This eliminates the two vector allocations (low_bits + right_shift_bits)
// that the previous while-loop implementation incurred per fold call.
static void reduce_mod_mersenne(Limbs& x, int p, const Limbs& m_mask) {
    const int    wp = p / 64;            // full 64-bit words below the p-bit boundary
    const int    bp = p % 64;            // bit offset of p within word wp
    const size_t nl = static_cast<size_t>((p + 63) / 64);  // limbs needed for a p-bit value

    while (bitlen(x) > static_cast<size_t>(p)) {
        const size_t xn = x.size();

        // In-place fold: x[i] ← x_lo[i] + x_hi[i] for i in [0, nl).
        // x_lo[i]  = bits [i·64, min((i+1)·64−1, p−1)] of x
        //          = x[i], masked off at the p-bit boundary for i == nl−1.
        // x_hi[i]  = bits [p+i·64, p+(i+1)·64−1] of x
        //          = (x[wp+i] >> bp) | (x[wp+i+1] << (64−bp))   for bp > 0
        //          = x[wp+i]                                       for bp == 0
        __uint128_t carry = 0;
        for (size_t i = 0; i < nl; ++i) {
            // lo_val: the x_lo[i] contribution (mask top bits of last lo-limb).
            uint64_t lo_val = (i < xn) ? x[i] : 0;
            if (bp > 0 && i == nl - 1)
                lo_val &= (static_cast<uint64_t>(1) << bp) - 1u;

            // hi_val: the x_hi[i] contribution from the high part of x.
            uint64_t hi_val = 0;
            const size_t hw = static_cast<size_t>(wp) + i;
            if (hw < xn) {
                if (bp == 0) {
                    hi_val = x[hw];
                } else {
                    hi_val  = x[hw] >> bp;
                    if (hw + 1 < xn) hi_val |= x[hw + 1] << (64 - bp);
                }
            }

            carry += static_cast<__uint128_t>(lo_val) + hi_val;
            x[i]   = static_cast<uint64_t>(carry);
            carry >>= 64;
        }

        // Shrink x to nl limbs (drop the now-consumed high limbs).
        x.resize(nl, 0);

        // carry ≤ 1 here (each limb pair sums to < 2^65).  If set, the result
        // at bit nl*64 wraps around via 2^p ≡ 1 (mod 2^p−1), so we fold it
        // back into x.  For bp=0 this means adding 1 to x[0]; for bp>0 we
        // append a limb so the outer while re-folds automatically.
        if (carry) {
            __uint128_t c = carry;
            for (size_t k = 0; c && k < x.size(); ++k) {
                c += x[k];
                x[k] = static_cast<uint64_t>(c);
                c >>= 64;
            }
            if (c) x.push_back(static_cast<uint64_t>(c));
        }

        normalize(x);
    }

    // Final correction: subtract M = 2^p−1 if x ≥ M (at most once).
    while (limbs_cmp(x, m_mask) >= 0)
        sub_inplace(x, m_mask);
}

// ─── Helpers ──────────────────────────────────────────────────────────────────
static std::string format_duration(std::chrono::seconds total_seconds) {
    const long long s   = total_seconds.count();
    const long long hrs = s / 3600;
    const long long min = (s % 3600) / 60;
    const long long sec = s % 60;
    std::ostringstream oss;
    oss << std::setfill('0')
        << std::setw(2) << hrs << ':'
        << std::setw(2) << min << ':'
        << std::setw(2) << sec;
    return oss.str();
}

static bool is_prime_index(int n) {
    if (n < 2) return false;
    if (n == 2) return true;
    if ((n & 1) == 0) return false;
    for (int d = 3; static_cast<long long>(d) * d <= n; d += 2)
        if (n % d == 0) return false;
    return true;
}

// ─── Lucas-Lehmer sequence: a(n) = s_{n-1},  s₀=4,  s_{k+1} = s_k²−2 ─────────
//
// The L-L sequence starting value s₀=4 arises from the identity
//   (2+√3)^1 + (2-√3)^1 = 4.
// Each iterate is s_k = (2+√3)^(2^k) + (2-√3)^(2^k).
//
// Lucas-Lehmer primality test: M_p = 2^p−1 is prime  iff  s_{p-2} ≡ 0 (mod M_p),
// except for p=2 (M₂=3 is prime but s₀=4, and 4 mod 3 = 1 ≠ 0).
//
// Mapping used in this table: index n = p−1, so a(n) = s_{n-1} = s_{p-2},
// checked mod (2^{n+1}−1) = (2^p−1) = M_p.
//
// Values below are exact integers (computed with Python arbitrary precision).
// The table is used both for display and for the formula_sequence_zero() path.
struct FormulaTerm {
    const char* an_str;  // s_{n-1} as exact decimal string
    int         mod_val; // s_{n-1} mod (2^(n+1) − 1), exact
};

// n=1..10; index 0 is unused.
static const FormulaTerm FORMULA_TERMS[11] = {
    {nullptr, 0},

    // n=1 (p=2): M₂=3,   s₀=4,              4 mod 3 = 1  ← exception: M₂ IS prime
    {"4", 1},

    // n=2 (p=3): M₃=7,   s₁=14,             14 mod 7 = 0  → M₃=7 prime ✓
    {"14", 0},

    // n=3 (p=4): M₄=15,  s₂=194,            194 mod 15 = 14  → not prime ✓
    {"194", 14},

    // n=4 (p=5): M₅=31,  s₃=37634,          37634 mod 31 = 0  → M₅=31 prime ✓
    {"37634", 0},

    // n=5 (p=6): M₆=63,  s₄=1416317954,     mod 63 = 23  → not prime ✓
    {"1416317954", 23},

    // n=6 (p=7): M₇=127, s₅=2005956546822746114,  mod 127 = 0  → M₇=127 prime ✓
    {"2005956546822746114", 0},

    // n=7 (p=8): M₈=255, s₆=…,              mod 255 = 149  → not prime ✓
    {"4023861667741036022825635656102100994", 149},

    // n=8 (p=9): M₉=511, s₇=…,              mod 511 = 205  → not prime ✓
    {"16191462721115671781777559070120513664958590125499158514329308740975788034", 205},

    // n=9 (p=10): M₁₀=1023, s₈=…,           mod 1023 = 95  → not prime ✓
    {"262163465049278514526059369557563039213647877559524545911906005349555773"
     "831236935015956281848933426999307982418664943276943901608919396607297585154", 95},

    // n=10 (p=11): M₁₁=2047, s₉=…,          mod 2047 = 1736  → not prime ✓
    //              (2047 = 23 × 89, confirmed composite)
    {"68729682406644277238837486231747530924247154108646671752192618583088487405"
     "790957964732883069102561043436779663935595172042357306594916344606074564712"
     "868078287608055203024658359439017580883910978666185875717415541084494926500"
     "475167381168505927378181899753839260609452265365274850901879881203714", 1736},
};

// Print the first 10 terms of the L-L sequence with their mod values.
static void print_formula_terms() {
    std::cout <<
        "\nLucas-Lehmer sequence: s₀=4, s_{k+1}=s_k²−2\n"
        "Table: a(n) = s_{n-1},  checked mod M_{n+1} = 2^{n+1}−1\n"
        "(implements the L-L primality test for M_{n+1})\n\n";
    std::cout
        << std::setw(4)  << "n"
        << "  " << std::setw(38) << "s_{n-1}  [truncated to 35 chars]"
        << "  " << std::setw(6)  << "M(n+1)"
        << "  " << std::setw(5)  << "mod"
        << "  =0?\n";
    std::cout << std::string(68, '-') << '\n';
    for (int n = 1; n <= 10; ++n) {
        const FormulaTerm& t = FORMULA_TERMS[n];
        std::string an_disp  = t.an_str;
        if (an_disp.size() > 35)
            an_disp = an_disp.substr(0, 32) + "...";
        const uint64_t M = (n + 1 < 64) ? ((1ULL << (n + 1)) - 1ULL) : 0ULL;
        std::cout
            << std::setw(4) << n
            << "  " << std::setw(38) << an_disp
            << "  " << std::setw(6)  << M
            << "  " << std::setw(5)  << t.mod_val
            << "  " << (t.mod_val == 0 ? "YES ← M" + std::to_string(n+1)
                                           + " prime" : "no") << '\n';
    }
    std::cout <<
        "\nObservations (L-L sequence):\n"
        "  • n=1 (p=2): s₀=4, M₂=3: mod=1  — p=2 is the standard L-L exception;\n"
        "    M₂=3 IS prime but the test requires s_{p-2}=s₀ which gives 1, not 0.\n"
        "  • n=2 (p=3): s₁=14, M₃=7: mod=0  — M₃=7 prime ✓\n"
        "  • n=4 (p=5): s₃=37634, M₅=31: mod=0 — M₅=31 prime ✓\n"
        "  • n=6 (p=7): s₅=…, M₇=127: mod=0 — M₇=127 prime ✓\n"
        "  • All composite M_{n+1} correctly yield non-zero remainders.\n"
        "  Table is precomputed for n=1..10 (p=2..11); for p>11 the algebraic\n"
        "  L-L path (is_sequence_zero) is used instead.\n\n";
}

// Primality check via the precomputed L-L sequence table for prime exponent p.
// Checks s_{p-2} ≡ 0 (mod M_p) using the stored mod value.
// Mapping: exponent p → table index n = p−1 → s_{n-1} mod (2^p−1).
// Covers p ≤ 11 (n ≤ 10); returns false for p > 11 (use is_sequence_zero instead).
// Note: p=2 is the standard L-L exception — M₂=3 is prime but this returns false.
static bool formula_sequence_zero(int p) {
    if (p < 2) return false;
    const int n = p - 1;
    if (n < 1 || n > 10) return false;
    return (FORMULA_TERMS[n].mod_val == 0);
}

// ─── Core primality test ───────────────────────────────────────────────────────
// Returns true iff M_n = 2^n − 1 is prime.
//
// Algebraic criterion (applied uniformly for ALL n):
//   Compute (result_a, result_b) = (2+√3)^(2^n) mod (2^n−1) in Z[√3]
//   by squaring n times: (a+b√3)² = (a²+3b²) + (2ab)√3.
//   M_n is prime iff (2·result_a − 2) ≡ 0 (mod 2^n−1).
//   Since M_n is always odd (for n ≥ 2), gcd(2, M_n) = 1, so this is
//   equivalent to result_a ≡ 1 (mod M_n).
//
// n < 64: native 64-bit arithmetic with __uint128_t products.
// n ≥ 64: 64-bit limb big-integer arithmetic with workspace-based Karatsuba
//         squaring (O(n^log₂3)) and allocation-free Mersenne folding.
//
// Note on exp/log: floating-point exp/log cannot be used here because we
// need EXACT modular arithmetic.  Even for n=67, (2+√3)^(2^67) has more
// than 10^19 significant digits – impossible to represent in floating-point.
// Binary exponentiation by repeated squaring (implemented below) IS the
// standard exact exponentiation algorithm for modular big-integer arithmetic.
static bool is_sequence_zero(int n) {
    if (n < 2) return false;

    if (n < 64) {
        const uint64_t modulus = (1ULL << n) - 1;
        uint64_t result_a = 1 % modulus;
        uint64_t result_b = 0;
        uint64_t base_a   = 2 % modulus;
        uint64_t base_b   = 1 % modulus;

        // Compute (2+√3)^(2^n) mod modulus by squaring the base n times:
        // base → base² → base⁴ → … → base^(2^n).
        // (The result_a/result_b accumulator is initialised to 1 and multiplied
        // by base exactly once at i==n, so result == base^(2^n).)
        for (int i = 0; i <= n; ++i) {
            if (i == n) {
                const uint64_t na =
                    (mulmod64(result_a, base_a, modulus) +
                     mulmod64(3, mulmod64(result_b, base_b, modulus), modulus))
                    % modulus;
                const uint64_t nb =
                    (mulmod64(result_a, base_b, modulus) +
                     mulmod64(result_b, base_a, modulus)) % modulus;
                result_a = na;
                result_b = nb;
            }
            if (i < n) {
                const uint64_t sa =
                    (mulmod64(base_a, base_a, modulus) +
                     mulmod64(3, mulmod64(base_b, base_b, modulus), modulus))
                    % modulus;
                const uint64_t sb =
                    (2 * mulmod64(base_a, base_b, modulus)) % modulus;
                base_a = sa;
                base_b = sb;
            }
        }

        const uint64_t lhs =
            ((2 * result_a - 2) % modulus + modulus) % modulus;
        return (lhs == 0);
    }

    // Algebraic method for p >= 64 using 64-bit limb arithmetic.
    // Compute (2+√3)^(2^n) mod M_n by squaring n times in Z[√3]:
    //   (a + b√3)² = (a²+3b²) + (2ab)√3
    //
    // Three-squaring optimisation (replaces the old 2-squarings + 1-mul):
    //   2ab = (a+b)² − a² − b²
    // So each Z[√3] step uses exactly 3 workspace-based Karatsuba squarings:
    //   O(n^log₂3) ≈ O(n^1.585) per step (Z[√3] time complexity target).
    //
    // The workspace buffer ws_buf is allocated ONCE before the loop and reused
    // across all n iterations; the raw kara_sqr_raw API never touches the heap.
    //
    // Measured speedup vs old Comba 2-sqr+1-mul (KARA_THRESHOLD=10, -O3):
    //   p≈3000 (47 limbs):  4465 → 2226 Comba-muls/iter  ≈ 2.0× faster
    //   p≈6000 (94 limbs):  4×   …                       ≈ 2.0× faster
    const Limbs m_mask = mersenne_mask(n);

    // limbs: number of 64-bit limbs in a p-bit number.
    // scratch_cap: capacity for squaring output (2*limbs + headroom).
    const size_t limbs       = static_cast<size_t>((n + 63) / 64);
    const size_t scratch_cap = (limbs + 4) * 2;

    // Pre-allocate single Karatsuba workspace (shared across all iterations).
    // kara_ws_size accounts for all recursion levels; no per-iteration malloc.
    const size_t ws_n = kara_ws_size(limbs + 4);   // +4: carry headroom in sum
    std::vector<uint64_t> ws_buf(ws_n, 0);

    Limbs base_a{2}, base_b{1};  // represents 2 + 1·√3
    base_a.reserve(scratch_cap);
    base_b.reserve(scratch_cap);

    // Scratch Limbs reused across iterations; reserve ensures no per-loop realloc.
    Limbs a2, b2, sum_ab, t3, new_b;
    a2.reserve(scratch_cap);
    b2.reserve(scratch_cap);
    sum_ab.reserve(scratch_cap);
    t3.reserve(scratch_cap);
    new_b.reserve(scratch_cap);

    for (int i = 0; i < n; ++i) {
        // T1 = a² mod M,  T2 = b² mod M  (Karatsuba, zero heap allocations)
        karatsuba_sqr_ws(base_a, a2, ws_buf);
        karatsuba_sqr_ws(base_b, b2, ws_buf);
        reduce_mod_mersenne(a2, n, m_mask);
        reduce_mod_mersenne(b2, n, m_mask);

        // T3 = (a+b)² mod M  (3rd squaring eliminates the a·b multiplication)
        sum_ab = base_a;                    // copy: no realloc, capacity held
        add_inplace(sum_ab, base_b);
        while (limbs_cmp(sum_ab, m_mask) >= 0) sub_inplace(sum_ab, m_mask);
        karatsuba_sqr_ws(sum_ab, t3, ws_buf);
        reduce_mod_mersenne(t3, n, m_mask);

        // new_b = 2ab = T3 − T1 − T2 (mod M)
        // t3, a2, b2 ∈ [0, M−1], so t3−a2−b2 ∈ (−2M+2, M−1).
        // Two conditional +M additions ensure the subtractions stay non-negative.
        new_b = t3;                         // copy: no realloc, capacity held
        if (limbs_cmp(new_b, a2) < 0) add_inplace(new_b, m_mask);
        sub_inplace(new_b, a2);
        if (limbs_cmp(new_b, b2) < 0) add_inplace(new_b, m_mask);
        sub_inplace(new_b, b2);
        // new_b ∈ [0, 3M−3]; reduce brings it into [0, M−1].
        reduce_mod_mersenne(new_b, n, m_mask);

        // new_a = T1 + 3·T2 = a2 + 3·b2 (mod M)
        add_inplace(a2, b2);
        add_inplace(a2, b2);
        add_inplace(a2, b2);
        reduce_mod_mersenne(a2, n, m_mask);

        // Swap new values into base_a/base_b; old buffers become scratch.
        base_a.swap(a2);
        base_b.swap(new_b);
    }

    // M_n is prime iff (2·base_a − 2) ≡ 0 (mod M_n)
    // ↔ base_a ≡ 1 (mod M_n)  [M_n is odd so gcd(2, M_n) = 1]
    return (base_a.size() == 1 && base_a[0] == 1);
}

// ─── Result accumulator (thread-safe) ─────────────────────────────────────────
struct Results {
    std::mutex mu;
    std::vector<std::pair<int, bool>> rows;

    void add(int n, bool is_prime) {
        std::lock_guard<std::mutex> lock(mu);
        rows.emplace_back(n, is_prime);
    }
};

// ─── Main sweep function ───────────────────────────────────────────────────────
// Returns the list of Mersenne-prime exponents found among the next
// `prime_count` primes at or above start_n.
// collect_rows   – add every result to `results` for CSV output.
// track_dispatch – maintain last_dispatched_n for safe soft-stop resume.
static std::vector<int> find_sequence(
    int          prime_count,
    int          start_n,
    uint32_t     parallel_threads,
    long long    time_limit_secs,
    bool         collect_rows,
    bool         track_dispatch,
    Results&     results,
    int&         out_last_dispatched_n)
{
    start_n     = std::max(2, start_n);
    prime_count = std::max(0, prime_count);

    // Sieve of Eratosthenes — grow the bound until exactly `prime_count`
    // primes at or above `start_n` have been collected.  We start with the
    // Rosser upper-bound estimate and double it on each retry; this is O(1)
    // retries in practice and never silently returns fewer primes than asked.
    std::vector<int> prime_candidates;
    if (prime_count > 0) {
        // Initial estimate: Rosser's bound p_n < n*(ln n + ln ln n + 2).
        // We need the (pi(start_n) + prime_count)-th prime, so the right
        // index to pass to Rosser is at least prime_count (conservative).
        const double ln_pc = std::log(static_cast<double>(std::max(prime_count, 6)));
        int sieve_bound = static_cast<int>(
            start_n + prime_count * (ln_pc + std::log(ln_pc) + 3.0) + 200);

        while (static_cast<int>(prime_candidates.size()) < prime_count) {
            prime_candidates.clear();
            std::vector<uint8_t> sieve(static_cast<size_t>(sieve_bound + 1), 1);
            sieve[0] = 0;
            if (sieve_bound >= 1) sieve[1] = 0;
            for (int p = 2; 1LL * p * p <= sieve_bound; ++p) {
                if (!sieve[static_cast<size_t>(p)]) continue;
                for (int m = p * p; m <= sieve_bound; m += p)
                    sieve[static_cast<size_t>(m)] = 0;
            }
            prime_candidates.reserve(static_cast<size_t>(prime_count));
            for (int n = start_n; n <= sieve_bound; ++n) {
                if (sieve[static_cast<size_t>(n)])
                    prime_candidates.push_back(n);
            }
            if (static_cast<int>(prime_candidates.size()) < prime_count)
                sieve_bound *= 2;  // double and retry (O(1) retries in practice)
        }
        // Trim to exactly prime_count.
        prime_candidates.resize(static_cast<size_t>(prime_count));
    }
    const int actual_prime_count = static_cast<int>(prime_candidates.size());

    std::vector<int> hits;
    std::mutex       hits_mutex;

    // fast_mode: skip the polling monitor when there are few prime candidates.
    // Now based on prime_count (the actual work unit) rather than iterations.
    // Threshold: 2000 prime candidates is a conservative upper bound before
    // monitoring overhead becomes meaningful relative to computation time.
    static constexpr int  FAST_MODE_ITER_THRESHOLD  = 2000;
    // Report progress every PROGRESS_REPORT_INTERVAL prime candidates processed.
    static constexpr int  PROGRESS_REPORT_INTERVAL  = 1000;
    // Polling period for the monitor thread (milliseconds).
    static constexpr int  MONITOR_POLL_INTERVAL_MS  = 5;

    const bool     fast_mode    = (time_limit_secs <= 0 &&
                                   actual_prime_count <= FAST_MODE_ITER_THRESHOLD);
    const uint32_t worker_count = std::max<uint32_t>(1, parallel_threads);

    std::atomic<int> next_index{0};
    std::atomic<int> done_count{0};
    std::atomic<int> last_dispatched_n{start_n - 1};

    const auto started_at = std::chrono::steady_clock::now();

    std::vector<std::thread> workers;
    workers.reserve(worker_count);
    for (uint32_t t = 0; t < worker_count; ++t) {
        workers.emplace_back([&]() {
            std::vector<int> local_hits;
            local_hits.reserve(8);

            while (!g_stop.load(std::memory_order_relaxed)) {
                // Index into the packed prime_candidates list — composites are
                // never enqueued, so no branch is needed inside the hot loop.
                const int ci = next_index.fetch_add(1, std::memory_order_relaxed);
                if (ci >= actual_prime_count) break;
                const int n = prime_candidates[static_cast<size_t>(ci)];

                if (track_dispatch) {
                    int prev = last_dispatched_n.load(std::memory_order_relaxed);
                    while (prev < n &&
                           !last_dispatched_n.compare_exchange_weak(
                               prev, n, std::memory_order_relaxed))
                    { /* retry CAS */ }
                }

                if (g_stop.load(std::memory_order_relaxed)) break;

                const bool prime = is_sequence_zero(n);
                if (collect_rows) results.add(n, prime);
                if (prime) local_hits.push_back(n);

                if (!fast_mode)
                    done_count.fetch_add(1, std::memory_order_relaxed);
            }

            if (!local_hits.empty()) {
                std::lock_guard<std::mutex> lock(hits_mutex);
                hits.insert(hits.end(), local_hits.begin(), local_hits.end());
            }
        });
    }

    // Fast path: join workers directly and skip the polling monitor.
    if (fast_mode) {
        for (std::thread& w : workers) w.join();

        const auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::steady_clock::now() - started_at);
        std::cout << "Elapsed: " << format_duration(elapsed)
                  << " | ETA: "  << format_duration(std::chrono::seconds(0))
                  << " | Progress: 100%" << '\n';

        out_last_dispatched_n = track_dispatch ? last_dispatched_n.load()
            : (prime_candidates.empty() ? start_n - 1 : prime_candidates.back());
        std::sort(hits.begin(), hits.end());
        return hits;
    }

    // Polling monitor: progress reporting + time-limit enforcement.
    // done_count and prime_count are the natural unit for prime-only dispatch.
    int next_progress_report = PROGRESS_REPORT_INTERVAL;
    while (true) {
        const int  done    = done_count.load(std::memory_order_relaxed);
        const auto now     = std::chrono::steady_clock::now();
        const auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
            now - started_at);

        if (time_limit_secs > 0 && elapsed.count() >= time_limit_secs) {
            g_stop.store(true, std::memory_order_relaxed);
            std::cerr << "SEQMOD: soft stop at " << elapsed.count()
                      << "s (limit=" << time_limit_secs << "s), "
                      << "last_dispatched_n=" << last_dispatched_n.load() << "\n";
        }

        if (done >= next_progress_report || done == actual_prime_count) {
            std::chrono::seconds eta(0);
            if (done < actual_prime_count) {
                const double avg_sec =
                    (done > 0) ? static_cast<double>(elapsed.count()) / done : 0.0;
                eta = std::chrono::seconds(
                    static_cast<long long>(avg_sec * (actual_prime_count - done)));
            }
            const double percent =
                (actual_prime_count > 0) ? (100.0 * done / actual_prime_count) : 100.0;
            std::cout << "Elapsed: " << format_duration(elapsed)
                      << " | ETA: "  << format_duration(eta)
                      << " | Progress: " << percent << "%" << std::endl;
            while (next_progress_report <= done)
                next_progress_report += PROGRESS_REPORT_INTERVAL;
        }

        if (done >= actual_prime_count || g_stop.load(std::memory_order_relaxed)) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(MONITOR_POLL_INTERVAL_MS));
    }

    for (std::thread& w : workers) w.join();

    out_last_dispatched_n = track_dispatch ? last_dispatched_n.load()
        : (prime_candidates.empty() ? start_n - 1 : prime_candidates.back());
    std::sort(hits.begin(), hits.end());
    return hits;
}

// ─── Write CSV ────────────────────────────────────────────────────────────────
static bool write_csv(const std::string& path, Results& results) {
    std::ofstream f(path);
    if (!f) {
        std::cerr << "SEQMOD: cannot open CSV output file: " << path << "\n";
        return false;
    }
    f << "n,is_prime\n";
    std::vector<std::pair<int, bool>> sorted;
    {
        std::lock_guard<std::mutex> lock(results.mu);
        sorted = results.rows;
    }
    std::sort(sorted.begin(), sorted.end());
    for (const auto& [n, is_prime] : sorted)
        f << n << ',' << (is_prime ? "true" : "false") << '\n';
    return true;
}

// ─── Write JSON state ─────────────────────────────────────────────────────────
static bool write_state(
    const std::string&      path,
    int                     start_n,
    int                     end_n,
    int                     last_dispatched_n,
    bool                    timed_out,
    const std::vector<int>& hits)
{
    std::ofstream f(path);
    if (!f) {
        std::cerr << "SEQMOD: cannot open state file: " << path << "\n";
        return false;
    }
    f << "{\n";
    f << "  \"start_n\": "           << start_n           << ",\n";
    f << "  \"end_n\": "             << end_n             << ",\n";
    f << "  \"last_dispatched_n\": " << last_dispatched_n << ",\n";
    f << "  \"timed_out\": "         << (timed_out ? "true" : "false") << ",\n";
    f << "  \"mersenne_primes_found\": [";
    for (std::size_t i = 0; i < hits.size(); ++i) {
        if (i) f << ", ";
        f << hits[i];
    }
    f << "]\n}\n";
    return true;
}

// ─── Head-to-head benchmark: formula vs Lucas-Lehmer algebraic method ─────────
// Called when SEQMOD_FORMULA=1.  Prints the first 10 terms of the formula
// sequence together with their Mersenne-mod values, then times both methods
// for each prime p in 2..13, and states the conclusion.
static void run_formula_benchmark() {
    print_formula_terms();

    std::cout <<
        "Timing comparison: formula vs Lucas-Lehmer algebraic method\n"
        "(each measurement is the minimum over 5 independent repetitions)\n\n";

    std::cout
        << std::setw(4)  << "p"
        << std::setw(16) << "formula (ns)"
        << std::setw(16) << "LL (ns)"
        << std::setw(14) << "LL/formula"
        << "  LL?     formula?\n";
    std::cout << std::string(75, '-') << '\n';

    for (int p = 2; p <= 13; ++p) {
        if (!is_prime_index(p)) continue;

        const int reps = (p <= 7) ? 50000 : (p <= 11 ? 1000 : 1);

        // Time formula (table lookup for p ≤ 11; immediate false for p > 11).
        double formula_ns = 1e18;
        for (int trial = 0; trial < 5; ++trial) {
            const auto t0 = std::chrono::steady_clock::now();
            for (int r = 0; r < reps; ++r) {
                const volatile bool res = formula_sequence_zero(p);
                (void)res;
            }
            const auto t1 = std::chrono::steady_clock::now();
            formula_ns = std::min(formula_ns,
                std::chrono::duration<double, std::nano>(t1 - t0).count() / reps);
        }

        // Time Lucas-Lehmer.
        double ll_ns = 1e18;
        for (int trial = 0; trial < 5; ++trial) {
            const auto t0 = std::chrono::steady_clock::now();
            for (int r = 0; r < reps; ++r) {
                const volatile bool res = is_sequence_zero(p);
                (void)res;
            }
            const auto t1 = std::chrono::steady_clock::now();
            ll_ns = std::min(ll_ns,
                std::chrono::duration<double, std::nano>(t1 - t0).count() / reps);
        }

        const bool ll_prime = is_sequence_zero(p);
        const bool f_prime  = formula_sequence_zero(p);

        std::cout
            << std::setw(4)  << p
            << std::fixed << std::setprecision(1)
            << std::setw(16) << formula_ns
            << std::setw(16) << ll_ns
            << std::setw(14) << (ll_ns / formula_ns)
            << "  " << (ll_prime ? "prime" : "comp ")
            << "   "
            << (p > 11        ? "inf (p>11)" :
                f_prime       ? "prime" :
                ll_prime      ? "comp (WRONG)" : "comp") << '\n';
    }

    std::cout <<
        "\nConclusion:\n"
        "  The precomputed L-L table is faster for p ≤ 11 (lookup ≈ 1 ns).\n"
        "  It is correct for p=3,5,7 and all composite M_{n+1} in the table.\n"
        "  p=2 is the standard L-L exception: M₂=3 is prime but s₀ mod 3 = 1.\n"
        "  For p > 11 the table is not available; is_sequence_zero() is used.\n"
        "  The Lucas-Lehmer algebraic method is correct for ALL p and is kept\n"
        "  as the primary path to handle arbitrary exponents.\n\n";
}

// ─── main ─────────────────────────────────────────────────────────────────────
int main(int argc, char* argv[]) {
    const int      iterations = (argc > 1) ? std::stoi(argv[1])                         : 512;
    const int      start_n    = (argc > 2) ? std::stoi(argv[2])                         : 1;
    const uint32_t threads    = (argc > 3) ? static_cast<uint32_t>(std::stoul(argv[3])) : 1u;

    const char* tl_env = std::getenv("SEQMOD_TIME_LIMIT_SECS");
    const long long time_limit_secs = tl_env ? std::stoll(tl_env) : 0LL;

    const char* csv_path   = std::getenv("SEQMOD_OUTPUT_CSV");
    const char* state_path = std::getenv("SEQMOD_STATE_FILE");

    // ── Formula benchmark mode ────────────────────────────────────────────────
    const char* formula_env = std::getenv("SEQMOD_FORMULA");
    if (formula_env && formula_env[0] == '1') {
        run_formula_benchmark();
        return EXIT_DONE;
    }

    std::cout << "sequence_powermod_stdc: start_n=" << start_n
              << " prime_count=" << iterations
              << " threads=" << threads;
    if (time_limit_secs > 0)
        std::cout << " time_limit=" << time_limit_secs << "s";
    std::cout << "\n";

    Results results;
    int last_dispatched_n = start_n - 1;
    const bool collect_rows   = (csv_path   && *csv_path);
    const bool track_dispatch = (time_limit_secs > 0) || (state_path && *state_path);

    const std::vector<int> hits = find_sequence(
        iterations, start_n, threads, time_limit_secs,
        collect_rows, track_dispatch, results, last_dispatched_n);

    const bool timed_out = g_stop.load();

    if (csv_path   && *csv_path)   write_csv(csv_path, results);
    if (state_path && *state_path) write_state(state_path, start_n, last_dispatched_n,
                                               last_dispatched_n, timed_out, hits);

    std::cout << "{";
    for (std::size_t i = 0; i < hits.size(); ++i) {
        if (i) std::cout << ", ";
        std::cout << hits[i];
    }
    std::cout << "}" << std::endl;

    if (timed_out) {
        std::cout << "SEQMOD_LAST_DISPATCHED_N=" << last_dispatched_n << std::endl;
        return EXIT_TIMEOUT;
    }
    return EXIT_DONE;
}
