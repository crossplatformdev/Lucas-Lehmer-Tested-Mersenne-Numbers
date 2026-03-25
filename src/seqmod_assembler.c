/* seqmod_assembler.c  –  Mersenne primality via a(n) = (2+√3)^(2^n)
 *
 * Criterion:
 *   Mod[ Ceil[ (2+√3)^(2^n) ], 2^(n+2)−1 ] = 0  ⟺  2^(n+2)−1  is prime
 *
 * Mathematical equivalence:
 *   Ceil[ (2+√3)^(2^k) ] = S_k  where  S_0=4, S_k = S_{k-1}²−2.
 *   (2-√3)^(2^k) ∈ (0,1) for all k ≥ 0, so Ceil[a(k)] = S_k exactly.)
 *   The criterion is therefore the Lucas–Lehmer test for M_p = 2^p−1:
 *     S_{p-2} ≡ 0 (mod M_p)  ⟺  M_p prime.
 *
 * Design (hot-path engineering):
 *   • Big integers: fixed-width uint64_t limb arrays (little-endian).
 *     Always exactly nlimbs = ⌈p/64⌉ limbs; no leading-zero trimming needed.
 *   • Comba squaring: n²/2 fused 128-bit MACs, symmetric cross-terms doubled.
 *   • Mersenne reduction: fold upper p bits into lower p bits – no division.
 *     After one fold, at most one more carry propagation brings the result
 *     into [0, 2^p−1].  2^p−1 is then normalised to 0.
 *   • Fused subtract-2: handles M_p–1 and M_p–2 edge cases without branching
 *     into a slow path except for the two rarest values (s ∈ {0, 1}).
 *   • Fast native path for p ≤ 62: single __uint128_t, no heap allocation.
 *   • All scratch buffers pre-allocated once per exponent (posix_memalign,
 *     64-byte aligned) and reused across every iteration – zero allocation
 *     inside the LL loop.
 *   • Persistent thread pool: N worker threads created at start-up and
 *     reused across all batches; no per-batch thread creation/destruction.
 *   • Progress output is sparse (every 64 completions or every 10 s).
 *
 * Build (release):
 *   gcc -O3 -march=native -mtune=native -std=c11 -pthread \
 *       -o bin/seqmod_assembler src/seqmod_assembler.c -lm
 *
 * Build (gprof profiling):
 *   gcc -O2 -march=native -std=c11 -pthread -pg \
 *       -o bin/seqmod_assembler_prof src/seqmod_assembler.c -lm
 *
 * Usage:
 *   seqmod_assembler [prime_count [start_n [threads]]]
 *     prime_count  – number of prime exponents to test (default: 512)
 *     start_n      – first candidate (default: 2)
 *     threads      – worker threads (default: 4; 0 = hardware concurrency)
 *
 * Environment:
 *   SEQMOD_OUTPUT_CSV       – write "n,is_prime" CSV to this path
 *   SEQMOD_TIME_LIMIT_SECS  – soft stop after N seconds (exit 42)
 *   SEQMOD_ASM_THREADS      – override threads argument
 *
 * Exit codes:  0 = done,  42 = soft-stop,  1 = error
 */

#define _GNU_SOURCE
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdbool.h>
#include <math.h>
#include <time.h>
#include <pthread.h>
#include <unistd.h>

/* ─── Exit codes ─────────────────────────────────────────────────────────── */
#define EXIT_DONE     0
#define EXIT_TIMEOUT  42
#define EXIT_ERROR    1

/* ─── Default configuration ──────────────────────────────────────────────── */
#define DEFAULT_COUNT    512
#define DEFAULT_START_N  2
#define DEFAULT_THREADS  4
#define MAX_THREADS      256

/* ─── Compiler hints ─────────────────────────────────────────────────────── */
#if defined(__GNUC__) || defined(__clang__)
#  define LIKELY(x)       __builtin_expect(!!(x), 1)
#  define UNLIKELY(x)     __builtin_expect(!!(x), 0)
#  define FORCE_INLINE    __attribute__((always_inline)) static inline
#  define NO_INLINE       __attribute__((noinline))
#else
#  define LIKELY(x)       (x)
#  define UNLIKELY(x)     (x)
#  define FORCE_INLINE    static inline
#  define NO_INLINE
#endif

/* ─── Timing ─────────────────────────────────────────────────────────────── */
FORCE_INLINE double clock_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* ─── Sieve of Eratosthenes ──────────────────────────────────────────────── */
/* Returns a malloc'd array of the first `want` primes ≥ start_n.
   *out_count is set to the actual count (≤ want). */
static int *sieve_primes(int start_n, int want, int *out_count) {
    if (want <= 0 || start_n < 2) { *out_count = 0; return NULL; }

    /* Upper bound for the (start_n_index + want)-th prime.
       Rosser: p_n < n*(ln n + ln ln n + 1.1) for n >= 6.               */
    int offset = 0;                         /* extra to account for start_n */
    if (start_n > 2) {
        /* rough number of primes < start_n by prime counting approx */
        double lns = log((double)start_n);
        offset = (int)(1.1 * (double)start_n / lns) + 100;
    }
    int N = want + offset + 100;
    double lnN = log((double)N + 2.0);
    int limit = (int)((double)N * (lnN + log(lnN) + 1.1)) + 1000;
    if (limit < 1000) limit = 1000;

    char *comp = (char *)calloc((size_t)(limit + 1), 1);
    if (!comp) { perror("calloc"); exit(EXIT_ERROR); }
    comp[0] = comp[1] = 1;
    for (int i = 2; (long long)i * i <= limit; i++)
        if (!comp[i])
            for (int j = i * i; j <= limit; j += i)
                comp[j] = 1;

    /* Count primes ≥ start_n */
    int cnt = 0;
    for (int i = (start_n < 2 ? 2 : start_n); i <= limit && cnt < want; i++)
        if (!comp[i]) cnt++;

    int *arr = (int *)malloc((size_t)cnt * sizeof(int));
    if (!arr) { perror("malloc"); free(comp); exit(EXIT_ERROR); }
    int idx = 0;
    for (int i = (start_n < 2 ? 2 : start_n); i <= limit && idx < cnt; i++)
        if (!comp[i]) arr[idx++] = i;

    free(comp);
    *out_count = idx;
    return arr;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Big-integer arithmetic for Lucas–Lehmer hot path
 * ═══════════════════════════════════════════════════════════════════════════
 * Representation: exactly nlimbs = ⌈p/64⌉ uint64_t limbs (little-endian).
 * Invariant: value ∈ [0, 2^p−2].  2^p−1 (= 0 mod M_p) is normalised to 0. */

#define LIMBS_FOR(p)  (((uint32_t)(p) + 63u) / 64u)

/* Mask for the top limb of a p-bit number. */
FORCE_INLINE uint64_t top_mask(uint32_t p) {
    uint32_t r = p & 63u;
    return r ? ((UINT64_C(1) << r) - 1u) : UINT64_MAX;
}

/* ─── comba_square ───────────────────────────────────────────────────────── */
/*
 * Compute sq[0..2n-1] = s[0..n-1]^2 using column-wise 3-accumulator Comba.
 *
 * This is fundamentally faster than scatter-accumulate Comba because:
 *   • sq[] is written exactly once per limb (sequential, cache-friendly).
 *   • No inner carry-propagation loop (scatter-based Comba needs "while carry").
 *   • The three 64-bit accumulators (w2,w1,w0) stay in registers.
 *
 * Algorithm:
 *   Maintain a 192-bit accumulator (w2,w1,w0) that carries across columns.
 *   For each output column k = 0 … 2n−1:
 *     • Add 2·s[i]·s[k−i]  for each cross-pair  (i < k−i, both indices < n)
 *     • Add   s[k/2]^2      for the diagonal term (k even, k/2 < n)
 *     • Write w0 → sq[k]
 *     • Rotate: w0←w1, w1←w2, w2←0   (carry forward)
 *
 * The accumulator never overflows 192 bits: each column has at most n/2
 * cross-terms and one diagonal; each contributes ≤ 129 bits; with n ≤ ~5000
 * limbs the column sum fits comfortably in 192 bits.
 */

/* Add 2·(a·b) to (w2,w1,w0).  a,b are uint64_t; w2/w1/w0 are the accumulators. */
#define ACC2(a, b)  do {                                            \
    __uint128_t _p  = (__uint128_t)(a) * (b);                      \
    uint64_t _plo   = (uint64_t)_p,  _phi = (uint64_t)(_p >> 64); \
    uint64_t _dlo   = _plo << 1;                                    \
    uint64_t _dhi   = (_phi << 1) | (_plo >> 63);                  \
    uint64_t _dov   = _phi >> 63;          /* 0 or 1 */            \
    w0 += _dlo;  uint64_t _c0 = (w0 < _dlo);                      \
    w1 += _dhi;  uint64_t _c1 = (w1 < _dhi);                      \
    w1 += _c0;           _c1 += (w1 < _c0);                        \
    w2 += _dov + _c1;                                               \
} while (0)

/* Add 1·(a·b) to (w2,w1,w0). */
#define ACC1(a, b)  do {                                            \
    __uint128_t _p  = (__uint128_t)(a) * (b);                      \
    uint64_t _plo   = (uint64_t)_p,  _phi = (uint64_t)(_p >> 64); \
    w0 += _plo;  uint64_t _c0 = (w0 < _plo);                      \
    w1 += _phi;  uint64_t _c1 = (w1 < _phi);                      \
    w1 += _c0;           _c1 += (w1 < _c0);                        \
    w2 += _c1;                                                      \
} while (0)

NO_INLINE static void comba_square(const uint64_t * restrict s, uint32_t n,
                                   uint64_t * restrict sq) {
    uint64_t w0 = 0, w1 = 0, w2 = 0;

    for (uint32_t k = 0; k < 2u * n; k++) {
        /* Range of valid i: [i0, i_end], where sq[k] gets contributions
           from pairs (i, k−i) with 0 ≤ i ≤ k−i < n.                   */
        const uint32_t i0    = (k + 1u > n) ? (k + 1u - n) : 0u;
        const uint32_t i_end = (k < n)      ? k             : (n - 1u);
        /* Cross-term range: i < k−i  ⟺  i < k/2 (C integer division). */
        const uint32_t cross = k >> 1u;   /* exclusive upper for cross terms */

        /* Cross terms: i = i0 … min(cross−1, i_end)
         * Condition: i < k−i  ⟺  2i < k  (use i+i < k to avoid rounding). */
        for (uint32_t i = i0; i + i < k; i++)
            ACC2(s[i], s[k - i]);

        /* Diagonal: k even, k/2 in [i0, i_end] */
        if (!(k & 1u) && cross >= i0 && cross <= i_end)
            ACC1(s[cross], s[cross]);

        sq[k] = w0;
        w0 = w1;  w1 = w2;  w2 = 0;
    }
}

#undef ACC2
#undef ACC1

/* ─── mersenne_reduce ────────────────────────────────────────────────────── */
/* Compute s[0..n-1] = sq[0..2n-1] mod (2^p − 1).
 *
 * Method: lo = sq mod 2^p,  hi = sq >> p.
 *   lo + hi ≡ sq  (mod 2^p−1)  because  2^p ≡ 1.
 *   lo + hi < 2·M_p < 2^(p+1), so one further fold suffices.
 * After folding, if s == 2^p−1 (≡ 0), normalise to 0.              */
NO_INLINE static void mersenne_reduce(uint64_t * restrict s,
                                      const uint64_t * restrict sq,
                                      uint32_t n, uint32_t p) {
    const uint32_t r     = p & 63u;
    const uint64_t tmask = r ? ((UINT64_C(1) << r) - 1u) : UINT64_MAX;
    __uint128_t carry = 0;

    if (r == 0) {
        /* p is a multiple of 64: lo = sq[0..n-1], hi = sq[n..2n-1]. */
        for (uint32_t i = 0; i < n; i++) {
            carry += (__uint128_t)sq[i] + sq[n + i];
            s[i]   = (uint64_t)carry;
            carry >>= 64;
        }
    } else {
        /* General case: hi = sq >> p.
         * hi[i] = (sq[n-1+i] >> r) | (sq[n+i] << (64-r))  for i = 0..n-1.
         * lo[i] = sq[i]  (i < n-1),  sq[n-1] & tmask  (i = n-1).       */
        const uint32_t shr = r, shl = 64u - r;
        for (uint32_t i = 0; i < n; i++) {
            uint64_t lo = (i < n - 1u) ? sq[i] : (sq[i] & tmask);
            uint64_t hi = (sq[n - 1u + i] >> shr) | (sq[n + i] << shl);
            carry += (__uint128_t)lo + hi;
            s[i]   = (uint64_t)carry;
            carry >>= 64;
        }
        /* s[n-1] may contain bits above position r from hi's contribution.
         * Extract them and fold (2^p ≡ 1 mod M_p).                       */
        carry += s[n-1] >> r;
        s[n-1] &= tmask;
    }

    /* Propagate remaining carry (at most a few bits; loop terminates fast). */
    while (carry) {
        __uint128_t c = carry;
        carry = 0;
        for (uint32_t i = 0; i < n && c; i++) {
            c    += s[i];
            s[i]  = (uint64_t)c;
            c   >>= 64;
        }
        if (r) {
            carry += s[n-1] >> r;
            s[n-1] &= tmask;
        } else {
            carry += (uint64_t)c;
        }
    }

    /* Normalise 2^p−1 → 0. */
    if (s[n-1] == tmask) {
        bool all_ff = true;
        for (uint32_t i = 0; i < n - 1u && all_ff; i++)
            all_ff = (s[i] == UINT64_MAX);
        if (all_ff) memset(s, 0, (size_t)n * sizeof(uint64_t));
    }
}

/* ─── sub2_mod_mersenne ──────────────────────────────────────────────────── */
/* Compute s = (s − 2) mod (2^p − 1) in place.
 * Precondition: s ∈ [0, 2^p−2] (already reduced by mersenne_reduce).
 * The fast path handles s[0] ≥ 2 with a single subtract (no branch into
 * the slow path for the vast majority of LL iterations).               */
FORCE_INLINE void sub2_mod_mersenne(uint64_t * restrict s,
                                    uint32_t n, uint32_t p) {
    /* Fast path: no borrow needed from limb 1 onwards. */
    if (LIKELY(s[0] >= 2u)) {
        s[0] -= 2u;
        return;
    }

    /* Slow path: subtract 2 with borrow propagation. */
    uint64_t borrow = 2u;
    for (uint32_t i = 0; i < n && borrow; i++) {
        uint64_t prev = s[i];
        s[i]   = prev - borrow;
        borrow = (s[i] > prev) ? 1u : 0u;
    }
    if (!borrow) return;

    /* Underflow: s < 2 as a big integer.  Correct by adding M_p = 2^p−1. */
    const uint32_t r     = p & 63u;
    const uint64_t tmask = r ? ((UINT64_C(1) << r) - 1u) : UINT64_MAX;
    __uint128_t acc = 0;
    for (uint32_t i = 0; i < n; i++) {
        uint64_t mp = (i < n - 1u) ? UINT64_MAX : tmask;
        acc  += (__uint128_t)s[i] + mp;
        s[i]  = (uint64_t)acc;
        acc >>= 64;
    }
    /* The carry (acc) represents 2^(64n) which vanishes mod 2^(64n). */
    if (r) s[n-1] &= tmask;
}

/* ─── LL workspace (pre-allocated per exponent) ──────────────────────────── */
typedef struct {
    uint64_t *s;      /* n limbs: current LL value              */
    uint64_t *sq;     /* 2n limbs: squaring scratch             */
    uint32_t  n;      /* ceil(p/64)                             */
    uint32_t  p;      /* prime exponent                         */
} LLCtx;

static LLCtx llctx_alloc(uint32_t p) {
    LLCtx ctx;
    ctx.p = p;
    ctx.n = LIMBS_FOR(p);
    size_t szs  = ((size_t)ctx.n * sizeof(uint64_t) + 63u) & ~(size_t)63u;
    size_t szsq = szs * 2u;
    if (posix_memalign((void **)&ctx.s,  64, szs)  != 0 ||
        posix_memalign((void **)&ctx.sq, 64, szsq) != 0) {
        perror("posix_memalign"); exit(EXIT_ERROR);
    }
    return ctx;
}

static void llctx_free(LLCtx *ctx) {
    free(ctx->s);  free(ctx->sq);
    ctx->s = ctx->sq = NULL;
}

/* ─── Native fast path (p ≤ 62) ─────────────────────────────────────────── */
static bool ll_native(uint32_t p) {
    if (p == 2u) return true;
    uint64_t mp = (UINT64_C(1) << p) - 1u;
    uint64_t s  = 4u;
    for (uint32_t k = 0; k < p - 2u; k++) {
        __uint128_t sq = (__uint128_t)s * s;
        uint64_t lo = (uint64_t)(sq & mp);
        uint64_t hi = (uint64_t)(sq >> p);
        s = lo + hi;
        if (s >= mp) s -= mp;
        if (s < 2u) s += mp;
        s -= 2u;
        if (s >= mp) s -= mp;
    }
    return s == 0u;
}

/* ─── General Lucas–Lehmer test ──────────────────────────────────────────── */
/* Returns true if M_p = 2^p − 1 is prime. */
static bool lucas_lehmer(uint32_t p) {
    if (p == 2u) return true;
    if (p <= 62u) return ll_native(p);

    LLCtx ctx = llctx_alloc(p);

    /* S_0 = 4 */
    memset(ctx.s, 0, (size_t)ctx.n * sizeof(uint64_t));
    ctx.s[0] = 4u;

    /* S_k = S_{k-1}^2 − 2  (mod 2^p−1),  k = 1 … p−2 */
    for (uint32_t k = 0; k < p - 2u; k++) {
        comba_square(ctx.s, ctx.n, ctx.sq);
        mersenne_reduce(ctx.s, ctx.sq, ctx.n, p);
        sub2_mod_mersenne(ctx.s, ctx.n, p);
    }

    /* S_{p-2} ≡ 0  ⟺  M_p prime */
    bool prime = true;
    for (uint32_t i = 0; i < ctx.n; i++)
        if (ctx.s[i]) { prime = false; break; }

    llctx_free(&ctx);
    return prime;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Persistent thread pool (zero thread creation in the hot path)
 * ═══════════════════════════════════════════════════════════════════════════
 * Design:
 *   • Each worker has its own mutex + two condition variables (start/done).
 *   • Main fills all slots, signals start for each, then waits for done.
 *   • Workers loop indefinitely until signalled to stop (p == 0).         */

typedef struct {
    pthread_mutex_t mu;
    pthread_cond_t  cv_start;   /* main → worker: work is ready */
    pthread_cond_t  cv_done;    /* worker → main: result is ready */

    /* Work item (written by main, read by worker): */
    uint32_t  p;          /* exponent; 0 = shut down */
    /* Result (written by worker, read by main): */
    bool      result;
    double    elapsed;
    bool      ready;      /* true once result is written */
    bool      has_work;   /* true once main has posted work */
} WorkSlot;

static void *worker_thread(void *arg) {
    WorkSlot *sl = (WorkSlot *)arg;
    for (;;) {
        pthread_mutex_lock(&sl->mu);
        while (!sl->has_work)
            pthread_cond_wait(&sl->cv_start, &sl->mu);
        sl->has_work = false;
        uint32_t p = sl->p;
        pthread_mutex_unlock(&sl->mu);

        if (p == 0u) break;   /* shutdown sentinel */

        double t0 = clock_seconds();
        bool   ok = lucas_lehmer(p);
        double el = clock_seconds() - t0;

        pthread_mutex_lock(&sl->mu);
        sl->result  = ok;
        sl->elapsed = el;
        sl->ready   = true;
        pthread_cond_signal(&sl->cv_done);
        pthread_mutex_unlock(&sl->mu);
    }
    return NULL;
}

typedef struct {
    WorkSlot  *slots;
    pthread_t *threads;
    int        n;
} ThreadPool;

static ThreadPool pool_create(int n) {
    ThreadPool tp;
    tp.n       = n;
    tp.slots   = (WorkSlot *)calloc((size_t)n, sizeof(WorkSlot));
    tp.threads = (pthread_t *)malloc((size_t)n * sizeof(pthread_t));
    if (!tp.slots || !tp.threads) { perror("malloc"); exit(EXIT_ERROR); }

    for (int i = 0; i < n; i++) {
        WorkSlot *sl = &tp.slots[i];
        pthread_mutex_init(&sl->mu, NULL);
        pthread_cond_init(&sl->cv_start, NULL);
        pthread_cond_init(&sl->cv_done, NULL);
        sl->has_work = false;
        sl->ready    = false;
        pthread_create(&tp.threads[i], NULL, worker_thread, sl);
    }
    return tp;
}

static void pool_destroy(ThreadPool *tp) {
    /* Signal all workers to shut down. */
    for (int i = 0; i < tp->n; i++) {
        WorkSlot *sl = &tp->slots[i];
        pthread_mutex_lock(&sl->mu);
        sl->p        = 0u;
        sl->has_work = true;
        pthread_cond_signal(&sl->cv_start);
        pthread_mutex_unlock(&sl->mu);
    }
    for (int i = 0; i < tp->n; i++)
        pthread_join(tp->threads[i], NULL);
    for (int i = 0; i < tp->n; i++) {
        pthread_mutex_destroy(&tp->slots[i].mu);
        pthread_cond_destroy(&tp->slots[i].cv_start);
        pthread_cond_destroy(&tp->slots[i].cv_done);
    }
    free(tp->slots);
    free(tp->threads);
    tp->slots = NULL; tp->threads = NULL;
}

/* Dispatch exponent p to slot i (non-blocking). */
static void pool_dispatch(ThreadPool *tp, int slot, uint32_t p) {
    WorkSlot *sl = &tp->slots[slot];
    pthread_mutex_lock(&sl->mu);
    sl->p        = p;
    sl->ready    = false;
    sl->has_work = true;
    pthread_cond_signal(&sl->cv_start);
    pthread_mutex_unlock(&sl->mu);
}

/* Wait for slot i to complete; fill *result and *elapsed. */
static void pool_collect(ThreadPool *tp, int slot, bool *result, double *elapsed) {
    WorkSlot *sl = &tp->slots[slot];
    pthread_mutex_lock(&sl->mu);
    while (!sl->ready)
        pthread_cond_wait(&sl->cv_done, &sl->mu);
    *result  = sl->result;
    *elapsed = sl->elapsed;
    pthread_mutex_unlock(&sl->mu);
}

/* ─── CSV writer ─────────────────────────────────────────────────────────── */
typedef struct {
    FILE *f;
} CSV;

static CSV csv_open(const char *path) {
    CSV c = {NULL};
    if (!path || !*path) return c;
    c.f = fopen(path, "w");
    if (!c.f) { perror(path); return c; }
    fprintf(c.f, "n,is_prime\n");
    return c;
}

static void csv_write(CSV *c, int n, bool is_prime) {
    if (c->f)
        fprintf(c->f, "%d,%s\n", n, is_prime ? "true" : "false");
}

static void csv_close(CSV *c) {
    if (c->f) { fclose(c->f); c->f = NULL; }
}

/* ─── main ───────────────────────────────────────────────────────────────── */
int main(int argc, char *argv[]) {

    /* ── Parse arguments ─────────────────────────────────────────────────── */
    int prime_count = DEFAULT_COUNT;
    int start_n     = DEFAULT_START_N;
    int nthreads    = DEFAULT_THREADS;

    if (argc > 1) prime_count = atoi(argv[1]);
    if (argc > 2) start_n     = atoi(argv[2]);
    if (argc > 3) nthreads    = atoi(argv[3]);

    /* Environment overrides */
    {
        const char *e;
        e = getenv("SEQMOD_ASM_THREADS");
        if (e && *e) nthreads = atoi(e);
    }

    /* Clamp */
    if (prime_count < 1)   prime_count = 1;
    if (start_n     < 2)   start_n     = 2;
    if (nthreads    < 1)   nthreads    = 1;
    if (nthreads    > MAX_THREADS) nthreads = MAX_THREADS;
    /* 0 → hardware concurrency */
    if (nthreads == 0) {
        long nc = sysconf(_SC_NPROCESSORS_ONLN);
        nthreads = (nc > 0) ? (int)nc : DEFAULT_THREADS;
    }

    /* ── Soft-stop timer ─────────────────────────────────────────────────── */
    double time_limit = 0.0;
    {
        const char *e = getenv("SEQMOD_TIME_LIMIT_SECS");
        if (e && *e) time_limit = atof(e);
    }

    /* ── CSV output ──────────────────────────────────────────────────────── */
    const char *csv_path = getenv("SEQMOD_OUTPUT_CSV");
    CSV csv = csv_open(csv_path);

    /* ── Prime list ──────────────────────────────────────────────────────── */
    int  nprimes = 0;
    int *primes  = sieve_primes(start_n, prime_count, &nprimes);

    fprintf(stderr,
            "seqmod_assembler v1.0.0: "
            "prime_count=%d  start_n=%d  threads=%d%s\n",
            nprimes, start_n, nthreads,
            time_limit > 0.0 ? "  (time-limited)" : "");

    /* ── Thread pool ─────────────────────────────────────────────────────── */
    ThreadPool tp = pool_create(nthreads);

    /* ── Main dispatch loop ──────────────────────────────────────────────── */
    double  t_start      = clock_seconds();
    double  t_last_prog  = t_start;
    int     hits         = 0;
    int     done         = 0;
    bool    timed_out    = false;

    /* Slots in-flight: [flight_base, flight_top) */
    int flight_base = 0;   /* next slot to collect    */
    int flight_top  = 0;   /* next slot to dispatch   */
    int next_prime  = 0;   /* index into primes[]     */

    /* Pre-fill the pipeline with up to nthreads items. */
    for (int s = 0; s < nthreads && next_prime < nprimes; s++) {
        pool_dispatch(&tp, s % nthreads, (uint32_t)primes[next_prime]);
        next_prime++;
        flight_top++;
    }

    while (flight_base < flight_top) {
        int slot = flight_base % nthreads;
        bool   result;
        double elapsed;
        pool_collect(&tp, slot, &result, &elapsed);
        flight_base++;
        done++;

        int p = primes[flight_base - 1];
        if (result) hits++;
        csv_write(&csv, p, result);

        /* Optional progress */
        double now = clock_seconds();
        if (done == 1 || done % 64 == 0 || now - t_last_prog >= 10.0) {
            double wall = now - t_start;
            double pct  = (nprimes > 0) ? 100.0 * done / nprimes : 0.0;
            fprintf(stderr,
                    "  [%6.1f s] %d/%d (%.1f%%)  primes_found=%d  "
                    "last_p=%d  last=%.3f s\n",
                    wall, done, nprimes, pct, hits, p, elapsed);
            t_last_prog = now;
        }

        /* Soft-stop check */
        if (time_limit > 0.0 && (now - t_start) >= time_limit) {
            timed_out = true;
            break;
        }

        /* Dispatch the next exponent (if any) to the slot just freed. */
        if (next_prime < nprimes) {
            pool_dispatch(&tp, slot, (uint32_t)primes[next_prime]);
            next_prime++;
            flight_top++;
        }
    }

    /* ── Drain in-flight work on soft-stop ───────────────────────────────── */
    while (flight_base < flight_top) {
        int slot = flight_base % nthreads;
        bool   result;
        double elapsed;
        pool_collect(&tp, slot, &result, &elapsed);
        flight_base++;
        (void)result; (void)elapsed;
    }

    pool_destroy(&tp);
    csv_close(&csv);
    free(primes);

    double total = clock_seconds() - t_start;
    fprintf(stderr,
            "seqmod_assembler: done=%d  mersenne_primes=%d  "
            "wall=%.3f s  avg=%.3f ms/prime\n",
            done, hits, total,
            done > 0 ? 1000.0 * total / done : 0.0);

    if (timed_out) {
        fprintf(stdout, "SEQMOD_TIMED_OUT=1\n");
        return EXIT_TIMEOUT;
    }
    return EXIT_DONE;
}
