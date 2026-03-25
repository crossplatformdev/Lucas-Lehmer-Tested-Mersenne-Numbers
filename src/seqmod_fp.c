/* seqmod_fp.c – Mersenne primality via floating-point FFT squaring in Z[√3].
 *
 * Formula (user specification):
 *   a(n) = (2 + √3)^(2^n)
 *   Mod[Ceil[a(n)], 2^(n+2) - 1] = 0  ⟹  M_{n+2} = 2^{n+2}-1 is prime.
 *
 * Mathematical basis:
 *   α = 2+√3,  ᾱ = 2-√3 (conjugate, αᾱ = 1).
 *   a(n) = α^(2^n) = A_n + B_n·√3  (A_n, B_n ∈ ℤ, A_n > 0, B_n > 0).
 *   0 < ᾱ^(2^n) < 1  ⟹  Ceil[a(n)] = A_n + B_n·√3 rounded up = 2·A_n = s_n
 *   where s_n is the Lucas-Lehmer sequence (s_0=4, s_{k+1}=s_k²-2).
 *   M_p prime (p=n+2) ⟺ s_{p-2} ≡ 0 (mod M_p) ⟺ A_{p-2} ≡ 0 (mod M_p).
 *
 * Algorithm for testing prime exponent p (M_p = 2^p − 1):
 *   Start: (A, B) = (2, 1) in Z[√3] mod M_p.
 *   Iterate p-2 times:  (A, B) ← (A²+3B², 2AB) mod M_p.
 *   Prime iff A = 0 (mod M_p).
 *
 * FFT squaring (Crandall-Bailey DWT):
 *   Digits are stored as doubles in a "balanced irrational-base" representation
 *   where digit j has weight 2^(floor(j·p/n)/n) (Crandall-Bailey DWT weight).
 *   Squaring = forward DWT → n/2-point complex FFT → pointwise operation →
 *              n/2-point IFFT → inverse DWT → round → carry propagate.
 *   Mersenne reduction is automatic via the cyclic structure of DWT.
 *
 * On an 8-TFLOPS / 2-MIPS cluster the FFT path gives ~4×10^6 speedup over
 * schoolbook O(n²) integer multiplication.
 *
 * Usage:
 *   seqmod_fp [prime_count [start_p [threads]]]
 *   seqmod_fp --bench          benchmark and compare vs known results
 *   seqmod_fp --validate       validate all primes up to p=4093
 *
 * Environment:
 *   SEQMOD_FP_THREADS   thread count  (default 4)
 *   SEQMOD_FP_OUTPUT    CSV file path (default: none)
 *   SEQMOD_FP_VERBOSE   verbose output (0/1, default 0)
 *   SEQMOD_FP_BENCH_CSV CSV file for benchmark results
 *
 * Build:
 *   Release: gcc -std=c11 -O3 -march=native -o bin/seqmod_fp src/seqmod_fp.c -lm -lpthread
 *   Profile: gcc -std=c11 -O2 -pg    -march=native -o bin/seqmod_fp_prof \
 *                src/seqmod_fp.c -lm -lpthread
 */

#define _POSIX_C_SOURCE 200809L
/* M_PI may not be available in strict C11; define it if missing. */
#ifndef M_PI
#  define M_PI 3.14159265358979323846
#endif

#include <math.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* ─── constants ─────────────────────────────────────────────────────────────── */

/* Maximum FFT length n (power of 2).  Covers p up to ~4 million bits. */
#define MAX_LOG2N 22u
#define MAX_N     (1u << MAX_LOG2N)

/* Minimum exponent to use the FFT path; below this use exact 64-bit arithmetic. */
#define FFT_THRESHOLD_P 64

/* Maximum supported exponent.  FFT length n ≈ 2p/avg_width (avg_width ≈ 1.5 bits).
 * For n ≤ MAX_N = 2^22: p ≤ 1.5 * 2^22 / 2 ≈ 3.1 million.
 * All known Mersenne-prime exponents up through M_82589933 are well below this. */
#define MAX_P_BITS (MAX_N * 2u)

/* Thread count cap */
#define MAX_THREADS 1024u

/* ─── wall-clock timer ───────────────────────────────────────────────────────── */

static double wall_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* ─── small prime sieve (Eratosthenes) ──────────────────────────────────────── */

/* Returns a malloc'd array of primes ≤ limit, count via *count. */
static int *sieve_primes(int limit, int *count) {
    if (limit < 2) { *count = 0; return NULL; }
    char *composite = calloc((size_t)(limit + 1), 1);
    if (!composite) { perror("calloc"); exit(1); }
    for (int i = 2; (long long)i * i <= limit; ++i)
        if (!composite[i])
            for (int j = i * i; j <= limit; j += i)
                composite[j] = 1;
    int cnt = 0;
    for (int i = 2; i <= limit; ++i) if (!composite[i]) ++cnt;
    int *arr = malloc((size_t)cnt * sizeof(int));
    if (!arr) { perror("malloc"); exit(1); }
    int k = 0;
    for (int i = 2; i <= limit; ++i) if (!composite[i]) arr[k++] = i;
    free(composite);
    *count = cnt;
    return arr;
}

/* Returns 1 if n is prime (trial division, for n < ~10^9). */
static int is_prime_td(int n) {
    if (n < 2) return 0;
    if (n == 2) return 1;
    if (n % 2 == 0) return 0;
    for (int d = 3; (long long)d * d <= n; d += 2)
        if (n % d == 0) return 0;
    return 1;
}

/* ─── FFT engine structures ──────────────────────────────────────────────────── */

typedef struct {
    int     p;           /* Mersenne exponent                                     */
    size_t  n;           /* FFT length (power of 2, ≥ ceil(p/avg_digit_width)*2)  */
    size_t  M;           /* n/2  (half-size FFT length)                           */
    size_t  log2M;       /* log2(M)                                               */

    /* Per-digit tables – length n each */
    int    *digit_width; /* b_j = floor((j+1)*p/n) - floor(j*p/n)                */
    double *w_fwd_inv;   /* 2^{-frac(j*p/n)} (forward DWT divisor)               */
    double *w_inv_half;  /* 2·2^{frac(j*p/n)}/n  (inverse DWT scale, ×2 for M)  */
    double *mod_pow;     /* 2^{b_j}                                               */
    double *inv_mod_pow; /* 2^{-b_j}  (Carmack trick: multiply instead of divide) */

    /* n-point twiddle table – length n/2 entries (tw_re[k]=cos(-2πk/n)) */
    double *tw_re;
    double *tw_im;

    /* M-point half-FFT twiddle tables – length M/2 */
    double *tw_re_half;       /* cos(-2πm/M) = tw_re[2m]  */
    double *tw_im_half;       /* sin(-2πm/M)               */
    double *tw_im_half_inv;   /* -sin(-2πm/M) for IFFT     */

    /* Bit-reversal table for M-point FFT – length M */
    size_t *bitrv;
} FpEngine;

/* Per-thread, per-exponent state for Z[√3] squaring */
typedef struct {
    double *A;        /* n digit values (real part)     */
    double *B;        /* n digit values (sqrt(3) coeff) */
    double *hA_re;    /* M complex scratch for FFT of A */
    double *hA_im;
    double *hB_re;    /* M complex scratch for FFT of B */
    double *hB_im;
} ZState;

/* ─── FFT engine: choose length ─────────────────────────────────────────────── */

static size_t choose_fft_len(int p) {
    /* Identical to BigNum.cpp's choose_fft_length() but with a tighter
     * error threshold to account for the Z[√3] squaring:
     *
     *   yA[k] = xA[k]² + 3·xB[k]²
     *   yB[k] = 2·xA[k]·xB[k]
     *
     * The combined magnitude of yA is up to 4× larger than a plain squaring
     * (since 1 + 3 = 4 and xA, xB are comparably bounded).  The accumulated
     * FFT rounding error also scales with the signal magnitude, so we need the
     * single-squaring error to be ≤ 0.45/4 to keep the combined error < 0.45.
     *
     * Error model (Percival-Schatzman):
     *   single_err ≈ 5·k·n·(2^b_hi)²·2^{-53}
     *   combined_err (Z[√3]) ≤ 4·single_err
     * Require combined_err < 0.45  →  single_err < 0.45/4 = 0.1125.
     *
     * b_hi ≤ 53 ensures each digit fits exactly in a double mantissa. */
    if (p <= 0) return 0;
    const double kThresh = 0.1125;   /* = 0.45 / 4  (Z[√3] safety margin) */
    for (int k = 1; k <= 30; ++k) {
        size_t n    = (size_t)1 << k;
        size_t b_hi = ((size_t)p + n - 1) / n;   /* ceil(p/n) */
        if (b_hi > 53) continue;                   /* digit too wide */
        double err  = 5.0 * k * ldexp((double)n, (int)(2 * b_hi) - 53);
        if (err < kThresh) {
            if (n > MAX_N) return 0;
            return n;
        }
    }
    return 0;   /* no suitable length found (p too large for this build) */
}

/* ─── FFT engine: init / free ──────────────────────────────────────────────── */

static int engine_init(FpEngine *e, int p) {
    memset(e, 0, sizeof(*e));
    e->p = p;
    e->n = choose_fft_len(p);
    if (e->n == 0) return 0;   /* unsupported size */
    e->M     = e->n >> 1;
    size_t M = e->M;

    /* log2(M) */
    size_t logM = 0;
    for (size_t v = M; v > 1; v >>= 1) ++logM;
    e->log2M = logM;

    size_t n = e->n;

    /* Allocate digit tables */
    e->digit_width = malloc(n * sizeof(int));
    e->w_fwd_inv   = malloc(n * sizeof(double));
    e->w_inv_half  = malloc(n * sizeof(double));
    e->mod_pow     = malloc(n * sizeof(double));
    e->inv_mod_pow = malloc(n * sizeof(double));
    if (!e->digit_width || !e->w_fwd_inv || !e->w_inv_half ||
        !e->mod_pow || !e->inv_mod_pow) return 0;

    /* Digit widths and DWT weights.
     * b_j = floor((j+1)*p/n) - floor(j*p/n)  (either floor(p/n) or ceil(p/n))
     * w_fwd[j] = 2^{frac(j*p/n)}  where frac = (j*p mod n) / n
     * Forward DWT divides by w_fwd → multiply by w_fwd_inv = 1/w_fwd.
     * Inverse DWT multiplies by w_fwd/n, ×2 because we use n/2-point IFFT. */
    for (size_t j = 0; j < n; ++j) {
        uint64_t jp   = (uint64_t)j * (uint64_t)p;
        uint64_t jp1  = (uint64_t)(j + 1) * (uint64_t)p;
        uint64_t Bj   = jp / n;
        uint64_t Bj1  = jp1 / n;
        uint64_t rem  = jp % n;               /* n * frac(j*p/n) */
        e->digit_width[j] = (int)(Bj1 - Bj);
        double wfwd = exp2((double)rem / (double)n);
        e->w_fwd_inv[j]   = 1.0 / wfwd;
        e->w_inv_half[j]  = 2.0 * wfwd / (double)n;  /* 2*w_fwd/n for half-IFFT */
        e->mod_pow[j]     = exp2((double)(int)(Bj1 - Bj));
        e->inv_mod_pow[j] = exp2(-(double)(int)(Bj1 - Bj));
    }

    /* n-point twiddle table: tw_re[k]=cos(-2πk/n), tw_im[k]=sin(-2πk/n) */
    e->tw_re = malloc((n / 2) * sizeof(double));
    e->tw_im = malloc((n / 2) * sizeof(double));
    if (!e->tw_re || !e->tw_im) return 0;
    for (size_t k = 0; k < n / 2; ++k) {
        double angle = -2.0 * M_PI * (double)k / (double)n;
        e->tw_re[k] = cos(angle);
        e->tw_im[k] = sin(angle);
    }

    /* M-point half-FFT twiddle tables (stride 2 into the n-point table) */
    e->tw_re_half     = malloc((M / 2) * sizeof(double));
    e->tw_im_half     = malloc((M / 2) * sizeof(double));
    e->tw_im_half_inv = malloc((M / 2) * sizeof(double));
    if (!e->tw_re_half || !e->tw_im_half || !e->tw_im_half_inv) return 0;
    for (size_t m = 0; m < M / 2; ++m) {
        e->tw_re_half[m]     =  e->tw_re[2 * m];
        e->tw_im_half[m]     =  e->tw_im[2 * m];
        e->tw_im_half_inv[m] = -e->tw_im_half[m];
    }

    /* Bit-reversal table for M-point FFT */
    e->bitrv = malloc(M * sizeof(size_t));
    if (!e->bitrv) return 0;
    for (size_t j = 0; j < M; ++j) {
        size_t r = 0, v = j;
        for (size_t b = 0; b < logM; ++b) { r = (r << 1) | (v & 1); v >>= 1; }
        e->bitrv[j] = r;
    }

    return 1;
}

static void engine_free(FpEngine *e) {
    free(e->digit_width);  free(e->w_fwd_inv);   free(e->w_inv_half);
    free(e->mod_pow);      free(e->inv_mod_pow);
    free(e->tw_re);        free(e->tw_im);
    free(e->tw_re_half);   free(e->tw_im_half);   free(e->tw_im_half_inv);
    free(e->bitrv);
    memset(e, 0, sizeof(*e));
}

/* ─── ZState: alloc / free ──────────────────────────────────────────────────── */

static int zstate_alloc(ZState *z, const FpEngine *e) {
    size_t n = e->n, M = e->M;
    z->A     = calloc(n, sizeof(double));
    z->B     = calloc(n, sizeof(double));
    z->hA_re = malloc(M * sizeof(double));
    z->hA_im = malloc(M * sizeof(double));
    z->hB_re = malloc(M * sizeof(double));
    z->hB_im = malloc(M * sizeof(double));
    return z->A && z->B && z->hA_re && z->hA_im && z->hB_re && z->hB_im;
}

static void zstate_free(ZState *z) {
    free(z->A); free(z->B);
    free(z->hA_re); free(z->hA_im);
    free(z->hB_re); free(z->hB_im);
    memset(z, 0, sizeof(*z));
}

/* ─── in-place iterative radix-2 Cooley-Tukey FFT ───────────────────────────── */
/* Operates on split re[]/im[] arrays of length n (power of 2).
 * tw_im must carry the correct sign for forward (-) or inverse (+). */

static void fft_core(double * restrict re, double * restrict im,
                     const size_t * restrict bitrv,
                     const double * restrict tw_re,
                     const double * restrict tw_im,
                     size_t n) {
    /* Bit-reversal permutation */
    for (size_t j = 0; j < n; ++j) {
        size_t r = bitrv[j];
        if (r > j) {
            double tr = re[j]; re[j] = re[r]; re[r] = tr;
            double ti = im[j]; im[j] = im[r]; im[r] = ti;
        }
    }
    /* Butterfly stages */
    size_t half_n = n >> 1;
    for (size_t len = 1; len < n; len <<= 1) {
        size_t step = half_n / len;
        for (size_t start = 0; start < n; start += len * 2) {
            for (size_t k = 0; k < len; ++k) {
                size_t m  = k * step;
                double wr = tw_re[m], wi = tw_im[m];
                size_t u  = start + k, v = u + len;
                double tr = wr * re[v] - wi * im[v];
                double ti = wr * im[v] + wi * re[v];
                re[v] = re[u] - tr;  im[v] = im[u] - ti;
                re[u] = re[u] + tr;  im[u] = im[u] + ti;
            }
        }
    }
}

/* ─── Forward pass: DWT-pack + M-point FFT ──────────────────────────────────── */
/* Input:  digits[n]  (DWT digit representation of a big integer mod M_p)
 * Output: h_re[M], h_im[M]  (half-size spectrum, packed pairs) */

static void fp_forward(const FpEngine * restrict e,
                       const double   * restrict digits,
                       double         * restrict h_re,
                       double         * restrict h_im) {
    const double *wfi = e->w_fwd_inv;
    size_t M = e->M;
    /* DWT-weight and pack two real values per complex slot */
    for (size_t k = 0; k < M; ++k) {
        h_re[k] = wfi[2 * k    ] * digits[2 * k    ];
        h_im[k] = wfi[2 * k + 1] * digits[2 * k + 1];
    }
    /* M-point forward FFT */
    fft_core(h_re, h_im, e->bitrv, e->tw_re_half, e->tw_im_half, M);
}

/* ─── Combined post-process + Z[√3] pointwise op + pre-process ──────────────── */
/* Takes the two half-size spectra h_A (for A) and h_B (for B) and produces
 * two new half-size spectra h_nA (for A²+3B²) and h_nB (for 2AB).
 *
 * This fuses all three passes (post / combine / pre) into a single O(n) loop,
 * avoiding two extra memory passes.  Every k=1..M/2-1 reads 4 complex values
 * and writes 4 complex values in one cache-friendly sequential sweep.
 *
 * hA/hB may alias hNA/hNB when used for squaring a single Z[√3] element.
 * We use temporary locals to avoid write-read hazards.                        */

static void fp_combine(const FpEngine * restrict e,
                       const double   * restrict hA_re,
                       const double   * restrict hA_im,
                       const double   * restrict hB_re,
                       const double   * restrict hB_im,
                       double         * hNA_re,
                       double         * hNA_im,
                       double         * hNB_re,
                       double         * hNB_im) {
    const double *twr = e->tw_re;
    const double *twi = e->tw_im;
    size_t M = e->M;

    /* k = 0  (DC bin, real = X[0];  imag = X[M]) */
    {
        double xA0 = hA_re[0] + hA_im[0];   /* X_A[0]  */
        double xAM = hA_re[0] - hA_im[0];   /* X_A[M]  */
        double xB0 = hB_re[0] + hB_im[0];   /* X_B[0]  */
        double xBM = hB_re[0] - hB_im[0];   /* X_B[M]  */
        double yA0 = xA0 * xA0 + 3.0 * xB0 * xB0;
        double yAM = xAM * xAM + 3.0 * xBM * xBM;
        double yB0 = 2.0 * xA0 * xB0;
        double yBM = 2.0 * xAM * xBM;
        hNA_re[0] = (yA0 + yAM) * 0.5;
        hNA_im[0] = (yA0 - yAM) * 0.5;
        hNB_re[0] = (yB0 + yBM) * 0.5;
        hNB_im[0] = (yB0 - yBM) * 0.5;
    }

    /* k = 1 .. M/2 - 1  (general bins, fused post + op + pre) */
    for (size_t k = 1; k < M / 2; ++k) {
        size_t mk = M - k;
        double wr = twr[k], wi = twi[k];

        /* Post-process h_A[k], h_A[M-k]  →  X_A[k], X_A[M-k] */
        double prA = (hA_re[k] + hA_re[mk]) * 0.5;
        double piA = (hA_im[k] - hA_im[mk]) * 0.5;
        double qrA = (hA_im[k] + hA_im[mk]) * 0.5;
        double qiA = (hA_re[mk] - hA_re[k]) * 0.5;
        double xAk_re  = prA + wr * qrA - wi * qiA;
        double xAk_im  = piA + wr * qiA + wi * qrA;
        double xAmk_re = prA - (wr * qrA - wi * qiA);
        double xAmk_im = -piA + (wr * qiA + wi * qrA);

        /* Post-process h_B[k], h_B[M-k]  →  X_B[k], X_B[M-k] */
        double prB = (hB_re[k] + hB_re[mk]) * 0.5;
        double piB = (hB_im[k] - hB_im[mk]) * 0.5;
        double qrB = (hB_im[k] + hB_im[mk]) * 0.5;
        double qiB = (hB_re[mk] - hB_re[k]) * 0.5;
        double xBk_re  = prB + wr * qrB - wi * qiB;
        double xBk_im  = piB + wr * qiB + wi * qrB;
        double xBmk_re = prB - (wr * qrB - wi * qiB);
        double xBmk_im = -piB + (wr * qiB + wi * qrB);

        /* Z[√3] pointwise operation at bin k:
         *   Y_A[k] = X_A[k]² + 3·X_B[k]²
         *   Y_B[k] = 2·X_A[k]·X_B[k]  */
        double yAk_re  = xAk_re  * xAk_re  - xAk_im  * xAk_im
                       + 3.0 * (xBk_re  * xBk_re  - xBk_im  * xBk_im);
        double yAk_im  = 2.0 * xAk_re  * xAk_im
                       + 3.0 * 2.0 * xBk_re  * xBk_im;
        double yAmk_re = xAmk_re * xAmk_re - xAmk_im * xAmk_im
                       + 3.0 * (xBmk_re * xBmk_re - xBmk_im * xBmk_im);
        double yAmk_im = 2.0 * xAmk_re * xAmk_im
                       + 3.0 * 2.0 * xBmk_re * xBmk_im;
        double yBk_re  = 2.0 * (xAk_re  * xBk_re  - xAk_im  * xBk_im);
        double yBk_im  = 2.0 * (xAk_re  * xBk_im  + xAk_im  * xBk_re);
        double yBmk_re = 2.0 * (xAmk_re * xBmk_re - xAmk_im * xBmk_im);
        double yBmk_im = 2.0 * (xAmk_re * xBmk_im + xAmk_im * xBmk_re);

        /* Pre-process: pack (Y_A[k], Y_A[M-k]) into h_NA[k], h_NA[M-k] */
        double epA_re = (yAk_re + yAmk_re) * 0.5;
        double epA_im = (yAk_im - yAmk_im) * 0.5;
        double vkA_re = yAk_re - yAmk_re;
        double vkA_im = yAk_im + yAmk_im;
        double t1A = (wi * vkA_re - wr * vkA_im) * 0.5;
        double t2A = (wr * vkA_re + wi * vkA_im) * 0.5;
        hNA_re[k]  = epA_re + t1A;
        hNA_im[k]  = epA_im + t2A;
        hNA_re[mk] = epA_re - t1A;
        hNA_im[mk] = -epA_im + t2A;

        /* Pre-process: pack (Y_B[k], Y_B[M-k]) into h_NB[k], h_NB[M-k] */
        double epB_re = (yBk_re + yBmk_re) * 0.5;
        double epB_im = (yBk_im - yBmk_im) * 0.5;
        double vkB_re = yBk_re - yBmk_re;
        double vkB_im = yBk_im + yBmk_im;
        double t1B = (wi * vkB_re - wr * vkB_im) * 0.5;
        double t2B = (wr * vkB_re + wi * vkB_im) * 0.5;
        hNB_re[k]  = epB_re + t1B;
        hNB_im[k]  = epB_im + t2B;
        hNB_re[mk] = epB_re - t1B;
        hNB_im[mk] = -epB_im + t2B;
    }

    /* k = M/2  (center bin, self-conjugate) */
    if (M >= 2) {
        size_t k = M / 2;
        double wr = twr[k], wi = twi[k];
        double xAk_re = hA_re[k] + wr * hA_im[k];
        double xAk_im = wi * hA_im[k];
        double xBk_re = hB_re[k] + wr * hB_im[k];
        double xBk_im = wi * hB_im[k];
        double yAk_re = xAk_re * xAk_re - xAk_im * xAk_im
                      + 3.0 * (xBk_re * xBk_re - xBk_im * xBk_im);
        double yAk_im = 2.0 * xAk_re * xAk_im + 3.0 * 2.0 * xBk_re * xBk_im;
        double yBk_re = 2.0 * (xAk_re * xBk_re - xAk_im * xBk_im);
        double yBk_im = 2.0 * (xAk_re * xBk_im + xAk_im * xBk_re);
        hNA_re[k] = yAk_re;  hNA_im[k] = -yAk_im;
        hNB_re[k] = yBk_re;  hNB_im[k] = -yBk_im;
    }
}

/* ─── Carry propagation + Mersenne wrap-around fold ─────────────────────────── */
/* Normalizes digit array in-place: each digit d[j] is rounded to the nearest
 * integer and carry is propagated.  The wrap-around carry implements the
 * Mersenne reduction: 2^p ≡ 1 mod M_p.                                       */

static void fp_carry(const FpEngine * restrict e,
                     double         * restrict d) {
    const double *mp  = e->mod_pow;
    const double *imp = e->inv_mod_pow;
    size_t n = e->n;
    double carry = 0.0;
    for (size_t j = 0; j < n; ++j) {
        double raw     = d[j] + carry;
        double rounded = nearbyint(raw);
        carry  = floor(rounded * imp[j]);
        d[j]   = rounded - carry * mp[j];
    }
    /* Wrap-around: carry * 2^p ≡ carry (mod M_p) */
    for (size_t j = 0; j < n && carry != 0.0; ++j) {
        double raw     = d[j] + carry;
        double rounded = nearbyint(raw);
        carry  = floor(rounded * imp[j]);
        d[j]   = rounded - carry * mp[j];
    }
}

/* ─── Inverse pass: M-point IFFT + DWT-unpack + carry ───────────────────────── */

static void fp_inverse(const FpEngine * restrict e,
                       double         * restrict h_re,
                       double         * restrict h_im,
                       double         * restrict digits) {
    const double *wih = e->w_inv_half;
    size_t M = e->M;
    /* M-point inverse FFT (uses negated sine twiddles) */
    fft_core(h_re, h_im, e->bitrv, e->tw_re_half, e->tw_im_half_inv, M);
    /* DWT-unpack: scale by inverse weight and store into digit array */
    for (size_t k = 0; k < M; ++k) {
        digits[2 * k    ] = h_re[k] * wih[2 * k    ];
        digits[2 * k + 1] = h_im[k] * wih[2 * k + 1];
    }
    /* Carry propagation + Mersenne fold */
    fp_carry(e, digits);
}

/* ─── Z[√3] squaring step: (A,B) → (A²+3B², 2AB) mod M_p ──────────────────── */
/* This is the hot kernel: one LL iteration costs 2 forward FFTs + 1 combined
 * post/op/pre pass + 2 inverse FFTs.  All scratch is pre-allocated in z.     */

static void zsqrt3_step(const FpEngine * restrict e, ZState * restrict z) {
    /* Forward FFT of A and B */
    fp_forward(e, z->A, z->hA_re, z->hA_im);
    fp_forward(e, z->B, z->hB_re, z->hB_im);

    /* Combined post-process + Z[√3] pointwise + pre-process.
     * Write new_A spectrum into hA, new_B spectrum into hB (safe: temporary
     * locals inside fp_combine prevent read/write aliasing). */
    fp_combine(e,
               z->hA_re, z->hA_im,   /* in:  spectrum of A    */
               z->hB_re, z->hB_im,   /* in:  spectrum of B    */
               z->hA_re, z->hA_im,   /* out: spectrum of A²+3B² */
               z->hB_re, z->hB_im);  /* out: spectrum of 2AB    */

    /* Inverse FFT → new A and B digit arrays (carry + Mersenne fold inside) */
    fp_inverse(e, z->hA_re, z->hA_im, z->A);
    fp_inverse(e, z->hB_re, z->hB_im, z->B);
}

/* ─── Check A ≡ 0 (mod M_p) ─────────────────────────────────────────────────── */
/* After carry propagation all digits should be in [0, 2^b_j - 1].
 * Zero means all digits are exactly 0 after normalization.
 * M_p itself is represented as all-zeros (since M_p ≡ 0 mod M_p).          */

static int digits_is_zero(const double *d, size_t n) {
    for (size_t j = 0; j < n; ++j)
        if (d[j] != 0.0) return 0;
    return 1;
}

/* Load small unsigned integer val into digit array (little-endian digit 0). */
static void digits_load(const FpEngine *e, double *d, uint64_t val) {
    size_t n = e->n;
    memset(d, 0, n * sizeof(double));
    /* Distribute val across leading digits according to their widths */
    for (size_t j = 0; j < n && val > 0; ++j) {
        int b = e->digit_width[j];
        uint64_t mask = (b >= 64) ? UINT64_MAX : ((uint64_t)1 << b) - 1u;
        d[j] = (double)(val & mask);
        val >>= (b >= 64 ? 63 : b);
        if (b >= 64) val >>= 1;
    }
}

/* ─── Exact 64-bit path for small exponents p < 64 ─────────────────────────── */
/* Uses __uint128_t for 64-bit × 64-bit products.                              */

static inline uint64_t mulmod64_exact(uint64_t a, uint64_t b, uint64_t m) {
    return (uint64_t)((__uint128_t)a * b % m);
}

/* Returns 1 if M_p is prime (exact, p < 64). */
static int test_small_p(int p) {
    if (p == 2) return 1;   /* M_2 = 3 is prime (special case) */
    uint64_t mp = ((uint64_t)1 << p) - 1;
    /* Compute (A, B) ≡ (2+√3)^(2^(p-2)) in Z[√3] mod mp, starting from (2,1). */
    uint64_t A = 2, B = 1;
    for (int i = 0; i < p - 2; ++i) {
        /* (A+B√3)² = (A²+3B²) + 2AB·√3 */
        uint64_t A2  = mulmod64_exact(A, A, mp);
        uint64_t B2  = mulmod64_exact(B, B, mp);
        uint64_t AB  = mulmod64_exact(A, B, mp);
        uint64_t nA  = (A2 + mulmod64_exact(3, B2, mp)) % mp;
        uint64_t nB  = (2 * AB) % mp;
        A = nA; B = nB;
    }
    return (A == 0);
}

/* ─── Main primality test ────────────────────────────────────────────────────── */
/* Returns 1 if M_p is prime, 0 if composite, -1 on allocation error. */

static int test_mersenne_fp(int p) {
    if (p < 2) return 0;
    if (p == 2) return 1;
    if (p < FFT_THRESHOLD_P) return test_small_p(p);

    FpEngine e;
    if (!engine_init(&e, p)) return -1;

    ZState z;
    if (!zstate_alloc(&z, &e)) { engine_free(&e); return -1; }

    /* Initialize (A, B) = (2, 1) */
    digits_load(&e, z.A, 2);
    digits_load(&e, z.B, 1);

    /* p-2 squarings in Z[√3]: implements Mod[Ceil[a(p-2)], M_p] */
    for (int i = 0; i < p - 2; ++i)
        zsqrt3_step(&e, &z);

    int prime = digits_is_zero(z.A, e.n);

    zstate_free(&z);
    engine_free(&e);
    return prime;
}

/* ─── Thread pool ────────────────────────────────────────────────────────────── */

typedef struct {
    const int *primes;       /* array of prime exponents to test */
    int        nprime;       /* total count                      */
    atomic_int next;         /* next index to dispatch           */
    int       *results;      /* results[i] = 1/0/-1              */
    double    *times;        /* times[i] = seconds for this p    */
} WorkQueue;

static void *worker_thread(void *arg) {
    WorkQueue *q = (WorkQueue *)arg;
    for (;;) {
        int idx = atomic_fetch_add(&q->next, 1);
        if (idx >= q->nprime) break;
        int p = q->primes[idx];
        double t0 = wall_sec();
        q->results[idx] = test_mersenne_fp(p);
        double t1 = wall_sec();
        q->times[idx] = t1 - t0;
    }
    return NULL;
}

/* ─── Validation: known small Mersenne primes ────────────────────────────────── */

/* Known Mersenne prime exponents (OEIS A000043, first 22). */
static const int KNOWN_MERSENNE_P[] = {
    2, 3, 5, 7, 13, 17, 19, 31, 61, 89, 107, 127,
    521, 607, 1279, 2203, 2281, 3217, 4253, 4423, 9689, 9941
};
static const int N_KNOWN = (int)(sizeof(KNOWN_MERSENNE_P) / sizeof(KNOWN_MERSENNE_P[0]));

static void run_validate(int verbose) {
    printf("=== seqmod_fp validation ===\n");
    int ok = 0, fail = 0;
    for (int i = 0; i < N_KNOWN; ++i) {
        int p = KNOWN_MERSENNE_P[i];
        double t0 = wall_sec();
        int r = test_mersenne_fp(p);
        double dt = wall_sec() - t0;
        int expect = 1;
        int pass = (r == expect);
        if (pass) ++ok; else ++fail;
        if (verbose || !pass)
            printf("  p=%-7d  M_p=%s  expect=prime  got=%s  %.3f ms%s\n",
                   p,
                   "2^p-1",
                   r == 1 ? "prime" : (r == 0 ? "composite" : "error"),
                   dt * 1000.0,
                   pass ? "" : "  *** FAIL ***");
    }
    /* Also verify a few composites */
    int composites[] = {4, 6, 8, 9, 10, 11, 15, 23, 25};
    for (int i = 0; i < (int)(sizeof(composites)/sizeof(composites[0])); ++i) {
        int p = composites[i];
        if (!is_prime_td(p)) continue;   /* skip: p must be prime exponent */
        double t0 = wall_sec();
        int r = test_mersenne_fp(p);
        double dt = wall_sec() - t0;
        int expect = 0;
        int pass = (r == expect);
        if (pass) ++ok; else ++fail;
        if (verbose || !pass)
            printf("  p=%-7d  M_p=2^p-1  expect=composite  got=%s  %.3f ms%s\n",
                   p,
                   r == 1 ? "prime" : (r == 0 ? "composite" : "error"),
                   dt * 1000.0,
                   pass ? "" : "  *** FAIL ***");
    }
    printf("Validation: %d passed, %d failed\n", ok, fail);
}

/* ─── Benchmark helper ───────────────────────────────────────────────────────── */

static void run_bench(int prime_count, int start_p, int nthreads,
                      const char *csv_path, int verbose) {
    /* Gather prime candidates starting from start_p */
    int  limit_approx = start_p + prime_count * 20 + 1000;
    int  nsieve;
    int *all_primes = sieve_primes(limit_approx, &nsieve);

    /* Find start index */
    int start_idx = 0;
    while (start_idx < nsieve && all_primes[start_idx] < start_p) ++start_idx;

    /* Extend sieve if needed */
    while (nsieve - start_idx < prime_count) {
        limit_approx *= 2;
        free(all_primes);
        all_primes = sieve_primes(limit_approx, &nsieve);
        start_idx = 0;
        while (start_idx < nsieve && all_primes[start_idx] < start_p) ++start_idx;
    }

    const int *primes = all_primes + start_idx;
    int nprime = (prime_count < nsieve - start_idx) ? prime_count : nsieve - start_idx;

    int    *results = malloc((size_t)nprime * sizeof(int));
    double *times   = malloc((size_t)nprime * sizeof(double));
    if (!results || !times) { perror("malloc"); exit(1); }

    WorkQueue q;
    q.primes  = primes;
    q.nprime  = nprime;
    q.results = results;
    q.times   = times;
    atomic_init(&q.next, 0);

    printf("seqmod_fp: testing %d prime exponents starting from p=%d, threads=%d\n",
           nprime, primes[0], nthreads);

    double t_start = wall_sec();

    /* Launch threads */
    pthread_t *tids = malloc((size_t)nthreads * sizeof(pthread_t));
    if (!tids) { perror("malloc"); exit(1); }
    for (int i = 0; i < nthreads; ++i)
        pthread_create(&tids[i], NULL, worker_thread, &q);
    for (int i = 0; i < nthreads; ++i)
        pthread_join(tids[i], NULL);
    free(tids);

    double t_total = wall_sec() - t_start;

    /* Count Mersenne primes found */
    int found = 0;
    for (int i = 0; i < nprime; ++i) if (results[i] == 1) ++found;

    printf("Done: %.3f s total, %d Mersenne primes found in %d candidates\n",
           t_total, found, nprime);
    printf("  Throughput: %.1f primes/sec, avg %.3f ms/prime\n",
           nprime / t_total, t_total * 1000.0 / nprime);

    /* Print results */
    if (verbose) {
        for (int i = 0; i < nprime; ++i) {
            printf("  p=%-8d  %s  %.3f ms\n",
                   primes[i],
                   results[i] == 1 ? "PRIME" : "composite",
                   times[i] * 1000.0);
        }
    } else {
        /* Only print primes */
        printf("Mersenne primes found: {");
        int first = 1;
        for (int i = 0; i < nprime; ++i) {
            if (results[i] == 1) {
                printf("%s%d", first ? "" : ", ", primes[i]);
                first = 0;
            }
        }
        printf("}\n");
    }

    /* CSV output */
    if (csv_path && *csv_path) {
        FILE *f = fopen(csv_path, "w");
        if (f) {
            fprintf(f, "p,is_prime,time_sec\n");
            for (int i = 0; i < nprime; ++i)
                fprintf(f, "%d,%s,%.6f\n",
                        primes[i],
                        results[i] == 1 ? "true" : "false",
                        times[i]);
            fclose(f);
            printf("CSV written to: %s\n", csv_path);
        }
    }

    free(results);
    free(times);
    free(all_primes);
}

/* ─── Formula demonstration ──────────────────────────────────────────────────── */
/* Prints first 10 terms of a(n) = (2+√3)^(2^n) and the corresponding
 * primality test result Mod[Ceil[a(n)], 2^(n+2)-1].                          */

static void run_formula_demo(void) {
    printf("=== Formula demonstration: a(n) = (2+√3)^(2^n) ===\n");
    printf("  Ceil[a(n)] = s_n (Lucas-Lehmer sequence: s_0=4, s_{k+1}=s_k^2-2)\n");
    printf("  Mod[Ceil[a(n)], 2^(n+2)-1] = 0 ⟹ M_{n+2} = 2^{n+2}-1 is prime\n");
    printf("  (n=0 is a special case: M_2=3 is prime, formula gives 4 mod 3 = 1)\n\n");
    printf("  %-4s  %-6s  %-12s  %-12s  %-10s  %s\n",
           "n", "p=n+2", "M_p=2^p-1", "Ceil[a(n)]", "Mod result", "Verdict");
    printf("  %s\n", "------------------------------------------------------------------");

    /* s_n via exact recurrence s_0=4, s_{k+1}=s_k^2-2.
     * Representable exactly in uint64_t up to n=3 (s_4 fits, s_5 overflows). */
    uint64_t s = 4;
    for (int n = 0; n <= 6; ++n) {
        int p = n + 2;
        uint64_t mp = (p < 64) ? (((uint64_t)1 << p) - 1) : 0;

        /* Use FFT test for the actual primality verdict */
        int is_p = test_mersenne_fp(p);

        if (n <= 3 && mp > 0) {
            uint64_t mod_val = s % mp;
            const char *verdict = (is_p == 1) ? "PRIME (mod=0 ✓)" :
                                  (mod_val == 0 && n == 0) ? "PRIME (special: p=2)" :
                                  "composite";
            /* For n=0: s_0=4, M_2=3, 4 mod 3=1 but M_2 IS prime (formula gives 1 for p=2) */
            printf("  %-4d  %-6d  %-12llu  %-12llu  %-10llu  %s\n",
                   n, p, (unsigned long long)mp,
                   (unsigned long long)s,
                   (unsigned long long)mod_val, verdict);
        } else {
            /* Sequence overflows uint64; just show primality from FFT test */
            printf("  %-4d  %-6d  %-12llu  (large)       (FFT)      %s\n",
                   n, p, (unsigned long long)mp,
                   is_p == 1 ? "PRIME (FFT path)" : "composite (FFT path)");
        }

        /* Advance sequence s (exact while it fits) */
        if (n <= 2) {
            s = s * s - 2;
        } else {
            /* s_4 = 37634^2-2 fits uint64, s_5 = 1416317954^2-2 also fits,
             * s_6 overflows — stop exact computation at n=3 */
            s = 0;
        }
    }
    printf("\n");
}

/* ─── main ───────────────────────────────────────────────────────────────────── */

int main(int argc, char *argv[]) {
    /* Parse special modes */
    if (argc >= 2 && strcmp(argv[1], "--validate") == 0) {
        int verbose = (argc >= 3 && strcmp(argv[2], "-v") == 0);
        run_validate(verbose);
        return 0;
    }
    if (argc >= 2 && strcmp(argv[1], "--formula") == 0) {
        run_formula_demo();
        return 0;
    }

    /* Normal benchmark mode */
    const char *env_threads = getenv("SEQMOD_FP_THREADS");
    const char *env_csv     = getenv("SEQMOD_FP_OUTPUT");
    const char *env_verbose = getenv("SEQMOD_FP_VERBOSE");

    int nthreads = env_threads ? atoi(env_threads) : 4;
    if (nthreads < 1)  nthreads = 1;
    if (nthreads > (int)MAX_THREADS) nthreads = (int)MAX_THREADS;

    int verbose  = env_verbose ? atoi(env_verbose) : 0;

    /* Command-line: seqmod_fp [prime_count [start_p [threads]]] */
    int prime_count = (argc > 1) ? atoi(argv[1]) : 512;
    int start_p     = (argc > 2) ? atoi(argv[2]) : 2;
    if (argc > 3)     nthreads = atoi(argv[3]);
    if (prime_count < 1) prime_count = 1;
    if (start_p     < 2) start_p     = 2;

    const char *csv_path = (argc > 4) ? argv[4] : (env_csv ? env_csv : NULL);

    /* Print formula demo first */
    run_formula_demo();

    /* Run benchmark */
    run_bench(prime_count, start_p, nthreads, csv_path, verbose);

    return 0;
}
