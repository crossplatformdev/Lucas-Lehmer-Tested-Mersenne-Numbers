/*
 * src/split_bucket_batches.c
 *
 * C replacement for scripts/split_bucket_batches.py.
 * Compiles to a self-contained binary with no external dependencies.
 *
 * Usage:
 *   split_bucket_batches [--bucket-start N] [--bucket-end N]
 *                        [--batch-size N]
 *                        [--time-limit-seconds S]
 *                        [--resume-from-exponent N]
 *                        [--target-workers N]
 *                        [--prime-half {full|lower_half|upper_half}]
 *                        [--chunk-index N]  [--chunk-size N]
 *                        [--count-chunks]
 *                        [--dry-run]
 *                        [--output FILE] [--count-only]
 *
 * Bucket definition (1-indexed, n in [1, 64]):
 *   B_1  = [2, 2]
 *   B_n  = [2^(n-1), 2^n - 1]  for n >= 2
 *   B_64 = [2^63, 2^64 - 1]
 *
 * --target-workers N
 *   Compute batch sizes so that the total number of batches is as close to N
 *   as possible.  The algorithm counts total remaining primes across all
 *   selected buckets and sets batch_size = ceil(total / N).  The per-bucket
 *   time-limit cap still applies so no worker exceeds the budget.
 *   --batch-size acts as a hard upper cap on top of this.
 *
 * --time-limit-seconds S
 *   Compute a per-bucket batch size using the inverse rule of three:
 *       batch_size = floor(S / worst_case_time_per_exponent)
 *   Timing data for buckets 9-16 is taken from benchmark measurements
 *   (REPORT_MERSENNE_EXPONENTS-256-65536.MD); buckets 17+ are extrapolated
 *   at ×4.5 per bucket.  The result is capped by --batch-size (default 1000).
 *
 * --resume-from-exponent N
 *   Skip all prime exponents ≤ N.  Buckets whose last prime is ≤ N are
 *   omitted entirely; the first remaining batch starts from the next
 *   untested prime.  Batch ordinal numbers are still reported relative to
 *   the full bucket prime list so worker names remain stable across resumes.
 *
 * --prime-half {full|lower_half|upper_half}
 *   Filter the batch matrix to cover only the specified half of all prime
 *   exponents (ordered ascending by value).  Applied before chunking so
 *   worker ordinals remain stable when the same sweep is split across runs.
 *   lower_half – first floor(total/2) primes; last batch may be trimmed.
 *   upper_half – last ceil(total/2) primes; first batch may be trimmed
 *                (batch_min_exponent is advanced to the correct value).
 *   full (default) – include all primes.
 *
 * --count-chunks
 *   Instead of emitting a JSON batch matrix, print a single JSON object:
 *     {"total_batches":<N>,"chunk_size":<K>,"chunk_count":<C>}
 *   where N is the total filtered batch count, K is the --chunk-size value
 *   (default 256), and C = ceil(N/K).  Useful for workflow orchestration.
 *
 * --chunk-index N  (default 0)
 *   Emit only batch entries in the Nth chunk (0-based).  The output is a
 *   slice of the full filtered matrix: entries [N*chunk_size, (N+1)*chunk_size).
 *
 * --chunk-size K  (default 256, max 256)
 *   Number of batches per chunk / GitHub Actions matrix limit.
 *   When --chunk-index / --count-chunks is used, this controls the slice size.
 *
 * Worker name format:
 *   bucket-{N:02d}-batch-{start_ordinal:04d}-{end_ordinal:04d}-exp-{pmin}-{pmax}
 *
 * Example:
 *   bucket-17-batch-0001-0850-exp-65537-87251
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <inttypes.h>
#include <math.h>

#define GITHUB_MATRIX_MAX   256
#define BATCH_SIZE_DEFAULT  1000
#define WORKER_NAME_MAX     128

/* =========================================================================
 * Per-bucket worst-case single-exponent test time (seconds).
 * Source: REPORT_MERSENNE_EXPONENTS-256-65536.MD benchmark run.
 *   Buckets 9-16  – measured "Longest (s)" column.
 *   Buckets 1-10  – sub-millisecond; floored at 0.001 s.
 *   Buckets 17-21 – extrapolated at ×4.5 per bucket.
 * ========================================================================= */
#define BUCKET_WORST_SEC_MAX 21   /* highest index in the table below */
static const double BUCKET_WORST_SEC[BUCKET_WORST_SEC_MAX + 1] = {
    /* 0  */ 0.000575,  /* unused – bucket indices start at 1 */
    /* 1  */ 0.000575,  /* 2  */ 0.000575,  /* 3  */ 0.000575,  /* 4  */ 0.000575,
    /* 5  */ 0.000575,  /* 6  */ 0.000575,  /* 7  */ 0.000575,  /* 8  */ 0.000575,
    /* 9  */ 0.000575,  /* 10 */ 0.000575,  /* measured: <1 ms            */
    /* 11 */ 0.002875,  /* measured                                     */
    /* 12 */ 0.015525,  /* measured                                     */
    /* 13 */ 0.053475,  /* measured                                     */
    /* 14 */ 0.186875,  /* measured                                     */
    /* 15 */ 0.737725,  /* measured                                     */
    /* 16 */ 3.246450,  /* measured                                     */
    /* 17 */ 14.609025, /* extrapolated: previous × 4.5                 */
    /* 18 */ 65.740900, /* extrapolated: previous × 4.5                 */
    /* 19 */ 295.832900,/* extrapolated: previous × 4.5                 */
    /* 20 */ 1331.247475,/* extrapolated: previous × 4.5                */
    /* 21 */ 5990.613350,/* extrapolated: previous × 4.5                */
};
#define BUCKET_WORST_GROWTH 4.5   /* extrapolation factor for buckets > 21 */

/* Safety margin applied to extrapolated estimates (buckets > 16) to account
 * for the uncertainty in the growth factor (observed trend: ×3.5, ×4.0, ×4.4,
 * still increasing).  A 20% buffer keeps workers safely inside the 6-h limit
 * even when the true per-exponent time slightly exceeds the 4.5× assumption. */
#define EXTRAPOLATED_SAFETY_MARGIN 1.2

/* Return the estimated worst-case single-exponent time for bucket n.
 * For buckets 1–16: measured values from the benchmark report.
 * For buckets 17+:  extrapolated at ×4.5 per bucket, then multiplied by the
 *                   safety margin to add a conservative buffer.                */
static double worst_sec_for_bucket(int n)
{
    double w;
    if (n <= BUCKET_WORST_SEC_MAX) {
        w = BUCKET_WORST_SEC[n > 0 ? n : 0];
    } else {
        /* Extrapolate beyond the table */
        w = BUCKET_WORST_SEC[BUCKET_WORST_SEC_MAX];
        for (int i = BUCKET_WORST_SEC_MAX; i < n; i++)
            w *= BUCKET_WORST_GROWTH;
    }
    /* Apply safety margin to all extrapolated buckets (> 16). */
    if (n > 16)
        w *= EXTRAPOLATED_SAFETY_MARGIN;
    return w;
}

/* Inverse rule of three: how many primes fit in time_limit_sec? */
static size_t batch_size_for_bucket(int n, double time_limit_sec)
{
    double worst = worst_sec_for_bucket(n);
    if (worst <= 0.0) return BATCH_SIZE_DEFAULT;
    size_t sz = (size_t)(time_limit_sec / worst);
    return sz < 1 ? 1 : sz;
}

/* =========================================================================
 * Primality test (trial division).
 * ========================================================================= */
static int is_prime(uint64_t n)
{
    if (n < 2)  return 0;
    if (n == 2) return 1;
    if ((n & 1) == 0) return 0;
    if (n == 3) return 1;
    if (n % 3 == 0) return 0;
    for (uint64_t i = 5; i <= n / i; i += 6) {
        if (n % i == 0 || n % (i + 2) == 0) return 0;
    }
    return 1;
}

/* =========================================================================
 * Bucket range: bucket n (1-indexed) covers [lo, hi].
 * ========================================================================= */
static void bucket_range(int n, uint64_t *lo, uint64_t *hi)
{
    if (n == 1) { *lo = 2; *hi = 2; return; }
    *lo = (uint64_t)1 << (n - 1);
    *hi = (n < 64) ? (((uint64_t)1 << n) - 1) : UINT64_MAX;
}

/* =========================================================================
 * Dynamic array of uint64_t.
 * ========================================================================= */
typedef struct { uint64_t *data; size_t size; size_t cap; } U64Vec;

static void u64vec_push(U64Vec *v, uint64_t x)
{
    if (v->size == v->cap) {
        size_t new_cap = v->cap ? v->cap * 2 : 256;
        uint64_t *p = realloc(v->data, new_cap * sizeof(uint64_t));
        if (!p) { perror("realloc"); exit(1); }
        v->data = p;
        v->cap  = new_cap;
    }
    v->data[v->size++] = x;
}

/* =========================================================================
 * Enumerate all prime exponents in bucket n, ascending.
 * ========================================================================= */
static U64Vec enumerate_bucket_primes(int n)
{
    uint64_t lo, hi;
    bucket_range(n, &lo, &hi);

    U64Vec v = {NULL, 0, 0};

    uint64_t p = (lo >= 2) ? lo : 2;
    if (p > 2 && (p & 1) == 0) p++;

    while (p <= hi) {
        if (is_prime(p)) u64vec_push(&v, p);
        if (p == 2) {
            p = 3;
        } else {
            if (hi - p < 2) break;
            p += 2;
        }
    }
    return v;
}

/* =========================================================================
 * Batch descriptor.
 * ========================================================================= */
typedef struct {
    int      bucket_n;
    uint64_t bucket_min;
    uint64_t bucket_max;
    size_t   batch_index;
    size_t   batch_count;
    size_t   batch_prime_start_index;
    size_t   batch_prime_end_index;
    uint64_t batch_min_exponent;
    uint64_t batch_max_exponent;
    size_t   batch_size;
    char     worker_name[WORKER_NAME_MAX];
} Batch;

typedef struct { Batch *data; size_t size; size_t cap; } BatchVec;

static void batch_push(BatchVec *v, const Batch *b)
{
    if (v->size == v->cap) {
        size_t new_cap = v->cap ? v->cap * 2 : 16;
        Batch *p = realloc(v->data, new_cap * sizeof(Batch));
        if (!p) { perror("realloc"); exit(1); }
        v->data = p;
        v->cap  = new_cap;
    }
    v->data[v->size++] = *b;
}

/* =========================================================================
 * Split a bucket's prime list into batches.
 *
 * batch_size:   max primes per batch (already per-bucket computed by caller)
 * resume_from:  skip primes ≤ this value; 0 = no skip
 *
 * Batch ordinals are reported relative to the full (un-skipped) prime list
 * so that worker names remain stable across resume runs.
 * ========================================================================= */
static BatchVec split_bucket_into_batches(int n, size_t batch_size,
                                           uint64_t resume_from)
{
    U64Vec primes = enumerate_bucket_primes(n);
    BatchVec result = {NULL, 0, 0};
    if (primes.size == 0) { free(primes.data); return result; }

    uint64_t lo, hi;
    bucket_range(n, &lo, &hi);

    size_t total = primes.size;

    /* Find the first prime that hasn't been tested yet. */
    size_t skip = 0;
    if (resume_from > 0) {
        while (skip < total && primes.data[skip] <= resume_from)
            skip++;
    }

    /* All primes in this bucket already tested – skip the bucket. */
    if (skip == total) {
        free(primes.data);
        return result;
    }

    /* Remaining primes to test. */
    uint64_t *rem   = primes.data + skip;
    size_t    rem_n = total - skip;
    size_t    batch_count = (rem_n + batch_size - 1) / batch_size;

    for (size_t i = 0; i < batch_count; i++) {
        size_t start_in_rem = i * batch_size;
        size_t chunk_size   = (start_in_rem + batch_size <= rem_n)
                              ? batch_size : (rem_n - start_in_rem);
        size_t end_in_rem   = start_in_rem + chunk_size - 1;

        /* Absolute ordinals in the full bucket prime list (0-based). */
        size_t abs_start = skip + start_in_rem;
        size_t abs_end   = skip + end_in_rem;

        Batch b;
        b.bucket_n                = n;
        b.bucket_min              = lo;
        b.bucket_max              = hi;
        b.batch_index             = i;
        b.batch_count             = batch_count;
        b.batch_prime_start_index = abs_start;
        b.batch_prime_end_index   = abs_end;
        b.batch_min_exponent      = rem[start_in_rem];
        b.batch_max_exponent      = rem[end_in_rem];
        b.batch_size              = chunk_size;

        snprintf(b.worker_name, WORKER_NAME_MAX,
                 "bucket-%02d-batch-%04zu-%04zu-exp-%" PRIu64 "-%" PRIu64,
                 n,
                 abs_start + 1, abs_end + 1,
                 b.batch_min_exponent, b.batch_max_exponent);

        batch_push(&result, &b);
    }

    free(primes.data);
    return result;
}

/* =========================================================================
 * Print a slice [slice_start, slice_end) of the batch matrix as JSON.
 * ========================================================================= */
static void print_json_matrix_slice(FILE *out, const BatchVec *m,
                                    size_t slice_start, size_t slice_end)
{
    if (slice_end > m->size) slice_end = m->size;
    fputc('[', out);
    for (size_t i = slice_start; i < slice_end; i++) {
        const Batch *b = &m->data[i];
        if (i > slice_start) fputc(',', out);
        fprintf(out,
                "{\"bucket_n\":%d"
                ",\"bucket_min\":%" PRIu64
                ",\"bucket_max\":%" PRIu64
                ",\"batch_index\":%zu"
                ",\"batch_count\":%zu"
                ",\"batch_prime_start_index\":%zu"
                ",\"batch_prime_end_index\":%zu"
                ",\"batch_min_exponent\":%" PRIu64
                ",\"batch_max_exponent\":%" PRIu64
                ",\"batch_size\":%zu"
                ",\"worker_name\":\"%s\"}",
                b->bucket_n,
                b->bucket_min, b->bucket_max,
                b->batch_index, b->batch_count,
                b->batch_prime_start_index, b->batch_prime_end_index,
                b->batch_min_exponent, b->batch_max_exponent,
                b->batch_size,
                b->worker_name);
    }
    fputs("]\n", out);
}

/* =========================================================================
 * Filter the batch matrix to the lower or upper half of all prime exponents
 * (ordered ascending by value).  Applied before chunking so worker ordinals
 * remain stable when the same sweep is split across multiple runs.
 *
 * prime_half: 0 = full (no-op), 1 = lower_half, 2 = upper_half
 *
 * Half definition:
 *   total = sum of batch_size across all batches
 *   mid   = total / 2  (integer, rounds down)
 *   lower = first mid primes (indices 0 .. mid-1, ascending by value)
 *   upper = last (total-mid) primes (indices mid .. total-1)
 *
 * Partial-batch handling:
 *   lower – the last batch crossing the boundary is trimmed: batch_size and
 *           batch_max_exponent are reduced to include only the first 'take'
 *           primes.
 *   upper – the first batch crossing the boundary is trimmed: batch_size is
 *           reduced and batch_min_exponent / batch_prime_start_index are
 *           advanced to the correct prime value.
 * ========================================================================= */
static BatchVec apply_prime_half(const BatchVec *m, int prime_half)
{
    BatchVec result = {NULL, 0, 0};

    if (prime_half == 0) {
        for (size_t i = 0; i < m->size; i++)
            batch_push(&result, &m->data[i]);
        return result;
    }

    /* Count total primes across all batches. */
    size_t total = 0;
    for (size_t i = 0; i < m->size; i++)
        total += m->data[i].batch_size;

    if (total == 0)
        return result;

    size_t mid = total / 2;  /* lower half: [0, mid); upper half: [mid, total) */

    if (prime_half == 1) {   /* lower_half */
        if (mid == 0)
            return result;

        size_t seen = 0;
        for (size_t i = 0; i < m->size; i++) {
            if (seen >= mid) break;
            const Batch *b = &m->data[i];
            size_t take = b->batch_size;
            if (seen + take > mid)
                take = mid - seen;

            if (take == b->batch_size) {
                batch_push(&result, b);
            } else {
                /* Partial batch: trim to 'take' primes from the start. */
                Batch partial = *b;
                partial.batch_size = take;
                size_t end_idx = b->batch_prime_start_index + take - 1;
                partial.batch_prime_end_index = end_idx;
                /* Look up the prime value at the trimmed end. */
                U64Vec primes = enumerate_bucket_primes(b->bucket_n);
                if (end_idx < primes.size)
                    partial.batch_max_exponent = primes.data[end_idx];
                free(primes.data);
                /* Regenerate worker_name to reflect the trimmed end ordinal/exponent. */
                snprintf(partial.worker_name, WORKER_NAME_MAX,
                         "bucket-%02d-batch-%04zu-%04zu-exp-%" PRIu64 "-%" PRIu64,
                         partial.bucket_n,
                         partial.batch_prime_start_index + 1,
                         end_idx + 1,
                         partial.batch_min_exponent,
                         partial.batch_max_exponent);
                batch_push(&result, &partial);
            }
            seen += take;
        }

    } else {   /* upper_half */
        size_t upper_count = total - mid;
        if (upper_count == 0)
            return result;

        size_t seen = 0;
        for (size_t i = 0; i < m->size; i++) {
            const Batch *b = &m->data[i];
            size_t batch_end = seen + b->batch_size;

            if (batch_end <= mid) {
                /* Entire batch is in the lower half – skip it. */
                seen = batch_end;
                continue;
            }

            if (seen < mid) {
                /* Straddling batch: skip the first (mid - seen) primes. */
                size_t skip_in_batch = mid - seen;
                Batch partial = *b;
                size_t new_start_idx = b->batch_prime_start_index + skip_in_batch;
                partial.batch_prime_start_index = new_start_idx;
                partial.batch_size = b->batch_size - skip_in_batch;
                /* Look up the prime value at the new start position. */
                U64Vec primes = enumerate_bucket_primes(b->bucket_n);
                if (new_start_idx < primes.size)
                    partial.batch_min_exponent = primes.data[new_start_idx];
                free(primes.data);
                /* Regenerate worker_name to reflect the advanced start ordinal/exponent. */
                snprintf(partial.worker_name, WORKER_NAME_MAX,
                         "bucket-%02d-batch-%04zu-%04zu-exp-%" PRIu64 "-%" PRIu64,
                         partial.bucket_n,
                         new_start_idx + 1,
                         partial.batch_prime_end_index + 1,
                         partial.batch_min_exponent,
                         partial.batch_max_exponent);
                batch_push(&result, &partial);
            } else {
                batch_push(&result, b);
            }

            seen = batch_end;
        }
    }

    return result;
}

/* =========================================================================
 * Argument helpers
 * ========================================================================= */
static size_t parse_positive_int(const char *opt, const char *str)
{
    char *end;
    long val = strtol(str, &end, 10);
    if (*end != '\0' || end == str || val <= 0) {
        fprintf(stderr, "ERROR: %s requires a positive integer, got '%s'\n",
                opt, str);
        exit(1);
    }
    return (size_t)val;
}

static size_t parse_nonneg_int(const char *opt, const char *str)
{
    if (str[0] == '-') {
        fprintf(stderr, "ERROR: %s requires a non-negative integer, got '%s'\n",
                opt, str);
        exit(1);
    }
    char *end;
    unsigned long val = strtoul(str, &end, 10);
    if (*end != '\0' || end == str) {
        fprintf(stderr, "ERROR: %s requires a non-negative integer, got '%s'\n",
                opt, str);
        exit(1);
    }
    return (size_t)val;
}

static uint64_t parse_nonneg_uint64(const char *opt, const char *str)
{
    char *end;
    unsigned long long val = strtoull(str, &end, 10);
    if (*end != '\0' || end == str) {
        fprintf(stderr, "ERROR: %s requires a non-negative integer, got '%s'\n",
                opt, str);
        exit(1);
    }
    return (uint64_t)val;
}

static double parse_nonneg_double(const char *opt, const char *str)
{
    char *end;
    double val = strtod(str, &end);
    if (*end != '\0' || end == str || val < 0.0) {
        fprintf(stderr, "ERROR: %s requires a non-negative number, got '%s'\n",
                opt, str);
        exit(1);
    }
    if (!isfinite(val)) {
        fprintf(stderr, "ERROR: %s requires a finite number, got '%s'\n",
                opt, str);
        exit(1);
    }
    return val;
}

/* =========================================================================
 * main
 * ========================================================================= */
int main(int argc, char **argv)
{
    int         bucket_start     = 1;
    int         bucket_end       = 64;
    size_t      batch_size       = BATCH_SIZE_DEFAULT;
    double      time_limit_sec   = 0.0;   /* 0 = use --batch-size only */
    uint64_t    resume_from      = 0;     /* 0 = no skip               */
    size_t      target_workers   = 0;     /* 0 = off                   */
    int         prime_half       = 0;     /* 0=full, 1=lower_half, 2=upper_half */
    size_t      chunk_index      = 0;     /* 0-based chunk index       */
    size_t      chunk_sz         = GITHUB_MATRIX_MAX; /* max batches per chunk */
    int         count_chunks_mode = 0;    /* --count-chunks flag        */
    int         chunking_enabled = 0;     /* true when chunk flags used */
    int         dry_run          = 0;
    int         count_only       = 0;
    const char *output_file      = NULL;

    for (int i = 1; i < argc; i++) {
        if      (!strcmp(argv[i], "--bucket-start")        && i + 1 < argc)
            bucket_start   = (int)parse_positive_int("--bucket-start", argv[++i]);
        else if (!strcmp(argv[i], "--bucket-end")          && i + 1 < argc)
            bucket_end     = (int)parse_positive_int("--bucket-end",   argv[++i]);
        else if (!strcmp(argv[i], "--batch-size")          && i + 1 < argc)
            batch_size     = parse_positive_int("--batch-size", argv[++i]);
        else if (!strcmp(argv[i], "--time-limit-seconds")  && i + 1 < argc)
            time_limit_sec = parse_nonneg_double("--time-limit-seconds", argv[++i]);
        else if (!strcmp(argv[i], "--resume-from-exponent") && i + 1 < argc)
            resume_from    = parse_nonneg_uint64("--resume-from-exponent", argv[++i]);
        else if (!strcmp(argv[i], "--target-workers")      && i + 1 < argc)
            target_workers = parse_nonneg_int("--target-workers", argv[++i]);
        else if (!strcmp(argv[i], "--prime-half")          && i + 1 < argc) {
            const char *val = argv[++i];
            if      (!strcmp(val, "lower_half")) prime_half = 1;
            else if (!strcmp(val, "upper_half")) prime_half = 2;
            else if (!strcmp(val, "full"))       prime_half = 0;
            else {
                fprintf(stderr,
                        "ERROR: --prime-half must be full/lower_half/upper_half,"
                        " got '%s'\n", val);
                return 1;
            }
        }
        else if (!strcmp(argv[i], "--chunk-index")         && i + 1 < argc) {
            chunk_index      = parse_nonneg_int("--chunk-index", argv[++i]);
            chunking_enabled = 1;
        }
        else if (!strcmp(argv[i], "--chunk-size")          && i + 1 < argc) {
            chunk_sz = parse_positive_int("--chunk-size", argv[++i]);
            if (chunk_sz > GITHUB_MATRIX_MAX) {
                fprintf(stderr,
                        "ERROR: --chunk-size must be in [1, %d], got %zu\n",
                        GITHUB_MATRIX_MAX, chunk_sz);
                return 1;
            }
            chunking_enabled = 1;
        }
        else if (!strcmp(argv[i], "--count-chunks")) {
            count_chunks_mode = 1;
            chunking_enabled  = 1;
        }
        else if (!strcmp(argv[i], "--dry-run"))
            dry_run        = 1;
        else if (!strcmp(argv[i], "--count-only"))
            count_only     = 1;
        else if (!strcmp(argv[i], "--output")              && i + 1 < argc)
            output_file    = argv[++i];
        else {
            fprintf(stderr, "Unknown argument: %s\n", argv[i]);
            return 1;
        }
    }

    /* Validate */
    if (bucket_start < 1 || bucket_start > 64) {
        fprintf(stderr, "ERROR: --bucket-start must be in [1, 64], got %d\n",
                bucket_start);
        return 1;
    }
    if (bucket_end < 1 || bucket_end > 64) {
        fprintf(stderr, "ERROR: --bucket-end must be in [1, 64], got %d\n",
                bucket_end);
        return 1;
    }
    if (bucket_start > bucket_end) {
        fprintf(stderr,
                "ERROR: --bucket-start (%d) must be <= --bucket-end (%d)\n",
                bucket_start, bucket_end);
        return 1;
    }

    /* ── Step 1: count total remaining primes across all buckets ────────── */
    size_t total_remaining = 0;
    if (target_workers > 0) {
        for (int n = bucket_start; n <= bucket_end; n++) {
            U64Vec primes = enumerate_bucket_primes(n);
            size_t skip = 0;
            if (resume_from > 0) {
                while (skip < primes.size && primes.data[skip] <= resume_from)
                    skip++;
            }
            total_remaining += (primes.size > skip) ? (primes.size - skip) : 0;
            free(primes.data);
        }
    }

    /* ── Step 2: compute global target batch size ─────────────────────── */
    size_t global_target_bs;
    if (target_workers > 0 && total_remaining > 0) {
        /* ceil division: batch_size = ceil(total / target_workers) */
        global_target_bs = (total_remaining + target_workers - 1) / target_workers;
    } else {
        global_target_bs = batch_size;
    }

    /* ── Step 3: enumerate batches with per-bucket caps ──────────────── */
    BatchVec matrix = {NULL, 0, 0};
    for (int n = bucket_start; n <= bucket_end; n++) {
        /* Start with the global target */
        size_t bs = global_target_bs;

        /* Hard upper cap from --batch-size */
        if (bs > batch_size) bs = batch_size;

        /* Per-bucket time-limit safety cap */
        if (time_limit_sec > 0.0) {
            size_t tl_bs = batch_size_for_bucket(n, time_limit_sec);
            if (tl_bs < bs) bs = tl_bs;
        }

        BatchVec bv = split_bucket_into_batches(n, bs, resume_from);
        for (size_t j = 0; j < bv.size; j++)
            batch_push(&matrix, &bv.data[j]);
        free(bv.data);
    }

    /* ── Step 4: apply prime-half filter ──────────────────────────────── */
    BatchVec filtered = apply_prime_half(&matrix, prime_half);
    free(matrix.data);

    /* --count-only: print filtered batch count and exit. */
    if (count_only) {
        printf("%zu\n", filtered.size);
        free(filtered.data);
        return 0;
    }

    /* --count-chunks: print JSON summary and exit. */
    if (count_chunks_mode) {
        size_t num_chunks = (filtered.size + chunk_sz - 1) / chunk_sz;
        printf("{\"total_batches\":%zu,\"chunk_size\":%zu,\"chunk_count\":%zu}\n",
               filtered.size, chunk_sz, num_chunks);
        free(filtered.data);
        return 0;
    }

    /* --dry-run: human-readable plan showing all filtered batches. */
    if (dry_run) {
        printf("bucket_start         : %d\n", bucket_start);
        printf("bucket_end           : %d\n", bucket_end);
        if (target_workers > 0)
            printf("target_workers       : %zu  (total remaining primes: %zu)\n",
                   target_workers, total_remaining);
        if (time_limit_sec > 0.0) {
            printf("time_limit_seconds   : %.0f  (%.1f h per worker)\n",
                   time_limit_sec, time_limit_sec / 3600.0);
            printf("batch_size cap       : %zu\n", batch_size);
        } else {
            printf("batch_size           : %zu\n", batch_size);
        }
        if (prime_half == 1)      printf("prime_half           : lower_half\n");
        else if (prime_half == 2) printf("prime_half           : upper_half\n");
        if (resume_from > 0)
            printf("resume_from_exponent : %" PRIu64
                   "  (skipping exponents <= %" PRIu64 ")\n",
                   resume_from, resume_from);
        printf("total batches        : %zu\n", filtered.size);
        if (chunking_enabled) {
            size_t num_chunks = (filtered.size + chunk_sz - 1) / chunk_sz;
            printf("chunk_size           : %zu\n", chunk_sz);
            printf("chunk_count          : %zu\n", num_chunks);
        }
        printf("\n");
        for (size_t i = 0; i < filtered.size; i++) {
            const Batch *b = &filtered.data[i];
            if (time_limit_sec > 0.0) {
                double w = worst_sec_for_bucket(b->bucket_n);
                printf("  [%3zu/%3zu]  bucket=%2d"
                       "  primes=%6zu-%6zu"
                       "  exp=%" PRIu64 "-%" PRIu64
                       "  size=%zu  est_worst=%.3fs/exp  est_max=%.2fh"
                       "  name=%s\n",
                       b->batch_index, b->batch_count,
                       b->bucket_n,
                       b->batch_prime_start_index, b->batch_prime_end_index,
                       b->batch_min_exponent, b->batch_max_exponent,
                       b->batch_size, w,
                       (double)b->batch_size * w / 3600.0,
                       b->worker_name);
            } else {
                printf("  [%3zu/%3zu]  bucket=%2d"
                       "  primes=%6zu-%6zu"
                       "  exp=%" PRIu64 "-%" PRIu64
                       "  name=%s\n",
                       b->batch_index, b->batch_count,
                       b->bucket_n,
                       b->batch_prime_start_index, b->batch_prime_end_index,
                       b->batch_min_exponent, b->batch_max_exponent,
                       b->worker_name);
            }
        }
        if (!chunking_enabled && filtered.size > GITHUB_MATRIX_MAX) {
            fprintf(stderr,
                    "\nWARNING: %zu batches exceeds the GitHub Actions matrix"
                    " limit of %d.  Use --chunk-size / --chunk-index or narrow"
                    " the bucket range.\n",
                    filtered.size, GITHUB_MATRIX_MAX);
        }
        free(filtered.data);
        return 0;
    }

    /* Normal mode: output the JSON matrix for the requested chunk. */
    size_t slice_start = chunk_index * chunk_sz;
    size_t slice_end   = slice_start + chunk_sz;
    if (slice_end > filtered.size) slice_end = filtered.size;

    /* Guard: without explicit chunking, error if total exceeds the limit. */
    if (!chunking_enabled && filtered.size > GITHUB_MATRIX_MAX) {
        fprintf(stderr,
                "ERROR: %zu batches exceeds the GitHub Actions matrix limit"
                " of %d.  Use --count-chunks / --chunk-index / --chunk-size"
                " or narrow the bucket range.\n",
                filtered.size, GITHUB_MATRIX_MAX);
        free(filtered.data);
        return 1;
    }

    /* Guard: chunk_index out of range. */
    if (slice_start >= filtered.size && filtered.size > 0) {
        fprintf(stderr,
                "ERROR: --chunk-index %zu is out of range"
                " (total filtered batches: %zu, chunk_size: %zu)\n",
                chunk_index, filtered.size, chunk_sz);
        free(filtered.data);
        return 1;
    }

    FILE *out = stdout;
    if (output_file) {
        out = fopen(output_file, "w");
        if (!out) { perror(output_file); free(filtered.data); return 1; }
    }

    print_json_matrix_slice(out, &filtered, slice_start, slice_end);

    if (output_file) {
        fclose(out);
        fprintf(stderr, "Wrote %zu batch entries to %s\n",
                slice_end - slice_start, output_file);
    }

    free(filtered.data);
    return 0;
}
