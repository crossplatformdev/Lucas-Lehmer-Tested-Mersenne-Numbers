/*
 * src/split_bucket_batches.c
 *
 * C replacement for scripts/split_bucket_batches.py.
 * Compiles to a self-contained binary with no external dependencies.
 *
 * Usage:
 *   split_bucket_batches [--bucket-start N] [--bucket-end N]
 *                        [--batch-size N] [--dry-run]
 *                        [--output FILE] [--count-only]
 *
 * Bucket definition (1-indexed, n in [1, 64]):
 *   B_1  = [2, 2]
 *   B_n  = [2^(n-1), 2^n - 1]  for n >= 2
 *   B_64 = [2^63, 2^64 - 1]
 *
 * Worker name format:
 *   bucket-{N:02d}-batch-{start_ordinal:04d}-{end_ordinal:04d}-exp-{pmin}-{pmax}
 *
 * Example:
 *   bucket-17-batch-0001-1000-exp-65537-65867
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <inttypes.h>

#define GITHUB_MATRIX_MAX   256
#define BATCH_SIZE_DEFAULT  1000
#define WORKER_NAME_MAX     128

/* -------------------------------------------------------------------------
 * Primality test (trial division).
 * Uses i <= n/i to avoid overflow when i*i might wrap for large n.
 * ------------------------------------------------------------------------- */
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

/* -------------------------------------------------------------------------
 * Bucket range: bucket n (1-indexed) covers [lo, hi].
 * ------------------------------------------------------------------------- */
static void bucket_range(int n, uint64_t *lo, uint64_t *hi)
{
    if (n == 1) { *lo = 2; *hi = 2; return; }
    *lo = (uint64_t)1 << (n - 1);
    *hi = (n < 64) ? (((uint64_t)1 << n) - 1) : UINT64_MAX;
}

/* -------------------------------------------------------------------------
 * Dynamic array of uint64_t.
 * ------------------------------------------------------------------------- */
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

/* -------------------------------------------------------------------------
 * Enumerate all prime exponents in bucket n, ascending.
 * Mirrors enumerate_bucket_primes() from generate_bucket_primes.py.
 * ------------------------------------------------------------------------- */
static U64Vec enumerate_bucket_primes(int n)
{
    uint64_t lo, hi;
    bucket_range(n, &lo, &hi);

    U64Vec v = {NULL, 0, 0};

    /* Start candidate at lo (or 2), skip even numbers above 2. */
    uint64_t p = (lo >= 2) ? lo : 2;
    if (p > 2 && (p & 1) == 0) p++;

    while (p <= hi) {
        if (is_prime(p)) u64vec_push(&v, p);
        if (p == 2) {
            p = 3;
        } else {
            /* Guard against overflow and against stepping past hi. */
            if (hi - p < 2) break;
            p += 2;
        }
    }
    return v;
}

/* -------------------------------------------------------------------------
 * Batch descriptor.
 * ------------------------------------------------------------------------- */
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

/* -------------------------------------------------------------------------
 * Split a bucket's prime list into batches of at most batch_size.
 * Mirrors split_bucket_into_batches() from split_bucket_batches.py.
 * ------------------------------------------------------------------------- */
static BatchVec split_bucket_into_batches(int n, size_t batch_size)
{
    U64Vec primes = enumerate_bucket_primes(n);
    BatchVec result = {NULL, 0, 0};
    if (primes.size == 0) { free(primes.data); return result; }

    uint64_t lo, hi;
    bucket_range(n, &lo, &hi);

    size_t total       = primes.size;
    size_t batch_count = (total + batch_size - 1) / batch_size;

    for (size_t i = 0; i < batch_count; i++) {
        size_t start_idx  = i * batch_size;
        size_t chunk_size = (start_idx + batch_size <= total)
                            ? batch_size
                            : (total - start_idx);
        size_t end_idx    = start_idx + chunk_size - 1;

        Batch b;
        b.bucket_n                = n;
        b.bucket_min              = lo;
        b.bucket_max              = hi;
        b.batch_index             = i;
        b.batch_count             = batch_count;
        b.batch_prime_start_index = start_idx;
        b.batch_prime_end_index   = end_idx;
        b.batch_min_exponent      = primes.data[start_idx];
        b.batch_max_exponent      = primes.data[end_idx];
        b.batch_size              = chunk_size;

        /* Worker name: bucket-{N:02d}-batch-{s:04d}-{e:04d}-exp-{pmin}-{pmax} */
        snprintf(b.worker_name, WORKER_NAME_MAX,
                 "bucket-%02d-batch-%04zu-%04zu-exp-%" PRIu64 "-%" PRIu64,
                 n,
                 start_idx + 1, end_idx + 1,
                 b.batch_min_exponent, b.batch_max_exponent);

        batch_push(&result, &b);
    }

    free(primes.data);
    return result;
}

/* -------------------------------------------------------------------------
 * Print the batch matrix as a JSON array to *out.
 * Compact format (no extra whitespace) — valid JSON understood by fromJson().
 * ------------------------------------------------------------------------- */
static void print_json_matrix(FILE *out, const BatchVec *m)
{
    fputc('[', out);
    for (size_t i = 0; i < m->size; i++) {
        const Batch *b = &m->data[i];
        if (i > 0) fputc(',', out);
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

/* -------------------------------------------------------------------------
 * Parse a positive integer from a string; die with a clear error on failure.
 * ------------------------------------------------------------------------- */
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

/* -------------------------------------------------------------------------
 * main
 * ------------------------------------------------------------------------- */
int main(int argc, char **argv)
{
    int         bucket_start = 1;
    int         bucket_end   = 64;
    size_t      batch_size   = BATCH_SIZE_DEFAULT;
    int         dry_run      = 0;
    int         count_only   = 0;
    const char *output_file  = NULL;

    for (int i = 1; i < argc; i++) {
        if      (!strcmp(argv[i], "--bucket-start") && i + 1 < argc)
            bucket_start = (int)parse_positive_int("--bucket-start", argv[++i]);
        else if (!strcmp(argv[i], "--bucket-end")   && i + 1 < argc)
            bucket_end   = (int)parse_positive_int("--bucket-end",   argv[++i]);
        else if (!strcmp(argv[i], "--batch-size")   && i + 1 < argc)
            batch_size   = parse_positive_int("--batch-size", argv[++i]);
        else if (!strcmp(argv[i], "--dry-run"))
            dry_run      = 1;
        else if (!strcmp(argv[i], "--count-only"))
            count_only   = 1;
        else if (!strcmp(argv[i], "--output")       && i + 1 < argc)
            output_file  = argv[++i];
        else {
            fprintf(stderr, "Unknown argument: %s\n", argv[i]);
            return 1;
        }
    }

    /* Validate inputs */
    if (bucket_start < 1 || bucket_start > 64) {
        fprintf(stderr,
                "ERROR: --bucket-start must be in [1, 64], got %d\n",
                bucket_start);
        return 1;
    }
    if (bucket_end < 1 || bucket_end > 64) {
        fprintf(stderr,
                "ERROR: --bucket-end must be in [1, 64], got %d\n",
                bucket_end);
        return 1;
    }
    if (bucket_start > bucket_end) {
        fprintf(stderr,
                "ERROR: --bucket-start (%d) must be <= --bucket-end (%d)\n",
                bucket_start, bucket_end);
        return 1;
    }
    if (batch_size < 1) {
        fprintf(stderr, "ERROR: --batch-size must be >= 1\n");
        return 1;
    }

    /* Enumerate all batches across the selected bucket range. */
    BatchVec matrix = {NULL, 0, 0};
    for (int n = bucket_start; n <= bucket_end; n++) {
        BatchVec bv = split_bucket_into_batches(n, batch_size);
        for (size_t j = 0; j < bv.size; j++)
            batch_push(&matrix, &bv.data[j]);
        free(bv.data);
    }

    /* --count-only: print the batch count and exit. */
    if (count_only) {
        printf("%zu\n", matrix.size);
        free(matrix.data);
        return 0;
    }

    /* --dry-run: print a human-readable plan and exit. */
    if (dry_run) {
        printf("bucket_start  : %d\n", bucket_start);
        printf("bucket_end    : %d\n", bucket_end);
        printf("batch_size    : %zu\n", batch_size);
        printf("total batches : %zu\n", matrix.size);
        printf("\n");
        for (size_t i = 0; i < matrix.size; i++) {
            const Batch *b = &matrix.data[i];
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
        if (matrix.size > GITHUB_MATRIX_MAX) {
            fprintf(stderr,
                    "\nWARNING: %zu batches exceeds the GitHub Actions matrix"
                    " limit of %d.  Narrow the bucket range or increase"
                    " --batch-size.\n",
                    matrix.size, GITHUB_MATRIX_MAX);
        }
        free(matrix.data);
        return 0;
    }

    /* Normal mode: validate matrix size and output JSON. */
    if (matrix.size > GITHUB_MATRIX_MAX) {
        fprintf(stderr,
                "ERROR: %zu batches exceeds the GitHub Actions matrix limit"
                " of %d. Narrow the bucket range or increase --batch-size.\n",
                matrix.size, GITHUB_MATRIX_MAX);
        free(matrix.data);
        return 1;
    }

    FILE *out = stdout;
    if (output_file) {
        out = fopen(output_file, "w");
        if (!out) { perror(output_file); free(matrix.data); return 1; }
    }

    print_json_matrix(out, &matrix);

    if (output_file) {
        fclose(out);
        fprintf(stderr, "Wrote %zu batch entries to %s\n",
                matrix.size, output_file);
    }

    free(matrix.data);
    return 0;
}
