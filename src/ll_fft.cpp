/*
    ll_fft.cpp

    ======================================================================
    OBJETIVO
    ======================================================================

    Dado n, calcular exactamente:

        res(n) = Ceil((2 + sqrt(3))^(2^n)) mod (2^(n+2) - 1)

    usando la equivalencia exacta con el residuo de Lucas-Lehmer.

    Si definimos:
        p   = n + 2
        M_p = 2^p - 1

    entonces:
        res(n) = s_{p-2} mod M_p

    donde:
        s_0 = 4
        s_{k+1} = s_k^2 - 2 mod M_p

    ======================================================================
    PIPELINE
    ======================================================================

    1) Calcular p = n + 2
    2) Comprobar si p es primo
       - Si no lo es, M_p es compuesto
    3) Buscar factores pequeños/medianos de M_p usando la forma especial
       de los divisores primos de números de Mersenne:
           q = 2*k*p + 1
           q ≡ ±1 (mod 8)
    4) Si sobrevive, ejecutar Lucas-Lehmer
    5) Imprimir el residuo exacto

    ======================================================================
    FUNDAMENTO MATEMÁTICO
    ======================================================================

    Sea:
        alpha = 2 + sqrt(3)
        beta  = 2 - sqrt(3)

    Entonces:
        alpha * beta = 1
        0 < beta < 1

    Definimos:
        u_m = alpha^m + beta^m

    Como 0 < beta^m < 1 para m >= 1, se cumple:
        Ceil(alpha^m) = u_m

    Además:
        u_{2m} = u_m^2 - 2

    Si definimos:
        s_0 = 4
        s_{k+1} = s_k^2 - 2

    entonces:
        s_k = u_{2^k}

    Por tanto:
        Ceil((2 + sqrt(3))^(2^n)) = s_n

    y el residuo buscado es:
        s_n mod (2^(n+2)-1)

    Como p = n+2, esto es exactamente:
        s_{p-2} mod (2^p - 1)

    ======================================================================
    IMPLEMENTACIÓN
    ======================================================================

        Esta versión usa:
            - uint64 para el prefiltro (primalidad de p y factores pequeños)
            - aritmética de limbs propia para el paso Lucas-Lehmer:
                    * multiplicación/square sobre arrays de limbs
                    * buffers fijos de limbs
                    * reducción específica módulo 2^p - 1

    ======================================================================
    LÍMITE REAL DE ESTA VERSIÓN
    ======================================================================

    Esta implementación es exacta y más seria que una basada solo en enteros
    nativos,
    pero no es un motor tipo GIMPS.

    El siguiente salto real para exponentes gigantes sería:
      - multiplicación FFT especializada
      - representación aún más diseñada para 2^p - 1
      - control/corrección numérica
      - tuning fino por CPU y cachés

    ======================================================================
    COMPILACIÓN
    ======================================================================

        g++ -O3 -std=c++17 -Wall -Wextra -o ll_fft ll_fft.cpp
*/

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <errno.h>
#include <inttypes.h>

typedef uint64_t mp_limb_t;
typedef size_t mp_size_t;
typedef uint64_t mp_bitcnt_t;
enum { GMP_NUMB_BITS = 8 * (int)sizeof(mp_limb_t) };

/* ====================================================================== */
/* Opciones                                                                */
/* ====================================================================== */

typedef struct {
    unsigned long n;
    uint64_t factor_limit;
    int output_hex;
    int force_ll;
    int selftest;
} options_t;

typedef struct {
    mp_bitcnt_t p;         /* exponente del número de Mersenne: M = 2^p - 1 */
    mp_size_t   L;         /* número de limbs necesarios para p bits */
    unsigned    rem_bits;  /* p mod GMP_NUMB_BITS */
    mp_limb_t   top_mask;  /* máscara del último limb */
} mersenne_ctx_t;

/* ====================================================================== */
/* Utilidades generales                                                    */
/* ====================================================================== */

static void die(const char *msg) {
    fprintf(stderr, "%s\n", msg);
    exit(EXIT_FAILURE);
}

template <typename T>
static T *xcalloc(size_t count) {
    T *ptr = static_cast<T *>(calloc(count, sizeof(T)));
    if (!ptr) die("Error de memoria.");
    return ptr;
}

static int parse_ulong_arg(const char *s, unsigned long *out) {
    char *end = NULL;
    errno = 0;
    unsigned long v = strtoul(s, &end, 10);
    if (errno != 0 || s == end || *end != '\0') return 0;
    *out = v;
    return 1;
}

static int parse_u64_arg(const char *s, uint64_t *out) {
    char *end = NULL;
    errno = 0;
    unsigned long long v = strtoull(s, &end, 10);
    if (errno != 0 || s == end || *end != '\0') return 0;
    *out = (uint64_t)v;
    return 1;
}

static void parse_args(int argc, char **argv, options_t *opt) {
    if (argc < 2) {
        fprintf(stderr,
            "Uso: %s <p> [--factor-limit N] [--hex] [--force-ll] [--selftest]\n",
            argv[0]);
        exit(EXIT_FAILURE);
    }

    if (!parse_ulong_arg(argv[1], &opt->n)) {
        die("Error: p debe ser un entero positivo válido.");
    }

    opt->factor_limit = 0;
    opt->output_hex = 0;
    opt->force_ll = 0;
    opt->selftest = 0;

    for (int i = 2; i < argc; ++i) {
        if (strcmp(argv[i], "--hex") == 0) {
            opt->output_hex = 1;
        } else if (strcmp(argv[i], "--force-ll") == 0) {
            opt->force_ll = 1;
        } else if (strcmp(argv[i], "--selftest") == 0) {
            opt->selftest = 1;
        } else if (strcmp(argv[i], "--factor-limit") == 0) {
            if (i + 1 >= argc) die("Error: falta valor tras --factor-limit");
            if (!parse_u64_arg(argv[i + 1], &opt->factor_limit)) {
                die("Error: valor inválido para --factor-limit");
            }
            ++i;
        } else {
            fprintf(stderr, "Opción desconocida: %s\n", argv[i]);
            exit(EXIT_FAILURE);
        }
    }
}

static mp_limb_t top_mask_for_bits(unsigned rem_bits) {
    if (rem_bits == 0) return (mp_limb_t)(-1);
    return (((mp_limb_t)1) << rem_bits) - 1;
}

static void ctx_init(mersenne_ctx_t *ctx, uint64_t p) {
    ctx->p = (mp_bitcnt_t)p;
    ctx->L = (mp_size_t)((ctx->p + GMP_NUMB_BITS - 1) / GMP_NUMB_BITS);
    ctx->rem_bits = (unsigned)(ctx->p % GMP_NUMB_BITS);
    ctx->top_mask = top_mask_for_bits(ctx->rem_bits);
}

/* ====================================================================== */
/* Aritmética modular uint64 para prefiltros                               */
/* ====================================================================== */

static uint64_t mul_mod_u64(uint64_t a, uint64_t b, uint64_t mod) {
#if defined(__SIZEOF_INT128__)
    __uint128_t z = (__uint128_t)a * (__uint128_t)b;
    return (uint64_t)(z % mod);
#else
    uint64_t res = 0;
    a %= mod;
    while (b) {
        if (b & 1) {
            res = (res >= mod - a) ? (res + a - mod) : (res + a);
        }
        b >>= 1;
        if (b) {
            a = (a >= mod - a) ? (a + a - mod) : (a + a);
        }
    }
    return res;
#endif
}

static uint64_t pow_mod_u64(uint64_t base, uint64_t exp, uint64_t mod) {
    uint64_t result = 1 % mod;
    base %= mod;

    while (exp) {
        if (exp & 1) result = mul_mod_u64(result, base, mod);
        exp >>= 1;
        if (exp) base = mul_mod_u64(base, base, mod);
    }
    return result;
}

/* ====================================================================== */
/* Test de primalidad determinista para uint64                             */
/* ====================================================================== */

static int is_prime_u64(uint64_t n) {
    static const uint64_t small_primes[] = {
        2ULL, 3ULL, 5ULL, 7ULL, 11ULL, 13ULL, 17ULL,
        19ULL, 23ULL, 29ULL, 31ULL, 37ULL
    };

    if (n < 2) return 0;
    for (size_t i = 0; i < sizeof(small_primes) / sizeof(small_primes[0]); ++i) {
        uint64_t p = small_primes[i];
        if (n == p) return 1;
        if (n % p == 0) return 0;
    }

    /* n - 1 = d * 2^s, con d impar */
    uint64_t d = n - 1;
    unsigned s = 0;
    while ((d & 1) == 0) {
        d >>= 1;
        ++s;
    }

    /*
        Bases deterministas válidas para todo uint64.
    */
    static const uint64_t bases[] = {
        2ULL, 325ULL, 9375ULL, 28178ULL,
        450775ULL, 9780504ULL, 1795265022ULL
    };

    for (size_t i = 0; i < sizeof(bases) / sizeof(bases[0]); ++i) {
        uint64_t a = bases[i] % n;
        if (a == 0) continue;

        uint64_t x = pow_mod_u64(a, d, n);
        if (x == 1 || x == n - 1) continue;

        int witness = 1;
        for (unsigned r = 1; r < s; ++r) {
            x = mul_mod_u64(x, x, n);
            if (x == n - 1) {
                witness = 0;
                break;
            }
        }
        if (witness) return 0;
    }

    return 1;
}

/* ====================================================================== */
/* Búsqueda dirigida de factores de M_p                                    */
/* ====================================================================== */

/*
    Si q | (2^p - 1) con p primo impar, entonces:
        q = 2*k*p + 1
        q ≡ ±1 (mod 8)

    Recorremos esa forma hasta factor_limit.
*/
static int find_factor_of_mersenne(uint64_t p, uint64_t factor_limit, uint64_t *factor_out) {
    if (factor_limit < 3 || p < 2) return 0;

    for (uint64_t k = 1;; ++k) {
        if (k > (UINT64_MAX - 1) / (2 * p)) break;

        uint64_t q = 2 * k * p + 1;
        if (q > factor_limit) break;

        uint64_t r8 = q & 7ULL;
        if (!(r8 == 1 || r8 == 7)) continue;
        if (!is_prime_u64(q)) continue;

        if (pow_mod_u64(2ULL, p, q) == 1ULL) {
            *factor_out = q;
            return 1;
        }
    }

    return 0;
}

/* ====================================================================== */
/* Operaciones básicas sobre limbs                                         */
/* ====================================================================== */

static void limbs_zero(mp_limb_t *dst, mp_size_t n) {
    memset(dst, 0, (size_t)n * sizeof(mp_limb_t));
}

static void limbs_copy(mp_limb_t *dst, const mp_limb_t *src, mp_size_t n) {
    memcpy(dst, src, (size_t)n * sizeof(mp_limb_t));
}

static int limbs_is_zero(const mp_limb_t *x, mp_size_t n) {
    for (mp_size_t i = 0; i < n; ++i) {
        if (x[i] != 0) return 0;
    }
    return 1;
}

static mp_limb_t limbs_add_n(
    mp_limb_t *dst,
    const mp_limb_t *a,
    const mp_limb_t *b,
    mp_size_t n
) {
#if defined(__SIZEOF_INT128__)
    __uint128_t carry = 0;
    for (mp_size_t i = 0; i < n; ++i) {
        __uint128_t sum = (__uint128_t)a[i] + (__uint128_t)b[i] + carry;
        dst[i] = (mp_limb_t)sum;
        carry = sum >> GMP_NUMB_BITS;
    }
    return (mp_limb_t)carry;
#else
#error "Se requiere __int128 para la aritmetica de limbs en esta version."
#endif
}

static void limbs_mul(
    mp_limb_t *out,
    const mp_limb_t *a,
    const mp_limb_t *b,
    mp_size_t n
) {
    limbs_zero(out, 2 * n);

#if defined(__SIZEOF_INT128__)
    for (mp_size_t i = 0; i < n; ++i) {
        __uint128_t carry = 0;
        for (mp_size_t j = 0; j < n; ++j) {
            mp_size_t k = i + j;
            __uint128_t sum = (__uint128_t)a[i] * (__uint128_t)b[j]
                            + (__uint128_t)out[k]
                            + carry;
            out[k] = (mp_limb_t)sum;
            carry = sum >> GMP_NUMB_BITS;
        }

        mp_size_t k = i + n;
        while (carry != 0 && k < 2 * n) {
            __uint128_t sum = (__uint128_t)out[k] + carry;
            out[k] = (mp_limb_t)sum;
            carry = sum >> GMP_NUMB_BITS;
            ++k;
        }
    }
#else
#error "Se requiere __int128 para la aritmetica de limbs en esta version."
#endif
}

/* ====================================================================== */
/* Comparación con M = 2^p - 1                                             */
/* ====================================================================== */

static int eq_mersenne(const mp_limb_t *x, const mersenne_ctx_t *ctx) {
    for (mp_size_t i = 0; i + 1 < ctx->L; ++i) {
        if (x[i] != (mp_limb_t)(-1)) return 0;
    }
    return x[ctx->L - 1] == ctx->top_mask;
}

static int ge_mersenne(const mp_limb_t *x, const mersenne_ctx_t *ctx) {
    for (mp_size_t i = ctx->L; i-- > 0;) {
        mp_limb_t m = (i == ctx->L - 1) ? ctx->top_mask : (mp_limb_t)(-1);
        if (x[i] > m) return 1;
        if (x[i] < m) return 0;
    }
    return 1;
}

/*
    x <- x - M, asumiendo x >= M y M = 2^p - 1.
*/
static void sub_mersenne(mp_limb_t *x, const mersenne_ctx_t *ctx) {
    mp_limb_t borrow = 0;

    for (mp_size_t i = 0; i + 1 < ctx->L; ++i) {
        mp_limb_t xi = x[i];
        mp_limb_t yi = (mp_limb_t)(-1) + borrow;
        x[i] = xi - yi;
        borrow = (xi < yi);
    }

    {
        mp_limb_t xi = x[ctx->L - 1];
        mp_limb_t yi = ctx->top_mask + borrow;
        x[ctx->L - 1] = xi - yi;
    }
}

/* ====================================================================== */
/* Normalización módulo 2^p - 1                                            */
/* ====================================================================== */

/*
    Dejamos x en [0, M-1], con M = 2^p - 1.

    Si p no es múltiplo de GMP_NUMB_BITS, puede haber bits sobrantes en el
    limb superior. Esos bits se pliegan abajo usando:
        2^p ≡ 1 (mod 2^p - 1)
*/
static void normalize_mod_mersenne(mp_limb_t *x, const mersenne_ctx_t *ctx) {
    if (ctx->rem_bits != 0) {
        for (;;) {
            mp_limb_t extra = x[ctx->L - 1] >> ctx->rem_bits;
            x[ctx->L - 1] &= ctx->top_mask;

            if (extra == 0) break;

            mp_limb_t carry = extra;
            for (mp_size_t i = 0; i < ctx->L && carry; ++i) {
                mp_limb_t old = x[i];
                x[i] += carry;
                carry = (x[i] < old);
            }
        }
    }

    if (eq_mersenne(x, ctx)) {
        limbs_zero(x, ctx->L);
        return;
    }

    while (ge_mersenne(x, ctx)) {
        sub_mersenne(x, ctx);
    }
}

/* ====================================================================== */
/* x <- x - 2 mod M                                                        */
/* ====================================================================== */

static void add_mersenne_inplace(mp_limb_t *x, const mersenne_ctx_t *ctx) {
    mp_limb_t carry = 0;

    for (mp_size_t i = 0; i + 1 < ctx->L; ++i) {
        mp_limb_t old = x[i];
        x[i] = old + (mp_limb_t)(-1) + carry;
        carry = (x[i] < old) || (carry && x[i] == old);
    }

    {
        mp_limb_t old = x[ctx->L - 1];
        x[ctx->L - 1] = old + ctx->top_mask + carry;
    }

    normalize_mod_mersenne(x, ctx);
}

static void sub2_mod_mersenne(mp_limb_t *x, const mersenne_ctx_t *ctx) {
    mp_limb_t borrow = 2;

    for (mp_size_t i = 0; i < ctx->L && borrow; ++i) {
        mp_limb_t old = x[i];
        x[i] = old - borrow;
        borrow = (old < borrow);
    }

    if (borrow) {
        add_mersenne_inplace(x, ctx);
    }
}

/* ====================================================================== */
/* Reducción de 2L limbs módulo 2^p - 1                                    */
/* ====================================================================== */

/*
    Reduce in (2L limbs) módulo M = 2^p - 1.

    Idea:
        in = low + 2^p * high
        in mod M == low + high mod M

    Si p cae en frontera de limb, el corte es directo.
    Si no, high empieza dentro del limb L-1 y hay que reconstruirlo.
*/
static void reduce_from_2L(
    mp_limb_t *out,            /* tamaño L */
    const mp_limb_t *in,       /* tamaño 2L */
    mp_limb_t *scratch,        /* tamaño L+1 */
    const mersenne_ctx_t *ctx
) {
    limbs_zero(scratch, ctx->L + 1);

    if (ctx->rem_bits == 0) {
        mp_limb_t cy = limbs_add_n(scratch, in, in + ctx->L, ctx->L);
        scratch[ctx->L] = cy;
    } else {
        const unsigned s = ctx->rem_bits;
        const unsigned rs = GMP_NUMB_BITS - s;

        /* low */
        for (mp_size_t i = 0; i + 1 < ctx->L; ++i) {
            scratch[i] = in[i];
        }
        scratch[ctx->L - 1] = in[ctx->L - 1] & ctx->top_mask;
        scratch[ctx->L] = 0;

        /* high, reconstruido y sumado a scratch */
        mp_limb_t carry = 0;
        for (mp_size_t i = 0; i < ctx->L; ++i) {
            mp_limb_t lo = in[ctx->L - 1 + i] >> s;
            mp_limb_t hi = 0;

            if (ctx->L - 1 + i + 1 < 2 * ctx->L) {
                hi = in[ctx->L - 1 + i + 1] << rs;
            }

            mp_limb_t h = lo | hi;

            mp_limb_t old = scratch[i];
            scratch[i] = old + h + carry;
            carry = (scratch[i] < old) || (carry && scratch[i] == old);
        }
        scratch[ctx->L] += carry;
    }

    /*
        Plegado final del limb extra.
    */
    while (scratch[ctx->L] != 0) {
        mp_limb_t carry = scratch[ctx->L];
        scratch[ctx->L] = 0;

        for (mp_size_t i = 0; i < ctx->L && carry; ++i) {
            mp_limb_t old = scratch[i];
            scratch[i] = old + carry;
            carry = (scratch[i] < old);
        }
        scratch[ctx->L] += carry;
    }

    limbs_copy(out, scratch, ctx->L);
    normalize_mod_mersenne(out, ctx);
}

/* ====================================================================== */
/* Iteración Lucas-Lehmer con limbs propios                                 */
/* ====================================================================== */

static void ll_iteration_mpn(
    mp_limb_t *state,          /* L limbs */
    mp_limb_t *square_buf,     /* 2L limbs */
    mp_limb_t *scratch,        /* L+1 limbs */
    const mersenne_ctx_t *ctx
) {
    limbs_mul(square_buf, state, state, ctx->L);
    reduce_from_2L(state, square_buf, scratch, ctx);
    sub2_mod_mersenne(state, ctx);
}

static void lucas_lehmer_residue_mpn(mp_limb_t *res, uint64_t p) {
    mersenne_ctx_t ctx;
    ctx_init(&ctx, p);

    mp_limb_t *state   = xcalloc<mp_limb_t>((size_t)ctx.L);
    mp_limb_t *sqbuf   = xcalloc<mp_limb_t>((size_t)(2 * ctx.L));
    mp_limb_t *scratch = xcalloc<mp_limb_t>((size_t)(ctx.L + 1));

    /*
        Caso especial p = 2:
            M_2 = 3
            residuo = s_0 mod 3 = 4 mod 3 = 1
    */
    if (p == 2) {
        limbs_zero(res, ctx.L);
        res[0] = 1;
        free(state);
        free(sqbuf);
        free(scratch);
        return;
    }

    state[0] = 4;
    normalize_mod_mersenne(state, &ctx);

    for (uint64_t i = 0; i < p - 2; ++i) {
        ll_iteration_mpn(state, sqbuf, scratch, &ctx);
    }

    limbs_copy(res, state, ctx.L);

    free(state);
    free(sqbuf);
    free(scratch);
}

/* ====================================================================== */
/* Conversión / impresión                                                  */
/* ====================================================================== */

static uint32_t divmod_u32_inplace(mp_limb_t *x, mp_size_t L, uint32_t div) {
#if defined(__SIZEOF_INT128__)
    __uint128_t rem = 0;
    for (mp_size_t i = L; i-- > 0;) {
        __uint128_t cur = (rem << GMP_NUMB_BITS) | x[i];
        x[i] = (mp_limb_t)(cur / div);
        rem = cur % div;
    }
    return (uint32_t)rem;
#else
#error "Se requiere __int128 para conversion decimal en esta version."
#endif
}

static void print_limbs_decimal(const mp_limb_t *x, mp_size_t L) {
    if (limbs_is_zero(x, L)) {
        printf("Residuo (dec): 0\n");
        return;
    }

    mp_limb_t *tmp = xcalloc<mp_limb_t>((size_t)L);
    limbs_copy(tmp, x, L);

    const uint32_t base10 = 1000000000U;
    size_t max_chunks = (size_t)((GMP_NUMB_BITS * L) / 29U) + 3U;
    uint32_t *chunks = xcalloc<uint32_t>(max_chunks);
    size_t chunks_len = 0;

    while (!limbs_is_zero(tmp, L)) {
        if (chunks_len >= max_chunks) {
            free(tmp);
            free(chunks);
            die("Error interno convirtiendo a decimal.");
        }
        chunks[chunks_len++] = divmod_u32_inplace(tmp, L, base10);
    }

    printf("Residuo (dec): %u", chunks[chunks_len - 1]);
    for (size_t i = chunks_len - 1; i-- > 0;) {
        printf("%09u", chunks[i]);
    }
    printf("\n");

    free(tmp);
    free(chunks);
}

static void print_limbs_hex(const mp_limb_t *x, mp_size_t L) {
    if (limbs_is_zero(x, L)) {
        printf("Residuo (hex): 0\n");
        return;
    }

    mp_size_t hi = L;
    while (hi > 0 && x[hi - 1] == 0) --hi;

    printf("Residuo (hex): %" PRIx64, (uint64_t)x[hi - 1]);
    for (mp_size_t i = hi - 1; i-- > 0;) {
        printf("%016" PRIx64, (uint64_t)x[i]);
    }
    printf("\n");
}

/* ====================================================================== */
/* Versión de referencia uint64 para selftest                              */
/* ====================================================================== */

static uint64_t lucas_lehmer_residue_u64(uint64_t p) {
    if (p == 2) return 1;

    uint64_t M = (1ULL << p) - 1ULL;
    uint64_t s = 4ULL % M;

    for (uint64_t i = 0; i < p - 2; ++i) {
#if defined(__SIZEOF_INT128__)
        __uint128_t z = (__uint128_t)s * (__uint128_t)s;
        s = (uint64_t)(z % M);
#else
        s = mul_mod_u64(s, s, M);
#endif
        s = (s >= 2ULL) ? (s - 2ULL) : (s + M - 2ULL);
    }

    return s;
}

/* ====================================================================== */
/* Selftest                                                                */
/* ====================================================================== */

static void run_selftest(void) {
    const uint64_t tests[] = {2, 3, 5, 7, 11, 13, 17, 19, 31, 61};
    const size_t nt = sizeof(tests) / sizeof(tests[0]);

    for (size_t i = 0; i < nt; ++i) {
        uint64_t p = tests[i];
        mersenne_ctx_t ctx;
        ctx_init(&ctx, p);

        mp_limb_t *res_mpn = xcalloc<mp_limb_t>((size_t)ctx.L);

        lucas_lehmer_residue_mpn(res_mpn, p);

        uint64_t ref = lucas_lehmer_residue_u64(p);
        if (res_mpn[0] != (mp_limb_t)ref) {
            fprintf(stderr,
                    "Selftest FAILED para p=%" PRIu64 "\nEsperado (u64): %" PRIu64 "\nObtenido limb0: %" PRIu64 "\n",
                    p, ref, (uint64_t)res_mpn[0]);
            free(res_mpn);
            exit(EXIT_FAILURE);
        }

        for (mp_size_t k = 1; k < ctx.L; ++k) {
            if (res_mpn[k] != 0) {
                fprintf(stderr, "Selftest FAILED para p=%" PRIu64 " (limb alto no nulo)\n", p);
                free(res_mpn);
                exit(EXIT_FAILURE);
            }
        }

        if (p == 3 || p == 5 || p == 7 || p == 13 || p == 17 || p == 19 || p == 31 || p == 61) {
            if (res_mpn[0] != 0) {
                fprintf(stderr, "Selftest FAILED para p=%" PRIu64 " (esperado residuo 0)\n", p);
                free(res_mpn);
                exit(EXIT_FAILURE);
            }
        }
        if (p == 11 && res_mpn[0] == 0) {
            fprintf(stderr, "Selftest FAILED para p=11 (M_11 es compuesto, residuo no debe ser 0)\n");
            free(res_mpn);
            exit(EXIT_FAILURE);
        }

        free(res_mpn);
    }

    fprintf(stderr, "Selftest OK.\n");
}

/* ====================================================================== */
/* main                                                                    */
/* ====================================================================== */

int main(int argc, char **argv) {
    options_t opt;
    parse_args(argc, argv, &opt);

    if (opt.selftest) {
        run_selftest();
    }

    uint64_t p = (uint64_t)opt.n;   /* opt.n holds the Mersenne exponent parsed from argv[1] */

    if (p < 2) {
        die("Error: p debe ser >= 2.");
    }

    printf("p = %" PRIu64 "\n", p);

    int p_is_prime = is_prime_u64(p);
    printf("p es %s\n", p_is_prime ? "primo" : "compuesto");

    int skip_ll = 0;

    if (!p_is_prime) {
        printf("Como p no es primo, M_p = 2^p - 1 es compuesto.\n");
        if (!opt.force_ll) {
            printf("Se omite Lucas-Lehmer. Usa --force-ll para forzarlo.\n");
            skip_ll = 1;
        }
    }

    if (opt.factor_limit > 0 && p_is_prime && p > 2) {
        uint64_t factor = 0;
        printf("Buscando factores q <= %" PRIu64 " de la forma q = 2*k*p + 1, q ≡ ±1 (mod 8)...\n",
               opt.factor_limit);

        if (find_factor_of_mersenne(p, opt.factor_limit, &factor)) {
            printf("Se encontró factor de M_p: q = %" PRIu64 "\n", factor);
            if (!opt.force_ll) {
                printf("Se omite Lucas-Lehmer. Usa --force-ll para forzarlo.\n");
                skip_ll = 1;
            }
        } else {
            printf("No se encontró factor en el rango probado.\n");
        }
    }

    if (!skip_ll) {
        mersenne_ctx_t ctx;
        ctx_init(&ctx, p);

        mp_limb_t *res = xcalloc<mp_limb_t>((size_t)ctx.L);
        printf("Calculando residuo Lucas-Lehmer exacto con limbs...\n");

        lucas_lehmer_residue_mpn(res, p);

        if (opt.output_hex) {
            print_limbs_hex(res, ctx.L);
        } else {
            print_limbs_decimal(res, ctx.L);
        }

        printf("Residuo %s 0\n", limbs_is_zero(res, ctx.L) ? "=" : "!=");

        free(res);
    }

    return 0;
}