// bench/microbench_fft.cpp – FFT Lucas–Lehmer microbenchmark.
//
// Includes src/BigNum.cpp with BIGNUM_NO_MAIN to access all backend classes
// without pulling in main() or the discover-mode infrastructure.
//
// Measures, for a fixed exponent p that exercises FftMersenneBackend:
//   (a) total wall-clock time for MICROBENCH_ITERS (or p-2 if MICROBENCH_ITERS == 0) LL iterations
//   (b) max_roundoff accumulated across all iterations
//   (c) whether the final residue matches the expected result for a full run
//
// Build:
//   make microbench                        (default p=44497, full p-2 iterations)
//   make microbench MICROBENCH_ITERS=500   (partial run, no residue check)
//
// Usage:
//   ./bin/microbench_fft [p]
//   p defaults to 44497 (known Mersenne prime exponent, exercises FFT backend).

#define BIGNUM_NO_MAIN
#include "../src/BigNum.cpp"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>

// ---------------------------------------------------------------------------
// Benchmark parameters
// ---------------------------------------------------------------------------

// Default exponent: p=44497 is a known Mersenne prime exponent.
// It exceeds kLimbFftCrossover (~4000), so FftMersenneBackend is used.
static constexpr uint32_t kDefaultP = 44497;

// Maximum supported exponent for the command-line argument.
// Keep this aligned with the library-wide LL_MAX_EXPONENT default
// (currently 200,000,000 in BigNum.cpp). This is independent of the
// LL_LIMB_FFT_CROSSOVER environment-variable sanity cap, which is limited
// to 1,000,000 in BigNum.cpp.
static constexpr long kMaxSupportedExponent = 200000000L;

// Number of LL iterations to run.  Defaults to p-2 (full test).
// Override at compile time: -DMICROBENCH_ITERS=N
#ifndef MICROBENCH_ITERS
#  define MICROBENCH_ITERS 0u  // 0 = use p-2 (full run)
#endif

// Known expected result for the full LL test on p=44497 (Mersenne prime).
static constexpr bool        kExpectedIsPrime = true;
static constexpr const char* kExpectedResidue = "0000000000000000";

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    // Optional: override p from command line.
    uint32_t p = kDefaultP;
    if (argc >= 2) {
        char* end = nullptr;
        const long v = std::strtol(argv[1], &end, 10);
        if (end == argv[1] || *end != '\0' || v < 2 || v > kMaxSupportedExponent) {
            std::fprintf(stderr,
                "Usage: %s [p]\n"
                "  p must be an integer in [2, %ld]\n",
                argv[0], kMaxSupportedExponent);
            return 2;
        }
        p = static_cast<uint32_t>(v);
    }

    // Determine iteration count.
    const uint32_t p_minus2   = p - 2u;
    const uint32_t bench_iters =
        (MICROBENCH_ITERS == 0u || MICROBENCH_ITERS >= p_minus2)
            ? p_minus2
            : static_cast<uint32_t>(MICROBENCH_ITERS);
    const bool full_run = (bench_iters == p_minus2);

    // Lucas–Lehmer for p=2 has zero iterations (p-2 == 0), so a per-iteration
    // microbenchmark is not meaningful and would lead to division by zero when
    // computing ns/iter. Require at least one LL iteration (p >= 3).
    if (bench_iters == 0u) {
        std::fprintf(stderr,
            "Error: p=%u yields zero Lucas-Lehmer iterations (p-2 == 0). "
            "microbench_fft requires p >= 3.\n",
            p);
        return 2;
    }

    std::printf("=== microbench_fft ===\n");
    std::printf("p                : %u\n", p);
    std::printf("iters            : %u%s\n", bench_iters, full_run ? " (full, p-2)" : " (partial)");
    std::printf("backend          : FftMersenneBackend\n");
    // Report intra-FFT thread configuration.
    {
        const char* fft_threads_env = std::getenv("LL_FFT_THREADS");
        const char* outer_env       = std::getenv("LL_THREADS");
        const unsigned fft_t = fft_threads_env ? static_cast<unsigned>(std::atoi(fft_threads_env)) : 1u;
        const unsigned outer = outer_env        ? static_cast<unsigned>(std::atoi(outer_env))        : 1u;
        std::printf("LL_FFT_THREADS   : %u (outer LL_THREADS=%u)\n", fft_t, outer);
    }
    std::fflush(stdout);

    // Ensure p is large enough to exercise the FFT backend.
    // The crossover threshold is approximately 4000 (the default value of
    // kLimbFftCrossover in BigNum.cpp; adjustable via LL_LIMB_FFT_CROSSOVER).
    // This check is advisory: FftMersenneBackend::init() may still fail for some p
    // if no suitable FFT length/plan is available under the current configuration.
    if (p < 4000u) {
        std::fprintf(stderr,
            "Warning: p=%u may not exercise FftMersenneBackend "
            "(crossover ~4000; see kLimbFftCrossover in BigNum.cpp). "
            "Use p >= 4000 for FFT coverage.\n", p);
    }

    // -----------------------------------------------------------------------
    // Initialise FFT backend state (allocates twiddle tables, DWT weights, etc.)
    // -----------------------------------------------------------------------
    backend::FftMersenneBackend::State st;
    try {
        st = backend::FftMersenneBackend::init(p);
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "FftMersenneBackend::init failed: %s\n", ex.what());
        return 1;
    }

    // -----------------------------------------------------------------------
    // Hot loop: run bench_iters LL steps and measure wall-clock time.
    // -----------------------------------------------------------------------
    const auto t0 = std::chrono::steady_clock::now();

    for (uint32_t i = 0u; i < bench_iters; ++i) {
        backend::FftMersenneBackend::step(st);
    }

    const auto t1 = std::chrono::steady_clock::now();

    // -----------------------------------------------------------------------
    // Collect results
    // -----------------------------------------------------------------------
    const double elapsed_s    = std::chrono::duration<double>(t1 - t0).count();
    const double ns_per_iter  = elapsed_s * 1.0e9 / static_cast<double>(bench_iters);
    const double max_roundoff = backend::FftMersenneBackend::max_roundoff(st);
    const bool   is_zero      = backend::FftMersenneBackend::is_zero(st);
    const std::string residue = backend::FftMersenneBackend::residue_hex(st);

    std::printf("\n--- results ---\n");
    std::printf("total time       : %.3f s\n", elapsed_s);
    std::printf("time/iter        : %.1f ns\n", ns_per_iter);
    std::printf("max_roundoff     : %.6f\n", max_roundoff);
    std::printf("is_zero          : %s\n", is_zero ? "true" : "false");
    std::printf("residue (hex)    : %s\n", residue.c_str());

    // -----------------------------------------------------------------------
    // (c) Residue validation: compare against expected for the full run.
    //     For p=kDefaultP=44497, the full LL test must yield is_prime=true.
    //     For other p values, rely on is_zero as a basic sanity check when
    //     the exponent is in the known Mersenne-prime list.
    // -----------------------------------------------------------------------
    int exit_code = 0;

    if (full_run) {
        // Only compare against the hardcoded expected residue for the default p.
        if (p == kDefaultP) {
            const bool prime_ok   = (is_zero == kExpectedIsPrime);
            const bool residue_ok = (residue == kExpectedResidue);

            std::printf("\n--- validation (p=%u is a known Mersenne prime) ---\n", p);
            std::printf("is_prime match   : %s  (expected=%s  got=%s)\n",
                        prime_ok   ? "PASS" : "FAIL",
                        kExpectedIsPrime ? "true" : "false",
                        is_zero          ? "true" : "false");
            std::printf("residue match    : %s  (expected=%s  got=%s)\n",
                        residue_ok ? "PASS" : "FAIL",
                        kExpectedResidue, residue.c_str());

            const bool all_ok = prime_ok && residue_ok;
            std::printf("\nOverall          : %s\n", all_ok ? "PASS" : "FAIL");
            exit_code = all_ok ? 0 : 1;
        } else {
            // For user-specified p, report the result but do not fail.
            std::printf("\n(no hardcoded expected value for p=%u; "
                        "result is_prime=%s)\n",
                        p, is_zero ? "true" : "false");
        }
    } else {
        std::printf("\n(partial run: %u/%u iters – residue comparison skipped)\n",
                    bench_iters, p_minus2);
    }

    return exit_code;
}
