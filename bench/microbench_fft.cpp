// bench/microbench_fft.cpp – FFT Lucas–Lehmer microbenchmark.
//
// Includes src/BigNum.cpp with BIGNUM_NO_MAIN to access all backend classes
// without pulling in main() or the discover-mode infrastructure.
//
// Measures, for a fixed exponent p that exercises FftMersenneBackend:
//   (a) total wall-clock time for MICROBENCH_ITERS (or p-2 if MICROBENCH_ITERS == 0) LL iterations
//   (b) max_roundoff accumulated across all iterations
//   (c) whether the final residue matches the expected result for a full run
//   (d) theoretical Teraflops and MIPS for the measured configuration,
//       with extrapolation to a configurable cluster (default: 256 workers,
//       4 threads each, 2 FFT threads, nested).
//
// Build:
//   make microbench                        (default p=44497, full p-2 iterations)
//   make microbench MICROBENCH_ITERS=500   (partial run, no residue check)
//
// Usage:
//   ./bin/microbench_fft [p [iters]]
//   p     defaults to 44497 (known Mersenne prime exponent, exercises FFT backend).
//   iters overrides the iteration count at runtime (overrides compile-time
//         MICROBENCH_ITERS).  Useful for large exponents (e.g., p=756839)
//         where a full p-2 run is infeasible: pass 200 to measure ns/iter
//         and extrapolate the full runtime.
//
// Cluster power environment variables (override defaults):
//   LL_CLUSTER_WORKERS    – number of worker nodes  (default 256)
//   LL_CLUSTER_THREADS    – LL threads per worker   (default 4)
//   LL_CLUSTER_FFT_THREADS– FFT threads per worker  (default taken from LL_FFT_THREADS, fallback 2)

#define BIGNUM_NO_MAIN
#include "../src/BigNum.cpp"

#include <chrono>
#include <cmath>
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
    // Optional: override p from command line (argv[1]).
    uint32_t p = kDefaultP;
    if (argc >= 2) {
        char* end = nullptr;
        const long v = std::strtol(argv[1], &end, 10);
        if (end == argv[1] || *end != '\0' || v < 2 || v > kMaxSupportedExponent) {
            std::fprintf(stderr,
                "Usage: %s [p [iters]]\n"
                "  p     must be an integer in [2, %ld]\n"
                "  iters must be a positive integer (default: p-2, full LL test;\n"
                "        values > p-2 are clamped to p-2)\n",
                argv[0], kMaxSupportedExponent);
            return 2;
        }
        p = static_cast<uint32_t>(v);
    }

    // Optional: override iteration count from command line (argv[2]).
    // Overrides the compile-time MICROBENCH_ITERS default.
    // Useful for large exponents (e.g., p=756839) where a full p-2 run is
    // infeasible on a CI runner: pass a small count (e.g., 200) to measure
    // ns/iter and extrapolate the full runtime.
    uint32_t runtime_iters = 0u;
    if (argc >= 3) {
        char* end = nullptr;
        const long v = std::strtol(argv[2], &end, 10);
        if (end == argv[2] || *end != '\0' || v < 1) {
            std::fprintf(stderr,
                "Usage: %s [p [iters]]\n"
                "  p     must be an integer in [2, %ld]\n"
                "  iters must be a positive integer (default: p-2, full LL test;\n"
                "        values > p-2 are clamped to p-2)\n",
                argv[0], kMaxSupportedExponent);
            return 2;
        }
        runtime_iters = static_cast<uint32_t>(v);
    }

    // Determine iteration count:
    //   runtime_iters (argv[2]) takes precedence when provided.
    //   Values larger than p-2 are clamped to p-2 (treated as full run).
    //   Otherwise fall back to compile-time MICROBENCH_ITERS (0 = full p-2 run).
    const uint32_t p_minus2   = p - 2u;
    const uint32_t bench_iters =
        (runtime_iters > 0u)
            ? std::min(runtime_iters, p_minus2)
            : ((MICROBENCH_ITERS == 0u || MICROBENCH_ITERS >= p_minus2)
                ? p_minus2
                : static_cast<uint32_t>(MICROBENCH_ITERS));
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

    // -----------------------------------------------------------------------
    // (d) Cluster power report.
    //
    // FLOPs per LL iteration (Crandall-Bailey DWT/FFT, real-FFT optimisation):
    //
    //   The hot path in FftMersenneBackend::step() performs one
    //   square-sub-2-mod-Mersenne via a real-input convolution:
    //     1. DWT forward weighting:         n   multiplications
    //     2. Forward n/2-point complex FFT: 5*(n/2)*log2(n/2)  FLOPs
    //     3. Real-FFT post + pointwise sq:  ~4*(n/2) FLOPs
    //     4. Real-FFT pre  + IFFT n/2-pt:  5*(n/2)*log2(n/2) + 4*(n/2) FLOPs
    //     5. DWT inverse weighting + norm:  n   multiplications
    //     6. Carry propagation (x,floor,-): ~3*n FLOPs
    //
    //   Summing all components:
    //     n + 5*(n/2)*(log2(n)-1) + 2*n + 2*n + 5*(n/2)*(log2(n)-1) + n + 3*n
    //     = 5*n*log2(n) + 4*n
    //
    //   The formula used here is 5*n*log2(n), which omits the +4*n term.
    //   For p=44497 (n=4096, log2(n)=12): +4*n adds ~6.7% to the total.
    //   The simplified form is the standard FFT literature convention and
    //   is used here for consistency with published FLOP-rate benchmarks.
    //
    // Instruction-count model for MIPS (theoretical estimate):
    //   Modern CPUs with FMA execute one multiply-add per instruction (2 FLOPs).
    //   Adding memory and integer overhead (~0.8x the FP instruction count):
    //     instructions_per_iter ~= flops_per_iter / 2 * (1 + 0.8)
    //                            = flops_per_iter * 0.9
    //   This is a theoretical estimate.  Actual MIPS depends on micro-
    //   architecture, vectorisation width, cache behaviour, and branch
    //   prediction.  Profile with `perf stat` for hardware-measured values.
    //
    // Cluster extrapolation (theoretical upper bound):
    //   The benchmark is run on a single node.  Cluster power is computed by
    //   multiplying per-thread figures by workers * ll_threads, assuming each
    //   worker node achieves the same throughput (ideal linear scaling).
    //   Actual cluster performance will be lower due to memory bandwidth
    //   contention, interconnect latency, and OS scheduling jitter.
    // -----------------------------------------------------------------------

    // --- Read cluster configuration (env vars, with defaults) ---
    auto read_env_uint = [](const char* name, unsigned def) -> unsigned {
        const char* s = std::getenv(name);
        if (!s || *s == '\0') return def;
        char* end = nullptr;
        const long v = std::strtol(s, &end, 10);
        return (end != s && v > 0) ? static_cast<unsigned>(v) : def;
    };

    // LL_FFT_THREADS is already used by the FftTeam inside the backend.
    // Use it as the default for fft_threads_per_worker so reported config
    // matches the actual run.
    const unsigned fft_threads_env       = read_env_uint("LL_FFT_THREADS",         1u);
    const unsigned cluster_workers       = read_env_uint("LL_CLUSTER_WORKERS",    256u);
    const unsigned ll_threads_per_worker = read_env_uint("LL_CLUSTER_THREADS",      4u);
    const unsigned fft_threads_per_worker= read_env_uint("LL_CLUSTER_FFT_THREADS", fft_threads_env);

    // --- FLOPs per iteration (5*n*log2(n), standard FFT literature convention) ---
    const size_t   fft_n           = st.n;   // transform length (power of 2)
    const double   log2_n          = std::log2(static_cast<double>(fft_n));
    const double   flops_per_iter  = 5.0 * static_cast<double>(fft_n) * log2_n;

    // --- Single-thread throughput (measured) ---
    const double iters_per_sec     = 1.0e9 / ns_per_iter;
    const double gflops_per_thread = flops_per_iter * iters_per_sec / 1.0e9;

    // --- Per-node throughput ---
    // ll_threads_per_worker exponents run in parallel; each FFT uses
    // fft_threads_per_worker threads (already reflected in ns_per_iter when
    // LL_FFT_THREADS == fft_threads_per_worker).
    const double gflops_per_node   = gflops_per_thread
                                     * static_cast<double>(ll_threads_per_worker);
    const double tflops_per_node   = gflops_per_node / 1.0e3;

    // --- Cluster-wide throughput (theoretical, ideal linear scaling) ---
    const double cluster_gflops    = gflops_per_node
                                     * static_cast<double>(cluster_workers);
    const double cluster_tflops    = cluster_gflops / 1.0e3;

    // --- MIPS estimate (per thread, then scaled) ---
    // instructions_per_iter ~= flops_per_iter * 0.9  (FMA + memory/integer overhead)
    const double inst_per_iter     = flops_per_iter * 0.9;
    const double mips_per_thread   = inst_per_iter * iters_per_sec / 1.0e6;
    const double cluster_mips      = mips_per_thread
                                     * static_cast<double>(cluster_workers)
                                     * static_cast<double>(ll_threads_per_worker);

    std::printf("\n--- cluster power (theoretical) ---\n");
    std::printf("FFT length (n)              : %zu\n",  fft_n);
    std::printf("FLOPs/iter                  : %.3e\n", flops_per_iter);
    std::printf("iters/sec   (1 thread)      : %.0f\n", iters_per_sec);
    std::printf("GFLOPs/s    (1 thread)      : %.3f\n", gflops_per_thread);
    std::printf("\nCluster config              : %u workers x %u LL-threads x %u FFT-threads (nested)\n",
                cluster_workers, ll_threads_per_worker, fft_threads_per_worker);
    std::printf("GFLOPs/s    (per node)      : %.3f\n", gflops_per_node);
    std::printf("TFLOPs/s    (per node)      : %.6f\n", tflops_per_node);
    std::printf("TFLOPs/s    (cluster)       : %.4f\n", cluster_tflops);
    std::printf("MIPS        (1 thread, est.): %.0f\n", mips_per_thread);
    std::printf("MIPS        (cluster, est.) : %.0f\n", cluster_mips);
    std::printf("\n(Note: cluster figures assume ideal linear scaling; actual"
                " performance\n will be lower due to memory bandwidth and"
                " scheduling overhead.)\n");

    return exit_code;
}
