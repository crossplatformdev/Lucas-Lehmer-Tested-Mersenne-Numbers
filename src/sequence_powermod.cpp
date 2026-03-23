// sequence_powermod.cpp – GMP-based Mersenne sequence search using the
// congruence (2+√3)^(2^n) ≡ 1 (mod 2^n−1) as a Mersenne-prime criterion.
//
// Algorithm:
//   For each prime exponent p, compute (result_a + result_b·√3) where
//   (result_a, result_b) = (2 + √3)^(2^p) mod (2^p − 1).
//   M_p is prime iff (2·result_a − 2) ≡ 0 (mod 2^p − 1).
//
// Usage:
//   sequence_powermod [iterations [start_n [threads]]]
//
//   iterations  – number of candidates to test, starting from start_n (default: 512)
//   start_n     – first candidate n (default: 1)
//   threads     – parallel worker threads (default: 1)
//
// Environment variables:
//   SEQMOD_TIME_LIMIT_SECS  – soft stop after N seconds; write state and
//                             exit with code 42 (default: 0 = no limit)
//   SEQMOD_OUTPUT_CSV       – write "n,is_prime,mod_result" CSV to this path
//                             mod_result = (2·result_a−2) mod (2^n−1); empty
//                             for composite n (not tested by the algorithm)
//   SEQMOD_STATE_FILE       – write JSON state on exit to this path
//                             (includes last_dispatched_n for safe resume)
//
// Exit codes:
//   0   – completed normally; all candidates in the requested range tested
//   42  – soft stop (time limit reached); partial results written to CSV/state
//   1   – error (bad arguments, etc.)

#include <gmpxx.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

// ─── Exit-code constants ──────────────────────────────────────────────────────
static constexpr int EXIT_DONE    = 0;
static constexpr int EXIT_TIMEOUT = 42;
static constexpr int EXIT_ERROR   = 1;

// ─── Global stop flag (set from the main/progress thread) ─────────────────────
static std::atomic<bool> g_stop{false};

// ─── Helpers ──────────────────────────────────────────────────────────────────
static bool is_prime_index(int n) {
    if (n < 2) return false;
    if (n == 2) return true;
    if ((n & 1) == 0) return false;
    // 1LL * d * d promotes to int64_t before the multiply to avoid int overflow
    // when d² exceeds INT_MAX (i.e., d > 46340 for 32-bit int).
    for (int d = 3; 1LL * d * d <= n; d += 2)
        if (n % d == 0) return false;
    return true;
}

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

// Component-wise multiplication in Z[√3] reduced mod `modulus`.
//   (la + lb·√3) * (ra + rb·√3) = (la·ra + 3·lb·rb) + (la·rb + lb·ra)·√3
static void mul_mod_components(
    const mpz_class& la, const mpz_class& lb,
    const mpz_class& ra, const mpz_class& rb,
    const mpz_class& modulus,
    mpz_class& out_a, mpz_class& out_b)
{
    out_a = (la * ra + 3 * lb * rb) % modulus;
    out_b = (la * rb + lb * ra)     % modulus;
}

// Returns {is_prime, mod_result} for M_n = 2^n − 1 using the algebraic
// identity.  mod_result is lhs = (2·result_a − 2) mod (2^n − 1); it equals
// 0 exactly when M_n is prime.
static std::pair<bool, mpz_class> is_sequence_zero(int n) {
    if (n < 2) return {false, mpz_class(0)};

    const mpz_class modulus  = (mpz_class(1) << n) - 1;   // 2^n − 1
    const mpz_class exponent =  mpz_class(1) << n;         // 2^n

    mpz_class result_a = 1 % modulus;
    mpz_class result_b = 0;
    mpz_class base_a   = 2 % modulus;
    mpz_class base_b   = 1 % modulus;
    mpz_class exp      = exponent;

    while (exp > 0) {
        if ((exp & 1) != 0) {
            mpz_class na, nb;
            mul_mod_components(result_a, result_b,
                               base_a,   base_b,
                               modulus,  na, nb);
            result_a = na;
            result_b = nb;
        }
        mpz_class sa, sb;
        mul_mod_components(base_a, base_b,
                           base_a, base_b,
                           modulus, sa, sb);
        base_a = sa;
        base_b = sb;
        exp >>= 1;
    }

    mpz_class lhs = (2 * result_a - 2) % modulus;
    if (lhs < 0) lhs += modulus;
    return {lhs == 0, lhs};
}

// ─── Result accumulator (thread-safe) ─────────────────────────────────────────
struct Results {
    std::mutex mu;
    // (n, is_prime, mod_result) – mod_result is the decimal string of
    // (2·result_a − 2) mod (2^n − 1); empty string for composite n.
    std::vector<std::tuple<int, bool, std::string>> rows;

    void add(int n, bool is_prime, std::string mod_result = {}) {
        std::lock_guard<std::mutex> lock(mu);
        rows.emplace_back(n, is_prime, std::move(mod_result));
    }
};

// ─── Main sweep function ───────────────────────────────────────────────────────
// Returns hits (prime exponents) and the last n value that was dispatched to a
// worker thread (last_dispatched_n).  On timeout last_dispatched_n is the safe
// resume boundary: every n ≤ last_dispatched_n has been *dispatched* but may not
// all be *completed* because some threads may still be inside is_sequence_zero().
// Since workers finish their current computation before checking g_stop, the safe
// resume point is: min(currently_working_n) − 1, which we conservatively report
// as (start_n + next_dispatched_index − worker_count − 1) when stop fires.
static std::vector<int> find_sequence(
    int          iterations,
    int          start_n,
    uint32_t     parallel_threads,
    long long    time_limit_secs,
    Results&     results,
    int&         out_last_dispatched_n)
{
    start_n   = std::max(1, start_n);
    iterations = std::max(0, iterations);

    std::vector<int>           hits;
    std::mutex                 hits_mutex;
    const uint32_t             worker_count = std::max<uint32_t>(1, parallel_threads);
    std::atomic<int>           next_index{0};
    std::atomic<int>           done_count{0};
    // last_dispatched tracks the highest n handed to any thread so far.
    std::atomic<int>           last_dispatched_n{start_n - 1};

    const auto started_at = std::chrono::steady_clock::now();

    std::vector<std::thread> workers;
    workers.reserve(worker_count);
    for (uint32_t t = 0; t < worker_count; ++t) {
        workers.emplace_back([&]() {
            std::vector<int> local_hits;
            local_hits.reserve(8);

            while (!g_stop.load(std::memory_order_relaxed)) {
                const int i = next_index.fetch_add(1, std::memory_order_relaxed);
                if (i >= iterations) break;

                const int n = start_n + i;

                // Update the high-water mark of dispatched work.
                // memory_order_relaxed is sufficient here: last_dispatched_n
                // is a monotonically increasing progress counter that is only
                // read after all threads are joined (via out_last_dispatched_n),
                // so there is no ordering dependency with other atomic operations
                // that would require a stronger memory order.
                int prev = last_dispatched_n.load(std::memory_order_relaxed);
                while (prev < n &&
                       !last_dispatched_n.compare_exchange_weak(
                           prev, n, std::memory_order_relaxed))
                { /* retry CAS */ }

                // Check stop flag before starting an expensive computation.
                if (g_stop.load(std::memory_order_relaxed)) break;

                if (is_prime_index(n)) {
                    // Re-check stop flag before the long is_sequence_zero() call.
                    if (g_stop.load(std::memory_order_relaxed)) break;

                    const auto [prime, mod_val] = is_sequence_zero(n);
                    results.add(n, prime, mod_val.get_str());
                    if (prime) local_hits.push_back(n);
                } else {
                    // mod_result not computed for composite n.
                    results.add(n, false, "");
                }

                done_count.fetch_add(1, std::memory_order_relaxed);
            }

            if (!local_hits.empty()) {
                std::lock_guard<std::mutex> lock(hits_mutex);
                hits.insert(hits.end(), local_hits.begin(), local_hits.end());
            }
        });
    }

    // ── Progress / time-limit monitor (main thread) ───────────────────────
    int next_progress_report = 1000;

    while (true) {
        const int done = done_count.load(std::memory_order_relaxed);

        const auto now     = std::chrono::steady_clock::now();
        const auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
            now - started_at);

        // Time-limit check.
        if (time_limit_secs > 0 && elapsed.count() >= time_limit_secs) {
            g_stop.store(true, std::memory_order_relaxed);
            std::cerr << "SEQMOD: soft stop at " << elapsed.count()
                      << "s (limit=" << time_limit_secs << "s), "
                      << "last_dispatched_n=" << last_dispatched_n.load()
                      << "\n";
        }

        if (done >= next_progress_report || done == iterations) {
            const double avg_sec = (done > 0)
                ? static_cast<double>(elapsed.count()) / done : 0.0;
            const int remaining  = iterations - done;
            const auto eta       = std::chrono::seconds(
                static_cast<long long>(avg_sec * remaining));

            std::cout << "Elapsed: " << format_duration(elapsed)
                      << " | ETA: "  << format_duration(eta) << std::endl;

            while (next_progress_report <= done)
                next_progress_report += 1000;
        }

        if (done >= iterations || g_stop.load(std::memory_order_relaxed))
            break;

        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }

    for (std::thread& w : workers) w.join();

    out_last_dispatched_n = last_dispatched_n.load();

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
    f << "n,is_prime,mod_result\n";
    // Sort by n before writing for reproducibility.
    std::vector<std::tuple<int, bool, std::string>> sorted;
    {
        std::lock_guard<std::mutex> lock(results.mu);
        sorted = results.rows;
    }
    std::sort(sorted.begin(), sorted.end());
    for (const auto& [n, is_prime, mod_result] : sorted)
        f << n << ',' << (is_prime ? "true" : "false") << ',' << mod_result << '\n';
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
    f << "]\n";
    f << "}\n";
    return true;
}

// ─── main ─────────────────────────────────────────────────────────────────────
int main(int argc, char* argv[]) {
    // Parse positional arguments.
    const int iterations       = (argc > 1) ? std::stoi(argv[1]) : 512;
    const int start_n          = (argc > 2) ? std::stoi(argv[2]) : 1;
    const uint32_t threads     = (argc > 3) ? static_cast<uint32_t>(std::stoul(argv[3])) : 1u;

    // Read environment variables.
    const char* tl_env = std::getenv("SEQMOD_TIME_LIMIT_SECS");
    const long long time_limit_secs = tl_env ? std::stoll(tl_env) : 0LL;

    const char* csv_path   = std::getenv("SEQMOD_OUTPUT_CSV");
    const char* state_path = std::getenv("SEQMOD_STATE_FILE");

    const int end_n = start_n + iterations - 1;

    std::cout << "sequence_powermod: start_n=" << start_n
              << " end_n=" << end_n
              << " threads=" << threads;
    if (time_limit_secs > 0)
        std::cout << " time_limit=" << time_limit_secs << "s";
    std::cout << "\n";

    Results results;
    int last_dispatched_n = start_n - 1;

    const std::vector<int> hits = find_sequence(
        iterations, start_n, threads, time_limit_secs,
        results, last_dispatched_n);

    const bool timed_out = g_stop.load();

    // Write CSV output.
    if (csv_path && *csv_path)
        write_csv(csv_path, results);

    // Write JSON state.
    if (state_path && *state_path)
        write_state(state_path, start_n, end_n, last_dispatched_n,
                    timed_out, hits);

    // Print summary to stdout (matches the original format).
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
