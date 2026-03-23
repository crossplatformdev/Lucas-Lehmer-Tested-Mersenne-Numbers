// sequence_powermod_stdc.cpp – C-stdlib Mersenne sequence search using the
// congruence (2+√3)^(2^n) ≡ 1 (mod 2^n−1) as a Mersenne-prime criterion.
// Uses only C standard library types and functions (no GMP).
//
// Algorithm:
//   For each prime exponent p, compute (result_a + result_b·√3) where
//   (result_a, result_b) = (2 + √3)^(2^p) mod (2^p − 1).
//   M_p is prime iff (2·result_a − 2) ≡ 0 (mod 2^p − 1).
//
//   For p < 64 the computation is done entirely in native 64-bit arithmetic
//   using __uint128_t for intermediate products.
//   For p ≥ 64 the standard Lucas-Lehmer test is used with a 64-bit limb
//   representation, Comba squaring, and Mersenne-specific folding reduction.
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

// ─── 64-bit limb big-integer helpers (used for p ≥ 64 Lucas-Lehmer) ──────────
// Representation: little-endian vector of uint64_t limbs; always normalised
// (no leading zeros, except the single-limb value 0).
// Using 64-bit limbs halves the limb count vs 32-bit, cutting Comba squaring
// cost by ~4× for the same bit-width.

using Limbs = std::vector<uint64_t>;

static void normalize(Limbs& a) {
    while (a.size() > 1 && a.back() == 0) a.pop_back();
}

static bool is_zero(const Limbs& a) {
    return a.size() == 1 && a[0] == 0;
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

// Subtract a small scalar v from a (assumes a >= v).
static void sub_small_inplace(Limbs& a, uint64_t v) {
    for (size_t i = 0; i < a.size() && v; ++i) {
        if (a[i] >= v) {
            a[i] -= v;
            v = 0;
        } else {
            // a[i] < v: borrow from the next limb.
            // Result limb = a[i] + 2^64 - v
            a[i] = static_cast<uint64_t>(
                (static_cast<__uint128_t>(1) << 64)
                + static_cast<__uint128_t>(a[i]) - v);
            v = 1;  // carry 1 into the next limb
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

// Comba squaring: out = a², exploiting a[i]*a[j] = a[j]*a[i].
// Operations: n diagonal + n(n-1)/2 cross terms ≈ n²/2 multiplications.
// Each 64-bit × 64-bit product is computed with __uint128_t.
static Limbs square_comba(const Limbs& a) {
    const size_t n = a.size();
    Limbs out(n * 2, 0);

    for (size_t i = 0; i < n; ++i) {
        // Diagonal: a[i]^2 → accumulate at position 2i
        {
            __uint128_t carry = static_cast<__uint128_t>(a[i]) * a[i];
            for (size_t k = 2 * i; carry; ++k) {
                carry += out[k];
                out[k] = static_cast<uint64_t>(carry);
                carry >>= 64;
            }
        }
        // Cross terms: 2 * a[i] * a[j] for j > i → position i+j
        for (size_t j = i + 1; j < n; ++j) {
            // prod = a[i] * a[j], fits in 128 bits.
            // 2*prod fits in 129 bits: split into 64-bit hi/lo before doubling.
            const __uint128_t prod     = static_cast<__uint128_t>(a[i]) * a[j];
            const uint64_t    prod_lo  = static_cast<uint64_t>(prod);
            const uint64_t    prod_hi  = static_cast<uint64_t>(prod >> 64);
            const uint64_t    cross_lo  = prod_lo << 1;
            const uint64_t    cross_mid = (prod_hi << 1) | (prod_lo >> 63);
            const uint64_t    cross_top = prod_hi >> 63;  // 0 or 1

            size_t k = i + j;
            __uint128_t carry = static_cast<__uint128_t>(out[k]) + cross_lo;
            out[k] = static_cast<uint64_t>(carry);
            carry  = (carry >> 64) + static_cast<__uint128_t>(out[k + 1]) + cross_mid;
            out[k + 1] = static_cast<uint64_t>(carry);
            carry = (carry >> 64) + cross_top;
            k += 2;
            while (carry) {
                carry += out[k];
                out[k] = static_cast<uint64_t>(carry);
                carry >>= 64;
                ++k;
            }
        }
    }

    normalize(out);
    return out;
}

// Return a >> shift (bit-level right shift).
static Limbs right_shift_bits(const Limbs& a, int shift) {
    if (shift <= 0) return a;
    const size_t word_shift = static_cast<size_t>(shift / 64);
    const int    bit_shift  = shift % 64;
    if (word_shift >= a.size()) return Limbs{0};

    Limbs out(a.size() - word_shift, 0);
    for (size_t i = word_shift; i < a.size(); ++i) {
        const uint64_t v    = a[i];
        const uint64_t next = (i + 1 < a.size()) ? a[i + 1] : 0;
        if (bit_shift == 0) {
            out[i - word_shift] = v;
        } else {
            // bit_shift in [1,63]: both operands are valid shift amounts.
            out[i - word_shift] = (v >> bit_shift) | (next << (64 - bit_shift));
        }
    }
    normalize(out);
    return out;
}

// Return the low `bits` bits of a.
static Limbs low_bits(const Limbs& a, int bits) {
    if (bits <= 0) return Limbs{0};
    const size_t full_words = static_cast<size_t>(bits / 64);
    const int    rem_bits   = bits % 64;
    size_t keep = full_words + (rem_bits ? 1u : 0u);
    keep = std::min(keep, a.size());
    Limbs out(a.begin(), a.begin() + keep);
    if (rem_bits && !out.empty())
        out.back() &= (static_cast<uint64_t>(1) << rem_bits) - 1u;
    normalize(out);
    return out;
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

// Reduce x modulo 2^p − 1 using repeated high-bit folding.
static void reduce_mod_mersenne(Limbs& x, int p, const Limbs& m_mask) {
    while (bitlen(x) > static_cast<size_t>(p)) {
        Limbs lo = low_bits(x, p);
        Limbs hi = right_shift_bits(x, p);
        add_inplace(lo, hi);
        x.swap(lo);
    }
    while (limbs_cmp(x, m_mask) >= 0)
        sub_inplace(x, m_mask);
}

// Convert Limbs to decimal string.
static std::string to_decimal(const Limbs& x) {
    if (is_zero(x)) return "0";
    Limbs t = x;
    std::string out;
    while (!is_zero(t)) {
        uint64_t rem = 0;
        for (size_t i = t.size(); i-- > 0;) {
            const __uint128_t cur =
                (static_cast<__uint128_t>(rem) << 64) | t[i];
            t[i] = static_cast<uint64_t>(cur / 10);
            rem  = static_cast<uint64_t>(cur % 10);
        }
        out.push_back(static_cast<char>('0' + rem));
        normalize(t);
    }
    std::reverse(out.begin(), out.end());
    return out;
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

// ─── Core primality test ───────────────────────────────────────────────────────
// Returns {is_prime, mod_result_string} for M_n = 2^n − 1.
//
// p < 64: algebraic criterion via native 64-bit arithmetic.
//   Computes (2 + √3)^(2^p) mod (2^p − 1) in Z[√3] using __uint128_t mulmod.
//   M_p is prime iff (2·result_a − 2) ≡ 0 (mod 2^p − 1).
//
// p ≥ 64: Lucas-Lehmer test via 64-bit Comba squaring.
//   Starts from s = 4 and iterates s = (s² − 2) mod (2^p − 1) for p−2 steps.
//   M_p is prime iff s ≡ 0.
static std::pair<bool, std::string> is_sequence_zero(int n) {
    if (n < 2) return {false, ""};

    if (n < 64) {
        const uint64_t modulus = (1ULL << n) - 1;
        uint64_t result_a = 1 % modulus;
        uint64_t result_b = 0;
        uint64_t base_a   = 2 % modulus;
        uint64_t base_b   = 1 % modulus;

        // Compute (2+√3)^(2^n) mod modulus: square base n times, then
        // multiply result by base once (exponent 2^n has a single 1-bit).
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
        const bool is_prime = (lhs == 0);
        return {is_prime, is_prime ? "0" : std::to_string(lhs)};
    }

    // Lucas-Lehmer for p >= 64 using 64-bit Comba squaring.
    const Limbs m_mask = mersenne_mask(n);
    Limbs s{4};
    for (int i = 0; i < n - 2; ++i) {
        Limbs sq = square_comba(s);
        sub_small_inplace(sq, 2);
        reduce_mod_mersenne(sq, n, m_mask);
        s.swap(sq);
    }
    const bool is_prime = is_zero(s);
    return {is_prime, is_prime ? "0" : to_decimal(s)};
}

// ─── Result accumulator (thread-safe) ─────────────────────────────────────────
struct Results {
    std::mutex mu;
    std::vector<std::tuple<int, bool, std::string>> rows;

    void add(int n, bool is_prime, std::string mod_result = {}) {
        std::lock_guard<std::mutex> lock(mu);
        rows.emplace_back(n, is_prime, std::move(mod_result));
    }
};

// ─── Main sweep function ───────────────────────────────────────────────────────
// Returns the list of Mersenne-prime exponents found in [start_n, start_n+iterations-1].
// collect_rows   – add every candidate to `results` for CSV output.
// track_dispatch – maintain last_dispatched_n for safe soft-stop resume.
static std::vector<int> find_sequence(
    int          iterations,
    int          start_n,
    uint32_t     parallel_threads,
    long long    time_limit_secs,
    bool         collect_rows,
    bool         track_dispatch,
    Results&     results,
    int&         out_last_dispatched_n)
{
    start_n    = std::max(1, start_n);
    iterations = std::max(0, iterations);

    const int end_n = start_n + iterations - 1;

    // Sieve of Eratosthenes to mark prime candidates up-front.
    std::vector<uint8_t> prime_flags(static_cast<size_t>(iterations), 0);
    if (iterations > 0) {
        std::vector<uint8_t> sieve(static_cast<size_t>(end_n + 1), 1);
        sieve[0] = 0;
        if (end_n >= 1) sieve[1] = 0;
        for (int p = 2; 1LL * p * p <= end_n; ++p) {
            if (!sieve[static_cast<size_t>(p)]) continue;
            for (int m = p * p; m <= end_n; m += p)
                sieve[static_cast<size_t>(m)] = 0;
        }
        for (int i = 0; i < iterations; ++i)
            prime_flags[static_cast<size_t>(i)] =
                sieve[static_cast<size_t>(start_n + i)];
    }

    std::vector<int> hits;
    std::mutex       hits_mutex;

    // fast_mode: for short, time-unlimited runs skip the polling monitor to
    // reduce latency.  Threshold: 2000 candidates fit comfortably within a
    // sub-second wall time for the typical exponent range tested here.
    static constexpr int  FAST_MODE_ITER_THRESHOLD  = 2000;
    // Report progress every PROGRESS_REPORT_INTERVAL candidates processed.
    static constexpr int  PROGRESS_REPORT_INTERVAL  = 1000;
    // Polling period for the monitor thread (milliseconds).
    static constexpr int  MONITOR_POLL_INTERVAL_MS  = 5;

    const bool     fast_mode    = (time_limit_secs <= 0 &&
                                   iterations <= FAST_MODE_ITER_THRESHOLD);
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
                const int i = next_index.fetch_add(1, std::memory_order_relaxed);
                if (i >= iterations) break;
                const int n = start_n + i;

                if (track_dispatch) {
                    int prev = last_dispatched_n.load(std::memory_order_relaxed);
                    while (prev < n &&
                           !last_dispatched_n.compare_exchange_weak(
                               prev, n, std::memory_order_relaxed))
                    { /* retry CAS */ }
                }

                if (g_stop.load(std::memory_order_relaxed)) break;

                if (prime_flags[static_cast<size_t>(i)]) {
                    if (g_stop.load(std::memory_order_relaxed)) break;
                    const auto [prime, mod_val] = is_sequence_zero(n);
                    if (collect_rows) results.add(n, prime, mod_val);
                    if (prime) local_hits.push_back(n);
                } else {
                    if (collect_rows) results.add(n, false, "");
                }

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

        out_last_dispatched_n = track_dispatch ? last_dispatched_n.load() : end_n;
        std::sort(hits.begin(), hits.end());
        return hits;
    }

    // Polling monitor: progress reporting + time-limit enforcement.
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

        if (done >= next_progress_report || done == iterations) {
            std::chrono::seconds eta(0);
            if (done < iterations) {
                const double avg_sec =
                    (done > 0) ? static_cast<double>(elapsed.count()) / done : 0.0;
                eta = std::chrono::seconds(
                    static_cast<long long>(avg_sec * (iterations - done)));
            }
            const double percent =
                (iterations > 0) ? (100.0 * done / iterations) : 100.0;
            std::cout << "Elapsed: " << format_duration(elapsed)
                      << " | ETA: "  << format_duration(eta)
                      << " | Progress: " << percent << "%" << std::endl;
            while (next_progress_report <= done)
                next_progress_report += PROGRESS_REPORT_INTERVAL;
        }

        if (done >= iterations || g_stop.load(std::memory_order_relaxed)) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(MONITOR_POLL_INTERVAL_MS));
    }

    for (std::thread& w : workers) w.join();

    out_last_dispatched_n = track_dispatch ? last_dispatched_n.load() : end_n;
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
    f << "]\n}\n";
    return true;
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

    const int end_n = start_n + iterations - 1;

    std::cout << "sequence_powermod_stdc: start_n=" << start_n
              << " end_n=" << end_n
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
    if (state_path && *state_path) write_state(state_path, start_n, end_n,
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
