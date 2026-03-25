// seqmod_assembler.cpp – Mersenne primality via a(p) = (2+√3)^(2^p)
//
// Criterion:
//   M_p = 2^p − 1  is prime  iff  S_{p-2} ≡ 0 (mod M_p),
//   where  S_0 = 4,  S_k = S_{k-1}^2 − 2.
//   This is the Lucas–Lehmer test.
//
// Computation:
//   All arithmetic is delegated to mersenne::lucas_lehmer() from BigNum.cpp.
//   The backend is chosen automatically by exponent size:
//     p < 128           → GenericBackend  (boost::multiprecision::cpp_int)
//     128 ≤ p < ~4000   → LimbBackend     (Comba squaring + Mersenne fold)
//     p ≥ ~4000         → FftMersenneBackend (Crandall–Bailey DWT/FFT)
//   Override the Limb↔FFT crossover at runtime with LL_LIMB_FFT_CROSSOVER.
//
// This file contains no big-integer arithmetic of its own — it is a thin
// dispatch layer over BigNum's fully optimised, profiled engine.  benchmark_mode
// is always set to true so the already-sieved exponents skip the redundant
// is_prime_exponent() check inside lucas_lehmer().
//
// Build (release):
//   g++ -std=c++20 -O3 -march=native -mtune=native -flto -pthread
//       -DBIGNUM_NO_MAIN src/BigNum.cpp src/seqmod_assembler.cpp
//       -o bin/seqmod_assembler -flto -pthread
//
// Build (gprof):
//   g++ -std=c++20 -O2 -march=native -pthread -pg
//       -DBIGNUM_NO_MAIN src/BigNum.cpp src/seqmod_assembler.cpp
//       -o bin/seqmod_assembler_prof
//
// Usage:
//   seqmod_assembler [prime_count [start_p [threads]]]
//     prime_count – number of prime exponents to test  (default: 512)
//     start_p     – first prime exponent to consider   (default: 2)
//     threads     – parallel workers                   (default: 4; 0 = hwconcurrency)
//
// Environment:
//   SEQMOD_OUTPUT_CSV      – write "p,is_prime" CSV to this path
//   SEQMOD_TIME_LIMIT_SECS – soft stop after N seconds (exit 42)
//   SEQMOD_STATE_FILE      – write JSON resume state on exit
//   SEQMOD_ASM_THREADS     – override the threads argument
//   LL_THREADS             – outer worker count (auto-set from threads if absent)
//   LL_LIMB_FFT_CROSSOVER  – override BigNum's Limb↔FFT crossover exponent
//   LL_FFT_THREADS         – FFT-internal thread count (passed through to BigNum)
//   LL_FFT_ALLOW_NESTED    – allow FFT threading inside outer worker pool
//
// Exit codes: 0 = done, 42 = soft-stop, 1 = error

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <utility>
#include <vector>
#if !defined(_WIN32)
#include <unistd.h>
#endif

// ─── BigNum engine forward declaration ────────────────────────────────────────
// Defined in BigNum.cpp (compiled with -DBIGNUM_NO_MAIN so its main() is omitted).
// benchmark_mode=true skips the redundant is_prime_exponent() check — callers
// already guarantee that p is prime via the sieve.
namespace mersenne {
bool lucas_lehmer(uint32_t p, bool progress, bool benchmark_mode);
} // namespace mersenne

// ─── Exit codes ───────────────────────────────────────────────────────────────
static constexpr int EXIT_DONE    = 0;
static constexpr int EXIT_TIMEOUT = 42;
static constexpr int EXIT_ERROR   = 1;

// ─── Sieve of Eratosthenes ────────────────────────────────────────────────────
// Returns a vector of the first `want` prime exponents ≥ start_p.
static std::vector<int> sieve_primes(int start_p, int want) {
    if (want <= 0 || start_p < 2) return {};

    // Rosser upper bound for the (want + offset)-th prime.
    int extra = 0;
    if (start_p > 2) {
        const double lns = std::log(static_cast<double>(start_p));
        extra = static_cast<int>(1.1 * start_p / lns) + 200;
    }
    const int N      = want + extra + 200;
    const double lnN = std::log(static_cast<double>(N + 2));
    int limit = static_cast<int>(static_cast<double>(N) * (lnN + std::log(lnN) + 1.1)) + 2000;
    if (limit < 2000) limit = 2000;

    std::vector<bool> comp(static_cast<size_t>(limit + 1), false);
    comp[0] = comp[1] = true;
    for (int i = 2; static_cast<long long>(i) * i <= limit; i++)
        if (!comp[i])
            for (int j = i * i; j <= limit; j += i)
                comp[j] = true;

    std::vector<int> out;
    out.reserve(static_cast<size_t>(want));
    for (int i = (start_p < 2 ? 2 : start_p); i <= limit && static_cast<int>(out.size()) < want; i++)
        if (!comp[i]) out.push_back(i);
    return out;
}

// ─── Shared completion queue ──────────────────────────────────────────────────
// Workers push their slot index here on completion.  The main thread picks
// whichever slot finishes first, eliminating head-of-line blocking that would
// arise from fixed round-robin collection when exponent runtimes vary.
struct CompletionQueue {
    std::mutex              mu;
    std::condition_variable cv;
    std::queue<int>         q;  // slot indices of finished workers

    void push(int slot) {
        {
            std::lock_guard<std::mutex> lk(mu);
            q.push(slot);
        }
        cv.notify_one();
    }

    int pop() {
        std::unique_lock<std::mutex> lk(mu);
        cv.wait(lk, [this] { return !q.empty(); });
        const int slot = q.front();
        q.pop();
        return slot;
    }
};

// ─── Per-worker slot ──────────────────────────────────────────────────────────
struct WorkSlot {
    // Written by main before dispatch; read by worker after cv_start wakes it.
    uint32_t p        = 0u;
    // Written by worker on completion; read by main after popping from CompletionQueue.
    bool     result   = false;
    double   elapsed  = 0.0;
    // Dispatch gate: worker waits here until has_work becomes true.
    bool     has_work = false;

    std::mutex              mu;
    std::condition_variable cv_start;
};


// ─── main ─────────────────────────────────────────────────────────────────────
int main(int argc, char* argv[]) {
    // ── Parse arguments ─────────────────────────────────────────────────────
    int prime_count = (argc > 1) ? std::atoi(argv[1]) : 512;
    int start_p     = (argc > 2) ? std::atoi(argv[2]) : 2;
    int nthreads    = (argc > 3) ? std::atoi(argv[3]) : 4;

    // Environment overrides.
    if (const char* e = std::getenv("SEQMOD_ASM_THREADS"))
        nthreads = std::atoi(e);

    if (prime_count < 1) prime_count = 1;
    if (start_p     < 2) start_p     = 2;
    // threads=0 → all hardware cores.
    if (nthreads == 0) {
        const int hw = static_cast<int>(std::thread::hardware_concurrency());
        nthreads = (hw > 0) ? hw : 4;
    }
    if (nthreads < 1)   nthreads = 1;
    if (nthreads > 256) nthreads = 256;

    // ── Propagate worker count to BigNum's FFT gating ────────────────────────
    // BigNum gates nested FFT threading on LL_THREADS.  Use setenv with
    // overwrite=0 so an existing caller-set value is preserved; if absent,
    // publish our actual worker count so BigNum correctly decides whether
    // nested FFT threads would oversubscribe the CPU.
    {
        const std::string val = std::to_string(nthreads);
#if defined(_WIN32)
        _putenv_s("LL_THREADS", val.c_str());
#else
        ::setenv("LL_THREADS", val.c_str(), /*overwrite=*/0);
#endif
    }

    // ── Soft-stop timer ─────────────────────────────────────────────────────
    double time_limit = 0.0;
    if (const char* e = std::getenv("SEQMOD_TIME_LIMIT_SECS"))
        time_limit = std::atof(e);

    // ── Output paths ─────────────────────────────────────────────────────────
    const char* csv_path   = std::getenv("SEQMOD_OUTPUT_CSV");
    const char* state_path = std::getenv("SEQMOD_STATE_FILE");

    // ── Build the prime candidate list via sieve ─────────────────────────────
    const std::vector<int> primes = sieve_primes(start_p, prime_count);
    const int nprimes = static_cast<int>(primes.size());

    std::fprintf(stderr,
        "seqmod_assembler v2.1.0 (bignum engine): "
        "prime_count=%d  start_p=%d  threads=%d%s\n",
        nprimes, start_p, nthreads,
        time_limit > 0.0 ? "  (time-limited)" : "");

    // ── Thread pool and completion queue ─────────────────────────────────────
    // Slots and threads are allocated once.  The CompletionQueue lets the main
    // thread always collect whichever worker finishes next, avoiding head-of-line
    // blocking when per-exponent runtimes vary across the pipeline.
    CompletionQueue cq;
    std::vector<WorkSlot> slots(static_cast<size_t>(nthreads));
    std::vector<std::thread> threads;
    threads.reserve(static_cast<size_t>(nthreads));

    // We need stable slot addresses for the worker pointer arithmetic in
    // cq.push(); use the index directly via a lambda capture instead.
    for (int i = 0; i < nthreads; i++) {
        WorkSlot* sl = &slots[static_cast<size_t>(i)];
        const int idx = i;
        threads.emplace_back([sl, idx, &cq] {
            for (;;) {
                uint32_t p;
                {
                    std::unique_lock<std::mutex> lk(sl->mu);
                    sl->cv_start.wait(lk, [sl] { return sl->has_work; });
                    sl->has_work = false;
                    p = sl->p;
                }
                if (p == 0u) break;

                const auto t0 = std::chrono::steady_clock::now();
                const bool ok = mersenne::lucas_lehmer(
                    p, /*progress=*/false, /*benchmark_mode=*/true);
                const auto t1 = std::chrono::steady_clock::now();

                sl->result  = ok;
                sl->elapsed = std::chrono::duration<double>(t1 - t0).count();
                cq.push(idx);
            }
        });
    }

    // ── Dispatch helper ───────────────────────────────────────────────────────
    // Lock is released before notify to avoid waking the worker into immediate
    // contention on the same mutex.
    const auto dispatch = [&](int slot, uint32_t p) {
        WorkSlot& sl = slots[static_cast<size_t>(slot)];
        {
            std::unique_lock<std::mutex> lk(sl.mu);
            sl.p        = p;
            sl.has_work = true;
        }
        sl.cv_start.notify_one();
    };

    // ── Result accumulators ──────────────────────────────────────────────────
    std::vector<std::pair<int, bool>> csv_rows;
    if (csv_path && *csv_path) csv_rows.reserve(static_cast<size_t>(nprimes));
    std::vector<int> hits;

    // ── Dispatch/collect loop ────────────────────────────────────────────────
    // Maintain a window of nthreads in-flight exponents so every worker is
    // always busy.  The CompletionQueue gives us any-order collection: the main
    // thread unblocks as soon as *any* worker finishes, not just the oldest one.
    const auto t_start    = std::chrono::steady_clock::now();
    auto       t_last_prog = t_start;
    int  done              = 0;
    int  in_flight         = 0;
    bool timed_out         = false;
    int  last_dispatched_p = start_p - 1; // updated at dispatch time
    int  next_prime        = 0;           // index of next prime to dispatch

    // Pre-fill the pipeline.
    for (int s = 0; s < nthreads && next_prime < nprimes; s++, in_flight++) {
        const int p_val = primes[next_prime++];
        last_dispatched_p = p_val;
        dispatch(s, static_cast<uint32_t>(p_val));
    }

    while (in_flight > 0) {
        // Block until any worker finishes — no head-of-line blocking.
        const int slot = cq.pop();
        in_flight--;
        done++;

        const WorkSlot& sl = slots[static_cast<size_t>(slot)];
        const int  p       = sl.p;
        const bool result  = sl.result;
        const double elapsed = sl.elapsed;

        if (result) hits.push_back(p);
        if (csv_path && *csv_path) csv_rows.emplace_back(p, result);

        // Progress output: sparse (every 64 completions or every 10 s).
        const auto now = std::chrono::steady_clock::now();
        const double wall       = std::chrono::duration<double>(now - t_start).count();
        const double since_prog = std::chrono::duration<double>(now - t_last_prog).count();
        if (done == 1 || done % 64 == 0 || since_prog >= 10.0) {
            const double pct = (nprimes > 0) ? 100.0 * done / nprimes : 0.0;
            std::fprintf(stderr,
                "  [%6.1f s] %d/%d (%.1f%%)  primes_found=%d  last_p=%d  last=%.3f s\n",
                wall, done, nprimes, pct,
                static_cast<int>(hits.size()), p, elapsed);
            t_last_prog = now;
        }

        // Soft-stop check.
        if (time_limit > 0.0 && wall >= time_limit) {
            timed_out = true;
            // Dispatch no more work; drain remaining in-flight workers below.
            break;
        }

        // Dispatch the next exponent to the slot just freed.
        if (next_prime < nprimes) {
            last_dispatched_p = primes[next_prime];
            dispatch(slot, static_cast<uint32_t>(primes[next_prime]));
            next_prime++;
            in_flight++;
        }
    }

    // ── Drain in-flight workers on soft-stop ──────────────────────────────────
    // Workers that were already dispatched will complete normally; collect and
    // record their results so they appear in CSV and state output.
    while (in_flight > 0) {
        const int slot = cq.pop();
        in_flight--;
        const WorkSlot& sl = slots[static_cast<size_t>(slot)];
        const int drained_p = static_cast<int>(sl.p);
        if (sl.result) hits.push_back(drained_p);
        if (csv_path && *csv_path) csv_rows.emplace_back(drained_p, sl.result);
    }

    // ── Shut down thread pool ────────────────────────────────────────────────
    // Send sentinel (p=0) to every worker; workers exit on p==0 without pushing
    // to the CompletionQueue, so we just join them directly.
    for (int i = 0; i < nthreads; i++) dispatch(i, 0u);
    for (auto& t : threads) t.join();

    // ── Write CSV ────────────────────────────────────────────────────────────
    if (csv_path && *csv_path && !csv_rows.empty()) {
        std::sort(csv_rows.begin(), csv_rows.end());
        std::ofstream f(csv_path);
        if (f) {
            f << "p,is_prime\n";
            for (const auto& [p, ok] : csv_rows)
                f << p << ',' << (ok ? "true" : "false") << '\n';
        } else {
            std::fprintf(stderr, "seqmod_assembler: cannot open CSV file: %s\n", csv_path);
        }
    }

    // ── Write JSON state (for resume support) ────────────────────────────────
    if (state_path && *state_path) {
        std::ofstream f(state_path);
        if (f) {
            f << "{\n"
              << "  \"start_p\": " << start_p << ",\n"
              << "  \"last_dispatched_p\": " << last_dispatched_p << ",\n"
              << "  \"timed_out\": " << (timed_out ? "true" : "false") << ",\n"
              << "  \"mersenne_primes_found\": [";
            for (size_t i = 0; i < hits.size(); i++) {
                if (i) f << ", ";
                f << hits[i];
            }
            f << "]\n}\n";
        }
    }

    // ── Summary ──────────────────────────────────────────────────────────────
    const double total = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t_start).count();
    std::fprintf(stderr,
        "seqmod_assembler: done=%d  mersenne_primes=%d  "
        "wall=%.3f s  avg=%.3f ms/prime\n",
        done, static_cast<int>(hits.size()), total,
        done > 0 ? 1000.0 * total / done : 0.0);

    if (timed_out) {
        // Print last dispatched exponent for the workflow to resume from.
        std::printf("SEQMOD_LAST_DISPATCHED_P=%d\n", last_dispatched_p);
        return EXIT_TIMEOUT;
    }
    return EXIT_DONE;
}
