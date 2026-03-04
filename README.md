# BigNum

Performance-focused Lucas–Lehmer primality test for Mersenne numbers, with
tiered backends, multi-mode operation, and a profiling/benchmark harness.

---

## Mathematics

### Mersenne numbers

A **Mersenne number** with prime exponent $p$ is defined as:

$$M_p = 2^p - 1$$

$M_p$ is a **Mersenne prime** when $M_p$ itself is prime. A necessary condition is that $p$ be prime, though not all prime $p$ yield a prime $M_p$.

### Lucas–Lehmer primality test

Define the sequence $\{s_k\}$ by:

$$s_0 = 4, \qquad s_{k+1} = s_k^2 - 2 \pmod{M_p}$$

**Theorem (Lucas–Lehmer):** For an odd prime $p$,

$$M_p \text{ is prime} \iff s_{p-2} \equiv 0 \pmod{M_p}$$

### Hot-path iteration

Each Lucas–Lehmer iteration computes the recurrence modulo $M_p = 2^p - 1$:

$$s \;\leftarrow\; \bigl(s^2 - 2\bigr) \bmod \bigl(2^p - 1\bigr)$$

The modular reduction exploits the Mersenne property: because $2^p \equiv 1 \pmod{M_p}$, a $2p$-bit product $q$ can be reduced by folding the upper half back onto the lower half,

$$q \bmod (2^p - 1) \;=\; (q \bmod 2^p) + \lfloor q / 2^p \rfloor$$

with at most one further subtraction of $M_p$. This avoids a general big-integer division and is the basis of the Crandall–Bailey DWT/FFT backend.

---

## Architecture

Each Lucas–Lehmer iteration computes `s = (s² − 2) mod (2^p − 1)`.
The backend is chosen automatically based on the exponent size:

| Exponent range | Backend | Algorithm |
|---|---|---|
| `p < 128` | `GenericBackend` | `boost::multiprecision::cpp_int` – reference path |
| `128 ≤ p < LL_LIMB_FFT_CROSSOVER` (default 4000) | `LimbBackend` | True Comba schoolbook squaring + recursive Karatsuba + Mersenne fold |
| `p ≥ LL_LIMB_FFT_CROSSOVER` | `FftMersenneBackend` | Crandall–Bailey DWT/FFT cyclic convolution |

The crossover threshold can be tuned at runtime via the `LL_LIMB_FFT_CROSSOVER`
environment variable (see [Environment variables](#environment-variables)).

### Performance optimisations

- **Real-FFT optimisation (Phase 4):** two n-point complex FFTs replaced by two
  n/2-point FFTs + O(n) pre/post-processing — ~46 % fewer butterflies per
  iteration.
- **Fused hot op:** `square_sub2_mod_mersenne()` performs the full LL step in a
  single FFT pair; no intermediate allocation.
- **Pre-computed tables:** twiddle factors, DWT weights, bit-reversal table, and
  digit-width table are allocated once per engine lifetime.
- **No heap allocation inside the hot loop:** all scratch buffers are
  pre-allocated.
- **True Comba squaring:** n*(n+1)/2 multiplications vs n² for generic multiply.
- **Multiply-by-reciprocal carry propagation (Phase 3):** avoids the 20+ cycle
  integer divide.
- **Hardware sqrt for trial division (Phase 1):** one `VSQRTSD` before the loop,
  plain 64-bit comparison inside.

---

## Build

```
make all          # release binary: bin/bignum
make plan-tool    # C plan helper:  bin/split_bucket_batches
```

**Compiler flags (defaults):**
`-std=c++20 -O3 -march=native -mtune=native -flto -pthread`

**Dependencies:** `g++`, `make`, `libboost-dev`
(`sudo apt-get install -y g++ make libboost-dev pkg-config`)

---

## Running

```
./bin/bignum [start_index] [threads] [progress]
```

| Argument | Default | Description |
|---|---|---|
| `start_index` | `0` | First index into the known Mersenne-prime exponent list |
| `threads` | max available cores | Worker thread count; `0` = max |
| `progress` | off | Any third argument enables per-iteration progress output |

The binary supports several **operating modes** selected via environment
variables:

| `LL_SWEEP_MODE` | Mode | Description |
|---|---|---|
| *(not set)* | **Benchmark / known-list** | Walk `known_mersenne_prime_exponents[]` from `start_index`; `is_prime_exponent()` check skipped by default |
| `n` | **Natural sweep** | All integers in `[LL_MIN_EXPONENT, LL_MAX_EXPONENT]` |
| `p` | **Prime sweep** | All prime exponents in `[LL_MIN_EXPONENT, LL_MAX_EXPONENT]` |
| `m` | **Mersenne-first sweep** | Known Mersenne-prime exponents first, then remaining primes in range |
| `discover` | **Discover mode** | Explore primes beyond the known list; writes results to `LL_OUTPUT_DIR` |
| `power_bucket_primes` | **Power-bucket mode** | Sweep one power-of-two exponent bucket (B_n = [2^(n-1), 2^n−1]) |

---

## Environment variables

### Core controls

| Variable | Default | Description |
|---|---|---|
| `LL_LIMB_FFT_CROSSOVER` | `4000` | Exponent threshold where `LimbBackend` gives way to `FftMersenneBackend` |
| `LL_STOP_AFTER_ONE` | `0` | `1` = run exactly one exponent then exit |
| `LL_BENCHMARK_MODE` | `1` for known-list/prime/Mersenne modes; `0` for natural mode | Skip `is_prime_exponent()` check (safe when the input list is already prime) |
| `LL_MAX_EXPONENT_INDEX` | *(list length)* | Exclusive upper bound on the known-list index (used to keep CI bounded) |
| `LL_BENCH_OUTPUT` | *(none)* | Path for machine-readable CSV output (`p,is_prime,time_sec`) |

### Sweep / range controls

| Variable | Default | Description |
|---|---|---|
| `LL_SWEEP_MODE` | *(none)* | Operating mode: `n`, `p`, `m`, `discover`, `power_bucket_primes` |
| `LL_MIN_EXPONENT` | `2` | Lower bound (inclusive) for sweep modes |
| `LL_MAX_EXPONENT` | last known Mersenne exponent | Upper bound (inclusive) for sweep modes |
| `LL_MAX_CASES` | *(no limit)* | Hard cap on exponents to process |
| `LL_SHARD_INDEX` | `0` | Which shard to run (0-based) |
| `LL_SHARD_COUNT` | `1` | Total number of shards (for parallel CI jobs) |
| `LL_REVERSE_ORDER` | `0` | `1` = reverse work list before running |
| `LL_LARGEST_FIRST` | `0` | `1` = sort work list largest-exponent-first |
| `LL_THREADS` | *(argv[2] or max)* | Override thread count via environment |

### Discover mode

| Variable | Default | Description |
|---|---|---|
| `LL_SINGLE_EXPONENT` | `0` | Explicit first exponent to test (0 = none) |
| `LL_MIN_EXPONENT` | `136279841` | Lower bound exclusive for discover range |
| `LL_MAX_EXPONENT` | `200000000` | Upper bound inclusive for discover range |
| `LL_STOP_AFTER_N_CASES` | `0` | Stop after N exponents (0 = no limit) |
| `LL_DRY_RUN` | `0` | `1` = print plan without running LL tests |
| `LL_PROGRESS` | `0` | `1` = show per-iteration LL progress |
| `LL_OUTPUT_DIR` | `discover_out/` | Directory for result files (CSV, JSON) |

### Power-bucket mode

| Variable | Default | Description |
|---|---|---|
| `LL_BUCKET_N` | `0` | Bucket number to run (1–64; 0 = all buckets from `LL_BUCKET_START` to `LL_BUCKET_END`) |
| `LL_BUCKET_START` | `1` | First bucket (inclusive) when `LL_BUCKET_N=0` |
| `LL_BUCKET_END` | `64` | Last bucket (inclusive) when `LL_BUCKET_N=0` |
| `LL_MAX_EXPONENTS_PER_JOB` | `0` | Cap on exponents per bucket (0 = no limit) |
| `LL_RESUME_FROM_EXPONENT` | `0` | Skip exponents below this value |
| `LL_STOP_AFTER_FIRST_PRIME_RESULT` | `0` | Stop after the first new Mersenne prime found |

### Progress / checkpointing

| Variable | Default | Description |
|---|---|---|
| `LL_PROGRESS_INTERVAL_ITERS` | `10000` | Checkpoint every N iterations |
| `LL_PROGRESS_INTERVAL_SECONDS` | `0` | Also checkpoint every N seconds (0 = off) |

---

## Makefile targets

| Target | Description |
|---|---|
| `make all` | Build release binary `bin/bignum` |
| `make plan-tool` | Build C plan helper `bin/split_bucket_batches` |
| `make unit` | Run unit / correctness tests (fast) |
| `make smoke` | Run a single tiny exponent to confirm the binary works |
| `make regression` | Run bounded regression: exponents[0..26], p ≤ 44 497 (< 15 s) |
| `make test` | Composite: `unit` + `smoke` + `regression` |
| `make bench` | Interactive benchmark: 1 thread vs max cores (starts at index 14) |
| `make bench-ci` | Bounded CI benchmark (indices 14–26); writes `bin/bench_ci_*.csv` |
| `make prof` | Build profiling binary and run `gprof`; output → `prof_report.txt` |
| `make discover` | Run discover mode (set env vars to control behaviour) |
| `make discover-dry-run` | Print discover plan without running tests |
| `make bucket` | Run power-bucket prime sweep |
| `make bucket-dry-run` | Print bucket plan without running tests |
| `make manual-sweep` | Configurable sweep via environment variables |
| `make clean` | Remove `bin/`, `prof_report.txt`, `gmon.out`, `discover_out/` |

Useful overrides:

```bash
make bench BENCH_START_INDEX=14        # start benchmark at index 14 (p=9689)
make discover LL_MAX_EXPONENT=150000000 LL_SHARD_COUNT=4 LL_SHARD_INDEX=0
make bucket LL_BUCKET_N=5
```

---

## Tests

```bash
make test     # unit + smoke + bounded regression
make unit     # unit tests only (< 1 s)
make smoke    # single-exponent smoke test
make regression  # bounded correctness run (< 15 s on a 2-core runner)
```

The test binary is compiled from `tests/test_bignum.cpp`, which includes
`src/BigNum.cpp` with `BIGNUM_NO_MAIN` defined.

---

## Profiling

```bash
make prof     # builds bin/bignum_prof (-O2 -pg), runs gprof, writes prof_report.txt
```

The profiling binary uses `-O2 -pg` (keeps enough optimisation to be
representative while preserving call structure for `gprof`).

---

## Benchmark

```bash
make bench                          # 1 thread vs max cores, index 14 → end
make bench BENCH_START_INDEX=20     # start later in the list
make bench-ci                       # CI-safe subset (indices 14–26, writes CSV)
scripts/run_benchmark.sh [CSV] [start] [max_index] [threads]
```

---

## CI/CD workflows

All workflows are in `.github/workflows/`.

### `ci.yml` — runs on every push and pull request

| Job | Description |
|---|---|
| `unit-smoke` | Build + unit tests + smoke test (timeout: 10 min) |
| `regression` | Bounded regression: exponents[0..26] (timeout: 15 min) |
| `profile` | Build profiling binary, run `gprof`, upload `prof_report.txt` artifact |
| `bench-ci` | **Optional** — bounded benchmark CSV; only when `workflow_dispatch` with `run_benchmark=true` |

### Other workflows

| File | Trigger | Description |
|---|---|---|
| `discover-primes.yml` | `workflow_dispatch` | Run discover mode; configurable via inputs |
| `power-range-prime-sweep.yml` | `workflow_dispatch` | Power-bucket sweep for one or more buckets |
| `prime-sweep-wrapper.yml` | `workflow_dispatch` | Wrapper that fans out prime-sweep jobs |
| `manual-sweep.yml` | `workflow_dispatch` | Ad-hoc configurable sweep |

---

## Scripts

| Script | Description |
|---|---|
| `scripts/run_benchmark.sh` | Run bounded benchmark, write CSV with thread column |
| `scripts/run_discovery_mode.sh` | Local mirror of the discover-mode workflow |
| `scripts/run_bucket.sh` | Run one power-of-two bucket sweep |
| `scripts/split_bucket_batches.py` | Python plan generator (superseded by `bin/split_bucket_batches`) |
| `scripts/generate_bucket_primes.py` | Enumerate prime exponents in a bucket |
| `scripts/generate_exponent_set.py` | Generate a custom exponent set |
| `scripts/merge_bucket_results.py` | Merge per-bucket result files |

---

## Known Mersenne-prime exponent list

The binary ships with 52 known Mersenne-prime exponents (M_2 through M_136279841).
In default benchmark mode the known list is used directly and
`is_prime_exponent()` is skipped.
Use `LL_MAX_EXPONENT_INDEX` to restrict how far into the list CI runs
(e.g. `LL_MAX_EXPONENT_INDEX=27` stops before M_86243 to keep CI under 15 minutes).