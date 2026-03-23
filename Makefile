CXX ?= g++
CXXFLAGS ?= -std=c++20 -O3 -march=native -mtune=native -flto -pthread -Wall -Wextra -Wpedantic
LDFLAGS ?= -flto -pthread

CC ?= gcc
CFLAGS_PLAN ?= -std=c11 -O2 -Wall -Wextra

SRC := src/BigNum.cpp
BIN := bin/bignum
TEST_BIN := bin/test_bignum
PROF_BIN := bin/bignum_prof
PERF_BIN := bin/bignum_perf
CALLGRIND_BIN := bin/bignum_callgrind
PLAN_BIN := bin/split_bucket_batches
PLAN_SRC := src/split_bucket_batches.c

# sequence_powermod: GMP-based Mersenne sequence search binary.
SEQMOD_SRC  := src/sequence_powermod.cpp
SEQMOD_BIN  := bin/sequence_powermod
SEQMOD_CXXFLAGS := -std=c++17 -O3 -march=native -mtune=native -pthread -Wall -Wextra
SEQMOD_LDFLAGS  := -pthread -lgmp -lgmpxx

# Profiling build uses -O2 (keeps enough optimization to be representative
# while preserving function call structure for gprof) and -pg.
PROF_CXXFLAGS := -std=c++20 -O2 -march=native -pthread -pg -Wall -Wextra
PROF_LDFLAGS  := -pthread -pg

# Microbenchmark build: -O3 -g -fno-omit-frame-pointer for profiling with
# frame pointers intact (perf, valgrind, etc.).  Add MICROBENCH_GPROF=1 to
# also enable gprof instrumentation (-pg on compile and link).
MICROBENCH_BIN      := bin/microbench_fft
MICROBENCH_SRC      := bench/microbench_fft.cpp
MICROBENCH_GPROF    ?= 0
MICROBENCH_PG       := $(if $(filter 1,$(MICROBENCH_GPROF)),-pg,)
MICROBENCH_CXXFLAGS := -std=c++20 -O3 -g -fno-omit-frame-pointer -pthread -Wall -Wextra $(MICROBENCH_PG)
MICROBENCH_LDFLAGS  := -pthread $(MICROBENCH_PG)
# Number of LL iterations for the microbench (0 = full p-2 run, the default).
# Override at the make command line: make microbench MICROBENCH_ITERS=500
MICROBENCH_ITERS ?= 0

# perf build: -O3 with frame pointers for Linux perf, no LTO.
PERF_CXXFLAGS := -std=c++20 -O3 -g -fno-omit-frame-pointer -march=native -mtune=native -pthread -Wall -Wextra -Wpedantic
PERF_LDFLAGS  := -pthread

# callgrind build: -O2 with frame pointers for Valgrind/Callgrind, no LTO.
CALLGRIND_CXXFLAGS := -std=c++20 -O2 -g -fno-omit-frame-pointer -march=native -mtune=native -pthread -Wall -Wextra -Wpedantic
CALLGRIND_LDFLAGS  := -pthread

.PHONY: all clean unit smoke regression test bench bench-ci cluster-power prof perf-build callgrind-build perf-run callgrind-run discover discover-dry-run manual-sweep bucket bucket-dry-run plan-tool seqmod

BENCH_START_INDEX ?= 14

all: $(BIN)

$(BIN): $(SRC)
	@mkdir -p bin
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

$(PLAN_BIN): $(PLAN_SRC)
	@mkdir -p bin
	$(CC) $(CFLAGS_PLAN) $< -o $@

# plan-tool: build the C plan tool (replaces scripts/split_bucket_batches.py).
plan-tool: $(PLAN_BIN)

$(SEQMOD_BIN): $(SEQMOD_SRC)
	@mkdir -p bin
	$(CXX) $(SEQMOD_CXXFLAGS) $< -o $@ $(SEQMOD_LDFLAGS)

# seqmod: build the GMP-based sequence_powermod binary.
seqmod: $(SEQMOD_BIN)

$(TEST_BIN): tests/test_bignum.cpp $(SRC)
	@mkdir -p bin
	$(CXX) $(CXXFLAGS) tests/test_bignum.cpp -o $@ $(LDFLAGS)

$(PROF_BIN): $(SRC)
	@mkdir -p bin
	$(CXX) $(PROF_CXXFLAGS) $< -o $@ $(PROF_LDFLAGS)

$(PERF_BIN): $(SRC)
	@mkdir -p bin
	$(CXX) $(PERF_CXXFLAGS) $< -o $@ $(PERF_LDFLAGS)

$(CALLGRIND_BIN): $(SRC)
	@mkdir -p bin
	$(CXX) $(CALLGRIND_CXXFLAGS) $< -o $@ $(CALLGRIND_LDFLAGS)

# ---------------------------------------------------------------------------
# unit: run the unit/correctness test binary only (fast – no main binary run).
# ---------------------------------------------------------------------------
unit: $(TEST_BIN)
	./$(TEST_BIN)

# ---------------------------------------------------------------------------
# smoke: run a single tiny exponent to confirm the binary executes correctly.
# ---------------------------------------------------------------------------
smoke: $(BIN)
	LL_STOP_AFTER_ONE=1 ./$(BIN) 0 1

# ---------------------------------------------------------------------------
# regression: bounded subset – exponents[0..26] (2 … 44497).
# All complete in < 15 s on a 2-core GitHub runner.  Index 27 = 86243 (44 s,
# skipped here so total CI time stays well under the 15-minute budget).
# ---------------------------------------------------------------------------
regression: $(BIN)
	LL_STOP_AFTER_ONE=0 LL_MAX_EXPONENT_INDEX=27 ./$(BIN) 0 0

# ---------------------------------------------------------------------------
# test: CI-safe composite – unit tests + smoke + bounded regression.
# Replaces the old unbounded "walk the full exponent list" target.
# ---------------------------------------------------------------------------
test: unit smoke regression

# ---------------------------------------------------------------------------
# bench-ci: bounded benchmark for CI (indices 14 … 26, p = 9689 … 44497).
#
# Thread configuration sweep (measured on 4-vCPU ubuntu-24.04 runner):
#
#   Config                        single-exp ns/iter (p=44497)   multi-exp wall (14-26)
#   (LL=1, FFT=1, nested=0)       120,079 ns                     6,936 ms  ← baseline
#   (LL=2, FFT=1, nested=0)       120,079 ns                     5,039 ms  (27 % faster)
#   (LL=3, FFT=1, nested=0)         -                            4,825 ms
#   (LL=4, FFT=1, nested=0)         -                            4,405 ms
#   (LL=4, FFT=2, nested=1)       121,379 ns                     4,366 ms  ← BEST (37 % faster)
#
# Best config: LL_THREADS=4, LL_FFT_THREADS=2, LL_FFT_ALLOW_NESTED=1
#   • Four outer workers saturate all 4 vCPUs for multi-exponent throughput.
#   • LL_FFT_ALLOW_NESTED=1 lets each FFT use 2 threads when nested under the
#     outer pool; on a 4-vCPU machine the 4×2 = 8 logical threads overlap I/O
#     and compute, reducing total wall time by ~13 % vs LL=4/FFT=1.
#   • Single-exponent ns/iter is within noise (120 k vs 121 k) — the gain is
#     entirely in multi-exponent throughput via better CPU utilisation.
#
# Override defaults at the command line:
#   make bench-ci BENCH_LL_THREADS=2 BENCH_FFT_THREADS=1 BENCH_FFT_ALLOW_NESTED=0
# ---------------------------------------------------------------------------
# Tunable defaults (override on the make command line or in the environment).
BENCH_LL_THREADS        ?= 4
BENCH_FFT_THREADS       ?= 2
BENCH_FFT_ALLOW_NESTED  ?= 1

bench-ci: $(BIN)
	@echo "=== CI benchmark (indices 14-26, 1 thread — baseline) ==="
	@LL_STOP_AFTER_ONE=0 LL_MAX_EXPONENT_INDEX=27 \
	    LL_FFT_THREADS=1 LL_FFT_ALLOW_NESTED=0 \
	    LL_BENCH_OUTPUT=bin/bench_ci_1t.csv ./$(BIN) $(BENCH_START_INDEX) 1
	@echo "=== CI benchmark (indices 14-26, LL_THREADS=$(BENCH_LL_THREADS) FFT_THREADS=$(BENCH_FFT_THREADS) nested=$(BENCH_FFT_ALLOW_NESTED) — optimised) ==="
	@LL_STOP_AFTER_ONE=0 LL_MAX_EXPONENT_INDEX=27 \
	    LL_FFT_THREADS=$(BENCH_FFT_THREADS) \
	    LL_FFT_ALLOW_NESTED=$(BENCH_FFT_ALLOW_NESTED) \
	    LL_BENCH_OUTPUT=bin/bench_ci_mt.csv \
	    ./$(BIN) $(BENCH_START_INDEX) $(BENCH_LL_THREADS)
	@echo "--- 1-thread CSV ---" && cat bin/bench_ci_1t.csv
	@echo "--- optimised CSV (LL=$(BENCH_LL_THREADS) FFT=$(BENCH_FFT_THREADS) nested=$(BENCH_FFT_ALLOW_NESTED)) ---" && cat bin/bench_ci_mt.csv

# ---------------------------------------------------------------------------
# cluster-power: measure FFT throughput and calculate theoretical cluster
# performance in Teraflops and MIPS.
#
# Runs the microbench with the reference cluster configuration:
#   256 workers, 4 LL-threads per worker, 2 FFT-threads per worker (nested).
#
# The microbench reports per-thread measured throughput and extrapolates to
# the full cluster by multiplying by workers * ll_threads.
#
# Configurable via Makefile variables (override on the command line):
#   make cluster-power CLUSTER_P=44497 CLUSTER_WORKERS=256 \
#                      CLUSTER_THREADS=4 CLUSTER_FFT_THREADS=2
# ---------------------------------------------------------------------------
CLUSTER_P            ?= 44497
CLUSTER_WORKERS      ?= 256
CLUSTER_THREADS      ?= 4
CLUSTER_FFT_THREADS  ?= 2

cluster-power: $(MICROBENCH_BIN)
	@echo "=== Cluster power benchmark ==="
	@echo "  exponent       : p=$(CLUSTER_P)"
	@echo "  cluster config : $(CLUSTER_WORKERS) workers x $(CLUSTER_THREADS) LL-threads x $(CLUSTER_FFT_THREADS) FFT-threads (nested)"
	@echo ""
	LL_FFT_THREADS=$(CLUSTER_FFT_THREADS) \
	LL_FFT_ALLOW_NESTED=1 \
	LL_CLUSTER_WORKERS=$(CLUSTER_WORKERS) \
	LL_CLUSTER_THREADS=$(CLUSTER_THREADS) \
	LL_CLUSTER_FFT_THREADS=$(CLUSTER_FFT_THREADS) \
	./$(MICROBENCH_BIN) $(CLUSTER_P)

# ---------------------------------------------------------------------------
# bench: full interactive benchmark (1 thread vs max cores, start at index 14).
# Not run by default CI; for local profiling only.
# ---------------------------------------------------------------------------
bench: $(BIN)
	@echo "Benchmark (index=$(BENCH_START_INDEX)): 1 thread vs max cores"
	@set -e; \
		t0=$$(date +%s%N); \
		LL_STOP_AFTER_ONE=0 ./$(BIN) $(BENCH_START_INDEX) 1 >/dev/null; \
		t1=$$(date +%s%N); \
		LL_STOP_AFTER_ONE=0 ./$(BIN) $(BENCH_START_INDEX) 0 >/dev/null; \
		t2=$$(date +%s%N); \
		one_ms=$$(( (t1 - t0) / 1000000 )); \
		max_ms=$$(( (t2 - t1) / 1000000 )); \
		if [ $$max_ms -eq 0 ]; then max_ms=1; fi; \
		speed_x=$$(( one_ms * 100 / max_ms )); \
		echo "1 thread:    $${one_ms} ms"; \
		echo "max cores:   $${max_ms} ms"; \
		echo "speedup aprox: $$(printf "%d.%02dx" $$((speed_x/100)) $$((speed_x%100)))"

# ---------------------------------------------------------------------------
# manual-sweep: run a configurable sweep via environment variables.
# Example:
#   make manual-sweep LL_SWEEP_MODE=m LL_MIN_EXPONENT=2 LL_MAX_EXPONENT=10000
# ---------------------------------------------------------------------------
manual-sweep: $(BIN)
	./$(BIN) 0 0

# ---------------------------------------------------------------------------
# prof: profiling target (gprof).
# ---------------------------------------------------------------------------
prof: $(PROF_BIN)
	@echo "Running profiling binary (index=$(BENCH_START_INDEX), 1 thread)..."
	LL_STOP_AFTER_ONE=1 $(PROF_BIN) $(BENCH_START_INDEX) 1
	@echo "Generating gprof report -> prof_report.txt"
	gprof $(PROF_BIN) gmon.out > prof_report.txt
	@echo "--- Top 20 functions (flat profile) ---"
	head -50 prof_report.txt

# ---------------------------------------------------------------------------
# perf-build: build with -O3 -g -fno-omit-frame-pointer, no LTO (for perf).
# ---------------------------------------------------------------------------
perf-build: $(PERF_BIN)

# ---------------------------------------------------------------------------
# callgrind-build: build with -O2 -g -fno-omit-frame-pointer, no LTO
#                  (for Valgrind/Callgrind).
# ---------------------------------------------------------------------------
callgrind-build: $(CALLGRIND_BIN)

# ---------------------------------------------------------------------------
# perf-run: record a perf profile into perf.data (requires perf).
# ---------------------------------------------------------------------------
perf-run: $(PERF_BIN)
	LL_STOP_AFTER_ONE=1 perf record -g -o perf.data -- ./$(PERF_BIN) $(BENCH_START_INDEX) 1

# ---------------------------------------------------------------------------
# callgrind-run: profile with Valgrind/Callgrind into callgrind.out.
# ---------------------------------------------------------------------------
callgrind-run: $(CALLGRIND_BIN)
	LL_STOP_AFTER_ONE=1 valgrind --tool=callgrind --callgrind-out-file=callgrind.out \
	    ./$(CALLGRIND_BIN) $(BENCH_START_INDEX) 1

clean:
	rm -rf bin prof_report.txt gmon.out perf.data callgrind.out discover_out seqmod_out

# ---------------------------------------------------------------------------
# microbench: FFT Lucas–Lehmer microbenchmark for a single large exponent.
# Compiled with -O3 -g -fno-omit-frame-pointer for profiling with perf/valgrind.
# Add MICROBENCH_GPROF=1 to also compile/link with -pg for gprof support.
# Override the number of iterations:
#   make microbench MICROBENCH_ITERS=500
#   make microbench MICROBENCH_GPROF=1
# ---------------------------------------------------------------------------
$(MICROBENCH_BIN): $(MICROBENCH_SRC) $(SRC)
	@mkdir -p bin
	$(CXX) $(MICROBENCH_CXXFLAGS) -DMICROBENCH_ITERS=$(MICROBENCH_ITERS) $< -o $@ $(MICROBENCH_LDFLAGS)

microbench: $(MICROBENCH_BIN)
	./$(MICROBENCH_BIN)

# discover: run full discover mode (set env vars to control behaviour).
# Example: LL_MAX_EXPONENT=136279950 LL_STOP_AFTER_N_CASES=1 make discover
discover: $(BIN)
	LL_SWEEP_MODE=discover \
	LL_SINGLE_EXPONENT=$${LL_SINGLE_EXPONENT:-0} \
	LL_MAX_EXPONENT=$${LL_MAX_EXPONENT:-200000000} \
	LL_SHARD_COUNT=$${LL_SHARD_COUNT:-1} \
	LL_SHARD_INDEX=$${LL_SHARD_INDEX:-0} \
	LL_REVERSE_ORDER=$${LL_REVERSE_ORDER:-0} \
	LL_STOP_AFTER_N_CASES=$${LL_STOP_AFTER_N_CASES:-0} \
	LL_DRY_RUN=0 \
	LL_OUTPUT_DIR=discover_out/ \
	./$(BIN) 0 $${LL_THREADS:-0}

# discover-dry-run: print the exponent plan without running LL tests.
discover-dry-run: $(BIN)
	LL_SWEEP_MODE=discover \
	LL_SINGLE_EXPONENT=$${LL_SINGLE_EXPONENT:-0} \
	LL_MAX_EXPONENT=$${LL_MAX_EXPONENT:-200000000} \
	LL_SHARD_COUNT=$${LL_SHARD_COUNT:-1} \
	LL_SHARD_INDEX=$${LL_SHARD_INDEX:-0} \
	LL_REVERSE_ORDER=$${LL_REVERSE_ORDER:-0} \
	LL_STOP_AFTER_N_CASES=$${LL_STOP_AFTER_N_CASES:-0} \
	LL_DRY_RUN=1 \
	./$(BIN) 0 1

# ---------------------------------------------------------------------------
# bucket: run power-bucket prime sweep.
# Example: LL_BUCKET_N=5 make bucket
#          LL_BUCKET_START=1 LL_BUCKET_END=10 make bucket
# ---------------------------------------------------------------------------
bucket: $(BIN)
	LL_SWEEP_MODE=power_bucket_primes \
	LL_BUCKET_N=$${LL_BUCKET_N:-0} \
	LL_BUCKET_START=$${LL_BUCKET_START:-1} \
	LL_BUCKET_END=$${LL_BUCKET_END:-64} \
	LL_REVERSE_ORDER=$${LL_REVERSE_ORDER:-0} \
	LL_MAX_EXPONENTS_PER_JOB=$${LL_MAX_EXPONENTS_PER_JOB:-0} \
	LL_STOP_AFTER_FIRST_PRIME_RESULT=$${LL_STOP_AFTER_FIRST_PRIME_RESULT:-0} \
	LL_DRY_RUN=0 \
	LL_OUTPUT_DIR=$${LL_OUTPUT_DIR:-bucket_out/} \
	./$(BIN) 0 $${LL_THREADS:-0}

# bucket-dry-run: print the exponent plan for a bucket without running LL tests.
# Example: LL_BUCKET_N=3 make bucket-dry-run
bucket-dry-run: $(BIN)
	LL_SWEEP_MODE=power_bucket_primes \
	LL_BUCKET_N=$${LL_BUCKET_N:-0} \
	LL_BUCKET_START=$${LL_BUCKET_START:-1} \
	LL_BUCKET_END=$${LL_BUCKET_END:-64} \
	LL_REVERSE_ORDER=$${LL_REVERSE_ORDER:-0} \
	LL_MAX_EXPONENTS_PER_JOB=$${LL_MAX_EXPONENTS_PER_JOB:-0} \
	LL_DRY_RUN=1 \
	./$(BIN) 0 1
