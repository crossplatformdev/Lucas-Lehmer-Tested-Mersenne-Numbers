CXX ?= g++
CXXFLAGS ?= -std=c++20 -O3 -march=native -mtune=native -flto -pthread -Wall -Wextra -Wpedantic
LDFLAGS ?= -flto -pthread

CC ?= gcc
CFLAGS_PLAN ?= -std=c11 -O2 -Wall -Wextra

SRC := src/BigNum.cpp
BIN := bin/bignum
TEST_BIN := bin/test_bignum
PROF_BIN := bin/bignum_prof
PLAN_BIN := bin/split_bucket_batches
PLAN_SRC := src/split_bucket_batches.c

# Profiling build uses -O2 (keeps enough optimization to be representative
# while preserving function call structure for gprof) and -pg.
PROF_CXXFLAGS := -std=c++20 -O2 -march=native -pthread -pg -Wall -Wextra
PROF_LDFLAGS  := -pthread -pg

.PHONY: all clean  unit smoke regression test bench bench-ci prof discover discover-dry-run manual-sweep bucket bucket-dry-run plan-tool

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

$(TEST_BIN): tests/test_bignum.cpp $(SRC)
	@mkdir -p bin
	$(CXX) $(CXXFLAGS) tests/test_bignum.cpp -o $@ $(LDFLAGS)

$(PROF_BIN): $(SRC)
	@mkdir -p bin
	$(CXX) $(PROF_CXXFLAGS) $< -o $@ $(PROF_LDFLAGS)

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
# bench-ci: bounded benchmark for CI (indices 14 … 26, 1 thread & max cores).
# Index 14 = p=9689 (~0.3 s), index 26 = p=44497 (~9 s); runs quickly.
# Writes machine-readable CSV to bin/bench_ci.csv.
# ---------------------------------------------------------------------------
bench-ci: $(BIN)
	@echo "=== CI benchmark (indices 14-26, 1 thread) ==="
	@LL_STOP_AFTER_ONE=0 LL_MAX_EXPONENT_INDEX=27 \
	    LL_BENCH_OUTPUT=bin/bench_ci_1t.csv ./$(BIN) $(BENCH_START_INDEX) 1
	@echo "=== CI benchmark (indices 14-26, max threads) ==="
	@LL_STOP_AFTER_ONE=0 LL_MAX_EXPONENT_INDEX=27 \
	    LL_BENCH_OUTPUT=bin/bench_ci_mt.csv ./$(BIN) $(BENCH_START_INDEX) 0
	@echo "--- 1-thread CSV ---" && cat bin/bench_ci_1t.csv
	@echo "--- max-thread CSV ---" && cat bin/bench_ci_mt.csv

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

clean:
	rm -rf bin prof_report.txt gmon.out discover_out

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
