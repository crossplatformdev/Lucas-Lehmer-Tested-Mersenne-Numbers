CXX ?= g++
CXXFLAGS ?= -std=c++20 -O3 -march=native -mtune=native -flto -pthread -Wall -Wextra -Wpedantic
LDFLAGS ?= -flto -pthread

SRC := src/BigNum.cpp
BIN := bin/bignum
TEST_BIN := bin/test_bignum

.PHONY: all clean test

BENCH_START_INDEX ?= 14

all: $(BIN)

$(BIN): $(SRC)
	@mkdir -p bin
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

$(TEST_BIN): tests/test_bignum.cpp $(SRC)
	@mkdir -p bin
	$(CXX) $(CXXFLAGS) tests/test_bignum.cpp -o $@ $(LDFLAGS)

test: $(TEST_BIN) $(BIN)
	./$(TEST_BIN)
	LL_STOP_AFTER_ONE=1 ./$(BIN) 0 1

bench: $(BIN)
	@echo "Benchmark (index=$(BENCH_START_INDEX)): 1 hilo vs máximo cores"
	@set -e; \
		t0=$$(date +%s%N); \
		LL_STOP_AFTER_ONE=1 ./$(BIN) $(BENCH_START_INDEX) 1 >/dev/null; \
		t1=$$(date +%s%N); \
		LL_STOP_AFTER_ONE=1 ./$(BIN) $(BENCH_START_INDEX) 0 >/dev/null; \
		t2=$$(date +%s%N); \
		one_ms=$$(( (t1 - t0) / 1000000 )); \
		max_ms=$$(( (t2 - t1) / 1000000 )); \
		if [ $$max_ms -eq 0 ]; then max_ms=1; fi; \
		speed_x=$$(( one_ms * 100 / max_ms )); \
		echo "1 hilo:      $${one_ms} ms"; \
		echo "max cores:   $${max_ms} ms"; \
		echo "speedup aprox: $$(printf "%d.%02dx" $$((speed_x/100)) $$((speed_x%100)))"

bench-ci: $(BIN)
	@echo "Running CI benchmark"
	@LL_STOP_AFTER_ONE=1 ./$(BIN) $(BENCH_START_INDEX) 1
	@LL_STOP_AFTER_ONE=1 ./$(BIN) $(BENCH_START_INDEX) 0

clean:
	rm -rf bin
