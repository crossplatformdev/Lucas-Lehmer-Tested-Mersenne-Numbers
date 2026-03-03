# Copilot Instructions - Lucas-Lehmer Mersenne Benchmark Optimization

## Mission
Redesign the C++ Lucas-Lehmer benchmark for Mersenne numbers for real throughput gains on large exponents. Do not do cosmetic cleanup. Preserve correctness while replacing the current generic big-integer hot path with a production-grade, benchmarked, profiled implementation.

## Current context
- `boost::multiprecision::cpp_int` is the fallback/reference backend (used for small exponents and correctness validation); the hot path is already `FftMersenneBackend` for large exponents and optionally `GmpBackend` for medium sizes.
- Each Lucas-Lehmer iteration is effectively: `s = (s * s - 2) mod (2^p - 1)`.
- The exponent list is a benchmark list of already-known Mersenne-prime exponents, so benchmark mode is not a search engine.
- Current bottlenecks are FFT squaring efficiency, allocation/storage overhead, and avoidable work inside the hot loop.
- Running different exponents on different threads is not the optimal model for a single very large exponent.

## Non-negotiable rules
- Prioritize measurable speedups over refactoring aesthetics.
- Keep `boost::multiprecision::cpp_int` only as a fallback and reference path for validation, not as the main engine for large exponents.
- Preserve correctness at every phase and validate every fast path against a reference path.
- Separate benchmark mode from search mode.
- In benchmark mode, skip unnecessary search-related work and keep results reproducible.

## Required architecture
Implement a backend architecture such as:
- `LucasLehmerEngine`
- `SquareModMersenneBackend`
- `GenericBackend`
- `GmpBackend`
- `FftMersenneBackend`

Add backend auto-selection thresholds based on measured crossovers and an autotuner or benchmark harness that chooses the fastest backend and transform length on the current CPU.

## Required arithmetic strategy
Use a tiered multiplication/squaring strategy:
- Small sizes: optimized schoolbook / Comba squaring
- Medium sizes: Karatsuba / Toom-Cook or GMP `mpn`-level squaring
- Large sizes: Mersenne-specific DWT/FFT, or NTT first if exactness is easier to land initially

For the large backend, exploit `mod (2^p - 1)` directly:
- Do not compute a generic `2p`-bit square and then reduce.
- Compute `square mod (2^p - 1)` directly as a cyclic or weighted convolution.
- Use a Crandall-Fagin / DWT-style representation so wraparound reduction is built into the transform path.

## Hot-path engineering requirements
Treat the Lucas-Lehmer kernel as a production-performance loop:
- in-place operations
- aligned memory
- contiguous SIMD-friendly buffers
- scratch buffers reused across iterations
- memory pools where useful
- minimal allocations
- cache-aware layout
- no repeated big-object construction/destruction inside the LL loop

Move from arbitrary-precision objects to plain limb or digit arrays.
Use 64-bit limbs or FFT-friendly floating chunks depending on backend.
Align buffers to cache-line and SIMD boundaries.
Eliminate false sharing.
Use prefetching only if it benchmarks faster.
Consider huge pages for very large transforms.
Use `-O3 -march=native`, and where appropriate LTO/PGO. Do not enable unsafe math optimizations unless validated against the reference path.
For non-FFT backends, consider tuned carry chains and BMI2/ADX paths such as `MULX`, `ADCX`, and `ADOX` if they are measurably faster.

## FFT / DWT backend requirements
- Choose radix/chunk sizes for transform efficiency, bounded carries, and safe rounding margins.
- Prefer balanced digits to reduce carry pressure and floating-point error.
- Exploit squaring-specific shortcuts, including symmetry and real-input optimizations where valid.
- Precompute and reuse twiddles, DWT weights, bit-reversal/permutation tables, and transform plans.
- Support multiple FFT lengths and pick the best smooth length per exponent.
- Consider split-radix or mixed-radix FFT, real FFT optimizations, FMA-enabled kernels, and AVX2 / AVX-512 vectorization.
- If using floating FFT, implement roundoff/error tracking, normalization checks, checkpoint residue validation, and an optional cross-check path.
- Keep an exact validation path via GMP/mpn or NTT.

## Lucas-Lehmer loop requirements
Fuse each iteration into a single hot operation:
- implemented as a single backend call (e.g. `backend.step(state)`) that performs `s = (s * s - 2) mod (2^p - 1)`

Also:
- avoid recomputing masks and temporaries
- avoid reallocating temporary buffers
- remove frequent timing calls from the hot loop
- make progress output optional and sparse
- replace `std::time` / `difftime` with `steady_clock`
- add checkpointing every N iterations with residue, elapsed time, max roundoff seen, and resumability
- ensure worker threads honor the caller's `progress` flag when invoking `lucas_lehmer(p, progress)`; avoid regressing to forcing `progress = true`
- skip `is_prime_exponent()` in benchmark mode when using the known-prime exponent list

## Parallelism requirements
For a single large exponent, parallelize the square/convolution itself rather than only running independent exponents.

Implement and benchmark:
- thread pool
- CPU affinity / pinning
- optional NUMA awareness
- barriers only where needed
- per-thread scratch buffers to avoid false sharing

Measure both:
- throughput mode: many exponents concurrently
- latency mode: one exponent using all cores

Report which mode wins by size range.

## Profiling requirements - mandatory
You must profile before changing the code, profile after every meaningful change, and profile the final result. Do not guess.

### Required profiler
Use the GNU/g++ toolchain profiler as a hard requirement:
- build a dedicated profiling target with `g++` profiling enabled
- use `-pg` and collect results with `gprof`

`gprof` is required. You may use additional profilers if helpful, but `gprof` must be part of the workflow and must appear in the report.

### What to profile
Profile every hot operation and every measurable atomic sub-operation in the kernel, not just the whole program. Break down and benchmark, at minimum:
- square
- square-sub-2 fused op
- Mersenne reduction / fold
- carry propagation
- normalization
- transform setup vs steady-state transform cost
- FFT forward/inverse stages
- pointwise squaring
- twiddle/table access overhead
- buffer allocation/reuse overhead
- thread synchronization and barrier cost
- per-backend dispatch overhead

Where useful, create isolated microbenchmarks for individual primitives and report per-operation metrics such as:
- ns/op
- cycles/op
- calls
- percent of total runtime
- speedup vs baseline

### Profiling workflow requirements
For every optimization phase:
1. capture a clean baseline
2. apply one substantial change
3. rerun the same benchmark set
4. rerun `gprof`
5. report before/after results
6. keep or revert the change based on measured results

Do not merge or present changes without a before/after performance comparison.

## Benchmark and reporting requirements
Produce a benchmark report against the original implementation, including:
- per exponent
- per backend
- per thread count
- memory usage
- speedup
- throughput mode vs latency mode
- profiler findings for the top hot functions and atomic operations

The report must clearly show:
- original runtime
- new runtime
- absolute improvement
- percentage improvement
- which change produced each gain

## Workflow and job-link requirements
When run in the workflow environment, always publish the workflow job/run link along with the benchmark results.

At minimum:
- include the workflow run URL in the final report / PR comment / summary
- include the specific job name
- attach links to benchmark artifacts generated by the workflow environment
- attach or reference profiler outputs and benchmark logs

If running in GitHub Actions, construct and print the run link from environment variables when needed, for example using:
- `GITHUB_SERVER_URL`
- `GITHUB_REPOSITORY`
- `GITHUB_RUN_ID`
- `GITHUB_JOB`

Expected output should include something like:
- workflow run link
- job name
- artifact names/links
- before/after benchmark table
- before/after profiler summary

## Correctness and validation requirements
Validate every optimized backend against the reference path.
Add tests for:
- Generic vs GMP vs FFT backend equivalence
- selected LL residues at chosen iterations
- random intermediate states
- Mersenne reduction correctness
- square-mod equivalence
- long-run stress cases to catch transform roundoff issues

## Search-mode scope
Keep benchmark mode pure and reproducible.
If search mode exists, it may include:
- trial factoring
- optional P-1
- optional PRP before LL

Do not let search-mode overhead contaminate benchmark-mode measurements.

## Implementation order
Implement in phases:
1. establish a clean baseline benchmark and profiler report
2. remove obvious benchmark overhead
3. introduce limb-array backend
4. add GMP/mpn backend
5. add FFT/DWT backend
6. add multithreaded FFT/convolution
7. autotune thresholds and transform selection
8. finalize validation and publish before/after benchmark and profiler reports

## Code-specific notes that must be preserved
- The exponent list is already a list of known Mersenne-prime exponents, so benchmark mode must continue to skip `is_prime_exponent()` checks.
- Worker threads must continue to honor the caller's `progress` flag; do not regress to forcing `progress = true`.
- For very large exponents, the right parallelization target is the multiply/square itself, not independent LL chains.

## Final deliverables
1. Redesigned C++ implementation with the backend architecture above
2. Benchmark report versus the original code
3. Profiler report using `gprof`, including hot functions and atomic-operation breakdowns
4. Workflow-environment output that includes the job/run link and links or references to benchmark/profiler artifacts
5. Short engineering note covering thresholds, transform strategy, error control, and why the final design is faster

## Working style
- Prefer phased, measurable changes.
- Keep a running changelog of performance wins and losses.
- If a change does not make the code faster or safer, do not keep it.
- Do not stop at macro benchmarks; inspect and optimize the atomic operations that dominate runtime.
