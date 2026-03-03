# BigNum

Implementación de Lucas-Lehmer enfocada en rendimiento para runners con muchos núcleos.

## Ejecución

- Compilar: `make all`
- Ejecutar: `./bin/bignum [start_index] [threads] [progress]`

Comportamiento por defecto:

- Si no se pasa `threads`, usa el máximo de núcleos disponibles (afinidad + hardware).
- Si `threads=0`, también usa el máximo disponible.

## Tests

- Ejecutar tests: `make test`
- Incluye pruebas unitarias y un smoke test del binario principal.

## Benchmark

- Ejecutar benchmark corto: `make bench`
- Compara 1 hilo vs máximo de cores disponibles.
- Puedes ajustar el caso con: `make bench BENCH_START_INDEX=14`

## CI/CD

Hay workflow en `.github/workflows/ci.yml` que se ejecuta en cada:

- `push` (cada commit)
- `pull_request`

El pipeline instala dependencias, compila y corre `make test`.

Benchmark opcional en CI:

- Se puede lanzar manualmente con `workflow_dispatch` activando `run_benchmark`.