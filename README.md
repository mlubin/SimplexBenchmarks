SimplexBenchmarks
=================
Benchmarks comparing individual operations of the Simplex method for linear programming in Julia and other languages. Uses modified version of **[jlSimplex]** to generate data from real instances.

[jlSimplex]: https://github.com/mlubin/jlSimplex

## To compile

Run make in C++ directory.

## To run

cd GenerateData; julia test.jl

cd ..;

julia Julia/runbench.jl GenerateData/GREENBEA.SIF.dump

C++/runbench GenerateData/GREENBEA.SIF.dump

## Timings on a laptop (Intel i5-3320M):

### Julia
- Matrix transpose-vector product with non-basic columns: 0.05101799964904785 sec
- Hyper-sparse matrix-transpose vector product: 0.04227614402770996 sec

### C++
- Matrix transpose-vector product with non-basic columns: 0.025541 sec
- Matrix transpose-vector product with non-basic columns (with bounds checking): 0.035027 sec
- Hyper-sparse matrix transpose-vector product: 0.028707 sec
- Hyper-sparse matrix transpose-vector product (with bounds checking): 0.032742 sec

### Python (PyPy)
- Matrix transpose-vector product with non-basic columns: 0.148302 sec
- Hyper-sparse matrix transpose-vector product: 0.126729 sec
