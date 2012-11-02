SimplexBenchmarks
=================
Benchmarks comparing individual operations of the Simplex method for linear programming in Julia and other languages. Uses modified version of **[jlSimplex]** to generate data from real instances.

[jlSimplex]: https://github.com/mlubin/jlSimplex

## To compile

Run make in C++ directory.

## To run

julia runBenchmarks.jl

## Timings on a laptop (Intel i5-3320M):

	Geometric mean (relative to C++bnd):
			Julia	C++		C++bnd	PyPy	Python
	mtvec:	1.31	0.76	1.00	4.25	86.33	
	smtvec:	1.20	0.86	1.00	22.49	577.88	
	rto2:	1.49	0.83	1.00	5.18	56.64	
	srto2:	1.45	0.89	1.00	22.07	45.66	

	Key:
	mtvec = Matrix-transpose-vector product with non-basic columns
	smtvec = Hyper-sparse matrix-transpose-vector product
	rto2 = Two-pass dual ratio test
	srto2 = Hyper-sparse two-pass dual ratio test
	C++bnd = C++ with bounds checking
