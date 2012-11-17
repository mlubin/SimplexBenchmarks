SimplexBenchmarks
=================
Benchmarks comparing individual operations of the Simplex method for linear programming in Julia and other languages. Uses modified version of **[jlSimplex]** to generate data from real instances.

[jlSimplex]: https://github.com/mlubin/jlSimplex

## To compile

Run ```make``` in ```C++``` directory.

## To run

Be sure to have, ```julia```, ```pypy``` and ```matlab``` in your path. If these aren't present, the code will hang (this is a **[bug]** in Julia). Then run: 

```julia runBenchmarks.jl```

[bug]: https://github.com/JuliaLang/julia/issues/1514

## Timings on a laptop (Intel i5-3320M):

	Geometric mean (relative to C++bnd):
			Julia	C++		C++bnd	matlab	PyPy	Python	
	mtvec:	1.44	0.76	1.00	8.72	4.11	82.12	
	smtvec:	1.29	0.90	1.00	5.79	19.20	417.16	
	rto2:	1.51	0.84	1.00	19.75	4.03	49.91	
	srto2:	1.59	0.96	1.00	13.98	13.81	48.39	
	
	Key:
	mtvec = Matrix-transpose-vector product with non-basic columns
	smtvec = Hyper-sparse matrix-transpose-vector product
	rto2 = Two-pass dual ratio test
	srto2 = Hyper-sparse two-pass dual ratio test
	C++bnd = C++ with bounds checking
