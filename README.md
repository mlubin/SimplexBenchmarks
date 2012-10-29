SimplexBenchmarks
=================
Benchmarks comparing individual operations of the Simplex method for linear programming in Julia and other languages.

First, run make in C++ directory.

To run:
cd GenerateData; julia test.jl
cd ..;
julia Julia/runbench.jl GenerateData/GREENBEA.SIF.dump
C++/runbench GenerateData/GREENBEA.SIF.dump

