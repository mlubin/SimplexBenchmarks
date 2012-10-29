#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cstdint>
#include <chrono>
#include <functional>
#include <cassert>

using namespace std;


struct SparseMatrixCSC {
	int nrow, ncol;
	vector<int64_t> colptr, rowval;
	vector<double> nzval;
};

enum VariableState { Basic = 1, AtLower = 2, AtUpper = 3 };

struct IterationData {
	vector<VariableState> variableState;
	vector<double> priceInput;
	bool valid;
};

SparseMatrixCSC readMat(istream &f) {
	
	SparseMatrixCSC A;
	int64_t nnz;
	f >> A.nrow;
	f >> A.ncol;
	f >> nnz;

	A.colptr.reserve(A.ncol+1);
	A.rowval.reserve(nnz);
	A.nzval.reserve(nnz);
	
	string line;
	getline(f,line);
	getline(f,line);
	{
		istringstream ss(line);
		
		for (int i = 0; i <= A.ncol; i++) {
			int64_t x;
			ss >> x;
			A.colptr.push_back(x-1); // adjust 1-based indices
			assert(!ss.fail() && !ss.bad());
		}
	}

	getline(f,line);
	{
		istringstream ss(line);
		
		for (int i = 0; i < nnz; i++) {
			int64_t x;
			ss >> x;
			A.rowval.push_back(x-1);
			assert(!ss.fail() && !ss.bad());
		}
	}

	getline(f,line);
	{
		istringstream ss(line);
		
		for (int i = 0; i < nnz; i++) {
			double x;
			ss >> x;
			A.nzval.push_back(x);
			assert(!ss.fail() && !ss.bad());
		}
	}

	return A;
}

IterationData readIteration(istream &f, SparseMatrixCSC const &A) {
	
	IterationData d;
	d.valid = true;
	string line;
	getline(f,line);
	{
		istringstream ss(line);
		
		for (int i = 0; i < A.nrow+A.ncol; i++) {
			int x;
			ss >> x;
			if (ss.fail()) {
				d.valid = false; return d;
			}
			d.variableState.push_back(static_cast<VariableState>(x));
		}
		
	}
	getline(f,line);
	{
		istringstream ss(line);
		
		for (int i = 0; i < A.nrow; i++) {
			double x;
			ss >> x;
			if (ss.fail()) {
				d.valid = false; return d;
			}
			d.priceInput.push_back(x);
		}
	}

	return d;

}

chrono::nanoseconds doPrice(SparseMatrixCSC const& A, IterationData const& d) {

	vector<double> output(A.nrow+A.ncol,0.);

	auto t = chrono::high_resolution_clock::now();

	for (int i = 0; i < A.ncol; i++) {
		if (d.variableState[i] == Basic) continue;
		double val = 0.;
		for (int64_t k = A.colptr[i]; k < A.colptr[i+1]; k++) {
			val += d.priceInput[A.rowval[k]]*A.nzval[k];
		}
		output[i] = val;
	}
	for (int i = 0; i < A.nrow; i++) {
		int k = i + A.ncol;
		if (d.variableState[i] == Basic) continue;
		output[k] = -d.priceInput[i];
	}

	auto t2 = chrono::high_resolution_clock::now();
	return chrono::duration_cast<chrono::nanoseconds>(t2-t);

}

chrono::nanoseconds doPriceBoundsCheck(SparseMatrixCSC const& A, IterationData const& d) {

	vector<double> output(A.nrow+A.ncol,0.);

	auto t = chrono::high_resolution_clock::now();

	for (int i = 0; i < A.ncol; i++) {
		if (d.variableState.at(i) == Basic) continue;
		double val = 0.;
		for (int64_t k = A.colptr[i]; k < A.colptr[i+1]; k++) {
			val += d.priceInput.at(A.rowval.at(k))*A.nzval.at(k);
		}
		output.at(i) = val;
	}
	for (int i = 0; i < A.nrow; i++) {
		int k = i + A.ncol;
		if (d.variableState.at(i) == Basic) continue;
		output.at(k) = -d.priceInput[i];
	}

	auto t2 = chrono::high_resolution_clock::now();
	return chrono::duration_cast<chrono::nanoseconds>(t2-t);

}

struct BenchmarkOperation {
	function<chrono::nanoseconds(SparseMatrixCSC const&, IterationData const&)> func;
	string name;
};


int main(int argc, char**argv) {

	assert(argc == 2);
	ifstream f(argv[1]);
	
	SparseMatrixCSC A = readMat(f);
	cout << "Problem is " << A.nrow << " by " << A.ncol << " with " << A.nzval.size() << " nonzeros\n";
	
	vector<BenchmarkOperation> benchmarks{ 
		{ doPrice, "Matrix transpose-vector product with non-basic columns" },
		{ doPriceBoundsCheck, "Matrix transpose-vector product with non-basic columns (with bounds checking)" }
	};
	vector<chrono::nanoseconds> timings(benchmarks.size(), chrono::nanoseconds::zero());

	int nruns = 0;
	while (true) {
		IterationData dat = readIteration(f,A);
		if (!dat.valid) break;
		for (int i = 0; i < benchmarks.size(); i++) {
			timings[i] += benchmarks[i].func(A,dat);
		}
		nruns++;
	}

	cout << nruns << " simulated iterations. Total timings:\n";
	for (int i = 0; i < benchmarks.size(); i++) {
		cout << benchmarks[i].name << ": " << timings[i].count()/1000000000. << " sec\n";
	}

	return 0;

}
