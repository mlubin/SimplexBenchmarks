#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cstdint>
#include <chrono>
#include <functional>
#include <cassert>
#include <cmath>

using namespace std;


struct SparseMatrixCSC {
	int nrow, ncol;
	vector<int64_t> colptr, rowval;
	vector<double> nzval;
};

enum VariableState { Basic = 1, AtLower = 2, AtUpper = 3 };

struct InstanceData {
	SparseMatrixCSC A, Atrans;
};

struct IterationData {
	vector<VariableState> variableState;
	vector<double> priceInput;
	bool valid;
};

struct IndexedVector {
	vector<double> elts;
	vector<int> nzidx;
	int nnz;
	IndexedVector(vector<double> const& densevec) : nzidx(densevec.size(),0), elts(densevec.size(),0.), nnz(0) {
		for (int i = 0; i < densevec.size(); i++) {
			if (fabs(densevec[i]) > 1e-50) {
				nzidx[nnz++] = i;
				elts[i] = densevec[i];
			}
		}
	}
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

InstanceData readInstance(istream &f) {
	SparseMatrixCSC A = readMat(f);
	SparseMatrixCSC Atrans = readMat(f);
	return { A, Atrans };
}

IterationData readIteration(istream &f, InstanceData const &instance) {
	
	SparseMatrixCSC const& A = instance.A;
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

chrono::nanoseconds doPrice(InstanceData const& instance, IterationData const& d) {

	SparseMatrixCSC const &A = instance.A;
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

chrono::nanoseconds doPriceBoundsCheck(InstanceData const& instance, IterationData const& d) {

	SparseMatrixCSC const &A = instance.A;
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
		output.at(k) = -d.priceInput.at(i);
	}

	auto t2 = chrono::high_resolution_clock::now();
	return chrono::duration_cast<chrono::nanoseconds>(t2-t);

}

chrono::nanoseconds doPriceHypersparse(InstanceData const& instance, IterationData const& d) {

	SparseMatrixCSC const &A = instance.A;
	SparseMatrixCSC const &Atrans = instance.Atrans;
	vector<double> outputelts(A.nrow+A.ncol,0.);
	vector<int> outputnzidx(A.nrow+A.ncol,0);
	int outputnnz = 0;

	IndexedVector rho(d.priceInput);


	auto t = chrono::high_resolution_clock::now();

	for (int k = 0; k < rho.nnz; k++) {
		int row = rho.nzidx[k];
		double elt = rho.elts[row];
		for (int64_t j = Atrans.colptr[row]; j < Atrans.colptr[row+1]; j++) {
			int idx = Atrans.rowval[j];
			if (outputelts[idx] != 0.) {
				outputelts[idx] += elt*Atrans.nzval[j];
				//if (outputelts[idx] == 0.) outputelts[idx] = 1e-50;
			} else {
				outputelts[idx] = elt*Atrans.nzval[j];
				assert(outputelts[idx] != 0.);
				outputnzidx[outputnnz++] = idx;
			}
		}
		outputelts[row+A.ncol] = -elt;
		outputnzidx[outputnnz++] = row+A.ncol;
	}


	auto t2 = chrono::high_resolution_clock::now();
	return chrono::duration_cast<chrono::nanoseconds>(t2-t);

}

chrono::nanoseconds doPriceHypersparseBoundsCheck(InstanceData const& instance, IterationData const& d) {

	SparseMatrixCSC const &A = instance.A;
	SparseMatrixCSC const &Atrans = instance.Atrans;
	vector<double> outputelts(A.nrow+A.ncol,0.);
	vector<int> outputnzidx(A.nrow+A.ncol,0);
	int outputnnz = 0;

	IndexedVector rho(d.priceInput);


	auto t = chrono::high_resolution_clock::now();

	for (int k = 0; k < rho.nnz; k++) {
		int row = rho.nzidx.at(k);
		double elt = rho.elts.at(row);
		for (int64_t j = Atrans.colptr[row]; j < Atrans.colptr[row+1]; j++) {
			int idx = Atrans.rowval.at(j);
			if (outputelts.at(idx) != 0.) {
				outputelts.at(idx) += elt*Atrans.nzval.at(j);
				//if (outputelts[idx] == 0.) outputelts[idx] = 1e-50;
			} else {
				outputelts.at(idx) = elt*Atrans.nzval.at(j);
				outputnzidx.at(outputnnz++) = idx;
			}
		}
		outputelts.at(row+A.ncol) = -elt;
		outputnzidx.at(outputnnz++) = row+A.ncol;
	}


	auto t2 = chrono::high_resolution_clock::now();
	return chrono::duration_cast<chrono::nanoseconds>(t2-t);

}

struct BenchmarkOperation {
	function<chrono::nanoseconds(InstanceData const&, IterationData const&)> func;
	string name;
};


int main(int argc, char**argv) {

	assert(argc == 2);
	ifstream f(argv[1]);
	
	InstanceData instance = readInstance(f);
	cout << "Problem is " << instance.A.nrow << " by " << instance.A.ncol << " with " << instance.A.nzval.size() << " nonzeros\n";
	
	vector<BenchmarkOperation> benchmarks{ 
		{ doPrice, "Matrix transpose-vector product with non-basic columns" },
		{ doPriceBoundsCheck, "Matrix transpose-vector product with non-basic columns (with bounds checking)" },
		{ doPriceHypersparse, "Hyper-sparse matrix transpose-vector product" },
		{ doPriceHypersparseBoundsCheck, "Hyper-sparse matrix transpose-vector product (with bounds checking)" },
	};
	vector<chrono::nanoseconds> timings(benchmarks.size(), chrono::nanoseconds::zero());

	int nruns = 0;
	while (true) {
		IterationData dat = readIteration(f,instance);
		if (!dat.valid) break;
		for (int i = 0; i < benchmarks.size(); i++) {
			timings[i] += benchmarks[i].func(instance,dat);
		}
		nruns++;
	}

	cout << nruns << " simulated iterations. Total timings:\n";
	for (int i = 0; i < benchmarks.size(); i++) {
		cout << benchmarks[i].name << ": " << timings[i].count()/1000000000. << " sec\n";
	}

	return 0;

}
