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

const double dualTol = 1e-7;

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
	vector<double> priceInput, reducedCosts, normalizedTableauRow;
	bool valid;
};

struct IndexedVector {
	vector<double> elts;
	vector<int> nzidx;
	int nnz;
	IndexedVector(vector<double> const& densevec) : elts(densevec.size(),0.), nzidx(densevec.size(),0), nnz(0) {
		for (size_t i = 0; i < densevec.size(); i++) {
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

bool readVector(string const& line, vector<double> &v, int len) {
	istringstream ss(line);
		
	for (int i = 0; i < len; i++) {
		double x;
		ss >> x;
		if (ss.fail()) {
			return false;
		}
		v.push_back(x);
	}

	return true;

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
	d.valid = d.valid && readVector(line, d.priceInput, A.nrow);

	getline(f,line);
	d.valid = d.valid && readVector(line, d.reducedCosts, A.ncol+A.nrow);

	getline(f,line);
	d.valid = d.valid && readVector(line, d.normalizedTableauRow, A.ncol+A.nrow);
	
	return d;

}

struct BenchmarkOperation {
	function<chrono::nanoseconds(InstanceData const&, IterationData const&)> func;
	string name;
};

