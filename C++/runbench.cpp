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
	vector<double> priceInput, reducedCosts, normalizedTableauRow;
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


chrono::nanoseconds doTwoPassRatioTest(InstanceData const& instance, IterationData const& d) {

	SparseMatrixCSC const &A = instance.A;
	int nrow = A.nrow, ncol = A.ncol;

	vector<int> candidates(ncol);
	int ncandidates = 0;
	double thetaMax = 1e25;
	const double pivotTol = 1e-7, dualTol = 1e-7;

	auto t = chrono::high_resolution_clock::now();

	for (int i = 0; i < nrow+ncol; i++) {
		VariableState thisState = d.variableState[i];
		if (thisState == Basic) continue;
		double pivotElt = d.normalizedTableauRow[i];
		if ( (thisState == AtLower && pivotElt > pivotTol) ||
		     (thisState == AtUpper && pivotElt < -pivotTol)) {
			double ratio;
			if (pivotElt < 0.) {
				ratio = (d.reducedCosts[i] - dualTol)/pivotElt;
			} else {
				ratio = (d.reducedCosts[i] + dualTol)/pivotElt;
			}
			if (ratio < thetaMax) {
				thetaMax = ratio;
				candidates[ncandidates++] = i;
			}
		}
	}

	int enter = -1;
	double maxAlpha = 0.;
	for (int k = 0; k < ncandidates; k++) {
		int i = candidates[k];
		double ratio = d.reducedCosts[i]/d.normalizedTableauRow[i];
		if (ratio <= thetaMax) {
			double absalpha = abs(d.normalizedTableauRow[i]);
			if (absalpha > maxAlpha) {
				maxAlpha = absalpha;
				enter = i;
			}
		}
	}

	auto t2 = chrono::high_resolution_clock::now();
	return chrono::duration_cast<chrono::nanoseconds>(t2-t);
}

chrono::nanoseconds doTwoPassRatioTestBoundsCheck(InstanceData const& instance, IterationData const& d) {

	SparseMatrixCSC const &A = instance.A;
	int nrow = A.nrow, ncol = A.ncol;

	vector<int> candidates(ncol);
	int ncandidates = 0;
	double thetaMax = 1e25;
	const double pivotTol = 1e-7, dualTol = 1e-7;

	auto t = chrono::high_resolution_clock::now();

	for (int i = 0; i < nrow+ncol; i++) {
		VariableState thisState = d.variableState.at(i);
		if (thisState == Basic) continue;
		double pivotElt = d.normalizedTableauRow.at(i);
		if ( (thisState == AtLower && pivotElt > pivotTol) ||
		     (thisState == AtUpper && pivotElt < -pivotTol)) {
			double ratio;
			if (pivotElt < 0.) {
				ratio = (d.reducedCosts.at(i) - dualTol)/pivotElt;
			} else {
				ratio = (d.reducedCosts.at(i) + dualTol)/pivotElt;
			}
			if (ratio < thetaMax) {
				thetaMax = ratio;
				candidates.at(ncandidates++) = i;
			}
		}
	}

	int enter = -1;
	double maxAlpha = 0.;
	for (int k = 0; k < ncandidates; k++) {
		int i = candidates.at(k);
		double ratio = d.reducedCosts.at(i)/d.normalizedTableauRow.at(i);
		if (ratio <= thetaMax) {
			double absalpha = abs(d.normalizedTableauRow.at(i));
			if (absalpha > maxAlpha) {
				maxAlpha = absalpha;
				enter = i;
			}
		}
	}

	auto t2 = chrono::high_resolution_clock::now();
	return chrono::duration_cast<chrono::nanoseconds>(t2-t);
}

chrono::nanoseconds doTwoPassRatioTestHypersparse(InstanceData const& instance, IterationData const& d) {

	SparseMatrixCSC const &A = instance.A;
	int nrow = A.nrow, ncol = A.ncol;

	vector<int> candidates(ncol);
	int ncandidates = 0;
	double thetaMax = 1e25;
	const double pivotTol = 1e-7, dualTol = 1e-7;

	IndexedVector tabrow(d.normalizedTableauRow);
	

	auto t = chrono::high_resolution_clock::now();

	for (int k = 0; k < tabrow.nnz; k++) {
		int i = tabrow.nzidx[k];
		VariableState thisState = d.variableState[i];
		if (thisState == Basic) continue;
		double pivotElt = tabrow.elts[i];
		if ( (thisState == AtLower && pivotElt > pivotTol) ||
		     (thisState == AtUpper && pivotElt < -pivotTol)) {
			double ratio;
			if (pivotElt < 0.) {
				ratio = (d.reducedCosts[i] - dualTol)/pivotElt;
			} else {
				ratio = (d.reducedCosts[i] + dualTol)/pivotElt;
			}
			if (ratio < thetaMax) {
				thetaMax = ratio;
				candidates[ncandidates++] = i;
			}
		}
	}

	int enter = -1;
	double maxAlpha = 0.;
	for (int k = 0; k < ncandidates; k++) {
		int i = candidates[k];
		double ratio = d.reducedCosts[i]/tabrow.elts[i];
		if (ratio <= thetaMax) {
			double absalpha = abs(tabrow.elts[i]);
			if (absalpha > maxAlpha) {
				maxAlpha = absalpha;
				enter = i;
			}
		}
	}

	auto t2 = chrono::high_resolution_clock::now();
	return chrono::duration_cast<chrono::nanoseconds>(t2-t);
}

chrono::nanoseconds doTwoPassRatioTestHypersparseBoundsCheck(InstanceData const& instance, IterationData const& d) {

	SparseMatrixCSC const &A = instance.A;
	int nrow = A.nrow, ncol = A.ncol;

	vector<int> candidates(ncol);
	int ncandidates = 0;
	double thetaMax = 1e25;
	const double pivotTol = 1e-7, dualTol = 1e-7;

	IndexedVector tabrow(d.normalizedTableauRow);
	

	auto t = chrono::high_resolution_clock::now();

	for (int k = 0; k < tabrow.nnz; k++) {
		int i = tabrow.nzidx.at(k);
		VariableState thisState = d.variableState.at(i);
		if (thisState == Basic) continue;
		double pivotElt = tabrow.elts.at(i);
		if ( (thisState == AtLower && pivotElt > pivotTol) ||
		     (thisState == AtUpper && pivotElt < -pivotTol)) {
			double ratio;
			if (pivotElt < 0.) {
				ratio = (d.reducedCosts.at(i) - dualTol)/pivotElt;
			} else {
				ratio = (d.reducedCosts.at(i) + dualTol)/pivotElt;
			}
			if (ratio < thetaMax) {
				thetaMax = ratio;
				candidates.at(ncandidates++) = i;
			}
		}
	}

	int enter = -1;
	double maxAlpha = 0.;
	for (int k = 0; k < ncandidates; k++) {
		int i = candidates[k];
		double ratio = d.reducedCosts.at(i)/tabrow.elts.at(i);
		if (ratio <= thetaMax) {
			double absalpha = abs(tabrow.elts.at(i));
			if (absalpha > maxAlpha) {
				maxAlpha = absalpha;
				enter = i;
			}
		}
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
		{ doPriceHypersparse, "Hyper-sparse matrix transpose-vector product" },
		{ doTwoPassRatioTest, "Two-pass dual ratio test" },
		{ doTwoPassRatioTestHypersparse, "Hyper-sparse two-pass dual ratio test" },
		{ doPriceBoundsCheck, "Matrix transpose-vector product with non-basic columns (with bounds checking)" },
		{ doPriceHypersparseBoundsCheck, "Hyper-sparse matrix transpose-vector product (with bounds checking)" },
		{ doTwoPassRatioTestBoundsCheck, "Two-pass dual ratio test (with bounds checking)" },
		{ doTwoPassRatioTestHypersparseBoundsCheck, "Hyper-sparse two-pass dual ratio test (with bounds checking)" },
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
