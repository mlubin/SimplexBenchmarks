#include "common.cpp"

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
		int i = candidates.at(k);
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

int main(int argc, char**argv) {

	assert(argc == 2);
	ifstream f(argv[1]);
	
	InstanceData instance = readInstance(f);
	cout << "Problem is " << instance.A.nrow << " by " << instance.A.ncol << " with " << instance.A.nzval.size() << " nonzeros\n";
	
	vector<BenchmarkOperation> benchmarks{ 
		{ doPriceBoundsCheck, "Matrix-transpose-vector product with non-basic columns" },
		{ doPriceHypersparseBoundsCheck, "Hyper-sparse matrix-transpose-vector product" },
		{ doTwoPassRatioTestBoundsCheck, "Two-pass dual ratio test" },
		{ doTwoPassRatioTestHypersparseBoundsCheck, "Hyper-sparse two-pass dual ratio test" },
	};
	vector<chrono::nanoseconds> timings(benchmarks.size(), chrono::nanoseconds::zero());

	int nruns = 0;
	while (true) {
		IterationData dat = readIteration(f,instance);
		if (!dat.valid) break;
		for (size_t i = 0; i < benchmarks.size(); i++) {
			timings[i] += benchmarks[i].func(instance,dat);
		}
		nruns++;
	}

	cout << nruns << " simulated iterations\n";
	for (size_t i = 0; i < benchmarks.size(); i++) {
		cout << benchmarks[i].name << ": " << timings[i].count()/1000000000. << " sec\n";
	}

	return 0;

}
