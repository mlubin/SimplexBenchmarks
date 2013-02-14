#include "common.cpp"

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



chrono::nanoseconds doTwoPassRatioTest(InstanceData const& instance, IterationData const& d) {

	SparseMatrixCSC const &A = instance.A;
	int nrow = A.nrow, ncol = A.ncol;

	vector<int> candidates(ncol);
	int ncandidates = 0;
	double thetaMax = 1e25;
	const double pivotTol = 1e-7;

	auto t = chrono::high_resolution_clock::now();

	for (int i = 0; i < nrow+ncol; i++) {
		VariableState thisState = d.variableState[i];
		if (thisState == Basic) continue;
		double pivotElt = d.normalizedTableauRow[i];
		if ( (thisState == AtLower && pivotElt > pivotTol) ||
		     (thisState == AtUpper && pivotElt < -pivotTol)) {
			candidates[ncandidates++] = i;
			double ratio;
			if (pivotElt < 0.) {
				ratio = (d.reducedCosts[i] - dualTol)/pivotElt;
			} else {
				ratio = (d.reducedCosts[i] + dualTol)/pivotElt;
			}
			if (ratio < thetaMax) {
				thetaMax = ratio;
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


chrono::nanoseconds doTwoPassRatioTestHypersparse(InstanceData const& instance, IterationData const& d) {

	SparseMatrixCSC const &A = instance.A;
	int nrow = A.nrow, ncol = A.ncol;

	vector<int> candidates(ncol);
	int ncandidates = 0;
	double thetaMax = 1e25;
	const double pivotTol = 1e-7;

	IndexedVector tabrow(d.normalizedTableauRow);
	

	auto t = chrono::high_resolution_clock::now();

	for (int k = 0; k < tabrow.nnz; k++) {
		int i = tabrow.nzidx[k];
		VariableState thisState = d.variableState[i];
		if (thisState == Basic) continue;
		double pivotElt = tabrow.elts[i];
		if ( (thisState == AtLower && pivotElt > pivotTol) ||
		     (thisState == AtUpper && pivotElt < -pivotTol)) {
			candidates[ncandidates++] = i;
			double ratio;
			if (pivotElt < 0.) {
				ratio = (d.reducedCosts[i] - dualTol)/pivotElt;
			} else {
				ratio = (d.reducedCosts[i] + dualTol)/pivotElt;
			}
			if (ratio < thetaMax) {
				thetaMax = ratio;
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

chrono::nanoseconds doUpdateDuals(InstanceData const& instance, IterationData const& d) {
	
	SparseMatrixCSC const &A = instance.A;
	int nrow = A.nrow, ncol = A.ncol;

	vector<double> redcost = d.reducedCosts;
	double stepsize = 1;

	auto t = chrono::high_resolution_clock::now();

	for (int i = 0; i < nrow+ncol; i++) {
		double dnew = redcost[i] - stepsize*d.normalizedTableauRow[i];

		if (d.variableState[i] == AtLower) {
			if (dnew >= dualTol) {
				redcost[i] = dnew;
			} else {
				redcost[i] = -dualTol;
			}
		} else if (d.variableState[i] == AtUpper) {
			if (dnew <= dualTol) {
				redcost[i] = dnew;
			} else {
				redcost[i] = dualTol;
			}
		}
	}

	auto t2 = chrono::high_resolution_clock::now();
	return chrono::duration_cast<chrono::nanoseconds>(t2-t);
}

chrono::nanoseconds doUpdateDualsHypersparse(InstanceData const& instance, IterationData const& d) {
	
	SparseMatrixCSC const &A = instance.A;
	int nrow = A.nrow, ncol = A.ncol;

	vector<double> redcost = d.reducedCosts;
	IndexedVector tabrow(d.normalizedTableauRow);
	double stepsize = 1;

	auto t = chrono::high_resolution_clock::now();

	for (int j = 0; j < tabrow.nnz; j++) {
		int i = tabrow.nzidx[j];
		double dnew = redcost[i] - stepsize*tabrow.elts[i];

		if (d.variableState[i] == AtLower) {
			if (dnew >= dualTol) {
				redcost[i] = dnew;
			} else {
				redcost[i] = -dualTol;
			}
		} else if (d.variableState[i] == AtUpper) {
			if (dnew <= dualTol) {
				redcost[i] = dnew;
			} else {
				redcost[i] = dualTol;
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
		{ doPrice, "Matrix-transpose-vector product with non-basic columns" },
		{ doPriceHypersparse, "Hyper-sparse matrix-transpose-vector product" },
		{ doTwoPassRatioTest, "Two-pass dual ratio test" },
		{ doTwoPassRatioTestHypersparse, "Hyper-sparse two-pass dual ratio test" },
		{ doUpdateDuals, "Update dual iterate with cost shifting" },
		{ doUpdateDualsHypersparse, "Hyper-sparse update dual iterate with cost shifting" },
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
