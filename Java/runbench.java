import java.io.*;

class SimplexBenchmarks {
    public static void main(String[] args) {
        try {
            FileReader fr = new FileReader(new File(args[0]));
            LineNumberReader lnr = new LineNumberReader(fr);

            InstanceData instance = new InstanceData(lnr);
            
            System.out.println("Problem is " + instance.A.nrow +
                                " by " + instance.A.ncol +
                                " with " + instance.A.nnz + " nonzeros");

            int nruns = 0;
            long total_price_time = 0;
            long total_pricehs_time = 0;
            long total_twopass_time = 0;
            long total_twopasshs_time = 0;
            long total_updual_time = 0;
            long total_updualhs_time = 0;

            while (true) {
                IterationData itdata = new IterationData(lnr);
                if (!itdata.valid) {
                    break;
                }

                nruns += 1;

                long price_time = doPrice(instance, itdata);
                total_price_time += price_time;
                // System.out.println(price_time);
                
                long pricehs_time = doPriceHypersparse(instance, itdata);
                total_pricehs_time += pricehs_time;
                // System.out.println(pricehs_time);
                
                long twopass_time = doTwoPassRatioTest(instance, itdata);
                total_twopass_time += twopass_time;
                // System.out.println(twopass_time);

                long twopasshs_time = doTwoPassRatioTestHypersparse(instance, itdata);
                total_twopasshs_time += twopasshs_time;
                // System.out.println(twopasshs_time);

                long updual_time = doUpdateDuals(instance, itdata);
                total_updual_time += updual_time;
                // System.out.println(updual_time);

                long updualhs_time = doUpdateDualsHypersparse(instance, itdata);
                total_updualhs_time += updualhs_time;
                // System.out.println(updualhs_time);
            }

            System.out.println(nruns + " simulated iterations");
            System.out.println("Matrix-transpose-vector product with non-basic columns: " + (total_price_time*1.0/nruns/1000000) + " sec");
            System.out.println("Hyper-sparse matrix-transpose-vector product:  " + (total_pricehs_time*1.0/nruns/1000000) + " sec");
            System.out.println("Two-pass dual ratio test: " + (total_twopass_time*1.0/nruns/1000000) + " sec");
            System.out.println("Hyper-sparse two-pass dual ratio test: " + (total_twopasshs_time*1.0/nruns/1000000) + " sec");
            System.out.println("Update dual iterate with cost shifting: " + (total_updual_time*1.0/nruns/1000000) + " sec");
            System.out.println("Hyper-sparse update dual iterate with cost shifting: " + (total_updualhs_time*1.0/nruns/1000000) + " sec");

        } catch(Exception e) {
            e.printStackTrace();
        }
    }


    public static long doPrice(InstanceData instance, IterationData d) {
        // Direct translation of the Python code
        SparseMatrixCSC A = instance.A;
        int nrow = A.nrow;
        int ncol = A.ncol;
        double[] output = new double[nrow+ncol];

        double[] rho = d.priceInput;
        int[] Arv = A.rowval;
        double[] Anz = A.nzval;
        int[] varstate = d.variableState;

        final long t = System.nanoTime();

        for (int i = 0; i < ncol; i++) {
            if (varstate[i] == 1) {  // 1 == BASIC
                continue;
            }
            double val = 0.0;
            for (int k = A.colptr[i]; k < A.colptr[i+1]; k++) {
                val += rho[Arv[k]] * Anz[k];
            }
            output[i] = val;
        }

        for (int i = 0; i < nrow; i++) {
            int k = i + ncol;
            if (varstate[i] == 1) {  // 1 == BASIC
                continue;
            }
            output[k] = -rho[i];
        }

        // Error checking
        // double sumout_less_norm = 0.0;
        // double sumout_plus_norm = 0.0;
        // for (int i = 0; i < nrow + ncol; i++) {
        //     sumout_less_norm += output[i] - d.normalizedTableauRow[i];
        //     sumout_plus_norm += output[i] + d.normalizedTableauRow[i];
        // }
        // System.out.println("sum(output - normalizedTableauRow)-1 = " + (sumout_less_norm-1));
        // System.out.println("sum(output + normalizedTableauRow)-1 = " + (sumout_plus_norm-1));

        return System.nanoTime() - t;
    }


    public static long doPriceHypersparse(InstanceData instance, IterationData d) {
        // Direct translation of the Python code
        SparseMatrixCSC A      = instance.A;
        SparseMatrixCSC Atrans = instance.Atrans;
        int nrow = A.nrow;
        int ncol = A.ncol;
        double[] outputelts  = new double[nrow+ncol];
        int[]    outputnzidx = new int[nrow+ncol];
        int      outputnnz   = 0;
        IndexedVector rho = new IndexedVector(d.priceInput);
        double[] rhoelts = rho.elts;
        int[]    rhoidx  = rho.nzidx;

        int[]    Atrv = Atrans.rowval;
        double[] Atnz = Atrans.nzval;

        final long t = System.nanoTime();

        for (int k = 0; k < rho.nnz; k++) {
            int    row = rhoidx[k];
            double elt = rhoelts[row];
            for (int j = Atrans.colptr[row]; j < Atrans.colptr[row+1]; j++) {
                int    idx = Atrv[j];
                double val = outputelts[idx];
                if (val != 0.0) {
                    val += elt * Atnz[j];
                    outputelts[idx] = val;
                } else {
                    outputelts[idx] = elt * Atnz[j];
                    outputnzidx[outputnnz] = idx;
                    outputnnz += 1;
                }
            }
            outputelts[row+ncol] = -elt;
            outputnzidx[outputnnz] = row + ncol;
            outputnnz += 1;
        }

        // Error checking
        // double sumout_less_norm = 0.0;
        // double sumout_plus_norm = 0.0;
        // for (int i = 0; i < nrow + ncol; i++) {
        //     sumout_less_norm += outputelts[i] - d.normalizedTableauRow[i];
        //     sumout_plus_norm += outputelts[i] + d.normalizedTableauRow[i];
        // }
        // System.out.println("sum(output - normalizedTableauRow)-1 = " + (sumout_less_norm-1));
        // System.out.println("sum(output + normalizedTableauRow)-1 = " + (sumout_plus_norm-1));

        return System.nanoTime() - t;
    }


    public static long doTwoPassRatioTest(InstanceData instance, IterationData d) {
        // Direct translation of the Python code
        SparseMatrixCSC A = instance.A;
        int nrow = A.nrow;
        int ncol = A.ncol;
        int[] candidates = new int[ncol];
        int ncandidates = 0;
        double thetaMax = 1e25;
        double pivotTol = 1e-7;
        double dualTol = 1e-7;

        double[]  redcost = d.reducedCosts;
        int[]    varstate = d.variableState;
        double[]   tabrow = d.normalizedTableauRow;

        final long t = System.nanoTime();

        for (int i = 0; i < ncol + nrow; i++) {
            int thisState = varstate[i];
            double pivotElt = tabrow[i];
            // 2 == ATLOWER, 3 == ATUPPER
            if ((thisState == 2 && pivotElt >  pivotTol) || 
                (thisState == 3 && pivotElt < -pivotTol)) {
                candidates[ncandidates] = i;
                ncandidates += 1;
                double ratio = 0.0;
                if (pivotElt < 0.0) {
                    ratio = (redcost[i] - dualTol)/pivotElt;
                } else {
                    ratio = (redcost[i] + dualTol)/pivotTol;
                }
                if (ratio < thetaMax) {
                    thetaMax = ratio;
                }
            }
        }

        int enter = -1;
        double maxAlpha = 0.0;

        for (int k = 0; k < ncandidates; k++) {
            int i = candidates[k];
            double ratio = redcost[i]/tabrow[i];
            if (ratio <= thetaMax) {
                double absalpha = Math.abs(tabrow[i]);
                if (absalpha > maxAlpha) {
                    maxAlpha = absalpha;
                    enter = i;
                }
            }
        }

        return System.nanoTime() - t;
    }


    public static long doTwoPassRatioTestHypersparse(InstanceData instance, IterationData d) {
        // Direct translation of the Python code
        SparseMatrixCSC A = instance.A;
        int nrow = A.nrow;
        int ncol = A.ncol;
        int[] candidates = new int[ncol];
        int ncandidates = 0;
        double thetaMax = 1e25;
        final double pivotTol = 1e-7;
        final double dualTol = 1e-7;

        double[]  redcost = d.reducedCosts;
        int[]    varstate = d.variableState;
        IndexedVector tabrow = new IndexedVector(d.normalizedTableauRow);
        double[]  tabrowelts = tabrow.elts;
        int[]      tabrowidx = tabrow.nzidx;

        long t = System.nanoTime();

        for (int k = 0; k < tabrow.nnz; k++) {
            int i = tabrowidx[k];
            int thisState = varstate[i];
            double pivotElt = tabrowelts[i];
            // 2 == ATLOWER, 3 == ATUPPER
            if ((thisState == 2 && pivotElt >  pivotTol) || 
                (thisState == 3 && pivotElt < -pivotTol)) {
                candidates[ncandidates] = i;
                ncandidates += 1;
                double ratio = 0.0;
                if (pivotElt < 0.0) {
                    ratio = (redcost[i] - dualTol)/pivotElt;
                } else {
                    ratio = (redcost[i] + dualTol)/pivotElt;
                }
                if (ratio < thetaMax) {
                    thetaMax = ratio;
                }
            }
        }

        int enter = -1;
        double maxAlpha = 0.0;

        for (int k = 0; k < ncandidates; k++) {
            int i = candidates[k];
            double ratio = redcost[i]/tabrowelts[i];
            if (ratio <= thetaMax) {
                double absalpha = Math.abs(tabrowelts[i]);
                if (absalpha > maxAlpha) {
                    maxAlpha = absalpha;
                    enter = i;
                }
            }
        }
        
        return System.nanoTime() - t;
    }


    public static long doUpdateDuals(InstanceData instance, IterationData d) {
        // Direct translation of the Python code
        SparseMatrixCSC A = instance.A;
        int nrow = A.nrow;
        int ncol = A.ncol;

        double[] redcost = new double[d.reducedCosts.length];
        for (int i = 0; i < d.reducedCosts.length; i++) {
            redcost[i] = d.reducedCosts[i];
        }
        int[] varstate = d.variableState;
        double[] tabrow = d.normalizedTableauRow;

        final double stepsize = 1.0;
        final double dualTol = 1e-7;

        long t = System.nanoTime();

        for (int i = 0; i < nrow + ncol; i++) {
            double dnew = redcost[i] - stepsize * tabrow[i];

            // AT LOWER
            if (varstate[i] == 2) {
                if (dnew >= dualTol) {
                    redcost[i] = dnew;
                } else {
                    redcost[i] = -dualTol;
                }
            // AT UPPER
            } else if (varstate[i] == 3) {
                if (dnew <= dualTol) {
                    redcost[i] = dnew;
                } else {
                    redcost[i] = dualTol;
                }
            }
        }
        
        return System.nanoTime() - t;
    }


    public static long doUpdateDualsHypersparse(InstanceData instance, IterationData d) {
        // Direct translation of the Python code
        SparseMatrixCSC A = instance.A;
        int nrow = A.nrow;
        int ncol = A.ncol;

        double[] redcost = new double[d.reducedCosts.length];
        for (int i = 0; i < d.reducedCosts.length; i++) {
            redcost[i] = d.reducedCosts[i];
        }
        int[] varstate = d.variableState;
        IndexedVector tabrow = new IndexedVector(d.normalizedTableauRow);
        double[]  tabrowelts = tabrow.elts;
        int[]      tabrowidx = tabrow.nzidx;

        final double stepsize = 1.0;
        final double dualTol = 1e-7;

        long t = System.nanoTime();

        for (int j = 0; j < tabrow.nnz; j++) {
            int i = tabrowidx[j];
            double dnew = redcost[i] - stepsize * tabrowelts[i];

            // AT LOWER
            if (varstate[i] == 2) {
                if (dnew >= dualTol) {
                    redcost[i] = dnew;
                } else {
                    redcost[i] = -dualTol;
                }
            // AT UPPER
            } else if (varstate[i] == 3) {
                if (dnew <= dualTol) {
                    redcost[i] = dnew;
                } else {
                    redcost[i] = dualTol;
                }
            }
        }

        return System.nanoTime() - t;
    }
}


class IndexedVector {
    int n;
    double[] elts;
    int[] nzidx;
    int nnz;

    public IndexedVector(double[] densevec) {
        n = densevec.length;
        elts  = new double[n];
        nzidx = new int[n];
        nnz = 0;
        for (int i = 0; i < n; i++) {
            elts[i]  = 0.0;
            nzidx[i] = 0;
        }
        for (int i = 0; i < n; i++) {
            if (Math.abs(densevec[i]) > 1e-50) {
                elts[i] = densevec[i];
                nzidx[nnz] = i;
                nnz += 1;
            }
        }
    }
}


class IterationData {
    public int[] variableState;
    public double[] priceInput;
    public double[] reducedCosts;
    public double[] normalizedTableauRow;
    public boolean valid;

    public IterationData(LineNumberReader lnr) {
        try {
            String[] varstate_strs = lnr.readLine().split(" ");
            variableState = new int[varstate_strs.length];
            for (int i = 0; i < varstate_strs.length; i++) {
                variableState[i] = Integer.parseInt(varstate_strs[i]);
            }

            String[] priceInput_strs = lnr.readLine().split(" ");
            priceInput = new double[priceInput_strs.length];
            for (int i = 0; i < priceInput_strs.length; i++) {
                priceInput[i] = Double.parseDouble(priceInput_strs[i]);
            }

            String[] reducedCosts_strs = lnr.readLine().split(" ");
            reducedCosts = new double[reducedCosts_strs.length];
            for (int i = 0; i < reducedCosts_strs.length; i++) {
                reducedCosts[i] = Double.parseDouble(reducedCosts_strs[i]);
            }

            String[] normalizedTableauRow_strs = lnr.readLine().split(" ");
            normalizedTableauRow = new double[normalizedTableauRow_strs.length];
            for (int i = 0; i < normalizedTableauRow_strs.length; i++) {
                normalizedTableauRow[i] = Double.parseDouble(normalizedTableauRow_strs[i]);
            }

            valid = true;
        } catch(Exception e) {
            valid = false;
        }
    }
}


class InstanceData {
    public SparseMatrixCSC A;
    public SparseMatrixCSC Atrans;

    public InstanceData(LineNumberReader lnr) throws Exception {
        A      = new SparseMatrixCSC(lnr);
        Atrans = new SparseMatrixCSC(lnr);
    }
}


class SparseMatrixCSC {
    public int nrow;
    public int ncol;
    public int nnz;
    public int[] colptr;
    public int[] rowval;
    public double[] nzval;

    public SparseMatrixCSC(LineNumberReader lnr) throws Exception {
        String matrix_sizes_str = lnr.readLine();
        String[] matrix_sizes = matrix_sizes_str.split(" ");
        nrow = Integer.parseInt(matrix_sizes[0]);
        ncol = Integer.parseInt(matrix_sizes[1]);
        nnz  = Integer.parseInt(matrix_sizes[2]);

        String[] colptr_strs = lnr.readLine().split(" ");
        colptr = new int[colptr_strs.length];
        for (int i = 0; i < colptr_strs.length; i++) {
            colptr[i] = Integer.parseInt(colptr_strs[i]) - 1;
        }

        String[] rowval_strs = lnr.readLine().split(" ");
        rowval = new int[rowval_strs.length];
        for (int i = 0; i < rowval_strs.length; i++) {
            rowval[i] = Integer.parseInt(rowval_strs[i]) - 1;
        }

        String[] nzval_strs = lnr.readLine().split(" ");
        nzval = new double[nzval_strs.length];
        for (int i = 0; i < nzval_strs.length; i++) {
            nzval[i] = Double.parseDouble(nzval_strs[i]);
        }
    }
}