import java.io.*;

class SimplexBenchmarks {
    public static void main(String[] args) {
        System.out.println("Hello World!"); // Display the string.

        try {
            FileReader fr = new FileReader(new File("../GenerateData/greenbea.dump"));
            LineNumberReader lnr = new LineNumberReader(fr);

            InstanceData instance = new InstanceData(lnr);
            
            System.out.println("Problem is " + instance.A.nrow +
                                " by " + instance.A.ncol +
                                " with " + instance.A.nnz + " nonzeros");

            while (true) {
                IterationData itdata = new IterationData(lnr);
                long price_time = doPrice(instance, itdata);
                System.out.println(price_time);
                long pricehs_time = doPriceHypersparse(instance, itdata);
                System.out.println(pricehs_time);
                break;
            }

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

        final long t = System.currentTimeMillis();

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
        double sumout_less_norm = 0.0;
        double sumout_plus_norm = 0.0;
        for (int i = 0; i < nrow + ncol; i++) {
            sumout_less_norm += output[i] - d.normalizedTableauRow[i];
            sumout_plus_norm += output[i] + d.normalizedTableauRow[i];
        }
        System.out.println("sum(output - normalizedTableauRow)-1 = " + (sumout_less_norm-1));
        System.out.println("sum(output + normalizedTableauRow)-1 = " + (sumout_plus_norm-1));

        return System.currentTimeMillis() - t;
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

        final long t = System.currentTimeMillis();

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
        double sumout_less_norm = 0.0;
        double sumout_plus_norm = 0.0;
        for (int i = 0; i < nrow + ncol; i++) {
            sumout_less_norm += outputelts[i] - d.normalizedTableauRow[i];
            sumout_plus_norm += outputelts[i] + d.normalizedTableauRow[i];
        }
        System.out.println("sum(output - normalizedTableauRow)-1 = " + (sumout_less_norm-1));
        System.out.println("sum(output + normalizedTableauRow)-1 = " + (sumout_plus_norm-1));

        return System.currentTimeMillis() - t;
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

    public IterationData(LineNumberReader lnr) throws Exception {
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