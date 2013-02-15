function [] = runbench(fileName)

    % Load data
    f = fopen(fileName,'r');
    A = fscanf(f,'%d %d %d',3);
    A_nrow = A(1);
    A_ncol = A(2);
    A_nnz  = A(3);
    A_colptr = fscanf(f,'%d',A_ncol+1);
    A_rowval = fscanf(f,'%d',A_nnz);
    A_nzval  = fscanf(f,'%g',A_nnz);
    disp(['Problem is ',num2str(A_nrow),' by ',num2str(A_ncol),' with ',num2str(A_nnz),' nonzeros'])

    AT = fscanf(f,'%d %d %d',3);
    AT_nrow = AT(1);
    AT_ncol = AT(2);
    AT_nnz  = AT(3);
    AT_colptr = fscanf(f,'%d',AT_ncol+1);
    AT_rowval = fscanf(f,'%d',AT_nnz);
    AT_nzval  = fscanf(f,'%g',AT_nnz);

    % Define some helpful constants
    BASIC = 1;
    ATLOWER = 2;
    ATUPPER = 3;

    % Loop over all iterations
    iterationCount = 0;
    doPriceTime = 0;
    doPriceHypersparseTime = 0;
    doTwoPassRatioTestTime = 0;
    doTwoPassRatioTestHypersparseTime = 0;
    doUpdateDualsTime = 0;
    doUpdateDualsHypersparseTime = 0;
    
    while 1==1
        % Attempt to read an iteration
        variableState = fscanf(f,'%d',A_nrow+A_ncol);
        priceInput = fscanf(f,'%g',A_nrow);
        redCost = fscanf(f,'%g',A_nrow+A_ncol);
        normTableauRow = fscanf(f,'%g',A_nrow+A_ncol);
        
        if numel(variableState) == 0
            % End of file
            break
        end
        
        iterationCount = iterationCount + 1;
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % DoPrice
        tic
        output = zeros(A_ncol+A_nrow,1);
        for i = 1:A_ncol
            if variableState(i) ~= BASIC
                val = 0;
                for k = A_colptr(i):(A_colptr(i+1)-1)
                    val = val + priceInput(A_rowval(k))*A_nzval(k);
                end
                output(i) = val;
            end
        end
        for i = 1:A_nrow
            k = i + A_ncol;
            if variableState(k) ~= BASIC
                output(k) = -priceInput(i);
            end
        end
        t=toc;
        doPriceTime = doPriceTime + t;
        
        err = min(abs(sum(output-normTableauRow)), abs(sum(output+normTableauRow)));
        if err > 1e-5
          disp(['Error in price: ',num2str(err)])
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % doPriceHypersparse
        rho_nnz = 0;
        rho_elts = zeros(A_nrow,1);
        rho_idx  = zeros(A_nrow,1);
        for i = 1:(A_nrow)
            if (abs(priceInput(i)) > 1e-50)
                rho_nnz = rho_nnz + 1;
                rho_idx(rho_nnz) = i;
                rho_elts(rho_nnz) = priceInput(i);
            end
        end
        
        output_nnz = 0;
        output_elts = zeros(A_ncol+A_nrow,1);
        output_idx  = zeros(A_ncol+A_nrow,1);
        
        tic
        for k = 1:rho_nnz
            row = rho_idx(k);
            elt = rho_elts(k);
            for j = AT_colptr(row):(AT_colptr(row+1)-1)
                idx = AT_rowval(j);
                val = output_elts(idx);
                if (val ~= 0)
                    val = val + elt*AT_nzval(j);
                    output_elts(idx) = val;
                else
                    output_elts(idx) = elt*AT_nzval(j);
                    output_nnz = output_nnz + 1;
                    output_idx(output_nnz) = idx;
                end
            end
            % Slack value
            output_elts(row+A_ncol) = -elt;
            output_nnz = output_nnz + 1;
            output_idx(output_nnz) = A_nrow + A_ncol;
        end
        t=toc;
        doPriceHypersparseTime = doPriceHypersparseTime + t;
        
        err = min(abs(sum(output_elts-normTableauRow)-1), abs(sum(output_elts+normTableauRow)-1));
        if err > 1e-5
          disp(['Error in price: ',num2str(err)])
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % doTwoPassRatioTest
        candidates = zeros(A_ncol,1);
        ncandidates = 0;
        thetaMax = 1e25;
        pivotTol = 1e-7;
        dualTol = 1e-7;
        
        tic
        for i = 1:(A_ncol+A_nrow)
           thisState = variableState(i);
           pivotElt = normTableauRow(i);
           if (thisState == ATLOWER && pivotElt > pivotTol) || (thisState == ATUPPER && pivotElt < -pivotTol)
			   ncandidates = ncandidates + 1;
			   candidates(ncandidates) = i;
               ratio = 0;
               if pivotElt < 0
                   ratio = (redCost(i) - dualTol)/pivotElt;
               else
                   ratio = (redCost(i) + dualTol)/pivotElt;
               end
               if ratio < thetaMax
                   thetaMax = ratio;
               end
           end
        end
        
        enter = -1;
        maxAlpha = 0;
        for k = 1:ncandidates
            i = candidates(k);
            ratio = redCost(i)/normTableauRow(i);
            if (ratio <= thetaMax)
                absalpha = abs(normTableauRow(i));
                if (absalpha > maxAlpha)
                    maxAlpha = absalpha;
                    enter = i;
                end
            end
        end
    
        t=toc;
        doTwoPassRatioTestTime = doTwoPassRatioTestTime + t;
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % doTwoPassRatioTestHypersparse
        candidates = zeros(A_ncol,1);
        ncandidates = 0;
        thetaMax = 1e25;
        pivotTol = 1e-7;

        tabrow_nnz = 0;
        tabrow_elts = zeros(A_nrow+A_ncol,1);
        tabrow_idx  = zeros(A_nrow+A_ncol,1);
        for i = 1:(A_nrow+A_ncol)
            if (abs(normTableauRow(i)) > 1e-50)
                tabrow_nnz = tabrow_nnz + 1;
                tabrow_idx(tabrow_nnz) = i;
                tabrow_elts(tabrow_nnz) = normTableauRow(i);
            end
        end
        
        tic;
        for k = 1:tabrow_nnz
            i = tabrow_idx(k);
            thisState = variableState(i);
            pivotElt = tabrow_elts(i);
            if (thisState == ATLOWER && pivotElt > pivotTol) || (thisState == ATUPPER && pivotElt < -pivotTol)
				ncandidates = ncandidates + 1;
				candidates(ncandidates) = i;
                ratio = 0;
                if pivotElt < 0
                    ratio = (redCost(i) - dualTol)/pivotElt;
                else
                    ratio = (redCost(i) + dualTol)/pivotElt;
                end
                if ratio < thetaMax
                    thetaMax = ratio;
                end
            end
        end
              
        enter = -1;
        maxAlpha = 0;
        for k = 1:ncandidates
            i = candidates(k);
            ratio = redCost(i)/tabrow_elts(i);
            if (ratio <= thetaMax)
                absalpha = abs(tabrow_elts(i));
                if (absalpha > maxAlpha)
                    maxAlpha = absalpha;
                    enter = i;
                end
            end
        end
        
        t=toc;
        doTwoPassRatioTestHypersparseTime = doTwoPassRatioTestHypersparseTime + t;


        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % doUpdateDuals
        stepsize = 1.;
        newRedCost = redCost; % make a copy
        tic;
        for i = 1:(A_nrow+A_ncol)
            thisState = variableState(i);
            dnew = newRedCost(i) - stepsize*tabrow_elts(i);

            if (thisState == ATLOWER)
                if (dnew >= dualTol)
                    newRedCost(i) = dnew;
                else
                    newRedCost(i) = -dualTol;
                end
            elseif (thisState == ATUPPER)
                if (dnew <= dualTol)
                    newRedCost(i) = dnew;
                else
                    newRedCost(i) = dualTol;
                end
            end
        end

        t=toc;
        doUpdateDualsTime = doUpdateDualsTime + t;

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % doUpdateDualsHypersparse
        newRedCost = redCost; 
        tic;
        for j = 1:tabrow_nnz
            i = tabrow_idx(j);
            thisState = variableState(i);
            dnew = newRedCost(i) - stepsize*tabrow_elts(i);

            if (thisState == ATLOWER)
                if (dnew >= dualTol)
                    newRedCost(i) = dnew;
                else
                    newRedCost(i) = -dualTol;
                end
            elseif (thisState == ATUPPER)
                if (dnew <= dualTol)
                    newRedCost(i) = dnew;
                else
                    newRedCost(i) = dualTol;
                end
            end
        end

        t=toc;
        doUpdateDualsHypersparseTime = doUpdateDualsHypersparseTime + t;

            
    end
    
    disp([num2str(iterationCount),' simulated iterations'])
    disp(['Matrix-transpose-vector product with non-basic columns: ',num2str(doPriceTime), ' sec'])
    disp(['Hyper-sparse matrix-transpose-vector product: ',num2str(doPriceHypersparseTime),' sec'])
    disp(['Two-pass dual ratio test: ',num2str(doTwoPassRatioTestTime),' sec']);
    disp(['Hyper-sparse two-pass dual ratio test: ',num2str(doTwoPassRatioTestHypersparseTime),' sec']);
    disp(['Update dual iterate with cost shifting: ',num2str(doUpdateDualsTime),' sec']);
    disp(['Hyper-sparse update dual iterate with cost shifting: ',num2str(doUpdateDualsHypersparseTime),' sec']);

end
