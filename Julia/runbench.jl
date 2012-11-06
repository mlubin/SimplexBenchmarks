load("sparse.jl")
load("profile.jl")

typealias VariableState Int
const Basic = 1
const AtLower = 2
const AtUpper = 3

type InstanceData
    A::SparseMatrixCSC{Float64,Int64}
    Atrans::SparseMatrixCSC{Float64,Int64}
end

type IterationData
    variableState::Vector{VariableState}
    priceInput::Vector{Float64}
    reducedCosts::Vector{Float64}
    normalizedTableauRow::Vector{Float64}
    valid::Bool
end

type IndexedVector
    elts::Vector{Float64}
    nzidx::Vector{Int}
    nnz::Int
end

function IndexedVector(densevec::Vector{Float64})
    elts = zeros(length(densevec))
    nzidx = zeros(Int,length(densevec))
    nnz = 0
    for i in 1:length(densevec)
        if abs(densevec[i]) > 1e-50
            nnz += 1
            nzidx[nnz] = i
            elts[i] = densevec[i]
        end
    end
    IndexedVector(elts,nzidx,nnz)
end

function readMat(f)
    s1 = split(readline(f))
    nrow,ncol,nnz = int(s1[1]),int(s1[2]),int(s1[3])
    colptr = convert(Vector{Int64},[int64(s) for s in split(readline(f))])
    rowval = convert(Vector{Int64},[int64(s) for s in split(readline(f))])
    nzval = [float64(s) for s in split(readline(f))]
    @assert length(colptr) == ncol + 1
    @assert colptr[ncol+1] == nnz+1
    @assert length(rowval) == nnz
    @assert length(nzval) == nnz
    
    SparseMatrixCSC(nrow, ncol, colptr, rowval, nzval)
end

function readInstance(f)
    A = readMat(f)
    Atrans = readMat(f)
    InstanceData(A,Atrans)
end

function readIteration(f)
    variableState = convert(Vector{VariableState},[int(s) for s in split(readline(f))])
    priceInput = convert(Vector{Float64},[float64(s) for s in split(readline(f))]) 
    reducedCosts = convert(Vector{Float64},[float64(s) for s in split(readline(f))]) 
    normalizedTableauRow = convert(Vector{Float64},[float64(s) for s in split(readline(f))]) 
    valid = (length(variableState) > 0 && length(priceInput) > 0)
    IterationData(variableState,priceInput,reducedCosts,normalizedTableauRow,valid)
end

# dot product with nonbasic columns
# assumes dense input
#@profile begin
function doPrice(instance::InstanceData,d::IterationData)
    A = instance.A
    nrow,ncol = size(A)
    output = zeros(nrow+ncol)

    # Eventually these references won't be needed, after improvements in the Julia compiler
    # Thanks @vtjnash
    rho = d.priceInput
    Arv = A.rowval
    Anz = A.nzval
    varstate = d.variableState

    t = time()
   
    for i in 1:ncol
        if (varstate[i] == Basic)
            continue
        end
        val = 0.
        for k in A.colptr[i]:(A.colptr[i+1]-1)
            val += rho[Arv[k]]*Anz[k]
        end
        output[i] = val
    end
    for i in 1:nrow
        k = i+ncol
        if (varstate[k] == Basic)
            continue
        end
        output[k] = -rho[i]
    end
    t = time() - t
    # check the answer
    err = min(abs(sum(output-d.normalizedTableauRow)),abs(sum(output+d.normalizedTableauRow)))
    if (err > 1e-5)
        println("Error in price: $err\n")
    end
    return t
end

# linear combination of rows
# costly to check basic/nonbasic status here. 
# instead, ignore and assume will be checked later
function doPriceHypersparse(instance::InstanceData,d::IterationData)
    A = instance.A
    Atrans = instance.Atrans
    nrow,ncol = size(A)
    output = IndexedVector(zeros(nrow+ncol))
    outputelts = output.elts
    outputnzidx = output.nzidx
    outputnnz = output.nnz
    rho = IndexedVector(d.priceInput)
    rhoelts = rho.elts
    rhoidx = rho.nzidx

    Atrv = Atrans.rowval
    Atnz = Atrans.nzval

    t = time()
  
    for k in 1:rho.nnz
        # add elt*(row of A) to output
        row = rhoidx[k]
        elt = rhoelts[row]
        for j in Atrans.colptr[row]:(Atrans.colptr[row+1]-1)
            idx = Atrv[j]
            val = outputelts[idx]
            if (val != 0.)
                val += elt*Atnz[j]
                outputelts[idx] = val
                #if (val == 0.) 
                #    outputelts[idx] = 1e-50
                #end
            else
                outputelts[idx] = elt*Atnz[j]
                outputnnz += 1
                outputnzidx[outputnnz] = idx
            end
        end
        # slack value
        outputelts[row+ncol] = -elt
        outputnnz += 1
        outputnzidx[outputnnz] = row+ncol
    end
    t = time() - t

    # check the answer
    err = min(abs(sum(outputelts-d.normalizedTableauRow)-1.),abs(sum(outputelts+d.normalizedTableauRow)-1.)) # should be off by 1 because we keep basic columns
    if (abs(err) > 1e-5) 
        println("Error in hypersparse price: $err\n")
    end

    return t
end

# Harris stabilizing two-pass ratio test, described in Koberstein thesis
function doTwoPassRatioTest(instance::InstanceData,d::IterationData)

    nrow,ncol = size(instance.A)

    candidates = zeros(Int,ncol) # don't count allocation time, assume reuse
    ncandidates = 0
    thetaMax = 1e25
    pivotTol = 1e-7
    dualTol = 1e-7

    redcost = d.reducedCosts
    varstate = d.variableState
    tabrow = d.normalizedTableauRow

    t = time()

    for i in 1:(ncol+nrow)
        thisState = varstate[i]
        if thisState == Basic # || d.boundClass[i] == Fixed
            continue
        end
        pivotElt = tabrow[i]
        if (thisState == AtLower && pivotElt > pivotTol) || (thisState == AtUpper && pivotElt < -pivotTol) # || (varstate[i] == Free && (alpha2[i] > pivotTol || alpha2[i] < -pivotTol))
            ratio = 0.
            if (pivotElt < 0.)
                ratio = (redcost[i] - dualTol)/pivotElt
            else
                ratio = (redcost[i] + dualTol)/pivotElt
            end
            if (ratio < thetaMax)
                thetaMax = ratio
                candidates[ncandidates += 1] = i
            end
        end
    end

    # pass 2
    enter = -1
    maxAlpha = 0.
    for k in 1:ncandidates
        i = candidates[k]
        ratio = redcost[i]/tabrow[i]
        if (ratio <= thetaMax)
            absalpha = abs(tabrow[i])
            if (absalpha > maxAlpha)
                maxAlpha = absalpha
                enter = i
            end
        end
    end
    # answer in enter. -1 means unbounded

    return time() - t

end

# same as before but now tableau row is a sparse indexed vector
function doTwoPassRatioTestHypersparse(instance::InstanceData,d::IterationData)

    nrow,ncol = size(instance.A)

    candidates = zeros(Int,ncol) 
    ncandidates = 0
    thetaMax = 1e25
    pivotTol = 1e-7
    dualTol = 1e-7

    redcost = d.reducedCosts
    varstate = d.variableState
    tabrow = IndexedVector(d.normalizedTableauRow)
    tabrowelts = tabrow.elts
    tabrowidx = tabrow.nzidx

    t = time()

    for k in 1:tabrow.nnz
        i = tabrowidx[k]
        thisState = varstate[i]
        if thisState == Basic # || d.boundClass[i] == Fixed
            continue
        end
        pivotElt = tabrowelts[i]
        if (thisState == AtLower && pivotElt > pivotTol) || (thisState == AtUpper && pivotElt < -pivotTol) # || (varstate[i] == Free && (alpha2[i] > pivotTol || alpha2[i] < -pivotTol))
            ratio = 0.
            if (pivotElt < 0.)
                ratio = (redcost[i] - dualTol)/pivotElt
            else
                ratio = (redcost[i] + dualTol)/pivotElt
            end
            if (ratio < thetaMax)
                thetaMax = ratio
                candidates[ncandidates += 1] = i
            end
        end
    end

    # pass 2
    enter = -1
    maxAlpha = 0.
    for k in 1:ncandidates
        i = candidates[k]
        ratio = redcost[i]/tabrowelts[i]
        if (ratio <= thetaMax)
            absalpha = abs(tabrowelts[i])
            if (absalpha > maxAlpha)
                maxAlpha = absalpha
                enter = i
            end
        end
    end
    # answer in enter. -1 means unbounded

    return time() - t

end

function doBenchmarks(inputname) 

    f = open(inputname,"r")
    instance = readInstance(f)
    println("Problem is $(instance.A.m) by $(instance.A.n) with $(length(instance.A.nzval)) nonzeros")
    benchmarks = [(doPrice,"Matrix-transpose-vector product with non-basic columns"),
        (doPriceHypersparse,"Hyper-sparse matrix-transpose-vector product"),
        (doTwoPassRatioTest,"Two-pass dual ratio test"),
        (doTwoPassRatioTestHypersparse,"Hyper-sparse two-pass dual ratio test")]
    timings = zeros(length(benchmarks))
    nruns = 0
    while true
        dat = readIteration(f)
        if !dat.valid
            break
        end
        for i in 1:length(benchmarks)
            func,name = benchmarks[i]
            timings[i] += func(instance,dat)
        end
        nruns += 1
    end

    println("$nruns simulated iterations")
    for i in 1:length(benchmarks)
        println("$(benchmarks[i][2]): $(timings[i]) sec")
    end

end

doBenchmarks(ARGS[1])
#@profile report
