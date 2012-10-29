load("sparse.jl")
load("profile.jl")

typealias VariableState Int
const Basic = 1
const AtLower = 2
const AtUpper = 3

type IterationData
    variableState::Vector{VariableState}
    priceInput::Vector{Float64}
    valid::Bool
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

function readIteration(f)
    variableState = convert(Vector{VariableState},[int(s) for s in split(readline(f))])
    priceInput = convert(Vector{Float64},[float64(s) for s in split(readline(f))]) 
    valid = (length(variableState) > 0 && length(priceInput) > 0)
    IterationData(variableState,priceInput,valid)
end

# dot product with nonbasic columns
# assumes dense input
#@profile begin
function doPrice(A::SparseMatrixCSC,d::IterationData)
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
    
    return time() - t
end
#end

function doBenchmarks(inputname) 

    f = open(inputname,"r")
    A = readMat(f)
    println("Problem is $(A.m) by $(A.n) with $(length(A.nzval)) nonzeros")
    benchmarks = [(doPrice,"Matrix transpose-vector product with non-basic columns")]
    timings = zeros(length(benchmarks))
    nruns = 0
    while true
        dat = readIteration(f)
        if !dat.valid
            break
        end
        for i in 1:length(benchmarks)
            func,name = benchmarks[i]
            timings[i] += func(A,dat)
        end
        nruns += 1
    end

    println("$nruns simulated iterations. Total timings:")
    for i in 1:length(benchmarks)
        println("$(benchmarks[i][2]): $(timings[i]) sec")
    end

end

doBenchmarks(ARGS[1])
#@profile report
