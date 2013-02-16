require("jlSimplex")

function dumpHead(d,f)
    nrow,ncol = size(d.data.A)
    write(f,"$nrow $ncol $(length(d.data.A.rowval))\n")
    write(f,strcat(join(d.data.A.colptr," "),"\n"))
    write(f,strcat(join(d.data.A.rowval," "),"\n"))
    write(f,strcat(join(d.data.A.nzval," "),"\n"))
    Atrans = d.data.A'
    write(f,"$ncol $nrow $(length(d.data.A.rowval))\n")
    write(f,strcat(join(Atrans.colptr," "),"\n"))
    write(f,strcat(join(Atrans.rowval," "),"\n"))
    write(f,strcat(join(Atrans.nzval," "),"\n"))

end

function doTests(mpsfile,dump)
    global f = open("$(mpsfile).dump","w")
    global dumpEvery = dump
    d = DualSimplexData(LPDataFromMPS(mpsfile));
    dumpHead(d,f)
    @time go(d)
    close(f)

    #println("Now with glpk:")
    #SolveMPSWithGLPK(mpsfile)
end

@assert length(ARGS) == 2
ENV["OMP_NUM_THREADS"] = 1 # multithreaded blas/lapack can slow down execution
doTests(ARGS[1],int(ARGS[2]))
