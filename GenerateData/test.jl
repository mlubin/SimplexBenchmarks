load("jlSimplex.jl")

function dumpHead(d,f)
    nrow,ncol = size(d.data.A)
    write(f,"$nrow $ncol $(length(d.data.A.rowval))\n")
    write(f,strcat(join(d.data.A.colptr," "),"\n"))
    write(f,strcat(join(d.data.A.rowval," "),"\n"))
    write(f,strcat(join(d.data.A.nzval," "),"\n"))


end

function doTests()

    mpsfile = "GREENBEA.SIF"
    global f = open("$(mpsfile).dump","w")
    global dumpEvery = 10
    d = DualSimplexData(LPDataFromMPS(mpsfile));
    dumpHead(d,f)
    @time go(d)
    close(f)

    println("Now with glpk:")
    SolveMPSWithGLPK(mpsfile)
end

doTests()
