using MPI

ENV["PARALLEL"] = "true"
ENV["JULIA_PROJECT"] = dirname(@__FILE__)

@show ENV["JULIA_PROJECT"]
np = 2
#np = 1

run(`$(mpiexec()) -n $np julia training.jl`)

