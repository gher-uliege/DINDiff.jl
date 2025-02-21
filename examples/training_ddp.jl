using MPI

ENV["PARALLEL"] = "true"

run(`$(mpiexec()) -n 2 julia training.jl`)

#run(`$(mpiexec()) -n 2 printenv`)
