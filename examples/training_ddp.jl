using MPI

ENV["PARALLEL"] = "true"
ENV["JULIA_PROJECT"] = dirname(@__FILE__)

@show ENV["JULIA_PROJECT"]
np = 2
#np = 1

run(`$(mpiexec()) -n $np julia training.jl`)

#run(`$(mpiexec()) -n 2 printenv`)

# mirlo
# 198.215347 seconds 1 GPU: loss 0.022
# 207 s 2 GPUs: 0.024
