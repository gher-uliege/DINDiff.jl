module DINDiff

import MLUtils: numobs, getobs, getobs!
using BSON
using CUDA
using Dates
using Flux
using NCDatasets
using Printf
using Random
using Statistics

include("diffusion_model.jl")

end
