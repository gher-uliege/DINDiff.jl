module DINDiff

using DataStructures
using Dates
using Lux
using JSON3
using NCDatasets
using Printf
using Random
using Statistics
using JLD2
using Optimisers
using Zygote
using KernelAbstractions

include("diffusion_model.jl")
include("data.jl")
include("my_unet.jl")

end
