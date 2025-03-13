parallel = get(ENV,"PARALLEL","false") == "true"

# load all modules
# all dependencies are assumed to be already installed

using Pkg
Pkg.activate(dirname(@__FILE__))

Pkg.status()

using JLD2
using DataStructures
using Dates
using Lux
using JSON3
using NCDatasets
using Printf
using Random
using Statistics
using Test
using Glob
using MLUtils: DataLoader
using DINDiff
using DINDiff: ncload, extend, train!, DatasetLoader,
    AuxData, naux_data, skipnan, savemodel, noise_schedule, genmodel
if parallel
    import MPI
end

if !isnothing(Sys.which("nvidia-smi"))
    import CUDA, cuDNN
    const GPU=CUDA
else
    import AMDGPU
    const GPU=AMDGPU
end

GPU.allowscalar(false)

function pprintln(backend,args...)
    MPI.Barrier(backend.comm)
    print("rank ",DistributedUtils.local_rank(backend),": ")
    println(args...)
end
pprintln(::Nothing,args...) = println(args...)

function gpusync()
    GPU.synchronize()
end

local_rank = 0
if parallel
    const backend_type = MPIBackend
    DistributedUtils.initialize(backend_type)
    backend = DistributedUtils.get_distributed_backend(backend_type)
    local_rank = DistributedUtils.local_rank(backend)
else
    backend = nothing
end

@show parallel
timestamp = Dates.format(Dates.now(),"yyyy-mm-ddTHHMMSS")

# training on CPU or GPU
#device = cpu_device()
device = gpu_device()

# NetCDF file with the training data
fname = expanduser("~/Data/NECCTON_Black_Sea/CHL2/cmems_obs-oc_blk_bgc-plankton_my_l3-olci-300m_P1D/patches_64_64_0.8.nc")
varname = "CHL"
datadir = dirname(fname)
datatrans = log10
isvalid = >(0)
datadir = dirname(fname)




fname = expanduser("~/Data/Global/MODIS/patches_sst_0.25_train.nc")
fname = expanduser("~/Data/Global/MODIS/patches_sst_0.25_train_512.nc")
varname = "sst"
datadir = expanduser("~/tmp/SST-diffusion-model")
datatrans = identity
isvalid = nothing



batch_size = 64
checkpoint_epoch = 200
nb_epochs = 140
nb_epochs =  20
#nb_epochs =  100
#nb_epochs = 160
learning_rate = 0.00018967415117200598
kernel_size = 3
T = 600
activation = selu
max_beta = 0.02031910864124268;
channels = (16,32,64,128,256,256)
#channels = (16,32,64,128,256)
learning_rate_drop_epoch = 70
learning_rate_factor = 0.8369710273382387
ntime_win = 1

@show batch_size
# quick test
#checkpoint_epoch = 1
#nb_epochs =  2
#T = 2
#channels = (16,32)
# end quick test

@info("$timestamp",batch_size,varname,checkpoint_epoch,nb_epochs,learning_rate,
      kernel_size,T,activation,max_beta,channels,learning_rate_drop_epoch,
      learning_rate_factor,ntime_win)

#datadir = dirname(fname)

resdir = joinpath(datadir,timestamp)
@show resdir

@info "$(Threads.nthreads()) thread(s) available"

@info "loading data: $fname"

train_input = ncload(fname,varname,datatrans; isvalid, backend, nmultiple = batch_size*8);
train_input = extend(train_input);

#@info "remove mean"
#train_input_m = mapslices(s -> mean(skipnan(s)),train_input,dims=(1,2));
#@. train_input = train_input - train_input_m

@info "sample size $(size(train_input))"

ds = NCDataset(fname);
lon = ds["lon"][:,:];
lat = ds["lat"][:,:];
time = ds["time"][:];


if occursin("cmems_obs-oc_blk_bgc-plankton_my_l3-olci-300m_P1D",fname)
    # single precision
    lonf = lon
    latf = lat

    # partical fixes for the used CMEMS data cmems_obs-oc_blk_bgc-plankton_my_l3-olci-300m_P1D
    # yes, the resolution are not round values
    Δlon = 0.0037530265
    Δlat = 0.0026990548
    Δtime = Day(1)

    lon = round.(Int, (lonf .- Δlon/2) / Δlon) * Δlon;
    lat = round.(Int, (latf .- Δlat/2) / Δlat) * Δlat;

    @debug begin
        using Test
        @test Float32.(lon) ≈ Float32.(lonf)
        @test Float32.(lat) ≈ Float32.(latf)
    end
end

sz = size(train_input)[1:2]

#auxdata_loader = nothing
Δlon = lon[2]-lon[1]
Δlat = lat[2]-lat[1]
Δtime = Day(1)

auxdata_loader = AuxData(
    (lon,lat,time),(Δlon,Δlat,Δtime),train_input,
    ntime_win;
    cycle = 365.25)

beta = collect(LinRange(0, max_beta, T))

mkpath(resdir)
model_fname = joinpath(resdir,"model_diffusion.jld2")

if local_rank == 0
    for fn in glob("*.jl",dirname(@__FILE__))
        println("[rank $local_rank]: copying $fn")
        if !isfile(joinpath(resdir,basename(fn)))
            cp(fn,joinpath(resdir,basename(fn)))
        end
    end

    for fn in glob("*.jl",dirname(pathof(DINDiff)))
        println("[rank $local_rank]: copying $fn")
        if !isfile(joinpath(resdir,basename(fn)))
            cp(fn,joinpath(resdir,basename(fn)))
        end
    end
end

@info "generate model"

in_channels = 1
out_channels = 1
if auxdata_loader !== nothing
    in_channels += naux_data(auxdata_loader)
    #    out_channels += naux_data(auxdata_loader)
end

model = genmodel(;
                 kernel_size = kernel_size,
                 activation = activation,
                 in_channels = in_channels+1,
                 out_channels = out_channels,
                 channels = channels
                 )

checkpoint_dirname = resdir


train_mean = Float32(mean(skipnan(Float64,train_input)))
train_std = Float32(std(skipnan(Float64,train_input)))

@info "save hyperparameters"

if local_rank == 0

    paramsname = joinpath(resdir,"params.json")

    open(paramsname,"w") do f
        JSON3.pretty(f,OrderedDict(
            "beta" => beta,
            "nb_epoch" => nb_epochs,
            "activation" => "$activation",
            "batch_size" => batch_size,
            "kernel_size" => kernel_size,
            "learning_rate" => learning_rate,
            "learning_rate_drop_epoch" => learning_rate_drop_epoch,
            "learning_rate_factor" => learning_rate_factor,
            "T" => T,
            "channels" => channels,
            "fname" => fname,
            "ntime_win" => ntime_win,
            "train_mean" => train_mean,
            "train_std" => train_std,
            "in_channels" => in_channels,
            "out_channels" => out_channels,
        ))
    end

end

@info "start training"

training = true

alpha,alpha_bar,sigma = device.(noise_schedule(beta))

rng = Random.GLOBAL_RNG
dd = DatasetLoader(train_input,rng,T,train_mean,train_std,device,alpha_bar,auxdata_loader,training)

@info "Data loader uses $(Threads.nthreads()) thread(s)"
dl = DataLoader(dd; batchsize = batch_size, shuffle=true,
                     parallel = Threads.nthreads() > 1,
                     partial = false);

# test run
(xt,tt,eps,mask) = first(dl);
#ϵ = model((xt, tt));

if size(train_input)[end] == 512
    @info "warm-up"
# warm-up
alpha, alpha_bar, sigma, losses, ps, st = train!(
    model,dl;
    nb_epochs = 2,
    device,
    learning_rate,
    batch_size,
    beta,
    learning_rate_drop_epoch,
    learning_rate_factor,
    checkpoint_dirname,
    checkpoint_epoch,
    auxdata_loader,
    train_mean,
    train_std,
    backend,
    gpusync,
);

end

alpha, alpha_bar, sigma, losses, ps, st = @time train!(
    model,dl;
    nb_epochs,
    device,
    learning_rate,
    batch_size,
    beta,
    learning_rate_drop_epoch,
    learning_rate_factor,
    checkpoint_dirname,
    checkpoint_epoch,
    auxdata_loader,
    train_mean,
    train_std,
    backend,
    gpusync,
);


if local_rank == 0
    savemodel((ps,st),model_fname,train_mean,train_std,beta,losses)
end

