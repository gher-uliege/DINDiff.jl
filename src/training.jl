# load all modules
# all dependencies are assumed to be already installed

import CUDA
using BSON
using DataStructures
using Dates
using Flux
using JSON3
using NCDatasets
using Printf
using Random
using Statistics
using Test

include("diffusion_model.jl")

CUDA.allowscalar(false)

timestamp = Dates.format(Dates.now(),"yyyy-mm-ddTHHMMSS")

# training on CPU or GPU
#device = cpu
device = gpu

# NetCDF file with the training data
fname = expanduser("~/Data/NECCTON_Black_Sea/CHL2/cmems_obs-oc_blk_bgc-plankton_my_l3-olci-300m_P1D/patches_64_64_0.8.nc")
batch_size = 60
batch_size = 2 # test
varname = "CHL"
checkpoint_epoch = 20
nb_epochs = 140
nb_epochs =  20
learning_rate = 0.00018967415117200598
kernel_size = 3
T = 600
activation = relu
max_beta = 0.02031910864124268;
channels = (16,32,64,128,256,256)
learning_rate_drop_epoch = 70
learning_rate_factor = 0.8369710273382387
ntime_win = 1

@info("$timestamp",batch_size,varname,checkpoint_epoch,nb_epochs,learning_rate,
      kernel_size,T,activation,max_beta,channels,learning_rate_drop_epoch,
      learning_rate_factor,ntime_win)

datadir = dirname(fname)

resdir = joinpath(datadir,timestamp)
@show resdir

@info "$(Threads.nthreads()) thread(s) available"

@info "loading data"

train_input = ncload(fname,varname);
train_input = extend(train_input);

@info "sample size $(size(train_input))"

ds = NCDataset(fname);
lonf = ds["lon"][:,:];
latf = ds["lat"][:,:];
time = ds["time"][:];

#lon[2,1]-lon[1,1]
#lat[2,1]-lat[1,1]

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

pi = PatchIndex((lon,lat,time),(Δlon,Δlat,Δtime))

sz = size(train_input)[1:2]

auxdata_loader = nothing

auxdata_loader = AuxData(
     (lon,lat,time),(Δlon,Δlat,Δtime),train_input,
     ntime_win;
     cycle = 365.25)


beta = collect(LinRange(0, max_beta, T))


mkpath(resdir)
model_fname = joinpath(resdir,"model_diffusion.bson")
model_parameters_fname = joinpath(resdir,"model_parameters_diffusion.bson")

cp(@__FILE__,joinpath(resdir,basename(@__FILE__)))

#for fn in ["diffusion_model.jl","test_diffusion_cmp.jl"]
#    cp(joinpath(dirname(@__FILE__),fn),joinpath(resdir,fn))
#end

@info "generate model"

in_channels = 1
out_channels = 1
if auxdata_loader !== nothing
    in_channels += 2*naux_data(auxdata_loader)
    out_channels += naux_data(auxdata_loader)
end

model = genmodel(
    kernel_size,activation;
    in_channels = in_channels,
    out_channels = out_channels,
    channels = channels);

model = model |> device;

checkpoint_dirname = resdir


train_mean = Float32(mean(skipnan(Float64,train_input)))
train_std = Float32(std(skipnan(Float64,train_input)))

@info "save hyperparameters"

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
    ))
end

@info "start training"

training = true

alpha,alpha_bar,sigma = device.(noise_schedule(beta))

rng = Random.GLOBAL_RNG
dd = DatasetLoader(train_input,rng,T,train_mean,train_std,device,alpha_bar,auxdata_loader,training)

@info "Data loader uses $(Threads.nthreads()) thread(s)"
dl = Flux.DataLoader(dd; batchsize = batch_size, shuffle=true,
                     parallel = Threads.nthreads() > 1,
                     partial = false);

# test run
(xt,tt,eps,mask) = first(dl)
ϵ = model((xt, tt))

alpha, alpha_bar, sigma, losses = @time train!(
    model,dl;
    device = gpu,
    nb_epochs = nb_epochs,
    learning_rate = learning_rate,
    batch_size = batch_size,
    beta = beta,
    learning_rate_drop_epoch = learning_rate_drop_epoch,
    learning_rate_factor = learning_rate_factor,
    checkpoint_dirname = checkpoint_dirname,
    checkpoint_epoch = checkpoint_epoch,
    auxdata_loader = auxdata_loader,
    train_mean = train_mean,
    train_std = train_std,
)

m = cpu(model)
BSON.@save model_fname m alpha train_mean train_std beta losses

model_parameters = cpu.(Flux.params(model));

BSON.@save model_parameters_fname model_parameters alpha train_mean train_std beta losses
