# import the modules

import CUDA
using BSON
using DataStructures
using Dates
using Flux
using Glob
using JSON3
using NCDatasets
using Printf
using Random
using Statistics
using Test

# name of the dataset (test or dev)
dataset = "test"

# variable name
varname = "CHL"

# training data
fname_train = expanduser("~/Data/NECCTON_Black_Sea/CHL2/cmems_obs-oc_blk_bgc-plankton_my_l3-olci-300m_P1D/patches_64_64_0.8.nc")

# data to be reconstructed
fname_orig = expanduser("~/Data/NECCTON_Black_Sea/CHL2/cmems_obs-oc_blk_bgc-plankton_my_l3-olci-300m_P1D/$(dataset)_log10.nc")

# directory with the experiments
expdir = expanduser("~/Data/NECCTON_Black_Sea/CHL2/cmems_obs-oc_blk_bgc-plankton_my_l3-olci-300m_P1D/")

# timestamp of the used model and epoch
timestamp = "2023-12-06T152517"
epoch = 100

# number of ensemble members to compute and compte the mean and standard
# deviation
Nsample = 64

# number of ensemble members to keep
Nsample_keep = 64

# resolution of the dataset
# (longitude and latitude are sadly stored in the netcdf file
# as single precision floats with is insufficient for a 300 m resolution
# dataset)
Δlon = 0.0037530265
Δlat = 0.0026990548
Δtime = Day(1)


#---

include("diffusion_model.jl")
CUDA.allowscalar(false)

fname_cv = replace(fname_orig,".nc" => "_add_clouds.nc")

ds = NCDataset(fname_orig)
data_orig = nomissing(ds[varname][:,:,:],NaN)
data_orig = reshape(data_orig,(size(data_orig,1),size(data_orig,2),1,size(data_orig,3)))
close(ds)


ds = NCDataset(fname_cv)
data_cv = nomissing(ds[varname][:,:,:],NaN)
data_cv = reshape(data_cv,(size(data_cv,1),size(data_cv,2),1,size(data_cv,3)))
lonf = repeat(ds["lon"][:],inner=(1,size(data_cv,4)))
latf = repeat(ds["lat"][:],inner=(1,size(data_cv,4)))
time = ds["time"][:];


ds_train = NCDataset(fname_train)
lon_range = extrema(ds_train["lon"][:,:])
lat_range = extrema(ds_train["lat"][:,:])
close(ds_train)


lon = round.(Int, (lonf .- Δlon/2) / Δlon) * Δlon;
lat = round.(Int, (latf .- Δlat/2) / Δlat) * Δlat;

pi = PatchIndex1((lon,lat,time),(Δlon,Δlat,Δtime));

sz = size(data_cv)[1:2]

epoch_str = @sprintf("%05d",epoch)

model_fname = joinpath(expdir,"$timestamp/model-checkpoint-$epoch_str.bson")

fname_cv_out = replace(model_fname,".bson" => "") * "_" * replace(basename(fname_cv),".nc" => "log10_filled.nc")
fname_cv_stat = replace(model_fname,".bson" => "") * "_" * replace(basename(fname_cv),".nc" => "log10_filled-$varname.json")

@show model_fname

BSON.@load model_fname beta train_mean train_std losses
BSON.@load model_fname m


params = JSON3.read(joinpath(dirname(model_fname),"params.json"))
ntime_win = get(params,:ntime_win,1)

auxdata_loader = AuxData(
    (lon,lat,time),(Δlon,Δlat,Δtime),data_cv,
    ntime_win;
    lon_range = lon_range,
    lat_range = lat_range,
    cycle = 365.25)

ds = NCDataset(fname_cv,"r")

fname_cv_out = "TESTTEST.nc"
isfile(fname_cv_out) && rm(fname_cv_out)

dsout = NCDataset(fname_cv_out,"c")

# Dimensions

dsout.dim["lon"] = ds.dim["lon"]
dsout.dim["lat"] = ds.dim["lat"]
dsout.dim["time"] = ds.dim["time"]

# Declare variables

nclon = defVar(dsout,"lon", Float64, ("lon", "time"))

nclat = defVar(dsout,"lat", Float64, ("lat", "time"))

nctime = defVar(dsout,"time", Float64, ("time",), attrib = OrderedDict(
    "units"                     => "days since 1970-01-01",
))

ncdata = defVar(dsout,varname, Float32, ("lon", "lat", "time"), attrib = OrderedDict(
    "_FillValue"                => Float32(-9999.0),
))


if Nsample_keep > 0
    dsout.dim["sample"] = Nsample_keep
    ncdatasample = defVar(dsout,varname * "_sample", Float32, ("lon", "lat", "time", "sample"), attrib = OrderedDict(
        "_FillValue"                => Float32(-9999.0),
    ))
end

ncdataerror = defVar(dsout,varname * "_error", Float32, ("lon", "lat", "time"), attrib = OrderedDict(
    "_FillValue"                => Float32(-9999.0),
))


if ndims(ds["lon"]) == 2
    nclon[:,:] = ds["lon"][:,:]
    nclat[:,:] = ds["lat"][:,:]
else
    nclon[:,:] = repeat(ds["lon"][:],inner=(1,ds.dim["time"]))
    nclat[:,:] = repeat(ds["lat"][:],inner=(1,ds.dim["time"]))
end
nctime[:] = ds["time"][:]

close(ds)

# number of steps
T = length(beta)
device = gpu
model = m |> device

alpha,alpha_bar,sigma = noise_schedule(beta)

training = false
rng = Random.GLOBAL_RNG
dd = Dataset6(data_cv,rng,T,train_mean,train_std,device,alpha_bar,auxdata_loader,training)

ntimes = 1:size(data_cv,4)

# time loop
for n = ntimes
    local x0
    local xc
    local mx
    local stdx

    x0, = getobs_orig(dd,n)
    x0 = x0 |> device

    xc = generate_cond(
        device, beta, model, train_mean, train_std, x0, Nsample;
        x_diff = nothing,
    );


    if any(isnan,xc)
        @warn "NaN in reconstruction at step $n"
        open(fname_cv_stat,"w") do f
            JSON3.pretty(f,OrderedDict(
            "cvrms" => 9999
            ))
        end
        break
    end

    xc = xc[:,:,1:1,:] # first slice is current time
    xc = cpu(xc)
    mx = mean(xc,dims=4)[:,:,1,1]
    stdx = std(xc,dims=4)[:,:,1,1]

    @show n,size(data_cv,4),extrema(mx)

    ncdata[:,:,n] = mx
    ncdataerror[:,:,n] = stdx

    if Nsample_keep > 0
        ncdatasample[:,:,n,:] = xc[:,:,1,1:Nsample_keep]
    end
end

close(dsout)
