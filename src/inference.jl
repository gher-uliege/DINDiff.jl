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


include("diffusion_model.jl")
CUDA.allowscalar(false)

dataset = "test"
fname_orig = "/home/abarth/Data/NECCTON_Black_Sea/CHL2/cmems_obs-oc_blk_bgc-plankton_my_l3-olci-300m_P1D/$(dataset)_log10.nc"
fname_cv = replace(fname_orig,".nc" => "_add_clouds.nc")
varname = "CHL"


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



fname_train = expanduser("~/Data/NECCTON_Black_Sea/CHL2/cmems_obs-oc_blk_bgc-plankton_my_l3-olci-300m_P1D/patches_64_64_0.8.nc")
ds_train = NCDataset(fname_train)
lon_range = extrema(ds_train["lon"][:,:])
lat_range = extrema(ds_train["lat"][:,:])

#debug
# lonf = ds_train["lon"][:,:]
# latf = ds_train["lat"][:,:]
# time = ds_train["time"][:];
# data_cv = nomissing(ds_train[varname][:,:,:],NaN)
# data_cv = reshape(data_cv,(size(data_cv,1),size(data_cv,2),1,size(data_cv,3)))
close(ds_train)

#lon[2,1]-lon[1,1]
#lat[2,1]-lat[1,1]

# yes, the resolution are not round values
Δlon = 0.0037530265
Δlat = 0.0026990548
Δtime = Day(1)

lon = round.(Int, (lonf .- Δlon/2) / Δlon) * Δlon;
lat = round.(Int, (latf .- Δlat/2) / Δlat) * Δlat;


pi = PatchIndex1((lon,lat,time),(Δlon,Δlat,Δtime));

sz = size(data_cv)[1:2]

epochs = 20:20:100
#epochs = 60:20:100

expdir = "/home/abarth/mnt/milan/Data/NECCTON_Black_Sea/CHL2/cmems_obs-oc_blk_bgc-plankton_my_l3-olci-300m_P1D/"

timestamp = "2023-12-06T152517" # best
epoch = 100

timestamp = "2023-12-04T163318"
epoch = 20


Nsample_keep = 64

epoch_str = @sprintf("%05d",epoch)

model_fname = "/home/abarth/mnt/milan/Data/NECCTON_Black_Sea/CHL2/cmems_obs-oc_blk_bgc-plankton_my_l3-olci-300m_P1D/$timestamp/model-checkpoint-$epoch_str.bson"

fname_cv_out = replace(model_fname,".bson" => "") * "_" * replace(basename(fname_cv),".nc" => "log10_filled.nc")
fname_cv_stat = replace(model_fname,".bson" => "") * "_" * replace(basename(fname_cv),".nc" => "log10_filled-$varname.json")

@show model_fname

BSON.@load model_fname beta train_mean train_std losses
BSON.@load model_fname m


params = JSON3.read(joinpath(dirname(model_fname),"params.json"))
ntime_win = get(params,:ntime_win,1)


#auxdata_loader = nothing
auxdata_loader = AuxData3(
    (lon,lat,time),(Δlon,Δlat,Δtime),data_cv,
    ntime_win;
    lon_range = lon_range,
    lat_range = lat_range,
    cycle = 365.25)

ds = NCDataset(fname_cv,"r")

isfile(fname_cv_out) && rm(fname_cv_out)

#varname2 = varname * "_log"
varname2 = varname

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

ncdata = defVar(dsout,varname2, Float32, ("lon", "lat", "time"), attrib = OrderedDict(
    "_FillValue"                => Float32(-9999.0),
))


if Nsample_keep > 0
    dsout.dim["sample"] = Nsample_keep
    ncdatasample = defVar(dsout,varname2 * "_sample", Float32, ("lon", "lat", "time", "sample"), attrib = OrderedDict(
        "_FillValue"                => Float32(-9999.0),
    ))
end

ncdataerror = defVar(dsout,varname2 * "_error", Float32, ("lon", "lat", "time"), attrib = OrderedDict(
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


#---

T = length(beta)
Nsample = 64
device = gpu
model = m |> device

alpha,alpha_bar,sigma = noise_schedule(beta)

training = false
rng = Random.GLOBAL_RNG
dd = Dataset6(data_cv,rng,T,train_mean,train_std,device,alpha_bar,auxdata_loader,training)

ntimes = 1:size(data_cv,4)

for n = ntimes
    local x0
    local xc
    local mx
    local stdx

    x0, = getobs_orig(dd,n)
    x0 = x0 |> device

    xc = generate_cond(
        device, nothing, beta, model, train_mean, train_std, x0, Nsample;
        x_diff = nothing,
    );


    if any(isnan,xc)
        @warn "NaN is reconstruction at $n"
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


