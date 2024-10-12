# import the modules

using Pkg
Pkg.activate("/home/abarth/.julia/dev/DINDiff")

import CUDA
using JLD2
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
using DINDiff
using DINDiff: genmodel, generate_cond, getobs_orig, AuxData, loadmodel, noise_schedule, DatasetLoader

# name of the dataset (test or dev)
dataset = "test"

# timestamp of the used model and epoch
#timestamp = "2023-12-06T152517"


# SST
fname = expanduser("~/Data/Global/MODIS/patches_sst_0.25_dev.nc")
varname = "sst"
expdir = expanduser("~/tmp/SST-diffusion-model")
fname_train = fname
fname_orig = fname_train
#expdir = dirname(fname_train)
datatrans = identity
isvalid = nothing
#timestamp = "2024-10-08T223706"
#timestamp = "2024-10-08T212832"
#timestamp = "2024-10-10T132004"

epoch = 100
epoch = 140

# quick test
#timestamp = "2024-10-02T174808"
#epoch = 20
# end quick test

# number of ensemble members to compute and compte the mean and standard
# deviation
Nsample = 64

# number of ensemble members to keep
Nsample_keep = 64



#---

CUDA.allowscalar(false)

fname_cv = replace(fname_orig,".nc" => "_add_clouds.nc")
fname_cv = fname_train



# ds = NCDataset(fname_orig)
# data_orig = nomissing(ds[varname][:,:,:],NaN)
# data_orig = reshape(data_orig,(size(data_orig,1),size(data_orig,2),1,size(data_orig,3)))
# close(ds)


tindex = 3:3


# ds_train = NCDataset(fname_train)
# lon_range = extrema(ds_train["lon"][:,:])
# lat_range = extrema(ds_train["lat"][:,:])
# close(ds_train)

lon_range = (-179.9499969482422, 178.35000610351562)
lat_range = (-79.94999694824219, 73.55000305175781)

if occursin("cmems_obs-oc_blk_bgc-plankton_my_l3-olci-300m_P1D",fname_train)
    # single precision
    lonf = lon
    latf = lat

    # resolution of the dataset
    # (longitude and latitude are sadly stored in the netcdf file
    # as single precision floats with is insufficient for a 300 m resolution
    # dataset)
    Δlon = 0.0037530265
    Δlat = 0.0026990548
    Δtime = Day(1)

    lon = round.(Int, (lonf .- Δlon/2) / Δlon) * Δlon;
    lat = round.(Int, (latf .- Δlat/2) / Δlat) * Δlat;
end


epoch_str = @sprintf("%05d",epoch)

model_fname = joinpath(expdir,timestamp,"model-checkpoint-$epoch_str.jld2")

fname_cv_out = replace(model_fname,".jld2" => "") * "_" * replace(basename(fname_cv),".nc" => "_filled.nc")
fname_cv_stat = replace(model_fname,".jld2" => "") * "_" * replace(basename(fname_cv),".nc" => "_filled-$varname.json")

@show model_fname

model,params = loadmodel(model_fname);

ntime_win = params.ntime_win
beta = params.beta
train_mean = params.train_mean
train_std = params.train_std


# #auxdata_loader = nothing
# auxdata_loader = AuxData(
#     (lon,lat,time),(Δlon,Δlat,Δtime),data_cv,
#     ntime_win;
#     lon_range = lon_range,
#     lat_range = lat_range,
#     cycle = 365.25)



ds_all = NCDataset(fname_cv,"r")
ds = view(ds_all,time = tindex)
(dsout,ncdata,ncdatasample,ncdataerror) = DINDiff.ncoutput(
    (ds["lon"],ds["lat"],ds["time"]),fname_cv_out, varname; Nsample_keep)

close(ds_all)

device = gpu
model = model |> device

training = false

dd = DatasetLoader(fname_cv, varname, beta;
                   tindex,
                   train_mean,
                   train_std,
                   ntime_win,
                   training,
                   device,
                   lon_range,
                   lat_range)

ntimes = 1:size(ncdata,3)



x_diff = nothing

# time loop
for n = ntimes
    local x0
    local xc
    local mx
    local stdx

    x0,x_mask,aux_data = device.(getobs_orig(dd,n))
    x_diff = zeros(size(x0)[1:3]...,Nsample,length(beta));

    #x0 = cat(x0,aux_data,dims=3)
    #x0 .= x0 .- mean(filter(isfinite,x0))

    #x_diff = nothing

    xc = generate_cond(
        device, beta, model, train_mean, train_std, x0, Nsample;
        auxdata = aux_data,
        x_diff = x_diff,
    );


    if !isnothing(x_diff)
        fname_out = joinpath(dirname(model_fname),"$(dataset)_diff_$(n).nc")
        if isfile(fname_out)
            rm(fname_out)
        end
        ds = NCDataset(fname_out,"c")
        defVar(ds,varname * "_diffusion",x_diff[:,:,1,:,:,],("lon","lat","sample","diffusion_step"))
        close(ds)
    end

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

    @show n,length(ntimes),extrema(mx)

    ncdata[:,:,n] = mx
    ncdataerror[:,:,n] = stdx

    if Nsample_keep > 0
        ncdatasample[:,:,n,:] = xc[:,:,1,1:Nsample_keep]
    end
end

close(dsout)




varname = "sst"
ds = NCDataset(fname_cv; maskingvalue = NaN)
ds_rec = NCDataset(fname_cv_out; maskingvalue = NaN)
n = 1
figure();
subplot(2,1,1); pcolormesh(ds[varname][:,:,3]')
subplot(2,1,2); pcolormesh(ds_rec[varname * "_sample"][:,:,n,1]')
