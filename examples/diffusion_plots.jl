using NCDatasets
using CairoMakie
using Dates
using Printf

include("diffusion_sst_common.jl")

timestamp = sort(readdir(expdir))[end]



model_fname = joinpath(expdir,timestamp,"model_diffusion.jld2")



#fname_rec = expanduser("~/tmp/SST-diffusion-model/2024-10-10T132004/model-checkpoint-00140_patches_sst_0.25_dev_filled.nc")

fname_cv_out = replace(model_fname,".jld2" => "") * "_" * replace(basename(fname_cv),".nc" => "_filled.nc")

fname_cv_out = "/home/abarth/tmp/SST-diffusion-model/2024-10-21T231928/model_diffusion_patches_sst_0.25_dev_filled_save.nc"

tindex = 1:1000

ds_mask_all = NCDataset(patchfile_mask,"r");
ds_mask = view(ds_mask_all,time = tindex)
ncmask_cv = ds_mask["mask_cv"]

ds = NCDataset(fname_cv; maskingvalue = NaN)
ds_rec = NCDataset(fname_cv_out; maskingvalue = NaN)
n = 1
k = 1
#n1 = tindex[1]
cl = (-1,1)
plon = 0
plat = 0

dsm = NCDataset(mask_fname)
lon = dsm["lon"][:];
lat = dsm["lat"][:];
mask = dsm["mask"][:,:];



function hm!(ax,x; kwargs...)
    heatmap!(ax,plon,plat,x; colorrange = cl, kwargs...)
    heatmap!(ax,plon,plat,landmask,colormap = :grays)
end

for (n,n1) in enumerate(tindex)
    @show n1
    global cl, lon, lat, landmask, plon, plat

    data_orig = ds[varname][:,:,n1]
    plon  = ds["lon"][:,n1]
    plat  = ds["lat"][:,n1]
    ptime  = ds["time"][n1]

    i = findfirst(==(plon[1]),lon) .+ (0:(length(plon)-1))
    j = findfirst(==(plat[1]),lat) .+ (0:(length(plat)-1))
    @assert lon[i] == plon
    @assert lat[j] == plat

    data_rec = ds_rec[varname * "_sample"][:,:,n,k]
    # mask land
    data_rec[mask[i,j] .== 0] .= NaN

    data = copy(data_orig)
    mask_cv = Bool.(ncmask_cv[:,:,n])
    data[mask_cv .== 0] .= NaN

    cl = extrema(filter(isfinite,data_orig))


    landmask = fill(NaN,size(mask[i,j]))
    landmask[mask[i,j] .== 0] .= 1

    fig = Figure(size=(500,500));
    ax1 = Axis(fig[1,1]; title = "original data", aspect=1,
               xlabel = "longitude", ylabel = "latitude");
    hm!(ax1,data_orig);
    ax2 = Axis(fig[1,2]; title = "with added clouds", aspect=1);
    hm!(ax2,data);
    ax2 = Axis(fig[2,1]; title = "reconstructed data", aspect=1);
    hm!(ax2,data_rec);
    cb = Colorbar(fig[:, 3]; limits=cl);
    fig[0, :] = Label(fig, string("sea surface temperature ",Dates.format(ptime,"yyyy-mm-dd")))
    fig

    figname = joinpath(dirname(model_fname),
                       string("sample_", @sprintf("%05d_%05d",n,k), ".png"))
    save(figname,fig)
end
