using NCDatasets
using CairoMakie
using Dates
using Printf

include("diffusion_sst_common.jl")


dsm = NCDataset(mask_fname)
lon = dsm["lon"][:];
lat = dsm["lat"][:];
mask = dsm["mask"][:,:];


nglobal = 400
tindex = [nglobal]

dataset = "dev"

timestamp = sort(readdir(expdir))[end]
model_fname = joinpath(expdir,timestamp,"model_diffusion.jld2")

fname_out = joinpath(dirname(model_fname),"$(dataset)_diff_$(nglobal).nc")

dirn = joinpath(dirname(model_fname),"$(dataset)_diff_$(nglobal)_plot")
mkpath(dirn)

isfile(fname_out)
ds_diff = NCDataset(fname_out)

x_diff = ds_diff[varname * "_diffusion"][:,:,1,:];

ds_mask_all = NCDataset(patchfile_mask,"r");
ds_mask = view(ds_mask_all,time = tindex)
ncmask_cv = ds_mask["mask_cv"]

ds = NCDataset(fname_cv; maskingvalue = NaN)

data_orig = ds[varname][:,:,nglobal]
plon  = ds["lon"][:,nglobal]
plat  = ds["lat"][:,nglobal]
ptime  = ds["time"][nglobal]

i = findfirst(==(plon[1]),lon) .+ (0:(length(plon)-1))
j = findfirst(==(plat[1]),lat) .+ (0:(length(plat)-1))
@assert lon[i] == plon
@assert lat[j] == plat

cl = extrema(filter(isfinite,data_orig))

landmask = fill(NaN,size(mask[i,j]))
landmask[mask[i,j] .== 0] .= 1

data = copy(data_orig)
mask_cv = Bool.(ncmask_cv[:,:,1])
data[mask_cv .== 0] .= NaN

k = 2
for k = 1:size(x_diff,3)
    @show k
    fig = Figure(size=(720,300));
    ax1 = Axis(fig[1,1]; title = "original data", aspect=1,
               xlabel = "longitude", ylabel = "latitude");
    hm!(ax1,data_orig);

    ax2 = Axis(fig[1,2]; title = "with added clouds", aspect=1);
    hm!(ax2,data);
    ax2 = Axis(fig[1,3]; title = "diffusion step $k", aspect=1);
    hm!(ax2,x_diff[:,:,k]);
    cb = Colorbar(fig[:,4]; limits=cl);

    fig[0, :] = Label(fig, string("sea surface temperature ",Dates.format(ptime,"yyyy-mm-dd")))


    figname = joinpath(dirn,@sprintf("diff_%05d.png",k))
    @show figname
    save(figname,fig)
end
