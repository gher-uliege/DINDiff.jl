
datadir = expanduser("~/Data/Global/MODIS")
fname_cv = expanduser("~/Data/Global/MODIS/patches_sst_0.25_dev.nc")
mask_fname = joinpath(datadir,"mask.nc")
varname = "sst"
expdir = expanduser("~/tmp/SST-diffusion-model")
datatrans = identity
isvalid = nothing

max_missing_fraction = 0.25
split_name = ["train","dev","test"]
ii = 2
patchfile_mask = joinpath(datadir,"patches_$(varname)_$(max_missing_fraction)_$(split_name[ii])_mask.nc")



# set plon, plat, cl, landmask
function hm!(ax,x; kwargs...)
    heatmap!(ax,plon,plat,x; colorrange = cl, kwargs...)
    heatmap!(ax,plon,plat,landmask,colormap = :grays)
end
