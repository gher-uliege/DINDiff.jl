using PythonPlot
using NCDatasets


model_fname = "/home/abarth/tmp/SST-diffusion-model/2024-10-08T212832/model-checkpoint-00140.jld2"

model_fname = "/home/abarth/tmp/SST-diffusion-model/2024-10-10T132004/model-checkpoint-00140.jld2"

fname_cv = expanduser("~/Data/Global/MODIS/patches_sst_0.25_dev.nc")


#fname_rec = expanduser("~/tmp/SST-diffusion-model/2024-10-10T132004/model-checkpoint-00140_patches_sst_0.25_dev_filled.nc")

fname_cv_out = replace(model_fname,".jld2" => "") * "_" * replace(basename(fname_cv),".nc" => "_filled.nc")

isfile(fname)


varname = "sst"
ds = NCDataset(fname_cv; maskingvalue = NaN)
ds_rec = NCDataset(fname_cv_out; maskingvalue = NaN)

n = 1

figure(); pcolormesh(ds[varname][:,:,3]')

pcolormesh(ds_rec[varname * "_sample"][:,:,n,1]')

pcolormesh(ds_rec[varname][:,:,n]')

pcolormesh(ds_rec[varname * "_error"][:,:,n]')
