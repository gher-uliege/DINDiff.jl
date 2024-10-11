



function DatasetLoader((lon,lat,time),(Δlon,Δlat,Δtime), data_cv, beta;
                       training = false,
                       ntime_win = 1,
                       device = cpu,
                       train_mean = 0,
                       train_std = 1,
                       rng = Random.GLOBAL_RNG,
                       kwargs...)

    #auxdata_loader = nothing
    auxdata_loader = AuxData(
        (lon,lat,time),(Δlon,Δlat,Δtime),data_cv,
        ntime_win;
        kwargs...)


    # number of steps
    T = length(beta)

    alpha,alpha_bar,sigma = noise_schedule(beta)
    dd = DatasetLoader(data_cv,rng,T,train_mean,train_std,device,alpha_bar,auxdata_loader,training)

    return dd
end

function DatasetLoader(fname_cv::AbstractString, varname, beta;
                       Δtime = Day(1), # FIXME
                       tindex = Colon(), kwargs...)
    ds = NCDataset(fname_cv)
    ds = view(ds,time = tindex)
    data_cv = nomissing(ds[varname][:,:,:],NaN)
    data_cv = reshape(data_cv,(size(data_cv,1),size(data_cv,2),1,size(data_cv,3)))
    #lon = repeat(ds["lon"][:],inner=(1,size(data_cv,4)))
    #lat = repeat(ds["lat"][:],inner=(1,size(data_cv,4)))
    lon = ds["lon"][:,:];
    lat = ds["lat"][:,:];
    time = ds["time"][:];

    Δlon = lon[2]-lon[1]
    Δlat = lat[2]-lat[1]

    dd = DatasetLoader((lon,lat,time),(Δlon,Δlat,Δtime), data_cv, beta; kwargs...)

    return dd
end



function ncoutput((lon,lat,time),fname_cv_out, varname; Nsample_keep = 0)

    isfile(fname_cv_out) && rm(fname_cv_out)

    dsout = NCDataset(fname_cv_out,"c")

    # Dimensions

    dsout.dim["lon"] = size(lon,1)
    dsout.dim["lat"] = size(lat,1)
    dsout.dim["time"] = length(time)

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
    else
        ncdatasample = nothing
    end

    ncdataerror = defVar(dsout,varname * "_error", Float32, ("lon", "lat", "time"), attrib = OrderedDict(
        "_FillValue"                => Float32(-9999.0),
    ))


    if ndims(lon) == 2
        nclon[:,:] = lon[:,:]
        nclat[:,:] = lat[:,:]
    else
        nclon[:,:] = repeat(lon[:],inner=(1,ds.dim["time"]))
        nclat[:,:] = repeat(lat[:],inner=(1,ds.dim["time"]))
    end
    nctime[:] = time[:]

    return (dsout,ncdata,ncdatasample,ncdataerror)
end
