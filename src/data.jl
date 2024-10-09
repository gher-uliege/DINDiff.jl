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

    return (ncdata,ncdatasample,ncdataerror)
end
