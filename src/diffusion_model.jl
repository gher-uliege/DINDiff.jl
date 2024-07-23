
function TimeAppender((x,t))
    # x is W x H x C x N   or   W x C x N
    sz = size(x)
    if ndims(t) == 0
        t = fill(t,(sz[1:end-2]...,1,sz[end]))
    end

    xt = cat(x,t,dims=ndims(x)-1)

    return xt
end


function generate1(device, sz, T, alpha, alpha_bar, sigma, model, train_mean, train_std)
    α = Float32.(cpu(alpha))
    ᾱ = cpu(alpha_bar)
    σ = cpu(sigma)

    x = randn(Float32,sz) |> device

    for t in T:-1:1
        tt = fill(t,ntuple(i -> (i == ndims(x) ? size(x,i) : 1),ndims(x)))
        tt = Float32.(tt ./ T .- 0.5)
        tt = repeat(tt,inner=ntuple(i -> (i > ndims(x)-2 ? 1 : size(x,i)),ndims(x))) |> device

        ϵ = model((x,tt))
        z =
            if t == 1
                zeros(Float32,size(x)) |> device
            else
                randn(Float32,size(x)) |> device
            end

        if α[t] == 1
            ratio = 0   # 0/√0
        else
            ratio = (1 - α[t]) / √(1 - ᾱ[t])
        end

        μ = (1 / √(α[t]) * (x - ratio * ϵ))

        x .= μ + σ[t] * z

        begin
            println("stat of x at step ", t, "NaN ",count(isnan.(x)), "range", extrema(x))
        end

    end

    return x * train_std .+ train_mean
end


"""

x0: width x height x channel
"""
function generate_cond(device, sz, beta, model, train_mean, train_std, x0, Nsample; x_diff = nothing, auxdata = nothing, noise = nothing)
    T = length(beta)

    α,ᾱ,σ = noise_schedule(device(beta))

    batchsize = size(x0,4)
    #x0 = (x0 .- train_mean) ./ train_std |> device;

    # flatten Nsample x batchsize
    x0 = repeat(x0,inner=(1,1,1,Nsample));
    sz = size(x0)

    x =
        if isnothing(noise)
            randn(Float32,sz)
        else
            # flatten Nsample and batchsize
            reshape(noise[:,:,:,:,1],sz);
        end

    x = x |> device;

    mask = isnan.(x0) |> device;
    x[.!mask] = x0[.!mask];

    if auxdata !== nothing
        auxdata = repeat(auxdata,inner=(1,1,1,Nsample)) |> device
    end

    for t in T:-1:1
        tt_index = fill(t,sz);
        tt_index[.!mask] .= 1;
        #@show size(x),size(tt)
        tt = Float32.((tt_index .- 1) ./ (T .- 1) .- 0.5) |> device;
        #@show size(x),size(tt),typeof(x),typeof(tt)
        #@show "call model"

        if auxdata !== nothing
            xin = cat(x,auxdata,dims=3)
        else
            xin = x
        end
        ϵ = model((xin,tt));
        islast = tt_index .== 1;

        #zt = randn(Float32,size(x));
        zt =
            if isnothing(noise)
                randn(Float32,sz)
            else
                reshape(noise[:,:,:,:,T-t+2],sz)
            end

        z = islast .* zeros(Float32,size(x)) + (1 .- islast) .* zt;
        z = z |> device;
        ratio = (1 .- α[tt_index]) ./ sqrt.(1 .- ᾱ[tt_index])
        ratio[isnan.(ratio)] .= 0
        μ = 1 ./ sqrt.(α[tt_index]) .* (x - ratio .* ϵ);
        x .= μ + σ[tt_index] .* z;
        #        x[.!mask] = x0[.!mask]

        @debug begin
        println("stat of x at step ", t,
                " count NaN ",count(isnan.(x)),
                " mean: ", mean(x),
                " std: ", std(x),
                " range: ", extrema(x))
        end

        if !isnothing(x_diff)
            x_diff[:,:,:,:,t] = cpu(train_std * x .+ train_mean)
            #@show "here", extrema(x),extrema(x_diff[:,:,:,:,t])
        end

        if any(isnan,x)
            @warn "NaN in x at step $t"
            break
        end
    end

    # unflatten
    x = reshape(x,(sz[1:3]...,Nsample,batchsize))
    return train_std * x .+ train_mean;
end


function showsize(m)
    return x -> begin
        @show m,size(x)
        return x
    end
end

function block(ks,activation,channels,level)
    if level == length(channels)
        return identity
        #return showsize("inner")
    else
        in = channels[level]
        out = channels[level+1]
        return SkipConnection(
            Chain(
                Conv((ks,ks),in=>out,activation,pad = SamePad()),
                Conv((ks,ks),out=>out,activation,pad = SamePad()),
                Conv((ks,ks),out=>out,activation,pad = SamePad()),
                MaxPool((2,2)),
                block(ks,activation,channels,level+1),
                ConvTranspose((ks,ks),out=>in,activation,pad=SamePad(),stride=2)),
            +)
    end
end

function genmodel2(kernel_size,activation;
                  channels = (16,32,64,128),
                  in_channels = 1,
                  out_channels = 1,
                  )

    ks = kernel_size
    # add time to every down/up sampling block?

    model = Chain(
        TimeAppender,
        Conv((ks,ks),(in_channels+1)=>16,activation,pad = SamePad()),
        Conv((ks,ks),16=>16,activation,pad = SamePad()),
        Conv((ks,ks),16=>16,activation,pad = SamePad()),

        block(ks,activation,channels,1),

        Conv((ks,ks),16=>16,activation,pad = SamePad()),
        Conv((ks,ks),16=>out_channels,pad = SamePad()),
    )
end



#skipnan(x) = filter(isfinite,x)
skipnan(x) = Iterators.filter(!isnan, x)

skipnan(T,x) = Iterators.map(T,Iterators.filter(!isnan, x))

function savemodel(model,dirn,eopch::Integer,train_mean,train_std,beta,losses=[])
    model_fname = joinpath(dirn,"model-checkpoint-" * @sprintf("%05d",eopch) * ".bson")
    savemodel(model,model_fname,train_mean,train_std,beta,losses)
end

function savemodel(model,model_fname,train_mean,train_std,beta,losses=[])
    @info "save model $(model_fname)"
    _savemodel(cpu(model),model_fname,cpu(train_mean),cpu(train_std),cpu(beta),losses)
end

function _savemodel(m,model_fname,train_mean,train_std,beta,losses=[])
    BSON.@save model_fname m train_mean train_std beta losses
end

function snapgrid(lon,Δlon)
    lon0 = minimum(lon);
    ii = round.(Int,(lon[1,:] .- lon0) ./ (Δlon * size(lon,1)))
end

function index_mapping((lon,lat,time),(Δlon,Δlat,Δtime))
    sz = (size(lon,1),size(lat,1))
    lon0 = minimum(lon)
    lat0 = minimum(lat)
    time0 = minimum(time)

    function from_lin_index((lon0,lat0,time0),(Δlon,Δlat,Δtime),(lon,lat,time),sz)
        i = round(Int,(lon - lon0) ./ (Δlon * sz[1]))+1
        j = round(Int,(lat - lat0) ./ (Δlat * sz[2]))+1
        n = round(Int,(time - time0) / Δtime) + 1
        return (i,j,n)
    end
    from_lin_index(l) = from_lin_index((lon0,lat0,time0),(Δlon,Δlat,Δtime),(lon[1,l],lat[1,l],time[l]),sz)
    function to_lin_index(i,j,n)
        if checkbounds(Bool,patch_index,i,j,n)
            return @inbounds patch_index[i,j,n]
        else
            return 0
        end
    end


    ii = snapgrid(lon,Δlon) .+ 1
    jj = snapgrid(lat,Δlat) .+ 1
    nn = round.(Int,(time - time0) ./ Δtime) .+ 1

    patch_index = zeros(Int,maximum(ii),maximum(jj),maximum(nn));
    for l = 1:size(lon,2)
        patch_index[ii[l],jj[l],nn[l]] = l
    end

    return from_lin_index,to_lin_index
end



struct Dataset6{T,N,Trng,Tdevice,Ta,Taux}
    train_input::Array{T,N}
    rng::Trng
    steps::Int64
    train_mean::T
    train_std::T
    device::Tdevice
    alpha_bar::Ta
    auxdata_loader::Taux
    training::Bool
end

numobs(d::Dataset6) = size(d.train_input)[end]

function getobs_orig(d::Dataset6,index::Union{AbstractVector,Integer})
    rng = d.rng
    auxdata_loader = d.auxdata_loader
    sz = size(d.train_input)[1:2]

    x0_cpu = d.train_input[:,:,:,index]

    if d.training
        index_mask = rand(rng,1:size(d.train_input,4),length(index))
        x_mask_cpu = d.train_input[:,:,:,index_mask]
    else
        # no masking
        x_mask_cpu = zeros(Float32,size(x0_cpu))
    end

    if auxdata_loader !== nothing
        aux_data = zeros(Float32,sz...,naux_data(auxdata_loader),length(index));
        load_aux_data!(auxdata_loader,index,aux_data)

        # do not mask additional data for other time instances
        aux_data_mask_cpu = zeros(Float32,size(aux_data))

        # try
        for k = 1:size(aux_data,3)
            index_mask = rand(rng,1:size(d.train_input,4),length(index))
            aux_data_mask_cpu[:,:,k,:] = d.train_input[:,:,1,index_mask]
        end

        x0_cpu = cat(x0_cpu,aux_data,dims=3)
        x_mask_cpu = cat(x_mask_cpu,aux_data_mask_cpu,dims=3)
    end

    x0_cpu = (x0_cpu .- d.train_mean) ./ d.train_std

    return (x0_cpu,x_mask_cpu)
end

function getobs(d::Dataset6,index::Union{AbstractVector,Integer})
    device = d.device
    alpha_bar = d.alpha_bar
    T = d.steps
    rng = d.rng
    sz = size(d.train_input)[1:2]

    x0_cpu,x_mask_cpu = getobs_orig(d,index)

    x0 = x0_cpu |> device
    x_mask = x_mask_cpu |> device

    has_no_data_orig = isnan.(x0)
    # where we pretend there is no data
    has_no_data = @. isnan(x_mask) || isnan(x0)
    # where to evaluate the loss function
    mask = @. isnan(x_mask) & !isnan(x0)

    # necessary because 0 * NaN is NaN
    x0[isnan.(x0)] .= 0;

    t = zeros(Int16,size(x0)) |> device
    t .= device(rand(rng,1:T,1,1,1,length(index)));
    t .= device(rand(rng,200:200,1,1,1,length(index)));
    #@show cpu(t)[1],T
    t[.!has_no_data] .= 1 # at uncorrupted stage
    t[has_no_data_orig] .= T # at fully corrupted stage
    eps = randn(rng,size(x0)) |> device

    xt = sqrt.(alpha_bar[t]) .* x0 + sqrt.(1 .- alpha_bar[t]) .* eps

    tt = Float32.((t .- 1) ./ (T .- 1) .- 0.5) |> device

    return (xt,tt,eps,mask)
end

function noise_schedule(beta)
    alpha = 1 .- beta
    alpha_bar = exp.(cumsum(log.(alpha)))
    sigma = sqrt.(beta)
    return alpha,alpha_bar,sigma
end

"""
All parameters, even with default values, can be domain specific
"""
function train!(model,dl;
                device = cpu,
                nb_epochs = 100,
                learning_rate = 1e-3,
                learning_rate_drop_epoch = 20,
                learning_rate_factor = 0.9,
                batch_size = 32,
                T = 1000,
                beta = LinRange(1e-4, 0.02, T),
                checkpoint_dirname = "",
                checkpoint_epoch = 0,
                auxdata_loader = nothing,
                rng = Random.GLOBAL_RNG,
                train_mean = 0,
                train_std = 1,
              )

    alpha,alpha_bar,sigma = device.(noise_schedule(beta))

    params = Flux.params(model)
    nb_parameters = sum(length.(params))
    println("nb_parameters ",nb_parameters)

    optimizer = ADAM(learning_rate)
    losses = Float32[]

    @time for k = 1:nb_epochs
        if k % learning_rate_drop_epoch == 0
            optimizer.eta *= learning_rate_factor
            @info "optimizer.eta " optimizer.eta
        end

        acc_loss = 0
        acc_count = 0

        #=
          (xt,tt,eps,mask) = first(dl)
        =#
        for (xt,tt,eps,mask) in dl
            loss, back = Flux.pullback(params) do
                ϵ = model((xt, tt))
                difference = (eps - ϵ) .* mask
                mean(difference.^2)
            end

            grad = back(1f0)

            Flux.update!(optimizer, params, grad)

            acc_loss += loss * size(xt)[end]
            acc_count += size(xt)[end]
        end

        push!(losses, acc_loss / acc_count)

        println("epoch: ",k," ",losses[end])

        if isnan(acc_loss)
            error("loss is NaN")
        end

        if (checkpoint_dirname != "") && (k % checkpoint_epoch == 0)
            savemodel(model,checkpoint_dirname,k,train_mean,train_std,beta,losses)
        end

        GC.gc()
        CUDA.reclaim()
    end

    return alpha, alpha_bar, sigma, losses
end


function prep_data!(data_input,fv,trans)
    @inbounds  for i in eachindex(data_input)
        if data_input[i] == fv
            data_input[i] = NaN
        elseif data_input[i] <= 0
            data_input[i] = NaN
        else
            data_input[i] = trans(data_input[i])
        end
    end
    nothing
end

function ncload(fname_train,varname,trans=log10)

    ds = NCDataset(fname_train)
    data_sz = size(ds[varname])
    train_input = zeros(Float32,(data_sz[1],data_sz[2],1,data_sz[3]));

    @inbounds NCDatasets.load!(ds[varname].var,
                               train_input,:,:,:)

    fv = get(ds[varname].attrib,"_FillValue",NaN)
    prep_data!(train_input,fv,trans)
    return train_input
end


function extend(train_input)
    sz = size(train_input)
    sz2 = 2 .^ ceil.(Int,log2.(sz[1:2]))

    if sz2 !== sz[1:2]
        @warn "input size is not a power of 2 $sz"
        train_input_bak = train_input;
        train_input = zeros(eltype(train_input),(sz2...,sz[3:end]...));
        train_input .= NaN
        train_input[1:sz[1],1:sz[2],:,:] .= train_input_bak;
    end
    return train_input
end

random(T,min,max) = min + (max-min) * rand(T)
random(min,max) = random(Float64,min,max)


struct PatchIndex1{N,T,TT,TD}
    sz::NTuple{N,Int}
    lon::Array{T,2}
    lat::Array{T,2}
    time::Vector{TT}
    lon0::T
    lat0::T
    time0::TT
    Δlon::T
    Δlat::T
    Δtime::TD
    patch_index::Array{Int,3}
end

function PatchIndex1((lon,lat,time),(Δlon,Δlat,Δtime))
    sz = (size(lon,1),size(lat,1))
    lon0 = minimum(lon)
    lat0 = minimum(lat)
    time0 = minimum(time)

    ii = snapgrid(lon,Δlon) .+ 1
    jj = snapgrid(lat,Δlat) .+ 1
    nn = round.(Int,(time - time0) ./ Δtime) .+ 1

    patch_index = zeros(Int,maximum(ii),maximum(jj),maximum(nn));
    for l = 1:size(lon,2)
        patch_index[ii[l],jj[l],nn[l]] = l
    end

    PatchIndex1(
        sz,
        lon,
        lat,
        time,
        lon0,
        lat0,
        time0,
        Δlon,
        Δlat,
        Δtime,
        patch_index,
    )
end


function _from_lin_index((lon0,lat0,time0),(Δlon,Δlat,Δtime),(lon,lat,time),sz)
    i = round(Int,(lon - lon0) ./ (Δlon * sz[1]))+1
    j = round(Int,(lat - lat0) ./ (Δlat * sz[2]))+1
    n = round(Int,(time - time0) / Δtime) + 1
    return (i,j,n)
end

from_lin_index(pi::PatchIndex1,l) = _from_lin_index(
    (pi.lon0,pi.lat0,pi.time0),(pi.Δlon,pi.Δlat,pi.Δtime),
    (pi.lon[1,l],pi.lat[1,l],pi.time[l]),pi.sz)

function to_lin_index(pi::PatchIndex1,i,j,n)
    if checkbounds(Bool,pi.patch_index,i,j,n)
        return @inbounds pi.patch_index[i,j,n]
    else
        return 0
    end
end


struct AuxData3{T,N,TPI}
    lon::Array{T,2}
    lat::Array{T,2}
    lon_range::NTuple{2,T}
    lat_range::NTuple{2,T}
    cos_time::Vector{T}
    sin_time::Vector{T}
    train_input::Array{T,4}
    ntime_win::Int
    pi::TPI
end

function AuxData3(
    coord,(Δlon,Δlat,Δtime),train_input,
    ntime_win;
    lon_range = extrema(coord[1]),
    lat_range = extrema(coord[2]),
    cycle = 365.25)

    (lon,lat,time) = coord

    cos_time = @. cos(2π * Dates.dayofyear(time)/cycle)
    sin_time = @. sin(2π * Dates.dayofyear(time)/cycle)

    pi = PatchIndex1((lon,lat,time),(Δlon,Δlat,Δtime))
    AuxData3{Float32,2,typeof(pi)}(
        lon,
        lat,
        lon_range,
        lat_range,
        cos_time,
        sin_time,
        train_input,
        ntime_win,
        pi)
end

#naux_data(auxd::AuxData3) = 2 + 2 + 2 * (auxd.ntime_win-1)
#naux_data(auxd::AuxData3) = 2 * (auxd.ntime_win-1)
naux_data(auxd::AuxData3) = (auxd.ntime_win-1)

normalize(x,x_range) = (x .- x_range[1]) ./ (x_range[2] - x_range[1])

function load_aux_data!(auxd::AuxData3,index,aux_data)
    ntime_win = auxd.ntime_win
    train_input = auxd.train_input

    for l = 1:length(index)
        i,j,n = from_lin_index(auxd.pi,l)

        # aux_data[:,:,1,l] .= normalize(auxd.lon[:,index[l]],auxd.lon_range)
        # aux_data[:,:,2,l] .= normalize(auxd.lat[:,index[l]],auxd.lat_range)'
        # aux_data[:,:,3,l] .= auxd.cos_time[l]
        # aux_data[:,:,4,l] .= auxd.sin_time[l]
        # baseindex = 5

        baseindex = 1

        for islice = ((1:ntime_win) .- (ntime_win+1)÷2)
            if islice == 0
                continue
            end

            n2 = n + islice
            index2 = to_lin_index(auxd.pi,i,j,n2)
            if index2 != 0
                aux_data[:,:,baseindex,  l] = replace(train_input[:,:,1,index2],NaN => 0)
#                aux_data[:,:,baseindex+1,l] = isfinite.(train_input[:,:,1,index2])
            else
                aux_data[:,:,baseindex,  l] .= 0
#                aux_data[:,:,baseindex+1,l] .= 0
            end
#            baseindex += 2
            baseindex += 1
        end
    end
end
