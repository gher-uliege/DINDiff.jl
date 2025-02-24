function showsize(msg)
    function f(x)
        #@show msg,size(x)
        return x
    end
end

cat_channels(x,y) = cat(x,y,dims=3)

n_out_channels(::typeof(+),c1,c2) = c1
n_out_channels(::typeof(cat_channels),c1,c2) = c1+c2

function CatChannelwise(xs)
    xt = cat(xs...,dims=ndims(xs[1])-1)
    return xt
end

function DConv(ks,(in,out), σ = identity; kwargs...)
    [
        Conv(ks,in => out; use_bias = false, kwargs...),
        BatchNorm(out),
        σ,
        Conv(ks,out => out; use_bias = false, kwargs...),
        BatchNorm(out),
        σ,
    ]
end

function block(channels; ks = 3, activation=relu, connection = cat_channels, pool = MaxPool)
    if length(channels) == 2
        inner_block = []
    else
        inner_block = block(channels[2:end]; ks, activation, connection, pool = pool)
    end

    nout = n_out_channels(connection,channels[1],channels[1])

    return [
        SkipConnection(
            Chain(
                pool((2,2)),
                DConv((ks,ks),channels[1]=>channels[2],activation,pad = SamePad())...,
                #showsize("before 3 $(channels[2])"),
                inner_block...,
                ConvTranspose((2,2),channels[2]=>channels[1],activation,pad=SamePad(),stride=2),
        ),
            connection),
        DConv((ks,ks),nout=>channels[1],activation,pad = SamePad())...,
    ]
end



function genmodel(;in_channels = 1,
                  channels = (64,128,256,512,1024),
                  activation = relu,
                  kernel_size = 3,
                  out_channels = 1,
                  connection = cat_channels,
                  out_activation = identity,
                  pool = MaxPool,
                  head = [CatChannelwise],
                  )

    ks = kernel_size
    inner_block = block(channels; activation, ks, connection, pool)

    model = Chain(
        head...,
        DConv((kernel_size,kernel_size),in_channels=>channels[1],activation,pad = SamePad())...,
        #showsize("before 0"),
        inner_block...,
        Conv((1,1),channels[1]=>out_channels,out_activation,pad = SamePad()),
    )
end


