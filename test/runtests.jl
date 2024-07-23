using DINDiff
using DINDiff: TimeAppender
using Test

@testset "DINDiff.jl" begin
    xt = TimeAppender((zeros(2,3,4,5),1.))
    @test size(xt) == (2,3,5,5)
    @test all(xt[:,:,1:end-1,:] .== 0)
    @test all(xt[:,:,end,:] .== 1)

    xt = TimeAppender((zeros(2,3,4,5),[1,2,3,4,5]))
    @test size(xt) == (2,3,5,5)
    @test all(xt[:,:,1:end-1,:] .== 0)
    @test all(xt[1,1,end,:] == 1:5)

    # Write your tests here.
end
