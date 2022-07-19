# helper.jl

using SPECTrecon: padzero!, padrepl!, pad2sizezero!, pad_it!
using SPECTrecon: fftshift!, ifftshift!, fftshift2!
using SPECTrecon: plus1di!, plus1dj!, plus2di!, plus2dj!
using SPECTrecon: plus3di!, plus3dj!, plus3dk!, scale3dj!, mul3dj!
using SPECTrecon: copy3dj!
using ImageFiltering: Fill, Pad, BorderArray
import OffsetArrays
using Test: @test, @testset, detect_ambiguities


@testset "padzero!" begin
    T = Float32
    x = randn(T, 7, 5)
    y = randn(T, 3, 3)
    padzero!(x, y, (2, 2, 1, 1))
    z = OffsetArrays.no_offset_view(BorderArray(y, Fill(0, (2, 1), (2, 1))))
    @test isequal(x, z)
end


@testset "padrepl!" begin
    T = Float32
    x = randn(T, 10, 9)
    y = randn(T, 5, 4)
    padrepl!(x, y, (1, 4, 3, 2)) # up, down, left, right
    z = OffsetArrays.no_offset_view(BorderArray(y, Pad(:replicate, (1, 3), (4, 2)))) # up, left, down, right
    @test isequal(x, z)
end


@testset "pad2size!" begin
    T = Float32
    ker = reshape(Int16(1):Int16(9), 3, 3)
    padsize = (8, 8)
    z = randn(T, padsize)
    pad2sizezero!(z, ker, padsize)
    tmp = pad_it!(ker, padsize)
    @test isequal(tmp, z)
end


@testset "fftshift2" begin
    T = Float32
    x = randn(T, 120, 128)
    y = similar(x)
    z = similar(x)
    fftshift!(y, x)
    fftshift2!(z, x)
    @test isequal(y, z)
end


@testset "plus1di" begin
    T = Float32
    x = randn(T, 4, 64)
    v = randn(T, 64)
    y = x[2, :] .+ v
    plus1di!(x, v, 2)
    @test isequal(x[2, :], y)
end


@testset "plus1dj!" begin
    T = Float32
    x = randn(T, 64, 4)
    v = randn(T, 64)
    y = x[:, 2] .+ v
    plus1dj!(x, v, 2)
    @test isequal(x[:, 2], y)
end


@testset "plus2di!" begin
    T = Float32
    x = randn(64)
    v = randn(4, 64)
    y = x .+ v[2, :]
    plus2di!(x, v, 2)
    @test isequal(x, y)
end


@testset "plus2dj!" begin
    T = Float32
    x = randn(T, 64)
    v = randn(T, 64, 4)
    y = x .+ v[:, 2]
    plus2dj!(x, v, 2)
    @test isequal(x, y)
end


@testset "plus3di!" begin
    T = Float32
    x = randn(T, 64, 64)
    v = randn(T, 4, 64, 64)
    y = x .+ v[2, :, :]
    plus3di!(x, v, 2)
    @test isequal(x, y)
end


@testset "plus3dj!" begin
    T = Float32
    x = randn(T, 64, 64)
    v = randn(T, 64, 4, 64)
    y = x .+ v[:, 2, :]
    plus3dj!(x, v, 2)
    @test isequal(x, y)
end


@testset "plus3dk!" begin
    T = Float32
    x = randn(T, 64, 64)
    v = randn(T, 64, 64, 4)
    y = x .+ v[:, :, 2]
    plus3dk!(x, v, 2)
    @test isequal(x, y)
end


@testset "scale3dj!" begin
    T = Float32
    x = randn(T, 64, 64)
    v = randn(T, 64, 4, 64)
    s = -0.5
    y = s * v[:, 2, :]
    scale3dj!(x, v, 2, s)
    @test isequal(x, y)
end


@testset "mul3dj!" begin
    T = Float32
    x = randn(T, 64, 4, 64)
    v = randn(T, 64, 64)
    y = x[:,2,:] .* v
    mul3dj!(x, v, 2)
    @test isequal(x[:,2,:], y)
end


@testset "copy3dj!" begin
    T = Float32
    x = randn(T, 64, 64)
    v = randn(T, 64, 4, 64)
    y = v[:,2,:]
    copy3dj!(x, v, 2)
    @test isequal(x, y)
end
