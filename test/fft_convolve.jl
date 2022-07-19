# fft_convolve.jl

using SPECTrecon: imfilter3!
using SPECTrecon: fft_conv!, fft_conv_adj!
using LinearAlgebra: dot
using FFTW: plan_fft!, plan_ifft!
using Random: seed!
using ImageFiltering: centered, imfilter
using Test: @test, @testset, detect_ambiguities


@testset "imfilter3!" begin
    N = 64
    T = Float32
    #img = zeros(T, N, N); img[20:50, 20:40] .= ones(31,21)
    img = rand(T, N, N) # random image that goes all the way to the edge!
    output = similar(img)
    ker = rand(T, 3, 3) / 9
    img_compl = similar(img, Complex{T})
    ker_compl = similar(img_compl)
    fft_plan = plan_fft!(img_compl)
    ifft_plan = plan_ifft!(img_compl)
    copyto!(img_compl, img)
    imfilter3!(output, img_compl, reverse(ker), ker_compl, fft_plan, ifft_plan)
    y = imfilter(img, centered(ker), "circular")
    @test isapprox(y, output)
end


@testset "fft-conv-adj-test" begin
    M = 200
    N = 64
    T = Float32
    testnum = 100
    fftpadsize = (28, 28, 32, 32)
    img_compl = zeros(Complex{T}, 256, 128)
    ker_compl = similar(img_compl)
    workmat = zeros(T, 256, 128)
    workvec1 = zeros(T, 128)
    workvec2 = zeros(T, 256)
    fft_plan = plan_fft!(img_compl)
    ifft_plan = plan_ifft!(img_compl)
    seed!(0) # todo: fails sometimes
    for t = 1:testnum
        x = randn(T, M, N)
        output_x = similar(x)
        y = randn(T, M, N)
        output_y = similar(y)
        ker = rand(T, 11, 11) # change from 5 -> 11
        ker /= sum(ker)
        kerev = reverse(ker)
        fft_conv!(output_x, workmat, x, ker, fftpadsize,
            img_compl, ker_compl, fft_plan, ifft_plan)

        fft_conv_adj!(output_y, workmat, workvec1, workvec2, y, kerev, fftpadsize,
            img_compl, ker_compl, fft_plan, ifft_plan)
        @test isapprox(dot(y, output_x), dot(x, output_y)) # todo tol?
    end
end
