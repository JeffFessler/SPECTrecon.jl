# fft_convolve.jl

using MIRTjim: jim
using BenchmarkTools: @btime
using Main.SPECTrecon: imfilter3!
using Main.SPECTrecon: fft_conv!, fft_conv_adj!
using FFTW: plan_fft!, plan_ifft!


function imfilter3_time()
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
    @btime imfilter3!($output, $img_compl, $ker, $ker_compl, $fft_plan, $ifft_plan)
    # 29.746 μs (0 allocations: 0 bytes)
    nothing
end


function fft_conv_time()
    M = 200
    N = 64
    T = Float32
    img = zeros(T, M, N)
    img[20:150, 20:40] .= rand(131,21)
    output = similar(img)
    ker = ones(T, 3, 3) / 9
    fftpadsize = (28, 28, 32, 32)

    img_compl = zeros(Complex{T}, 256, 128)
    workmat = zeros(T, 256, 128)
    ker_compl = similar(img_compl)
    fft_plan = plan_fft!(img_compl)
    ifft_plan = plan_ifft!(img_compl)
    @btime fft_conv!($output, $workmat, $img, $ker, $fftpadsize,
            $img_compl, $ker_compl, $fft_plan, $ifft_plan)
    # 455.066 μs (0 allocations: 0 bytes)
    nothing
end


function fft_conv_adj_time()
    M = 200
    N = 64
    T = Float32
    img = zeros(T, M, N)
    img[20:150, 20:40] .= rand(131,21)
    output = similar(img)
    ker = ones(T, 3, 3) / 9
    fftpadsize = (28, 28, 32, 32)

    img_compl = zeros(Complex{T}, 256, 128)
    workmat = zeros(T, 256, 128)
    workvec1 = zeros(T, 128)
    workvec2 = zeros(T, 256)
    ker_compl = similar(img_compl)
    fft_plan = plan_fft!(img_compl)
    ifft_plan = plan_ifft!(img_compl)
    @btime fft_conv_adj!($output, $workmat, $workvec1, $workvec2, $img, $ker,
        $fftpadsize, $img_compl, $ker_compl, $fft_plan, $ifft_plan)
    # 449.281 μs (0 allocations: 0 bytes)
    nothing
end


# run all functions, time may vary on different machines, but should be all zero allocation.
imfilter3_time()
fft_conv_time()
fft_conv_adj_time()
