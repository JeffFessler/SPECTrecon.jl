# rotate3.jl

using Main.SPECTrecon: linearinterp!, rotate_x!, rotate_y!
using Main.SPECTrecon: rotate_x_adj!, rotate_y_adj!
using Main.SPECTrecon: rotl90!, rotr90!, rot180!
using Main.SPECTrecon: imrotate3!, imrotate3_adj!
using BenchmarkTools: @btime
using ImageFiltering: padarray
using OffsetArrays


function linearinterp_time()
    T = Float32
    x = rand() * ones(T, 100)
    interp_x = SparseInterpolator(LinearSpline(T), x, length(x))
    @btime linearinterp!($interp_x, $x) # 421.226 ns (0 allocations: 0 bytes)
    nothing
end


function rotate_x_time()
    N = 100 # assume M = N
    T = Float32
    img = rand(T, N, N)
    output = rand(T, N, N)
    θ = T(3*π/11)
    xi = T.(1:N)
    yi = T.(1:N)
    interp_x = SparseInterpolator(LinearSpline(T), xi, length(xi))
    workvec = rand(T, N)
    c_y = 1
    # todo: test values, how to test values?
    @btime rotate_x!($output, $img, $θ, $xi, $yi, $interp_x, $workvec, $c_y)
    # 69.822 μs (0 allocations: 0 bytes)
    nothing
end


function rotate_x_adj_time()
    N = 100 # assume M = N
    T = Float32
    img = rand(T, N, N)
    output = rand(T, N, N)
    θ = T(3*π/11)
    xi = T.(1:N)
    yi = T.(1:N)
    interp_x = SparseInterpolator(LinearSpline(T), xi, length(xi))
    workvec = rand(T, N)
    c_y = 1
    @btime rotate_x_adj!($output, $img, $θ, $xi, $yi, $interp_x, $workvec, $c_y)
    # 84.405 μs (0 allocations: 0 bytes)
    nothing
end


function rotate_y_time()
    N = 100 # assume M = N
    T = Float32
    img = rand(T, N, N)
    output = rand(T, N, N)
    θ = T(3*π/11)
    xi = T.(1:N)
    yi = T.(1:N)
    interp_x = SparseInterpolator(LinearSpline(T), xi, length(xi))
    workvec = rand(T, N)
    c_x = 1
    @btime rotate_y!($output, $img, $θ, $xi, $yi, $interp_x, $workvec, $c_x)
    # 72.389 μs (0 allocations: 0 bytes)
    nothing
end


function rotate_y_adj_time()
    N = 100 # assume M = N
    T = Float32
    img = rand(T, N, N)
    output = rand(T, N, N)
    θ = T(3*π/11)
    xi = T.(1:N)
    yi = T.(1:N)
    interp_x = SparseInterpolator(LinearSpline(T), xi, length(xi))
    workvec = rand(T, N)
    c_x = 1
    @btime rotate_y_adj!($output, $img, $θ, $xi, $yi, $interp_x, $workvec, $c_x)
    # 89.847 μs (0 allocations: 0 bytes)
    nothing
end


function rotl90_time()
    T = Float32
    N = 100
    A = rand(T, N, N)
    B = rand(T, N, N)
    @btime rotl90!($B, $A) # 3.608 μs (0 allocations: 0 bytes)
    nothing
end


function rotr90_time()
    T = Float32
    N = 100
    A = rand(T, N, N)
    B = rand(T, N, N)
    @btime rotr90!($B, $A) # 3.700 μs (0 allocations: 0 bytes)
    nothing
end


function rot180_time()
    T = Float32
    N = 100
    A = rand(T, N, N)
    B = rand(T, N, N)
    @btime rot180!($B, $A) # 3.798 μs (0 allocations: 0 bytes)
    nothing
end


function imrotate3_1d_time()
    T = Float32
    M = 100
    N = 100
    pad_x = ceil(Int, 1 + M * sqrt(2)/2 - M / 2)
    pad_y = ceil(Int, 1 + N * sqrt(2)/2 - N / 2)
    workvec_x = zeros(T, M + 2 * pad_x)
    workvec_y = zeros(T, N + 2 * pad_y)
    A_x = SparseInterpolator(LinearSpline(T), workvec_x, length(workvec_x))
    A_y = SparseInterpolator(LinearSpline(T), workvec_y, length(workvec_y))
    img = zeros(T, M, N)
    img[30:50, 20:60] .= 1
    output = similar(img)
    workmat1 = OffsetArrays.no_offset_view(padarray(img, Fill(0, (pad_x, pad_y))))
    workmat2 = similar(workmat1)
    θ = 3*π/16
    @btime imrotate3!($output, $workmat1, $workmat2, $img, $θ, $A_x, $A_y, $workvec_x, $workvec_y)
    # 557.079 μs (0 allocations: 0 bytes)
    nothing
end


function imrotate3_1d_adj_time()
    T = Float32
    M = 100
    N = 100
    pad_x = ceil(Int, 1 + M * sqrt(2)/2 - M / 2)
    pad_y = ceil(Int, 1 + N * sqrt(2)/2 - N / 2)
    workvec_x = zeros(T, M + 2 * pad_x)
    workvec_y = zeros(T, N + 2 * pad_y)
    A_x = SparseInterpolator(LinearSpline(T), workvec_x, length(workvec_x))
    A_y = SparseInterpolator(LinearSpline(T), workvec_y, length(workvec_y))
    img = zeros(T, M, N)
    img[30:50, 20:60] .= 1
    output = similar(img)
    workmat1 = OffsetArrays.no_offset_view(padarray(img, Fill(0, (pad_x, pad_y))))
    workmat2 = similar(workmat1)
    θ = 3*π/16
    @btime imrotate3_adj!($output, $workmat1, $workmat2, $img, $θ, $A_x, $A_y, $workvec_x, $workvec_y)
    # 544.053 μs (0 allocations: 0 bytes)
    nothing
end


function imrotate3_2d_time()
    T = Float32
    M = 100
    N = 100
    pad_x = ceil(Int, 1 + M * sqrt(2)/2 - M / 2)
    pad_y = ceil(Int, 1 + N * sqrt(2)/2 - N / 2)
    img = zeros(T, M, N)
    img[30:60,20:70] .= 1
    output = similar(img)
    workmat1 = OffsetArrays.no_offset_view(padarray(img, Fill(0, (pad_x, pad_y))))
    workmat2 = similar(workmat1)
    θ = 3π/16
    @btime imrotate3!($output, $workmat1, $workmat2, $img, $θ)
    # 220.708 μs (0 allocations: 0 bytes)
    nothing
end


function imrotate3_2d_adj_time()
    T = Float32
    M = 100
    N = 100
    pad_x = ceil(Int, 1 + M * sqrt(2)/2 - M / 2)
    pad_y = ceil(Int, 1 + N * sqrt(2)/2 - N / 2)
    img = zeros(T, M, N)
    img[30:60,20:70] .= 1
    output = similar(img)
    workmat1 = OffsetArrays.no_offset_view(padarray(img, Fill(0, (pad_x, pad_y))))
    workmat2 = similar(workmat1)
    θ = 3π/16
    @btime imrotate3_adj!($output, $workmat1, $workmat2, $img, $θ)
    # 216.833 μs (0 allocations: 0 bytes)
    nothing
end


# run all functions, time may vary on different machines, but should be all zero allocation.
linearinterp_time()
rotate_x_time()
rotate_x_adj_time()
rotate_y_time()
rotate_y_adj_time()
rotl90_time()
rotr90_time()
rot180_time()
imrotate3_1d_time()
imrotate3_1d_adj_time()
imrotate3_2d_time()
imrotate3_2d_adj_time()
