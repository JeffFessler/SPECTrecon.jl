# helper.jl

using BenchmarkTools: @btime
using Main.SPECTrecon: padzero!, padrepl!, pad2sizezero!, pad_it!
using Main.SPECTrecon: fftshift!, ifftshift!, fftshift2!
using Main.SPECTrecon: plus1di!, plus1dj!, plus2di!, plus2dj!
using Main.SPECTrecon: plus3di!, plus3dj!, plus3dk!, scale3dj!, mul3dj!
using Main.SPECTrecon: copy3dj!


function padzero_time()
    T = Float32
    x = randn(T, 7, 5)
    y = randn(T, 3, 3)
    @btime padzero!($x, $y, (2, 2, 1, 1)) # 10.648 ns (0 allocations: 0 bytes)
    nothing
end


function padrepl_time()
    T = Float32
    x = randn(T, 10, 9)
    y = randn(T, 5, 4)
    @btime padrepl!($x, $y, (1, 4, 3, 2)) # 61.129 ns (0 allocations: 0 bytes)
    nothing
end


function pad2size_time()
    T = Float32
    ker = reshape(Int16(1):Int16(9), 3, 3)
    padsize = (8, 8)
    z = randn(T, padsize)
    @btime pad2sizezero!($z, $ker, $padsize) # 16.899 ns (0 allocations: 0 bytes)
    @btime pad_it!($ker, $padsize) #32.320 ns (0 allocations: 0 bytes)
    nothing
end

function fft_time()
    b = [1 2;3 4]
    x = [b 2*b;3*b 4*b]
    y = similar(x)
    @btime fftshift!($y, $x) # 76.447 ns (0 allocations: 0 bytes)
    @btime ifftshift!($x, $y) # 82.149 ns (0 allocations: 0 bytes)
    nothing
end


function fft2_time()
    T = Float32
    x = randn(T, 120, 128)
    z = similar(x)
    @btime fftshift!($z, $x) # 3.095 μs (0 allocations: 0 bytes)
    @btime fftshift2!($z, $x) # 2.776 μs (0 allocations: 0 bytes)
    nothing
end


function plus1di_time()
    T = Float32
    x = randn(T, 4, 64)
    v = randn(T, 64)
    @btime plus1di!($x, $v, 2) # 28.569 ns (0 allocations: 0 bytes)
    nothing
end


function plus1dj_time()
    T = Float32
    x = randn(T, 64, 4)
    v = randn(T, 64)
    @btime plus1dj!($x, $v, 2) # 5.257 ns (0 allocations: 0 bytes)
    nothing
end


function plus2di_time()
    T = Float32
    x = randn(64)
    v = randn(4, 64)
    @btime plus2di!($x, $v, 2) # 28.383 ns (0 allocations: 0 bytes)
    nothing
end


function plus2dj_time()
    T = Float32
    x = randn(T, 64)
    v = randn(T, 64, 4)
    @btime plus2dj!($x, $v, 2) # 6.207 ns (0 allocations: 0 bytes)
    nothing
end


function plus3di_time()
    T = Float32
    x = randn(T, 64, 64)
    v = randn(T, 4, 64, 64)
    @btime plus3di!($x, $v, 2) # 1.711 μs (0 allocations: 0 bytes)
    nothing
end


function plus3dj_time()
    T = Float32
    x = randn(T, 64, 64)
    v = randn(T, 64, 4, 64)
    @btime plus3dj!($x, $v, 2) # 304.141 ns (0 allocations: 0 bytes)
    nothing
end


function plus3dk_time()
    T = Float32
    x = randn(T, 64, 64)
    v = randn(T, 64, 64, 4)
    @btime plus3dk!($x, $v, 2) # 279.628 ns (0 allocations: 0 bytes)
    nothing
end


function scale3dj_time()
    T = Float32
    x = randn(T, 64, 64)
    v = randn(T, 64, 4, 64)
    s = -0.5
    @btime scale3dj!($x, $v, 2, $s) # 408.392 ns (0 allocations: 0 bytes)
    nothing
end


function mul3dj_time()
    T = Float32
    x = randn(T, 64, 4, 64)
    v = randn(T, 64, 64)
    @btime mul3dj!($x, $v, 2) # 15.197 μs (0 allocations: 0 bytes)
    nothing
end


function copy3dj_time()
    T = Float32
    x = randn(T, 64, 64)
    v = randn(T, 64, 4, 64)
    @btime copy3dj!($x, $v, 2) # 226.772 ns (0 allocations: 0 bytes)
    nothing
end


# run all functions, time may vary on different machines, but should be all zero allocation.
padzero_time()
padrepl_time()
pad2size_time()
fft_time()
fft2_time()
plus1di_time()
plus1dj_time()
plus2di_time()
plus2dj_time()
plus3di_time()
plus3dj_time()
plus3dk_time()
scale3dj_time()
mul3dj_time()
copy3dj_time()
