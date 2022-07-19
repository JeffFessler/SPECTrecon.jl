# project.jl

using BenchmarkTools: @btime
using Main.SPECTrecon: SPECTplan, Workarray
using Main.SPECTrecon: project!


function project_time()
    T = Float32
    nx = 64
    ny = 64
    nz = 40
    nview = 60

    mumap = rand(T, nx, ny, nz)

    nx_psf = 19
    nz_psf = 19
    psfs = rand(T, nx_psf, nz_psf, ny, nview)
    psfs = psfs .+ mapslices(reverse, psfs, dims = [1, 2])
    psfs = psfs ./ mapslices(sum, psfs, dims = [1, 2])

    xtrue = rand(T, nx, ny, nz)

    dy = T(4.7952)

    plan1d = SPECTplan(mumap, psfs, dy; interpidx = 1)
    plan2d = SPECTplan(mumap, psfs, dy; interpidx = 2)

    workarray1d = Vector{Workarray}(undef, plan1d.ncore)
    workarray2d = Vector{Workarray}(undef, plan2d.ncore)

    for i = 1:plan1d.ncore
        workarray1d[i] = Workarray(plan1d.T, plan1d.imgsize, plan1d.pad_fft, plan1d.pad_rot) # allocate
    end

    for i = 1:plan2d.ncore
        workarray2d[i] = Workarray(plan2d.T, plan2d.imgsize, plan2d.pad_fft, plan2d.pad_rot) # allocate
    end


    views1d = zeros(T, nx, nz, nview)
    views2d = zeros(T, nx, nz, nview)

    @btime project!($views1d, $xtrue, $plan1d, $workarray1d) # 626.873 ms (129631 allocations: 4.54 MiB)
    @btime project!($views2d, $xtrue, $plan2d, $workarray2d) # 512.367 ms (129672 allocations: 4.54 MiB)
    nothing
end

# run all functions, time may vary on different machines, will allocate ~4MB memory.
project_time()
