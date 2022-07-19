# project-mem.jl
# examine memory

using SPECTrecon
using BenchmarkTools: @btime

T = Float32
nx = 64
nz = 10
mumap = ones(T, nx,nx,nz)

nview = 20
psfs = ones(T, 7, 7, nx, 20)

xtrue = rand(T, nx, nx, nz)

dy = T(4.7952)
nview = size(psfs, 4)
plan = SPECTplan(mumap, psfs, nview, dy; interpidx = 1)
workarray = Vector{Workarray}(undef, plan.ncore)
for i = 1:plan.ncore
    workarray[i] = Workarray(plan.T, plan.imgsize, plan.pad_fft, plan.pad_rot) # allocate
end
(nx, ny, nz) = size(xtrue)
nviews = size(psfs, 4)
views = zeros(T, nx, nz, nviews)

view1 = copy(views[:,:,1])
tmp = project!(view1, xtrue, plan, workarray, 1)
@btime project!($view1, $xtrue, $plan, $workarray, 1)

#tmp = project!(views, xtrue, plan, workarray)
#@btime project!($views, $xtrue, $plan, $workarray)
nothing
