#---------------------------------------------------------
# # [SPECTrecon PSF](@id 3-psf)
#---------------------------------------------------------

# This page explains the PSF portion of the Julia package
# [`SPECTrecon.jl`](https://github.com/JeffFessler/SPECTrecon.jl).

# ### Setup

# Packages needed here.

using SPECTrecon
using MIRTjim: jim, prompt
using Plots: scatter, scatter!, plot!, default
default(markerstrokecolor=:auto, markersize=3)

# The following line is helpful when running this example.jl file as a script;
# this way it will prompt user to hit a key after each figure is displayed.

isinteractive() ? jim(:prompt, true) : prompt(:draw);

# ### Overview

# After rotating the image and the attenuation map,
# second step in SPECT image forward projection
# is to apply depth-dependent point spread function (PSF).
# Each (rotated) image plane
# is a certain distance from the SPECT detector
# and must be convolved with the 2D PSF appropriate
# for that plane.

# Because SPECT has relatively poor spatial resolution,
# the PSF is usually fairly wide,
# so convolution using FFT operations
# is typically more efficient
# than direct spatial convolution.

# Following other libraries like
# [FFTW.jl](https://github.com/JuliaMath/FFTW.jl),
# the PSF operations herein start with a `plan`
# where work arrays are preallocated
# for subsequent use.
# The `plan` is a `Vector` of `PlanPSF` objects:
# one for each thread.
# (Parallelism is across planes for a 3D image volume.)
# The number of threads defaults to `Threads.nthreads()`.


# ### Example

# Start with a 3D image volume.

T = Float32 # work with single precision to save memory
nx = 32
nz = 30
image = zeros(T, nx, nx, nz) # ny = nx required
image[1nx÷4, 1nx÷4, 3nz÷4] = 1
image[2nx÷4, 2nx÷4, 2nz÷4] = 1
image[3nx÷4, 3nx÷4, 1nz÷4] = 1
jim(image, "Original image")


# Create a synthetic depth-dependent PSF for a single view

function fake_psf(nx::Int, nx_psf::Int; factor::Real=0.9)
    psf = zeros(Float32, nx_psf, nx_psf, nx)

    for iy in 1:nx # depth-dependent blur
        r = (-(nx_psf-1)÷2):((nx_psf-1)÷2)
        r2 = abs2.((r / nx_psf) * iy.^0.9)
        tmp = @. exp(-(r2 + r2') / 2)
        psf[:,:,iy] = tmp / maximum(tmp)
    end
    return psf
end

nx_psf = 11
nview = 1 # for simplicity in this illustration
psf = zeros(nx_psf, nx_psf, nx, nview)

psf[:,:,:,1] = fake_psf(nx, nx_psf)
jim(psf, "PSF for each of $nx planes")


# Now plan the PSF modeling
# by specifying
# * the image size (must be square)
# * the PSF size: must be `nx_psf × nx_psf × nx × nview`
# * the `DataType` used for the work arrays.

plan = plan_psf(nx, nz, nx_psf; T)

# Here are the internals for the plan for the first thread:

plan[1]


# With this `plan` pre-allocated, now we can apply the depth-dependent PSF
# to the image volume (assumed already rotated here).

result = similar(image) # allocate memory for the result
fft_conv!(result, image, psf[:,:,:,1], plan) # mutates the first argument
jim(result, "After applying PSF")


# ### Adjoint

# To ensure adjoint consistency between SPECT forward- and back-projection,
# there is also an adjoint routine:

adj = similar(result)
fft_conv_adj!(adj, result, psf[:,:,:,1], plan)
jim(adj, "Adjoint of PSF modeling")


# The adjoint is *not* the same as the inverse
# so one does not expect the output here to match the original image!


# ### LinearMap

# One can form a linear map corresponding to PSF modeling using `LinearMapAA`.
# Perhaps the main purpose is simply for verifying adjoint correctness.

using LinearMapsAA: LinearMapAA

nx, nz, nx_psf = 10, 7, 5 # small size for illustration
psf3 = fake_psf(nx, nx_psf)
plan = plan_psf(nx, nz, nx_psf; T)
idim = (nx,nx,nz)
odim = (nx,nx,nz)
forw! = (y,x) -> fft_conv!(y, x, psf3, plan)
back! = (x,y) -> fft_conv_adj!(x, y, psf3, plan)
A = LinearMapAA(forw!, back!, (prod(odim),prod(idim)); T, odim, idim)

Afull = Matrix(A)
Aadj = Matrix(A')
jim(cat(dims=3, Afull, Aadj'), "Linear map for PSF modeling and its adjoint")


# The following check verifies adjoint consistency:

@assert Afull ≈ Aadj'
