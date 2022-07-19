# spectplan.jl

export SPECTplan, Workarray

"""
    SPECTplan
Struct for storing key factors for a SPECT system model
- `T` datatype of work arrays
- `imgr [nx, ny, nz]` 3D rotated version of image
- `add_img [nx, ny, nz]` 3D image for adding views and backprojection
- `mumap [nx,ny,nz]` attenuation map, must be 3D, possibly zeros()
- `mumapr [nx, ny, nz]` 3D rotated mumap
- `psfs [nx_psf,nz_psf,ny,nview]` point spread function, must be 4D, with `nx_psf` and `nz_psf` odd, and symmetric for each slice
- `nview` number of views, must be integer
- `viewangle` set of view angles, must be from 0 to 2π
- `interpidx` interpolation method, 1 means 1d, 2 means 2d
- `dy` voxel size in y direction (dx is the same value)
- `imgsize{nx, ny, nz}` number of voxels in {x,y,z} direction of the image, must be integer
- `psfsize{nx_psf, nz_psf}` number of voxels in {x, z} direction of the psf, must be integer
- `pad_fft{padu_fft, pad_fft, padl_fft, padr_fft} pixels padded for {left,right,up,down} direction for convolution with psfs, must be integer
- `pad_rot{padu_rot, padd_rot, padl_rot, padr_rot}` padded pixels for {left,right,up,down} direction for image rotation
- `ncore` number of CPU cores used to process data, must be integer
Currently code assumes each of the `nview` projection views is `[nx,nz]`
Currently code assumes `nx = ny`
Currently code assumes uniform angular sampling
Currently code uses multiprocessing using # of cores specified by Threads.nthreads() in Julia
Currently code assumes psf is symmetric
"""
struct SPECTplan
    T::DataType # default type for work arrays etc.
    imgr::AbstractArray{<:RealU, 3} # 3D rotated image, (nx, ny, nz)
    add_img::AbstractArray{<:RealU, 3}
    mumap::AbstractArray{<:RealU, 3} # [nx,ny,nz] attenuation map, must be 3D, possibly zeros()
    mumapr::AbstractArray{<:RealU, 3} # 3D rotated mumap, (nx, ny, nz)
    psfs::AbstractArray{<:RealU, 4} # PSFs must be 4D, [nx_psf, nz_psf, ny, nview], finally be centered psf
    nview::Int # number of views
    viewangle::AbstractVector{<:RealU}
    interpidx::Int
    dy::RealU
    imgsize::NTuple{3, Int}
    psfsize::NTuple{2, Int}
    pad_fft::NTuple{4, Int}
    pad_rot::NTuple{4, Int}
    ncore::Int # number of cores
    # other options for how to do the projection?
    function SPECTplan(
        mumap::AbstractArray{<:RealU, 3},
        psfs::AbstractArray{<:RealU, 4},
        dy::RealU;
        viewangle::AbstractVector{<:RealU} = (0:size(psfs, 4) - 1) / size(psfs, 4) * (2π), # set of view angles
        interpidx::Int = 2, # 1 is for 1d interpolation, 2 is for 2d interpolation
        T::DataType = promote_type(eltype(mumap), Float32),
    )
        # check nx = ny ? typically 128 x 128 x 81
        (nx, ny, nz) = size(mumap)
        @assert isequal(nx, ny)
        @assert iseven(nx) && iseven(ny)

        # check psf
        nx_psf = size(psfs, 1)
        nz_psf = size(psfs, 2)
        nview = size(psfs, 4)
        @assert isodd(nx_psf) && isodd(nz_psf)
        @assert all(mapslices(x -> x == reverse(x), psfs, dims = [1, 2]))

        # check interpidx
        @assert interpidx == 1 || interpidx == 2

        padu_fft = _padup(mumap, psfs)
        padd_fft = _paddown(mumap, psfs)
        padl_fft = _padleft(mumap, psfs)
        padr_fft = _padright(mumap, psfs)
        pad_fft = (padu_fft, padd_fft, padl_fft, padr_fft)

        padu_rot = ceil(Int, 1 + nx * sqrt(2)/2 - nx / 2)
        padd_rot = ceil(Int, 1 + nx * sqrt(2)/2 - nx / 2)
        padl_rot = ceil(Int, 1 + ny * sqrt(2)/2 - ny / 2)
        padr_rot = ceil(Int, 1 + ny * sqrt(2)/2 - ny / 2)
        pad_rot = (padu_rot, padd_rot, padl_rot, padr_rot)

        # size for fft plan
        fftplan_size = Array{Complex{T}}(undef, nx + padu_fft + padd_fft, nz + padl_fft + padr_fft)
        fft_plan = plan_fft!(fftplan_size)
        ifft_plan = plan_ifft!(fftplan_size)

        imgsize = (nx, ny, nz)
        psfsize = (nx_psf, nz_psf)
        ncore = Threads.nthreads()

        # imgr stores 3D image in different view angles
        imgr = zeros(T, nx, ny, nz)
        # add_img stores 3d image for backprojection
        add_img = zeros(T, nx, ny, nz)
        # mumapr stores 3D mumap in different view angles
        mumapr = zeros(T, nx, ny, nz)

        new(T, imgr, add_img, mumap, mumapr, psfs, nview, viewangle, interpidx,
            dy, imgsize, psfsize, pad_fft, pad_rot, ncore)
        #  creates objects of the block's type (inner constructor methods).
    end
end


"""
    Workarray
Struct for storing keys of the work array for a single thread
add tmp vectors to avoid allocating in rotate_x and rotate_y
For fft convolution:
- `workmat_fft [nx+padu_fft+padd_fft, nz+padl_fft+padr_fft]`: 2D padded image for imfilter3
- `workvec_fft_1 [nz+padl_fft+padr_fft,]`: 1D work vector
- `workvec_fft_2 [nx+padu_fft+padd_fft,]`: 1D work vector
- `img_compl [nx+padu_fft+padd_fft, nz+padl_fft+padr_fft]`: 2D [complex] padded image for fft
- `ker_compl [nx+padu_fft+padd_fft, nz+padl_fft+padr_fft]`: 2D [complex] padded image for fft
- `fft_plan` plan for doing fft, see plan_fft!
- `ifft_plan` plan for doing ifft, see plan_ifft!
For image rotation:
- `workmat_rot_1 [nx+padu_rot+padd_rot, ny+padl_rot+padr_rot]`: 2D padded image for image rotation
- `workmat_rot_2 [nx+padu_rot+padd_rot, ny+padl_rot+padr_rot]`: 2D padded image for image rotation
- `workvec_rot_x [nx+padu_rot+padd_rot,]`: 1D work vector for image rotation
- `workvec_rot_y [ny+padl_rot+padr_rot,]`: 1D work vector for image rotation
- `interp_x` sparse interpolator for rotating in x direction
- `interp_y` sparse interpolator for rotating in y direction
For attenuation:
- `exp_mumapr [nx, nz]` 2D exponential rotated mumap
"""
struct Workarray
    workmat_fft::AbstractArray{<:RealU, 2}
    workvec_fft_1::AbstractVector{<:RealU}
    workvec_fft_2::AbstractVector{<:RealU}
    img_compl::AbstractMatrix{<:Any}
    ker_compl::AbstractMatrix{<:Any}
    fft_plan::Union{AbstractFFTs.ScaledPlan, FFTW.cFFTWPlan}
    ifft_plan::Union{AbstractFFTs.ScaledPlan, FFTW.cFFTWPlan}
    workmat_rot_1::AbstractArray{<:RealU, 2}
    workmat_rot_2::AbstractArray{<:RealU, 2}
    workvec_rot_x::AbstractVector{<:RealU}
    workvec_rot_y::AbstractVector{<:RealU}
    interp_x::SparseInterpolator
    interp_y::SparseInterpolator
    exp_mumapr::AbstractArray{<:RealU, 2}
    # add_view::AbstractArray{<:RealU, 2}

    function Workarray(
        T::DataType,
        imgsize::NTuple{3, Int},
        pad_fft::NTuple{4, Int},
        pad_rot::NTuple{4, Int},
    )
        (nx, ny, nz) = imgsize
        (padu_fft, padd_fft, padl_fft, padr_fft) = pad_fft
        (padu_rot, padd_rot, padl_rot, padr_rot) = pad_rot

        # allocate working buffers for each thread:
        # For fft convolution:
        workmat_fft = zeros(T, nx+padu_fft+padd_fft, nz+padl_fft+padr_fft)
        workvec_fft_1 = zeros(T, nz+padl_fft+padr_fft)
        workvec_fft_2 = zeros(T, nx+padu_fft+padd_fft)

        # complex padimg
        img_compl = zeros(Complex{T}, nx+padu_fft+padd_fft, nz+padl_fft+padr_fft)
        # complex kernel
        ker_compl = zeros(Complex{T}, nx+padu_fft+padd_fft, nz+padl_fft+padr_fft)

        fft_plan = plan_fft!(ker_compl)
        ifft_plan = plan_ifft!(ker_compl)

        # For image rotation:
        workmat_rot_1 = zeros(T, nx+padu_rot+padd_rot, ny+padl_rot+padr_rot)
        workmat_rot_2 = zeros(T, nx+padu_rot+padd_rot, ny+padl_rot+padr_rot)
        workvec_rot_x = zeros(T, nx+padu_rot+padd_rot)
        workvec_rot_y = zeros(T, ny+padl_rot+padr_rot)

        interp_x = SparseInterpolator(LinearSpline(T), workvec_rot_x, length(workvec_rot_x))
        interp_y = SparseInterpolator(LinearSpline(T), workvec_rot_y, length(workvec_rot_y))

        # For attenuation:
        exp_mumapr = zeros(T, nx, nz)

        new(workmat_fft, workvec_fft_1, workvec_fft_2, img_compl, ker_compl,
            fft_plan, ifft_plan, workmat_rot_1, workmat_rot_2, workvec_rot_x,
            workvec_rot_y, interp_x, interp_y, exp_mumapr)
    end
end
