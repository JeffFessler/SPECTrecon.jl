# rotate3.jl

export imrotate3!, imrotate3_adj!

using LinearInterpolators: LinearSpline
using LinearInterpolators: SparseInterpolator, AffineTransform2D, rotate
using LinearInterpolators: TwoDimensionalTransformInterpolator


"""
    linearinterp!(A, x)
Assign key values in `SparseInterpolator` (linear) `A` that are calculated from `x`.
`x` must be a constant vector
"""
function linearinterp!(
    A::SparseInterpolator{<:AbstractFloat},
    x::AbstractVector{<:RealU},
)
    # x must be a constant vector
    dec = ceil(Int, x[1]) - x[1]
    ncoeff = length(A.C)
    ncol = length(x)
    for i = 1:ncoeff
        if isodd(i)
            A.C[i] = dec
        else
            A.C[i] = 1 - dec
        end
    end
    if x[end] <= ncol
        A.J[end] = ceil(Int, x[end])
        for i = ncoeff-1:-1:1
            A.J[i] = max(1, A.J[end] - ceil(Int, (ncoeff - i) / 2))
        end
    else
        A.J[1] = floor(Int, x[1])
        for i = 2:ncoeff
            A.J[i] = min(ncol, A.J[1] + ceil(Int, (i - 1) / 2))
        end
    end
    return A
end


"""
    rotate_x!(output, img, tan_θ, xi, yi, interp, workvec, c_y)
Rotate a 2D image along x axis in clockwise direction using 1d linear interpolation,
storing results in `output`.
Sample locations `xi` and `yi` must be in increasing order.
"""
function rotate_x!(
    output::AbstractMatrix{<:RealU},
    img::AbstractMatrix{<:RealU},
    tan_θ::Real,
    xi::AbstractVector{<:RealU},
    yi::AbstractVector{<:RealU},
    interp::SparseInterpolator{<:AbstractFloat},
    workvec::AbstractVector{<:RealU},
    c_y::Real,
)

    for i = 1:length(yi)
        workvec .= yi[i]
        broadcast!(-, workvec, workvec, c_y)
        broadcast!(*, workvec, workvec, tan_θ)
        broadcast!(+, workvec, workvec, xi)
        linearinterp!(interp, workvec)
        mul!((@view output[:, i]), interp, (@view img[:, i])) # need mul! to avoid allocating
    end
    return output
end


"""
    rotate_x_adj!(output, img, tan_θ, xi, yi, interp, workvec, c_y)
The adjoint of rotating a 2D image along x axis in clockwise direction using 1d linear interpolation,
storing results in `output`
xi and yi must be in increasing order.
"""
function rotate_x_adj!(
    output::AbstractMatrix{<:RealU},
    img::AbstractMatrix{<:RealU},
    tan_θ::Real,
    xi::AbstractVector{<:RealU},
    yi::AbstractVector{<:RealU},
    interp::SparseInterpolator,
    workvec::AbstractVector{<:RealU},
    c_y::Real,
)

    for i = 1:length(yi)
        workvec .= yi[i]
        broadcast!(-, workvec, workvec, c_y)
        broadcast!(*, workvec, workvec, tan_θ)
        broadcast!(+, workvec, workvec, xi)
        linearinterp!(interp, workvec)
        mul!((@view output[:, i]), interp', (@view img[:, i]))
    end
    return output
end


"""
    rotate_y!(output, img, sin_θ, xi, yi, interp, workvec, c_x)
Rotate a 2D image along y axis in clockwise direction using 1d linear interpolation,
storing results in `output`
xi and yi must be in increasing order.
"""
function rotate_y!(
    output::AbstractMatrix{<:RealU},
    img::AbstractMatrix{<:RealU},
    sin_θ::Real,
    xi::AbstractVector{<:RealU},
    yi::AbstractVector{<:RealU},
    interp::SparseInterpolator,
    workvec::AbstractVector{<:RealU},
    c_x::Real,
)

    for i = 1:length(xi)
        workvec .= xi[i]
        broadcast!(-, workvec, workvec, c_x)
        broadcast!(*, workvec, workvec, sin_θ)
        broadcast!(+, workvec, workvec, yi)
        linearinterp!(interp, workvec)
        mul!((@view output[i, :]), interp, (@view img[i, :]))
    end
    return output
end


"""
    rotate_y_adj!(output, img, sin_θ, xi, yi, interp, workvec, c_x)
Adjoint of rotating a 2D image along y axis in clockwise direction using 1d linear interpolation,
storing results in `output`
"""
function rotate_y_adj!(
    output::AbstractMatrix{<:RealU},
    img::AbstractMatrix{<:RealU},
    sin_θ::Real,
    xi::AbstractVector{<:RealU},
    yi::AbstractVector{<:RealU},
    interp::SparseInterpolator,
    workvec::AbstractVector{<:RealU},
    c_x::Real,
)

    for i = 1:length(xi)
        workvec .= xi[i]
        broadcast!(-, workvec, workvec, c_x)
        broadcast!(*, workvec, workvec, sin_θ)
        broadcast!(+, workvec, workvec, yi)
        linearinterp!(interp, workvec)
        mul!((@view output[i, :]), interp', (@view img[i, :]))
    end
    return output
end



"""
    rotl90!(B::AbstractMatrix, A::AbstractMatrix)
In place version of `rotl90`, returning rotation of `A` in `B`.
"""
function rotl90!(B::AbstractMatrix, A::AbstractMatrix)
    ind1, ind2 = axes(A)
    n = first(ind2) + last(ind2)
    for i = axes(A, 1), j = ind2
        B[n - j, i] = A[i, j]
    end
    return B
end


"""
    rotr90!(B::AbstractMatrix, A::AbstractMatrix)
In place version of `rotr90`, returning rotation of `A` in `B`.
"""
function rotr90!(B::AbstractMatrix, A::AbstractMatrix)
    ind1, ind2 = axes(A)
    m = first(ind1) + last(ind1)
    for i = ind1, j = axes(A, 2)
        B[j, m - i] = A[i, j]
    end
    return B
end


"""
    rot180!(B::AbstractMatrix, A::AbstractMatrix)
In place version of `rot180`, returning rotation of `A` in `B`.
"""
function rot180!(B::AbstractMatrix, A::AbstractMatrix)
    ind1, ind2 = axes(A,1), axes(A,2)
    m, n = first(ind1)+last(ind1), first(ind2)+last(ind2)
    for j=ind2, i=ind1
        B[m-i,n-j] = A[i, j]
    end
    return B
end


"""
    rot_f90!(output, img, m)
In-place version of rotating an image by 90/180/270 degrees
"""
function rot_f90!(
    output::AbstractMatrix,
    img::AbstractMatrix,
    m::Int,
)
    if m == 0
        output .= img
    elseif m == 1
        rotl90!(output, img)
    elseif m == 2
        rot180!(output, img)
    elseif m == 3
        rotr90!(output, img)
    else
        throw("invalid m!")
    end
    return output
end


"""
    rot_f90_adj!(output, img, m)
    The adjoint of rotating an image by 90/180/270 degrees
"""
function rot_f90_adj!(
    output::AbstractMatrix,
    img::AbstractMatrix,
    m::Int,
)
    if m == 0
        output .= img
    elseif m == 1
        rotr90!(output, img)
    elseif m == 2
        rot180!(output, img)
    elseif m == 3
        rotl90!(output, img)
    else
        throw("invalid m!")
    end
    return output
end


"""
    imrotate3!(output, workmat1, workmat2, img, θ, interp_x, interp_y, workvec_x, workvec_y)
Rotate an image by angle θ (must be within 0 to 2π) in clockwise direction
using a series of 1d linear interpolations.
"""
function imrotate3!(
    output::AbstractMatrix{<:RealU},
    workmat1::AbstractMatrix{<:RealU},
    workmat2::AbstractMatrix{<:RealU},
    img::AbstractMatrix{<:RealU},
    θ::Real,
    interp_x::SparseInterpolator{<:AbstractFloat},
    interp_y::SparseInterpolator{<:AbstractFloat},
    workvec_x::AbstractVector{<:RealU},
    workvec_y::AbstractVector{<:RealU},
)
    @boundscheck size(output) == size(img) || throw("size")
    @boundscheck size(img, 1) == size(img, 2) || throw("row != col")
    if mod(θ, 2π) ≈ 0
        output .= img
        return output # todo: check!
    end
    m = mod(floor(Int, 0.5 + θ/(π/2)), 4)
    M = size(img, 1)
    N = size(img, 2)
    pad_x = Int((size(workmat1, 1) - M) / 2)
    pad_y = Int((size(workmat1, 2) - N) / 2)
    if θ ≈ m * (π/2)
        padzero!(workmat2, img, (pad_x, pad_x, pad_y, pad_y))
        rot_f90!(workmat1, workmat2, m)
    else
        mod_theta = θ - m * (π/2) # make sure it is between -45 and 45 degree
        tan_mod_theta = tan(mod_theta / 2)
        sin_mod_theta = - sin(mod_theta)
        xi = 1 : M + 2 * pad_x
        yi = 1 : N + 2 * pad_y
        c_x = (length(xi)+1)/2 # center of xi
        c_y = (length(yi)+1)/2 # center of yi
        padzero!(workmat1, img, (pad_x, pad_x, pad_y, pad_y))
        rot_f90!(workmat2, workmat1, m)
        rotate_x!(workmat1, workmat2, tan_mod_theta, xi, yi, interp_x, workvec_x, c_y)
        rotate_y!(workmat2, workmat1, sin_mod_theta, xi, yi, interp_y, workvec_y, c_x)
        rotate_x!(workmat1, workmat2, tan_mod_theta, xi, yi, interp_x, workvec_x, c_y)
    end
    output .= (@view workmat1[pad_x + 1 : pad_x + M, pad_y + 1 : pad_y + N])
    return output
end


"""
    imrotate3_adj!(output, workmat1, workmat2, img, θ, interp_x, interp_y, workvec_x, workvec_y)
The adjoint of rotating an image by angle θ (must be ranging from 0 to 2π) in clockwise direction
using a series of 1d linear interpolation
"""
function imrotate3_adj!(
    output::AbstractMatrix{<:RealU},
    workmat1::AbstractMatrix{<:RealU},
    workmat2::AbstractMatrix{<:RealU},
    img::AbstractMatrix{<:RealU},
    θ::Real,
    interp_x::SparseInterpolator{<:AbstractFloat},
    interp_y::SparseInterpolator{<:AbstractFloat},
    workvec_x::AbstractVector{<:RealU},
    workvec_y::AbstractVector{<:RealU},
)

    @boundscheck size(output) == size(img) || throw("size")
    @boundscheck size(img, 1) == size(img, 2) || throw("row != col")

    if mod(θ, 2π) ≈ 0
        output .= img
        return output
    end
    m = mod(floor(Int, 0.5 + θ/(π/2)), 4)
    (M, N) = size(img)
    pad_x = Int((size(workmat1, 1) - M) / 2)
    pad_y = Int((size(workmat1, 2) - N) / 2)
    if θ ≈ m * (π/2)
        padzero!(workmat2, img, (pad_x, pad_x, pad_y, pad_y))
        rot_f90_adj!(workmat1, workmat2, m)
    else
        mod_theta = θ - m * (π/2) # make sure it is between -45 and 45 degree
        tan_mod_theta = tan(mod_theta / 2)
        sin_mod_theta = - sin(mod_theta)
        xi = 1 : M + 2 * pad_x
        yi = 1 : N + 2 * pad_y
        c_x = (length(xi)+1)/2
        c_y = (length(yi)+1)/2
        padzero!(workmat1, img, (pad_x, pad_x, pad_y, pad_y))
        rotate_x_adj!(workmat2, workmat1, tan_mod_theta, xi, yi, interp_x, workvec_x, c_y)
        rotate_y_adj!(workmat1, workmat2, sin_mod_theta, xi, yi, interp_y, workvec_y, c_x)
        rotate_x_adj!(workmat2, workmat1, tan_mod_theta, xi, yi, interp_x, workvec_x, c_y)
        rot_f90_adj!(workmat1, workmat2, m) # must be two different arguments
    end
    output .= (@view workmat1[pad_x + 1 : pad_x + M, pad_y + 1 : pad_y + N])
    return output
end


"""
    imrotate3!(output, workmat1, workmat2, img, θ)
Rotate an image by angle θ in clockwise direction using 2d linear interpolation
Source code is here: https://github.com/emmt/LinearInterpolators.jl
"""
function imrotate3!(
    output::AbstractMatrix{<:RealU},
    workmat1::AbstractMatrix{<:RealU},
    workmat2::AbstractMatrix{<:RealU},
    img::AbstractMatrix{<:RealU},
    θ::Real,
)

    @boundscheck size(output) == size(img) || throw("size")
    @boundscheck size(img, 1) == size(img, 2) || throw("row != col")

    if mod(θ, 2π) ≈ 0
        output .= img
        return output
    end
    (M, N) = size(img)
    pad_x = Int((size(workmat1, 1) - M) / 2)
    pad_y = Int((size(workmat1, 2) - N) / 2)
    T = eltype(img)
    ker = LinearSpline(T)
    rows = size(workmat2)
    cols = size(workmat1)
    c = ((1 + rows[1]) / 2, (1 + rows[2]) / 2)
    R = c + rotate(2π - θ, AffineTransform2D{T}() - c)
    A = TwoDimensionalTransformInterpolator(rows, cols, ker, R)
    padzero!(workmat1, img, (pad_x, pad_x, pad_y, pad_y))
    mul!(workmat2, A, workmat1)
    output .= (@view workmat2[pad_x + 1 : pad_x + M, pad_y + 1 : pad_y + N])
    return output
end


"""
    imrotate3_adj!(output, workmat1, workmat2, img, θ)
The adjoint of rotating an image by angle θ in clockwise direction using 2d linear interpolation
Source code is here: https://github.com/emmt/LinearInterpolators.jl
"""
function imrotate3_adj!(
    output::AbstractMatrix{<:RealU},
    workmat1::AbstractMatrix{<:RealU},
    workmat2::AbstractMatrix{<:RealU},
    img::AbstractMatrix{<:RealU},
    θ::Real,
)

    @boundscheck size(output) == size(img) || throw("size")
    @boundscheck size(img, 1) == size(img, 2) || throw("row != col")

    if mod(θ, 2π) ≈ 0
        output .= img
        return output
    end
    (M, N) = size(img)
    pad_x = Int((size(workmat1, 1) - M) / 2)
    pad_y = Int((size(workmat1, 2) - N) / 2)
    T = eltype(img)
    ker = LinearSpline(T)
    rows = size(workmat2)
    cols = size(workmat1)
    c = ((1 + rows[1]) / 2, (1 + rows[2]) / 2)
    R = c + rotate(2π - θ, AffineTransform2D{T}() - c)
    A = TwoDimensionalTransformInterpolator(rows, cols, ker, R)
    padzero!(workmat1, img, (pad_x, pad_x, pad_y, pad_y))
    mul!(workmat2, A', workmat1)
    output .= (@view workmat2[pad_x + 1 : pad_x + M, pad_y + 1 : pad_y + N])
    return output
end

