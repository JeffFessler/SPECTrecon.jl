# helper.jl
# A lot of helper functions

using LinearAlgebra
import OffsetArrays
using ImageFiltering: BorderArray, Fill, Pad
using FFTW


Power2 = x -> 2^(ceil(Int, log2(x)))
_padup(mumap, psfs) = ceil(Int, (Power2(size(mumap, 1) + size(psfs, 1) - 1) - size(mumap, 1)) / 2)
_paddown(mumap, psfs) = floor(Int, (Power2(size(mumap, 1) + size(psfs, 1) - 1) - size(mumap, 1)) / 2)
_padleft(mumap, psfs) = ceil(Int, (Power2(size(mumap, 3) + size(psfs, 2) - 1) - size(mumap, 3)) / 2)
_padright(mumap, psfs) = floor(Int, (Power2(size(mumap, 3) + size(psfs, 2) - 1) - size(mumap, 3)) / 2)


"""
    padzero!(output, img, pad_x, pad_y)
Mutating version of padding a 2D image by filling zeros.
Output has size `(size(img, 1) + padsize[1] + padsize[2], size(img, 2) + padsize[3] + padsize[4])`.
"""
Base.@propagate_inbounds function padzero!(
    output::AbstractMatrix{T},
    img::AbstractMatrix,
    padsize::NTuple{4, <:Int}, # up, down, left, right
) where {T}

    @boundscheck size(output) ==
        size(img) .+ (padsize[1] + padsize[2], padsize[3] + padsize[4]) || throw("size")
    M, N = size(img)
    output .= zero(T)
    for j = padsize[3] + 1:padsize[3] + N, i = padsize[1] + 1:padsize[1] + M
        @inbounds output[i, j] = img[i - padsize[1], j - padsize[3]]
    end
    return output
end



"""
    padrepl!(output, img, padsize)
Pad with replication from `img` into `output`
"""
Base.@propagate_inbounds function padrepl!(
    output::AbstractMatrix,
    img::AbstractMatrix,
    padsize::NTuple{4, <:Int}, # up, down, left, right
)

    @boundscheck size(output) ==
        size(img) .+ (padsize[1] + padsize[2], padsize[3] + padsize[4]) || throw("size")
    M, N = size(img)
    for j = 1:size(output, 2), i = 1:size(output, 1)
        @inbounds output[i, j] = img[clamp(i - padsize[1], 1, M), clamp(j - padsize[3], 1, N)]
    end
    return output
end



"""
    pad2sizezero!(output, img, padsize)

Non-allocating version of padding:
`output[pad_dims[1]+1 : pad_dims[1]+dims[1],
        pad_dims[2]+1 : pad_dims[2]+dims[2]]) .= img
"""
Base.@propagate_inbounds function pad2sizezero!(
    output::AbstractMatrix{T},
    img::AbstractMatrix,
    padsize::Tuple{<:Int, <:Int},
) where {T}
    @boundscheck size(output) == padsize || throw("size")

    dims = size(img)
    pad_dims = ceil.(Int, (padsize .- dims) ./ 2)
    output .= zero(T)
    for j = pad_dims[2]+1:pad_dims[2]+dims[2], i = pad_dims[1]+1:pad_dims[1]+dims[1]
        @inbounds output[i, j] = img[i - pad_dims[1], j - pad_dims[2]]
    end
    return output
end


"""
    pad_it!(X, padsize)
Zero-pad `X` to `padsize`
"""
function pad_it!(X::AbstractArray{T,D}, padsize::NTuple{D,<:Int}) where {D, T <: Number}
    dims = size(X)
    return OffsetArrays.no_offset_view(
        BorderArray(X,
            Fill(zero(T),
               (ceil.(Int, (padsize .- dims) ./ 2)),
               (floor.(Int, (padsize .- dims) ./ 2)),
            )
        )
    )
end



fftshift!(dst::AbstractArray, src::AbstractArray) = circshift!(dst, src, size(src) .÷ 2)

ifftshift!(dst::AbstractArray, src::AbstractArray) = circshift!(dst, src, size(src) .÷ -2)



"""
    fftshift2!(dst, src)
Same as `fftshift` in 2d, but non-allocating
"""
Base.@propagate_inbounds function fftshift2!(
    dst::AbstractMatrix,
    src::AbstractMatrix,
)
    @boundscheck (iseven(size(src, 1)) && iseven(size(src, 2))) || throw("odd")
    @boundscheck size(src) == size(dst) || throw("size")
    m, n = div.(size(src), 2)
    for j = 1:n, i = 1:m
        @inbounds dst[i, j] = src[m+i, n+j]
    end
    for j = n+1:2n, i = 1:m
        @inbounds dst[i, j] = src[m+i, j-n]
    end
    for j = 1:n, i = m+1:2m
        @inbounds dst[i, j] = src[i-m, j+n]
    end
    for j = n+1:2n, i = m+1:2m
        @inbounds dst[i, j] = src[i-m, j-n]
    end
    return dst
end



"""
    plus1di!(mat2d, mat1d)
Non-allocating `mat2d[i, :] += mat1d`
"""
Base.@propagate_inbounds function plus1di!(
    mat2d::AbstractMatrix,
    mat1d::AbstractVector,
    i::Int,
)
    @boundscheck (size(mat1d, 1) == size(mat2d, 2) || throw("size2"))
    @boundscheck (1 ≤ i ≤ size(mat2d, 1) || throw("bad i"))
    for m in 1:size(mat2d, 2)
        @inbounds mat2d[i, m] += mat1d[m]
    end
    return mat2d
end



"""
    plus1dj!(mat2d, mat1d)
Non-allocating `mat2d[:, j] += mat1d`
"""
Base.@propagate_inbounds function plus1dj!(
    mat2d::AbstractMatrix,
    mat1d::AbstractVector,
    j::Int,
)
    @boundscheck (size(mat1d, 1) == size(mat2d, 1) || throw("size1"))
    @boundscheck (1 ≤ j ≤ size(mat2d, 2) || throw("bad j"))
    for n in 1:size(mat2d, 1)
        @inbounds mat2d[n, j] += mat1d[n]
    end
    return mat2d
end


"""
    plus2di!(mat1d, mat2d, i)
Non-allocating `mat1d += mat2d[i,:]`
"""
Base.@propagate_inbounds function plus2di!(
    mat1d::AbstractVector,
    mat2d::AbstractMatrix,
    i::Int,
)
    @boundscheck (size(mat1d, 1) == size(mat2d, 2) || throw("size2"))
    @boundscheck (1 ≤ i ≤ size(mat2d, 1) || throw("bad i"))
    for m in 1:size(mat2d, 2)
        @inbounds mat1d[m] += mat2d[i, m]
    end
    return mat1d
end


"""
    plus2dj!(mat1d, mat2d, j)
Non-allocating `mat1d += mat2d[:,j]`
"""
Base.@propagate_inbounds function plus2dj!(
    mat1d::AbstractVector,
    mat2d::AbstractMatrix,
    j::Int,
)
    @boundscheck (size(mat1d, 1) == size(mat2d, 1) || throw("size1"))
    @boundscheck (1 ≤ j ≤ size(mat2d, 2) || throw("bad j"))
    for n in 1:size(mat2d, 1)
        @inbounds mat1d[n] += mat2d[n, j]
    end
    return mat1d
end


"""
    plus3di!(mat2d, mat3d, i)
Non-allocating `mat2d += mat3d[i,:,:]`
"""
Base.@propagate_inbounds function plus3di!(
    mat2d::AbstractMatrix,
    mat3d::AbstractArray{<:Any, 3},
    i::Int,
)
    @boundscheck (size(mat2d, 1) == size(mat3d, 2) || throw("size2"))
    @boundscheck (size(mat2d, 2) == size(mat3d, 3) || throw("size3"))
    @boundscheck (1 ≤ i ≤ size(mat3d, 1) || throw("bad i"))
    for n in 1:size(mat2d, 2), m in 1:size(mat2d, 1)
        @inbounds mat2d[m, n] += mat3d[i, m, n]
    end
    return mat2d
end


"""
    plus3dj!(mat2d, mat3d, j)
Non-allocating `mat2d += mat3d[:,j,:]`
"""
Base.@propagate_inbounds function plus3dj!(
    mat2d::AbstractMatrix,
    mat3d::AbstractArray{<:Any, 3},
    j::Int,
)
    @boundscheck (size(mat2d, 1) == size(mat3d, 1) || throw("size1"))
    @boundscheck (size(mat2d, 2) == size(mat3d, 3) || throw("size3"))
    @boundscheck (1 ≤ j ≤ size(mat3d, 2) || throw("bad j"))
    for n in 1:size(mat2d, 2), m in 1:size(mat2d, 1)
        @inbounds mat2d[m, n] += mat3d[m, j, n]
    end
    return mat2d
end


"""
    plus3dk!(mat2d, mat3d, k)
Non-allocating `mat2d += mat3d[:,:,k]`
"""
Base.@propagate_inbounds function plus3dk!(
    mat2d::AbstractMatrix,
    mat3d::AbstractArray{<:Any, 3},
    k::Int,
)
    @boundscheck (size(mat2d, 1) == size(mat3d, 1) || throw("size1"))
    @boundscheck (size(mat2d, 2) == size(mat3d, 2) || throw("size2"))
    @boundscheck (1 ≤ k ≤ size(mat3d, 3) || throw("bad k"))
    for n in 1:size(mat2d, 2), m in 1:size(mat2d, 1)
        @inbounds mat2d[m, n] += mat3d[m, n, k]
    end
    return mat2d
end


"""
    scale3dj!(mat2d, mat3d, j, s)
Non-allocating `mat2d = s * mat3d[:,j,:]`
"""
Base.@propagate_inbounds function scale3dj!(
    mat2d::AbstractMatrix,
    mat3d::AbstractArray{<:Any, 3},
    j::Int,
    s::RealU,
)
    @boundscheck (size(mat2d, 1) == size(mat3d, 1) || throw("size1"))
    @boundscheck (size(mat2d, 2) == size(mat3d, 3) || throw("size3"))
    @boundscheck (1 ≤ j ≤ size(mat3d, 2) || throw("bad j"))
    for n in 1:size(mat2d, 2), m in 1:size(mat2d, 1)
        @inbounds mat2d[m, n] = s * mat3d[m, j, n]
    end
    return mat2d
end


"""
    mul3dj!(mat3d, mat2d, j)
Non-allocating `mat3d[:,j,:] *= mat2d`
"""
Base.@propagate_inbounds function mul3dj!(
    mat3d::AbstractArray{<:Any, 3},
    mat2d::AbstractMatrix,
    j::Int,
)
    @boundscheck (size(mat3d, 1) == size(mat2d, 1) || throw("size1"))
    @boundscheck (size(mat3d, 3) == size(mat2d, 2) || throw("size3"))
    @boundscheck (1 ≤ j ≤ size(mat3d, 2) || throw("bad j"))
    for n in 1:size(mat3d, 3), m in 1:size(mat3d, 1)
        @inbounds mat3d[m, j, n] *= mat2d[m, n]
    end
    return mat3d
end


"""
    copy3dj!(mat2d, mat3d, j)
Non-allocating `mat2d .= mat3d[:,j,:]`
"""
Base.@propagate_inbounds function copy3dj!(
    mat2d::AbstractMatrix,
    mat3d::AbstractArray{<:Any, 3},
    j::Int,
)
    @boundscheck (size(mat3d, 1) == size(mat2d, 1) || throw("size1"))
    @boundscheck (size(mat3d, 3) == size(mat2d, 2) || throw("size3"))
    @boundscheck (1 ≤ j ≤ size(mat3d, 2) || throw("bad j"))
    for n in 1:size(mat3d, 3), m in 1:size(mat3d, 1)
        @inbounds mat2d[m,n] = mat3d[m, j, n]
    end
    return mat2d
end

