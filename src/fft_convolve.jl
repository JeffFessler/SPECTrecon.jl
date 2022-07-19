# fft_convolve.jl

export fft_conv!, fft_conv_adj!

import AbstractFFTs
import FFTW
using FFTW: plan_fft!, plan_ifft!


const AbsMatComplex = AbstractMatrix{<:Complex{T}} where T <: AbstractFloat # Real -> AbstractFloat

"""
    imfilter3!(output, img_compl, ker, ker_compl, fft_plan, ifft_plan)
FFT-based convolution between `img_compl` and kernel `ker` (not centered)
putting result in `output`.
"""
function imfilter3!(
    output::AbstractMatrix{<:RealU},
    img_compl::AbsMatComplex,
    ker::AbstractMatrix{<:RealU},
    ker_compl::AbsMatComplex,
    fft_plan::Union{AbstractFFTs.ScaledPlan, FFTW.cFFTWPlan},
    ifft_plan::Union{AbstractFFTs.ScaledPlan, FFTW.cFFTWPlan},
)

    # pad the kernel with zeros
    pad2sizezero!(ker_compl, ker, size(img_compl))
    mul!(img_compl, fft_plan, img_compl)
    mul!(ker_compl, fft_plan, ker_compl)
    img_compl .*= ker_compl 
    mul!(img_compl, ifft_plan, img_compl)
    fftshift2!(ker_compl, img_compl)
    output .= real.(ker_compl)
    return output
end


"""
    imfilter3_adj!(output, img_compl, kerev, ker_compl, fft_plan, ifft_plan)
Apply FFT convolution between `img_compl` and *REVERSED* kernel (not centered),
assuming the kernel is already be in reversed order.
"""
function imfilter3_adj!(
    output::AbstractMatrix{<:RealU},
    img_compl::AbsMatComplex,
    kerev::AbstractMatrix{<:RealU}, # input kernel should already be in reversed order
    ker_compl::AbsMatComplex,
    fft_plan::Union{AbstractFFTs.ScaledPlan, FFTW.cFFTWPlan},
    ifft_plan::Union{AbstractFFTs.ScaledPlan, FFTW.cFFTWPlan},
)

    # pad kernel with zeros
    pad2sizezero!(ker_compl, kerev, size(img_compl))
    mul!(img_compl, fft_plan, img_compl)
    mul!(ker_compl, fft_plan, ker_compl)
    img_compl .*= ker_compl
    mul!(img_compl, ifft_plan, img_compl)
    fftshift2!(ker_compl, img_compl)
    output .= real.(ker_compl)
    return output
end


"""
    fft_conv!(output, workmat, img, ker, fftpadsize, img_compl, ker_compl, fft_plan, ifft_plan)
Convolve `img` with `ker` using FFT, and store the result in `output`
"""
function fft_conv!(
    output::AbstractMatrix{<:RealU},
    workmat::AbstractMatrix{<:RealU},
    img::AbstractMatrix{<:RealU},
    ker::AbstractMatrix{<:RealU},
    fftpadsize::NTuple{4, <:Int},
    img_compl::AbsMatComplex,
    ker_compl::AbsMatComplex,
    fft_plan::Union{AbstractFFTs.ScaledPlan, FFTW.cFFTWPlan},
    ifft_plan::Union{AbstractFFTs.ScaledPlan, FFTW.cFFTWPlan},
)
    @boundscheck size(output) == size(img) || throw("size")
    # filter the image with a kernel, using replicate padding and fft convolution
    padrepl!(img_compl, img, fftpadsize)
    imfilter3!(workmat, img_compl, ker, ker_compl, fft_plan, ifft_plan)
    (M, N) = size(img)
    copyto!(output, (@view workmat[fftpadsize[1]+1:fftpadsize[1]+M,
                                   fftpadsize[3]+1:fftpadsize[3]+N]))
    return output
end


"""
    fft_conv_adj!(output, workmat, workvec1, workvec2, img, ker,
                  fftpadsize, img_compl, ker_compl, fft_plan, ifft_plan)
"""
function fft_conv_adj!(
    output::AbstractMatrix{<:RealU},
    workmat::AbstractMatrix{<:RealU},
    workvec1::AbstractVector{T},
    workvec2::AbstractVector{T},
    img::AbstractMatrix{<:RealU},
    ker::AbstractMatrix{<:RealU},
    fftpadsize::NTuple{4, <:Int},
    img_compl::AbsMatComplex,
    ker_compl::AbsMatComplex,
    fft_plan::Union{AbstractFFTs.ScaledPlan, FFTW.cFFTWPlan},
    ifft_plan::Union{AbstractFFTs.ScaledPlan, FFTW.cFFTWPlan},
) where {T}

    @boundscheck size(output) == size(img) || throw("size")
    padzero!(img_compl, img, fftpadsize)
    imfilter3_adj!(workmat, img_compl, ker, ker_compl, fft_plan, ifft_plan)
    (M, N) = size(img)
    # adjoint of replicate padding
    workvec1 .= zero(T)
    for i = 1:fftpadsize[1]
        plus2di!(workvec1, workmat, i)
    end
    plus1di!(workmat, workvec1, 1+fftpadsize[1])

    workvec1 .= zero(T)
    for i = fftpadsize[1]+M+1:size(workmat, 1)
        plus2di!(workvec1, workmat, i)
    end
    plus1di!(workmat, workvec1, M+fftpadsize[1])
    workvec2 .= zero(T)
    for j = 1:fftpadsize[3]
        plus2dj!(workvec2, workmat, j)
    end
    plus1dj!(workmat, workvec2, 1+fftpadsize[3])

    workvec2 .= zero(T)
    for j = fftpadsize[3]+N+1:size(workmat, 2)
        plus2dj!(workvec2, workmat, j)
    end
    plus1dj!(workmat, workvec2, N+fftpadsize[3])

    copyto!(output, (@view workmat[fftpadsize[1]+1:fftpadsize[1]+M,
                                   fftpadsize[3]+1:fftpadsize[3]+N]))
    return output
end
