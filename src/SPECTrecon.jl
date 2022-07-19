"""
    SPECTrecon
System matrix (forward and back-projector) for SPECT image reconstruction.
"""
module SPECTrecon

    const RealU = Number # Union{Real, Unitful.Length}

    include("helper.jl")
    include("rotate3.jl")
    include("fft_convolve.jl")
    include("spectplan.jl")
    include("project.jl")
    include("backproject.jl")

end # module
