# runtests.jl

using SPECTrecon
using Test: @test, @testset, detect_ambiguities

include("helper.jl")
include("rotatez.jl")
include("fftconv.jl")
include("psf-gauss.jl")
include("adjoint-rotate.jl")
include("adjoint-project.jl")
include("ml-os-em.jl")

@testset "SPECTrecon" begin
    @test isempty(detect_ambiguities(SPECTrecon))
end
