# runtests.jl

using SPECTrecon
using Test: @test, @testset, detect_ambiguities

include("helper.jl")
include("rotate3.jl")
include("fft_convolve.jl")
include("project.jl")

@testset "SPECTrecon" begin
    @test isempty(detect_ambiguities(SPECTrecon))
end
