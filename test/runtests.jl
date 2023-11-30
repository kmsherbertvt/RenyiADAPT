using LinearAlgebra
using RenyiADAPT
using Test

#= 
    Add your test file to this list. Your file should have name
    test_<yourname>.jl in this directory. Just add the <yourname>
    part to this loop and this will include it.
=#
for filename in (
    "density_matrix",
    "renyi_divergence",
)
    name = replace(filename, "_" => " ") |> titlecase
    @testset "$name" begin
        include("test_$filename.jl")
    end
end
