#= How does Renyi divergence compare to more traditional distance measures?

The hypothesis is that, as an unbounded function, it is less prone to barren plateaus.
To show that, we need to show a *bounded* cost-function which is minimized at the same state
    (ie. perfect overlap with a target state) *does* run into problems with barren plateaus.
Uh, I don't know what *bounded* means, exactly,
    but let's just see what happens with the traditional distance measure,

    Tr |rho - sigma| / 2

=#


import LinearAlgebra: tr

import ADAPT
import ADAPT: AbstractAnsatz, AdaptProtocol, Generator, GeneratorList, QuantumState

import .partial_trace, .ispure

Optional{T} = Union{T, Nothing}

struct TraceDistance{F <: AbstractFloat}
    ρ::DensityMatrix{F}
    nH::Int
    nV::Int

    function TraceDistance(ρ::AbstractMatrix, nH::Int)
        F = real(eltype(ρ))
        nV = trunc(Int, log2(size(ρ, 1)))
        return new{F}(convert(DensityMatrix{F}, ρ), nH, nV)
    end
end

ADAPT.typeof_energy(::TraceDistance{F}) where {F} = F

"""Returns whether or not the training model σ(θ) is pure or mixed"""
ispure(D::TraceDistance, σ₀::QuantumState) = (D.nH == 0) && ispure(σ₀)

function ADAPT.evaluate(
    D::TraceDistance,
    ψ::QuantumState,
)
    σ = ψ * ψ'
    σV = partial_trace(σ, D.nH)
    C = D.ρ .- σV
    A = sqrt(C' * C)
    return real(tr( A )) / 2
end
