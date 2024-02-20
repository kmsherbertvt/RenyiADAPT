#= How does Renyi divergence compare to more traditional distance measures?

The hypothesis is that, as an unbounded function, it is less prone to barren plateaus.
To show that, we need to show a *bounded* cost-function which is minimized at the same state
    (ie. perfect overlap with a target state) *does* run into problems with barren plateaus.
Uh, I don't know what *bounded* means, exactly,
    but let's just see what happens with the traditional distance measure,

    Tr (sqrt(rho) sigma sqrt(rho))

=#


import LinearAlgebra: tr

import ADAPT
import ADAPT: AbstractAnsatz, AdaptProtocol, Generator, GeneratorList, QuantumState

import .partial_trace, .ispure

Optional{T} = Union{T, Nothing}

struct Infidelity{F <: AbstractFloat}
    sqrtρ::DensityMatrix{F}
    nH::Int
    nV::Int

    function Infidelity(sqrtρ::AbstractMatrix, nH::Int)
        F = real(eltype(sqrtρ))
        nV = trunc(Int, log2(size(sqrtρ, 1)))
        return new{F}(convert(DensityMatrix{F}, sqrtρ), nH, nV)
    end
end

function Infidelity(
    ρ::Optional{DensityMatrix} = nothing;
    sqrtρ::Optional{DensityMatrix} = nothing,
    nH::Int,
)
    # Assert that the user supplied only one of ρ, ρk
    xor(isnothing(ρ), isnothing(sqrtρ)) || throw(ArgumentError("Only one of ρ, sqrtρ may be supplied"))

    #=
        If the user supplies just ρ, we calculate sqrtρ.
    =#
    if isnothing(sqrtρ)
        sqrtρ = sqrt(ρ)
    end

    return MaximalRenyiDivergence(sqrtρ, nH)
end

ADAPT.typeof_energy(::Infidelity{F}) where {F} = F

"""Returns whether or not the training model σ(θ) is pure or mixed"""
ispure(D::Infidelity, σ₀::QuantumState) = (D.nH == 0) && ispure(σ₀)

function ADAPT.evaluate(
    D::Infidelity,
    ψ::QuantumState,
)
    σ = ψ * ψ'
    σV = partial_trace(σ, D.nH)

    C = D.sqrtρ * σV * D.sqrtρ
    return 1 - abs2(tr( sqrt(C) ))
end



##########################################################################
#= Super lazy, super slow.

These should actually pry be defaults in ADAPT.jl... =#

function ADAPT.partial(
    k::Int,
    ansatz::AbstractAnsatz,
    D::Infidelity,
    ψ₀::QuantumState,
)
    return ADAPT.gradient(ansatz, D, ψ₀)[k]
end

function ADAPT.calculate_score(
    ansatz::AbstractAnsatz,
    protocol::AdaptProtocol,
    G::Generator,
    D::Infidelity,
    ψ₀::QuantumState,
)
    L = length(ansatz)
    candidate = deepcopy(ansatz)
    push!(candidate, G => zero(ADAPT.typeof_parameter(ansatz)))
    return abs(ADAPT.partial(L+1, candidate, D, ψ₀))
end


import FiniteDifferences

function ADAPT.gradient!(
    grad::AbstractVector,
    ansatz::AbstractAnsatz,
    D::Infidelity,
    ψ₀::QuantumState,
)
    cfd = FiniteDifferences.central_fdm(5, 1)
    x0 = copy(ADAPT.angles(ansatz))
    fn = ADAPT.Basics.make_costfunction(ansatz, D, ψ₀)
    grad .= FiniteDifferences.grad(cfd, fn, x0)[1]
    return grad
end

