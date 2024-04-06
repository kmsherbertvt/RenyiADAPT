#= How does Renyi divergence compare to more traditional distance measures?

The hypothesis is that, as an unbounded function, it is less prone to barren plateaus.
To show that, we need to show a *bounded* cost-function which is minimized at the same state
    (ie. perfect overlap with a target state) *does* run into problems with barren plateaus.
Uh, I don't know what *bounded* means, exactly,
    but let's just see what happens with the traditional distance measure,

    Tr (sqrt(rho) sigma sqrt(rho))

=#


import LinearAlgebra: tr, pinv, kron, transpose

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
    # return 1 - abs2(tr( sqrt(C) ))
    return 1 - abs2(tr( sqrt(C) ))
end

# import FiniteDifferences
# function ADAPT.gradient!(
#     result::AbstractVector,
#     ansatz::AbstractAnsatz,
#     observable::Infidelity,
#     reference::QuantumState,
# )
#     # TODO: We are fiddling with the order, to try and get higher accuracies,
#     #   to get a feel for infidelity without needing the analytical gradient.
#     cfd = FiniteDifferences.central_fdm(10, 1)
#     x0 = copy(ADAPT.angles(ansatz))
#     fn = ADAPT.make_costfunction(ansatz, observable, reference)
#     result .= FiniteDifferences.grad(cfd, fn, x0)[1]
#     return result
# end

function ADAPT.gradient!(
    result::AbstractVector,
    ansatz::AbstractAnsatz,
    D::Infidelity,
    ψREF::QuantumState,
)
    Uk = Matrix(size(ψREF, 1), ansatz)
    ψ = Uk * ψREF
    σ = ψ * ψ'
    σv = partial_trace(σ, D.nH)

    C = D.sqrtρ * σv * D.sqrtρ
    Cs = sqrt(C)
    Cks = inv(Cs)       # Handle sparse states robustly?

    #= Basically copying this from what I did for Renyi gradient.
        Not necessarily the most efficient? But it works. =#
    σk = Uk' * σ * Uk

    for k in eachindex(ansatz)
        Gk, θk = ansatz[k]

        ################################################
        # CONSTRUCT ∂k_σv AND FILL IN GRADIENT

        Hk = Matrix(Gk) # Pauli*Matrix not implemented, so cast Pauli to matrix for now
        commuted = commutator(σk, Hk)
        conjugated = Uk * commuted * Uk'
        ∂k_σv = im .* partial_trace(conjugated, D.nH)
        ∂k_C = D.sqrtρ * ∂k_σv * D.sqrtρ

        ∂k_Cs = (1/2) .* (Cks * ∂k_C)
        result[k] = real( -tr(∂k_Cs) * tr(Cs') - tr(Cs) * tr(∂k_Cs') )


        ################################################
        # MOVE THE exp(-iθG) FOR THIS STEP FROM Uk TO σk

        #= TODO: The evolution could happen on a statevector initialized to reference;
                    we'd have to re-convert it to a density matrix after each step.
                For now, the following is a lazy (but even more expensive) way
                    to evolve the density matrix. =#
        Ek = exp(-im*θk*Hk)
        σk = Ek * σk * Ek'

        Uk = Uk * Ek'
        #= TODO: At a glance the right_evolve_unitary! function looks like it should work,
            but I get the wrong answer if I replace the above line with

            Uk = right_evolve_unitary!(Gk, -θk, Uk)

            :(

            Not taking the time to debug right now;
                the problem could easily be in ether Jim's hack or my evolution.

             =#
    end

    return result
end



function modelpregradient!(
    grad::AbstractVector,
    ansatz::AbstractAnsatz,
    D::MaximalRenyiDivergence,
    ψ₀::QuantumState,
)
    # Do faster evolution with pure state, then calculate the gradient
    Uk = Matrix(size(ψ₀, 1), ansatz)
    ψ = Uk * ψ₀
    σ = ψ * ψ'
    return gradient!(grad, ansatz, D, σ, Uk)
end

function modelgradient!(
    grad::AbstractVector,
    ansatz::AbstractAnsatz,
    D::MaximalRenyiDivergence,
    σ::DensityMatrix,               # expectation: the totally-evolved σ on the whole space.
    Uk::Matrix                      # expectation: the unitary for the whole ansatz.
)
    σv = partial_trace(σ, D.nH)
    scale = -im / tr(σv^2 * D.ρk)   # -im comes from constant in ∂k_σv, divisor from ∂ log

    renyi_div = ∂k_σv -> tr(anticommutator(∂k_σv, σv) * D.ρk)

    #= I'd like to partially evolve σ rather than H,
        and I want to start with the reference state.
        But I don't want to change anything Jim did except in this method,
            and that means I can't put the reference state in the arguments.
        So, uh, I'm going to get σREF through the silliest of ways. =#
    σk = Uk' * σ * Uk

    for k in eachindex(ansatz)
        Gk, θk = ansatz[k]

        ################################################
        # CONSTRUCT ∂k_σv AND FILL IN GRADIENT

        Hk = Matrix(Gk) # Pauli*Matrix not implemented, so cast Pauli to matrix for now
        commuted = commutator(Hk, σk)
        conjugated = Uk * commuted * Uk'
        ∂k_σv = partial_trace(conjugated, D.nH)

        grad[k] = realifclose(scale * renyi_div(∂k_σv))
        grad[k] = real(scale * renyi_div(∂k_σv))

        ################################################
        # MOVE THE exp(-iθG) FOR THIS STEP FROM Uk TO σk

        #= TODO: The evolution could happen on a statevector initialized to reference;
                    we'd have to re-convert it to a density matrix after each step.
                For now, the following is a lazy (but even more expensive) way
                    to evolve the density matrix. =#
        Ek = exp(-im*θk*Hk)
        σk = Ek * σk * Ek'

        Uk = Uk * Ek'
        #= TODO: At a glance the right_evolve_unitary! function looks like it should work,
            but I get the wrong answer if I replace the above line with

            Uk = right_evolve_unitary!(Gk, -θk, Uk)

            :(

            Not taking the time to debug right now;
                the problem could easily be in ether Jim's hack or my evolution.

             =#
    end

    return grad
end