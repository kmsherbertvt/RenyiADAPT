#= This file is for implementing the RenyiDivergence observable type,
    compatible with DensityMatrix quantum states.
=#

import LinearAlgebra: rank, tr, inv, pinv

import ADAPT
import ADAPT: AbstractAnsatz, AdaptProtocol, GeneratorList, QuantumState

Optional{T} = Union{T, Nothing}

"""
    MaximalRenyiDivergence{F}(ρk, nH)

A cost function describing similarity with a target quantum state.

It is an upper bound on the Renyi divergence, which is trickier to calculate.

This code is restricted to order parameter α=2.

# Attributes
- `ρk`: the matrix inverse of a (mixed) quantum state `ρ` represented as a density matrix.
- `nH`: the number of "hidden" qubits, ie. ancillae emulating the environment
- `nV`: the number of "visible" qubits, fixed by `ρk`

"""
struct MaximalRenyiDivergence{F <: AbstractFloat}
    ρk::DensityMatrix{F}
    nH::Int
    nV::Int

    function MaximalRenyiDivergence(ρk::AbstractMatrix, nH::Int)
        F = real(eltype(ρk))
        nV = trunc(Int, log2(size(ρk, 1)))
        return new{F}(convert(DensityMatrix{F}, ρk), nH, nV)
    end
end

ADAPT.typeof_energy(::MaximalRenyiDivergence{F}) where {F} = F

function MaximalRenyiDivergence(
    ρ::Optional{DensityMatrix} = nothing;
    ρk::Optional{DensityMatrix} = nothing,
    nH::Int,
)
    # Assert that the user supplied only one of ρ, ρk
    xor(isnothing(ρ), isnothing(ρk)) || throw(ArgumentError("Only one of ρ, ρk may be supplied"))

    #=
        If the user supplies just ρ, we calculate ρk by inverting ρ.
        For full-rank ρ, we can use LA.inv(). Otherwise, we need to compute
        a pseudo-inverse.
    =#
    if isnothing(ρk)
        full_rank = rank(ρ) == size(ρ, 1)
        ρk = (full_rank) ? inv(ρ) : pinv(ρ)
    end

    return MaximalRenyiDivergence(ρk, nH)
end

"""Returns whether or not the training model σ(θ) is pure or mixed"""
ispure(D::MaximalRenyiDivergence, σ₀::QuantumState) = (D.nH == 0) && ispure(σ₀)


"""
    evaluate(D::MaximalRenyiDivergence, ψ::QuantumState)

Calculates the Renyi divergence for a pure state ψ.

# Parameters
- `D`: The Renyi divergence object
- `ψ`: The state to evaluate
"""
function ADAPT.evaluate(
    D::MaximalRenyiDivergence,
    ψ::QuantumState,
)
    #= 
        Compute log(Tr(σ²̢ρ⁻¹))
        For pure states, ρ^2 = ρ, and the trace becomes
        an expectation value
            log(<ψ|ρ⁻¹|ψ>)
    =#
    return log(ψ' * D.ρk * ψ)
end

"""
    evaluate(D::MaximalRenyiDivergence, ψ::QuantumState)

Calculates the Renyi divergence for a potentially mixed state ρ.

# Parameters
- `D`: The Renyi divergence object
- `ρ`: The state to evaluate
"""
function ADAPT.evaluate(
    D::MaximalRenyiDivergence,
    σ::DensityMatrix
)
    # Calculate log(Tr(σ²ρ⁻¹))
    return log(tr(σ^2 * D.ρk))
end

function ADAPT.partial(
    k::Int,
    ansatz::AbstractAnsatz,
    D::MaximalRenyiDivergence,
    ψ₀::QuantumState,
)
    #= TODO:
        Evaluate σ = U σ0 U' by evolving σ0 with ansatz
        * As Jim pointed out, σ can just be a statevector on visible and hidden nodes...
        Construct Hₖ
        Evaluate Eqn 8 from Kieferova 2021.

        * As Carlos pointed out, this inevitably involves computing σV,
            which should certainly be its own function.
        * As Jim pointed out, there should be a `gradient` function
                computing all partials simultaneously,
                enabling shared calculations such as σV.
            Kyle thought he already had that, but it looks like he did not.
            He will make it in short order.
    =#
    return 0.0
end

function ADAPT.partial(
    k::Int,
    ansatz::AbstractAnsatz,
    D::MaximalRenyiDivergence,
    σ₀::DensityMatrix,
)
    # Todo: This requires density matrix evolution of the adapt ansatz
    NotImplementedError(_density_matrix_evolution_error)
end

commutator(x, y) = x*y - y*x
anticommutator(x, y) = x*y + y*x

function ADAPT.calculate_scores(
    ansatz::AbstractAnsatz,
    protocol::AdaptProtocol,
    pool::GeneratorList,
    D::MaximalRenyiDivergence,
    ψ₀::QuantumState,
)
    # If our reference state is a ket, take the outer
    # product of |ψ₀>. Then dispatch to our normal function
    σ₀ = ψ₀ * ψ₀'

    return ADAPT.calculate_scores(ansatz, protocol, pool, D, σ₀)
end

function ADAPT.calculate_scores(
    ansatz::AbstractAnsatz,
    ::AdaptProtocol,
    pool::GeneratorList,
    D::MaximalRenyiDivergence,
    σ₀::DensityMatrix,
)
    # Construct unitary for our circuit
    U = ADAPT.to_unitary(ansatz)
    Ud = U'

    # Evolve the reference state, and calculate the partial trace to give our
    # visible model σv. Also calculate the denominator of the Renyi divergence
    σ = U * σ₀ * Ud
    σv = partial_trace(σ, D.nH)
    scale = -im / tr(σv^2 * D.ρk)

    # Define a closure function that computes the Renyi divergence for a
    # pool operator given the quantities we precomputed above.
    pool_renyi_div = Hj -> U * Hj * Ud |>
                           x -> commutator(x, σ) |>
                           x -> partial_trace(x, D.nH) |>
                           x -> anticommutator(x, σv) |>
                           x -> tr(x * D.ρk)

    # Todo: Possibly parallelize this?
    grad = pool_renyi_div.(pool)

    #=
        Scale the gradient and try to return all real values if possible.
        If there is an error and we get imaginary gradients, this will propagate
        downstream.
    =#
    return realifclose.(scale .* grad)
end

function gradient(
    ansatz::AbstractAnsatz,
    ::AdaptProtocol,
    D::MaximalRenyiDivergence,
    ψ₀::QuantumState,
)
    #=
        Fixme: Need implementation in ADAPT.jl
        Could probably just implement evolve_unitary!(U, G, θ), then reuse that
        to implement to_unitary. Something like..

        function to_unitary(ansatz::AbstractAnsatz)
            U = I
            for (G, θ) in ansatz
                evolve_unitary!(U, G, θ)
            end
            return U
        end
    =#
    # The unitary implementing our circuit, ∏_{j = N → 1} e^{-iθj Gj}
    Uk = ADAPT.to_unitary(ansatz)

    #=
        Evolve our state by reusing the unitary, then taking the outer product.
        Because matrix-vector multiplication is O(n^2), while matrix-matrix is O(n^3),
        first evolving then doing the outer product is faster than constructing σ₀ and
        computing Uσ₀U†.
    =#
    ψ = Uk * ψ₀
    σ = ψ * ψ'
    σv = partial_trace(σ, D.nH)
    scale = -im / tr(σv^2 * D.ρk)

    renyi_div = Hk -> commutator(Hk, σ) |>
                      x -> partial_trace(x, D.nH) |>
                      x -> anticommutator(x, σv) |>
                      x -> tr(x * D.ρk)

    #=
        Because Hk̃ = U_{k-1}† Hk U_{k-1}, we iterate in reverse order.
        Performing the operation
            U_{k-1} = e^{+iθk Hk} U_k,
        where
            U_k = ∏_{j = k → 1} e^{-iθj Hj},
        allows us to reuse as many intermediate products as possible.
    =#
    grad = zeros(ComplexF64, length(ansatz))
    for k in reverse(eachindex(ansatz))
        Gk, θk = ansatz[k]

        # Fixme: This needs a function in ADAPT.jl
        evolve_unitary!(Uk, Gk, -θk)

        Hk = Uk' * Gk * Uk
        grad[k] = renyi_div(Hk)
    end

    return realifclose.(scale .* grad)
end

function gradient(
    ansatz::AbstractAnsatz,
    ::AdaptProtocol,
    D::MaximalRenyiDivergence,
    σ₀::DensityMatrix,
)
    # Todo: density matrix evolution
    NotImplementedError(_density_matrix_evolution_error)
end
