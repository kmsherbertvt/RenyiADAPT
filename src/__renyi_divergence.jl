#= This file is for implementing the RenyiDivergence observable type,
    compatible with DensityMatrix quantum states.
=#

import LinearAlgebra: rank, tr, inv, pinv

import ADAPT
import ADAPT: AbstractAnsatz, AdaptProtocol, GeneratorList, QuantumState
using TimerOutputs: @timeit

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

        TODO: Jim implemented the pure and mixed state versions as though
                it was a pure or mixed state on just the visible nodes,
                but in this function, the quantum state should be understood as the ansatz.
            That includes visible and hidden nodes,
                meaning we need to trace out the hidden nodes
                when calculating the actual divergence.
            This version with a pure state
                (now understood as a pure state over visible and hidden nodes,
                presumably a mixed state after tracing out the hidden nodes)
                is now corrected, but due to laziness I'm delegating to Jim's version for mixed states.
            Properly speaking, that version should be renamed to a different, private method,
                since the ADAPT.evaluate method is explicitly for ansatze,
                rather than intermediate calculations.

    =#
    σ = ψ * ψ'
    σV = partial_trace(σ, D.nH)
        #= TODO: partial_trace is defined in __density_matrices.
        I'd like it explicitly identified in this file, eg. by courtesy import. =#
    return ADAPT.evaluate(D, σV)
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
    return realifclose(log(tr(σ^2 * D.ρk)))
    #= Following Jim's example, try to return all real values if possible.
        If there is an error and we get imaginary divergences, this will propagate
        downstream. =#
    #= TODO: As noted above, this implementation assumes σ is defined over visible nodes;
        the function should really take the whole ansatz,
        defined over visible and hidden nodes.
    We do need this implementation (in fact I use it above),
        but it should have a different function name probably.
    =#
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

# Fixme: commutator(x, y) fails if one is a Pauli because they lack
# right multiplication support. Workaround by casting to matrix. Didn't
# implement for anticommutator because it always comes after a call to commutator
commutator(x, y) = x*y - y*x
commutator(x, y::AnyPauli) = commutator(x, Matrix(y))
commutator(x::AnyPauli, y) = commutator(Matrix(x), y)
anticommutator(x, y) = x*y + y*x

function ADAPT.calculate_scores(
    ansatz::AbstractAnsatz,
    protocol::AdaptProtocol,
    pool::GeneratorList,
    D::MaximalRenyiDivergence,
    ψ₀::QuantumState,
)
    #=
        If our reference state is a ket, take the outer
        product of |ψ₀>. Then dispatch to our normal function.
        Because the bulk of the computation will be in the generator
        multiplication, we don't need to care about evolving the state
        first by Uk.
    =#
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
    @timeit "Pool Gradient" grad = begin

        # Construct unitary for our circuit
        U = Matrix(size(σ₀, 1), ansatz)

        # Evolve the reference state, and calculate the partial trace to give our
        # visible model σv. Also calculate the denominator of the Renyi divergence
        σ = U * σ₀ * U'
        σv = partial_trace(σ, D.nH)
        scale = -im / tr(σv^2 * D.ρk)

        # Define a closure function that computes the Renyi divergence for a
        # pool operator given the quantities we precomputed above.
        pool_renyi_div = Hj -> commutator(Hj, σ) |>
                            x -> partial_trace(x, D.nH) |>
                            x -> anticommutator(x, σv) |>
                            x -> tr(x * D.ρk)

        # Todo: Possibly parallelize this?
        # grad = pool_renyi_div.(pool)
        grad = zeros(ComplexF64, length(pool))
        for (k, Hj) in enumerate(pool)
            @timeit "Per operator" begin
                grad[k] = pool_renyi_div(Hj)
            end
        end

        #=
            Scale the gradient and try to return all real values if possible.
            If there is an error and we get imaginary gradients, this will propagate
            downstream.
        =#
        realifclose.(scale .* grad)
    end

    return grad
end

function ADAPT.gradient!(
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

function ADAPT.gradient!(
    grad::AbstractVector,
    ansatz::AbstractAnsatz,
    D::MaximalRenyiDivergence,
    σ₀::DensityMatrix
)
    Uk = Matrix(size(σ₀, 1), ansatz)
    σ = Uk * σ₀ * Uk'
    return gradient!(grad, ansatz, D, σ, Uk)
end

function right_evolve_unitary!(G, θ, U)
    # Fixme: Not sure if there's a better way to do this

    # UA = (A†U†)†, A† = e^{-i(-θ)G}
    ADAPT.evolve_unitary!(G, -θ, U')
    U = U'
    return U
end

#= NOTE: Jim wrote the following auxiliary function.
    Near as I can tell,
        it correctly implements the gradient expression defined in the paper,
        but it does not match one produced by a finite difference.
    I re-derived my own expression for the gradient,
        and implemented that below, and it does match.
    I *THINK* the expression in the paper is actually incomplete,
        and maybe I even have an idea of how the mistake was made,
        but of course the code behind the paper worked so I'm not 100% certain. =#
function broken_gradient!(
    grad::AbstractVector,
    ansatz::AbstractAnsatz,
    D::MaximalRenyiDivergence,
    σ::DensityMatrix,
    Uk::Matrix
)
    σv = partial_trace(σ, D.nH)
    scale = -im / tr(σv^2 * D.ρk)

    renyi_div = Hk -> commutator(Hk, σ) |>
                      x -> partial_trace(x, D.nH) |>
                      x -> anticommutator(x, σv) |>
                      x -> tr(x * D.ρk)

    #=
        Because Hk̃ = U_{k+1} Hk U_{k+1}†, we iterate in forward order.
        Performing the operation
            U_{k+1} = U_k e^{+iθk Hk},
        where
            U_k = ∏_{j = k → 1} e^{-iθj Hj},
        allows us to reuse as many intermediate products as possible.
    =#
    for k in eachindex(ansatz)
        Gk, θk = ansatz[k]

        Uk = right_evolve_unitary!(Gk, -θk, Uk)

        #=
            Fixme: (PauliOperators)
            Base.* is not defined for matrix * pauli
            so we group the multiplication of Gk * Uk first which is defined.
        =#
        Hk = Uk' * (Matrix(Gk) * Uk)
            #= TODO: I don't think Gk * Uk *is* defined?
                Brute-force casting to matrix for now,
                    but I would need to think a bit about what the "right" solution is.
            =#
        grad[k] = realifclose(scale * renyi_div(Hk))
    end

    return grad
end


#= TODO: This re-write hopefully fixes the broken gradient for our two-local pool,
        but I'm pretty sure a more complicated fix is needed for generic pool operators
        which may involve sums of non-commuting terms.
    The ADAPT.jl library has such a fix already implemented for simple observables;
        we need only adapt it if we ever choose to use such a pool. =#
function gradient!(
    grad::AbstractVector,
    ansatz::AbstractAnsatz,
    D::MaximalRenyiDivergence,
    σ::DensityMatrix,               # expectation: the totally-evolved σ on the whole space.
    Uk::Matrix                      # expectation: the unitary for the whole ansatz.
)
    @timeit "Ansatz gradient" begin
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
            @timeit "Per Parameter" begin
                Gk, θk = ansatz[k]

                ################################################
                # CONSTRUCT ∂k_σv AND FILL IN GRADIENT

                Hk = Matrix(Gk) # Pauli*Matrix not implemented, so cast Pauli to matrix for now
                commuted = commutator(Hk, σk)
                conjugated = Uk * commuted * Uk'
                ∂k_σv = partial_trace(conjugated, D.nH)

                grad[k] = realifclose(scale * renyi_div(∂k_σv))

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
        end
    end

    return grad
end