#= This file is for implementing the RenyiDivergence observable type,
    compatible with DensityMatrix quantum states.
=#

import ADAPT

"""
    MaximalRenyiDivergence{F}(ρk, nH)

A cost function describing similarity with a target quantum state.

It is an upper bound on the Renyi divergence, which is trickier to calculate.

This code is restricted to order parameter α=2.

# Attributes
- `ρk`: the matrix inverse of a (mixed) quantum state `ρ` represented as a density matrix
- `nH`: the number of "hidden" qubits, ie. ancillae emulating the environment
- `nV`: the number of "visible" qubits, fixed by `ρk`

"""
struct MaximalRenyiDivergence{F<:AbstractFloat}
    ρk::Matrix{Complex{F}}
    nH::Int
    nV::Int

    function MaximalRenyiDivergence(ρk::AbstractMatrix, nH::Int)
        F = real(eltype(ρk))
        NV = size(ρk,1)             # SIZE OF HILBERT SPACE
        nV = round(Int, log2(NV))   # NUMBER OF QUBITS
        return new{real(eltype(ρk))}(convert(Matrix{Complex{F}}, ρk), nH, nV)
    end
end

ADAPT.typeof_energy(::MaximalRenyiDivergence{F}) where {F} = F

##########################################################################################

function ADAPT.evaluate(
    D::MaximalRenyiDivergence,
    σ::ADAPT.QuantumState,
)
    #= TODO:
        Calculate log(Tr(σ²ρ⁻¹))
        * Strictly speaking, if optimization is rewritten to use only gradient,
            this function may not be needed for the algorithm.
            But we'll certainly want it anyway.
    =#
    return 0.0
end

function ADAPT.partial(
    k::Int,
    ansatz::ADAPT.AbstractAnsatz,
    D::MaximalRenyiDivergence,
    σ0::ADAPT.QuantumState,
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

function ADAPT.calculate_score(
    ansatz::ADAPT.AbstractAnsatz,
    ::ADAPT.AdaptProtocol,
    G::ADAPT.Generator,
    D::MaximalRenyiDivergence,
    σ0::ADAPT.QuantumState,
)
    #= TODO:
        I haven't thought about how to implement this.

        At worst, we could:
        - deepcopy the ansatz
        - add G with parameter 0
        - call partial with index=length(new ansatz)

        I suppose that wouldn't even be that bad, resource-wise,
            though I could wish for a more elegant solution that doesn't involve deepcopy.

    =#
    return 0.0
end