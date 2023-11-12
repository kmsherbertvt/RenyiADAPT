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

    # TODO: nV is determined by ρk, so an internal constructor is appropriate.
end

ADAPT.typeof_energy(::RenyiDivergence{F}) where {F} = F

function MaximalRenyiDivergence(ρ::DensityMatrix)
    #= TODO:
        Calculate ρk = inv(ρ).

        Wait what do you do if ρ is not full rank?
        I guess pseudo-inverse makes sense for density matrices
            but does that affect the interpretation of the equations???

        Anyway, I don't imagine we use this for thermal states,
            since it is easier to construct ρk ~= exp(H) directly.
    =#
end

#= TODO: We could consider a convenience constructor taking Hamiltonian,
    which calculates ρk ~= exp(-H) then calls internal constructor.

    But then we'd have to specify a standard form for the Hamiltonian,
        and there's no need for that. The script can handle it, I say.
=#

##########################################################################################

function ADAPT.evaluate(
    D::MaximalRenyiDivergence,
    σ::QuantumState,
)
    #= TODO:
        Calculate log(Tr(σ²ρ⁻¹))
    =#
end

function ADAPT.partial(
    k::Int,
    ansatz::AbstractAnsatz,
    D::MaximalRenyiDivergence,
    σ0::DensityMatrix,
)
    #= TODO:
        Evaluate σ = U σ0 U' by evolving σ0 with ansatz
        Construct Hₖ
        Evaluate Eqn 8 from Kieferova 2021.
    =#
end

function ADAPT.calculate_score(
    ansatz::AbstractAnsatz,
    ::AdaptProtocol,
    G::Generator,
    D::MaximalRenyiDivergence,
    σ0::DensityMatrix,
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
end