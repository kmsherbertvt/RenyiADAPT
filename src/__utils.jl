import ADAPT
import PauliOperators: PauliSum, ScaledPauliVector, Pauli, ScaledPauli, FixedPhasePauli
import PauliOperators: KetBitString, SparseKetBasis

import IterTools

function one_local_pool(n::Int64, axes=["I","X","Y","Z"])
    pool = ScaledPauliVector(n)
    for i in 1:n
        "X" in axes && (push!(pool, ScaledPauli(Pauli(n; X=i))))
        "Y" in axes && (push!(pool, ScaledPauli(Pauli(n; Y=i))))
        "Z" in axes && (push!(pool, ScaledPauli(Pauli(n; Z=i))))
    end
    return pool
end

function two_local_pool(n::Int64, axes=["I","X","Y","Z"])
    pool = ScaledPauliVector(n)
    for pair in IterTools.product(ntuple(i->1:n, 2)...)
        i,j = pair
        if i < j
            for pair2 in IterTools.product(ntuple(i->axes, 2)...)
                a,b = pair2
                if a == "I" || b == "I"   # NOTE: Use this to leave out one-local terms.
                # if a == b == "I"    # NOTE: Use this to include one-local terms.
                    continue
                end
                l = "I"^(i-1)*a*("I"^(j-i-1))*b*"I"^(n-j)
                pauli = ScaledPauli(Pauli(l))
                push!(pool, pauli)
            end
        end
    end
    return pool
end

function oneandtwo_local_pool(nV, nH)
    return vcat(
        one_local_pool(nV + nH),
        two_local_pool(nV + nH),
    )
end

function CNOT_generator(n, ctl, tgt)
    pauli = ScaledPauliVector(n)
    push!(pauli,  (1/2) * Pauli(n))
    push!(pauli,  (1/2) * Pauli(n; Z=ctl))
    push!(pauli,  (1/2) * Pauli(n; X=tgt))
    push!(pauli, (-1/2) * Pauli(n; Z=ctl, X=tgt))
    return (1/2) * pauli
end

num_qubits(state::SparseKetBasis{N,T}) where {N,T} = N
num_qubits(state::AbstractVector) = round(Int, log2(length(state)))

scale!(state::SparseKetBasis, c) = PauliOperators.scale!(state, c)
scale!(state::AbstractVector, c) = state .*= c

function apply_CNOT!(state, ctl, tgt)
    n = num_qubits(state)
    G = CNOT_generator(n, ctl, tgt)
    ADAPT.evolve_state!(G, π, state)
    scale!(state, im)                   # Correct the global phase.
end

function apply_Ry!(state, q, θ)
    n = num_qubits(state)
    G = (1/2) * Pauli(n, Y=q)
    ADAPT.evolve_state!(G, θ, state)
end






""" Generate H and calculate its thermal state, and the inversion thereof.

We have decided to test generic two-local Hamiltonians,
    so we assign a random number from N(μ=0,σ=1) to each 2-local Pauli word.

"""
function randomtwolocalhamiltonian(nV)
    # SAMPLE A STANDARD NORMAL VARIABLE FOR EACH 1- AND 2-LOCAL PAULI
    H = PauliSum(nV)
    for op in one_local_pool(nV)
        sum!(H, randn() * op)
    end
    for op in two_local_pool(nV)
        sum!(H, randn() * op)
    end

    # NORMALIZE SO THE ENERGY SCALE DOESN'T CHANGE WITH SYSTEM SIZE
    Q = sum(abs2, values(H.ops))
    foreach(key -> H[key] /= √Q, keys(H))

    # DO THE EXPENSIVE LINEAR ALGEBRA STUFF
    Hm = Matrix(H)              # NOTE: Not cheap!

    ρ = exp(-Hm)                # NOTE: Not cheap!
    ρ ./= LinearAlgebra.tr(ρ)

    ρk = exp(Hm)                # NOTE: Not cheap!
    ρk ./= LinearAlgebra.tr(ρk)

    ρs = sqrt(ρ)             # NOTE: Not cheap!

    return H, ρ, ρk, ρs
end

""" Generate ψREF.

Following the original Gibbs-ADAPT paper,
    we choose to prepare a random product state (using Ry rotation only),
    then CNOT each hidden unit with each visible unit.

"""
function randomentangledvector(nV, nH)
    # INITIALIZE A DENSE STATEVECTOR TO |0..0⟩
    ψREF = zeros(ComplexF64, 1<<(nV+nH)); ψREF[1] = 1

    # ROTATE BY AN ANGLE UNIFORMLY SAMPLED FROM [-π,π)
    for q in 1:(nV+nH)
        θ = (2*rand() - 1) * π
        apply_Ry!(ψREF, q, θ)
        #= NOTE: Up to a global phase,
            Ry(π/2) on |0⟩ is equivalent to a Hadamard, and
            Ry(π) on |0⟩ is equivalent to a Y gate. =#
    end

    # CNOT EACH HIDDEN UNIT WITH EACH VISIBLE UNIT
    for qH in nV+1:nV+nH
        for qV in 1:nV
            apply_CNOT!(ψREF, qH, qV)
        end
    end

    return ψREF
end




##########################################################################################
#= A new kind of ADAPT.Callback.
I'll probably add it to ADAPT.jl if it proves itself worthy here. =#

struct Serializer <: ADAPT.AbstractCallback
    ansatz_file::String
    trace_file::String
end
#= TODO: Is there any reason at all to permit serializing one file without the other? =#

function (serializer::Serializer)(
    ::ADAPT.Data, ansatz::ADAPT.AbstractAnsatz, trace::ADAPT.Trace,
    ::ADAPT.AdaptProtocol, ::ADAPT.GeneratorList, ::ADAPT.Observable, ::ADAPT.QuantumState,
)
    Serialization.serialize(serializer.ansatz_file, ansatz)
    Serialization.serialize(serializer.trace_file, trace)
    return false
end