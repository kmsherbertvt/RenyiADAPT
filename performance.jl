#= Generate an arbitrary Hamiltonian with 1- and 2-local terms,
    then let Renyi-ADAPT prepare the thermal state using a 1- and 2-local pool.

We will use a reference of |0..0⟩ rotated into a random entangling state,
    by applying Ry(random angles) to each qubit,
    followed by CNOTs between each hidden::visible pair.

=#

import ADAPT
import RenyiADAPT
import PauliOperators: PauliSum, ScaledPauliVector, Pauli, ScaledPauli
import PauliOperators: KetBitString, SparseKetBasis

import LinearAlgebra
import Random; Random.seed!(1234)
import IterTools
import TimerOutputs

##########################################################################################
#= HELPFUL FUNCTIONS =#

fidelity(ρ,σ) = (ρ12=sqrt(ρ); real(LinearAlgebra.tr(sqrt(ρ12*σ*ρ12))))

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


##########################################################################################
#= DESIGNATE THE COST-FUNCTION =#

function createLossFunction(nV, nH)
    n = nV + nH

    # BUILD OUT THE PROBLEM HAMILTONIAN
    H = PauliSum(nV)
    for op in one_local_pool(nV)
        sum!(H, randn() * op)   # For now, the coefficient is a random number from N(μ=0,σ=1).
    end
    for op in two_local_pool(nV)
        sum!(H, randn() * op)   # For now, the coefficient is a random number from N(μ=0,σ=1).
    end
    # NORMALIZE SO THE ENERGY SCALE DOESN'T CHANGE WITH SYSTEM SIZE
    Q = sum(abs2, values(H.ops))
    foreach(key -> H[key] /= √Q, keys(H))
    Hm = Matrix(H)              # NOTE: Not cheap!

    # CALCULATE ρ ~= exp(-H) (normalized), just for comparison
    ρ = exp(-Hm)                # NOTE: Not cheap!
    ρ ./= LinearAlgebra.tr(ρ)

    # CALCULATE ρk ~= exp(H) (normalized), for the actual algorithm
    ρk = exp(Hm)                # NOTE: Not cheap!
    ρk ./= LinearAlgebra.tr(ρk)

    # PREPARE THE RENYI DIVERGENCE OBJECT
    D = RenyiADAPT.MaximalRenyiDivergence(ρk, nH)
    @assert nV == D.nV

    return D, ρ
end

##########################################################################################
#= PREPARE THE OPERATOR POOL =#

createPool(n) = vcat(
    one_local_pool(n),
    two_local_pool(n),
)   # Well that was easy. Thanks, Karunya!



##########################################################################################
#= CONSTRUCT THE REFERENCE STATE =#

function createReferenceState(nV, nH)
    n = nV + nH
    # # INITIALIZE A SPARSE STATEVECTOR TO |0..0⟩
    # ψREF = SparseKetBasis(n; T=ComplexF64); ψREF[KetBitString{n}(0)] = 1
    # NOTE: Ry's on each qubit means the sparse statevector gains us nothing. ;)

    # INITIALIZE A DENSE STATEVECTOR TO |0..0⟩
    ψREF = zeros(ComplexF64, 1<<n); ψREF[1] = 1     # This gives the statevector with |0..0⟩.

    # EVOLVE BY RANDOM ANGLES
    for q in 1:n
        θ = (2*rand() - 1) * π      # For now, the angle is uniformly drawn from [-π,π)
                                    #= NOTE: Up to a global phase,
                                        Ry(π/2) on |0⟩ is equivalent to a Hadamard, and
                                        Ry(π) on |0⟩ is equivalent to a Y gate. =#
        apply_Ry!(ψREF, q, θ)
    end

    # CNOT EACH HIDDEN UNIT WITH EACH VISIBLE UNIT
    for qH in nV+1:n
        for qV in 1:nV
            apply_CNOT!(ψREF, qH, qV)
        end
    end

    return ψREF
end

##########################################################################################
#= RUN THE ADAPT ALGORITHM =#

function run_adapt(nV, nH; output=true)
    output && println("Creating loss function and hamiltonian")
    D, ρ = createLossFunction(nV, nH)
    output && println("Creating reference state")
    ψREF = createReferenceState(nV, nH)
    output && println("Creating operator pool")
    pool = createPool(nV + nH)

    # SELECT THE PROTOCOLS
    adapt = ADAPT.VANILLA
    vqe = ADAPT.OptimOptimizer(:BFGS; g_tol=1e-6)

    # SELECT THE CALLBACKS
    callbacks = [
        ADAPT.Callbacks.Tracer(:energy, :selected_index, :selected_score, :scores, :g_norm),
        ADAPT.Callbacks.ParameterTracer(),
        ADAPT.Callbacks.ScoreStopper(1e-3),
        ADAPT.Callbacks.ParameterStopper(length(pool)), # Don't exceed expense of naive way.
    ]

    if output
        cb = ADAPT.Callbacks.Printer(:energy, :selected_generator, :selected_score, :g_norm)
        push!(callbacks, cb)
    end

    # RUN THE ADAPT ALGORITHM
    ansatz = ADAPT.Ansatz(Float64, pool)
    trace = ADAPT.Trace()

    output && println("Running ADAPT")
    TimerOutputs.reset_timer!()
    TimerOutputs.@timeit "RenyiADAPT" finished = ADAPT.run!(ansatz, trace, adapt, vqe, pool, D, ψREF, callbacks)

    to = TimerOutputs.get_defaulttimer()
    #= NOTE: `finished` is true iff ADAPT converged.
        (If not, the last optimization was probably too hard.
        It may be that it just needs more iterations,
            in which case simply calling run! again
            will let the optimization pick up from where it left off.
        But if the optimization needs so many iterations, something is probably wrong.) =#

    ψEND = ADAPT.evolve_state(ansatz, ψREF)
    σV = RenyiADAPT.partial_trace(ψEND * ψEND', nH)
    f = fidelity(ρ, σV)

    return to, f, finished
end

function profile_adapt(nV, nH; output=true)
    # Get timer and final fidelity
    to, fidelity_, finished = run_adapt(nV, nH, output=output)

    if output
        # Save results to text file
        open("perf-$(nV).txt", "w") do file
            # Hyperparameters
            println(file, "Visible Qubits: ", nV)
            println(file, "Hidden Qubits: ", nH)
            println(file, "Total Qubits: ", nV + nH)

            # Results
            println(file, "Fidelity: ", fidelity_)
            println(file, "Finished?: ", finished)
            println(file)

            # Performance
            println(file, to)
        end
    end
end

function main()
    # Usage: $ julia performance.jl nV
    @assert length(ARGS) == 1
    nV = nH = parse(Int, ARGS[1])

    # Precompile the package
    profile_adapt(1, 1, output=false)
    profile_adapt(nV, nH)
end

main()
