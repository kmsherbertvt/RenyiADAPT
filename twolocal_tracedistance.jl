#= twolocal.jl, modified to use trace distance instead of Renyi divergence.

I do not recommend running this script;
    the gradient (or perhaps just the finite difference?) is unstable;
    linesearches take forever and the code takes an hour to complete a four-qubit run.

=#

import ADAPT
import RenyiADAPT
import PauliOperators: PauliSum, ScaledPauliVector, Pauli, ScaledPauli
import PauliOperators: KetBitString, SparseKetBasis

import LinearAlgebra
import Random; Random.seed!(1234)
import IterTools

##########################################################################################
#= HELPFUL FUNCTIONS =#

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

nV = 2
nH = 2
n = nV + nH

# BUILD OUT THE PROBLEM HAMILTONIAN
H = PauliSum(nV)
for op in one_local_pool(nV)
    sum!(H, randn() * op)   # For now, the coefficient is a random number from N(μ=0,σ=1).
end
for op in two_local_pool(nV)
    sum!(H, randn() * op)   # For now, the coefficient is a random number from N(μ=0,σ=1).
end
Hm = Matrix(H)              # NOTE: Not cheap!

# CALCULATE ρ ~= exp(-H) (normalized), just for comparison
ρ = exp(-Hm)                # NOTE: Not cheap!
ρ ./= LinearAlgebra.tr(ρ)

# PREPARE THE TRACE DISTANCE OBJECT
D = RenyiADAPT.TraceDistance(ρ, nH)
@assert nV == D.nV

##########################################################################################
#= PREPARE THE OPERATOR POOL =#
pool = vcat(
    one_local_pool(n),
    two_local_pool(n),
)   # Well that was easy. Thanks, Karunya!



##########################################################################################
#= CONSTRUCT THE REFERENCE STATE =#

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
for qH in nV+1:nV+nH
    for qV in 1:nV
        apply_CNOT!(ψREF, qH, qV)
    end
end


##########################################################################################
#= RUN THE ADAPT ALGORITHM =#

# SELECT THE PROTOCOLS
adapt = ADAPT.VANILLA
vqe = ADAPT.OptimOptimizer(:BFGS; g_tol=1e-6)
# vqe = RenyiADAPT.GradientDescent()    # TODO: switch to this if/when it matters.

# SELECT THE CALLBACKS
callbacks = [
    ADAPT.Callbacks.Tracer(
        :energy, :g_norm, :elapsed_f_calls, :elapsed_g_calls,
        :selected_index, :selected_score, :scores,
    ),
    ADAPT.Callbacks.ParameterTracer(),
    # ADAPT.Callbacks.Printer(:selected_score, :selected_generator),
    ADAPT.Callbacks.ScoreStopper(1e-3),
    ADAPT.Callbacks.ParameterStopper(length(pool)), # Don't exceed expense of naive way.
]

# INITIALIZE ANSATZ AND TRACE
ansatz = ADAPT.Ansatz(Float64, pool)
trace = ADAPT.Trace()

# # FOR TESTING: VALIDATE THAT ALL THESE ADAPT OBJECTS FIT TOGETHER CORRECTLY
# display(ADAPT.validate(
#     deepcopy(ansatz), adapt, vqe, pool[1:2], D, ψREF;
#     evaluation=nothing,     # Trace distance not exactly linear, no matrix cast.
#     gradient=1e-6,          # Finite difference, don't expect too much.
# ))

# RUN ADAPT
println("Running ADAPT...")
@time finished = ADAPT.run!(ansatz, trace, adapt, vqe, pool, D, ψREF, callbacks)
#= NOTE: `finished` is true iff ADAPT converged.
    (If not, the last optimization was probably too hard.
    It may be that it just needs more iterations,
        in which case simply calling run! again
        will let the optimization pick up from where it left off.
    But if the optimization needs so many iterations, something is probably wrong.) =#

ψEND = ADAPT.evolve_state(ansatz, ψREF)
σV = RenyiADAPT.partial_trace(ψEND * ψEND', nH)

println("ADAPT Finished? $finished")
println("Total iterations: $(last(trace[:iteration]))")
totalf = sum(trace[:elapsed_f_calls][trace[:adaptation][2:end]])
finished || (totalf += last(trace[:elapsed_f_calls]))
println("Total f calls: $totalf")
println()

##########################################################################################
#= RUN A VQE SIMPLY USING EVERY POSSIBLE POOL OPERATOR EXACTLY ONCE =#

# INITIALIZE THE ANSATZ
vqe_ansatz = ADAPT.Ansatz(Float64, pool)
ADAPT.set_optimized!(vqe_ansatz, false)
    # Empty ADAPT ansatze are implicitly optimized, but we'll be adding items manually.
for op in pool
    push!(vqe_ansatz, op => 0.0)    # Reference is randomized, so no need to perturb?
end

# RUN THE OPTIMIZATION ALGORITHM
vqe_trace = ADAPT.Trace()
println("Running VQE with all operators...")
@time result = ADAPT.optimize!(vqe_ansatz, vqe_trace, vqe, D, ψREF, callbacks)
# NOTE: result contains the `Optim.OptimizationResult` from BFGS

vqe_ψEND = ADAPT.evolve_state(vqe_ansatz, ψREF)
vqe_σV = RenyiADAPT.partial_trace(vqe_ψEND * vqe_ψEND', nH)

println("VQE Converged? $(result.g_converged)")
println("Total iterations: $(last(vqe_trace[:iteration]))")
println("Total f calls: $(last(vqe_trace[:elapsed_f_calls]))")
println()


##########################################################################################
#= CONTRAST RESULTS =#

# Construct a couple more states for comparison purposes.
referencestate = RenyiADAPT.partial_trace(ψREF * ψREF', nH)
maximallymixedstate = Matrix{ComplexF64}(LinearAlgebra.I, 1<<nV, 1<<nV) ./ (1<<nV)

purity = RenyiADAPT.purity
entropy = RenyiADAPT.von_neumann_entropy
tracedistance(ρ,σ) = LinearAlgebra.tr(abs.(ρ .- σ)) / 2
fidelity(ρ,σ) = (ρ12=sqrt(ρ); real(LinearAlgebra.tr(sqrt(ρ12*σ*ρ12))))


labels = [
    "ADAPT σV ($(length(ansatz)) ops)",
    "  VQE σV ($(length(vqe_ansatz)) ops)",
    "From Reference\t",
    "I/N \t\t",
    "exp(-H)",
]
println("  Metric"*"\t"*join(labels, "\t"))
println("-"^150)
states = [σV, vqe_σV, referencestate, maximallymixedstate, ρ]
println("  Purity"*"\t"*join((purity(σ) for σ in states), "\t"))
println(" Entropy"*"\t"*join((entropy(σ) for σ in states), "\t"))
println("Distance"*"\t"*join((tracedistance(ρ,σ) for σ in states), "\t"))
println("Fidelity"*"\t"*join((fidelity(ρ,σ) for σ in states), "\t"))
println()


#= TODO

This cost function appears to be somehow unstable,
    especially perhaps on operators acting only on hidden nodes?
Linesearches take forever.

=#