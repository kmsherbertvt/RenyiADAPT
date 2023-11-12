#= Just a rough sketch of Renyi ADAPT on thermal states. =#

import ADAPT
import RenyiADAPT
import PauliOperators: ScaledPauliVector, FixedPhasePauli, KetBitString, SparseKetBasis


# BUILD OUT THE PROBLEM HAMILTONIAN
H = [1 0; 0 1]
# TODO: Input a real H somehow.

# CALCULATE ρk ~= exp(H) (normalized))
ρk = exp(H)
ρk ./= trace(ρk)

# PREPARE THE RENYI DIVERGENCE OBJECT
nH = 3
D = RenyiADAPT.MaximalRenyiDivergence(ρk, nH)
nV = D.nV

# BUILD OUT THE OPERATOR POOL
pool = ScaledPauliVector{2L}[]
for i in   1:2L
for j in i+1:2L
    push!(pool, ADAPT.Operators.qubitexcitation(2L, i, j))

    for k in j+1:2L
    for l in k+1:2L
        push!(pool, ADAPT.Operators.qubitexcitation(2L, i, j, k, l))
    end; end
end; end
# TODO: Use a hopefully-non-arbitrary pool.

# CONSTRUCT A REFERENCE STATE
N = 1 << nV << nH               # TOTAL HILBERT SPACE
ρ0 = RenyiADAPT.DensityMatrix{Float64}(I, N, N)
# TODO: Use a better reference state than the maximally mixed..?

# INITIALIZE THE ANSATZ AND TRACE
ansatz = ADAPT.Ansatz(Float64, pool)
trace = ADAPT.Trace()

# SELECT THE PROTOCOLS
adapt = ADAPT.VANILLA
vqe = ADAPT.OptimOptimizer(:BFGS; g_tol=1e-6)

# SELECT THE CALLBACKS
callbacks = [
    ADAPT.Callbacks.Tracer(:energy, :selected_index, :selected_score, :scores),
    ADAPT.Callbacks.Printer(:energy, :selected_generator, :selected_score),
    ADAPT.Callbacks.ScoreStopper(1e-3),
    ADAPT.Callbacks.ParameterStopper(1),
]

# RUN THE ALGORITHM
ADAPT.run!(ansatz, trace, adapt, vqe, pool, H, ψ0, callbacks)