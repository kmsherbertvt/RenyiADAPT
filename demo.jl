#= Just a rough sketch of Renyi ADAPT on thermal states. =#

import ADAPT
import RenyiADAPT
import PauliOperators: ScaledPauliVector, Pauli, ScaledPauli
import PauliOperators: KetBitString, SparseKetBasis

import LinearAlgebra


# BUILD OUT THE PROBLEM HAMILTONIAN
H = [1 1; 1 -1]
#= TODO: Input a real H somehow.
Most likely, the most convenient choices for our purposes are:
1. Generate a random Hermitian matrix, or
2. Generate a molecular Hamiltonian with pyscf / openfermion,
    save it to a file (eg. with `NPZ`), then load it here.
=#

# CALCULATE ρk ~= exp(H) (normalized))
ρk = exp(H)
ρk ./= LinearAlgebra.tr(ρk)

# PREPARE THE RENYI DIVERGENCE OBJECT
nH = 3
D = RenyiADAPT.MaximalRenyiDivergence(ρk, nH)
nV = D.nV
n = nH + nV

# BUILD OUT THE OPERATOR POOL
pool = ScaledPauliVector{n}[]
for i in   1:n
for j in i+1:n
    push!(pool, ScaledPauli{n}[
        (1/2) * Pauli(n, X=i, Z=j),
        (1/2) * Pauli(n, X=j, Z=i),
    ])
    push!(pool, ScaledPauli{n}[
        (1/2) * Pauli(n, Y=i, Z=j),
        (1/2) * Pauli(n, Y=j, Z=i),
    ])

    for k in j+1:n
    for l in k+1:n
        push!(pool, ScaledPauli{n}[
            (1/8) * Pauli(n, X=[i,j,k], Z=l),
            (1/8) * Pauli(n, X=[i,j,l], Z=k),
            (1/8) * Pauli(n, X=[i,k,l], Z=j),
            (1/8) * Pauli(n, X=[j,k,l], Z=i),
            (1/8) * Pauli(n, X=i, Z=[j,k,l]),
            (1/8) * Pauli(n, X=j, Z=[i,k,l]),
            (1/8) * Pauli(n, X=k, Z=[i,j,l]),
            (1/8) * Pauli(n, X=l, Z=[i,j,k]),
        ])
        push!(pool, ScaledPauli{n}[
            (1/8) * Pauli(n, Y=[i,j,k], Z=l),
            (1/8) * Pauli(n, Y=[i,j,l], Z=k),
            (1/8) * Pauli(n, Y=[i,k,l], Z=j),
            (1/8) * Pauli(n, Y=[j,k,l], Z=i),
            (1/8) * Pauli(n, Y=i, Z=[j,k,l]),
            (1/8) * Pauli(n, Y=j, Z=[i,k,l]),
            (1/8) * Pauli(n, Y=k, Z=[i,j,l]),
            (1/8) * Pauli(n, Y=l, Z=[i,j,k]),
        ])
    end; end
end; end
# TODO: Use a hopefully-non-arbitrary pool.

# CONSTRUCT A REFERENCE STATE
#= In this example, assume H is represented in Hartree-Fock basis,
    so that mean-field solution is |1..10..0⟩.
=#
Ne = nV ÷ 2     # NUMBER OF ELECTRONS
    #= TODO: This is a placeholder.
            nV/2 is the number for common demos like STO-3G H chains and Hubbard models,
                but generally this needs to be input manually or extracted from pyscf. =#
HF_ketstring = "1"^Ne * "0"^(n-Ne)
HF_ket = KetBitString(n, parse(Int, HF_ketstring; base=2))
N = 1 << nV << nH               # TOTAL HILBERT SPACE
ψ0 = zeros(ComplexF64, N); ψ0[1+HF_ket.v] = 1

# INITIALIZE THE ANSATZ AND TRACE
ansatz = ADAPT.Ansatz(Float64, pool)
trace = ADAPT.Trace()

# SELECT THE PROTOCOLS
adapt = ADAPT.VANILLA
vqe = ADAPT.OptimOptimizer(:BFGS; g_tol=1e-6)
# vqe = RenyiADAPT.GradientDescent()    # TODO: switch to this if/when it matters.

# SELECT THE CALLBACKS
callbacks = [
    ADAPT.Callbacks.Tracer(:energy, :selected_index, :selected_score, :scores),
    ADAPT.Callbacks.Printer(:energy, :selected_generator, :selected_score),
    ADAPT.Callbacks.ScoreStopper(1e-3),
    ADAPT.Callbacks.ParameterStopper(1),
]

# RUN THE ALGORITHM
ADAPT.run!(ansatz, trace, adapt, vqe, pool, D, ψ0, callbacks)