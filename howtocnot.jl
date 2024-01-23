#= What is the generator for a CNOT?

We'll need to apply CNOT's to match the random initialization used in Gibbs-ADAPT,
    so I'd like to apply a cis(???) with fixed angle.
I could also just write a function that manually CNOTS each basisvector,
    but I'd like to know the answer to the above question anyway.

Process:
1. Take the log of CNOT (we'll do both orderings, to ensure the Paulis simply reverse).
    This gives -iθ(??)/2. In analogy to Ry(π) giving the reflection Y, take θ=π.
    Multiply by 2i/π to get just ??, best analogy to Pauli.
    But since our evolution does exp(-ixG), the G we construct needs to have the 1/2,
        just as, when we hack in Ry(x) as exp(-ixG), that G needs to be Y/2.
2. Decompose ?? into Paulis. I anticipate it should be something pretty simple.
3. Construct a multi-qubit version of the CNOT operator,
    and check that it prepares the correct Bell state on the correct qubits...


=#


CNOT  = [1 0 0 0; 0 1 0 0; 0 0 0 1; 0 0 1 0]    # CONTROL ON FIRST QUBIT
CNOT_ = [1 0 0 0; 0 0 0 1; 0 0 1 0; 0 1 0 0]    # CONTROL ON SECOND QUBIT

# TAKE THE LOG AND DIVIDE OFF THE FLUFF
G = log(CNOT) * 2im/π

##########################################################################################
#= DECOMPOSE INTO PAULIS =#
using LinearAlgebra
# TODO: It occurs to me that the PauliOperators package should be able to do this...

paulistrings = ["I", "X", "Y", "Z"]
σ = [
    1   0;   0  1;;;
    0   1;   1  0;;;
    0 -im;  im  0;;;
    1   0;   0 -1;;;
]

σσ = Array{eltype(σ)}(undef, 4, 4, 4, 4)
for i in 1:4; for j in 1:4
    σσ[:,:,i,j] .= kron(σ[:,:,i], σ[:,:,j])
end; end

generatorpauliweights = Matrix{Float64}(undef, 4, 4)
for i in 1:4; for j in 1:4
    generatorpauliweights[i,j] = real(tr(σσ[:,:,i,j] * CNOT)) / 4
    # NOTE: tr is real as long as σσ and CNOT are both Hermitian.
end; end

using PauliOperators
generatorpauli = PauliSum(2)
for i in 1:4; for j in 1:4
    abs(generatorpauliweights[i,j]) ≤ 1e-8 && continue
    paulistring = paulistrings[i] * paulistrings[j]
    sum!(generatorpauli, generatorpauliweights[i,j] * Pauli(paulistring))
end; end

#= It seems to look a lot like "(1/2) * (I+Z)_ctl ⊗ (I+X)_tgt",
    except that the joint term ZX has a minus sign,
    rendering the sum unfactorizable (which it needs to be, in order to entangle).
    Neat! =#

##########################################################################################
#= CONSTRUCT A MULTI-QUBIT VERSION =#

# NOTE: We're putting back in a factor of 1/2 now, so an angle π does a full reflection.

# function CNOT_generator(n, ctl, tgt)
#     pauli = PauliSum(n)
#     sum!(pauli,  (1/2) * Pauli(n))
#     sum!(pauli,  (1/2) * Pauli(n; Z=ctl))
#     sum!(pauli,  (1/2) * Pauli(n; X=tgt))
#     sum!(pauli, (-1/2) * Pauli(n; Z=ctl, X=tgt))
#     return (1/2) * pauli
# end
#= NOTE: I seem to have not implemented evolution of a SparseKetBasis by a PauliSum;
    I vaguely recall that may have been intentional. :?
    Anyway, all these terms commute so evolving by a ScaledPauliVector works just as well.
    =#

function CNOT_generator(n, ctl, tgt)
    pauli = ScaledPauliVector(n)
    push!(pauli,  (1/2) * Pauli(n))
    push!(pauli,  (1/2) * Pauli(n; Z=ctl))
    push!(pauli,  (1/2) * Pauli(n; X=tgt))
    push!(pauli, (-1/2) * Pauli(n; Z=ctl, X=tgt))
    return (1/2) * pauli
end


#= Let's have 4 qubits,
        apply a Hadamard to qubits 3 and 4,
        and a CNOT, controlled by 3 and targeted on 1.
    That should produce the Bell state |00⟩+|11⟩ between qubits 1 and 3,
        while 2 and 4 factor out as |0⟩ and |0⟩+|1⟩.
    So, for an initial state
            1/2 (|0000⟩ + |0010⟩ + |0001⟩ + |0011⟩)
        evolving the CNOT generator by π should yield:
            1/2 (|0000⟩ + |1010⟩ + |0001⟩ + |1011⟩)
=#
using ADAPT
ψ0 = SparseKetBasis(4; T=ComplexF64)
ψ0[KetBitString([0,0,0,0])] = 1/2
ψ0[KetBitString([0,0,1,0])] = 1/2
ψ0[KetBitString([0,0,0,1])] = 1/2
ψ0[KetBitString([0,0,1,1])] = 1/2

println("Hadamard state, pre-CNOT:")
display(ψ0)
println()

G = CNOT_generator(4,3,1)
bell = ADAPT.evolve_state(G, π, ψ0)
clip!(bell; thresh=eps(Float64))

println("Bell state, post-CNOT:")
display(bell)
println()

bellvector = Vector(bell)
bellvector = ADAPT.evolve_state(G, π, Vector(ψ0))

println("Bell state when evolving dense vector:")
display(bellvector)
println()

println("Vector evolution matches sparse evolution?")
display(bellvector .- Vector(bell))
println()