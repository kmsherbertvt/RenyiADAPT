using Combinatorics: combinations

include("__utils.jl")

@testset "Basic DensityMatrix Properties" begin
    ρ = random_product_state(4)
    @test ρ isa DensityMatrix
    @test tr(ρ) ≈ 1
    @test purity(ρ) ≈ 1

    ρ = random_pure_state(4)
    @test ρ isa DensityMatrix
    @test tr(ρ) ≈ 1
    @test purity(ρ) ≈ 1

    ρ = random_mixed_state(4)
    @test ρ isa DensityMatrix
    @test tr(ρ) ≈ 1
    @test purity(ρ) < 1
end

@testset "Ptrace Product States" begin
    # Generate multiple product states from 2-4 qubits.
    for qubits in 2:4
        ρ = random_product_state(qubits)

        #Trace out a variable number of qubits, each one should be pure
        for nH in 1:qubits-1
            nV = qubits - nH
            rdm = partial_trace(ρ, nH)
            @test rdm isa DensityMatrix
            @test tr(rdm) ≈ 1

            @test size(rdm) == (2^nV, 2^nV)
            @test purity(rdm) ≈ 1
        end

        # Trace out 0 qubits, returning the original density matrix
        rdm = partial_trace(ρ, 0)
        @test ρ ≈ rdm
    end
end

@testset "Ptrace Product States (variable qubit indices)" begin
    for qubits in 2:4
        # Construct random product state, saving the individual states
        states = [random_ket(1) for _ in 1:qubits]
        ψ = tensor(states)
        ρ = ψ * ψ'

        for nV in 1:qubits
            for keep in combinations(1:qubits, nV)
                keep = sort(keep)
                rdm = partial_trace(ρ, keep)

                @test rdm isa DensityMatrix
                @test tr(rdm) ≈ 1
                @test size(rdm) == (2^nV, 2^nV)

                @test purity(rdm) ≈ 1

                # Compute the tensor product of the states
                # we wish to keep, and get that density matrix
                ψ = tensor(states[sort(keep)])
                exact_rdm = ψ * ψ'

                @test rdm ≈ exact_rdm
            end
        end
    end
end

@testset "Ptrace Pure States" begin
    # Generate multiple product states from 2-4 qubits.
    for qubits in 2:4
        ρ = random_pure_state(qubits)

        #Trace out a variable number of qubits, each one should be pure
        for nH in 1:qubits-1
            nV = qubits - nH
            rdm = partial_trace(ρ, nH)

            @test rdm isa DensityMatrix
            @test tr(rdm) ≈ 1
            @test size(rdm) == (2^nV, 2^nV)

            # In general, entangled states will result in mixed states after reduction
            @test purity(rdm) <= 1
        end
    end
end

@testset "Ptrace Pure States (variable qubit indices)" begin
    for qubits in 2:4
        ρ = random_pure_state(qubits)

        for nV in 1:qubits
            for keep in combinations(1:qubits, nV)
                rdm = partial_trace(ρ, keep)

                @test rdm isa DensityMatrix
                @test tr(rdm) ≈ 1
                @test size(rdm) == (2^nV, 2^nV)

                @test purity(rdm) <= 1
            end
        end
    end
end

@testset "Bell State" begin
    # Bell State |00> - |11>
    ψ = 1 / √(2) .* [1 + 0im, 0, 0, -1 + 0im]
    ρ = ψ * ψ'

    # We get the same result, regardless which qubit we trace out.
    # Should be the maximally mixed state
    rdm = partial_trace(ρ, [1])
    @test purity(rdm) ≈ 0.5
    @test rdm ≈ [0.5 0; 0 0.5]

    rdm = partial_trace(ρ, [2])
    @test purity(rdm) ≈ 0.5
    @test rdm ≈ [0.5 0; 0 0.5]
end

@testset "|+0>" begin
    # Product state |+> ⊗ |0>
    plus = 1 / √(2) .* [1 + 0im, 1 + 0im]
    zero = [1 + 0im, 0im]
    ψ = tensor([plus, zero])
    ρ = ψ * ψ'

    rdm = partial_trace(ρ, [1])
    @test rdm ≈ plus * plus'

    rdm = partial_trace(ρ, 1)
    @test rdm ≈ plus * plus'

    rdm = partial_trace(ρ, [2])
    @test rdm ≈ zero * zero'
end

@testset "|0+->" begin
    plus = 1 / √(2) .* [1 + 0im, 1 + 0im]
    zero = [1 + 0im, 0im]
    minus = 1 / √(2) .* [1 + 0im, -1 + 0im]

    # Product state |0> ⊗ |+> ⊗ |->
    ψ = tensor([zero, plus, minus])
    ρ = ψ * ψ'

    # Keep just the |+> subsystem
    rdm = partial_trace(ρ, [2])
    @test rdm ≈ plus * plus'

    # Keep the first and last qubit, |0> ⊗ |->
    rdm = partial_trace(ρ, [1, 3])
    ψ0 = tensor([zero, minus])
    ρ0 = ψ0 * ψ0'
    @test rdm ≈ ρ0

    # Keep first 2 qubits, yielding |0> ⊗ |+>, which we should also
    # get by eliminating the last hidden qubit
    rdm = partial_trace(ρ, [1, 2])
    ψ0 = tensor([zero, plus])
    ρ0 = ψ0 * ψ0'
    @test rdm ≈ ρ0
    @test rdm ≈ partial_trace(ρ, 1)
end
