using RenyiADAPT
import RenyiADAPT: ADAPT

include("__utils.jl")

allzero(v::Vector{R}) where R <: Real = isapprox.(v, 0, atol=1e-9) |> all
somenonzero(v::Vector{R}) where R <: Real = !allzero(v)
allnonzero(v::Vector{R}) where R <: Real = .!isapprox.(v, 0, atol=1e-9) |> all

@testset "Create RenyiDivergence" begin
    # Maximally mixed state, is full rank
    ρ = 1 / 16 .* Matrix{ComplexF64}(I, 16, 16)
    D = MaximalRenyiDivergence(ρ, nH=1)

    @test D isa MaximalRenyiDivergence
    @test D.nH == 1
    @test D.nV == 4
    @test ρ != D.ρk

    # Random state, likely is not full rank
    ρ = random_pure_state(4)
    full_rank = rank(ρ) == size(ρ, 1)
    D = MaximalRenyiDivergence(ρ, nH=1)

    @test D isa MaximalRenyiDivergence
    @test D.nH == 1
    @test D.nV == 4
    @test ρ != D.ρk
    @test D.ρk ≈ (full_rank ? inv(ρ) : pinv(ρ))

    D = MaximalRenyiDivergence(ρk=ρ, nH=1)
    @test D.nH == 1
    @test D.nV == 4
    @test D.ρk == ρ

    # Can't provide both
    @test_throws ArgumentError MaximalRenyiDivergence(ρ, ρk=ρ, nH=0)
    # Must provide nH
    @test_throws UndefKeywordError MaximalRenyiDivergence(ρ)
end

@testset "Parameter Gradient (Pauli, |0>)" begin
    # Start with a reference state of |00000>
    n = 5
    ψ₀ = zeros(ComplexF64, 2^n)
    ψ₀[1] = 1
    σ₀ = ψ₀ * ψ₀'

    # Now we'd like to learn the maximally mixed state
    ρ = maximally_mixed_state(3)
    D = MaximalRenyiDivergence(ρ, nH=2)

    pool = [
        ADAPT.PauliOperators.Pauli("XXXXX")
    ]

    #=
        Construct an ansatz that is equivalent to having chosen the operator
        X⊗5 from the pool, but before BFGS optimization. The commutator of
        [X⊗5, |0><0|] = |1><0| - |0><1|. The partial trace of this is 0. Hence,
        we should get a zero gradient
    =#
    ansatz = ADAPT.Ansatz(Float64, pool)
    push!(ansatz, pool[1] => 0.0)

    # Calculate the gradient, and check its properties
    grad = ADAPT.gradient(ansatz, D, σ₀)

    @test length(grad) == 1
    @test allzero(grad)

    # Now construct the same ansatz, with new parameter. We should get
    # a zero gradient, since any other parameter will decrease the mixing.
    ansatz = ADAPT.Ansatz(Float64, pool)
    push!(ansatz, pool[1] => π/4)

    @test length(grad) == 1
    @test allzero(grad)
end

@testset "Pool Gradient (Pauli, |0>)" begin
    # Start with a reference state of |00000>
    n = 5
    ψ₀ = zeros(ComplexF64, 2^n)
    ψ₀[1] = 1
    σ₀ = ψ₀ * ψ₀'

    # Now we'd like to learn the maximally mixed state
    ρ = maximally_mixed_state(3)
    D = MaximalRenyiDivergence(ρ, nH=2)

    pool = [
        ADAPT.PauliOperators.Pauli(join(s, ""))
        for s in Iterators.product(Iterators.repeated("IXYZ", 5)...)
    ]
    pool = reshape(pool, 4^5)
    @test length(pool) == 4^5

    # Brand new, empty ansatz
    ansatz = ADAPT.Ansatz(Float64, pool)

    # Calculate the gradient, and check its properties. If we have all 0s,
    # this doesn't bode well for RenyiADAPT
    grad = ADAPT.calculate_scores(ansatz, ADAPT.VANILLA, pool, D, σ₀)
    @test length(grad) == length(pool)
    @test allzero(grad)
end

@testset "Pool Gradient (Pauli, |0> + |1>)" begin
    # Start with a reference state of |00000> + |11111>
    n = 5
    ψ₀ = zeros(ComplexF64, 2^n)
    ψ₀[1] = 1 / √(2)
    ψ₀[end] = 1 / √(2)
    σ₀ = ψ₀ * ψ₀'

    # Now we'd like to learn the maximally mixed state
    ρ = maximally_mixed_state(3)
    D = MaximalRenyiDivergence(ρ, nH=2)

    pool = [
        ADAPT.PauliOperators.Pauli(join(s, ""))
        for s in Iterators.product(Iterators.repeated("IXYZ", 5)...)
    ]
    pool = reshape(pool, 4^5)
    @test length(pool) == 4^5

    # Brand new, empty ansatz
    ansatz = ADAPT.Ansatz(Float64, pool)

    # Calculate the gradient, and check its properties. If we have all 0s,
    # this doesn't bode well for RenyiADAPT
    grad = ADAPT.calculate_scores(ansatz, ADAPT.VANILLA, pool, D, σ₀)
    @test length(grad) == length(pool)
    @test allzero(grad)
end

@testset "Pool Gradient (Pauli, pure_state)" begin
    # Start with a random pure state
    n = 5
    σ₀ = random_pure_state(5)

    # Now we'd like to learn the maximally mixed state
    ρ = maximally_mixed_state(3)
    D = MaximalRenyiDivergence(ρ, nH=2)

    pool = [
        ADAPT.PauliOperators.Pauli(join(s, ""))
        for s in Iterators.product(Iterators.repeated("IXYZ", 5)...)
    ]
    pool = reshape(pool, 4^5)
    @test length(pool) == 4^5

    # Brand new, empty ansatz
    ansatz = ADAPT.Ansatz(Float64, pool)

    # Calculate the gradient, and check its properties. If we have all 0s,
    # this doesn't bode well for RenyiADAPT
    grad = ADAPT.calculate_scores(ansatz, ADAPT.VANILLA, pool, D, σ₀)
    @test length(grad) == length(pool)
    @test somenonzero(grad)
end
