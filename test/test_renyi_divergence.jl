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