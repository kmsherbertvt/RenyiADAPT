using LinearAlgebra
using Distributions: Uniform

tensor(ψ::Vector{Vector{Complex{F}}}) where {F} = reduce(kron, ψ)

function random_ket(n::Int)
    ψ = rand(ComplexF64, 2^n)
    ψ = ψ ./ norm(ψ)
    return ψ
end

function random_product_state(n::Int)
    ψ = tensor([random_ket(1) for _ in 1:n])
    ρ = ψ * ψ'
    return ρ
end

function random_pure_state(n::Int)
    ψ = random_ket(n)
    ρ = ψ * ψ'
    return ρ
end

function random_mixed_state(qubits::Int)
    # Generate a random number of pure states, 2-5. Then assign
    # classical probabilities to each of them
    n_states = rand(2:5)
    p = rand(Uniform(0, 1), n_states)
    p = p ./ sum(p)

    ρ = zeros(ComplexF64, (2^qubits, 2^qubits))
    for pᵢ in p
        ρ += pᵢ .* random_pure_state(qubits)
    end

    return ρ
end

function maximally_mixed_state(qubits::Int)
    dim = 2 ^ qubits
    return Matrix{ComplexF64}(1 / dim * I, dim, dim)
end