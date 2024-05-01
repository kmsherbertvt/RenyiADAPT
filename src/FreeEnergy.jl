#= Set up structs related to the exact and truncated (approximate) free energies. =#

using OpenQuantumTools
import PauliOperators: Pauli, PauliSum, ScaledPauli, ScaledPauliVector, FixedPhasePauli, KetBitString, SparseKetBasis, clip!, mul!
using LinearAlgebra

function logm_expansion_coeff(n, n_max)
    """
    Coefficients for the taylor expansion of -log(rho).
    -log(rho) = (1-rho) + (rho-1)^2/2 - (rho-1)^3/3 + (rho-1)^4/4 - (rho-1)^5/5 + ...
    
    # Parameters
    - `n`: index of the desired coefficient
    - `n_max`: highest order in the truncated Taylor expansion

    # Returns
    - `C_n`: coefficient of rho^n in the truncated Taylor expansion of -log(rho), 
    - where -log(rho) = expansion_coeffs[1]*(rho^0) + expansion_coeffs[2]*(rho^1) + ... 
                            + expansion_coeffs[expansion_order+1]*(rho^expansion_order)
    """
    coeff = 0.0
    for i in range(max(n, 1), n_max)
        coeff += binomial(i, n) / i * ((-1)^(n&1))
    end
    return coeff
end

function density_operator(state::Vector{ComplexF64}, n_ancilla::Int, n_data::Int)
    """
    Given a pure state in a Hilbert space which can be partitioned into the
    Euclidean product of two smaller Hilbert spaces, H = H_data x H_ancilla,
    computes the reduced density matrix produced by doing the partial trace over
    H_ancilla.

    # Parameters
    - `state`: The pure state we're starting with. Must have dimension
        2^(n_ancilla + n_data)
    - `n_ancilla`: Number of ancillary qubits being traced out (H_ancilla)
    - `n_data`: Number of data qubits whose density operator we're calculating (H_data)
    
    # Returns
    - The (dense) density matrix computed
    """    
    density_matrix = state*state'
    return partial_trace(density_matrix, [2^n_data,2^n_ancilla], [1])
end

struct ModifiedObjective
    """
    Implements the modified objective function
    C(\rho) = -Tr(e^{-H/T} \rho) / Z + Tr(\rho^2) / 2
    which has the Gibbs state as its minimum, but approximates e^{-H/T}
    with a truncated Taylor series, and approximates Z with its trace.

    # Parameters
    - `n_data`: number of data qubits in our register. Assumed to be
        the first (most significant) n_data qubits
    - `n_ancilla`: number of ancilla qubits in our register. Assumed
        to be the final (least significant) n_ancilla qubits.
    - `ham`: Hamiltonian to be used to compute energies. Should act
        on only data qubits, i.e. should have dimension 2^n_data
    - `temp`: System temperature
    - `expansion_order`: Highest order of H to include in the Taylor
        expansion. Set to -1 for the full infinite series
    
    # Attributes
    - `expop`: approximation of -e^{-H/T}/Z
    """    
    n_data::Int64
    n_ancilla::Int64
    ham::PauliSum
    temperature::Float64
    expansion_order::Int8
    expop #::PauliSum
    
    function ModifiedObjective(n_data, n_ancilla, ham, temperature, expansion_order=-1)
        if expansion_order == -1
            Hmat = Matrix(ham)
            expop = exp(-1.0*Hmat / temperature)
        elseif expansion_order > 0
            opPower = (-1.0/temperature)*ham
            expop = PauliSum(n_data)
            expop += (1 + 0.0im)*Pauli("I"^n_data)
            for i in range(1, expansion_order)
                expop += (1.0/factorial(i))*opPower
                opPower = opPower*((-1.0/temperature)*ham)
            end
        else
            error("Bad expansion order")
        end
        tr_exp = tr(expop)
        expop = (-1.0/tr_exp)*expop
        new(n_data, n_ancilla, ham, temperature, expansion_order, expop)
    end
end

function modified_objective(MO::ModifiedObjective, state::Vector{ComplexF64})
    """
    Computes the objective function for the given state

    # Parameters
    - `state`: state whose objective function we're computing

    # Returns
    - Objective function evaluated for the given state
        C(\rho) = -Tr(e^{-H/T} \rho) / Z + Tr(\rho^2) / 2
    """
    density = density_operator(state, MO.n_ancilla, MO.n_data)
    if typeof(MO.expop)==PauliSum{MO.n_data}
        return real(tr(density*(Matrix(MO.expop) + density / 2.0))) 
    elseif typeof(MO.expop)==Matrix{ComplexF64}
        return real(tr(density*(MO.expop + density / 2.0))) 
    end
end

function gradient_function(MO::ModifiedObjective, shifted::Vector{ComplexF64}, unshifted::Vector{ComplexF64})
    """
    Compute the function used to compute gradients of the objective
    function via parameter-shift methods.

    # Parameters
    - `shifted`: The state at shifted parameters
    - `unshifted': The state without parameter shifts

    # Returns
    - A function used to compute the gradient of the objective function.
    """
    shifted_density = density_operator(shifted, MO.n_ancilla, MO.n_data)
    unshifted_density = density_operator(unshifted, MO.n_ancilla, MO.n_data)
    return tr(shifted_density*(Matrix(MO.expop) + unshifted_density))
end

function gradient_operator(MO::ModifiedObjective, state::Vector{ComplexF64})
    """
    Get a Hermitian operator which can be used to compute gradients of the
    objective function

    # Parameters
    - `state`: state at which the gradient is ultimately to be evaluated
    # Returns
    - A Hermitian operator whose gradient for the given function is
        equal to the gradient of the free energy itself
    """
    density = density_operator(state, MO.n_ancilla, MO.n_data)
    op = Matrix(MO.expop) + density
    op = kron(sparse(op), sparse(I,2^MO.n_ancilla,2^MO.n_ancilla))
    return op
end

struct GibbsStateFidelity
    """
    Creates a new GibbsStateFidelity, which computes the fidelity of a state w.r.t. the exact Gibbs state,
    which is given by rho_G = exp(-H/T)/trace(exp(-H/T)).
    
    # Parameters
    - `n_data`: Number of data qubits in our register. Data qubits are
        assumed to be the first (most significant) n_data qubits.
    - `n_ancilla`: Number of ancilla qubits in the register. Assumed to be
        the final (least significant) n_ancilla qubits.
    - `ham`: Hamiltonian to use for energy calculation. Should act only on
        the data qubits, i.e. it should have dim 2^n_data
    - `temp`: Temperature at which we should compute the free energy        
    """
    n_data::Int64
    n_ancilla::Int64
    ham::PauliSum #::Matrix{ComplexF64}
    temperature::Float64
    sqrtGibbs
    
    function GibbsStateFidelity(n_data, n_ancilla, Hamiltonian, temperature)
        Hmat = Matrix(Hamiltonian)
        sqrt_Gibbs_state = exp(-0.5*Hmat/temperature)/sqrt(tr(exp(-Hmat/temperature)))
        new(n_data, n_ancilla, Hamiltonian, temperature, sqrt_Gibbs_state)
    end
end    

function fidelity(GSF::GibbsStateFidelity, state::Vector{ComplexF64})
    """
    Calculates the fidelity of a given state with respect to an exact Gibbs state.
    
    # Parameters
    - `state`: state whose fidelity we are calculating

    # Returns
    - fidelity of the given state w.r.t. the Gibbs state
    """        
    density = density_operator(state, GSF.n_ancilla, GSF.n_data)
    fid = (sum(sqrt.(Complex.(eigvals(GSF.sqrtGibbs * density * GSF.sqrtGibbs)))))^2
    @assert isapprox(imag(fid), 0.0; atol=1e-8) "Imaginary part of fidelity should be 0."
    return real(fid)
end

struct TruncatedFreeEnergy
    """
    # Parameters
    - `n_data`: Number of data qubits in our register. Data qubits are
        assumed to be the first (most significant) n_data qubits.
    - `n_ancilla`: Number of ancilla qubits in the register. Assumed to be
        the final (least significant) n_ancilla qubits.
    - `ham`: Hamiltonian to use for energy calculation. Should act only on
        the data qubits, i.e. it should have dim 2**n_data
    - `expansion_order`: Maximum order of the entropy expansion
    - `temp`: Temperature at which we should compute the free energy    
    
    # Calculates
    - `expansion_coeffs`: Coefficients for the powers of rho in the taylor expansion of -log(rho), 
       with the coefficients going from 0, ..., expansion_order
    """
    n_data::Int64
    n_ancilla::Int64
    ham::PauliSum
    expansion_order::Int64
    expansion_coeffs::Vector{Float64}
    temperature::Float64
    
    function TruncatedFreeEnergy(n_data, n_ancilla, Hamiltonian, expansion_order, temperature)
        expansion_coeffs = [logm_expansion_coeff(n, expansion_order) for n in range(0,expansion_order)]
        new(n_data, n_ancilla, Hamiltonian, expansion_order, expansion_coeffs, temperature)
    end
end

function truncated_free_energy(TFE::TruncatedFreeEnergy, state::Vector{ComplexF64})
    """
    Computes an approximation of the free energy for a given state.
    # Parameters
    - `state`: State whose free energy to compute
    - `for_grad`: Are we trying to compute a gradient?
    - `unshifted': Used in gradient calculations. The state without any parameter shifts

    # Returns
    - `free_energy`: The truncated free energy of the data qubits in the provided state, 
       or if for_grad is true, a value which can be used to compute the gradient via 
       parameter shifts.
    
    # Given that
    -log(rho) = expansion_coeffs[1]*(rho^0) + expansion_coeffs[2]*(rho^1) + ... 
                            + expansion_coeffs[expansion_order+1]*(rho^expansion_order)
    - Thus, the entropy = - trace[ rho * log(rho) ]
                        = trace[ expansion_coeffs[1]*(rho^1) + expansion_coeffs[2]*(rho^2) + ... 
                            + expansion_coeffs[expansion_order+1]*(rho^(expansion_order+1)) ]
    """    
    density = density_operator(state, TFE.n_ancilla, TFE.n_data)
    
    energy_expectation = tr(density*TFE.ham)
    
    entropy = TFE.expansion_coeffs[1]  # initialize

    density_pow = 1.0 * density
    for coeff in TFE.expansion_coeffs[2:end]
        density_pow = density_pow * density
        entropy += coeff * tr(density_pow)
    end

    free_energy = energy_expectation - TFE.temperature * entropy
    @assert isapprox(imag(free_energy), 0.0; atol=1e-8) "Imaginary part of free energy should be 0."
    return real(free_energy)
end

function gradient_function(TFE::TruncatedFreeEnergy, shifted::Vector{ComplexF64}, unshifted::Vector{ComplexF64})
    """
    Compute the function used to compute gradients of the free energy via parameter-shift methods.
    # Parameters
    - `shifted`: The state at shifted parameters
    - `unshifted': The state without parameter shifts

    # Returns
    - A function used to compute the gradient of the truncated free energy.
    """        
    shifted_density = density_operator(shifted, TFE.n_ancilla, TFE.n_data)
    unshifted_density = density_operator(unshifted, TFE.n_ancilla, TFE.n_data)

    energy = tr(shifted_density*TFE.ham)

    entropy = 0.0
    
    density_pow = 1.0 * shifted_density
    for i in range(1, length(TFE.expansion_coeffs)-1)
        density_pow = density_pow*unshifted_density
        entropy += (i + 1) * TFE.expansion_coeffs[i+1] * tr(density_pow)
    end

    func_val = energy - TFE.temperature * entropy
    return func_val
end

function gradient_operator(TFE::TruncatedFreeEnergy, state::Vector{ComplexF64})
    """
    Get an Hermitian operator which can be used to compute gradients of the
    objective function

    # Parameters
    - `state`:The state at which the gradient is ultimately to be evaluated

    # Returns
    - A Hermitian operator whose gradient for the given function is
        equal to the gradient of the free energy itself
    """
    density = density_operator(state, TFE.n_ancilla, TFE.n_data)

    energy = 1.0 * TFE.ham

    entropy = 0.0
    
    density_pow = 1.0 * density
    for i in range(1, length(TFE.expansion_coeffs)-1)
        entropy += (i + 1) * TFE.expansion_coeffs[i+1] * density_pow
        density_pow = density_pow*(density)
    end

    op = energy - TFE.temperature * entropy
    op = kron(sparse(op), sparse(I,2^TFE.n_ancilla,2^TFE.n_ancilla))
    return op
end

struct ExactFreeEnergy
    """
    Computes the Gibbs free energy of given quantum states.
    
    # Parameters
    - `n_data`: Number of data qubits in our register. Data qubits are
        assumed to be the first (most significant) n_data qubits.
    - `n_ancilla`: Number of ancilla qubits in the register. Assumed to be
        the final (least significant) n_ancilla qubits.
    - `ham`: Hamiltonian to use for energy calculation. Should act only on
        the data qubits, i.e. it should have dim 2**n_data
    - `temp`: Temperature at which we should compute the free energy        
    """
    n_data::Int64
    n_ancilla::Int64
    ham::PauliSum
    temperature::Float64
    
    function ExactFreeEnergy(n_data, n_ancilla, Hamiltonian, temperature)
        new(n_data, n_ancilla, Hamiltonian, temperature)
    end
end

function exact_free_energy(EFE::ExactFreeEnergy, state::Vector{ComplexF64})
    """
    Compute the free energy of the provided state exactly.
    
    # Parameters
    - `state`: State whose free energy to compute

    # Returns
    - `free_energy`: The exact free energy of the given state.    
    """    
    density = density_operator(state, EFE.n_ancilla, EFE.n_data)
     #  mul!(out::Matrix{T}, p::PauliSum{N}, in::Matrix{T})
    energy_expectation = tr(density*EFE.ham)
    
    entropy = -tr(density*log(density))

    free_energy = energy_expectation - EFE.temperature * entropy
    
    @assert isapprox(imag(free_energy), 0.0; atol=1e-8) "Imaginary part of free energy should be 0."
    return real(free_energy)
end

function gradient_function(EFE::ExactFreeEnergy, shifted::Vector{ComplexF64}, unshifted::Vector{ComplexF64})
    """
    Compute the function used to compute gradients of the free energy via parameter-shift methods.
    # Parameters
    - `shifted`: The state at shifted parameters
    - `unshifted': The state without parameter shifts

    # Returns
    - A function used to compute the gradient of the free energy.
    """        
    shifted_density = density_operator(shifted, EFE.n_ancilla, EFE.n_data)
    unshifted_density = density_operator(unshifted, EFE.n_ancilla, EFE.n_data)

    energy = tr(shifted_density*EFE.ham)

    entropy = -tr(shifted_density*log(unshifted_density))

    func_val = energy - EFE.temperature * entropy
    
    return func_val
end

function gradient_operator(EFE::ExactFreeEnergy, state::Vector{ComplexF64})
    """
    Hermitian operator which can be used to compute gradients of the free energy.

    # Parameters
    - `state`: state at which the gradient is ultimately to be evaluated

    # Returns
    - A Hermitian operator whose gradient for the given function is
        equal to the gradient of the free energy itself
    """
    density = density_operator(state, EFE.n_ancilla, EFE.n_data)

    op = EFE.ham + EFE.temperature * log(density)
    op = kron(sparse(op), sparse(I,2^EFE.n_ancilla,2^EFE.n_ancilla))
    return op
end

struct HamObjective
    """
    Creates a new HamObjective object, which is used to compute expectation
    values of operators

    @param ham The operator whose expectation value we will compute.
        Assumed to be Hermitian.
    """
    ham::PauliSum    
    function HamObjective(ham)
        new(ham)
    end
end

function hamiltonian_expectation(HOb::HamObjective, state::SparseKetBasis)
    """
    Compute the expectation value

    @param state The state whose expectation value we're computing
    @return <state|ham|state>
    """
    return real(dot(state, HOb.ham * state))
end

function gradient_operator(HOb::HamObjective, state)
    """
    Get a Hermitian operator which can be used to compute gradients of the
    energy

    @param state Not used here
    @return Returns the Hamiltonian for gradient calculation purposes
    """
    return 1.0 * HOb.ham
end