import PauliOperators: Pauli, PauliSum, ScaledPauli, ScaledPauliVector, FixedPhasePauli, KetBitString, SparseKetBasis, clip!
using LinearAlgebra
# using KrylovKit

function Base.adjoint(ψ::SparseKetBasis)
       ψ_ = deepcopy(ψ)
       for (ket, coeff) in ψ_
           ψ_[ket] = coeff'
       end
       return ψ_
end

mutable struct PseudoTrotterAnsatz
    """
    Create a new PseudoTrotterAnsatz object. Intended as a base class for
    other ansatze with different gradient calculation mechanisms.
    
    @param objective_function A callable object which, when provided with a
        state, will return the value of the objective function to be
        minimized
    @param generators An object containing the generators which generate our ansatz
    @param ref_state Base reference state to use
    """
    objective_function
    generators
    ref_state
    iter
    curr_obj
    curr_grad

    function PseudoTrotterAnsatz(objective_function, generators, ref_state)
        iter = 0
        curr_obj = 0
        curr_grad = []
        new(objective_function, generators, ref_state, iter, curr_obj, curr_grad)
    end
end

function prepare_state(PTA::PseudoTrotterAnsatz, params, base_ket = nothing)
    """ 
    Prepares the state
    exp(- im * theta_n * generator_n) ... exp(- im * theta_1 * generator_1) |ket>
    
    @param params The coefficients of each generators
    @param base_ket If not None, use this for |ket>. Else, use the
        reference state
    @returns The prepared state
    """
    if !isnothing(base_ket)
        new_state = 1.0 * base_ket
    else
        new_state = 1.0 * PTA.ref_state
    end
    for i in range(1,length(PTA.generators))
        ADAPT.evolve_state!(PTA.generators[i], params[i], new_state)
    end
    return new_state
end

function eval_obj(PTA::PseudoTrotterAnsatz, params)
    """
    Evaluates the objective function for the state
    exp(- im * theta_n * generator_n) ... exp(- im * theta_1 * generator_1) |ket>
    
    @param params List of parameters. Should be the same size as the
        generator list
    @returns The objective function evaluated at the given parameters
    """
    state = prepare_state(PTA, params)
    obj = modified_objective(PTA.objective_function,state)
    PTA.curr_obj = obj
    return obj
end

function gradient(PTA::PseudoTrotterAnsatz, params)
    """
    Computes the gradient of all the parameters in the ansatz evaluated at
    the provided parameters.
    
    @param params List of parameters at which we evaluate the gradient.
        Should be same size as generator list.
    @returns An array whose entries are the derivatives of the
        objective function w.r.t. the corresponding parameter, evaluated at
        the provided parameters. Computed via commutators assuming
        Hermitian generators.
    """
    grad = similar(params)
    curr_ket = prepare_state(PTA, params)
    curr_bra = ((curr_ket)')*gradient_operator(PTA.objective_function,curr_ket) 
    for i in reverse(eachindex(params))
        ADAPT.evolve_state!(PTA.generators[i], -params[i], curr_ket)
        ADAPT.evolve_state!(PTA.generators[i], -params[i], curr_bra')
        overlap = curr_bra*Matrix(PTA.generators[i])*(curr_ket)
        grad[i] = 2.0 * imag(overlap)        
    end
    PTA.curr_grad = copy(grad)
    return grad
end

function finite_diff_grad!(
    result::AbstractVector,
    PTA::PseudoTrotterAnsatz,
    params,
)
    """
    Computes the gradient of all the parameters in the ansatz evaluated at
    the provided parameters using finite differences.
    
    @param params List of parameters at which we evaluate the gradient.
        Should be same size as generator list.
    @returns An array whose entries are the derivatives of the
        objective function w.r.t. the corresponding parameter, evaluated at
        the provided parameters. Computed via finite differences.
    """    
    cfd = FiniteDifferences.central_fdm(5, 1)
    x0 = copy(params)
    fn = parameters -> eval_obj(PTA, parameters) 
    result .= FiniteDifferences.grad(cfd, fn, x0)[1]
    return result
end

struct ParameterShiftAnsatz
    """
    Creates a new ParameterShiftAnsatz object.
    
    @param objective_function A callable object which, when provided with a
        state, will return the value of the objective function to be
        minimized. May optionally implement a gradient_func for parameter-
        shift gradient calculations. Otherwise, the objective function
        itself will be used.
    @param generators A list of operators which generate our ansatz
    @param r_values List of the r-values for all generators. Used for
        parameter shift calculations
    @param ref_state Base reference state to use
    """
    PTA::PseudoTrotterAnsatz
    r_values
    gradient_function
    
    function ParameterShiftAnsatz(objective_function, generators, r_values, ref_state)
        PTA = PseudoTrotterAnsatz(objective_function, generators, ref_state)
        gradient_function = true
        new(PTA,r_values,gradient_function)
    end
end

function gradient(PSA::ParameterShiftAnsatz, params)
    """
    Computes the gradient of all the parameters in the ansatz evaluated at
    the provided parameters using the parameter-shift rule.
    
    @param params List of parameters at which we evaluate the gradient.
        Should be same size as generator list.
    @returns An array whose entries are the derivatives of the
        objective function w.r.t. the corresponding parameter, evaluated at
        the provided parameters. Computed via parameter shifts.
    """
    grad = similar(params)

    curr_state = 1.0 * PSA.PTA.ref_state
    unshifted = prepare_state(PSA.PTA, params)
    shifted_params = copy(params)

    for i=1:length(params)
        shifted_params[i] += pi / (4.0*PSA.r_values[i])
        upshift_state = prepare_state(PSA.PTA, shifted_params, curr_state)
        shifted_params[i] -= 2 * pi / (4.0*PSA.r_values[i])
        downshift_state = prepare_state(PSA.PTA, shifted_params, curr_state)
        if !PSA.gradient_function
            upshift = modified_objective(PSA.PTA.objective_function, upshift_state) 
            downshift = modified_objective(PSA.PTA.objective_function, downshift_state)
        else
            upshift = gradient_function(PSA.PTA.objective_function, upshift_state, unshifted)
            downshift = gradient_function(PSA.PTA.objective_function, downshift_state, unshifted)
        end
        derivative = PSA.r_values[i] * (upshift - downshift)
        @assert isapprox(imag(derivative), 0.0; atol=1e-8) "Imaginary part of derivative should be 0."
        grad[i] = real(derivative)
    
        shifted_params[i] = 0
        if !isapprox(params[i], 0; atol=1e-8)
            ADAPT.evolve_state!(PSA.PTA.generators[i], params[i], curr_state)
        end
    end
    PSA.PTA.curr_grad = copy(grad)
    return grad
end