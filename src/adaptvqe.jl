import PauliOperators: Pauli, PauliSum, ScaledPauli, ScaledPauliVector, FixedPhasePauli, KetBitString, SparseKetBasis, clip!
using LinearAlgebra
import Optim: trace, optimize, BFGS, Options, f_trace, minimizer
using FiniteDifferences

include("Ansatz.jl")

function adapt_vqe(objective_func::ModifiedObjective, pool::Vector{Vector{ScaledPauli{N}}}, reference_ket, ansatz, true_objective::GibbsStateFidelity, true_thermal_state, write_infidelity, write_grad, infidelity_logfile, gradient_logfile, adapt_thresh= 1e-3, adapt_maxiter = 300) where {N}
    """
    Performs ADAPT-VQE algorithm

    @param objective_func A callable object which, when provided with a state,
        will compute the objective function to be minimized.
    @param pool Operator pool to use for ansatz generation
    @param reference_ket Initial state to use for ADAPT algorithm
    @param ansatz An object containing the generators and parameters in the ansatz
    @param true_objective If provided, will be evaluated and reported at the end
        of each iteration.
    @param true_thermal_state Exact Gibbs state used to calculate the exact minimum of the objective fn.
    @param write_infidelity Whether to write out the infidelities or not.
    @param write_grad Whether to write out the gradients or not.    
    @param infidelity_logfile Output file to write out the infidelities.
    @param gradient_logfile Output file to write out the largest pool gradients.
    @param adapt_thresh Convergence threshold. ADAPT completes when gradient
        norm is less than this
    @param adapt_maxiter Maximum number of iterations
    @returns Tuple containing the final best value of the objective function,
        final state, final ansatz, and whether convergence was reached before 
        the maximum number of iterations.
    """
    println("ADAPT starting")
    
    if write_infidelity infid_io = open(infidelity_logfile,"w") end
    if write_grad grad_io = open(gradient_logfile,"w") end    
    
    println("Reference state objective value: ", modified_objective(objective_func, reference_ket))
    println("Reference state fidelity value: ", fidelity(true_objective, reference_ket))

    curr_state = 1.0 * reference_ket
    converged = false
    pool_r_values = ones(length(pool))

    for n_iter=1:adapt_maxiter
        println("                      ADAPT-VQE iteration: ", n_iter)
        
        # Fast way of computing gradient of each operator in pool were we to add it next
        PSA = ParameterShiftAnsatz(objective_func, pool, pool_r_values, curr_state)
        grad = gradient(PSA, zeros(length(pool)))  # analytical gradient
        
        # write out the gradients
#         if write_output write(io, string(n_iter)) end              
#         for i in range(1,length(grad))
#             if write_output write(io, "\t"*string(abs(grad[i]))) end              
#         end
#         if write_output write(io, "\n") end

        grad_norm = norm(grad); 
        # println("\tGrad norm", grad_norm); println("\tMax grad",maximum(abs.(grad)))

        if grad_norm < adapt_thresh
            converged = true
            println("converged!")
            break
        end

        next_op_idx = argmax(abs.(grad)); # println("\tAdd operator ", next_op_idx)
        push!(ansatz, pool[next_op_idx] => 0.0)
        if write_grad write(grad_io, string(n_iter)*"\t", string(maximum(abs.(grad)))*"\t", string(next_op_idx)*"\n") end    

        PSA_trial = ParameterShiftAnsatz(objective_func, copy(ansatz.generators), ones(length(ansatz.generators)), reference_ket)
        
#         grad = similar(ansatz.parameters)
        
        # RUN OPTIMIZATION
        result = optimize(
            params_to_optimize -> eval_obj(PSA_trial.PTA, params_to_optimize),
            params_to_optimize -> gradient(PSA_trial, params_to_optimize),
            copy(ansatz.parameters),
            BFGS(),Options(store_trace=true, iterations = 5000, g_tol=1e-6); inplace = false
        )        
        optimized_parameters = minimizer(result); ADAPT.bind!(ansatz, optimized_parameters); println("\tFinished iteration: ")

        curr_state = prepare_state(PSA_trial.PTA, optimized_parameters)
        curr_obj = modified_objective(objective_func, curr_state); println("\tCurrent objective value: ", curr_obj)
        curr_fid = fidelity(true_objective, curr_state); println("\tCurrent fidelity value (true): ", curr_fid)
        
        if write_infidelity write(infid_io, string(n_iter)*"\t", string((1.0-curr_fid))*"\n") end        
    end
    final_objective = modified_objective(objective_func, curr_state)

    println("ADAPT finished")
    println("Number of operators in ansatz: ", length(ansatz.parameters))
    println("Final objective value: ", final_objective)
    println("Final fidelity value: ", fidelity(true_objective, curr_state))
    
    if write_infidelity close(infid_io) end
    if write_grad close(grad_io) end    
    
    return (final_objective, curr_state, ansatz, converged)
end    
    