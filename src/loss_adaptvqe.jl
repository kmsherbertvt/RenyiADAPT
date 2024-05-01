import PauliOperators: Pauli, PauliSum, ScaledPauli, ScaledPauliVector, FixedPhasePauli, KetBitString, SparseKetBasis, clip!
using LinearAlgebra
import Optim: trace, optimize, BFGS, Options, f_trace, minimizer
using FiniteDifferences

include("Ansatz.jl")

function loss_adapt_vqe(objective_func::ModifiedObjective, pool::Vector{Vector{ScaledPauli{N}}}, reference_ket, ansatz, true_objective::GibbsStateFidelity, true_thermal_state, write_output, logfile1, logfile2, adapt_thresh= 1e-3, adapt_maxiter = 300) where {N}
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
    @param write_output Whether to write out the loss or not. 
    @param logfile1, logfile2 Output files to write out the ADAPT loss values and the ADAPT steps.
    @param adapt_thresh Convergence threshold. ADAPT completes when gradient
        norm is less than this
    @param adapt_maxiter Maximum number of iterations
    @returns Tuple containing the final best value of the objective function,
        final state, final ansatz, and whether convergence was reached before 
        the maximum number of iterations.
    """
    println("ADAPT starting")
    
    if write_output 
        io = open(logfile1,"w") 
        io2 = open(logfile2,"w")   
    end
    
    # EXACT MINIMUM FOR OBJECTIVE FN.
    exact_obj = real(tr(true_thermal_state*(Matrix(objective_func.expop) + (true_thermal_state / 2.0) ) )) 
    println("Reference state objective value: ", modified_objective(objective_func, reference_ket))
    println("Reference state fidelity value: ", fidelity(true_objective, reference_ket))

    curr_state = 1.0 * reference_ket
    converged = false
    pool_r_values = ones(length(pool))

    num_BFGS_iters = 0
    for n_iter=1:adapt_maxiter
        println("                      ADAPT-VQE iteration: ", n_iter)
        
        # Fast way of computing gradient of each operator in pool were we to add it next
        PSA = ParameterShiftAnsatz(objective_func, pool, pool_r_values, curr_state)
        grad = gradient(PSA, zeros(length(pool)))  # analytical gradient
        
        grad_norm = norm(grad)
        # println("\tGrad norm", grad_norm); println("\tMax grad",maximum(abs.(grad)))

        if grad_norm < adapt_thresh
            converged = true
            println("converged!")
            break
        end

        next_op_idx = argmax(abs.(grad)); # println("\tAdd operator ", next_op_idx)
        push!(ansatz, pool[next_op_idx] => 0.0)

        PSA_trial = ParameterShiftAnsatz(objective_func, copy(ansatz.generators), ones(length(ansatz.generators)), reference_ket)
        
#         grad = similar(ansatz.parameters)
        
        # RUN OPTIMIZATION
        result = optimize(
            params_to_optimize -> eval_obj(PSA_trial.PTA, params_to_optimize),
            params_to_optimize -> gradient(PSA_trial, params_to_optimize),
            copy(ansatz.parameters),
            BFGS(),Options(store_trace=true, iterations = 5000, g_tol=1e-6); inplace = false
        )
        
        # write out the objective fn. (ADAPT loss) values
        for i in range(1,length(f_trace(result)))
            write(io, string(exact_obj)*"\t", string(f_trace(result)[i])*"\n")
        end
        
        num_BFGS_iters += length(f_trace(result)); println("\tCurrent BFGS iters: ", num_BFGS_iters)
        optimized_parameters = minimizer(result); ADAPT.bind!(ansatz, optimized_parameters)
        curr_state = prepare_state(PSA_trial.PTA, optimized_parameters)
        curr_obj = modified_objective(objective_func, curr_state); println("\tCurrent objective value: ", curr_obj)
        # write out the loss value for this ADAPT step
        write(io2, string(num_BFGS_iters)*"\t", string(abs(curr_obj - exact_obj))*"\n")
        curr_fid = fidelity(true_objective, curr_state); println("\tCurrent fidelity value (true): ", curr_fid)
    end
    final_objective = modified_objective(objective_func, curr_state)

    println("ADAPT finished")    
    close(io); close(io2); 
    return (final_objective, curr_state, ansatz, converged)
end    
    