import PauliOperators: Pauli, PauliSum, ScaledPauli, ScaledPauliVector, FixedPhasePauli, KetBitString, SparseKetBasis, clip!
using LinearAlgebra
import Optim: trace, optimize, BFGS, Options, f_trace, minimizer
using FiniteDifferences

include("Ansatz.jl")

function vqe(objective_func::ModifiedObjective, pool::Vector{Vector{ScaledPauli{N}}}, reference_ket, ansatz, true_objective::GibbsStateFidelity, true_thermal_state, write_output, logfile3) where {N}
    """
    Performs VQE algorithm

    @param objective_func A callable object which, when provided with a state,
        will compute the objective function to be minimized.
    @param pool Operator pool to use for ansatz generation    
    @param reference_ket Initial state to use for VQE
    @param ansatz An object containing the generators and parameters in the ansatz
    @param true_objective If provided, will be evaluated and reported at the end
        of each iteration.
    @returns Whether the VQE converged
    """
    println("VQE starting")
    
    io3 = open(logfile3,"w")
    
    # EXACT MINIMUM FOR OBJECTIVE FN.
    exact_obj = real(tr(true_thermal_state*(Matrix(objective_func.expop) + (true_thermal_state / 2.0) ) )) 
    println("Reference state objective value: ", modified_objective(objective_func, reference_ket))
    println("Reference state fidelity value: ", fidelity(true_objective, reference_ket))

    converged = false

    # CONSTRUCT ANSATZ
    for op in pool
        push!(ansatz, op => 0.0)
    end

    PSA_trial = ParameterShiftAnsatz(objective_func, copy(ansatz.generators), ones(length(ansatz.generators)), reference_ket)    
        
    # RUN OPTIMIZATION
    result = optimize(
        params_to_optimize -> eval_obj(PSA_trial.PTA, params_to_optimize),
        params_to_optimize -> gradient(PSA_trial, params_to_optimize),
        copy(ansatz.parameters),
        BFGS(),Options(store_trace=true, iterations = 500, g_tol=1e-6); inplace = false
    )

    # write out the objective fn. values
    for i in range(1,length(f_trace(result)))
        write(io3, string(exact_obj)*"\t", string(f_trace(result)[i])*"\n")
    end

    println("VQE finished")    
    close(io3)
    return converged
end    
    