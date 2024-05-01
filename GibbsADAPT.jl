#= Run GibbsADAPT on random two-local Hamiltonians with the two-local pool. =#
#= The infidelity (gradient) portion generates data for Figure 2 (Figure 5) of the Renyi-ADAPT manuscript. =#
#= For the infidelity data, nV: 1 -> 4, nH : nV -> nV, seed: 1 -> 20. =#
#= For the gradient data, nV: 3 -> 3, nH : nV -> nV, seed: 1 -> 20. =#

using OpenQuantumTools
import Serialization: serialize, deserialize
using NPZ
import ADAPT
import PauliOperators: Pauli, PauliSum, ScaledPauli, ScaledPauliVector, FixedPhasePauli, KetBitString, SparseKetBasis, clip!
using LinearAlgebra
using SparseArrays

include("src/FreeEnergy.jl")
include("src/adaptvqe.jl")
include("src/Ansatz.jl")
include("src/Hamiltonian_and_Pool.jl")

temperature = 1.0
expansion_order = 5
write_infidelity = true; write_grad = true

for nV in 1:4
    for seed in 1:20
        nV >= 4 && seed > 1 && continue
        
        # COMPARE TO EXACT THERMAL STATE
        exact_filename = "thermalstates/setup/thermalstates/twolocal."*string(nV)*"."*string(seed)*".npy"
        exact_gibbs = npzread(exact_filename); sqrt_gibbs = sqrt(exact_gibbs)
        
        for nH in nV:nV
            infidelity_output = "thermalstates/gibbsresults/nV_"*string(nV)*"_nH_"*string(nH)*"_seed_"*string(seed)*"infidelity_vs_params.dat"
            gradient_output = "thermalstates/gibbsresults/max_gradients/nV_"*string(nV)*"_nH_"*string(nH)*"_seed_"*string(seed)*"_maxgrads.dat"

            t_0 = time()
            println("\nnV = $nV, nH = $nH, seed = $seed")
            n_system = nV+nH
            
            # BUILD THE PROBLEM HAMILTONIAN: # Hamiltonians are named as "twolocal.nV.seed" 
            Ham_filename = "thermalstates/setup/hamiltonians/twolocal."*string(nV)*"."*string(seed)
            mydict = deserialize(Ham_filename)
            H = PauliSum{length(first(keys(mydict)))}(Dict(FixedPhasePauli(uppercase(label)) => coeff for (label, coeff) in mydict)) 
            
            # CONSTRUCT THE REFERENCE STATE # Reference vectors are named "entangled.nV.nH.seed.npy"
            ref_vec_filename = "thermalstates/setup/referencevectors/entangled."*string(nV)*"."*string(nH)*"."*string(seed)*".npy"
            y = npzread(ref_vec_filename); ψREF = copy(y)
            
            # BUILD THE OPERATOR POOL
            pool = oneandtwo_local_pool(nV, nH)
            pool_r_values = ones(length(pool)); # Operators are chosen s.t. r = 1 (where r = 0.5*(λ1 - λ0), and λ are the two distinct eigenvalues of each pool operator, to be used in the parameter-shift rule)
            
            # DEFINE THE OBJECTIVE FUNCTION, FREE ENERGY AND GIBBS STATE FIDELITY
            Obj_objective = ModifiedObjective(nV, nH, H, temperature, expansion_order)
            Obj_exact_free_energy = ExactFreeEnergy(nV, nH, H, temperature)
            Obj_gibbs_state_fidelity = GibbsStateFidelity(nV, nH, H, temperature)
            
            # INITIALIZE THE ANSATZ
            ansatz = ADAPT.Ansatz(Float64, pool)
            
            # RUN THE ALGORITHM
            (final_objective, final_state, final_ansatz, adapt_converged) = adapt_vqe(Obj_objective, pool, ψREF, ansatz, Obj_gibbs_state_fidelity, exact_gibbs, write_infidelity, write_grad, infidelity_output, gradient_output)
            
            # RESULTS
            ADAPT_Gibbs_state = density_operator(final_state, nH, nV)
            fin_fidelity = (LinearAlgebra.tr(sqrt(sqrt_gibbs*ADAPT_Gibbs_state*sqrt_gibbs)))^2
            println("Fidelity w.r.t exact Gibbs state: ",real(fin_fidelity))
            t_f = time(); dt = (t_f - t_0)/60.0; println("The walltime for running ADAPT-VQE-Gibbs for nV = $nV, nH = $nH, seed = $seed was $dt minutes\n")   
        end
    end
end