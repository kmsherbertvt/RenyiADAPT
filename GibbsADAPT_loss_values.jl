#= Run GibbsADAPT on random two-local Hamiltonians with the two-local pool. =#
#= This generates data for Figure 1 of the Renyi-ADAPT manuscript. =#
#= For the loss data, nV: 3 -> 3, nH : nV -> nV, seed: 1 -> 1. =#

using OpenQuantumTools
import Serialization: serialize, deserialize
using NPZ
import ADAPT
import PauliOperators: Pauli, PauliSum, ScaledPauli, ScaledPauliVector, FixedPhasePauli, KetBitString, SparseKetBasis, clip!
using LinearAlgebra
using SparseArrays

include("src/FreeEnergy.jl")
include("src/loss_adaptvqe.jl")
include("src/vqe.jl")
include("src/Ansatz.jl")
include("src/Hamiltonian_and_Pool.jl")

temperature = 1.0
expansion_order = 5
write_output = true

for nV in 3:3
    for seed in 1:1
        nV >= 4 && seed > 1 && continue
        
        # COMPARE TO EXACT THERMAL STATE
        exact_filename = "thermalstates/setup/thermalstates/twolocal."*string(nV)*"."*string(seed)*".npy"
        exact_gibbs = npzread(exact_filename); sqrt_gibbs = sqrt(exact_gibbs)        
        
        for nH in nV:nV
            output_file1 = "thermalstates/gibbsresults/loss/nV_"*string(nV)*"_nH_"*string(nH)*"_seed_"*string(seed)*"_ADAPT_loss.dat"
            output_file2 = "thermalstates/gibbsresults/loss/nV_"*string(nV)*"_nH_"*string(nH)*"_seed_"*string(seed)*"_ADAPT_steps.dat"
            output_file3 = "thermalstates/gibbsresults/loss/nV_"*string(nV)*"_nH_"*string(nH)*"_seed_"*string(seed)*"_VQE_loss.dat"
            
            println("\nnV = $nV, nH = $nH, seed = $seed")
            n_system = nV+nH
            
            # BUILD OUT THE PROBLEM HAMILTONIAN: # Hamiltonians are named as "twolocal.nV.seed" 
            Ham_filename = "thermalstates/setup/hamiltonians/twolocal."*string(nV)*"."*string(seed)
            mydict = deserialize(Ham_filename)
            H = PauliSum{length(first(keys(mydict)))}(Dict(FixedPhasePauli(uppercase(label)) => coeff for (label, coeff) in mydict)) 
            
            # CONSTRUCT A REFERENCE STATE # Reference vectors are named "entangled.nV.nH.seed.npy"
            ref_vec_filename = "thermalstates/setup/referencevectors/entangled."*string(nV)*"."*string(nH)*"."*string(seed)*".npy"
            y = npzread(ref_vec_filename); ψREF = copy(y)
            
            # BUILD OUT THE OPERATOR POOL
            pool = oneandtwo_local_pool(nV, nH)
            pool_r_values = ones(length(pool)); # Operators are chosen s.t. r = 1 (where r = 0.5*(λ1 - λ0), and λ are the two distinct eigenvalues of each pool operator, to be used in the parameter-shift rule)
            
            # DEFINE THE OBJECTIVE FUNCTION, FREE ENERGY AND GIBBS STATE FIDELITY
            Obj_objective = ModifiedObjective(nV, nH, H, temperature, expansion_order)
            Obj_exact_free_energy = ExactFreeEnergy(nV, nH, H, temperature)
            Obj_gibbs_state_fidelity = GibbsStateFidelity(nV, nH, H, temperature)
            
            # INITIALIZE THE ANSATZ
            ansatz = ADAPT.Ansatz(Float64, pool)
            
            # RUN THE ADAPT-VQE ALGORITHM
            (final_objective, final_state, final_ansatz, adapt_converged) = loss_adapt_vqe(Obj_objective, pool, ψREF, ansatz, Obj_gibbs_state_fidelity, exact_gibbs, write_output, output_file1, output_file2)

            # RUN THE VQE
            ansatz = ADAPT.Ansatz(Float64, pool); ψREF = copy(y)          
            vqe_converged = vqe(Obj_objective, pool, ψREF, ansatz, Obj_gibbs_state_fidelity, exact_gibbs, write_output, output_file3)
        end
    end
end