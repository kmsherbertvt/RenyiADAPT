import CSV
import DataFrames
import RenyiADAPT
import RenyiADAPT.ThermalStatesExperiment: init_dataframe, Params, Float, get_hamiltonian, get_referencevector
import DelimitedFiles: readdlm
import Serialization
import LinearAlgebra: norm, tr


# # DATA FOR FIGURE 1: LOSS VS. BFGS ITERATION
# loading it directly as delimited file to plot in optimization.jl


# DATA FOR FIGURE 2 AND FIGURE 5: INFIDELITY VS. ADAPT ITERATION
for nV in 1:4
    for nH in nV:nV
        for seed in 1:20
            nV >= 4 && seed > 1 && continue
            dfg = init_dataframe()
            
            if nV == 3 && nH == 3 
                max_grad_file = "thermalstates/gibbsresults/max_gradients/nV_"*string(nV)*"_nH_"*string(nH)*"_seed_"*string(seed)*"_maxgrads.dat"
                max_grad_data = readdlm(max_grad_file,Float64)
            end
            input_file = "thermalstates/gibbsresults/nV_"*string(nV)*"_nH_"*string(nH)*"_seed_"*string(seed)*"infidelity_vs_params.dat"
            mydata = readdlm(input_file,Float64)
            
            for i in axes(mydata,1)
                push!(dfg, [
                    "twolocal", #enum_H,
                    "entangled", #enum_ψREF,
                    "twolocal", #enum_pool,
                    "gibbs", #enum_method,
                    nV,
                    nH,
                    seed, # seed_H,
                    seed, # seed_ψ,
                    mydata[i,1], #numparams,
                    0, #numiters,
                    0.0, #runtime,
                    0.0, # purity,
                    0.0, # entropy,
                    0.0, # distance,
                    1 - mydata[i,2], # fidelity, # (the input file contains INfidelity data)
                    nV == 3 ? max_grad_data[i,2] : 0.0, # maxpoolgradient
                ])
                output_file = "thermalstates/gibbsresults/csv/twolocal.entangled.twolocal.gibbs."*string(nV)*"."*string(nH)*"."*string(seed)*"."*string(seed)*".csv"
                CSV.write(output_file, dfg)
            end
        end
    end
end


# DATA FOR FIGURE 3 AND FIGURE 4: LARGEST POOL GRADIENT VS. SYSTEM SIZE
firststeps_df = DataFrames.DataFrame(
    # INPUT PARAMETERS
    :enum_H => String[],
    :enum_ψREF => String[],
    :enum_pool => String[],
    :enum_method => String[],
    :nV => Int[],
    :nH => Int[],
    :seed_H => Int[],
    :seed_ψREF => Int[],
    # SCORE VECTOR NORMS
    :GL => Int[],       # Number of non-zero scores.
    :G0 => Float[],
    :G1 => Float[],
    :G2 => Float[],
    :G∞ => Float[],
    # DISTANCE MEASURES
    :fidelity => Float[],
    :distance => Float[],
)
for nV in 1:7
    for nH in nV:nV
        for seed in 1:20
            nV ≥ 4 && seed > 1 && continue
            setup = Params(
                "twolocal",
                "entangled",
                "twolocal",
                "gibbs",
                nV,
                nH,
                seed,
                seed,
            )

            filename = "thermalstates/gibbsresults/gradients/nV_"*string(nV)*"_nH_"*string(nH)*"_seed_"*string(seed)*"_pool_grads.dat"
            mydata = readdlm(filename,Float64)

            # SCORE VECTOR NORMS
            grad_vec1 = mydata[1, 2:end]
            GL = count(g -> abs(g) > eps(Float64), grad_vec1)
            G0 = norm(grad_vec1, 0); G1 = norm(grad_vec1, 1); G2 = norm(grad_vec1, 2); G∞ = norm(grad_vec1, Inf)

            # FETCH ρ AND σ0 FOR DISTANCE MEASURES
            _, ρ, _, ρs = get_hamiltonian(setup)
            ψREF = get_referencevector(setup)
            σ = ψREF * ψREF'
            σ0 = RenyiADAPT.partial_trace(σ, setup.nH)

            push!(firststeps_df, [
                "twolocal",
                "entangled",
                "twolocal",
                "gibbs",
                nV,
                nH,
                seed,
                seed,
                GL,
                G0,
                G1,
                G2,
                G∞,
                # DISTANCE MEASURES
                real( tr(sqrt(ρs*σ0*ρs)) ),
                real( tr(abs.(ρ .- σ0)) / 2 ),
            ])            
        end
    end
end

output_file = "thermalstates/gibbsresults/csv/largest_pool_grad_first_steps.csv"
CSV.write(output_file, firststeps_df)