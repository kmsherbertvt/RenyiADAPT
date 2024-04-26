#= Script that constructs a dataframe compatible with `distance.jl`,
    containing all of Karunya's Gibbs data =#
import RenyiADAPT
import RenyiADAPT.ThermalStatesExperiment as JOB

using DelimitedFiles: readdlm
import DataFrames
import LinearAlgebra: norm, tr

# gibbs_df = DataFrames.DataFrame()
gibbs_df = DataFrames.DataFrame(
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
    :G0 => JOB.Float[],
    :G1 => JOB.Float[],
    :G2 => JOB.Float[],
    :G∞ => JOB.Float[],
    # DISTANCE MEASURES
    :fidelity => JOB.Float[],
    :distance => JOB.Float[],
)


for nV in 1:7
    for nH in nV:nV
        for seed in 1:20
            nV ≥ 4 && seed > 1 && continue


            setup = JOB.Params(
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

#             header = [:iter,:grad]
#             my_df = DataFrames.DataFrame(mydata,vec(header))



            # FETCH ρ AND σ0 FOR DISTANCE MEASURES
            _, ρ, _, ρs = JOB.get_hamiltonian(setup)
            ψREF = JOB.get_referencevector(setup)
            σ = ψREF * ψREF'
            σ0 = RenyiADAPT.partial_trace(σ, setup.nH)

            push!(gibbs_df, [
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

#             if nV == 1
#                 append!(df1, my_df)
#             elseif nV == 2
#                 append!(df2, my_df)
#             elseif nV == 3
#                 append!(df3, my_df)
#             end
        end
    end
end
# println(gibbs_df)
# for nV in 4:4
#     for nH in nV:nV
#         for seed in 1:1
#             filename = "../Gibbs/results/nV_"*string(nV)*"_nH_"*string(nH)*"_seed_"*string(seed)*"_pool_grads.dat"
#             mydata = readdlm(filename,Float64)
#             header = [:iter,:grad]
#             my_df = DataFrames.DataFrame(mydata,vec(header))

#             if nV == 4
#                 append!(df4, my_df)
#             end
#         end
#     end
# end