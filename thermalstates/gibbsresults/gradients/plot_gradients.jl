using DelimitedFiles: readdlm
import DataFrames
import LinearAlgebra: norm

gibbs_df = DataFrames.DataFrame()

for nV in 1:3
    for nH in nV:nV
        for seed in 1:20
            filename = "../Gibbs/results/gradients/nV_"*string(nV)*"_nH_"*string(nH)*"_seed_"*string(seed)*"_pool_grads.dat"
            mydata = readdlm(filename,Float64)
            
            # SCORE VECTOR NORMS
            grad_vec1 = mydata[1, 2:end]           
            GL = count(g -> abs(g) > eps(Float64), grad_vec1)
            G0 = norm(grad_vec1, 0); G1 = norm(grad_vec1, 1); G2 = norm(grad_vec1, 2); G∞ = norm(grad_vec1, Inf)
            
#             header = [:iter,:grad]
#             my_df = DataFrames.DataFrame(mydata,vec(header))
            
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
println(gibbs_df)
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