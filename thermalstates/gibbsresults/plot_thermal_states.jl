import Plots
import ColorSchemes
using DelimitedFiles
import DataFrames   # Julia's version of Python's `pandas`

#= Infidelity vs. ADAPT iteration:=#

plt = Plots.plot(;
    xlabel = "ADAPT Iterations",
    ylabel = "Infidelity",
    ylims = [0.0, 1.0],
    legend = :topright,
)

df1 = DataFrames.DataFrame()
df2 = DataFrames.DataFrame()
df3 = DataFrames.DataFrame()
df4 = DataFrames.DataFrame()

for nV in 1:3
    for nH in nV:nV
        for seed in 1:20
            filename = "../Gibbs/results/nV_"*string(nV)*"_nH_"*string(nH)*"_seed_"*string(seed)*"infidelity_vs_params.dat"
            mydata = readdlm(filename,Float64)
            header = [:numparams,:fidelity]
            my_df = DataFrames.DataFrame(mydata,vec(header))
            
            if nV == 1
                append!(df1, my_df)
            elseif nV == 2
                append!(df2, my_df)
            elseif nV == 3
                append!(df3, my_df)
            end
        end
    end
end

for nV in 4:4
    for nH in nV:nV
        for seed in 1:1
            filename = "../Gibbs/results/nV_"*string(nV)*"_nH_"*string(nH)*"_seed_"*string(seed)*"infidelity_vs_params.dat"
            mydata = readdlm(filename,Float64)
            header = [:numparams,:fidelity]
            my_df = DataFrames.DataFrame(mydata,vec(header))
            
            if nV == 4
                append!(df4, my_df)
            end
        end
    end
end


curves = Dict()

key1 = Dict(); key1[:nV] = 1; key1[:nH] = 1
curves[key1] = DataFrames.combine(
        DataFrames.groupby(df1, :numparams),
        :fidelity => (itr -> DataFrames.quantile(itr, 0.00)) => :q0,
        :fidelity => (itr -> DataFrames.quantile(itr, 0.25)) => :q1,
        :fidelity => (itr -> DataFrames.quantile(itr, 0.50)) => :q2,
        :fidelity => (itr -> DataFrames.quantile(itr, 0.75)) => :q3,
        :fidelity => (itr -> DataFrames.quantile(itr, 1.00)) => :q4,
    )
sort!(curves[key1], :numparams)

key2 = Dict(); key2[:nV] = 2; key2[:nH] = 2
curves[key2] = DataFrames.combine(
        DataFrames.groupby(df2, :numparams),
        :fidelity => (itr -> DataFrames.quantile(itr, 0.00)) => :q0,
        :fidelity => (itr -> DataFrames.quantile(itr, 0.25)) => :q1,
        :fidelity => (itr -> DataFrames.quantile(itr, 0.50)) => :q2,
        :fidelity => (itr -> DataFrames.quantile(itr, 0.75)) => :q3,
        :fidelity => (itr -> DataFrames.quantile(itr, 1.00)) => :q4,
    )
sort!(curves[key2], :numparams)

key3 = Dict(); key3[:nV] = 3; key3[:nH] = 3
curves[key3] = DataFrames.combine(
        DataFrames.groupby(df3, :numparams),
        :fidelity => (itr -> DataFrames.quantile(itr, 0.00)) => :q0,
        :fidelity => (itr -> DataFrames.quantile(itr, 0.25)) => :q1,
        :fidelity => (itr -> DataFrames.quantile(itr, 0.50)) => :q2,
        :fidelity => (itr -> DataFrames.quantile(itr, 0.75)) => :q3,
        :fidelity => (itr -> DataFrames.quantile(itr, 1.00)) => :q4,
    )
sort!(curves[key3], :numparams)

key4 = Dict(); key4[:nV] = 4; key4[:nH] = 4
curves[key4] = DataFrames.combine(
        DataFrames.groupby(df4, :numparams),
        :fidelity => (itr -> DataFrames.quantile(itr, 0.00)) => :q0,
        :fidelity => (itr -> DataFrames.quantile(itr, 0.25)) => :q1,
        :fidelity => (itr -> DataFrames.quantile(itr, 0.50)) => :q2,
        :fidelity => (itr -> DataFrames.quantile(itr, 0.75)) => :q3,
        :fidelity => (itr -> DataFrames.quantile(itr, 1.00)) => :q4,
    )
sort!(curves[key4], :numparams)


function get_args(key)
    args = Dict{Symbol,Any}()

    args[:linewidth] = 2

    args[:linestyle] = :solid

    args[:seriescolor] = ColorSchemes.tab10[key[:nV]]
    args[:seriesalpha] = 0.2 + 0.8 * (key[:nH] / key[:nV])

    args[:label] = all((
        key[:nV] == key[:nH],
    )) ? "n=$(key[:nV])" : false

    return args
end

for (key, curve) in pairs(curves)
    Plots.plot!(plt,
        curve[!,:numparams],
        curve[!,:q2];
        ribbon = (
            curve[!,:q2] .- curve[!,:q3],   # BOTTOM ERROR
            curve[!,:q1] .- curve[!,:q2],   # TOP ERROR
        ),
        get_args(key)...
    )
end

Plots.savefig(plt, "../Gibbs/GibbsADAPT_infidelity_vs_params.pdf")           
