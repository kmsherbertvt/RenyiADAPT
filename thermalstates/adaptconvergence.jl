#= Plot pool gradient convergence as a function of % completion.

% completion is a little arbitrary,
    but the goal is to visualize the shape of how gradients decay.

Please note that gradient decay here is a GOOD thing;
    it implies CONVERGENCE, and NOT barren plateaus. ;)

=#

import RenyiADAPT.ThermalStatesExperiment as JOB
import ADAPT

##########################################################################################
#= LOAD ALL DATA =#

import CSV
import DataFrames   # Julia's version of Python's `pandas`

df = DataFrames.DataFrame()
for file in readdir(JOB.METRIC, join=true)
    try
        csv = CSV.File(file)
        csv_df = DataFrames.DataFrame(csv)

        if !("maxpoolgradient" in names(csv_df))
            println(file)
            continue
        end

        append!(df, csv_df)
    catch end
end


# append!(df, dfg) # TODO: Drop in Karunya's data once it includes gradient ∞-norms.


#= TODO: Not all the metrics have the updated maxpoolgradient column.
    Write a specialized script to literally rewrite every existing CSV file. ^_^
    =#

#= TODO: Calculate the % convergence. For each *group*.
   I guess we can't really group on seed.
   We might not want to plot statistics after all, but individual trajectories, possibly.
    =#

##########################################################################################
#= PREPARE FOR PLOTTING =#

import ColorSchemes
import Plots

#= Infidelity vs. ADAPT iteration:

- Fidelity vs numparams.
- Aggregate on enums and nV and nH.
  - Linestyle fixed by enums.
  - Primary color fixed by nV (from rainbow).
  - Alpha fixed by ratio nH/nV.
- Collect quartiles over seed_H, seed_ψ.
- Plot interquartile range of fidelity vs numparams.

=#

pdf = DataFrames.groupby(df, [
    :enum_H,
    :enum_ψREF,
    :enum_pool,
    :enum_method,
    :nV,
    :nH,
])

curves = Dict()
for (key, curve) in pairs(pdf)
    curves[key] = DataFrames.combine(
        DataFrames.groupby(curve, :numparams),
        :numiters,
        :maxpoolgradient => (itr -> DataFrames.quantile(itr, 0.00)) => :q0,
        :maxpoolgradient => (itr -> DataFrames.quantile(itr, 0.25)) => :q1,
        :maxpoolgradient => (itr -> DataFrames.quantile(itr, 0.50)) => :q2,
        :maxpoolgradient => (itr -> DataFrames.quantile(itr, 0.75)) => :q3,
        :maxpoolgradient => (itr -> DataFrames.quantile(itr, 1.00)) => :q4,
    )

    sort!(curves[key], :numparams)
end

curves = sort(curves; by=key->key.nV)

function get_args(key)
    args = Dict{Symbol,Any}()

    args[:linewidth] = 2

    args[:linestyle] = (
        renyi = :solid,
        overlap = :dash,
        gibbs = :dot,
    )[Symbol(key.enum_method)]

    args[:seriescolor] = ColorSchemes.tab10[key.nV]
    args[:seriesalpha] = 0.2 + 0.8 * (key.nH / key.nV)

    args[:label] = all((
        key.enum_method == "renyi",
        key.nV == key.nH,
    )) ? "n=$(key.nV)" : false

    return args
end



##########################################################################################
#= PLOT SOME DATA: Log-y plot with ribbons. =#

include_it(key) = all((
    key.enum_ψREF == "entangled",
    key.nV == key.nH,
    any((
        key.enum_method == "renyi",
        key.enum_method == "overlap",
        key.enum_method == "gibbs",
    )),
    key.nV ≤ 4,
))

function get_args(key)
    args = Dict{Symbol,Any}()

    args[:linewidth] = 3

    # args[:linestyle] = (
    #     renyi = :solid,
    #     overlap = :solid,
    #     gibbs = :solid,
    # )[Symbol(key.enum_method)]

    # args[:shape] = (
    #     renyi = :square,
    #     overlap = :circle,
    #     gibbs = :utriangle,
    # )[Symbol(key.enum_method)]

    args[:seriescolor] = (
        renyi = 1,
        overlap = 2,
        gibbs = 3,
    )[Symbol(key.enum_method)]

    args[:linestyle] = [
        :dot,
        :dashdot,
        :dash,
        :solid,
    ][key.nV]
    args[:seriesalpha] = 0.8

    args[:label] = all((
        # key.enum_method == "renyi",
        key.nV == 4,
    )) ? (
        renyi = "Renyi",
        overlap = "Overlap",
        gibbs = "Gibbs",
    )[Symbol(key.enum_method)] : false

    return args
end

xticks = [1, 10, 100]
for (key, curve) in pairs(curves)
    include_it(key) || continue
    key.enum_method == "renyi" || continue
    push!(xticks, last(curve[!,:numparams]))
end

plt = Plots.plot(;
    xlabel = "ADAPT Iterations",
    xscale = :log10,
    xlims = [1, 220],
    xticks = (xticks, map(string, xticks)),
    ylabel = "Pool Gradient ∞-norm",
    ylims = [1e-16, 1e2],
    yscale = :log10,
    yticks = 10.0 .^ (-16:2:2),
    legend = :bottomright,
)

for (key, curve) in pairs(curves)
    include_it(key) || continue

    Plots.plot!(plt,
        curve[!,:numparams],
        1 .- curve[!,:q0];  # Worst-case infidelity from lowest-obtained fidelity.
        get_args(key)...
    )
end

# Dummy curves for linestyle.
Plots.plot!(plt, [0], [-1]; color=:black, lw=3, ls=:dot, label="n=1")
Plots.plot!(plt, [0], [-1]; color=:black, lw=3, ls=:dashdot, label="n=2")
Plots.plot!(plt, [0], [-1]; color=:black, lw=3, ls=:dash, label="n=3")
Plots.plot!(plt, [0], [-1]; color=:black, lw=3, ls=:solid, label="n=4")

Plots.savefig(plt, "thermalstates/adaptconvergence.pdf")