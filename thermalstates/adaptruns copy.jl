#= Prepare the publication-quality figures showing each ADAPT run.

Probably we want iF vs. P for each n=nV=nH run,
    with ribbons for spread when available.
Unfortunately n=4 takes too long to have more than one good trial for. :/

Log y makes sense in principle, iff we do omit nH<nV runs.
    But honestly the n=4 case converges much "sooner" than the others
    that it doesn't actually look good.
Log x might make sense iff we really care about showing off nV<4;
    I'm not at all sure we do.

This backup is before Karunya's additions to include Gibbs-ADAPT.

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
        append!(df, DataFrames.DataFrame(csv))
    catch end
end

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
        :fidelity => (itr -> DataFrames.quantile(itr, 0.00)) => :q0,
        :fidelity => (itr -> DataFrames.quantile(itr, 0.25)) => :q1,
        :fidelity => (itr -> DataFrames.quantile(itr, 0.50)) => :q2,
        :fidelity => (itr -> DataFrames.quantile(itr, 0.75)) => :q3,
        :fidelity => (itr -> DataFrames.quantile(itr, 1.00)) => :q4,
        # :fidelity => DataFrames.median,
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

plt = Plots.plot(;
    xlabel = "ADAPT Iterations",
    ylabel = "Infidelity",
    ylims = [1e-16, 1e2],
    yscale = :log10,
    yticks = 10.0 .^ (-16:2:2),
    legend = :bottomright,
)

for (key, curve) in pairs(curves)
    key.enum_ψREF == "entangled" || continue
    key.nV == key.nH || continue
    key.enum_method == "renyi" || continue
    key.nV ≤ 4 || continue

    Plots.plot!(plt,
        curve[!,:numparams],
        1 .- curve[!,:q2];
        ribbon = (
            curve[!,:q3] .- curve[!,:q2],   # BOTTOM ERROR (backwards 'cause INfidelity)
            curve[!,:q2] .- curve[!,:q1],   # TOP ERROR (backwards 'cause INfidelity)
        ),
        get_args(key)...
    )
end
Plots.savefig(plt, "thermalstates/infidelityvsparameters.renyi.pdf")



plt = Plots.plot(;
    xlabel = "ADAPT Iterations",
    ylabel = "Infidelity",
    ylims = [1e-16, 1e2],
    yscale = :log10,
    yticks = 10.0 .^ (-16:2:2),
    legend = :bottomright,
)

for (key, curve) in pairs(curves)
    key.enum_ψREF == "entangled" || continue
    key.nV == key.nH || continue
    key.enum_method == "overlap" || continue
    key.nV ≤ 4 || continue

    Plots.plot!(plt,
        curve[!,:numparams],
        1 .- curve[!,:q2];
        ribbon = (
            curve[!,:q3] .- curve[!,:q2],   # BOTTOM ERROR (backwards 'cause INfidelity)
            curve[!,:q2] .- curve[!,:q1],   # TOP ERROR (backwards 'cause INfidelity)
        ),
        get_args(key)...
    )
end
Plots.savefig(plt, "thermalstates/infidelityvsparameters.overlap.pdf")


##########################################################################################
#= PLOT SOME DATA: Full log plot sans ribbons. Plot max infidelity rather than median. =#

include_it(key) = all((
    key.enum_ψREF == "entangled",
    key.nV == key.nH,
    key.enum_method == "renyi" || key.enum_method == "overlap",
    key.nV ≤ 4,
))

xticks = [1, 10, 100]
for (key, curve) in pairs(curves)
    include_it(key) || continue
    key.enum_method == "renyi" || continue
    push!(xticks, last(curve[!,:numparams]))
end

plt = Plots.plot(;
    xlabel = "ADAPT Iterations",
    xscale = :log10,
    xlims = [1, 200],
    xticks = (xticks, map(string, xticks)),
    ylabel = "Infidelity",
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
Plots.savefig(plt, "thermalstates/infidelityvsparameters.log.pdf")