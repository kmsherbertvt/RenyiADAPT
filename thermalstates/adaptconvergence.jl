#= Plot pool gradient convergence as a function of % completion.

% completion is a little arbitrary,
    but the goal is to visualize the shape of how gradients decay.

Please note that gradient decay here is a GOOD thing;
    it implies CONVERGENCE, and NOT barren plateaus. ;)
 
Easier for this plot would be to just use a single trial (ie. seed_H=seed_ψREF=1).
And if we do it here, maybe we'd best do it for the infidelity plot also,
    to be consistent.
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
    catch
    end
end

# ADD IN Karunya's GIBBS-ADAPT PLOTS

for nV in 1:4
    for nH in nV:nV
        for seed in 1:20
            nV >= 4 && seed > 1 && continue            
            filename = "thermalstates/gibbsresults/csv/twolocal.entangled.twolocal.gibbs."*string(nV)*"."*string(nH)*"."*string(seed)*"."*string(seed)*".csv"
            Gibbs_csv = CSV.File(filename)
            append!(df, DataFrames.DataFrame(Gibbs_csv))
        end
    end
end

##########################################################################################
#= PREPARE FOR PLOTTING =#

import ColorSchemes
import Plots

#= Max pool gradient vs. completion:

completion = numparams/max(numparams)

- maxpoolgradient vs completion.
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
    :seed_H,
    :seed_ψ,
])

curves = Dict()
for (key, curve) in pairs(pdf)
    # Skip all dataframes where there's 1 row
    DataFrames.nrow(curve) <= 1 && continue
    curve[end, :maxpoolgradient] > 1e-3 && continue

    # Transform the numparams column into a completion column
    curve = DataFrames.transform(curve, :numparams => (itr -> (itr ./ maximum(itr))) => :completion)
    
    # Create the quartiles and sort by completion
    curves[key] = DataFrames.combine(
        DataFrames.groupby(curve, :completion),
        :completion,
        :maxpoolgradient => (itr -> DataFrames.quantile(itr, 0.00)) => :q0,
        :maxpoolgradient => (itr -> DataFrames.quantile(itr, 0.25)) => :q1,
        :maxpoolgradient => (itr -> DataFrames.quantile(itr, 0.50)) => :q2,
        :maxpoolgradient => (itr -> DataFrames.quantile(itr, 0.75)) => :q3,
        :maxpoolgradient => (itr -> DataFrames.quantile(itr, 1.00)) => :q4,
        # :maxpoolgradient => DataFrames.median,
    )

    sort!(curves[key], :completion)
end

curves = sort(curves; by=key -> key.nV)

function get_args(key)
    args = Dict{Symbol,Any}()

    args[:linewidth] = 2

    color_idx = (
        renyi = 1,
        overlap = 2,
    )[Symbol(key.enum_method)]

    args[:seriescolor] = ColorSchemes.tab10[color_idx]
    args[:seriesalpha] = 0.2 + 0.8 * (key.nH / key.nV)

    args[:label] = "$(key.enum_method)"

    return args
end



# ##########################################################################################
# #= PLOT SOME DATA: Log-y plot with ribbons. =#

# plt = Plots.plot(;
#     xlabel="Completion",
#     ylabel="Gradient ∞-norm",
#     ylims=[1e-7, 1e1],
#     yscale=:log10,
#     yticks=10.0 .^ (-16:2:2),
#     legend=:bottomright,
# )

# for (key, curve) in pairs(curves)
#     key.enum_ψREF == "entangled" || continue
#     key.nV == key.nH || continue
#     key.enum_method == "renyi" || continue
#     key.nV == 3 && key.nH == 3 || continue

#     Plots.plot!(plt,
#         curve[!, :completion],
#         curve[!, :q2];
#         ribbon=(
#             curve[!, :q2] .- curve[!, :q1],   # BOTTOM ERROR
#             curve[!, :q3] .- curve[!, :q2],   # TOP ERROR
#         ),
#         get_args(key)...
#     )
# end
# Plots.savefig(plt, "thermalstates/completion.renyi.pdf")



# plt = Plots.plot(;
#     xlabel="Completion",
#     ylabel="Gradient ∞-norm",
#     ylims=[1e-7, 1e1],
#     yscale=:log10,
#     yticks=10.0 .^ (-16:2:2),
#     legend=:bottomright,
# )

# for (key, curve) in pairs(curves)
#     key.enum_ψREF == "entangled" || continue
#     key.nV == key.nH || continue
#     key.enum_method == "overlap" || continue
#     key.nV == 3 && key.nH == 3 || continue

#     Plots.plot!(plt,
#         curve[!, :completion],
#         curve[!, :q2];
#         ribbon=(
#             curve[!, :q2] .- curve[!, :q1],   # BOTTOM ERROR
#             curve[!, :q3] .- curve[!, :q2],   # TOP ERROR
#         ),
#         get_args(key)...
#     )
# end
# Plots.savefig(plt, "thermalstates/completion.overlap.pdf")


# ##########################################################################################
# #= PLOT SOME DATA: Full log plot sans ribbons. Plot max infidelity rather than median. =#

# include_it(key) = all((
#     key.enum_ψREF == "entangled",
#     key.nV == key.nH,
#     key.enum_method == "renyi" || key.enum_method == "overlap",
#     key.nV == 3 && key.nH == 3,
# ))

# xticks = [1, 10, 100]
# for (key, curve) in pairs(curves)
#     include_it(key) || continue
#     key.enum_method == "renyi" || continue
#     push!(xticks, last(curve[!, :completion]))
# end

# plt = Plots.plot(;
#     xlabel="Completion",
#     # xscale=:log10,
#     xlims=[0, 1],
#     # xticks=(xticks, map(string, xticks)),
#     ylabel="Gradient ∞-norm",
#     ylims=[1e-7, 1e1],
#     yscale=:log10,
#     yticks=10.0 .^ (-16:2:2),
#     legend=:bottomright,
# )

# for (key, curve) in pairs(curves)
#     include_it(key) || continue

#     Plots.plot!(plt,
#         curve[!, :completion],
#         curve[!, :q2];  # Median pool gradient,
#         ribbon=(
#             curve[!, :q2] .- curve[!, :q1],   # BOTTOM ERROR,
#             curve[!, :q3] .- curve[!, :q2],   # TOP ERROR
#         ),
#         get_args(key)...
#     )
# end
# Plots.savefig(plt, "thermalstates/completion.all.pdf")


# #######################################################
# # Sanity check: plot max and min quartiles
# #######################################################
# include_it(key) = all((
#     key.enum_ψREF == "entangled",
#     key.nV == key.nH,
#     key.enum_method == "renyi" || key.enum_method == "overlap",
#     key.nV == 3 && key.nH == 3,
# ))

# xticks = [1, 10, 100]
# for (key, curve) in pairs(curves)
#     include_it(key) || continue
#     key.enum_method == "renyi" || continue
#     push!(xticks, last(curve[!, :completion]))
# end

# plt = Plots.plot(;
#     xlabel="Completion",
#     # xscale=:log10,
#     xlims=[0, 1],
#     # xticks=(xticks, map(string, xticks)),
#     ylabel="Gradient ∞-norm",
#     ylims=[1e-7, 1e1],
#     yscale=:log10,
#     yticks=10.0 .^ (-16:2:2),
#     legend=:bottomright,
# )

# for (key, curve) in pairs(curves)
#     include_it(key) || continue

#     Plots.plot!(plt,
#         curve[!, :completion],
#         curve[!, :q2];  # Median pool gradient,
#         ribbon=(
#             curve[!, :q2] .- curve[!, :q0],   # BOTTOM ERROR,
#             curve[!, :q4] .- curve[!, :q2],   # TOP ERROR
#         ),
#         get_args(key)...
#     )
# end
# Plots.savefig(plt, "thermalstates/completion.allquartiles.pdf")


##########################################################################################
#= Scatter plot each curve, score against % convergence. =#

function get_args(key)
    args = Dict{Symbol,Any}()

    args[:lw] = 1
    args[:msw] = 0
    args[:ms] = 3

    args[:shape] = (
        renyi = :square,
        overlap = :circle,
        gibbs = :utriangle,
    )[Symbol(key.enum_method)]

    color_idx = (
        renyi = 1,
        overlap = 2,
        gibbs = 3,
    )[Symbol(key.enum_method)]

    args[:seriescolor] = ColorSchemes.tab10[color_idx]
    args[:seriesalpha] = 0.8
    args[:linealpha] = 0.4

    args[:label] = key.seed_H == 1 ? (
        renyi = "Renyi",
        overlap = "Overlap",
        gibbs = "Gibbs",
    )[Symbol(key.enum_method)] : false

    return args
end

include_it(key) = all((
    key.enum_ψREF == "entangled",
    key.nV == key.nH,
    any((
        key.enum_method == "renyi",
        key.enum_method == "overlap",
        key.enum_method == "gibbs",
    )),
    key.nV == 3 && key.nH == 3,
))

plt = Plots.plot(;
    xlabel="Completion",
    xlims=[0, 1],
    ylabel="Gradient ∞-norm",
    ylims=[1e-7, 1e1],
    yscale=:log10,
    yticks=10.0 .^ (-16:2:2),
    legend=:bottomright,
)

for (key, curve) in pairs(curves)
    include_it(key) || continue

    Plots.plot!(plt,
        curve[!, :completion],
        curve[!, :q2];  # Median pool gradient. 'course, it is a median of one here...
        get_args(key)...
    )
end
Plots.savefig(plt, "thermalstates/completion.scatter.pdf")



##########################################################################################
#= Scatter plot each curve, score against % convergence. =#

function get_args(key)
    args = Dict{Symbol,Any}()

    args[:lw] = 1
    args[:msw] = 0
    args[:ms] = 3

    args[:shape] = (
        renyi = :square,
        overlap = :circle,
        gibbs = :utriangle,
    )[Symbol(key.enum_method)]

    color_idx = (
        renyi = 1,
        overlap = 2,
        gibbs = 3,
    )[Symbol(key.enum_method)]

    args[:seriescolor] = ColorSchemes.tab10[color_idx]
    args[:seriesalpha] = 0.8
    args[:linealpha] = 0.4

    args[:label] = key.seed_H == 1 ? (
        renyi = "Renyi",
        overlap = "Overlap",
        gibbs = "Gibbs",
    )[Symbol(key.enum_method)] : false

    return args
end

include_it(key) = all((
    key.enum_ψREF == "entangled",
    key.nV == key.nH,
    any((
        key.enum_method == "renyi",
        key.enum_method == "overlap",
        key.enum_method == "gibbs",
    )),
    key.nV == 3 && key.nH == 3,
))

plt = Plots.plot(;
    xlabel="Completion",
    xlims=[0.7, 1.01],
    ylabel="Gradient ∞-norm",
    ylims=[1e-7, 1e1],
    yscale=:log10,
    yticks=10.0 .^ (-16:2:2),
    legend=:bottomleft,
)

for (key, curve) in pairs(curves)
    include_it(key) || continue

    Plots.plot!(plt,
        curve[!, :completion],
        curve[!, :q2];  # Median pool gradient. 'course, it is a median of one here...
        get_args(key)...
    )
end
Plots.savefig(plt, "thermalstates/completion.tighter.pdf")
