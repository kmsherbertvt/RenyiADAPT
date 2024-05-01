#= Plot initial ∞-norm against distance between ρ and σ0.

Unfortunately, even though distance is recorded in the CSV,
    I'm afraid it's inconsistent whether that distance refers to that of σ0,
    or that after a single optimization round.
So I will unfortunately need to explicitly calculate that for each seed index.

=#

import RenyiADAPT
import RenyiADAPT.ThermalStatesExperiment as JOB
import ADAPT
import Serialization
import Statistics
import LinearAlgebra: norm, tr

##########################################################################################
#= LOAD ALL DATA =#

import CSV
import DataFrames   # Julia's version of Python's `pandas`

PATTERN = Regex(join((
    "(twolocal)",                   # enum_H
    "(entangled|zipf_\\d\\.\\d+)",  # enum_ψREF
    "(twolocal)",                   # enum_pool
    "(renyi|overlap)",              # enum_method
    "(\\d+)",                       # nV
    "(\\d+)",                       # nH
    "(\\d+)",                       # seed_H
    "(\\d+)",                       # seed_ψREF
), "\\."))

function parsefile(file)
    re = match(PATTERN, file)
    isnothing(re) && return nothing
    return JOB.Params(
        re[1], re[2], re[3], re[4],
        parse(Int, re[5]), parse(Int, re[6]),
        parse(Int, re[7]), parse(Int, re[8]),
    )
end

df = DataFrames.DataFrame(
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
for file in readdir(JOB.TRACE, join=true)
    setup = parsefile(file)
    isnothing(setup) && continue

    trace = Ref(ADAPT.Trace())
    try
        trace[] = Serialization.deserialize(file)
    catch
        println("Failed to deserialize $file")
        continue
    end

    # FETCH POOL GRADIENT FOR SCORE NORMS
    G  = first(trace[][:scores])

    # FETCH ρ AND σ0 FOR DISTANCE MEASURES
    _, ρ, _, ρs = JOB.get_hamiltonian(setup)
    ψREF = JOB.get_referencevector(setup)
    σ = ψREF * ψREF'
    σ0 = RenyiADAPT.partial_trace(σ, setup.nH)

    push!(df, [
        # INPUT PARAMETERS
        [getfield(setup, field) for field in fieldnames(typeof(setup))]...,
        # SCORE VECTOR NORMS
        count(g -> abs(g) > eps(JOB.Float), G),
        norm(G, 0),
        norm(G, 1),
        norm(G, 2),
        norm(G, Inf),
        # DISTANCE MEASURES
        real( tr(sqrt(ρs*σ0*ρs)) ),
        real( tr(abs.(ρ .- σ0)) / 2 ),
    ])
end

# ADD IN Karunya's DATA
# include("./gibbsGnormdf.jl")
# append!(df, gibbs_df)
filename = "thermalstates/gibbsresults/csv/largest_pool_grad_first_steps.csv"
Gibbs_csv = CSV.File(filename)
append!(df, DataFrames.DataFrame(Gibbs_csv))


# SET A COLUMN EXPLICITLY GIVING THE DISTANCE TO A FULLY CONTROLLABLE SYSTEM
df[!,:Δn] = df[!,:nV] .- df[!,:nH]

##########################################################################################
#= PREPARE FOR PLOTTING =#

import ColorSchemes
import Plots

#= Score vs. distance:

- G∞ vs distance.
- Aggregate on enums and nV and nV-nH.
  - Marker fixed by enums.
  - Primary color fixed by nV.
- Scatter-plot all available data!

=#

pdf = DataFrames.groupby(df, [
    :enum_H,
    :enum_ψREF,
    :enum_pool,
    :enum_method,
    :nV,
    :Δn,
])

# curves = Dict()
# for (key, curve) in pairs(pdf)
#     curves[key] = DataFrames.combine(
#         DataFrames.groupby(curve, :nV),
#         :G∞ => (itr -> DataFrames.quantile(itr, 0.00)) => :q0,
#         :G∞ => (itr -> DataFrames.quantile(itr, 0.25)) => :q1,
#         :G∞ => (itr -> DataFrames.quantile(itr, 0.50)) => :q2,
#         :G∞ => (itr -> DataFrames.quantile(itr, 0.75)) => :q3,
#         :G∞ => (itr -> DataFrames.quantile(itr, 1.00)) => :q4,
#         # TODO: Maybe u and σ also? But I doubt it.
#         :fidelity,
#         :distance,
#     )

#     sort!(curves[key], :nV)
# end

# curves = sort(curves; by=key->key.Δn)

function get_args(key)
    args = Dict{Symbol,Any}()

    # args[:linewidth] = 2

    # args[:linestyle] = (
    #     renyi = :solid,
    #     overlap = :solid,
    #     gibbs = :solid,
    # )[Symbol(key.enum_method)]

    args[:shape] = (
        renyi = :square,
        overlap = :circle,
        gibbs = :utriangle,
    )[Symbol(key.enum_method)]

    args[:seriescolor] = (
        renyi = 1,
        overlap = 2,
        gibbs = 3,
    )[Symbol(key.enum_method)]
    args[:seriesalpha] = 0.2 + 0.8 * (key.nV/4)

    args[:label] = all((
        key.Δn == 0,
        key.nV == 4,
    )) ? (
        renyi = "Renyi",
        overlap = "Overlap",
        gibbs = "Gibbs",
    )[Symbol(key.enum_method)] : false

    return args
end


##########################################################################################
#= PLOT SOME DATA: Scatter G∞ vs trace distance. =#

include_it(key) = all((
    key.enum_ψREF == "entangled",
    key.Δn == 0,
    any((
        key.enum_method == "renyi",
        key.enum_method == "overlap",
        key.enum_method == "gibbs",
    )),
))

plt = Plots.plot(;
    xlabel = "Trace Distance",
    ylabel = "Initial Gradient ∞-norm",
    ylims = [1e-3, 1e1],
    yscale = :log10,
    yticks = 10.0 .^ (-16:1:2),
    legend = :bottomleft,
)

for (key, curve) in pairs(pdf)
    include_it(key) || continue

    Plots.scatter!(plt,
        curve[!,:distance],
        curve[!,:G∞];
        get_args(key)...
    )
end
Plots.savefig(plt, "thermalstates/distance.distance.pdf")




##########################################################################################
#= PLOT SOME DATA: Scatter G∞ vs fidelity. =#

plt = Plots.plot(;
    xlabel = "Fidelity",
    ylabel = "Initial Gradient ∞-norm",
    ylims = [1e-3, 1e1],
    yscale = :log10,
    yticks = 10.0 .^ (-16:1:2),
    legend = :bottomright,
)

for (key, curve) in pairs(pdf)
    include_it(key) || continue

    Plots.scatter!(plt,
        curve[!,:fidelity],
        curve[!,:G∞];
        get_args(key)...
    )
end
Plots.savefig(plt, "thermalstates/distance.fidelity.pdf")