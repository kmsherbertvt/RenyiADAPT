#= Sandbox for generating, inspecting, and plotting data from thermalstates experiment. =#

import RenyiADAPT.ThermalStatesExperiment as JOB

""" Convenience constructor for input variables.

Experiment is designed so that,
    if we decide to use a different kind of Hamiltonian or reference vector or pool,
    the data format is still applicable,
    but for now, the first three args are fixed.

Also, while re-using random numbers is not generaly okay,
    since H is generated from randn and ψREF from rand
    (ie. different distributions),
    I think we can get away with using the same seeds for both.


"""
Setup(enum, nV, nH, seed) = JOB.Params(
    "twolocal", "entangled", "twolocal",
    enum,
    nV, nH,
    seed, seed,
)

##########################################################################################
#= GENERATE SOME DATA

The methods are designed so that, if data for a run already exists,
    it will just pick up where it left off,
    meaning nothing expensive happens if you ask to run a job that is already done.
=#

# # RUN ADAPT - do this to generate new data
# for nV in 1:4; for nH in 0:nV; for seed in 2:20
#     setup = Setup("renyi", nV, nH, seed)
#     display(setup)

#     ansatz, trace, adapt, vqe, pool, O, ψREF, callbacks = JOB.get_adapt(setup; run=true)

#     df = JOB.get_dataframe(setup)
#     display(df)

#     println("\n"^3)
# end; end; end

# # CALCULATE METRICS - do this to re-calculate metrics on existing data
# for nV in 1:4; for nH in 0:nV
#     setup = Setup("renyi", nV, nH, 1)
#     JOB.get_dataframe(setup; load=false)
# end; end


##########################################################################################
#= LOAD ALL DATA =#

import CSV
import DataFrames   # Julia's version of Python's `pandas`

df = DataFrames.DataFrame()
for file in readdir(JOB.METRIC, join=true)
    csv = CSV.File(file)
    append!(df, DataFrames.DataFrame(csv))
end

##########################################################################################
#= PLOT SOME DATA =#

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

plt = Plots.plot(;
    xlabel = "ADAPT Iterations",
    ylabel = "Infidelity",
    ylims = [0.0, 1.0],
    legend = :topright,
)

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

for (key, curve) in pairs(curves)
    key.enum_method == "renyi" || continue

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
Plots.savefig(plt, "thermalstates/infidelityvsparameters.pdf")







# Plots.gui()