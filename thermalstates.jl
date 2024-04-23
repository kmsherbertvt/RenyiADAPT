#= Sandbox for generating, inspecting, and plotting data from thermalstates experiment. =#

import RenyiADAPT.ThermalStatesExperiment as JOB
import ADAPT

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

# ASSIGN MORE CALLBACKS
more_callbacks = ADAPT.AbstractCallback[
    ADAPT.Callbacks.ParameterStopper(0),    # For scaling evidence on barren plateaus.
]


# RUN ADAPT - do this to generate new data
for nV in 7:10; for nH in nV:nV; for seed in 1:1
    setup = JOB.Params("twolocal", "entangled", "twolocal", "renyi", nV, nH, seed, seed)
    display(setup)

    ansatz, trace, adapt, vqe, pool, O, ψREF, callbacks = JOB.get_adapt(setup;
        more_callbacks=more_callbacks,
        run=true,
    )

    df = JOB.get_dataframe(setup)
    display(df)

    println("\n"^3)
end; end; end

# # CALCULATE METRICS - do this to re-calculate metrics on existing data
# for nV in 1:4; for nH in 0:nV
#     setup = JOB.Params("twolocal", "entangled", "twolocal", "renyi", nV, nH, seed, seed)
#     JOB.get_dataframe(setup; load=false)
# end; end


# ##########################################################################################
# #= LOAD ALL DATA =#

# import CSV
# import DataFrames   # Julia's version of Python's `pandas`

# df = DataFrames.DataFrame()
# for file in readdir(JOB.METRIC, join=true)
#     try
#         csv = CSV.File(file)
#         append!(df, DataFrames.DataFrame(csv))
#     catch end
# end

# ##########################################################################################
# #= PLOT SOME DATA =#

# import ColorSchemes
# import Plots

# #= Infidelity vs. ADAPT iteration:

# - Fidelity vs numparams.
# - Aggregate on enums and nV and nH.
#   - Linestyle fixed by enums.
#   - Primary color fixed by nV (from rainbow).
#   - Alpha fixed by ratio nH/nV.
# - Collect quartiles over seed_H, seed_ψ.
# - Plot interquartile range of fidelity vs numparams.

# =#

# pdf = DataFrames.groupby(df, [
#     :enum_H,
#     :enum_ψREF,
#     :enum_pool,
#     :enum_method,
#     :nV,
#     :nH,
# ])

# curves = Dict()
# for (key, curve) in pairs(pdf)
#     curves[key] = DataFrames.combine(
#         DataFrames.groupby(curve, :numparams),
#         :numiters,
#         :fidelity => (itr -> DataFrames.quantile(itr, 0.00)) => :q0,
#         :fidelity => (itr -> DataFrames.quantile(itr, 0.25)) => :q1,
#         :fidelity => (itr -> DataFrames.quantile(itr, 0.50)) => :q2,
#         :fidelity => (itr -> DataFrames.quantile(itr, 0.75)) => :q3,
#         :fidelity => (itr -> DataFrames.quantile(itr, 1.00)) => :q4,
#         # :fidelity => DataFrames.median,
#     )

#     sort!(curves[key], :numparams)
# end

# function get_args(key)
#     args = Dict{Symbol,Any}()

#     args[:linewidth] = 2

#     args[:linestyle] = (
#         renyi = :solid,
#         overlap = :dash,
#     )[Symbol(key.enum_method)]

#     args[:seriescolor] = ColorSchemes.tab10[key.nV]
#     args[:seriesalpha] = 0.2 + 0.8 * (key.nH / key.nV)

#     args[:label] = all((
#         key.enum_method == "renyi",
#         key.nV == key.nH,
#     )) ? "n=$(key.nV)" : false

#     return args
# end

# plt = Plots.plot(;
#     xlabel = "ADAPT Iterations",
#     ylabel = "Infidelity",
#     ylims = [1e-16, 1e2],
#     yscale = :log10,
#     yticks = 10.0 .^ (-16:2:2),
#     legend = :topright,
# )

# for (key, curve) in pairs(curves)
#     key.enum_ψREF == "entangled" || continue
#     # key.nV == key.nH || continue
#     # key.enum_method == "renyi" || continue

#     Plots.plot!(plt,
#         curve[!,:numparams],
#         1 .- curve[!,:q2];
#         ribbon = (
#             curve[!,:q3] .- curve[!,:q2],   # BOTTOM ERROR (backwards 'cause INfidelity)
#             curve[!,:q2] .- curve[!,:q1],   # TOP ERROR (backwards 'cause INfidelity)
#         ),
#         get_args(key)...
#     )
# end
# Plots.savefig(plt, "thermalstates/infidelityvsparameters.log.pdf")


# plt = Plots.plot(;
#     xlabel = "ADAPT Iterations",
#     ylabel = "Infidelity",
#     ylims = [0.0, 1.0],
#     legend = :topright,
# )

# for (key, curve) in pairs(curves)
#     key.enum_ψREF == "entangled" || continue
#     # key.nV == key.nH || continue
#     # key.enum_method == "renyi" || continue

#     Plots.plot!(plt,
#         curve[!,:numparams],
#         1 .- curve[!,:q2];
#         ribbon = (
#             curve[!,:q3] .- curve[!,:q2],   # BOTTOM ERROR (backwards 'cause INfidelity)
#             curve[!,:q2] .- curve[!,:q1],   # TOP ERROR (backwards 'cause INfidelity)
#         ),
#         get_args(key)...
#     )
# end
# # Plots.savefig(plt, "thermalstates/infidelityvsparameters.png")
# Plots.savefig(plt, "thermalstates/infidelityvsparameters.linear.pdf")







# Plots.gui()