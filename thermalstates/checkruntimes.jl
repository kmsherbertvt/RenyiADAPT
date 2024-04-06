#= Plot comparing 3/3 runtimes, for renyi, overlap, and overlap w/finite difference.

Comparing renyi and overlap gives an idea of advantages for one method vs another.
Comparing overlap w and w/out finite difference gives an idea of how "deep"
    we can plausibly go with a finite difference.

=#

import RenyiADAPT.ThermalStatesExperiment as JOB

##########################################################################################
#= LOAD 3/3 DATA =#

import CSV
import DataFrames   # Julia's version of Python's `pandas`

prefix = "thermalstates/results/metrics"
suffix = "3.3.1.1.csv"
dfR = DataFrames.DataFrame(CSV.File(
        "$prefix/twolocal.entangled.twolocal.renyi.$suffix"))
dfI = DataFrames.DataFrame(CSV.File(
        "$prefix/twolocal.entangled.twolocal.overlap.$suffix"))
dfIΔ = DataFrames.DataFrame(CSV.File(
        "$prefix/overlap_cfd_p10/twolocal.entangled.twolocal.overlap.$suffix"))

##########################################################################################
#= PLOT SOME DATA =#

import ColorSchemes
import Plots

plt = Plots.plot(;
    xlabel = "ADAPT Iterations",
    ylabel = "Runtime (s)",
    yscale = :log10,
    ylims = [1e-3, 1e3],
    legend = :bottomright,
)
Plots.plot!(plt, dfR[!,:numparams], dfR[!,:runtime];
    lw=2, α=0.5,
    label="Renyi Divergence",
)
Plots.plot!(plt, dfI[!,:numparams], dfI[!,:runtime];
    lw=2, α=0.5,
    label="Overlap",
)
Plots.plot!(plt, dfIΔ[!,:numparams], dfIΔ[!,:runtime];
    lw=2, α=0.5,
    label="Overlap (Finite)",
)
Plots.savefig(plt, "thermalstates/runtime.pdf")







# Plots.gui()