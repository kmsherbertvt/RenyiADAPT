#= Plot the optimization traces for nV=3, nH=3, seed=1.

Best if we can do this for all three methods side-by-side,
    and also side-by-side with non-adaptive all-66-ops traces.

Maybe use log scale on x-axis to showcase curve shapes better?
Don't worry about the vertical lines,
    but do use markers for each adapt iteration.
    (that way we can do different ADAPT runs side-by-side).

This script will have to serialize the VQE runs for Renyi and Overlap.

BACKUP from when trying to plot all methods on one plot.
That is problematic since it plots different units on the same axis.
Avoid it by just giving each method its own loss plot.

=#

import RenyiADAPT.ThermalStatesExperiment as JOB
import ADAPT

enum_H = "twolocal"
enum_ψREF = "entangled"
enum_pool = "twolocal"
nV = 3
nH = 3
seed_H = 1
seed_ψREF = 1

##########################################################################################
#= LOAD ADAPT RUNS =#

setup_R = JOB.Params(enum_H, enum_ψREF, enum_pool, "renyi", nV, nH, seed_H, seed_ψREF)
ansatz_R, trace_R, adapt, vqe, pool, D, ψREF, callbacks = JOB.get_adapt(
    setup_R;
    run=false,
)

# Most of the objects are redundant, for overlap.
setup_O = JOB.Params(enum_H, enum_ψREF, enum_pool, "overlap", nV, nH, seed_H, seed_ψREF)
ansatz_O, trace_O, _, _, _, iF, _, _ = JOB.get_adapt(setup_O; run=false)

##########################################################################################
#= RUN VQE IF IT HASN'T ALREADY FINISHED =#

jobs = [
    setup_R => D,       # Renyi job
    setup_O => iF,      # Overlap job
]

result = Ref{Any}()     # Save optimization result for debugging purposes.
for (setup, O) in jobs
    name = JOB.name_result(setup)

    # INITIALIZE TRACE AND ANSATZ FILES, IF NECESSARY
    if !all((
        isfile("thermalstates/vqe/ansatze/$name"),
        isfile("thermalstates/vqe/traces/$name"),
    ))
        # INITIALIZE THE ANSATZ TO RUN VQE WITH THE RENYI DIVERGENCE
        ansatz = ADAPT.Ansatz(Float64, pool)
        ADAPT.set_optimized!(ansatz, false)
        for op in pool
            push!(ansatz, op => 0.0)
        end

        # COMMIT FRESH ANSATZ AND TRACE TO FILE
        Serialization.serialize("thermalstates/vqe/ansatze/$name", ansatz)
        Serialization.serialize("thermalstates/vqe/traces/$name", ADAPT.Trace())
    end

    # LOAD ANSATZ AND TRACE
    ansatz = Serialization.deserialize("thermalstates/vqe/ansatze/$name")
    trace  = Serialization.deserialize("thermalstates/vqe/traces/$name")

    # RUN VQE - NOTE: nothing happens if ansatz is flagged as optimized.
    result[] = ADAPT.optimize!(ansatz, trace, vqe, O, ψREF, callbacks)

    # SAVE ANSATZ AND TRACE
    Serialization.serialize("thermalstates/vqe/ansatze/$name", ansatz)
    Serialization.serialize("thermalstates/vqe/traces/$name", trace)
end

##########################################################################################
#= LOAD VQE RUNS =#

name_R = JOB.name_result(setup_R)
ansatz_vR = Serialization.deserialize("thermalstates/vqe/ansatze/$name_R")
trace_vR = Serialization.deserialize("thermalstates/vqe/traces/$name_R")

name_O = JOB.name_result(setup_O)
ansatz_vO = Serialization.deserialize("thermalstates/vqe/ansatze/$name_O")
trace_vO = Serialization.deserialize("thermalstates/vqe/traces/$name_O")

##########################################################################################
#= PLOT THE TRACES!

Each method gets three curves:
- Solid line for ADAPT trace.
- Markers only for ADAPT iterations.
- Dotted line for VQE trace.

=#

import Plots
import ColorSchemes

plt = Plots.plot(;
    xlabel = "BFGS Iterations",
    ylabel = "Loss Function - Min Loss Function",
    ylims = [1e-16, 1e2],
    yscale = :log10,
    yticks = 10.0 .^ (-16:2:2),
    legend = :bottom,
)

# DUMMY CURVE FOR VQE LEGEND
Plots.plot!(plt, [0], [-1]; color=:black, lw=2, ls=:dot, label="VQE")


# PLOT RENYI

E0 = min(
    minimum(trace_R[:energy]),
    minimum(trace_vR[:energy]),
)

color = 1
shape = :square
label = "Renyi"

Plots.plot!(plt,
    trace_R[:iteration],
    trace_R[:energy] .- E0;
    lw=2, ls=:solid, label=false, color=color,
)
Plots.scatter!(plt,
    trace_R[:adaptation][2:end],
    trace_R[:energy][trace_R[:adaptation][2:end]] .- E0;
    label=label, shape=shape, color=color,
)
Plots.plot!(plt,
    trace_vR[:iteration],
    trace_vR[:energy] .- E0;
    lw=2, ls=:dot, label=false, color=color,
)

# PLOT RENYI

E0 = min(
    minimum(trace_O[:energy]),
    minimum(trace_vO[:energy]),
)

color = 2
shape = :circle
label = "Overlap"

Plots.plot!(plt,
    trace_O[:iteration],
    trace_O[:energy] .- E0;
    lw=2, ls=:solid, label=false, color=color,
)
Plots.scatter!(plt,
    trace_O[:adaptation][2:end],
    trace_O[:energy][trace_O[:adaptation][2:end]] .- E0;
    label=label, shape=shape, color=color,
)
Plots.plot!(plt,
    trace_vO[:iteration],
    trace_vO[:energy] .- E0;
    lw=2, ls=:dot, label=false, color=color,
)

Plots.savefig(plt, "thermalstates/optimization.pdf")