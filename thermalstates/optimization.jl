#= Plot the optimization traces for nV=3, nH=3, seed=1.

Best if we can do this for all three methods side-by-side,
    and also side-by-side with non-adaptive all-66-ops traces.

Maybe use log scale on x-axis to showcase curve shapes better?
Don't worry about the vertical lines,
    but do use markers for each adapt iteration.
    (that way we can do different ADAPT runs side-by-side).

This script will have to serialize the VQE runs for Renyi and Overlap.

=#

import RenyiADAPT.ThermalStatesExperiment as JOB
import ADAPT
import Serialization

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

#= LOAD Karunya's GIBBS-ADAPT DATA =#
    
import DelimitedFiles: readdlm
    
filename1 = "thermalstates/gibbsresults/loss/nV_"*string(nV)*"_nH_"*string(nH)*"_seed_"*string(seed_H)*"_ADAPT_loss.dat"
filename2 = "thermalstates/gibbsresults/loss/nV_"*string(nV)*"_nH_"*string(nH)*"_seed_"*string(seed_H)*"_ADAPT_steps.dat"
filename3 = "thermalstates/gibbsresults/loss/nV_"*string(nV)*"_nH_"*string(nH)*"_seed_"*string(seed_H)*"_VQE_loss.dat"

ADAPTloss_data = readdlm(filename1,Float64)
ADAPTsteps_data = readdlm(filename2,Float64)
VQEloss_data = readdlm(filename3,Float64)

    
##########################################################################################
#= PLOT THE TRACES!

Each method gets three curves:
- Solid line for ADAPT trace.
- Markers only for ADAPT iterations.
- Dotted line for VQE trace.

=#

import Plots
import ColorSchemes


# PLOT RENYI

plt = Plots.plot(;
    xlabel = "BFGS Iterations",
    ylabel = "Loss (Maximal Renyi Divergence)",
    ylims = [1e-16, 1e2],
    yscale = :log10,
    yticks = 10.0 .^ (-16:2:2),
    legend = :topright,
)

E0 = 0  # Set an "exact" energy.

ER = trace_R[:energy]
EvR = trace_vR[:energy]

#= TODO: Awkward. We normalized ρk incorrectly so there is an extra log(K) in here.

ρ = exp(-H) / Tr[exp(-H)]
ρk = inv(ρ) = exp(H) * Tr[exp(-H)]

Unfortunately, what I've done is

my_ρk = exp(H) / Tr[exp(H)]

So my_ρk = ρk / { Tr[exp(-H)] * Tr[exp(H)] } ≡ ρk / K

Thus, D(σ,my_ρk)= log(Tr[σ² my_ρk])
                = log(Tr[σ² ρk / K])
                = log(Tr[σ² ρk] / K)
                = log(Tr[σ² ρk]) - log K
    So the true D = my_D + log K

It's just a constant appearing in the loss function,
    it shouldn't affect the gradient (or therefore the optimization) in the slightest.
Except perhaps edge case numerical effects. :/

=#
H, ρ, ρk, ρs = JOB.get_hamiltonian(setup_R)
Hm = Matrix(H)
expH_ = exp(-Hm)
expH  = exp(Hm)
import LinearAlgebra: tr
K = real(tr(expH_) * tr(expH))
ER = ER .+ log(K)
EvR = EvR .+ log(K)



color = 1
shape = :square

Plots.plot!(plt,
    trace_R[:iteration],
    ER .- E0;
    lw=2, ls=:solid, label="ADAPT Loss", color=color,
)
Plots.scatter!(plt,
    trace_R[:adaptation][2:end],
    ER[trace_R[:adaptation][2:end]] .- E0;
    label="ADAPT Step", shape=shape, color=color,
)
Plots.plot!(plt,
    trace_vR[:iteration],
    EvR .- E0;
    lw=2, ls=:dot, label="VQE Loss", color=color,
)

Plots.savefig(plt, "thermalstates/optimization.renyi.pdf")

# PLOT OVERLAP

plt = Plots.plot(;
    xlabel = "BFGS Iterations",
    ylabel = "Loss (Infidelity)",
    ylims = [1e-16, 1e2],
    yscale = :log10,
    yticks = 10.0 .^ (-16:2:2),
    legend = :topright,
)

E0 = 0  # Set an "exact" energy.

color = 2
shape = :circle

Plots.plot!(plt,
    trace_O[:iteration],
    trace_O[:energy] .- E0;
    lw=2, ls=:solid, label="ADAPT Loss", color=color,
)
Plots.scatter!(plt,
    trace_O[:adaptation][2:end],
    trace_O[:energy][trace_O[:adaptation][2:end]] .- E0;
    label="ADAPT Step", shape=shape, color=color,
)
Plots.plot!(plt,
    trace_vO[:iteration],
    trace_vO[:energy] .- E0;
    lw=2, ls=:dot, label="VQE Loss", color=color,
)

Plots.savefig(plt, "thermalstates/optimization.overlap.pdf")
    
    
# PLOT GIBBS
    
plt = Plots.plot(;
    xlabel = "BFGS Iterations",
    ylabel = "Loss Function - Min Loss Function",
    ylims = [1e-16, 1e2],
    yscale = :log10,
    yticks = 10.0 .^ (-16:2:2),
    legend = :bottom,
)
    
color = 3
shape = :utriangle
label = "Gibbs"

Plots.plot!(plt,
    ADAPTloss_data[:,2] .- ADAPTloss_data[:,1];
    lw=2, ls=:solid, label="ADAPT loss", color=color,
)
Plots.scatter!(plt,
    ADAPTsteps_data[:,1],
    ADAPTsteps_data[:,2];
    label="ADAPT step", shape=shape, color=color,
)
Plots.plot!(plt,
    VQEloss_data[:,2] .- VQEloss_data[:,1];
    lw=2, ls=:dot, label="VQE loss", color=color,
)
Plots.savefig(plt, "thermalstates/optimization.gibbs.pdf")