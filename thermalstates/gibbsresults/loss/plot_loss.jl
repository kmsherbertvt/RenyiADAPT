import Plots
import ColorSchemes
using DelimitedFiles
import DataFrames   # Julia's version of Python's `pandas`


plt = Plots.plot(;
    xlabel = "BFGS Iterations",
    ylabel = "Loss Function - Min Loss Function",
    ylims = [1e-16, 1e2],
    yscale = :log10,
    yticks = 10.0 .^ (-16:2:2),
    legend = :bottom,
)

nV = 3; nH = 3; seed = 1

filename1 = "../Gibbs/gibbsresults/loss/nV_"*string(nV)*"_nH_"*string(nH)*"_seed_"*string(seed)*"_ADAPT_loss.dat"
filename2 = "../Gibbs/gibbsresults/loss/nV_"*string(nV)*"_nH_"*string(nH)*"_seed_"*string(seed)*"_ADAPT_steps.dat"
filename3 = "../Gibbs/gibbsresults/loss/nV_"*string(nV)*"_nH_"*string(nH)*"_seed_"*string(seed)*"_VQE_loss.dat"

ADAPTloss_data = readdlm(filename1,Float64)
ADAPTsteps_data = readdlm(filename2,Float64)
VQEloss_data = readdlm(filename3,Float64)

# PLOT GIBBS

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

Plots.savefig(plt, "../Gibbs/optimization.gibbs.pdf")
