#= Parse score data from one or two steps to try and map out barren plateau scaling. =#

import RenyiADAPT.ThermalStatesExperiment as JOB
import ADAPT
import Serialization
import Statistics

# LOAD DATA
Setup(enum_ψREF, enum_method, nV, nH) = JOB.Params(
    "twolocal", enum_ψREF, "twolocal", enum_method,
    nV, nH, 1, 1,
)

function fill_data!(data, setup)
    name = JOB.name_result(setup)
    file = "$(JOB.TRACE)/$name"
    trace = Serialization.deserialize(file)

    s1 = abs.(trace[:scores][1])
    s2 = abs.(trace[:scores][2])

    data[setup] = (
        u1 = Statistics.mean(s1),
        σ1 = Statistics.std(s1),

        u2 = Statistics.mean(s2),
        σ2 = Statistics.std(s2),
    )

    return data
end

data = Dict()
for enum_ψREF in ("entangled", "zipf_1.0", "zipf_0.1", "zipf_0.01")
for enum_method in ("renyi", "overlap")
for nV in 1:5; for nH in 0:nV
    setup = Setup(enum_ψREF, enum_method, nV, nH)
    fill_data!(data, setup)
end; end; end; end

curves = Dict()
for enum_ψREF in ("entangled", "zipf_1.0", "zipf_0.1", "zipf_0.01")
for enum_method in ("renyi", "overlap")
    setup(n) = Setup(enum_ψREF, enum_method, n, n)
    curves[enum_ψREF, enum_method] = (
        u1=[data[setup(n)].u1 for n in 1:5],
        σ1=[data[setup(n)].σ1 for n in 1:5],
        u2=[data[setup(n)].u2 for n in 1:5],
        σ2=[data[setup(n)].σ2 for n in 1:5],
    )
end; end

# PLOT DATA
import Plots

plt = Plots.plot(;
    xlabel = "# qubits (n=nV=nH)",
    ylabel = "",
    yscale = :log10,
    legend = :bottomleft,
)

for (key, curve) in pairs(curves)
    enum_ψREF, enum_method = key
    label = "$enum_ψREF $enum_method"
    color = convert(Int, hash(enum_ψREF)>>1)
    shape = enum_method == "renyi" ? :circle : :square
    Plots.plot!(plt, curve.u1; label=false, color=color, shape=shape, lw=2, ls=:solid)
    Plots.plot!(plt, curve.σ1; label=false, color=color, shape=shape, lw=2, ls=:dashdot)
    Plots.plot!(plt, curve.u2; label=false, color=color, shape=shape, lw=2, ls=:dashdotdot)
    Plots.plot!(plt, curve.σ2; label=false, color=color, shape=shape, lw=2, ls=:dot)
end

Plots.gui()