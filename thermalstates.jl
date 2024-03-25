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

# GENERATE SOME DATA
for nV in 4:4; for nH in 0:nV
    setup = Setup("overlap", nV, nH, 1)
    display(setup)

    ansatz, trace, adapt, vqe, pool, O, ψREF, callbacks = JOB.get_adapt(setup; run=true)

    df = JOB.get_dataframe(setup)
    display(df)

    println("\n"^3)
end; end