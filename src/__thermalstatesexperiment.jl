Float = Float64

import LinearAlgebra
import Random

import ADAPT
import RenyiADAPT
import PauliOperators: PauliSum, ScaledPauliVector, Pauli, ScaledPauli, FixedPhasePauli

import NPZ
import Serialization

import CSV
import DataFrames

##########################################################################################
#= Directory names. =#


EXPERIMENT = "thermalstates"

SETUP = "$EXPERIMENT/setup"
RESULTS = "$EXPERIMENT/results"

HAM = "$SETUP/hamiltonians"
RHO = "$SETUP/thermalstates"
INV = "$SETUP/invertedthermalstates"
SRT = "$SETUP/rootedthermalstates"
PSI = "$SETUP/referencevectors"

ANSATZ = "$RESULTS/ansatze"
TRACE = "$RESULTS/traces"
METRIC = "$RESULTS/metrics"

##########################################################################################
#= Type definitions. =#

struct Params
    enum_H::String          # How to generate Hamiltonian.
    enum_ψREF::String       # How to generate reference state.
    enum_pool::String       # Which pool.
    enum_method::String     # What ADAPT method (Renyi, Overlap, etc.).
    nV::Int                 # Size of system register.
    nH::Int                 # Size of ancilla register.
    seed_H::Int             # Which Hamiltonian is generated.
    seed_ψ::Int             # Which reference state is generated.
end

""" A single row of a metric CSV. """
struct Metrics
    numparams::Int          # Number of optimizable parameters in ADAPT ansatz.
    numiters::Int           # Number of BFGS iterations to complete all optimizations.
    runtime::Float          # Number of seconds to complete all optimizations.
    purity::Float           # Purity of system register.
    entropy::Float          # vo Neumann entropy of system register.
    distance::Float         # Distance of system register from target state.
    fidelity::Float         # Fidelity of system register with target state.
    # TODO: Dynamical Lie Rank?
    # TODO: CNOT count?
    # TODO: A more accurate depth metric?
end

""" Get a row of metrics for the a'th adaptation. """
function Metrics(trace, pool, ψREF, nH, ρ, ρs, a)
    # CONSTRUCT AN ANSATZ FOR THE a'th ADAPTATION
    x = trace[:parameters][a,:]     # ROW OF ANGLES
    L = findlast(x .≠ 0)            # NUMBER OF NON-ZERO ANGLES
    isnothing(L) && (L = 0)

    ansatz = ADAPT.Ansatz(eltype(x), pool)
    for i in 1:L
        G = pool[trace[:selected_index][i]]
        θ = x[i]
        push!(ansatz, G => θ)
    end

    # PREPARE THE FINAL STATE
    ψEND = ADAPT.evolve_state(ansatz, ψREF)
    σV = RenyiADAPT.partial_trace(ψEND * ψEND', nH)

    # CALCULATE ALL THE METRICS
    numiters = trace[:adaptation][a]
    if :elapsed_time in keys(trace)
        runtime = sum(trace[:elapsed_time][trace[:adaptation][2:a]], init=0.0)
    else
        # Handle traces which never actually did any optimization...
        runtime = 0.0
    end
    purity = RenyiADAPT.purity(σV)
    entropy = real(RenyiADAPT.von_neumann_entropy(σV))
    distance = LinearAlgebra.tr(abs.(ρ .- σV)) / 2
    fidelity = real(LinearAlgebra.tr(sqrt(ρs*σV*ρs)))

    return Metrics(L, numiters, runtime, purity, entropy, distance, fidelity)
end

##########################################################################################
#= Experiment functions. =#

""" Interact with file system. (Loads H,ρ,ρk if already there, or saves them if not. """
function get_hamiltonian(setup)
    setup.enum_H == "twolocal" || error("Unsupported H enum.")
    name = "$(setup.enum_H).$(setup.nV).$(setup.seed_H)"

    # IF *ALL* OBJECTS EXIST, ASSUME IT IS SAFE TO LOAD
    if all((
        isfile("$HAM/$name"),
        isfile("$RHO/$name.npy"),
        isfile("$INV/$name.npy"),
        isfile("$SRT/$name.npy"),
    ))
        # LOAD HAMILTONIAN
        H_asdict = Serialization.deserialize("$HAM/$name")
        N = length(first(keys(H_asdict)))
        H = PauliSum{N}(
            Dict(FixedPhasePauli(uppercase(label)) => coeff
                for (label, coeff) in H_asdict)
        )

        # LOAD THERMAL STATE AND ITS INVERSE
        ρ  = NPZ.npzread("$RHO/$name.npy")
        ρk = NPZ.npzread("$INV/$name.npy")
        ρs = NPZ.npzread("$SRT/$name.npy")

        return H, ρ, ρk, ρs
    end

    # IF *ANY* OBJECTS ARE MISSING, GENERATE THEM ALL
    Random.seed!(setup.seed_H)
    H, ρ, ρk, ρs = RenyiADAPT.Utils.randomtwolocalhamiltonian(setup.nV)

    # SAVE HAMILTONIAN - serialize a dict for enhanced portability
    H_asdict = Dict(string(pauli) => coeff for (pauli, coeff) in H.ops)
    Serialization.serialize("$HAM/$name", H_asdict)

    # SAVE THERMAL STATE AND ITS INVERSE
    NPZ.npzwrite("$RHO/$name.npy", ρ)
    NPZ.npzwrite("$INV/$name.npy", ρk)
    NPZ.npzwrite("$SRT/$name.npy", ρs)

    return H, ρ, ρk, ρs
end

""" Interact with file system. (Loads ψREF if already there, or saves it if not. """
function get_referencevector(setup::Params)
    name = "$(setup.enum_ψREF).$(setup.nV).$(setup.nH).$(setup.seed_ψ)"

    # IF FILE OBJECT EXISTS, ASSUME IT IS SAFE TO LOAD
    isfile("$PSI/$name.npy") && return NPZ.npzread("$PSI/$name.npy")

    # OTHERWISE, GENERATE IT...
    Random.seed!(setup.seed_ψ)

    if setup.enum_ψREF == "entangled"
        ψREF = RenyiADAPT.Utils.randomentangledvector(setup.nV, setup.nH)
    elseif startswith(setup.enum_ψREF, "zipf_")
        p = parse(Float64, only(match(r"zipf_(.*)", setup.enum_ψREF)))
        ψREF = RenyiADAPT.Utils.zipfvector(setup.nV, setup.nH, p)
    else
        error("Unsupported ψREF enum.")
    end

    # ...AND NOW SAVE IT!
    NPZ.npzwrite("$PSI/$name.npy", ψREF)

    return ψREF
end

""" Construct pool.

Pools are easy to assemble;
    I don't think there is anything to gain from trying to save it.
"""
function get_pool(setup::Params)
    setup.enum_pool == "twolocal" || error("Unsupported pool enum")
    return RenyiADAPT.Utils.oneandtwo_local_pool(setup.nV, setup.nH)
end

""" Make names for the ansatz, trace, and metric csv. """
function name_result(setup::Params)
    return join((getfield(setup, field) for field in fieldnames(typeof(setup))), ".")
end

""" Construct all the objects needed to start ADAPT. """
function init_adapt(setup::Params; more_callbacks=ADAPT.AbstractCallback[])
    name = name_result(setup)

    adapt = ADAPT.VANILLA
    vqe = ADAPT.OptimOptimizer(:BFGS; g_tol=1e-6)
    #= NOTE: If we ever want to change `adapt` or `vqe`,
        even just the convergence criterion of BFGS,
        we will likely need to designate a new `enum_method`.

    Eg. "renyi_loosebfgs" or something.
    And of course this method should dispatch affected variables on `enum_method`
        like it currently does for `observable`. =#

    pool = get_pool(setup)

    H, ρ, ρk, ρs = get_hamiltonian(setup)
    observable = (
        setup.enum_method == "renyi" ?
            RenyiADAPT.MaximalRenyiDivergence(ρk, setup.nH) :
        setup.enum_method == "overlap" ?
            RenyiADAPT.Infidelity(ρs, setup.nH) :
        error("Unsupported method enum")
    )

    reference = get_referencevector(setup)

    callbacks = [
        ADAPT.Callbacks.Tracer(
            :energy, :g_norm, :elapsed_time, :elapsed_f_calls, :elapsed_g_calls,
            :selected_index, :selected_score, :scores,
        ),
        ADAPT.Callbacks.ParameterTracer(),
        # ADAPT.Callbacks.Printer(:energy, :selected_generator, :selected_score),
        RenyiADAPT.Utils.Serializer("$ANSATZ/$name", "$TRACE/$name"),
        ADAPT.Callbacks.ScoreStopper(1e-3),
        ADAPT.Callbacks.ParameterStopper(length(pool)),
        more_callbacks...,
    ]
    #= NOTE: Changing convergence criteria for ADAPT does *not* need a new `enum_method`,
        since it does not change the trajectory.

    We can always just resume an existing run with the revised convergence rules.

    Unless we add something to the trace. That's a pain. Try not to do that please. =#

    ansatz = ADAPT.Ansatz(Float64, pool)
    trace = ADAPT.Trace()

    return ansatz,trace,adapt,vqe,pool,observable,reference,callbacks
end

""" Interact with file system. Run adapt if the existing ansatz isn't converged. """
function get_adapt(setup::Params; more_callbacks=ADAPT.AbstractCallback[], run=false)
    name = name_result(setup)
    ansatz,trace,adapt,vqe,pool,observable,reference,callbacks = init_adapt(
        setup;
        more_callbacks=more_callbacks,
    )

    if all((
        isfile("$ANSATZ/$name"),
        isfile("$TRACE/$name"),
    ))
        ansatz = Serialization.deserialize("$ANSATZ/$name")
        trace = Serialization.deserialize("$TRACE/$name")
    end

    if run
        fin = ADAPT.run!(ansatz,trace,adapt,vqe,pool,observable,reference,callbacks)
        fin || println("""
        --- WARNING! ---

        ADAPT has not (yet) converged.

        Simply rerunning this trial will load the current run and attempt to continue,
            but you should inspect the results for a deeper problem.

        --- -------- ---
        """)
    end

    # Save a checkpoint.
    Serialization.serialize("$ANSATZ/$name", ansatz)
    Serialization.serialize("$TRACE/$name", trace)

    return ansatz,trace,adapt,vqe,pool,observable,reference,callbacks
end

""" Initialize an empty data rame with all column headers in place. """
function init_dataframe()
    nP, tP = fieldnames(Params),  fieldtypes(Params)
    nM, tM = fieldnames(Metrics), fieldtypes(Metrics)
    return DataFrames.DataFrame(
        (string(nP[i]) => tP[i][] for i in eachindex(nP))...,
        (string(nM[i]) => tM[i][] for i in eachindex(nM))...,
    )
end

function get_dataframe(
    setup::Params;
    more_callbacks=ADAPT.AbstractCallback[], load=true, run=false,
)
    name = name_result(setup)

    if load && isfile("$METRIC/$name.csv")
        return DataFrames.DataFrame(CSV.File("$METRIC/$name.csv"))
    end

    # FETCH ALL THE INTERMEDIATE OBJECTS
    H, ρ, ρk, ρs = get_hamiltonian(setup)
    ansatz,trace,adapt,vqe,pool,observable,reference,callbacks = get_adapt(
        setup;
        run=run,
        more_callbacks=more_callbacks,
    )

    # INITIALIZE THE DATA FRAME
    df = init_dataframe()

    # CHECK IF THERE *IS* ANY DATA
    haskey(trace, :adaptation) || return df
    # TODO: Decide if the empty csv should be written. I say no for now.

    # FILL THE DATA FRAME
    for a in eachindex(trace[:adaptation])
        results = Metrics(trace, pool, reference, setup.nH, ρ, ρs, a)   # Not cheap.
        push!(df, [
            (getfield(setup, field) for field in fieldnames(Params))...,
            (getfield(results, field) for field in fieldnames(Metrics))...,
        ])
    end

    # SAVE OUTPUT
    CSV.write("$METRIC/$name.csv", df)

    return df
end