#= This file is for implementing an `OptimizationProtocol` that needs only `gradient`
    (ie. each partial), never `evaluate`.
=#

import ADAPT

#= NOTE: I call this "GradientDescent" as a place-holder;
    I don't know what optimization protocol to use.
    My sense is that most of the sophisticated gradient-based optimizers
        DO use direct function evaluations,
        at least in the "linesearch" performed in each iteration.
    A pure gradient-descent algorithm shouldn't need to do that,
        though its optimization trajectory is demonstrably worse in my experience.
    But I've not got much background here.
=#
struct GradientDescent <: ADAPT.OptimizationProtocol
    # TODO: Hyperparameters, at minimum convergence thresholds I guess?
end

function optimize!(
    ansatz::ADAPT.AbstractAnsatz,
    trace::ADAPT.Trace,
    VQE::GradientDescent,
    H::ADAPT.Observable,   # NOTE: no reason to make this specific to Renyi divergence
    ψ0::ADAPT.QuantumState,
    callbacks::ADAPT.CallbackList,
)
    #= TODO: low priority, can just use `OptimOptimizer` for simulations

    Implementation Documentation:

    Callbacks must be called in each "iteration".
    The optimization protocol is free to decide what an "iteration" is,
        but it should generally correspond to "any time the ansatz is changed".
    That's not a hard-fast rule, though -
        for example, it doesn't necessarily make sense to call the callbacks
        for each function evaluation in a linesearch.

    Any implementation of this method must be careful to obey the following contract:

    1. In each iteration, update the ansatz parameters and
            do whatever calculations you need to do.
    Fill up a `data` dict with as much information as possible.
    See the `Callbacks` module for some standard choices.

    2. Call each callback in succession, passing it the `data` dict.
    If any callback returns `true`, terminate without calling any more callbacks,
            and discontinue the optimization.

    3. After calling all callbacks, check if the ansatz has been flagged as optimized.
    If so, discontinue the optimization.

    3. If the optimization protocol terminates successfully without interruption by callbacks,
            call `set_optimized!(ansatz, true)`.
    Be careful to ensure the ansatz parameters actually are the ones found by the optimizer!

    Standard operating procedure is to let callbacks do all the updates to `trace`.
    Thus, implementations of this method should normally ignore `trace` entirely
        (except in passing it along to the callbacks).
    That said, this rule is a "style" guideline, not a contract.

    The return type of this method is intentionally unspecified,
        so that implementations can return something helpful for debugging,
        eg. an `Optim` result object.
    If the callbacks interrupt your optimization,
        it may be worthwhile to check if they flagged the `ansatz` as converged,
        and modify this return object accordingly if possible.
    =#
end