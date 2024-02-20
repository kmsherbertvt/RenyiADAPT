module RenyiADAPT
    _not_implemented_default = "Method not implemented for these arguments"
    _density_matrix_evolution_error = "Density matrix evolution not implemented"
    NotImplementedError(msg::String=_not_implemented_default) = error(msg)

    include("__density_matrices.jl")
    export DensityMatrix
    export partial_trace, purity, ispure, von_neumann_entropy
    #=
    Jim pointed out that evolving density matrices is a bit redundant
        as long as our reference state is a pure state on visible and hidden units.
    I...am truly puzzled why I hadn't already realized that. ^_^
    Anyway, IGNORE the above file; I'll add its contents to ADAPT.jl some day,
        but we won't need it in this repo right away, if ever.
                                            -Kyle
    =#

    # Implementing evaluation and gradient of the maximal renyi divergence.
    include("__renyi_divergence.jl")
    export MaximalRenyiDivergence

    # Implementing evaluation and gradient of the trace distance.
    include("__trace_distance.jl")
    export TraceDistance

    # Implementing evaluation and gradient of the trace distance.
    include("__infidelity.jl")
    export Infidelity

    # Implementing an optimization that uses only the gradient of the loss function.
    include("__gradient_descent.jl")
    export GradientDescent  # NOTE: the name of the protocol is a placeholder...

end # module RenyiADAPT
