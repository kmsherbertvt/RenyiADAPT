#= This file contains all the methods needed to get density matrices
        to work with "standard" ADAPT objects.

    The idea is that stuff here could be absorbed directly into `ADAPT.jl` some day.

    NOTE from Kyle
    --------------
    The standard way of extending Julia packages is to implement their functions
        but now dispatching on types of your own creation.
    It *looks* like that's what we're doing in this file, but *actually*
        the `DensityMatrix` type is just an alias of Julia's own `Matrix`.

    Because of that, this file is implicitly defining a whole bunch of methods
        for functions in an external package (`ADAPT.jl`),
        dispatched only on external types.

    Julia has a special name for this.

    It's called *type piracy*.

    It is EXTRAORDINARILY dangerous, and VERY LIKELY to BREAK functionality.

    The ONLY reason this file (and ONLY this file) is allowed to do this (by my fiat),
        is because I have every intent of absorbing this file into `ADAPT.jl`
        as soon as I am content with its content.

    Indeed, I am expecting to drop the alias, and just let it be understood
        that plain matrices are to be understood as density matrices.
    But I am not wholly convinced of that just yet;
        see the NOTE in the `DensityMatrix` docstring for more.
=#

"""
    DensityMatrix{F}

Semantic alias for a complex matrix, representing the density matrix of a quantum state.

# Type Parameters
- `F`: a float type, determining precision. Usually `Float64`.

# NOTE from Kyle

When this code is absorbed into the `ADAPT.jl` package,
    I anticipate dropping the alias, working with `Matrix` types explicitly.
I (presently) feel this is a perfectly intuitive identification,
    and making a new type is redundant and spaghettifies code unnecessarily.

To be clear, that means

    function dothisthing(ρ::DensityMatrix) ... end

will (in my long-term plan, most likely) be changed to

    function dothisthing(ρ::Matrix{Complex{<:AbstractFloat}}) ... end

BUT strictly speaking, not just any complex matrix can be a density matrix,
    so it may make more semantic sense to define the density matrix thus:

    struct DensityMatrix{F<:AbstractFloat} matrix::Matrix{Complex{F}} end

or even possibly something more complicated like a pure-state decomposition.

To reiterate, I do not intend to do this.
But I can be fickle, especially when it comes to pedantic semantics. ;)

The value of the alias is that function headers need not change,
    even if the type definition changes.
Moreover, it may be we choose to adopt a more complicated representation
    for this package which I choose not to absorb into `ADAPT.jl`.
That said, (almost) every occurrence of `ρ` would need to be replaced by `ρ.matrix`,
    which is much less concise and is the main reason I don't want to do this.

"""
DensityMatrix{F<:AbstractFloat} = Matrix{Complex{F}}

import ADAPT
import ADAPT.Basics: MyPauliOperators
import ADAPT.Basics.MyPauliOperators: SparseKetBasis
import ADAPT.Basics.MyPauliOperators: AbstractPauli, FixedPhasePauli, ScaledPauliVector, PauliSum
PauliOperators = MyPauliOperators
#= TODO @Kyle:
    Replace `ADAPT.Basics.MyPauliOperators` with `PauliOperators`
    once `ADAPT.Basics.MyPauliOperators` extension is absorbed into `PauliOperators` package.
=#

AnyPauli = Union{AbstractPauli, PauliSum, ScaledPauliVector}

function ADAPT.evolve_state!(
    G::AnyPauli,
    θ::ADAPT.Parameter,
    ρ::DensityMatrix,
)
    #= TODO:
        Let U = exp(-iθG).
        This function mutates ρ ← UρU'.

        Likely wants a different method for each Pauli type.
        Likely wants to use functions directly from `PauliOperators`
            which do not actually exist yet.

        As a matter of style, should return ρ at the end,
            though it is not explictly mandated in the interface.
    =#
end

function ADAPT.evaluate(
    H::AnyPauli,
    ρ::DensityMatrix,
)
    # NOTE: Not necessarily needed for Renyi divergence!
    #= TODO:
        Evaluate Tr(ρH).

        I've not yet thought about how to do this efficiently
            (ie. without converting H to a matrix).
        Possibly `ρ` will need to be diagonalized.
    =#
end

function ADAPT.partial(
    ::Int,
    ::ADAPT.AbstractAnsatz,
    ::AnyPauli,
    ρ::DensityMatrix,
)
    # NOTE: Not necessarily needed for Renyi divergence!
    #= TODO:
        Actually I think the implementation in `ADAPT.jl`
            ought to work for density matrices already,
            and this method can be omitted from this file.
        But someone should double-check me on that.
        And test it, of course...
    =#
end

function ADAPT.calculate_score(
    ::ADAPT.AbstractAnsatz,
    ::ADAPT.AdaptProtocol,
    G::AnyPauli,
    H::AnyPauli,
    ρ0::DensityMatrix,
)
    # NOTE: Not necessarily needed for Renyi divergence!
    #= TODO:
        Evolve ρ ← U ρ0 U'
        Construct [G,H]
        Evaluate Tr(ρ[G,H])
    =#
end
