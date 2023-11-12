# Renyi ADAPT

This package is meant to faciliate variational optimization of the Renyi divergence,
    builiding the ansatz with ADAPT.

## Software Framework

This package is largely an extension of `ADAPT.jl`, hosted here: https://github.com/kmsherbertvt/ADAPT.jl

Most of the "library" code that needs to be written will be implementing functions defined in that module.

The `ADAPT.jl` package is organized around the following types:
- `Ansatz`
- `Generator`
- `Observable`
- `QuantumState`
- `AdaptProtocol`
- `OptimizationProtocol`

The first two (`Ansatz` and `Generator`) represent a sequence of `exp(iθH)` rotations,
    where `H` is a Hermitian operator.
The "standard" implementation in `ADAPT.jl` uses Pauli decompositions,
    and that should work just fine for us, so we needn't worry about these.

The latter two (`AdaptProtocol` and `OptimizationProtocol`) are just "enum" classes
    to identify the precise methods by which one grows and then optimizes the ansatz.
The "standard" implementations in `ADAPT.jl` should work fine for our purposes,
    at least until we know things are working and want to get fancy.

The middle two (`Observable` and `QuantumState`) are where most of the work needs to be done.
See the files `src/__density_matrices.jl` and `src/__renyi_divergence.jl`
    for a skeleton of what needs to be done
    (or at least as much of it as my foresight allows).