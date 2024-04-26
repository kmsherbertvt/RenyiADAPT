#= Plot pool gradient convergence as a function of % completion.

% completion is a little arbitrary,
    but the goal is to visualize the shape of how gradients decay.

Please note that gradient decay here is a GOOD thing;
    it implies CONVERGENCE, and NOT barren plateaus. ;)

=#

#= TODO: You'd think we want to do the exact same thing as `adaptruns.jl`,
    but selecting the largest score to plot for each adapt iteration
    rather than the largest infidelity.

Unfortunately, that isn't data I already have in the CSV file,
    so it would be a bit painful to collect those statistics,
    loading each trace individually.
Easiest solution on that front is to redefine `Metrics` to include the largest score
    and re-generate all the csv files from the (existing) traces,
    but that still sounds painful and tedious,
    especially since I'm not sure I have been consistently defining a "zeroth" row.
Runs that I explictly stopped at the first iteration
    report data for the reference state itself,
    but all other runs' first row is after a round of optimization.
Okay, so maybe redefining metrics and generating them consistently is a *good* idea...
    ...but I don't know off-hand if it can be done robustly.

Easier for this plot would be to just use a single trial (ie. seed_H=seed_ψREF=1).
And if we do it here, maybe we'd best do it for the infidelity plot also,
    to be consistent.

=#