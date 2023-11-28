# Testing

Here are some instructions for running the unit tests.

## Setup

We need to setup the testing environment

```bash
# Navigate to the test directory
RenyiADAPT $ cd test
RenyiADAPT/test $ julia

# Activate the test environment
julia> ]
(@v1.9) pkg> activate .
(test) pkg> instantiate
```

## Running tests

Navigate to the root directory and open the julia terminal. The `test` command should
automatically handle everything.

```bash
RenyiADAPT $ julia
julia> ]
(@v1.9) pkg> activate .
(RenyiADAPT) pkg> test
```
