#= Set up the Hamiltonian and operator pool for GibbsADAPT. =#

using OpenQuantumTools
import PauliOperators: Pauli, PauliSum, ScaledPauli, ScaledPauliVector, FixedPhasePauli, KetBitString, SparseKetBasis, clip!
using LinearAlgebra

function xyz_model(L::Int,Jx::Float64,Jy::Float64,Jz::Float64,PBCs::Bool)
    """
    Set up an XYZ Hamiltonian on L qubits with Periodic Boundary Conditions = PBCs.
    """      
    H = PauliSum(L)

    for i=1:L-1
        term = PauliSum(L)
        term += Jx*Pauli(L; X=[i,i+1])
        term += Jy*Pauli(L; Y=[i,i+1])        
        term += Jz*Pauli(L; Z=[i,i+1])        
        clip!(term)
        sum!(H,term)
    end
    if PBCs
        term = PauliSum(L)
        term += Jx*Pauli(L; X=[L,1])
        term += Jy*Pauli(L; Y=[L,1])        
        term += Jz*Pauli(L; Z=[L,1])        
        clip!(term)
        sum!(H,term)
    end

    return H
end

function fullpauli(n::Int)
    """
    Returns the pool of (4^n - 1) Pauli strings on n qubits, excluding II...I
    """          
    pool = ScaledPauliVector{n}[]
    for plist in Iterators.product(ntuple(i->["I","X","Y","Z"],n)...)
        pstr = join(plist)
        pauli = [ScaledPauli(Pauli(pstr))]
        push!(pool, pauli)
    end
    pool = pool[2:end] # skip the first entry, which is just "III...."  
    return pool
end

function one_local_pool(n::Int64, axes=["I","X","Y","Z"])
    pool = ScaledPauliVector{n}[]
    for i in 1:n
        "X" in axes && (push!(pool, [ScaledPauli(Pauli(n; X=i))]))
        "Y" in axes && (push!(pool, [ScaledPauli(Pauli(n; Y=i))]))
        "Z" in axes && (push!(pool, [ScaledPauli(Pauli(n; Z=i))]))
    end
    return pool
end

function two_local_pool(n::Int64, axes=["I","X","Y","Z"])
    pool = ScaledPauliVector{n}[]
    for pair in Iterators.product(ntuple(i->1:n, 2)...)
        i,j = pair
        if i < j
            for pair2 in Iterators.product(ntuple(i->axes, 2)...)
                a,b = pair2
                if a == "I" || b == "I"  # to include 1-local strings, use: if a == b == "I"
                    continue
                end
                l = "I"^(i-1)*a*("I"^(j-i-1))*b*"I"^(n-j)
                pauli = [ScaledPauli(Pauli(l))]
                push!(pool, pauli)
            end
        end
    end
    return pool
end

function oneandtwo_local_pool(nV, nH)
    return vcat(
        one_local_pool(nV + nH),
        two_local_pool(nV + nH),
    )
end