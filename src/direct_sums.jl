module DirectSums

export direct_sum

using SparseArrays
using SparseArrays: blockdiag
using TestItems

using ..QuantumSystems
using ..Isomorphisms: operator_to_iso_vec, iso_vec_to_operator


"""
    direct_sum(A::AbstractMatrix, B::AbstractMatrix)

Returns the direct sum of two matrices.
"""
function direct_sum(A::AbstractMatrix, B::AbstractMatrix)
    return [A spzeros((size(A, 1), size(B, 2))); spzeros((size(B, 1), size(A, 2))) B]
end

"""
    direct_sum(A::SparseMatrixCSC, B::SparseMatrixCSC)

Returns the direct sum of two sparse matrices.
"""
function direct_sum(A::SparseMatrixCSC, B::SparseMatrixCSC)
    return blockdiag(A, B)
end

"""
    direct_sum(Ã⃗::AbstractVector, B̃⃗::AbstractVector)

Returns the direct sum of two iso_vec operators.
"""
function direct_sum(Ã⃗::AbstractVector, B̃⃗::AbstractVector)
    return operator_to_iso_vec(
        direct_sum(
            iso_vec_to_operator(Ã⃗),
            iso_vec_to_operator(B̃⃗)
        )
    )
end

"""
    direct_sum(sys1::QuantumSystem, sys2::QuantumSystem)

Returns the direct sum of two `QuantumSystem` objects.

Constructs a new system where the Hilbert space is the direct sum of the two input systems:
H = H₁ ⊕ H₂ = [H₁  0 ]
               [0   H₂]

Both systems must have the same number of drives. The resulting system uses sys1's T_max and drive_bounds.

# Example
```julia
sys1 = QuantumSystem([PAULIS[:X]], 10.0, [(-1.0, 1.0)])
sys2 = QuantumSystem([PAULIS[:Y]], 10.0, [(-1.0, 1.0)])
sys_combined = direct_sum(sys1, sys2)
```
"""
function direct_sum(sys1::QuantumSystem, sys2::QuantumSystem)
    @assert sys1.n_drives == sys2.n_drives "System 1 drives ($(sys1.n_drives)) must equal System 2 drives ($(sys2.n_drives))"
    n_drives = sys1.n_drives
    H = (u, t) -> direct_sum(sys1.H(u, t), sys2.H(u, t))
    
    # Get combined drive bounds from both systems - but they should be the same length since n_drives must match
    drive_bounds = sys1.drive_bounds  # They should be the same as sys2.drive_bounds
    
    # Create new QuantumSystem with the Hamiltonian function
    return QuantumSystem(H, sys1.T_max, drive_bounds)
end

direct_sum(systems::AbstractVector{<:QuantumSystem}) = reduce(direct_sum, systems)


# *************************************************************************** #

@testitem "Test matrix direct sum" begin
    using SparseArrays
    A = [1 2; 3 4]
    B = [5 6; 7 8]
    @test direct_sum(A, B) == [1 2 0 0; 3 4 0 0; 0 0 5 6; 0 0 7 8]

    A = sparse([1 2; 3 4])
    B = sparse([5 6; 7 8])
    @test direct_sum(A, B) == sparse([1 2 0 0; 3 4 0 0; 0 0 5 6; 0 0 7 8])
end

@testitem "Test quantum system direct sum" begin
    sys1 = QuantumSystem(ComplexF64[1 2; 3 4], 1.0)
    sys2 = QuantumSystem(ComplexF64[5 6; 7 8], 1.0)
    sys = direct_sum(sys1, sys2)
    @test sys.H(Float64[], 0.0) == ComplexF64[1 2 0 0; 3 4 0 0; 0 0 5 6; 0 0 7 8]
    @test sys.n_drives == 0
end

end
