
# ----------------------------------------------------------------------------- #
# QuantumSystem
# ----------------------------------------------------------------------------- #

"""
    QuantumSystem <: AbstractQuantumSystem

A struct for storing quantum dynamics.

# Fields
- `H::Function`: The Hamiltonian function: (u, t) -> H(u, t), where u is the control vector and t is time
- `G::Function`: The isomorphic generator function: (u, t) -> G(u, t), including the Hamiltonian mapped to superoperator space
- `H_drift::SparseMatrixCSC{ComplexF64, Int}`: The drift Hamiltonian (time-independent component)
- `H_drives::Vector{SparseMatrixCSC{ComplexF64, Int}}`: The drive Hamiltonians (control-dependent components)
- `T_max::Float64`: Maximum evolution time
- `drive_bounds::Vector{Tuple{Float64, Float64}}`: Drive amplitude bounds for each control (lower, upper)
- `n_drives::Int`: The number of control drives in the system
- `levels::Int`: The number of levels (dimension) in the system

See also [`OpenQuantumSystem`](@ref), [`VariationalQuantumSystem`](@ref).
"""
struct QuantumSystem{F1<:Function, F2<:Function} <: AbstractQuantumSystem
    H::F1
    G::F2
    H_drift::SparseMatrixCSC{ComplexF64, Int}
    H_drives::Vector{SparseMatrixCSC{ComplexF64, Int}}
    T_max::Float64
    drive_bounds::Vector{Tuple{Float64, Float64}}
    n_drives::Int
    levels::Int
end

"""
    QuantumSystem(H::Function, T_max::Float64, drive_bounds::DriveBounds)

Construct a QuantumSystem from a Hamiltonian function.

# Arguments
- `H::Function`: Hamiltonian function with signature (u, t) -> H(u, t) where u is the control vector and t is time
- `T_max::Float64`: Maximum evolution time
- `drive_bounds::DriveBounds`: Drive amplitude bounds for each control. Can be:
  - Tuples `(lower, upper)` for asymmetric bounds
  - Scalars which are interpreted as symmetric bounds `(-value, value)`

# Example
```julia
# Define a time-dependent Hamiltonian
H = (u, t) -> PAULIS[:Z] + u[1] * PAULIS[:X] + u[2] * PAULIS[:Y]
# Using symmetric bounds (scalars)
sys = QuantumSystem(H, 10.0, [1.0, 1.0])
# Equivalent to: [(-1.0, 1.0), (-1.0, 1.0)]
```
"""
function QuantumSystem(
    H::Function,
    T_max::Float64,
    drive_bounds::DriveBounds
)
    drive_bounds = normalize_drive_bounds(drive_bounds)

    # Extract drift by evaluating with zero controls
    H_drift = H(zeros(length(drive_bounds)), 0.0)
    levels = size(H_drift, 1)

    return QuantumSystem(
        H,
        (u, t) -> Isomorphisms.G(H(u, t)),
        sparse(H_drift),
        Vector{SparseMatrixCSC{ComplexF64, Int}}(),  # Empty drives vector for function-based systems
        T_max,
        drive_bounds,
        length(drive_bounds),
        levels
    )
end

"""
    QuantumSystem(
        H_drift::AbstractMatrix{<:Number},
        H_drives::Vector{<:AbstractMatrix{<:Number}},
        T_max::Float64,
        drive_bounds::DriveBounds
    )

Construct a QuantumSystem from drift and drive Hamiltonian terms.

# Arguments
- `H_drift::AbstractMatrix`: The drift (time-independent) Hamiltonian
- `H_drives::Vector{<:AbstractMatrix}`: Vector of drive Hamiltonians, one for each control
- `T_max::Float64`: Maximum evolution time
- `drive_bounds::DriveBounds`: Drive amplitude bounds for each control. Can be:
  - Tuples `(lower, upper)` for asymmetric bounds
  - Scalars which are interpreted as symmetric bounds `(-value, value)`

The resulting Hamiltonian is: H(u, t) = H_drift + Σᵢ uᵢ * H_drives[i]

# Example
```julia
sys = QuantumSystem(
    PAULIS[:Z],                    # drift
    [PAULIS[:X], PAULIS[:Y]],      # drives
    10.0,                          # T_max
    [1.0, 1.0]                     # symmetric bounds: [(-1.0, 1.0), (-1.0, 1.0)]
)
```
"""
function QuantumSystem(
    H_drift::AbstractMatrix{<:Number},
    H_drives::Vector{<:AbstractMatrix{<:Number}},
    T_max::Float64,
    drive_bounds::DriveBounds
)
    drive_bounds = normalize_drive_bounds(drive_bounds)

    H_drift = sparse(H_drift)
    G_drift = sparse(Isomorphisms.G(H_drift))

    n_drives = length(H_drives)
    H_drives = sparse.(H_drives)
    G_drives = sparse.(Isomorphisms.G.(H_drives))

    if n_drives == 0
        H = (u, t) -> H_drift
        G = (u, t) -> G_drift 
    else
        H = (u, t) -> H_drift + sum(u .* H_drives)
        G = (u, t) -> G_drift + sum(u .* G_drives)
    end

    levels = size(H_drift, 1)

    return QuantumSystem(
        H,
        G,
        H_drift,
        H_drives,
        T_max,
        drive_bounds,
        n_drives,
        levels
    )
end

# Convenience constructors
"""
    QuantumSystem(H_drives::Vector{<:AbstractMatrix}, T_max::Float64, drive_bounds::DriveBounds)

Convenience constructor for a system with no drift Hamiltonian (H_drift = 0).

# Arguments
- `H_drives::Vector{<:AbstractMatrix}`: Vector of drive Hamiltonians
- `T_max::Float64`: Maximum evolution time
- `drive_bounds::DriveBounds`: Drive amplitude bounds for each control. Can be:
  - Tuples `(lower, upper)` for asymmetric bounds
  - Scalars which are interpreted as symmetric bounds `(-value, value)`

# Example
```julia
# Using scalars for symmetric bounds
sys = QuantumSystem([PAULIS[:X], PAULIS[:Y]], 10.0, [1.0, 1.0])
# Equivalent to: drive_bounds = [(-1.0, 1.0), (-1.0, 1.0)]
```
"""
function QuantumSystem(H_drives::Vector{<:AbstractMatrix{ℂ}}, T_max::Float64, drive_bounds::DriveBounds) where ℂ <: Number
    @assert !isempty(H_drives) "At least one drive is required"
    return QuantumSystem(spzeros(ℂ, size(H_drives[1])), H_drives, T_max, drive_bounds)
end

"""
    QuantumSystem(H_drift::AbstractMatrix, T_max::Float64)

Convenience constructor for a system with only a drift Hamiltonian (no drives).

# Example
```julia
sys = QuantumSystem(PAULIS[:Z], 10.0)
```
"""
function QuantumSystem(H_drift::AbstractMatrix{ℂ}, T_max::Float64) where ℂ <: Number 
    QuantumSystem(H_drift, Matrix{ℂ}[], T_max, Float64[])
end

# ******************************************************************************* #

@testitem "System creation" begin
    using PiccoloQuantumObjects: PAULIS, QuantumSystem, get_drift, get_drives
    using SparseArrays: sparse
    
    H_drift = PAULIS.Z
    H_drives = [PAULIS.X, PAULIS.Y]
    n_drives = length(H_drives)
    T_max = 1.0
    u_bounds = ones(n_drives)
    
    system = QuantumSystem(H_drift, H_drives, T_max, u_bounds)
    @test system isa QuantumSystem
    @test get_drift(system) == H_drift
    @test get_drives(system) == H_drives

    # repeat with a bigger system
    H_drift = kron(PAULIS.Z, PAULIS.Z)
    H_drives = [kron(PAULIS.X, PAULIS.I), kron(PAULIS.I, PAULIS.X),
                kron(PAULIS.Y, PAULIS.I), kron(PAULIS.I, PAULIS.Y)]
    n_drives = length(H_drives)
    u_bounds = ones(n_drives)

    system = QuantumSystem(H_drift, H_drives, T_max, u_bounds)
    @test system isa QuantumSystem
    @test get_drift(system) == H_drift
    @test get_drives(system) == H_drives
end

@testitem "No drift system creation" begin
    using PiccoloQuantumObjects: PAULIS, QuantumSystem, get_drift, get_drives
    using SparseArrays: spzeros
    
    H_drift = zeros(ComplexF64, 2, 2)
    H_drives = [PAULIS.X, PAULIS.Y]
    T_max = 1.0
    u_bounds = [1.0, 1.0]

    sys1 = QuantumSystem(H_drift, H_drives, T_max, u_bounds)
    sys2 = QuantumSystem(H_drives, T_max, u_bounds)

    @test get_drift(sys1) == get_drift(sys2) == H_drift
    @test get_drives(sys1) == get_drives(sys2) == H_drives
end

@testitem "No drive system creation" begin
    using PiccoloQuantumObjects: PAULIS, QuantumSystem, get_drift, get_drives
    
    H_drift = PAULIS.Z
    H_drives = Matrix{ComplexF64}[]
    T_max = 1.0
    u_bounds = Float64[]

    sys1 = QuantumSystem(H_drift, H_drives, T_max, u_bounds)
    sys2 = QuantumSystem(H_drift, T_max)

    @test get_drift(sys1) == get_drift(sys2) == H_drift
    @test get_drives(sys1) == get_drives(sys2) == H_drives
end

@testitem "System creation with Hamiltonian function" begin
    using PiccoloQuantumObjects: PAULIS, QuantumSystem, get_drift, get_drives

    # test one drive

    H_drift = PAULIS.Z
    H_drives = [PAULIS.X]
    
    system = QuantumSystem(
        (a, t) -> H_drift + sum(a .* H_drives), 
        1.0, 
        [1.0]
    )
    @test system isa QuantumSystem
    @test get_drift(system) == H_drift 
    @test get_drives(system) == H_drives

    # test no drift + three drives

    H_drives = [PAULIS.X, PAULIS.Y, PAULIS.Z]
    system = QuantumSystem(
        (a, t) -> sum(a .* H_drives),
        1.0, 
        [1.0, 1.0, 1.0]
    )
    @test system isa QuantumSystem
    @test get_drift(system) == zeros(2, 2)
    @test get_drives(system) == H_drives 
end

@testitem "QuantumSystem drive_bounds conversion" begin
    using PiccoloQuantumObjects: PAULIS, QuantumSystem

    # Test scalar bounds are converted to symmetric tuples
    H_drift = PAULIS.Z
    H_drives = [PAULIS.X, PAULIS.Y]
    T_max = 1.0

    # Test with scalar bounds
    sys_scalar = QuantumSystem(H_drift, H_drives, T_max, [1.0, 1.5])
    @test sys_scalar.drive_bounds == [(-1.0, 1.0), (-1.5, 1.5)]

    # Test with tuple bounds
    sys_tuple = QuantumSystem(H_drift, H_drives, T_max, [(-0.5, 1.0), (-1.5, 0.5)])
    @test sys_tuple.drive_bounds == [(-0.5, 1.0), (-1.5, 0.5)]

    # Test with mixed bounds (scalars and tuples) - requires explicit type annotation
    mixed_bounds = Union{Float64, Tuple{Float64,Float64}}[1.0, (-0.5, 1.5)]
    sys_mixed = QuantumSystem(H_drift, H_drives, T_max, mixed_bounds)
    @test sys_mixed.drive_bounds == [(-1.0, 1.0), (-0.5, 1.5)]

    # Test with function-based Hamiltonian
    H = (u, t) -> H_drift + sum(u .* H_drives)
    sys_func = QuantumSystem(H, T_max, [0.8, 1.2])
    @test sys_func.drive_bounds == [(-0.8, 0.8), (-1.2, 1.2)]
end
