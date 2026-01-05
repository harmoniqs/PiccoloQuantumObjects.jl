
# ----------------------------------------------------------------------------- #
# QuantumSystem
# ----------------------------------------------------------------------------- #

"""
    is_hermitian(H::AbstractMatrix; tol=1e-10)

Check if a matrix is Hermitian within a tolerance.
"""
function is_hermitian(H::AbstractMatrix; tol=1e-10)
    return norm(H - H') < tol
end

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
- `time_dependent::Bool`: Whether the Hamiltonian has explicit time dependence beyond control modulation

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
    time_dependent::Bool
end

"""
    QuantumSystem(H::Function, T_max::Float64, drive_bounds::Vector{<:Union{Tuple{Float64, Float64}, Float64}}; time_dependent::Bool=false)

Construct a QuantumSystem from a Hamiltonian function.

# Arguments
- `H::Function`: Hamiltonian function with signature (u, t) -> H(u, t) where u is the control vector and t is time
- `T_max::Float64`: Maximum evolution time
- `drive_bounds::DriveBounds`: Drive amplitude bounds for each control. Can be:
  - Tuples `(lower, upper)` for asymmetric bounds
  - Scalars which are interpreted as symmetric bounds `(-value, value)`

# Keyword Arguments
- `time_dependent::Bool=false`: Set to `true` if the Hamiltonian has explicit time dependence (e.g., cos(ωt) modulation)

# Example
```julia
# Define a time-dependent Hamiltonian
H = (u, t) -> PAULIS[:Z] + u[1] * cos(ω * t) * PAULIS[:X]
sys = QuantumSystem(H, 10.0, [(-1.0, 1.0)]; time_dependent=true)
```
"""
function QuantumSystem(
    H::Function,
    T_max::Float64,
    drive_bounds::Vector{<:Union{Tuple{Float64, Float64}, Float64}};
    time_dependent::Bool=false
)
    drive_bounds = normalize_drive_bounds(drive_bounds)

    n_drives = length(drive_bounds)

    # Extract drift by evaluating with zero controls
    H_drift = H(zeros(n_drives), 0.0)
    levels = size(H_drift, 1)
    
    # Check that H_drift is Hermitian
    @assert is_hermitian(H_drift) "Drift Hamiltonian H(u=0, t=0) is not Hermitian"
    
    # Check that Hamiltonian is Hermitian for sample control values
    u_test = [b isa Tuple ? (b[1] + b[2])/2 : 0.0 for b in drive_bounds]
    H_test = H(u_test, 0.0)
    @assert is_hermitian(H_test) "Hamiltonian H(u, t=0) is not Hermitian for test control values u=$u_test"

    return QuantumSystem(
        H,
        (u, t) -> Isomorphisms.G(H(u, t)),
        sparse(H_drift),
        Vector{SparseMatrixCSC{ComplexF64, Int}}(),  # Empty drives vector for function-based systems
        T_max,
        drive_bounds,
        n_drives,
        levels,
        time_dependent
    )
end

"""
    QuantumSystem(
        H_drift::AbstractMatrix{<:Number},
        H_drives::Vector{<:AbstractMatrix{<:Number}},
        T_max::Float64,
        drive_bounds::Vector{<:Union{Tuple{Float64, Float64}, Float64}};
        time_dependent::Bool=false
    )

Construct a QuantumSystem from drift and drive Hamiltonian terms.

# Arguments
- `H_drift::AbstractMatrix`: The drift (time-independent) Hamiltonian
- `H_drives::Vector{<:AbstractMatrix}`: Vector of drive Hamiltonians, one for each control
- `T_max::Float64`: Maximum evolution time
- `drive_bounds::DriveBounds`: Drive amplitude bounds for each control. Can be:
  - Tuples `(lower, upper)` for asymmetric bounds
  - Scalars which are interpreted as symmetric bounds `(-value, value)`

# Keyword Arguments
- `time_dependent::Bool=false`: Set to `true` if using time-dependent modulation (typically handled at a higher level)

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
    drive_bounds::Vector{<:Union{Tuple{Float64, Float64}, Float64}};
    time_dependent::Bool=false
)
    drive_bounds = [
        b isa Tuple ? b : (-b, b) for b in drive_bounds
    ]
    
    # Check that H_drift is Hermitian
    @assert is_hermitian(H_drift) "Drift Hamiltonian H_drift is not Hermitian"
    
    # Check that all drive Hamiltonians are Hermitian
    for (i, H_drive) in enumerate(H_drives)
        @assert is_hermitian(H_drive) "Drive Hamiltonian H_drives[$i] is not Hermitian"
    end

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
        levels,
        time_dependent
    )
end

# Convenience constructors
"""
    QuantumSystem(H_drives::Vector{<:AbstractMatrix}, T_max::Float64, drive_bounds::Vector; time_dependent::Bool=false)

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
function QuantumSystem(H_drives::Vector{<:AbstractMatrix{ℂ}}, T_max::Float64, drive_bounds::Vector{<:Union{Tuple{Float64, Float64}, Float64}}; time_dependent::Bool=false) where ℂ <: Number
    @assert !isempty(H_drives) "At least one drive is required"
    return QuantumSystem(spzeros(ℂ, size(H_drives[1])), H_drives, T_max, drive_bounds; time_dependent=time_dependent)
end

"""
    QuantumSystem(H_drift::AbstractMatrix, T_max::Float64; time_dependent::Bool=false)

Convenience constructor for a system with only a drift Hamiltonian (no drives).

# Example
```julia
sys = QuantumSystem(PAULIS[:Z], 10.0)
```
"""
function QuantumSystem(H_drift::AbstractMatrix{ℂ}, T_max::Float64; time_dependent::Bool=false) where ℂ <: Number 
    QuantumSystem(H_drift, Matrix{ℂ}[], T_max, Float64[]; time_dependent=time_dependent)
end

# ----------------------------------------------------------------------------- #
# Constructors without T_max (duration lives in Pulse, not System)
# ----------------------------------------------------------------------------- #

"""
    QuantumSystem(H_drift, H_drives, drive_bounds; time_dependent=false)

Construct a QuantumSystem without specifying T_max. Duration is specified by the Pulse.

# Arguments
- `H_drift::AbstractMatrix`: The drift (time-independent) Hamiltonian
- `H_drives::Vector{<:AbstractMatrix}`: Vector of drive Hamiltonians
- `drive_bounds::DriveBounds`: Drive amplitude bounds for each control

# Example
```julia
sys = QuantumSystem(PAULIS[:Z], [PAULIS[:X], PAULIS[:Y]], [1.0, 1.0])
```
"""
function QuantumSystem(
    H_drift::AbstractMatrix{<:Number},
    H_drives::Vector{<:AbstractMatrix{<:Number}},
    drive_bounds::Vector{<:Union{Tuple{Float64, Float64}, Float64}};
    time_dependent::Bool=false
)
    return QuantumSystem(H_drift, H_drives, NaN, drive_bounds; time_dependent=time_dependent)
end

"""
    QuantumSystem(H_drives, drive_bounds; time_dependent=false)

Construct a QuantumSystem with no drift and no T_max.

# Example
```julia
sys = QuantumSystem([PAULIS[:X], PAULIS[:Y]], [1.0, 1.0])
```
"""
function QuantumSystem(
    H_drives::Vector{<:AbstractMatrix{ℂ}}, 
    drive_bounds::Vector{<:Union{Tuple{Float64, Float64}, Float64}}; 
    time_dependent::Bool=false
) where ℂ <: Number
    @assert !isempty(H_drives) "At least one drive is required"
    return QuantumSystem(spzeros(ℂ, size(H_drives[1])), H_drives, NaN, drive_bounds; time_dependent=time_dependent)
end

"""
    QuantumSystem(H_drift; time_dependent=false)

Construct a QuantumSystem with only drift (no drives, no T_max).

# Example
```julia
sys = QuantumSystem(PAULIS[:Z])
```
"""
function QuantumSystem(H_drift::AbstractMatrix{ℂ}; time_dependent::Bool=false) where ℂ <: Number
    return QuantumSystem(H_drift, Matrix{ℂ}[], NaN, Float64[]; time_dependent=time_dependent)
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
        [1.0, 1.0, 1.0],
        time_dependent=false
    )
    @test system isa QuantumSystem
    @test get_drift(system) == zeros(2, 2)
    @test get_drives(system) == H_drives 
end

@testitem "Hermiticity check" begin
    using PiccoloQuantumObjects: PAULIS, QuantumSystem
    using LinearAlgebra: I
    
    # Non-Hermitian drift should fail
    H_drift_bad = [1.0 1.0im; 0.0 1.0]  # Not Hermitian
    @test_throws AssertionError QuantumSystem(H_drift_bad, [PAULIS.X], 1.0, [1.0])
    
    # Non-Hermitian drive should fail  
    H_drive_bad = [1.0 1.0im; 0.0 1.0]  # Not Hermitian
    @test_throws AssertionError QuantumSystem(PAULIS.Z, [H_drive_bad], 1.0, [1.0])
    
    # Hermitian matrices should succeed
    H_drift = PAULIS.Z
    H_drives = [PAULIS.X, PAULIS.Y]
    sys = QuantumSystem(H_drift, H_drives, 1.0, [1.0, 1.0])
    @test sys isa QuantumSystem
    
    # Function-based: non-Hermitian should fail
    H_bad = (u, t) -> [1.0 1.0im; 0.0 1.0]
    @test_throws AssertionError QuantumSystem(H_bad, 1.0, [1.0])
    
    # Function-based: Hermitian should succeed
    H_good = (u, t) -> PAULIS.Z + u[1] * PAULIS.X
    sys2 = QuantumSystem(H_good, 1.0, [1.0])
    @test sys2 isa QuantumSystem
end

@testitem "System creation without T_max" begin
    using PiccoloQuantumObjects: PAULIS, QuantumSystem, get_drift, get_drives
    
    # Test with drift, drives, and bounds (no T_max)
    H_drift = PAULIS.Z
    H_drives = [PAULIS.X, PAULIS.Y]
    u_bounds = [1.0, 1.0]
    
    sys = QuantumSystem(H_drift, H_drives, u_bounds)
    @test sys isa QuantumSystem
    @test get_drift(sys) == H_drift
    @test get_drives(sys) == H_drives
    @test isnan(sys.T_max)
    @test sys.n_drives == 2
    
    # Test with drives only (no drift, no T_max)
    sys2 = QuantumSystem(H_drives, u_bounds)
    @test sys2 isa QuantumSystem
    @test get_drift(sys2) == zeros(ComplexF64, 2, 2)
    @test get_drives(sys2) == H_drives
    @test isnan(sys2.T_max)
    
    # Test with drift only (no drives, no T_max)
    sys3 = QuantumSystem(H_drift)
    @test sys3 isa QuantumSystem
    @test get_drift(sys3) == H_drift
    @test isempty(get_drives(sys3))
    @test isnan(sys3.T_max)
    @test sys3.n_drives == 0
end
