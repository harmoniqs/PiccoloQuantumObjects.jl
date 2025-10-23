using ..Isomorphisms
using SparseArrays: sparse, spzeros

export QuantumSystem
export OpenQuantumSystem
export VariationalQuantumSystem

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
    QuantumSystem(H::Function, T_max::Float64, drive_bounds::Vector{<:Union{Tuple{Float64, Float64}, Float64}})

Construct a QuantumSystem from a Hamiltonian function.

# Arguments
- `H::Function`: Hamiltonian function with signature (u, t) -> H(u, t) where u is the control vector and t is time
- `T_max::Float64`: Maximum evolution time
- `drive_bounds::Vector`: Drive amplitude bounds. Can be tuples `(lower, upper)` or scalars (interpreted as `(-bound, bound)`)

# Example
```julia
# Define a time-dependent Hamiltonian
H = (u, t) -> PAULIS[:Z] + u[1] * PAULIS[:X] + u[2] * PAULIS[:Y]
sys = QuantumSystem(H, 10.0, [(-1.0, 1.0), (-1.0, 1.0)])
```
"""
function QuantumSystem(
    H::Function,
    T_max::Float64,
    drive_bounds::Vector{<:Union{Tuple{Float64, Float64}, Float64}}
)
    drive_bounds = [
        b isa Tuple ? b : (-b, b) for b in drive_bounds
    ]

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
        drive_bounds::Vector{<:Union{Tuple{Float64, Float64}, Float64}}
    )

Construct a QuantumSystem from drift and drive Hamiltonian terms.

# Arguments
- `H_drift::AbstractMatrix`: The drift (time-independent) Hamiltonian
- `H_drives::Vector{<:AbstractMatrix}`: Vector of drive Hamiltonians, one for each control
- `T_max::Float64`: Maximum evolution time
- `drive_bounds::Vector`: Drive amplitude bounds for each control. Can be tuples `(lower, upper)` or scalars

The resulting Hamiltonian is: H(u, t) = H_drift + Σᵢ uᵢ * H_drives[i]

# Example
```julia
sys = QuantumSystem(
    PAULIS[:Z],                    # drift
    [PAULIS[:X], PAULIS[:Y]],      # drives
    10.0,                          # T_max
    [(-1.0, 1.0), (-1.0, 1.0)]     # bounds
)
```
"""
function QuantumSystem(
    H_drift::AbstractMatrix{<:Number},
    H_drives::Vector{<:AbstractMatrix{<:Number}},
    T_max::Float64,
    drive_bounds::Vector{<:Union{Tuple{Float64, Float64}, Float64}}
)
    drive_bounds = [
        b isa Tuple ? b : (-b, b) for b in drive_bounds
    ]

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
    QuantumSystem(H_drives::Vector{<:AbstractMatrix}, T_max::Float64, drive_bounds::Vector)

Convenience constructor for a system with no drift Hamiltonian (H_drift = 0).

# Example
```julia
sys = QuantumSystem([PAULIS[:X], PAULIS[:Y]], 10.0, [1.0, 1.0])
```
"""
function QuantumSystem(H_drives::Vector{<:AbstractMatrix{ℂ}}, T_max::Float64, drive_bounds::Vector{<:Union{Tuple{Float64, Float64}, Float64}}) where ℂ <: Number
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


# ----------------------------------------------------------------------------- #
# OpenQuantumSystem
# ----------------------------------------------------------------------------- #

"""
    OpenQuantumSystem <: AbstractQuantumSystem

A struct for storing open quantum dynamics.

# Fields
- `H::Function`: The Hamiltonian function: (u, t) -> H(u, t)
- `𝒢::Function`: The Lindbladian generator function: u -> 𝒢(u)
- `H_drift::SparseMatrixCSC{ComplexF64, Int}`: The drift Hamiltonian
- `H_drives::Vector{SparseMatrixCSC{ComplexF64, Int}}`: The drive Hamiltonians
- `T_max::Float64`: Maximum evolution time
- `drive_bounds::Vector{Tuple{Float64, Float64}}`: Drive amplitude bounds
- `n_drives::Int`: The number of control drives
- `levels::Int`: The number of levels in the system
- `dissipation_operators::Vector{SparseMatrixCSC{ComplexF64, Int}}`: The dissipation operators
- `params::Dict{Symbol, Any}`: Additional parameters

See also [`QuantumSystem`](@ref).
"""
struct OpenQuantumSystem{F1<:Function, F2<:Function} <: AbstractQuantumSystem
    H::F1
    𝒢::F2
    H_drift::SparseMatrixCSC{ComplexF64, Int}
    H_drives::Vector{SparseMatrixCSC{ComplexF64, Int}}
    T_max::Float64
    drive_bounds::Vector{Tuple{Float64, Float64}}
    n_drives::Int
    levels::Int
    dissipation_operators::Vector{SparseMatrixCSC{ComplexF64, Int}}
    params::Dict{Symbol, Any}
end

"""
    OpenQuantumSystem(
        H_drift::AbstractMatrix{<:Number},
        H_drives::AbstractVector{<:AbstractMatrix{<:Number}},
        T_max::Float64,
        drive_bounds::Vector{<:Union{Tuple{Float64, Float64}, Float64}};
        dissipation_operators::AbstractVector{<:AbstractMatrix{<:Number}}=Matrix{ComplexF64}[],
        params::Dict{Symbol, <:Any}=Dict{Symbol, Any}()
    )
    OpenQuantumSystem(
        H_drift::AbstractMatrix{<:Number}, 
        T_max::Float64, 
        drive_bounds::Vector{<:Union{Tuple{Float64, Float64}, Float64}};
        dissipation_operators::AbstractVector{<:AbstractMatrix{<:Number}}=Matrix{ComplexF64}[],
        params::Dict{Symbol, <:Any}=Dict{Symbol, Any}()
    )
    OpenQuantumSystem(
        H_drives::Vector{<:AbstractMatrix{<:Number}},
        T_max::Float64, 
        drive_bounds::Vector{<:Union{Tuple{Float64, Float64}, Float64}};
        dissipation_operators::AbstractVector{<:AbstractMatrix{<:Number}}=Matrix{ComplexF64}[],
        params::Dict{Symbol, <:Any}=Dict{Symbol, Any}()
    )
    OpenQuantumSystem(
        H::Function, 
        T_max::Float64,
        drive_bounds::Vector{<:Union{Tuple{Float64, Float64}, Float64}};
        dissipation_operators::Vector{<:AbstractMatrix{<:Number}}=Matrix{ComplexF64}[],
        params::Dict{Symbol, <:Any}=Dict{Symbol, Any}()
    )
    OpenQuantumSystem(
        system::QuantumSystem; 
        dissipation_operators::Vector{<:AbstractMatrix{<:Number}}=Matrix{ComplexF64}[],
        params::Dict{Symbol, <:Any}=Dict{Symbol, Any}()
    )

Constructs an OpenQuantumSystem object from the drift and drive Hamiltonian terms and
dissipation operators. All constructors require T_max (maximum time) and drive_bounds
(control bounds for each drive) to be explicitly specified.
"""
function OpenQuantumSystem end

# Main constructor from Hamiltonian components 
function OpenQuantumSystem(
    H_drift::AbstractMatrix{<:Number},
    H_drives::Vector{<:AbstractMatrix{<:Number}},
    T_max::Float64,
    drive_bounds::Vector{<:Union{Tuple{Float64, Float64}, Float64}};
    dissipation_operators::Vector{<:AbstractMatrix{<:Number}}=Matrix{ComplexF64}[],
    params::Dict{Symbol, <:Any}=Dict{Symbol, Any}()
)
    drive_bounds = [
        b isa Tuple ? b : (-b, b) for b in drive_bounds
    ]

    H_drift_sparse = sparse(H_drift)
    𝒢_drift = Isomorphisms.G(Isomorphisms.ad_vec(H_drift_sparse))

    n_drives = length(H_drives)
    H_drives_sparse = sparse.(H_drives)
    𝒢_drives = [Isomorphisms.G(Isomorphisms.ad_vec(H_drive)) for H_drive in H_drives_sparse]
    
    # Build dissipator
    if isempty(dissipation_operators)
        𝒟 = spzeros(size(𝒢_drift))
    else
        𝒟 = sum(Isomorphisms.iso_D(sparse(L)) for L in dissipation_operators)
    end

    if n_drives == 0
        H = (u, t) -> H_drift_sparse
        𝒢 = u -> 𝒢_drift + 𝒟
    else
        H = (u, t) -> H_drift_sparse + sum(u .* H_drives_sparse)
        𝒢 = u -> 𝒢_drift + sum(u .* 𝒢_drives) + 𝒟
    end

    levels = size(H_drift, 1)

    return OpenQuantumSystem(
        H,
        𝒢,
        H_drift_sparse,
        H_drives_sparse,
        T_max,
        drive_bounds,
        n_drives,
        levels,
        sparse.(dissipation_operators),
        params
    )
end

# Convenience constructors
function OpenQuantumSystem(
    H_drives::Vector{<:AbstractMatrix{ℂ}}, 
    T_max::Float64, 
    drive_bounds::Vector{<:Union{Tuple{Float64, Float64}, Float64}};
    dissipation_operators::Vector{<:AbstractMatrix{<:Number}}=Matrix{ComplexF64}[],
    params::Dict{Symbol, <:Any}=Dict{Symbol, Any}()
) where ℂ <: Number
    @assert !isempty(H_drives) "At least one drive is required"
    return OpenQuantumSystem(spzeros(ℂ, size(H_drives[1])), H_drives, T_max, drive_bounds;
                            dissipation_operators=dissipation_operators, params=params)
end

function OpenQuantumSystem(
    H_drift::AbstractMatrix{ℂ}, 
    T_max::Float64; 
    dissipation_operators::Vector{<:AbstractMatrix{<:Number}}=Matrix{ComplexF64}[],
    params::Dict{Symbol, <:Any}=Dict{Symbol, Any}()
) where ℂ <: Number 
    return OpenQuantumSystem(H_drift, Matrix{ℂ}[], T_max, Float64[];
                            dissipation_operators=dissipation_operators, params=params)
end

function OpenQuantumSystem(
    H::F, 
    T_max::Float64,
    drive_bounds::Vector{<:Union{Tuple{Float64, Float64}, Float64}};
    dissipation_operators::Vector{<:AbstractMatrix{ℂ}}=Matrix{ComplexF64}[],
    params::Dict{Symbol, <:Any}=Dict{Symbol, Any}()
) where {F <: Function, ℂ <: Number}
    
    drive_bounds = [
        b isa Tuple ? b : (-b, b) for b in drive_bounds
    ]

    n_drives = length(drive_bounds)
    
    # Extract drift by evaluating with zero controls
    H_drift = H(zeros(n_drives), 0.0)
    levels = size(H_drift, 1)
    
    # Build dissipator
    if isempty(dissipation_operators)
        𝒟 = spzeros(ComplexF64, levels^2, levels^2)
    else
        𝒟 = sum(Isomorphisms.iso_D(sparse(L)) for L in dissipation_operators)
    end

    return OpenQuantumSystem(
        H,
        u -> Isomorphisms.G(Isomorphisms.ad_vec(sparse(H(u, 0.0)))) + 𝒟,
        sparse(H_drift),
        Vector{SparseMatrixCSC{ComplexF64, Int}}(),  # Empty drives vector for function-based systems
        T_max,
        drive_bounds,
        n_drives,
        levels,
        sparse.(dissipation_operators),
        params
    )
end

function OpenQuantumSystem(
    system::QuantumSystem; 
    dissipation_operators::Vector{<:AbstractMatrix{<:Number}}=Matrix{ComplexF64}[],
    params::Dict{Symbol, <:Any}=Dict{Symbol, Any}()
)
    return OpenQuantumSystem(
        system.H_drift, system.H_drives, system.T_max, system.drive_bounds;
        dissipation_operators=dissipation_operators,
        params=params
    )
end

# ----------------------------------------------------------------------------- #
# VariationalQuantumSystem
# ----------------------------------------------------------------------------- #

# TODO: Open quantum systems?

"""
    VariationalQuantumSystem <: AbstractQuantumSystem

A struct for storing variational quantum dynamics, used for sensitivity and robustness analysis.

Variational systems allow exploring how the dynamics change under perturbations to the Hamiltonian.
The variational operators represent directions of uncertainty or perturbation in the system.

# Fields
- `H::Function`: The Hamiltonian function: (u, t) -> H(u, t)
- `G::Function`: The isomorphic generator function: (u, t) -> G(u, t)
- `G_vars::AbstractVector{<:Function}`: Variational generator functions, one for each perturbation direction
- `n_drives::Int`: The number of control drives in the system
- `levels::Int`: The number of levels (dimension) in the system
- `params::Dict{Symbol, Any}`: Additional parameters for the system

See also [`QuantumSystem`](@ref), [`OpenQuantumSystem`](@ref).
"""
struct VariationalQuantumSystem{F1<:Function, F2<:Function, F⃗3<:AbstractVector{<:Function}} <: AbstractQuantumSystem
    H::F1 
    G::F2
    G_vars::F⃗3
    n_drives::Int 
    levels::Int 
    params::Dict{Symbol, Any}
end

"""
    VariationalQuantumSystem(
        H_drift::AbstractMatrix,
        H_drives::AbstractVector{<:AbstractMatrix},
        H_vars::AbstractVector{<:AbstractMatrix};
        params::Dict{Symbol,Any}=Dict{Symbol,Any}()
    )

Construct a VariationalQuantumSystem from drift, drive, and variational Hamiltonian terms.

# Arguments
- `H_drift::AbstractMatrix`: The drift (time-independent) Hamiltonian
- `H_drives::AbstractVector{<:AbstractMatrix}`: Vector of drive Hamiltonians for control
- `H_vars::AbstractVector{<:AbstractMatrix}`: Vector of variational Hamiltonians representing perturbation directions
- `params::Dict{Symbol,Any}`: Optional additional parameters

The variational operators allow sensitivity analysis by exploring how dynamics change
under perturbations: H_perturbed = H + Σᵢ εᵢ * H_vars[i]

# Example
```julia
varsys = VariationalQuantumSystem(
    PAULIS[:Z],                    # drift
    [PAULIS[:X], PAULIS[:Y]],      # drives
    [PAULIS[:X]]                   # variational perturbations
)
```
"""
function VariationalQuantumSystem end

function VariationalQuantumSystem(
    H_drift::AbstractMatrix{<:Number},
    H_drives::AbstractVector{<:AbstractMatrix{<:Number}},
    H_vars::AbstractVector{<:AbstractMatrix{<:Number}};
    params::Dict{Symbol,Any}=Dict{Symbol,Any}()
)
    @assert !isempty(H_vars) "At least one variational operator is required"

    levels = size(H_drift, 1)
    H_drift = sparse(H_drift)
    G_drift = sparse(Isomorphisms.G(H_drift))

    n_drives = length(H_drives)
    H_drives = sparse.(H_drives)
    G_drives = sparse.(Isomorphisms.G.(H_drives))

    G_vars = [a -> Isomorphisms.G(sparse(H)) for H in H_vars]

    if n_drives == 0
        H = a -> H_drift
        G = a -> G_drift
    else
        H = a -> H_drift + sum(a .* H_drives)
        G = a -> G_drift + sum(a .* G_drives)
    end

    return VariationalQuantumSystem(
        H, G, G_vars, n_drives, levels, params
    )
end

function VariationalQuantumSystem(
    H_drives::AbstractVector{<:AbstractMatrix{ℂ}},
    H_vars::AbstractVector{<:AbstractMatrix{<:Number}};
    params::Dict{Symbol, <:Any}=Dict{Symbol, Any}()
) where ℂ <: Number
    @assert !isempty(H_drives) "At least one drive is required"
    @assert !isempty(H_vars) "At least one variational operator is required"
    return VariationalQuantumSystem(
        spzeros(ℂ, size(H_drives[1])), 
        H_drives, 
        H_vars; 
        params=params
    )
end

function VariationalQuantumSystem(
    H::F1,
    H_vars::F⃗2,
    n_drives::Int; 
    params::Dict{Symbol, <:Any}=Dict{Symbol, Any}()
) where {F1 <: Function, F⃗2 <: AbstractVector{<:Function}}
    @assert !isempty(H_vars) "At least one variational operator is required"
    G = a -> Isomorphisms.G(sparse(H(a)))
    G_vars = Function[a -> Isomorphisms.G(sparse(H_v(a))) for H_v in H_vars]
    levels = size(H(zeros(n_drives)), 1)
    return VariationalQuantumSystem(H, G, G_vars, n_drives, levels, params)
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

@testitem "Open system creation" begin
    using PiccoloQuantumObjects: PAULIS, OpenQuantumSystem, get_drift, get_drives, Isomorphisms
    
    H_drift = PAULIS.Z
    # don't want drives == levels
    H_drives = [PAULIS.X]
    dissipation_operators = [PAULIS.Z, PAULIS.X]
    T_max = 1.0
    drive_bounds = [1.0]

    system = OpenQuantumSystem(H_drift, H_drives, T_max, drive_bounds, dissipation_operators=dissipation_operators)
    @test system isa OpenQuantumSystem
    @test get_drift(system) == H_drift
    @test get_drives(system) == H_drives
    @test system.dissipation_operators == dissipation_operators

    # test dissipation
    𝒢_drift = Isomorphisms.G(Isomorphisms.ad_vec(H_drift))
    @test system.𝒢(zeros(system.n_drives)) != 𝒢_drift
end

@testitem "Open system alternate constructors" begin
    using PiccoloQuantumObjects: PAULIS, OpenQuantumSystem, get_drift, get_drives
    
    H_drift = PAULIS.Z
    # don't want drives == levels
    H_drives = [PAULIS.X]
    dissipation_operators = [PAULIS.Z, PAULIS.X]
    T_max = 1.0
    drive_bounds = [1.0]

    system = OpenQuantumSystem(
        H_drift, H_drives, T_max, drive_bounds, dissipation_operators=dissipation_operators
    )
    @test system isa OpenQuantumSystem
    @test get_drift(system) == H_drift
    @test get_drives(system) == H_drives
    @test system.dissipation_operators == dissipation_operators

    # no drift
    system = OpenQuantumSystem(H_drives, T_max, drive_bounds, dissipation_operators=dissipation_operators)
    @test system isa OpenQuantumSystem
    @test get_drift(system) == zeros(size(H_drift))
    @test get_drives(system) == H_drives
    @test system.dissipation_operators == dissipation_operators

    # no drives
    system = OpenQuantumSystem(
        H_drift, T_max, dissipation_operators=dissipation_operators
    )
    @test system isa OpenQuantumSystem
    @test system isa OpenQuantumSystem
    @test get_drift(system) == H_drift
    @test get_drives(system) == []
    @test system.dissipation_operators == dissipation_operators

    # function
    H = (u, t) -> PAULIS.Z + u[1] * PAULIS.X
    system = OpenQuantumSystem(H, T_max, drive_bounds, dissipation_operators=dissipation_operators)
    @test system isa OpenQuantumSystem
    @test get_drift(system) == H_drift
    @test get_drives(system) == H_drives
    @test system.dissipation_operators == dissipation_operators

    # from QuantumSystem
    qsys = QuantumSystem(H_drift, H_drives, T_max, drive_bounds)
    system = OpenQuantumSystem(qsys, dissipation_operators=dissipation_operators)
    @test system isa OpenQuantumSystem
    @test get_drift(system) == H_drift
    @test get_drives(system) == H_drives
    @test system.dissipation_operators == dissipation_operators

end

@testitem "Variational system creation" begin
    using PiccoloQuantumObjects: PAULIS, VariationalQuantumSystem, Isomorphisms
    using LinearAlgebra: I
    
    # default
    varsys1 = VariationalQuantumSystem(
        0.0 * PAULIS.Z,
        [PAULIS.X, PAULIS.Y],
        [PAULIS.X, PAULIS.Y] 
    )

    # no drift
    varsys2 = VariationalQuantumSystem(
        [PAULIS.X, PAULIS.Y],
        [PAULIS.X, PAULIS.Y] 
    )

    a = [1.0; 2.0]
    G_X = Isomorphisms.G(PAULIS.X)
    G_Y = Isomorphisms.G(PAULIS.Y)
    G = a[1] * G_X + a[2] * G_Y
    for varsys in [varsys1, varsys2]
        @test varsys isa VariationalQuantumSystem
        @test varsys.n_drives == 2
        @test length(varsys.G_vars) == 2
        @test varsys.G(a) ≈ G
        @test varsys.G_vars[1](a) ≈ G_X
        @test varsys.G_vars[2](a) ≈ G_Y
    end

    # single sensitivity
    varsys = VariationalQuantumSystem(
        [PAULIS.X, PAULIS.Y],
        [PAULIS.X] 
    )
    @test varsys isa VariationalQuantumSystem
    @test varsys.n_drives == 2
    @test length(varsys.G_vars) == 1
    @test varsys.G(a) ≈ G
    @test varsys.G_vars[1](a) ≈ G_X

    # functional sensitivity
    varsys = VariationalQuantumSystem(
        a -> a[1] * PAULIS.X + a[2] * PAULIS.Y,
        [a -> a[1] * PAULIS.X, a -> PAULIS.Y],
        2
    )
    @test varsys isa VariationalQuantumSystem
    @test varsys.n_drives == 2
    @test length(varsys.G_vars) == 2
    @test varsys.G(a) ≈ G
    @test varsys.G_vars[1](a) ≈ a[1] * G_X
    @test varsys.G_vars[2](a) ≈ G_Y
end