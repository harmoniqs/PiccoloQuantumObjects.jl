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
