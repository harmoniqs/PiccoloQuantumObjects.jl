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
end

"""
    OpenQuantumSystem(
        H_drift::AbstractMatrix{<:Number},
        H_drives::AbstractVector{<:AbstractMatrix{<:Number}},
        T_max::Float64,
        drive_bounds::Vector{<:Union{Tuple{Float64, Float64}, Float64}};
        dissipation_operators::AbstractVector{<:AbstractMatrix{<:Number}}=Matrix{ComplexF64}[]
    )
    OpenQuantumSystem(
        H_drift::AbstractMatrix{<:Number}, 
        T_max::Float64;
        dissipation_operators::AbstractVector{<:AbstractMatrix{<:Number}}=Matrix{ComplexF64}[]
    )
    OpenQuantumSystem(
        H_drives::Vector{<:AbstractMatrix{<:Number}},
        T_max::Float64, 
        drive_bounds::Vector{<:Union{Tuple{Float64, Float64}, Float64}};
        dissipation_operators::AbstractVector{<:AbstractMatrix{<:Number}}=Matrix{ComplexF64}[]
    )
    OpenQuantumSystem(
        H::Function, 
        T_max::Float64,
        drive_bounds::Vector{<:Union{Tuple{Float64, Float64}, Float64}};
        dissipation_operators::Vector{<:AbstractMatrix{<:Number}}=Matrix{ComplexF64}[]
    )
    OpenQuantumSystem(
        system::QuantumSystem; 
        dissipation_operators::Vector{<:AbstractMatrix{<:Number}}=Matrix{ComplexF64}[]
    )

Constructs an OpenQuantumSystem object from the drift and drive Hamiltonian terms and
dissipation operators. All constructors require T_max (maximum time) and drive_bounds
(control bounds for each drive) to be explicitly specified.
"""
function OpenQuantumSystem(
    H_drift::AbstractMatrix{<:Number},
    H_drives::Vector{<:AbstractMatrix{<:Number}},
    T_max::Float64,
    drive_bounds::Vector{<:Union{Tuple{Float64, Float64}, Float64}};
    dissipation_operators::Vector{<:AbstractMatrix{<:Number}}=Matrix{ComplexF64}[]
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
        sparse.(dissipation_operators)
    )
end

# Convenience constructors
function OpenQuantumSystem(
    H_drives::Vector{<:AbstractMatrix{ℂ}}, 
    T_max::Float64, 
    drive_bounds::Vector{<:Union{Tuple{Float64, Float64}, Float64}};
    dissipation_operators::Vector{<:AbstractMatrix{<:Number}}=Matrix{ComplexF64}[]
) where ℂ <: Number
    @assert !isempty(H_drives) "At least one drive is required"
    return OpenQuantumSystem(spzeros(ℂ, size(H_drives[1])), H_drives, T_max, drive_bounds;
                            dissipation_operators=dissipation_operators)
end

function OpenQuantumSystem(
    H_drift::AbstractMatrix{ℂ}, 
    T_max::Float64; 
    dissipation_operators::Vector{<:AbstractMatrix{<:Number}}=Matrix{ComplexF64}[]
) where ℂ <: Number 
    return OpenQuantumSystem(H_drift, Matrix{ℂ}[], T_max, Float64[];
                            dissipation_operators=dissipation_operators)
end

function OpenQuantumSystem(
    H::F, 
    T_max::Float64,
    drive_bounds::Vector{<:Union{Tuple{Float64, Float64}, Float64}};
    dissipation_operators::Vector{<:AbstractMatrix{ℂ}}=Matrix{ComplexF64}[]
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
        sparse.(dissipation_operators)
    )
end

function OpenQuantumSystem(
    system::QuantumSystem; 
    dissipation_operators::Vector{<:AbstractMatrix{<:Number}}=Matrix{ComplexF64}[]
)
    return OpenQuantumSystem(
        system.H_drift, system.H_drives, system.T_max, system.drive_bounds;
        dissipation_operators=dissipation_operators
    )
end

# ******************************************************************************* #

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
    using PiccoloQuantumObjects: PAULIS, OpenQuantumSystem, QuantumSystem, get_drift, get_drives
    
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
