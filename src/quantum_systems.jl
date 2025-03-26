module QuantumSystems

export AbstractQuantumSystem
export QuantumSystem
export OpenQuantumSystem
export ParameterizedQuantumSystem

export get_drift
export get_drives

using ..Isomorphisms
using ..QuantumObjectUtils

using LinearAlgebra
using SparseArrays
using TestItems
using ForwardDiff

function generator_jacobian(G::Function)
    return function ∂G(a::AbstractVector{Float64})
        ∂G⃗ = ForwardDiff.jacobian(a_ -> vec(G(a_)), a)
        dim = Int(sqrt(size(∂G⃗, 1)))
        return [reshape(∂G⃗ⱼ, dim, dim) for ∂G⃗ⱼ ∈ eachcol(∂G⃗)]
    end
end

# ----------------------------------------------------------------------------- #
# AbstractQuantumSystem
# ----------------------------------------------------------------------------- #

"""
    AbstractQuantumSystem

Abstract type for defining systems.
"""
abstract type AbstractQuantumSystem end

# ----------------------------------------------------------------------------- #
# AbstractQuantumSystem methods
# ----------------------------------------------------------------------------- #

"""
    get_drift(sys::AbstractQuantumSystem)

Returns the drift Hamiltonian of the system.
"""
get_drift(sys::AbstractQuantumSystem) = sys.H(zeros(sys.n_drives))

"""
    get_drives(sys::AbstractQuantumSystem)

Returns the drive Hamiltonians of the system.
"""
function get_drives(sys::AbstractQuantumSystem)
    H_drift = get_drift(sys)
    # Basis vectors for controls will extract drive operators
    return [sys.H(I[1:sys.n_drives, i]) - H_drift for i ∈ 1:sys.n_drives]
end


# ----------------------------------------------------------------------------- #
# QuantumSystem
# ----------------------------------------------------------------------------- #

"""
    QuantumSystem <: AbstractQuantumSystem

A struct for storing quantum dynamics and the appropriate gradients.

# Fields
- `H::Function`: The Hamiltonian function, excluding dissipation: a -> H(a).
- `G::Function`: The isomorphic generator function, including dissipation, a -> G(a).
- `∂G::Function`: The generator jacobian function, a -> ∂G(a).
- `levels::Int`: The number of levels in the system.
- `n_drives::Int`: The number of drives in the system.

# Constructors
- QuantumSystem(H_drift::AbstractMatrix{<:Number}, H_drives::Vector{<:AbstractMatrix{<:Number}}; kwargs...)
- QuantumSystem(H_drift::AbstractMatrix{<:Number}; kwargs...)
- QuantumSystem(H_drives::Vector{<:AbstractMatrix{<:Number}}; kwargs...)
- QuantumSystem(H::Function, n_drives::Int; kwargs...)

"""
struct QuantumSystem <: AbstractQuantumSystem
    H::Function
    G::Function
    ∂G::Function
    n_drives::Int
    levels::Int
    params::Dict{Symbol, Any}

    """
        QuantumSystem(H_drift::Matrix{<:Number}, H_drives::Vector{Matrix{<:Number}}; kwargs...)
        QuantumSystem(H_drift::Matrix{<:Number}; kwargs...)
        QuantumSystem(H_drives::Vector{Matrix{<:Number}}; kwargs...)
        QuantumSystem(H::Function, n_drives::Int; kwargs...)

    Constructs a `QuantumSystem` object from the drift and drive Hamiltonian terms.
    """
    function QuantumSystem end

    function QuantumSystem(
        H_drift::AbstractMatrix{<:Number},
        H_drives::Vector{<:AbstractMatrix{<:Number}};
        params::Dict{Symbol, <:Any}=Dict{Symbol, Any}()
    )
        levels = size(H_drift, 1)
        H_drift = sparse(H_drift)
        G_drift = sparse(Isomorphisms.G(H_drift))

        n_drives = length(H_drives)
        H_drives = sparse.(H_drives)
        G_drives = sparse.(Isomorphisms.G.(H_drives))

        if n_drives == 0
            H = a -> H_drift
            G = a -> G_drift
            ∂G = a -> 0
        else
            H = a -> H_drift + sum(a .* H_drives)
            G = a -> G_drift + sum(a .* G_drives)
            ∂G = a -> G_drives
        end

        return new(
            H,
            G,
            ∂G,
            n_drives,
            levels,
            params
        )
    end

    function QuantumSystem(H_drives::Vector{<:AbstractMatrix{ℂ}}; kwargs...) where ℂ <: Number
        @assert !isempty(H_drives) "At least one drive is required"
        return QuantumSystem(spzeros(ℂ, size(H_drives[1])), H_drives; kwargs...)
    end

    QuantumSystem(H_drift::AbstractMatrix{ℂ}; kwargs...) where ℂ <: Number =
        QuantumSystem(H_drift, Matrix{ℂ}[]; kwargs...)

    function QuantumSystem(
        H::Function, 
        n_drives::Int; 
        params::Dict{Symbol, <:Any}=Dict{Symbol, Any}()
    )
        G = a -> Isomorphisms.G(sparse(H(a)))
        ∂G = generator_jacobian(G)
        levels = size(H(zeros(n_drives)), 1)
        return new(H, G, ∂G, n_drives, levels, params)
    end

end

# ----------------------------------------------------------------------------- #
# OpenQuantumSystem
# ----------------------------------------------------------------------------- #

"""
    OpenQuantumSystem <: AbstractQuantumSystem

A struct for storing open quantum dynamics and the appropriate gradients.

# Additional fields
- `dissipation_operators::Vector{AbstractMatrix}`: The dissipation operators.

See also [`QuantumSystem`](@ref).

# Constructors
- OpenQuantumSystem(
        H_drift::AbstractMatrix{<:Number},
        H_drives::AbstractVector{<:AbstractMatrix{<:Number}}
        dissipation_operators::AbstractVector{<:AbstractMatrix{<:Number}};
        kwargs...
    )
- OpenQuantumSystem(
        H_drift::Matrix{<:Number}, H_drives::AbstractVector{Matrix{<:Number}};
        dissipation_operators::AbstractVector{<:AbstractMatrix{<:Number}}=Matrix{ComplexF64}[],
        kwargs...
    )
- OpenQuantumSystem(H_drift::Matrix{<:Number}; kwargs...)
- OpenQuantumSystem(H_drives::Vector{Matrix{<:Number}}; kwargs...)
- OpenQuantumSystem(H::Function, n_drives::Int; kwargs...)

"""
struct OpenQuantumSystem <: AbstractQuantumSystem
    H::Function
    𝒢::Function
    ∂𝒢::Function
    n_drives::Int
    levels::Int
    dissipation_operators::Vector{Matrix{ComplexF64}}
    params::Dict{Symbol, Any}

    """
    OpenQuantumSystem(
        H_drift::AbstractMatrix{<:Number},
        H_drives::AbstractVector{<:AbstractMatrix{<:Number}}
        dissipation_operators::AbstractVector{<:AbstractMatrix{<:Number}};
        kwargs...
    )
    OpenQuantumSystem(
        H_drift::Matrix{<:Number}, H_drives::AbstractVector{Matrix{<:Number}};
        dissipation_operators::AbstractVector{<:AbstractMatrix{<:Number}}=Matrix{ComplexF64}[],
        kwargs...
    )
    OpenQuantumSystem(H_drift::Matrix{<:Number}; kwargs...)
    OpenQuantumSystem(H_drives::Vector{Matrix{<:Number}}; kwargs...)
    OpenQuantumSystem(H::Function, n_drives::Int; kwargs...)

    Constructs an `OpenQuantumSystem` object from the drift and drive Hamiltonian terms and
    dissipation operators.
    """
    function OpenQuantumSystem end

    function OpenQuantumSystem(
        H_drift::AbstractMatrix{<:Number},
        H_drives::AbstractVector{<:AbstractMatrix{<:Number}};
        dissipation_operators::AbstractVector{<:AbstractMatrix{<:Number}}=Matrix{ComplexF64}[],
        params::Dict{Symbol, <:Any}=Dict{Symbol, Any}()
    )
        levels = size(H_drift, 1)
        H_drift = sparse(H_drift)
        𝒢_drift = Isomorphisms.G(Isomorphisms.ad_vec(H_drift))

        n_drives = length(H_drives)
        H_drives = sparse.(H_drives)
        𝒢_drives = Isomorphisms.G.(Isomorphisms.ad_vec.(H_drives))

        if isempty(dissipation_operators)
            𝒟 = zeros(size(𝒢_drift))
        else
            𝒟 = sum(Isomorphisms.iso_D(L) for L ∈ sparse.(dissipation_operators))
        end

        if n_drives == 0
            H = a -> H_drift
            𝒢 = a -> 𝒢_drift + 𝒟
            ∂𝒢 = a -> 0
        else
            H = a -> H_drift + sum(a .* H_drives)
            𝒢 = a -> 𝒢_drift + sum(a .* 𝒢_drives) + 𝒟
            ∂𝒢 = a -> 𝒢_drives
        end

        return new(
            H,
            𝒢,
            ∂𝒢,
            n_drives,
            levels,
            dissipation_operators,
            params
        )
    end

    function OpenQuantumSystem(
        H_drift::AbstractMatrix{<:Number},
        H_drives::AbstractVector{<:AbstractMatrix{<:Number}},
        dissipation_operators::AbstractVector{<:AbstractMatrix{<:Number}};
        params::Dict{Symbol, <:Any}=Dict{Symbol, Any}()
    )
        return OpenQuantumSystem(
            H_drift, H_drives;
            dissipation_operators=dissipation_operators,
            params=params
        )
    end

    function OpenQuantumSystem(
        H_drives::AbstractVector{<:AbstractMatrix{ℂ}}; kwargs...
    ) where ℂ <: Number
        @assert !isempty(H_drives) "At least one drive is required"
        return OpenQuantumSystem(spzeros(ℂ, size(H_drives[1])), H_drives; kwargs...)
    end

    OpenQuantumSystem(H_drift::AbstractMatrix{T}; kwargs...) where T <: Number =
        OpenQuantumSystem(H_drift, Matrix{T}[]; kwargs...)

    function OpenQuantumSystem(
        H::Function, n_drives::Int;
        dissipation_operators::AbstractVector{<:AbstractMatrix{ℂ}}=Matrix{ComplexF64}[],
        params::Dict{Symbol, <:Any}=Dict{Symbol, Any}()
    ) where ℂ <: Number
        G = a -> Isomorphisms.G(Isomorphisms.ad_vec(sparse(H(a))))
        ∂G = generator_jacobian(G)
        levels = size(H(zeros(n_drives)), 1)
        return new(H, G, ∂G, n_drives, levels, dissipation_operators, params)
    end

    OpenQuantumSystem(system::QuantumSystem; kwargs...) = OpenQuantumSystem(
        system.H, system.n_drives; kwargs...
    )

end

# ****************************************************************************** #

struct ParameterizedQuantumSystem <: AbstractQuantumSystem
    H::Function 
    G::Function
    Gₐ::Vector{Function} 
    ∂G::Function 
    n_drives::Int 
    levels::Int 
    params::Dict{Symbol, Any}

    function ParameterizedQuantumSystem end 

    function ParameterizedQuantumSystem(base_system::QuantumSystem, sensitivity_systems::Vector{QuantumSystem})
        return new(
            base_system.H,
            base_system.G,
            [sys.G for sys ∈ sensitivity_systems],
            base_system.∂G,
            base_system.n_drives,
            base_system.levels,
            base_system.params
        )
    end

    function ParameterizedQuantumSystem(system::ParameterizedQuantumSystem, indices::AbstractVector{Int64})
        return new(
            system.H,
            system.G,
            system.Gₐ[indices],
            system.∂G,
            system.n_drives,
            system.levels,
            system.params
        )
    end

end





# struct ParameterizedQuantumSystem <: AbstractQuantumSystem
#     H::Function 
#     G::Function
#     Gₐ::Function 
#     ∂G::Function 
#     n_drives::Int 
#     levels::Int 
#     params::Dict{Symbol, Any}

#     function ParameterizedQuantumSystem end 

#     function ParameterizedQuantumSystem(
#         H_drift::AbstractMatrix{<:Number},
#         H_drives::Vector{<:AbstractMatrix{<:Number}},
#         Hₐ_drives::Vector{<:AbstractMatrix{<:Number}};
#         params::Dict{Symbol, <:Any}=Dict{Symbol, Any}()
#     )
#         levels = size(H_drift, 1)
#         H_drift = sparse(H_drift)
#         G_drift = sparse(Isomorphisms.G(H_drift))

#         n_drives = length(H_drives)
#         H_drives = sparse.(H_drives)
#         G_drives = sparse.(Isomorphisms.G.(H_drives))

#         Gₐ_drives = sparse.(Isomorphisms.G.(Hₐ_drives))
        
#         if n_drives == 0
#             H = a -> H_drift
#             G = a -> G_drift
#             Gₐ = a -> 0
#             ∂G = a -> 0
#         else
#             H = a -> H_drift + sum(a .* H_drives)
#             G = a -> G_drift + sum(a .* G_drives)
#             Gₐ = a -> sum(a .* Gₐ_drives)
#             ∂G = a -> G_drives
#         end

#         return new(
#             H,
#             G,
#             Gₐ,
#             ∂G,
#             n_drives,
#             levels,
#             params
#         )
#     end

#     function ParameterizedQuantumSystem(
#         H_drift::AbstractMatrix{<:Number},
#         H_drives::Vector{<:AbstractMatrix{<:Number}},
#         Hₐ_drift::AbstractMatrix{<:Number};
#         params::Dict{Symbol, <:Any}=Dict{Symbol, Any}()
#     )
#         levels = size(H_drift, 1)
#         H_drift = sparse(H_drift)
#         G_drift = sparse(Isomorphisms.G(H_drift))

#         n_drives = length(H_drives)
#         H_drives = sparse.(H_drives)
#         G_drives = sparse.(Isomorphisms.G.(H_drives))

#         Gₐ_drift = sparse(Isomorphisms.G(Hₐ_drift))
        
#         if n_drives == 0
#             H = a -> H_drift
#             G = a -> G_drift
#             Gₐ = a -> Gₐ_drift
#             ∂G = a -> 0
#         else
#             H = a -> H_drift + sum(a .* H_drives)
#             G = a -> G_drift + sum(a .* G_drives)
#             Gₐ = a -> Gₐ_drift
#             ∂G = a -> G_drives
#         end

#         return new(
#             H,
#             G,
#             Gₐ,
#             ∂G,
#             n_drives,
#             levels,
#             params
#         )
#     end

#     function ParameterizedQuantumSystem(H_drives::Vector{<:AbstractMatrix{ℂ}},Hₐ_drives::Vector{<:AbstractMatrix{ℂ}}; kwargs...) where ℂ <: Number
#         @assert !isempty(H_drives) "At least one drive is required"
#         @assert length(Hₐ_drives) == length(H_drives) "Size has to be the same"
#         return ParameterizedQuantumSystem(spzeros(ℂ, size(H_drives[1])), H_drives, Hₐ_drives; kwargs...)
#     end
    
#     function ParameterizedQuantumSystem(H_drift::AbstractMatrix{ℂ},Hₐ_drives::Vector{<:AbstractMatrix{ℂ}}; kwargs...) where ℂ <: Number
#         @assert !isempty(H_drives) "At least one drive is required"
#         @assert length(Hₐ_drives) == length(H_drives) "Size has to be the same"
#         return ParameterizedQuantumSystem(H_drift, Matrix{ℂ}[], Hₐ_drives; kwargs...)
#     end

#     function ParameterizedQuantumSystem(H_drives::Vector{<:AbstractMatrix{ℂ}},Hₐ_drift::AbstractMatrix{ℂ}; kwargs...) where ℂ <: Number
#         @assert !isempty(H_drives) "At least one drive is required"
#         @assert length(Hₐ_drives) == length(H_drives) "Size has to be the same"
#         return ParameterizedQuantumSystem(spzeros(ℂ, size(H_drives[1])), H_drives, Hₐ_drift; kwargs...)
#     end
    
#     function ParameterizedQuantumSystem(H_drift::AbstractMatrix{ℂ},Hₐ_drift::AbstractMatrix{ℂ}; kwargs...) where ℂ <: Number
#         @assert !isempty(H_drives) "At least one drive is required"
#         @assert length(Hₐ_drives) == length(H_drives) "Size has to be the same"
#         return ParameterizedQuantumSystem(H_drift, Matrix{ℂ}[], Hₐ_drift; kwargs...)
#     end

#     function ParameterizedQuantumSystem(system::QuantumSystem,Hₐ_drift::AbstractMatrix{ℂ}; kwargs...) where ℂ <: Number
#         return new(
#             system.H,
#             system.G,
#             a -> sparse(Isomorphisms.G(Hₐ_drift)),
#             system.∂G,
#             system.n_drives,
#             system.levels,
#             system.params
#         )

#     end

#     function ParameterizedQuantumSystem(system::QuantumSystem,Hₐ_drives::Vector{<:AbstractMatrix{ℂ}}; kwargs...) where ℂ <: Number
        
#         Gₐ_drives = sparse.(Isomorphisms.G.(Hₐ_drives))
#         return new(
#             system.H,
#             system.G,
#             a -> sum(a .* Gₐ_drives),
#             system.∂G,
#             system.n_drives,
#             system.levels,
#             system.params
#         )

#     end

# end


#***********************************************************************************************#


@testitem "System creation" begin
    H_drift = PAULIS[:Z]
    H_drives = [PAULIS[:X], PAULIS[:Y]]
    n_drives = length(H_drives)

    system = QuantumSystem(H_drift, H_drives)
    @test system isa QuantumSystem
    @test get_drift(system) == H_drift
    @test get_drives(system) == H_drives

    # test jacobians
    a = randn(n_drives)
    ∂G = system.∂G(a)
    @test length(∂G) == system.n_drives
    @test all(∂G .≈ QuantumSystems.generator_jacobian(system.G)(a))

    # repeat with a bigger system
    H_drift = kron(PAULIS[:Z], PAULIS[:Z])
    H_drives = [kron(PAULIS[:X], PAULIS[:I]), kron(PAULIS[:I], PAULIS[:X]),
                kron(PAULIS[:Y], PAULIS[:I]), kron(PAULIS[:I], PAULIS[:Y])]
    n_drives = length(H_drives)

    system = QuantumSystem(H_drift, H_drives)
    @test system isa AbstractQuantumSystem
    @test system isa QuantumSystem
    @test get_drift(system) == H_drift
    @test get_drives(system) == H_drives

    # test jacobians
    a = randn(n_drives)
    ∂G = system.∂G(a)
    @test length(∂G) == system.n_drives
    @test all(∂G .≈ QuantumSystems.generator_jacobian(system.G)(a))
end

@testitem "Parametric system creation" begin
    system = QuantumSystem(PAULIS[:Z], [PAULIS[:X]], params=Dict(:a => 1))
    @test system.params[:a] == 1

    open_system = OpenQuantumSystem(PAULIS[:Z], [PAULIS[:X]], params=Dict(:a => 1))
    @test open_system.params[:a] == 1
end

@testitem "No drift system creation" begin
    H_drift = zeros(2, 2)
    H_drives = [PAULIS[:X], PAULIS[:Y]]

    sys1 = QuantumSystem(H_drift, H_drives)
    sys2 = QuantumSystem(H_drives)

    @test get_drift(sys1) == get_drift(sys2) == H_drift
    @test get_drives(sys1) == get_drives(sys2) == H_drives
end

@testitem "No drive system creation" begin
    H_drift = PAULIS[:Z]
    H_drives = Matrix{ComplexF64}[]

    sys1 = QuantumSystem(H_drift, H_drives)
    sys2 = QuantumSystem(H_drift)

    @test get_drift(sys1) == get_drift(sys2) == H_drift
    @test get_drives(sys1) == get_drives(sys2) == H_drives
end

@testitem "System creation with Hamiltonian function" begin
    H(a) = PAULIS[:Z] + a[1] * PAULIS[:X]
    system = QuantumSystem(H, 1)
    @test system isa QuantumSystem
    @test get_drift(system) == PAULIS[:Z]
    @test get_drives(system) == [PAULIS[:X]]

    # test jacobians
    compare = QuantumSystem(PAULIS[:Z], [PAULIS[:X]])
    a = randn(system.n_drives)
    @test system.∂G(a) == compare.∂G(a)

    # test three drives
    H(a) = a[1] * PAULIS[:X] + a[2] * PAULIS[:Y] + a[3] * PAULIS[:Z]
    system = QuantumSystem(H, 3)
    @test system isa QuantumSystem
    @test get_drift(system) == zeros(2, 2)
    @test get_drives(system) == [PAULIS[:X], PAULIS[:Y], PAULIS[:Z]]

end

@testitem "Open system creation" begin
    H_drift = PAULIS[:Z]
    # don't want drives == levels
    H_drives = [PAULIS[:X]]
    dissipation_operators = [PAULIS[:Z], PAULIS[:X]]

    system = OpenQuantumSystem(H_drift, H_drives, dissipation_operators)
    @test system isa AbstractQuantumSystem
    @test system isa OpenQuantumSystem
    @test get_drift(system) == H_drift
    @test get_drives(system) == H_drives
    @test system.dissipation_operators == dissipation_operators

    # test dissipation
    𝒢_drift = Isomorphisms.G(Isomorphisms.ad_vec(H_drift))
    @test system.𝒢(zeros(system.n_drives)) != 𝒢_drift

    # test jacobians (disspiation is constant)
    a = randn(system.n_drives)
    ∂𝒢 = system.∂𝒢(a)
    @test length(∂𝒢) == system.n_drives
    @test all(∂𝒢 .≈ QuantumSystems.generator_jacobian(system.𝒢)(a))

end

@testitem "Open system alternate constructors" begin
    H_drift = PAULIS[:Z]
    # don't want drives == levels
    H_drives = [PAULIS[:X]]
    dissipation_operators = [PAULIS[:Z], PAULIS[:X]]

    system = OpenQuantumSystem(
        H_drift, H_drives, dissipation_operators=dissipation_operators
    )
    @test system isa OpenQuantumSystem
    @test get_drift(system) == H_drift
    @test get_drives(system) == H_drives
    @test system.dissipation_operators == dissipation_operators

    # no drift
    system = OpenQuantumSystem(H_drives, dissipation_operators=dissipation_operators)
    @test system isa OpenQuantumSystem
    @test get_drift(system) == zeros(size(H_drift))
    @test get_drives(system) == H_drives
    @test system.dissipation_operators == dissipation_operators

    # no drives
    system = OpenQuantumSystem(
        H_drift, dissipation_operators=dissipation_operators
    )
    @test system isa OpenQuantumSystem
    @test system isa OpenQuantumSystem
    @test get_drift(system) == H_drift
    @test get_drives(system) == []
    @test system.dissipation_operators == dissipation_operators

    # function
    H = a -> PAULIS[:Z] + a[1] * PAULIS[:X]
    system = OpenQuantumSystem(H, 1, dissipation_operators=dissipation_operators)
    @test system isa OpenQuantumSystem
    @test get_drift(system) == H_drift
    @test get_drives(system) == H_drives
    @test system.dissipation_operators == dissipation_operators

end

@testitem "Generator jacobian types" begin
    GX = Isomorphisms.G(PAULIS.X)
    GY = Isomorphisms.G(PAULIS.Y)
    GZ = Isomorphisms.G(PAULIS.Z)
    G(a) = GX + a[1] * GY + a[2] * GZ
    ∂G = QuantumSystems.generator_jacobian(G)

    traj_a = randn(Float64, 2, 3)
    a₀ = traj_a[:, 1]
    aᵥ = @views traj_a[:, 1]

    @test ∂G(a₀) isa AbstractVector{<:AbstractMatrix{Float64}}
    @test ∂G(a₀)[1] isa AbstractMatrix

    @test ∂G(aᵥ) isa AbstractVector{<:AbstractMatrix{Float64}}
    @test ∂G(aᵥ)[1] isa AbstractMatrix{Float64}
end

end
