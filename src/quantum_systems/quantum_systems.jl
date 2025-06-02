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
- `H::Function`: The Hamiltonian function, excluding dissipation: a -> H(a).
- `G::Function`: The isomorphic generator function, including dissipation, a -> G(a).
- `levels::Int`: The number of levels in the system.
- `n_drives::Int`: The number of drives in the system.

"""
struct QuantumSystem{F1<:Function, F2<:Function} <: AbstractQuantumSystem
    H::F1
    G::F2
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
        else
            H = a -> H_drift + sum(a .* H_drives)
            G = a -> G_drift + sum(a .* G_drives)
        end

        return new{typeof(H), typeof(G)}(
            H,
            G,
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
        H::F, 
        n_drives::Int; 
        params::Dict{Symbol, <:Any}=Dict{Symbol, Any}()
    ) where F <: Function
        G = a -> Isomorphisms.G(sparse(H(a)))
        levels = size(H(zeros(n_drives)), 1)
        return new{F, typeof(G)}(H, G, n_drives, levels, params)
    end

end

# ----------------------------------------------------------------------------- #
# OpenQuantumSystem
# ----------------------------------------------------------------------------- #

"""
    OpenQuantumSystem <: AbstractQuantumSystem

A struct for storing open quantum dynamics.

# Additional fields
- `dissipation_operators::Vector{AbstractMatrix}`: The dissipation operators.

See also [`QuantumSystem`](@ref).

"""
struct OpenQuantumSystem{F1<:Function, F2<:Function} <: AbstractQuantumSystem
    H::F1
    𝒢::F2
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
        H_drift::Matrix{<:Number}, 
        H_drives::AbstractVector{Matrix{<:Number}};
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
            𝒟 = spzeros(size(𝒢_drift))
        else
            𝒟 = sum(
                Isomorphisms.iso_D(L) 
                    for L ∈ sparse.(dissipation_operators)
            )
        end

        if n_drives == 0
            H = a -> H_drift
            𝒢 = a -> 𝒢_drift + 𝒟
        else
            H = a -> H_drift + sum(a .* H_drives)
            𝒢 = a -> 𝒢_drift + sum(a .* 𝒢_drives) + 𝒟
        end

        return new{typeof(H), typeof(𝒢)}(
            H,
            𝒢,
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
        H::F, n_drives::Int;
        dissipation_operators::AbstractVector{<:AbstractMatrix{ℂ}}=Matrix{ComplexF64}[],
        params::Dict{Symbol, <:Any}=Dict{Symbol, Any}()
    ) where {F <: Function, ℂ <: Number}
        G = a -> Isomorphisms.G(Isomorphisms.ad_vec(sparse(H(a))))
        levels = size(H(zeros(n_drives)), 1)
        return new{F, typeof(G)}(H, G, n_drives, levels, dissipation_operators, params)
    end

    OpenQuantumSystem(system::QuantumSystem; kwargs...) = OpenQuantumSystem(
        system.H, system.n_drives; kwargs...
    )

end

# ----------------------------------------------------------------------------- #
# VariationalQuantumSystem
# ----------------------------------------------------------------------------- #

# TODO: Open quantum systems?

"""
    VariationalQuantumSystem <: AbstractQuantumSystem

A struct for storing variational quantum dynamics.

# Additional fields
- `G_vars::AbstractVector{<:Function}`: Variational generator functions

See also [`QuantumSystem`](@ref).

"""
struct VariationalQuantumSystem{F1<:Function, F2<:Function, F⃗3<:AbstractVector{<:Function}} <: AbstractQuantumSystem
    H::F1 
    G::F2
    G_vars::F⃗3
    n_drives::Int 
    levels::Int 
    params::Dict{Symbol, Any}

    """
        VariationalQuantumSystem(
            H_drift::AbstractMatrix{<:Number},
            H_drives::AbstractVector{<:AbstractMatrix{<:Number}},
            H_vars::AbstractVector{<:AbstractMatrix{<:Number}};
            kwargs...
        )
        VariationalQuantumSystem(
            H_drives::AbstractVector{<:AbstractMatrix{<:Number}},
            H_vars::AbstractVector{<:AbstractMatrix{<:Number}};
            kwargs...
        )
        VariationalQuantumSystem(
            H::F1,
            H_vars::F⃗2,
            n_drives::Int;
            kwargs...
        )

    Constructs a `VariationalQuantumSystem` object from the drift and drive Hamiltonian 
    terms and variational Hamiltonians, for sensitivity and robustness analysis.

    """
    function VariationalQuantumSystem end

    function VariationalQuantumSystem(
        H_drift::AbstractMatrix{<:Number},
        H_drives::AbstractVector{<:AbstractMatrix{<:Number}},
        H_vars::AbstractVector{<:AbstractMatrix{<:Number}};
        params::Dict{Symbol, <:Any}=Dict{Symbol, Any}()
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

        return new{typeof(H), typeof(G), typeof(G_vars)}(
            H, G, G_vars, n_drives, levels, params
        )
    end

    function VariationalQuantumSystem(
        H_drives::AbstractVector{<:AbstractMatrix{ℂ}},
        H_vars::AbstractVector{<:AbstractMatrix{<:Number}};
        kwargs...
    ) where ℂ <: Number
        @assert !isempty(H_drives) "At least one drive is required"
        @assert !isempty(H_vars) "At least one variational operator is required"
        return VariationalQuantumSystem(
            spzeros(ℂ, size(H_drives[1])), 
            H_drives, 
            H_vars; 
            kwargs...
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
        return new{F1, typeof(G), F⃗2}(H, G, G_vars, n_drives, levels, params)
    end
end

# ******************************************************************************* #

@testitem "Test system show methods" begin
    function showtest(x)
        sprint() do io
            Base.show(IOContext(io), MIME"text/plain"(), x)
        end
    end

    H_drift = PAULIS[:Z]
    H_drives = [PAULIS[:X], PAULIS[:Y]]
    n_drives = length(H_drives)

    system = QuantumSystem(H_drift, H_drives)
    @test showtest(system) == "QuantumSystem: levels = 2, n_drives = 2"

    open_system = OpenQuantumSystem(PAULIS[:Z], [PAULIS[:X]], params=Dict(:a => 1))
    @test showtest(open_system) == "OpenQuantumSystem: levels = 2, n_drives = 1"
end

@testitem "System creation" begin
    H_drift = PAULIS[:Z]
    H_drives = [PAULIS[:X], PAULIS[:Y]]
    n_drives = length(H_drives)

    system = QuantumSystem(H_drift, H_drives)
    @test system isa QuantumSystem
    @test get_drift(system) == H_drift
    @test get_drives(system) == H_drives

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
end

@testitem "System creation with params" begin
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
    system = QuantumSystem(
        a -> PAULIS[:Z] + a[1] * PAULIS[:X], 1
    )
    @test system isa QuantumSystem
    @test get_drift(system) == PAULIS[:Z]
    @test get_drives(system) == [PAULIS[:X]]

    # test three drives
    system = QuantumSystem(
        a -> a[1] * PAULIS[:X] + a[2] * PAULIS[:Y] + a[3] * PAULIS[:Z], 3
    )
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

@testitem "Variational system creation" begin
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
        @assert varsys isa VariationalQuantumSystem
        @assert varsys.n_drives == 2
        @assert length(varsys.G_vars) == 2
        @assert varsys.G(a) ≈ G
        @assert varsys.G_vars[1](a) ≈ G_X
        @assert varsys.G_vars[2](a) ≈ G_Y
    end

    # single sensitivity
    varsys = VariationalQuantumSystem(
        [PAULIS.X, PAULIS.Y],
        [PAULIS.X] 
    )
    @assert varsys isa VariationalQuantumSystem
    @assert varsys.n_drives == 2
    @assert length(varsys.G_vars) == 1
    @assert varsys.G(a) ≈ G
    @assert varsys.G_vars[1](a) ≈ G_X

    # functional sensitivity
    varsys = VariationalQuantumSystem(
        a -> a[1] * PAULIS.X + a[2] * PAULIS.Y,
        [a -> a[1] * PAULIS.X, a -> PAULIS.Y],
        2
    )
    @assert varsys isa VariationalQuantumSystem
    @assert varsys.n_drives == 2
    @assert length(varsys.G_vars) == 2
    @assert varsys.G(a) ≈ G
    @assert varsys.G_vars[1](a) ≈ a[1] * G_X
    @assert varsys.G_vars[2](a) ≈ G_Y
end