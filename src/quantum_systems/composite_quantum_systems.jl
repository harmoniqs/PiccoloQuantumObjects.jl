export lift_operator
export CompositeQuantumSystem

# ----------------------------------------------------------------------------- #
# Lift operators
# ----------------------------------------------------------------------------- #

@doc raw"""
    lift_operator(operator::AbstractMatrix{<:Number}, i::Int, subsystem_levels::Vector{Int})
    lift_operator(operator::AbstractMatrix{<:Number}, i::Int, n_qubits::Int; kwargs...)
    lift_operator(operators::AbstractVector{<:AbstractMatrix{T}}, indices::AbstractVector{Int}, subsystem_levels::Vector{Int})
    lift_operator(operators::AbstractVector{<:AbstractMatrix{T}}, indices::AbstractVector{Int}, n_qubits::Int; kwargs...)
    lift_operator(operator::AbstractMatrix{T}, indices::AbstractVector{Int}, subsystem_levels::AbstractVector{Int})
    lift_operator(operator::AbstractMatrix{T}, indices::AbstractVector{Int}, n_qubits::Int; kwargs...)

Lift an `operator` acting on the `i`-th subsystem within `subsystem_levels` to an operator
acting on the entire system spanning `subsystem_levels`.
"""
function lift_operator end

function lift_operator(operator::AbstractMatrix{T}, i::Int, subsystem_levels::AbstractVector{Int}
) where T <: Number
    @assert size(operator, 1) == subsystem_levels[i] "Operator must match subsystem level."
    Is = [Matrix{T}(I(l)) for l ∈ subsystem_levels]
    Is[i] = operator
    return reduce(kron, Is)
end

function lift_operator(
    operator::AbstractMatrix{T}, i::Int, n_qubits::Int;
    levels::Int=size(operator, 1)
) where T <: Number
    return lift_operator(operator, i, fill(levels, n_qubits))
end

function lift_operator(
    operators::AbstractVector{<:AbstractMatrix{T}},
    indices::AbstractVector{Int},
    subsystem_levels::AbstractVector{Int}
) where T <: Number
    @assert length(operators) == length(indices)
    return prod([lift_operator(op, i, subsystem_levels) for (op, i) ∈ zip(operators, indices)])
end

function lift_operator(
    operators::AbstractVector{<:AbstractMatrix{T}},
    indices::AbstractVector{Int},
    n_qubits::Int;
    levels::Int=size(operators[1], 1)
) where T <: Number
    return prod(
        [lift_operator(op, i, n_qubits, levels=levels) for (op, i) ∈ zip(operators, indices)]
    )
end

function lift_operator(
    operator::AbstractMatrix{T},
    indices::AbstractVector{Int},
    subsystem_levels::AbstractVector{Int},
) where T <: Number
    N = length(subsystem_levels)
    L = [subsystem_levels[i] for i ∈ indices]
    Lᶜ = [subsystem_levels[i] for i ∈ setdiff(1:N, indices)]
    @assert prod(L) == size(operator, 1)

    # Start with operator at the leading position
    shape = vcat(L, Lᶜ)
    array_shape = reverse(vcat(shape, shape))
    full_operator = kron(operator, [Matrix{T}(I(l)) for l ∈ Lᶜ]...)

    # Permute the array to match the actual subsystem order
    order = vcat(indices, [i for i ∈ setdiff(1:N, indices)])
    perm = sortperm(order)
    array_perm = reverse(2length(perm) + 1 .- vcat(perm, perm .+ length(perm)))

    return reshape(
        PermutedDimsArray(reshape(full_operator, array_shape...), array_perm),
        size(full_operator)
    )
end

function lift_operator(
    operator::AbstractMatrix{T},
    indices::AbstractVector{Int},
    n_qubits::Int;
    levels::Int=2
) where T <: Number
    return lift_operator(operator, indices, fill(levels, n_qubits))
end

# ----------------------------------------------------------------------------- #
# Composite Quantum Systems
# ----------------------------------------------------------------------------- #

"""
    CompositeQuantumSystem <: AbstractQuantumSystem

A composite quantum system consisting of `subsystems`. Couplings between subsystems can
be additionally defined. Subsystem drives are always appended to any new coupling drives.
"""
struct CompositeQuantumSystem{F1<:Function, F2<:Function} <: AbstractQuantumSystem
    H::F1
    G::F2
    n_drives::Int
    levels::Int
    params::Dict{Symbol, Any}
    subsystem_levels::Vector{Int}
    subsystems::Vector{QuantumSystem}

    function CompositeQuantumSystem(
        H_drift::AbstractMatrix{<:Number},
        H_drives::AbstractVector{<:AbstractMatrix{<:Number}},
        subsystems::AbstractVector{<:QuantumSystem};
        params::Dict{Symbol, Any}=Dict{Symbol, Any}()
    )
        subsystem_levels = [sys.levels for sys ∈ subsystems]
        levels = prod(subsystem_levels)

        H_drift = sparse(H_drift)
        for (i, sys) ∈ enumerate(subsystems)
            H_drift += lift_operator(get_drift(sys), i, subsystem_levels)
        end

        H_drives = sparse.(H_drives)
        for (i, sys) ∈ enumerate(subsystems)
            for H_drive ∈ get_drives(sys)
                push!(H_drives, lift_operator(H_drive, i, subsystem_levels))
            end
        end

        n_drives = length(H_drives)
        H_drives = sparse.(H_drives)
        G_drives = sparse.(Isomorphisms.G.(H_drives))

        # At least provide one drive
        H = a -> H_drift + sum(a .* H_drives)
        G = a -> G_drift + sum(a .* G_drives)

        return new{typeof(H), typeof(G)}(
            H,
            G,
            n_drives,
            levels,
            params,
            subsystem_levels,
            subsystems
        )
    end

    function CompositeQuantumSystem(
        H_drives::AbstractVector{<:AbstractMatrix{T}},
        subsystems::AbstractVector{<:QuantumSystem};
        kwargs...
    ) where T <: Number
        @assert !isempty(H_drives) "At least one drive is required"
        return CompositeQuantumSystem(
            spzeros(T, size(H_drives[1])),
            H_drives,
            subsystems;
            kwargs...
        )
    end

    function CompositeQuantumSystem(
        H_drift::AbstractMatrix{T},
        subsystems::AbstractVector{<:QuantumSystem};
        kwargs...
    ) where T <: Number
        return CompositeQuantumSystem(
            H_drift, 
            Matrix{T}[], 
            subsystems;
            kwargs...
        )
    end

    function CompositeQuantumSystem(
        subsystems::AbstractVector{<:QuantumSystem};
        kwargs...
    )
        @assert !isempty(subsystems) "At least one subsystem is required"
        T = eltype(get_drift(subsystems[1]))
        levels = prod([sys.levels for sys ∈ subsystems])
        return CompositeQuantumSystem(
            spzeros(T, (levels, levels)), 
            Matrix{T}[], 
            subsystems; 
            kwargs...
        )
    end
end

# ****************************************************************************** #

@testitem "Lift_operator subsystems" begin
    using LinearAlgebra
    @test lift_operator(PAULIS[:X], 1, [2, 3]) ≈ kron(PAULIS[:X], I(3))
    @test lift_operator(PAULIS[:Y], 2, [4, 2]) ≈ kron(I(4), PAULIS[:Y])
    @test lift_operator(PAULIS[:X], 2, [3, 2, 4]) ≈ reduce(kron, [I(3), PAULIS[:X], I(4)])
end

@testitem "Lift_operator qubits" begin
    using LinearAlgebra
    @test lift_operator(PAULIS[:X], 1, 2) ≈ kron(PAULIS[:X], I(2))
    @test lift_operator(PAULIS[:Y], 2, 2) ≈ kron(I(2), PAULIS[:Y])
    @test lift_operator(PAULIS[:X], 2, 3) ≈ reduce(kron, [I(2), PAULIS[:X], I(2)])
end

@testitem "Lift_operator multiple operators" begin
    using LinearAlgebra
    pair = [PAULIS[:X], PAULIS[:Y]]
    @test lift_operator(pair, [1, 2], [2, 2]) ≈ kron(PAULIS[:X], PAULIS[:Y])
    @test lift_operator(pair, [2, 1], [2, 2]) ≈ kron(PAULIS[:Y], PAULIS[:X])
    @test lift_operator(pair, [1, 2], [2, 2, 3]) ≈ kron(PAULIS[:X], PAULIS[:Y], I(3))
    @test lift_operator(pair, [2, 3], [4, 2, 2]) ≈ kron(I(4), PAULIS[:X], PAULIS[:Y])

    # Pass number of qubits
    @test lift_operator(pair, [1, 2], 3) ≈ kron(PAULIS[:X], PAULIS[:Y], I(2))
end

@testitem "Lift_operator single operator into disjoint levels" begin
    U = haar_random(2)
    UU = kron(U, U)
    UUU = kron(UU, U)
    I2 = [1 0; 0 1]
    I3 = [1 0 0; 0 1 0; 0 0 1]
;

    @test lift_operator(UU, [1,2], [2,2,2]) ≈ kron(U, U, I2)
    @test lift_operator(UU, [2,3], [2,2,2]) ≈ kron(I2, U, U)
    @test lift_operator(UU, [1,3], [2,2,2]) ≈ kron(U, I2, U)

    @test lift_operator(UU, [1], [4, 2, 3]) ≈ kron(UU, I2, I3)
    @test lift_operator(UU, [2], [2, 4, 3]) ≈ kron(I2, UU, I3)
    @test lift_operator(UU, [3], [2, 3, 4]) ≈ kron(I2, I3, UU)

    @test lift_operator(UUU, [1, 3, 4], [2, 3, 2, 2]) ≈ kron(U, I3, UU)
    @test lift_operator(UUU, [1, 2, 4], [2, 2, 3, 2]) ≈ kron(UU, I3, U)

    # Test qubit interface
    @test lift_operator(U, [1], 3) ≈ kron(U, I2, I2)
    @test lift_operator(UU, [2,3], 3) ≈ kron(I2, U, U)
    @test lift_operator(UU, [1,3], 3) ≈ kron(U, I2, U)
end

@testitem "Composite system" begin
    subsystem_levels = [4, 2, 2]
    sys1 = QuantumSystem(kron(PAULIS[:Z], PAULIS[:Z]), [kron(PAULIS[:X], PAULIS[:Y])])
    sys2 = QuantumSystem([PAULIS[:Y], PAULIS[:Z]])
    sys3 = QuantumSystem(zeros(2, 2))
    subsystems = [sys1, sys2, sys3]
    g12 = 0.1 * lift_operator([kron(PAULIS[:X], PAULIS[:X]), PAULIS[:X]], [1, 2], subsystem_levels)
    g23 = 0.2 * lift_operator([PAULIS[:Y], PAULIS[:Y]], [2, 3], subsystem_levels)

    # Construct composite system
    csys = CompositeQuantumSystem(g12, [g23], [sys1, sys2, sys3])
    @test csys.levels == prod(subsystem_levels)
    @test csys.n_drives == 1 + sum([sys.n_drives for sys ∈ subsystems])
    @test csys.subsystems == subsystems
    @test csys.subsystem_levels == subsystem_levels
    @test get_drift(csys) ≈ g12 + lift_operator(kron(PAULIS[:Z], PAULIS[:Z]), 1, subsystem_levels)
end

@testitem "Composite system from drift" begin
    using LinearAlgebra

    subsystem_levels = [2, 2]
    sys1 = QuantumSystem([PAULIS[:X], PAULIS[:Y]])
    sys2 = QuantumSystem([PAULIS[:Y], PAULIS[:Z]])
    subsystems = [sys1, sys2]
    g12 = 0.1 * kron(PAULIS[:X], PAULIS[:X])

    # Construct composite system from drift
    csys = CompositeQuantumSystem(g12, [sys1, sys2])
    @test csys.levels == prod(subsystem_levels)
    @test csys.n_drives == sum([sys.n_drives for sys ∈ subsystems])
    @test csys.subsystems == subsystems
    @test csys.subsystem_levels == subsystem_levels
    @test get_drift(csys) ≈ g12
end

@testitem "Composite system from drives" begin
    subsystem_levels = [2, 2, 2]
    sys1 = QuantumSystem(PAULIS[:Z], [PAULIS[:X], PAULIS[:Y]])
    sys2 = QuantumSystem([PAULIS[:Y], PAULIS[:Z]])
    sys3 = QuantumSystem(zeros(2, 2))
    subsystems = [sys1, sys2, sys3]
    g12 = 0.1 * lift_operator([PAULIS[:X], PAULIS[:X]], [1, 2], subsystem_levels)
    g23 = 0.2 * lift_operator([PAULIS[:Y], PAULIS[:Y]], [2, 3], subsystem_levels)

    csys = CompositeQuantumSystem([g12, g23], [sys1, sys2, sys3])
    @test csys.levels == prod(subsystem_levels)
    @test csys.n_drives == 2 + sum([sys.n_drives for sys ∈ subsystems])
    @test csys.subsystems == subsystems
    @test csys.subsystem_levels == subsystem_levels
    @test get_drift(csys) ≈ lift_operator(PAULIS[:Z], 1, subsystem_levels)
end
