export TimeDependentQuantumSystem

# ----------------------------------------------------------------------------- #
# TimeDependentQuantumSystem
# ----------------------------------------------------------------------------- #

# TODO: Open System

"""
    TimeDependentQuantumSystem <: AbstractQuantumSystem

A struct for storing time-dependent quantum dynamics and the appropriate gradients.

# Additional fields
- `H::Function`: The Hamiltonian function with time: a, t -> H(a, t).
- `G::Function`: The isomorphic generator function with time, a, t -> G(a, t).
- `∂G::Function`: The generator jacobian function with time, a, t -> ∂G(a, t).
- `n_drives::Int`: The number of drives in the system.
- `levels::Int`: The number of levels in the system.
- `params::Dict{Symbol, Any}`: A dictionary of parameters.

"""
struct TimeDependentQuantumSystem{F1<:Function, F2<:Function} <: AbstractQuantumSystem
    H::F1
    G::F2
    n_drives::Int
    levels::Int 
    params::Dict{Symbol, Any}

    function TimeDependentQuantumSystem(
        H_drift::AbstractMatrix{<:Number},
        H_drives::Vector{<:AbstractMatrix{<:Number}};
        carriers::AbstractVector{<:Number}=zeros(length(H_drives)),
        phases::AbstractVector{<:Number}=zeros(length(H_drives)),
        params::Dict{Symbol, <:Any}=Dict{Symbol, Any}()
    )
        levels = size(H_drift, 1)
        H_drift = sparse(H_drift)
        G_drift = sparse(Isomorphisms.G(H_drift))

        n_drives = length(H_drives)
        H_drives = sparse.(H_drives)
        G_drives = sparse.(Isomorphisms.G.(H_drives))

        if n_drives == 0
            H = (a, t) -> H_drift
            G = (a, t) -> G_drift
        else
            H = (a, t) -> H_drift + sum(a[i] * cos(carriers[i] * t + phases[i]) .* H_drives[i] for i in eachindex(H_drives))
            G = (a, t) -> G_drift + sum(a[i] * cos(carriers[i] * t + phases[i]) .* G_drives[i] for i in eachindex(G_drives))
        end

        # save carries and phases
        params[:carriers] = carriers
        params[:phases] = phases

        return new{typeof(H), typeof(G)}(
            H,
            G,
            n_drives,
            levels,
            params
        )
    end

    function TimeDependentQuantumSystem(
        H_drives::Vector{<:AbstractMatrix{ℂ}}; 
        kwargs...
    ) where ℂ <: Number
        @assert !isempty(H_drives) "At least one drive is required."
        return TimeDependentQuantumSystem(
            spzeros(ℂ, size(H_drives[1])), 
            H_drives; 
            kwargs...
        )
    end

    TimeDependentQuantumSystem(H_drift::AbstractMatrix{ℂ}; kwargs...) where ℂ <: Number = 
        TimeDependentQuantumSystem(H_drift, Matrix{ℂ}[]; kwargs...)

    function TimeDependentQuantumSystem(
        H::F,
        n_drives::Int;
        params::Dict{Symbol, <:Any}=Dict{Symbol, Any}()
    ) where F <: Function
        G = (a, t) -> Isomorphisms.G(sparse(H(a, t)))
        levels = size(H(zeros(n_drives), 0.0), 1)
        return new{F, typeof(G)}(H, G, n_drives, levels, params)
    end
end

get_drift(sys::TimeDependentQuantumSystem) = t -> sys.H(zeros(sys.n_drives), t)

function get_drives(sys::TimeDependentQuantumSystem)
    H_drift = get_drift(sys)
    # Basis vectors for controls will extract drive operators
    return [t -> sys.H(I[1:sys.n_drives, i], t) - H_drift(t) for i ∈ 1:sys.n_drives]
end

# ****************************************************************************** #

@testitem "System creation" begin
    H_drift = kron(PAULIS.Z, PAULIS.I)
    H_drives = [kron(PAULIS.X, PAULIS.I), kron(PAULIS.Y, PAULIS.I)]
    n_drives = length(H_drives)
    carriers = [1.0, 2.0]
    phases = [0.1, 0.2]

    sys = TimeDependentQuantumSystem(H_drift, H_drives, carriers=carriers, phases=phases)

    @test sys isa TimeDependentQuantumSystem

    # Params
    @test sys.params[:carriers] == carriers
    @test sys.params[:phases] == phases

    # Time-dependent Hamiltonians
    @test get_drift(sys)(0.0) == H_drift
    @test [H(0.0) for H in get_drives(sys)] == H_drives .* cos.(phases)

    t = 2.0
    @test get_drift(sys)(t) == H_drift
    @test [H(t) for H in get_drives(sys)] == H_drives .* cos.(carriers .* t .+ phases)
end

@testitem "No drift system creation" begin
    H_drift = zeros(2, 2)
    H_drives = [PAULIS.X, PAULIS.Y]
    sys1 = TimeDependentQuantumSystem(H_drift, H_drives)
    sys2 = TimeDependentQuantumSystem(H_drives)

    @test get_drift(sys1)(0.0) == get_drift(sys2)(0.0) == H_drift
    @test [H(0.0) for H in get_drives(sys1)] == [H(0.0) for H in get_drives(sys2)] == H_drives
end

@testitem "No drive system creation" begin
    H_drift = PAULIS.Z
    H_drives = Matrix{ComplexF64}[]

    sys1 = TimeDependentQuantumSystem(H_drift, H_drives)
    sys2 = TimeDependentQuantumSystem(H_drift)

    @test get_drift(sys1)(0.0) == get_drift(sys2)(0.0) == H_drift
    @test [H(0.0) for H in get_drives(sys1)] == [H(0.0) for H in get_drives(sys2)] == H_drives
end

@testitem "System creation with Hamiltonian function" begin
    n_drives = 1
    H_drift = t -> PAULIS.Z * cos(t)
    H_drive = t -> PAULIS.X * cos(t)
    system = TimeDependentQuantumSystem((a, t) -> H_drift(t) + a[1] * H_drive(t), n_drives)

    @test system isa TimeDependentQuantumSystem
    @test system.n_drives == n_drives
    @test get_drift(system)(0.0) == H_drift(0.0)
    @test get_drives(system)[1](0.0) == H_drive(0.0)

    t = 2.0
    @test get_drift(system)(t) == H_drift(t)
    @test get_drives(system)[1](t) == H_drive(t)

    # test three drives
    n_drives = 3
    Xₜ = t -> PAULIS.X * cos(t)
    Yₜ = t -> PAULIS.Y * sin(t)
    system = TimeDependentQuantumSystem((a, t) -> a[1] * Xₜ(t) + a[2] * Yₜ(t) + a[3] * PAULIS.Z, n_drives)

    @test system isa TimeDependentQuantumSystem
    @test get_drift(system)(0.0) == zeros(2, 2)
    @test get_drift(system)(1.0) == zeros(2, 2)

    t = 0.0
    @test [H(t) for H in get_drives(system)] == [Xₜ(t), Yₜ(t), PAULIS.Z]

    t = 2.0
    @test [H(t) for H in get_drives(system)] == [Xₜ(t), Yₜ(t), PAULIS.Z]
end