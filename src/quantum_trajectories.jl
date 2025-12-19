module QuantumTrajectories

"""
Quantum trajectory types and constructors for optimal control.

Provides high-level trajectory types that wrap NamedTrajectory with quantum-specific metadata:
- AbstractQuantumTrajectory: Base type
- UnitaryTrajectory: For unitary gate synthesis
- KetTrajectory: For quantum state transfer
- DensityTrajectory: For open quantum systems

Also includes helper functions for trajectory initialization:
- unitary_geodesic: Geodesic interpolation on the unitary manifold
- unitary_linear_interpolation: Linear interpolation of unitaries
- linear_interpolation: Generic linear interpolation
"""

export AbstractQuantumTrajectory
export UnitaryTrajectory
export KetTrajectory
export DensityTrajectory
export EnsembleTrajectory
export SamplingTrajectory
export get_trajectory, get_system, get_goal, get_state_name, get_control_name, get_state, get_controls
export get_ensemble_state_names, get_systems, get_weights, get_combined_trajectory
export unitary_geodesic, unitary_linear_interpolation, linear_interpolation
export build_sampling_trajectory, build_ensemble_trajectory, build_ensemble_trajectory_from_trajectories
export update_base_trajectory

using NamedTrajectories
using LinearAlgebra
using TestItems

# Import from other PiccoloQuantumObjects modules
using ..QuantumSystems: AbstractQuantumSystem, QuantumSystem, OpenQuantumSystem, get_drift
using ..Isomorphisms: operator_to_iso_vec, iso_vec_to_operator, ket_to_iso, density_to_iso_vec
using ..EmbeddedOperators: AbstractPiccoloOperator, EmbeddedOperator, unembed, embed

# ============================================================================= #
#                        Trajectory Initialization Helpers                      #
# ============================================================================= #

"""
    linear_interpolation(x, y, n)

Linear interpolation between vectors or matrices.
"""
linear_interpolation(x::AbstractVector, y::AbstractVector, n::Int) = hcat(range(x, y, n)...)
linear_interpolation(X::AbstractMatrix, Y::AbstractMatrix, n::Int) =
    hcat([X + (Y - X) * t for t in range(0, 1, length=n)]...)

"""
    unitary_linear_interpolation(U_init, U_goal, samples)

Compute a linear interpolation of unitary operators with `samples` samples.
"""
function unitary_linear_interpolation(
    U_init::AbstractMatrix{<:Number},
    U_goal::AbstractMatrix{<:Number},
    samples::Int
)
    Ũ⃗_init = operator_to_iso_vec(U_init)
    Ũ⃗_goal = operator_to_iso_vec(U_goal)
    Ũ⃗s = [Ũ⃗_init + (Ũ⃗_goal - Ũ⃗_init) * t for t ∈ range(0, 1, length=samples)]
    Ũ⃗ = hcat(Ũ⃗s...)
    return Ũ⃗
end

function unitary_linear_interpolation(
    U_init::AbstractMatrix{<:Number},
    U_goal::EmbeddedOperator,
    samples::Int
)
    return unitary_linear_interpolation(U_init, U_goal.operator, samples)
end

"""
    unitary_geodesic(U_init, U_goal, times; kwargs...)

Compute the geodesic connecting U_init and U_goal at the specified times.

# Arguments
- `U_init::AbstractMatrix{<:Number}`: The initial unitary operator.
- `U_goal::AbstractMatrix{<:Number}`: The goal unitary operator.
- `times::AbstractVector{<:Number}`: The times at which to evaluate the geodesic.

# Keyword Arguments
- `return_unitary_isos::Bool=true`: If true returns a matrix where each column is a unitary 
    isovec, i.e. vec(vcat(real(U), imag(U))). If false, returns a vector of unitary matrices.
- `return_generator::Bool=false`: If true, returns the effective Hamiltonian generating 
    the geodesic.
- `H_drift::AbstractMatrix{<:Number}=zeros(size(U_init))`: Drift Hamiltonian for time-dependent systems.
"""
function unitary_geodesic(
    U_init::AbstractMatrix{<:Number},
    U_goal::AbstractMatrix{<:Number},
    times::AbstractVector{<:Number};
    return_unitary_isos=true,
    return_generator=false,
    H_drift::AbstractMatrix{<:Number}=zeros(size(U_init)),
)
    t₀ = times[1]
    T = times[end] - t₀

    U_drift(t) = exp(-im * H_drift * t)
    H = im * log(U_drift(T)' * (U_goal * U_init')) / T
    # -im prefactor is not included in H
    U_geo = [U_drift(t) * exp(-im * H * (t - t₀)) * U_init for t ∈ times]

    if !return_unitary_isos
        if return_generator
            return U_geo, H
        else
            return U_geo
        end
    else
        Ũ⃗_geo = stack(operator_to_iso_vec.(U_geo), dims=2)
        if return_generator
            return Ũ⃗_geo, H
        else
            return Ũ⃗_geo
        end
    end
end

function unitary_geodesic(
    U_goal::AbstractPiccoloOperator,
    samples::Int;
    kwargs...
)
    return unitary_geodesic(
        I(size(U_goal, 1)),
        U_goal,
        samples;
        kwargs...
    )
end

function unitary_geodesic(
    U_init::AbstractMatrix{<:Number},
    U_goal::EmbeddedOperator,
    samples::Int;
    H_drift::AbstractMatrix{<:Number}=zeros(size(U_init)),
    kwargs...
)
    H_drift = unembed(H_drift, U_goal)
    U1 = unembed(U_init, U_goal)
    U2 = unembed(U_goal)
    Ũ⃗ = unitary_geodesic(U1, U2, samples; H_drift=H_drift, kwargs...)
    return hcat([
        operator_to_iso_vec(embed(iso_vec_to_operator(Ũ⃗ₜ), U_goal))
        for Ũ⃗ₜ ∈ eachcol(Ũ⃗)
    ]...)
end

function unitary_geodesic(
    U_init::AbstractMatrix{<:Number},
    U_goal::AbstractMatrix{<:Number},
    samples::Int;
    kwargs...
)
    return unitary_geodesic(U_init, U_goal, range(0, 1, samples); kwargs...)
end

"""
    get_initial_controls(sys::AbstractQuantumSystem, N::Int)

Generate initial controls sampled uniformly between drive bounds, with zeros at boundaries.

# Arguments
- `sys::AbstractQuantumSystem`: The quantum system with drive_bounds
- `N::Int`: Number of timesteps in the trajectory

# Returns
- `Matrix{Float64}`: Control matrix of size `(n_drives, N)` with zeros at first and last timesteps
"""
function get_initial_controls(sys::AbstractQuantumSystem, N::Int)
    n_drives = sys.n_drives
    u_lower = [sys.drive_bounds[i][1] for i in 1:n_drives]
    u_upper = [sys.drive_bounds[i][2] for i in 1:n_drives]
    u_inner = u_lower .+ (u_upper .- u_lower) .* rand(n_drives, N - 2)
    return hcat(zeros(n_drives), u_inner, zeros(n_drives))
end

export get_initial_controls

# ============================================================================= #
#                        Quantum Trajectory Types                               #
# ============================================================================= #

"""
    AbstractQuantumTrajectory

Abstract type for quantum trajectories that wrap a `NamedTrajectory` with quantum-specific metadata.

Subtypes should implement:
- `get_trajectory(qtraj)`: Return the underlying `NamedTrajectory`
- `get_system(qtraj)`: Return the quantum system
- `get_state_name(qtraj)`: Return the state variable name
- `get_control_name(qtraj)`: Return the control variable name
- `get_goal(qtraj)`: Return the goal state/operator
"""
abstract type AbstractQuantumTrajectory end

# Accessor functions
get_trajectory(qtraj::AbstractQuantumTrajectory) = qtraj.trajectory
get_system(qtraj::AbstractQuantumTrajectory) = qtraj.system
get_state_name(qtraj::AbstractQuantumTrajectory) = qtraj.state_name
get_control_name(qtraj::AbstractQuantumTrajectory) = qtraj.control_name
get_goal(qtraj::AbstractQuantumTrajectory) = qtraj.goal

# Delegate common operations to underlying NamedTrajectory
Base.getindex(qtraj::AbstractQuantumTrajectory, key) = getindex(get_trajectory(qtraj), key)
Base.setindex!(qtraj::AbstractQuantumTrajectory, value, key) = setindex!(get_trajectory(qtraj), value, key)

# Delegate property access to underlying NamedTrajectory, except for AbstractQuantumTrajectory's own fields
function Base.getproperty(qtraj::AbstractQuantumTrajectory, symb::Symbol)
    # Access AbstractQuantumTrajectory's own fields
    if symb ∈ fieldnames(typeof(qtraj))
        return getfield(qtraj, symb)
    # Otherwise delegate to the underlying NamedTrajectory
    else
        return getproperty(getfield(qtraj, :trajectory), symb)
    end
end

function Base.propertynames(qtraj::AbstractQuantumTrajectory)
    # Combine properties from both the wrapper and the trajectory
    wrapper_fields = fieldnames(typeof(qtraj))
    traj_props = propertynames(getfield(qtraj, :trajectory))
    return tuple(wrapper_fields..., traj_props...)
end

# Convenience accessors
state(qtraj::AbstractQuantumTrajectory) = get_trajectory(qtraj)[get_state_name(qtraj)]
controls(qtraj::AbstractQuantumTrajectory) = get_trajectory(qtraj)[get_control_name(qtraj)]

# New get_* versions
get_state(qtraj::AbstractQuantumTrajectory) = state(qtraj)
get_controls(qtraj::AbstractQuantumTrajectory) = controls(qtraj)

"""
    UnitaryTrajectory <: AbstractQuantumTrajectory

A trajectory for unitary gate synthesis problems.

# Fields
- `trajectory::NamedTrajectory`: The underlying trajectory data (stored as a copy)
- `system::QuantumSystem`: The quantum system
- `state_name::Symbol`: Name of the state variable (typically `:Ũ⃗`)
- `control_name::Symbol`: Name of the control variable (typically `:u`)
- `goal::AbstractPiccoloOperator`: Target unitary operator
"""
struct UnitaryTrajectory <: AbstractQuantumTrajectory
    trajectory::NamedTrajectory
    system::QuantumSystem
    state_name::Symbol
    control_name::Symbol
    goal::AbstractPiccoloOperator
    
    function UnitaryTrajectory(
        sys::QuantumSystem,
        U_goal::AbstractMatrix{<:Number},
        N::Int;
        U_init::AbstractMatrix{<:Number}=Matrix{ComplexF64}(I(size(sys.H_drift, 1))),
        Δt_min::Union{Float64, Nothing}=nothing,
        Δt_max::Union{Float64, Nothing}=nothing,
        Δt_bounds::Union{Tuple{Float64, Float64}, Nothing}=nothing,
        free_time::Bool=true,
        geodesic::Bool=true
    )
        Δt = sys.T_max / (N - 1)
        n_drives = sys.n_drives
        
        # Handle Δt_bounds: prioritize Δt_bounds tuple if provided, else use Δt_min/Δt_max
        if !isnothing(Δt_bounds)
            _Δt_min, _Δt_max = Δt_bounds
        else
            _Δt_min = isnothing(Δt_min) ? Δt / 2 : Δt_min
            _Δt_max = isnothing(Δt_max) ? 2 * Δt : Δt_max
        end
        
        # Initialize unitary trajectory
        if geodesic
            H_drift = Matrix(get_drift(sys))
            Ũ⃗ = unitary_geodesic(U_init, U_goal, N, H_drift=H_drift)
        else
            Ũ⃗ = unitary_linear_interpolation(U_init, U_goal, N)
        end

        # Initialize controls
        u = get_initial_controls(sys, N)
        
        # Timesteps
        Δt_vec = fill(Δt, N)
        
        # Initial and final constraints
        Ũ⃗_init = operator_to_iso_vec(U_init)
        Ũ⃗_goal = operator_to_iso_vec(U_goal)
        
        initial = (Ũ⃗ = Ũ⃗_init, u = zeros(n_drives))
        final = (u = zeros(n_drives),)
        goal_constraint = (Ũ⃗ = Ũ⃗_goal,)
        
        # Time data (automatic for time-dependent systems)
        if sys.time_dependent
            t_data = cumsum([0.0; Δt_vec[1:end-1]])
            initial = merge(initial, (t = [0.0],))
        end
        
        # Bounds - convert drive_bounds from Vector{Tuple} to Tuple of Vectors
        u_lower = [sys.drive_bounds[i][1] for i in 1:n_drives]
        u_upper = [sys.drive_bounds[i][2] for i in 1:n_drives]
        _Δt_bounds = free_time ? (_Δt_min, _Δt_max) : (Δt, Δt)
        bounds = (
            u = (u_lower, u_upper),
            Δt = _Δt_bounds
        )
        
        # Build component data
        comps_data = (Ũ⃗ = Ũ⃗, u = u, Δt = reshape(Δt_vec, 1, N))
        
        if sys.time_dependent
            comps_data = merge(comps_data, (t = reshape(t_data, 1, N),))
        end
        
        traj = NamedTrajectory(
            comps_data;
            controls = (:u, :Δt),
            timestep = :Δt,
            initial = initial,
            final = final,
            goal = goal_constraint,
            bounds = bounds
        )
        
        return new(traj, sys, :Ũ⃗, :u, U_goal)
    end
end

"""
    KetTrajectory <: AbstractQuantumTrajectory

A trajectory for quantum state transfer problems.

# Fields
- `trajectory::NamedTrajectory`: The underlying trajectory data (stored as a copy)
- `system::QuantumSystem`: The quantum system
- `state_name::Symbol`: Name of the state variable (typically `:ψ̃`)
- `control_name::Symbol`: Name of the control variable (typically `:u`)
- `goal::AbstractVector{ComplexF64}`: Target ket state

For multiple state transfers with a shared system, use `EnsembleTrajectory` wrapping
multiple `KetTrajectory` instances.
"""
struct KetTrajectory <: AbstractQuantumTrajectory
    trajectory::NamedTrajectory
    system::QuantumSystem
    state_name::Symbol
    control_name::Symbol
    goal::AbstractVector{ComplexF64}
    
    function KetTrajectory(
        sys::QuantumSystem,
        ψ_init::AbstractVector{ComplexF64},
        ψ_goal::AbstractVector{ComplexF64},
        N::Int;
        state_name::Symbol=:ψ̃,
        Δt_min::Union{Float64, Nothing}=nothing,
        Δt_max::Union{Float64, Nothing}=nothing,
        Δt_bounds::Union{Tuple{Float64, Float64}, Nothing}=nothing,
        free_time::Bool=true
    )
        Δt = sys.T_max / (N - 1)
        n_drives = sys.n_drives
        
        # Handle Δt_bounds: prioritize Δt_bounds tuple if provided, else use Δt_min/Δt_max
        if !isnothing(Δt_bounds)
            _Δt_min, _Δt_max = Δt_bounds
        else
            _Δt_min = isnothing(Δt_min) ? Δt / 2 : Δt_min
            _Δt_max = isnothing(Δt_max) ? 2 * Δt : Δt_max
        end
        
        # Convert to iso representation
        ψ̃_init = ket_to_iso(ψ_init)
        ψ̃_goal = ket_to_iso(ψ_goal)
        
        # Linear interpolation of state
        ψ̃ = linear_interpolation(ψ̃_init, ψ̃_goal, N)
        
        # Initialize controls
        u = get_initial_controls(sys, N)
        
        # Timesteps
        Δt_vec = fill(Δt, N)
        
        # Initial and final constraints
        initial = (; state_name => ψ̃_init, :u => zeros(n_drives))
        final = (u = zeros(n_drives),)
        goal_constraint = (; state_name => ψ̃_goal)
        
        # Time data (automatic for time-dependent systems)
        if sys.time_dependent
            t_data = [0.0; cumsum(Δt_vec)[1:end-1]]
            initial = merge(initial, (t = [0.0],))
        end
        
        # Bounds - convert drive_bounds from Vector{Tuple} to Tuple of Vectors
        u_lower = [sys.drive_bounds[i][1] for i in 1:n_drives]
        u_upper = [sys.drive_bounds[i][2] for i in 1:n_drives]
        _Δt_bounds = free_time ? (_Δt_min, _Δt_max) : (Δt, Δt)
        bounds = (
            u = (u_lower, u_upper),
            Δt = _Δt_bounds
        )
        
        # Build component data
        comps_data = (; state_name => ψ̃, :u => u, :Δt => reshape(Δt_vec, 1, N))
        
        if sys.time_dependent
            comps_data = merge(comps_data, (t = reshape(t_data, 1, N),))
        end
        
        traj = NamedTrajectory(
            comps_data;
            controls = (:u, :Δt),
            timestep = :Δt,
            initial = initial,
            final = final,
            goal = goal_constraint,
            bounds = bounds
        )
        
        return new(traj, sys, state_name, :u, ψ_goal)
    end
end

"""
    DensityTrajectory <: AbstractQuantumTrajectory

A trajectory for open quantum system problems.

# Fields
- `trajectory::NamedTrajectory`: The underlying trajectory data (stored as a copy)
- `system::OpenQuantumSystem`: The open quantum system
- `state_name::Symbol`: Name of the state variable (typically `:ρ⃗̃`)
- `control_name::Symbol`: Name of the control variable (typically `:u`)
- `goal::AbstractMatrix`: Target density matrix
"""
struct DensityTrajectory <: AbstractQuantumTrajectory
    trajectory::NamedTrajectory
    system::OpenQuantumSystem
    state_name::Symbol
    control_name::Symbol
    goal::AbstractMatrix
    
    function DensityTrajectory(
        sys::OpenQuantumSystem,
        ρ_init::AbstractMatrix,
        ρ_goal::AbstractMatrix,
        N::Int;
        Δt_min::Union{Float64, Nothing}=nothing,
        Δt_max::Union{Float64, Nothing}=nothing,
        Δt_bounds::Union{Tuple{Float64, Float64}, Nothing}=nothing,
        free_time::Bool=true
    )
        Δt = sys.T_max / (N - 1)
        n_drives = sys.n_drives
        
        # Handle Δt_bounds: prioritize Δt_bounds tuple if provided, else use Δt_min/Δt_max
        if !isnothing(Δt_bounds)
            _Δt_min, _Δt_max = Δt_bounds
        else
            _Δt_min = isnothing(Δt_min) ? Δt / 2 : Δt_min
            _Δt_max = isnothing(Δt_max) ? 2 * Δt : Δt_max
        end
        
        # Convert to iso representation
        ρ⃗̃_init = density_to_iso_vec(ρ_init)
        ρ⃗̃_goal = density_to_iso_vec(ρ_goal)
        
        # Linear interpolation of state
        ρ⃗̃ = linear_interpolation(ρ⃗̃_init, ρ⃗̃_goal, N)
        
        # Initialize controls
        u = get_initial_controls(sys, N)
        
        # Timesteps
        Δt_vec = fill(Δt, N)
        
        # Initial and final constraints
        initial = (ρ⃗̃ = ρ⃗̃_init, u = zeros(n_drives))
        final = (u = zeros(n_drives),)
        goal_constraint = (ρ⃗̃ = ρ⃗̃_goal,)
        
        # Time data (automatic for time-dependent systems)
        if sys.time_dependent
            t_data = [0.0; cumsum(Δt_vec)[1:end-1]]
            initial = merge(initial, (t = [0.0],))
        end
        
        # Bounds - convert drive_bounds from Vector{Tuple} to Tuple of Vectors
        u_lower = [sys.drive_bounds[i][1] for i in 1:n_drives]
        u_upper = [sys.drive_bounds[i][2] for i in 1:n_drives]
        _Δt_bounds = free_time ? (_Δt_min, _Δt_max) : (Δt, Δt)
        bounds = (
            u = (u_lower, u_upper),
            Δt = _Δt_bounds
        )
        
        # Build component data
        comps_data = (ρ⃗̃ = ρ⃗̃, u = u, Δt = reshape(Δt_vec, 1, N))
        
        if sys.time_dependent
            comps_data = merge(comps_data, (t = reshape(t_data, 1, N),))
        end
        
        traj = NamedTrajectory(
            comps_data;
            controls = (:u, :Δt),
            timestep = :Δt,
            initial = initial,
            final = final,
            goal = goal_constraint,
            bounds = bounds
        )
        
        return new(traj, sys, :ρ⃗̃, :u, ρ_goal)
    end
end

# ============================================================================= #
#                          Sampling Trajectory Type                             #
# ============================================================================= #

"""
    SamplingTrajectory{T<:AbstractQuantumTrajectory} <: AbstractQuantumTrajectory

A trajectory wrapper for robust/sampling optimization over an ensemble of systems.

This type wraps a base quantum trajectory and extends it to handle multiple quantum
systems with different parameters. Each system in the ensemble gets its own state 
variable (e.g., `Ũ⃗_sample_1`, `Ũ⃗_sample_2`), while controls are shared across all systems.

Use this for:
- Robust optimization over parameter uncertainty
- Ensemble control where different physical systems share the same pulse
- Sampling-based optimization over system variations

# Fields
- `base_trajectory::T`: The base quantum trajectory (nominal system)
- `systems::Vector{<:AbstractQuantumSystem}`: The ensemble of systems to optimize over
- `weights::Vector{Float64}`: Weights for each system in the objective
- `sample_state_names::Vector{Symbol}`: Names of state variables for each system

# Accessors
- `get_systems(qtraj)`: Return all systems in the ensemble
- `get_weights(qtraj)`: Return the weights for each system
- `get_ensemble_state_names(qtraj)`: Return the state variable names for each system

The standard `AbstractQuantumTrajectory` interface methods forward to the base trajectory:
- `get_trajectory(qtraj)`: Returns the base trajectory's NamedTrajectory
- `get_system(qtraj)`: Returns the nominal (first) system
- `get_goal(qtraj)`: Returns the goal from the base trajectory
- `get_state_name(qtraj)`: Returns the base state name (not sample names)
- `get_control_name(qtraj)`: Returns the control name

# Example
```julia
# Create base trajectory with nominal system
sys_nominal = QuantumSystem(H_drift, H_drives, T, bounds)
qtraj = UnitaryTrajectory(sys_nominal, U_goal, N)

# Create sampling trajectory with perturbed systems for robust optimization
sys_perturbed = QuantumSystem(1.1 * H_drift, H_drives, T, bounds)
systems = [sys_nominal, sys_perturbed]

sampling_traj = SamplingTrajectory(qtraj, systems)
```

See also: [`EnsembleTrajectory`](@ref) for multiple initial/goal states with a shared system.
"""
struct SamplingTrajectory{T<:AbstractQuantumTrajectory} <: AbstractQuantumTrajectory
    base_trajectory::T
    systems::Vector{<:AbstractQuantumSystem}
    weights::Vector{Float64}
    sample_state_names::Vector{Symbol}
    
    function SamplingTrajectory(
        base_trajectory::T,
        systems::Vector{<:AbstractQuantumSystem};
        weights::Vector{Float64}=fill(1.0, length(systems))
    ) where {T<:AbstractQuantumTrajectory}
        @assert length(weights) == length(systems) "weights must match number of systems"
        
        state_sym = get_state_name(base_trajectory)
        state_names = _sample_state_names(state_sym, length(systems))
        
        return new{T}(base_trajectory, systems, weights, state_names)
    end
    
    # Inner constructor for direct field initialization
    function SamplingTrajectory{T}(
        base_trajectory::T,
        systems::Vector{<:AbstractQuantumSystem},
        weights::Vector{Float64},
        sample_state_names::Vector{Symbol}
    ) where {T<:AbstractQuantumTrajectory}
        return new{T}(base_trajectory, systems, weights, sample_state_names)
    end
end

# Non-parametric constructor for explicit state names
function SamplingTrajectory(
    base_trajectory::T,
    systems::Vector{<:AbstractQuantumSystem},
    weights::Vector{Float64},
    sample_state_names::Vector{Symbol}
) where {T<:AbstractQuantumTrajectory}
    return SamplingTrajectory{T}(base_trajectory, systems, weights, sample_state_names)
end

# ============================================================================= #
# SamplingTrajectory Interface Implementation
# ============================================================================= #

# Forward AbstractQuantumTrajectory interface to base_trajectory
get_trajectory(qtraj::SamplingTrajectory) = get_trajectory(qtraj.base_trajectory)
get_system(qtraj::SamplingTrajectory) = get_system(qtraj.base_trajectory)  # Nominal system
get_goal(qtraj::SamplingTrajectory) = get_goal(qtraj.base_trajectory)
get_state_name(qtraj::SamplingTrajectory) = get_state_name(qtraj.base_trajectory)
get_control_name(qtraj::SamplingTrajectory) = get_control_name(qtraj.base_trajectory)

# Sampling-specific accessors
"""
    get_systems(qtraj::SamplingTrajectory) -> Vector{<:AbstractQuantumSystem}

Return all systems in the sampling ensemble.
"""
get_systems(qtraj::SamplingTrajectory) = qtraj.systems

"""
    get_weights(qtraj::SamplingTrajectory) -> Vector{Float64}

Return the weights for each system in the sampling ensemble.
"""
get_weights(qtraj::SamplingTrajectory) = qtraj.weights

"""
    get_ensemble_state_names(qtraj::SamplingTrajectory) -> Vector{Symbol}

Return the state variable names for each system in the sampling ensemble.
"""
get_ensemble_state_names(qtraj::SamplingTrajectory) = qtraj.sample_state_names

# Forward indexing and property access
Base.getindex(qtraj::SamplingTrajectory, key) = getindex(qtraj.base_trajectory, key)
Base.setindex!(qtraj::SamplingTrajectory, value, key) = setindex!(qtraj.base_trajectory, value, key)

function Base.getproperty(qtraj::SamplingTrajectory, symb::Symbol)
    if symb ∈ fieldnames(SamplingTrajectory)
        return getfield(qtraj, symb)
    else
        return getproperty(getfield(qtraj, :base_trajectory), symb)
    end
end

function Base.propertynames(qtraj::SamplingTrajectory)
    return tuple(fieldnames(SamplingTrajectory)..., propertynames(getfield(qtraj, :base_trajectory))...)
end

# ============================================================================= #
#                          Ensemble Trajectory Type                             #
# ============================================================================= #

"""
    EnsembleTrajectory{T<:AbstractQuantumTrajectory} <: AbstractQuantumTrajectory

A trajectory wrapper for ensemble optimization over multiple initial/goal states with a shared system.

This type wraps multiple quantum trajectories that share the same system but have different
initial and goal states. Each trajectory in the ensemble gets its own state variable 
(e.g., `Ũ⃗_init_1`, `Ũ⃗_init_2`), while controls and the system are shared.

Use this for:
- Training a gate on the computational basis states
- Multi-state transfer problems
- Any scenario where you want the same pulse to achieve different state→goal mappings

# Fields
- `trajectories::Vector{T}`: The quantum trajectories (each with different init/goal)
- `weights::Vector{Float64}`: Weights for each trajectory in the objective
- `ensemble_state_names::Vector{Symbol}`: Names of state variables for each trajectory

# Accessors
- `get_weights(qtraj)`: Return the weights for each trajectory
- `get_ensemble_state_names(qtraj)`: Return the state variable names for each trajectory

The standard `AbstractQuantumTrajectory` interface methods use the first trajectory:
- `get_trajectory(qtraj)`: Returns the first trajectory's NamedTrajectory
- `get_system(qtraj)`: Returns the shared system
- `get_goal(qtraj)`: Returns goals from all trajectories
- `get_state_name(qtraj)`: Returns the base state name
- `get_control_name(qtraj)`: Returns the control name

# Example
```julia
# Create system
sys = QuantumSystem(H_drift, H_drives, T, bounds)

# Create trajectories for different state transfers
qtraj1 = KetTrajectory(sys, ψ0, ψ1, N)  # |0⟩ → |1⟩
qtraj2 = KetTrajectory(sys, ψ1, ψ0, N)  # |1⟩ → |0⟩

ensemble_traj = EnsembleTrajectory([qtraj1, qtraj2])
```

See also: [`SamplingTrajectory`](@ref) for optimization over different systems.
"""
struct EnsembleTrajectory{T<:AbstractQuantumTrajectory} <: AbstractQuantumTrajectory
    trajectories::Vector{T}
    weights::Vector{Float64}
    ensemble_state_names::Vector{Symbol}
    
    function EnsembleTrajectory(
        trajectories::Vector{T};
        weights::Vector{Float64}=fill(1.0, length(trajectories))
    ) where {T<:AbstractQuantumTrajectory}
        @assert length(trajectories) > 0 "Must provide at least one trajectory"
        @assert length(weights) == length(trajectories) "weights must match number of trajectories"
        
        # Verify all trajectories share the same system
        sys = get_system(trajectories[1])
        for qtraj in trajectories[2:end]
            @assert get_system(qtraj) === sys "All trajectories must share the same system"
        end
        
        state_sym = get_state_name(trajectories[1])
        state_names = _ensemble_state_names(state_sym, length(trajectories))
        
        return new{T}(trajectories, weights, state_names)
    end
    
    # Inner constructor for direct field initialization
    function EnsembleTrajectory{T}(
        trajectories::Vector{T},
        weights::Vector{Float64},
        ensemble_state_names::Vector{Symbol}
    ) where {T<:AbstractQuantumTrajectory}
        return new{T}(trajectories, weights, ensemble_state_names)
    end
end

# Non-parametric constructor for explicit state names
function EnsembleTrajectory(
    trajectories::Vector{T},
    weights::Vector{Float64},
    ensemble_state_names::Vector{Symbol}
) where {T<:AbstractQuantumTrajectory}
    return EnsembleTrajectory{T}(trajectories, weights, ensemble_state_names)
end

# ============================================================================= #
# EnsembleTrajectory Interface Implementation
# ============================================================================= #

# Forward AbstractQuantumTrajectory interface to first trajectory
get_trajectory(qtraj::EnsembleTrajectory) = get_trajectory(qtraj.trajectories[1])
get_system(qtraj::EnsembleTrajectory) = get_system(qtraj.trajectories[1])  # Shared system
get_state_name(qtraj::EnsembleTrajectory) = get_state_name(qtraj.trajectories[1])
get_control_name(qtraj::EnsembleTrajectory) = get_control_name(qtraj.trajectories[1])

# Get all goals from all trajectories
get_goal(qtraj::EnsembleTrajectory) = [get_goal(t) for t in qtraj.trajectories]

# Ensemble-specific accessors
"""
    get_weights(qtraj::EnsembleTrajectory) -> Vector{Float64}

Return the weights for each trajectory in the ensemble.
"""
get_weights(qtraj::EnsembleTrajectory) = qtraj.weights

"""
    get_ensemble_state_names(qtraj::EnsembleTrajectory) -> Vector{Symbol}

Return the state variable names for each trajectory in the ensemble.
"""
get_ensemble_state_names(qtraj::EnsembleTrajectory) = qtraj.ensemble_state_names

# Forward indexing and property access to first trajectory
Base.getindex(qtraj::EnsembleTrajectory, key) = getindex(qtraj.trajectories[1], key)
Base.setindex!(qtraj::EnsembleTrajectory, value, key) = setindex!(qtraj.trajectories[1], value, key)

function Base.getproperty(qtraj::EnsembleTrajectory, symb::Symbol)
    if symb ∈ fieldnames(EnsembleTrajectory)
        return getfield(qtraj, symb)
    else
        return getproperty(getfield(qtraj, :trajectories)[1], symb)
    end
end

function Base.propertynames(qtraj::EnsembleTrajectory)
    return tuple(fieldnames(EnsembleTrajectory)..., propertynames(getfield(qtraj, :trajectories)[1])...)
end

# ============================================================================= #
# Shared Trajectory Utilities
# ============================================================================= #

"""
    _sample_state_names(state_sym::Symbol, n_samples::Int) -> Vector{Symbol}

Generate state variable names for each sample in a sampling trajectory.

# Examples
```julia
_sample_state_names(:Ũ⃗, 3)  # [:Ũ⃗_sample_1, :Ũ⃗_sample_2, :Ũ⃗_sample_3]
```
"""
_sample_state_names(state_sym::Symbol, n_samples::Int) = 
    [Symbol(state_sym, :_sample_, i) for i in 1:n_samples]

"""
    _ensemble_state_names(state_sym::Symbol, n_members::Int) -> Vector{Symbol}

Generate state variable names for each member in an ensemble trajectory.

# Examples
```julia
_ensemble_state_names(:Ũ⃗, 3)  # [:Ũ⃗_init_1, :Ũ⃗_init_2, :Ũ⃗_init_3]
```
"""
_ensemble_state_names(state_sym::Symbol, n_members::Int) = 
    [Symbol(state_sym, :_init_, i) for i in 1:n_members]

"""
    build_sampling_trajectory(
        base_traj::NamedTrajectory,
        state_sym::Symbol,
        n_samples::Int
    ) -> Tuple{NamedTrajectory, Vector{Symbol}}

Create a new trajectory with duplicated state variables for sampling optimization.

This function takes a base trajectory and creates a new one where:
- All non-state components (controls, derivatives, timesteps) are shared
- The state variable is duplicated for each sample with names like `:Ũ⃗_sample_1`, `:Ũ⃗_sample_2`
- Initial conditions are duplicated for each sample's state
- Goals are duplicated for each sample's state

Returns a tuple of (new_trajectory, sample_state_names).

# Arguments
- `base_traj::NamedTrajectory`: The base trajectory to duplicate
- `state_sym::Symbol`: The name of the state variable to duplicate
- `n_samples::Int`: Number of samples in the ensemble

# Returns
- `Tuple{NamedTrajectory, Vector{Symbol}}`: New trajectory and list of sample state names

# Example
```julia
base_traj = get_trajectory(qtraj)
new_traj, state_names = build_sampling_trajectory(base_traj, :Ũ⃗, 3)
# state_names = [:Ũ⃗_sample_1, :Ũ⃗_sample_2, :Ũ⃗_sample_3]
```
"""
function build_sampling_trajectory(
    base_traj::NamedTrajectory,
    state_sym::Symbol,
    n_samples::Int
)
    state_names = _sample_state_names(state_sym, n_samples)
    
    # Shared components (controls, derivatives, etc.)
    shared_components = (
        name => base_traj[name] 
        for name in base_traj.names if name != state_sym
    )
    
    # Duplicated state components for each sample
    sample_components = (
        name => copy(base_traj[state_sym]) 
        for name in state_names
    )
    
    # Merge all components
    new_components = merge(
        NamedTuple(shared_components),
        NamedTuple(sample_components)
    )
    
    # Build initial conditions: copy non-state, duplicate state
    base_init_state = base_traj.initial[state_sym]
    init_shared = (k => v for (k, v) in pairs(base_traj.initial) if k != state_sym)
    init_sample = (name => base_init_state for name in state_names)
    new_initial = merge(NamedTuple(init_shared), NamedTuple(init_sample))
    
    # Build final conditions (without state - handled by objective)
    new_final = NamedTuple(k => v for (k, v) in pairs(base_traj.final) if k != state_sym)
    
    # Build goal conditions: duplicate if present
    new_goal = if haskey(base_traj.goal, state_sym)
        base_goal = base_traj.goal[state_sym]
        NamedTuple(name => base_goal for name in state_names)
    else
        NamedTuple()
    end
    
    new_traj = NamedTrajectory(
        new_components;
        controls=base_traj.control_names,
        timestep=base_traj.timestep,
        initial=new_initial,
        final=new_final,
        goal=new_goal,
        bounds=base_traj.bounds
    )
    
    return new_traj, state_names
end

# Alias for backward compatibility
const build_ensemble_trajectory = build_sampling_trajectory

"""
    build_ensemble_trajectory_from_trajectories(
        trajectories::Vector{<:AbstractQuantumTrajectory}
    ) -> Tuple{NamedTrajectory, Vector{Symbol}}

Create a new trajectory from multiple quantum trajectories with different initial/goal states.

Each trajectory's state is included with names like `:Ũ⃗_init_1`, `:Ũ⃗_init_2`.
Controls and other components are taken from the first trajectory.

# Arguments
- `trajectories::Vector{<:AbstractQuantumTrajectory}`: Trajectories with shared system

# Returns
- `Tuple{NamedTrajectory, Vector{Symbol}}`: New trajectory and list of ensemble state names
"""
function build_ensemble_trajectory_from_trajectories(
    trajectories::Vector{<:AbstractQuantumTrajectory}
)
    @assert length(trajectories) > 0 "Must provide at least one trajectory"
    
    state_sym = get_state_name(trajectories[1])
    state_names = _ensemble_state_names(state_sym, length(trajectories))
    
    base_traj = get_trajectory(trajectories[1])
    
    # Shared components from first trajectory (controls, derivatives, etc.)
    shared_components = (
        name => base_traj[name] 
        for name in base_traj.names if name != state_sym
    )
    
    # State components from each trajectory
    ensemble_components = (
        state_names[i] => copy(get_trajectory(trajectories[i])[state_sym])
        for i in 1:length(trajectories)
    )
    
    # Merge all components
    new_components = merge(
        NamedTuple(shared_components),
        NamedTuple(ensemble_components)
    )
    
    # Build initial conditions from each trajectory
    init_shared = (k => v for (k, v) in pairs(base_traj.initial) if k != state_sym)
    init_ensemble = (
        state_names[i] => get_trajectory(trajectories[i]).initial[state_sym]
        for i in 1:length(trajectories)
    )
    new_initial = merge(NamedTuple(init_shared), NamedTuple(init_ensemble))
    
    # Build final conditions (without state - handled by objective)
    new_final = NamedTuple(k => v for (k, v) in pairs(base_traj.final) if k != state_sym)
    
    # Build goal conditions from each trajectory
    new_goal = if haskey(base_traj.goal, state_sym)
        NamedTuple(
            state_names[i] => get_trajectory(trajectories[i]).goal[state_sym]
            for i in 1:length(trajectories)
        )
    else
        NamedTuple()
    end
    
    new_traj = NamedTrajectory(
        new_components;
        controls=base_traj.control_names,
        timestep=base_traj.timestep,
        initial=new_initial,
        final=new_final,
        goal=new_goal,
        bounds=base_traj.bounds
    )
    
    return new_traj, state_names
end

"""
    build_ensemble_trajectory(qtraj::EnsembleTrajectory) -> Tuple{NamedTrajectory, Vector{Symbol}}

Build a combined NamedTrajectory from an EnsembleTrajectory.

Creates a trajectory with separate state variables for each member (e.g., `:ψ̃_init_1`, `:ψ̃_init_2`).
Controls and other components are shared.

# Returns
- `Tuple{NamedTrajectory, Vector{Symbol}}`: Combined trajectory and list of state names
"""
function build_ensemble_trajectory(qtraj::EnsembleTrajectory)
    return build_ensemble_trajectory_from_trajectories(qtraj.trajectories)
end

"""
    get_trajectory(qtraj::EnsembleTrajectory) -> NamedTrajectory

Return the combined trajectory with all ensemble states.

This builds a NamedTrajectory with separate state variables for each ensemble member.
For efficiency, consider caching this result if calling multiple times.
"""
function get_combined_trajectory(qtraj::EnsembleTrajectory)
    traj, _ = build_ensemble_trajectory(qtraj)
    return traj
end

"""
    update_base_trajectory(qtraj::SamplingTrajectory, new_base::T) where {T<:AbstractQuantumTrajectory}

Create a new SamplingTrajectory with an updated base trajectory.
Preserves the systems, weights, and sample state names.
"""
function update_base_trajectory(
    qtraj::SamplingTrajectory{T}, 
    new_base::T
) where {T<:AbstractQuantumTrajectory}
    return SamplingTrajectory{T}(
        new_base, 
        qtraj.systems, 
        qtraj.weights, 
        qtraj.sample_state_names
    )
end

# ============================================================================= #
# Tests
# ============================================================================= #


@testitem "UnitaryTrajectory high-level constructor" begin
    using PiccoloQuantumObjects
    using NamedTrajectories
    
    # Create a simple quantum system
    sys = QuantumSystem(
        GATES[:Z],              # H_drift
        [GATES[:X], GATES[:Y]], # H_drives
        1.0,                    # T_max
        [1.0, 1.0];             # drive_bounds
        time_dependent=false
    )
    
    N = 10
    U_goal = GATES[:H]
    
    # Test basic constructor
    qtraj = UnitaryTrajectory(sys, U_goal, N)
    @test qtraj isa UnitaryTrajectory
    @test size(qtraj[:Ũ⃗], 2) == N
    @test size(qtraj[:u], 2) == N
    @test size(qtraj[:u], 1) == 2  # 2 drives
    @test get_system(qtraj) === sys
    @test get_goal(qtraj) === U_goal
    @test get_state_name(qtraj) == :Ũ⃗
    @test get_control_name(qtraj) == :u
    
    # Test with custom initial unitary
    U_init = GATES[:I]
    qtraj2 = UnitaryTrajectory(sys, U_goal, N; U_init=U_init)
    @test qtraj2 isa UnitaryTrajectory
    @test size(qtraj2[:Ũ⃗], 2) == N
    
    # Test with fixed time (free_time=false)
    qtraj3 = UnitaryTrajectory(sys, U_goal, N; free_time=false)
    @test qtraj3 isa UnitaryTrajectory
    Δt_val = sys.T_max / (N - 1)
    @test qtraj3.bounds[:Δt][1][1] == Δt_val
    @test qtraj3.bounds[:Δt][2][1] == Δt_val
    
    # Test with custom Δt bounds
    qtraj4 = UnitaryTrajectory(sys, U_goal, N; Δt_min=0.05, Δt_max=0.2)
    @test qtraj4 isa UnitaryTrajectory
    @test qtraj4.bounds[:Δt][1][1] == 0.05
    @test qtraj4.bounds[:Δt][2][1] == 0.2
    
    # Test that time is NOT stored for non-time-dependent systems
    @test !haskey(qtraj.components, :t)
    @test :t ∉ keys(qtraj.components)
    
    # Test with linear interpolation (geodesic=false)
    qtraj6 = UnitaryTrajectory(sys, U_goal, N; geodesic=false)
    @test qtraj6 isa UnitaryTrajectory
    @test size(qtraj6[:Ũ⃗], 2) == N
end

@testitem "KetTrajectory high-level constructor" begin
    using PiccoloQuantumObjects
    using NamedTrajectories
    
    # Create a simple quantum system
    sys = QuantumSystem(
        GATES[:Z],              # H_drift
        [GATES[:X], GATES[:Y]], # H_drives
        1.0,                    # T_max
        [1.0, 1.0]             # drive_bounds
    )
    
    N = 10
    ψ_init = ComplexF64[1.0, 0.0]
    ψ_goal = ComplexF64[0.0, 1.0]
    
    # Test single state constructor
    qtraj = KetTrajectory(sys, ψ_init, ψ_goal, N)
    @test qtraj isa KetTrajectory
    @test size(qtraj[:ψ̃], 2) == N
    @test size(qtraj[:u], 2) == N
    @test size(qtraj[:u], 1) == 2  # 2 drives
    @test get_system(qtraj) === sys
    @test get_goal(qtraj) == ψ_goal
    @test get_state_name(qtraj) == :ψ̃
    @test get_control_name(qtraj) == :u
    
    # Test with fixed time
    qtraj3 = KetTrajectory(sys, ψ_init, ψ_goal, N; free_time=false)
    @test qtraj3 isa KetTrajectory
    Δt_val = sys.T_max / (N - 1)
    @test qtraj3.bounds[:Δt][1][1] == Δt_val
    @test qtraj3.bounds[:Δt][2][1] == Δt_val
    
    # Test with custom Δt bounds
    qtraj4 = KetTrajectory(sys, ψ_init, ψ_goal, N; Δt_min=0.05, Δt_max=0.2)
    @test qtraj4 isa KetTrajectory
    @test qtraj4.bounds[:Δt][1][1] == 0.05
    @test qtraj4.bounds[:Δt][2][1] == 0.2
    
    # Test with custom state name
    qtraj5 = KetTrajectory(sys, ψ_init, ψ_goal, N; state_name=:ψ̃_custom)
    @test qtraj5 isa KetTrajectory
    @test get_state_name(qtraj5) == :ψ̃_custom
    @test haskey(qtraj5.components, :ψ̃_custom)
    
    # Test that time is NOT stored for non-time-dependent systems
    @test !haskey(qtraj.components, :t)
    @test :t ∉ keys(qtraj.components)
end

@testitem "DensityTrajectory high-level constructor" begin
    using PiccoloQuantumObjects
    using NamedTrajectories
    
    # Create an open quantum system
    sys = OpenQuantumSystem(
        GATES[:Z],              # H_drift
        [GATES[:X], GATES[:Y]], # H_drives
        1.0,                    # T_max
        [1.0, 1.0]             # drive_bounds
    )
    
    N = 10
    ρ_init = ComplexF64[1.0 0.0; 0.0 0.0]  # |0⟩⟨0|
    ρ_goal = ComplexF64[0.0 0.0; 0.0 1.0]  # |1⟩⟨1|
    
    # Test basic constructor
    qtraj = DensityTrajectory(sys, ρ_init, ρ_goal, N)
    @test qtraj isa DensityTrajectory
    @test size(qtraj[:ρ⃗̃], 2) == N
    @test size(qtraj[:u], 2) == N
    @test size(qtraj[:u], 1) == 2  # 2 drives
    @test get_system(qtraj) === sys
    @test get_goal(qtraj) == ρ_goal
    @test get_state_name(qtraj) == :ρ⃗̃
    @test get_control_name(qtraj) == :u
    
    # Test with fixed time
    qtraj3 = DensityTrajectory(sys, ρ_init, ρ_goal, N; free_time=false)
    @test qtraj3 isa DensityTrajectory
    Δt_val = sys.T_max / (N - 1)
    @test qtraj3.bounds[:Δt][1][1] == Δt_val
    @test qtraj3.bounds[:Δt][2][1] == Δt_val
    
    # Test with custom Δt bounds
    qtraj4 = DensityTrajectory(sys, ρ_init, ρ_goal, N; Δt_min=0.05, Δt_max=0.2)
    @test qtraj4 isa DensityTrajectory
    @test qtraj4.bounds[:Δt][1][1] == 0.05
    @test qtraj4.bounds[:Δt][2][1] == 0.2
    
    # Test that time is NOT stored for non-time-dependent systems
    @test !haskey(qtraj.components, :t)
    @test :t ∉ keys(qtraj.components)
end

@testitem "Time-dependent Hamiltonians with UnitaryTrajectory" begin
    using PiccoloQuantumObjects
    using NamedTrajectories
    using LinearAlgebra
    
    # Create time-dependent Hamiltonian: H(u, t) = H_drift + u(t) * cos(ω*t) * H_drive
    ω = 2π * 5.0  # Drive frequency
    H_drift = GATES[:Z]
    H_drive = GATES[:X]
    
    H(u, t) = H_drift + u[1] * cos(ω * t) * H_drive
    
    # Create system with time-dependent flag
    sys = QuantumSystem(H, 1.0, [1.0]; time_dependent=true)
    
    N = 10
    U_goal = GATES[:H]
    
    # Test that time storage is automatic for time-dependent systems
    qtraj = UnitaryTrajectory(sys, U_goal, N)
    @test qtraj isa UnitaryTrajectory
    @test haskey(qtraj.components, :t)
    @test size(qtraj[:t], 2) == N
    @test qtraj[:t][1] ≈ 0.0
    @test qtraj.initial[:t][1] ≈ 0.0
    
    # Verify time values are cumulative sums of Δt
    Δt_cumsum = [0.0; cumsum(qtraj[:Δt][:])[1:end-1]]
    @test qtraj[:t][:] ≈ Δt_cumsum
    
    # Test with custom time bounds
    qtraj2 = UnitaryTrajectory(sys, U_goal, N; Δt_min=0.05, Δt_max=0.15)
    @test qtraj2 isa UnitaryTrajectory
    @test haskey(qtraj2.components, :t)
    
    # Test that time is included in components (but not controls)
    @test :t ∈ keys(qtraj.components)
    @test :t ∉ qtraj.control_names
end

@testitem "Time-dependent Hamiltonians with KetTrajectory" begin
    using PiccoloQuantumObjects
    using NamedTrajectories
    using LinearAlgebra
    
    # Create time-dependent Hamiltonian
    ω = 2π * 5.0
    H_drift = GATES[:Z]
    H_drive = GATES[:X]
    
    H(u, t) = H_drift + u[1] * cos(ω * t) * H_drive
    
    # Create system with time-dependent flag
    sys = QuantumSystem(H, 1.0, [1.0]; time_dependent=true)
    
    N = 10
    ψ_init = ComplexF64[1.0, 0.0]
    ψ_goal = ComplexF64[0.0, 1.0]
    
    # Test single state with time-dependent system (automatic time storage)
    qtraj = KetTrajectory(sys, ψ_init, ψ_goal, N)
    @test qtraj isa KetTrajectory
    @test haskey(qtraj.components, :t)
    @test size(qtraj[:t], 2) == N
    @test qtraj[:t][1] ≈ 0.0
    @test qtraj.initial[:t][1] ≈ 0.0
    
    # Verify time values are cumulative sums of Δt
    Δt_cumsum = [0.0; cumsum(qtraj[:Δt][:])[1:end-1]]
    @test qtraj[:t][:] ≈ Δt_cumsum
    
    # Test that time is included in components (but not controls)
    @test :t ∈ keys(qtraj.components)
    @test :t ∉ qtraj.control_names
end

@testitem "Time-dependent Hamiltonians with DensityTrajectory" begin
    using PiccoloQuantumObjects
    using NamedTrajectories
    using LinearAlgebra
    
    # Create time-dependent Hamiltonian
    ω = 2π * 5.0
    H_drift = GATES[:Z]
    H_drive = GATES[:X]
    
    H(u, t) = H_drift + u[1] * cos(ω * t) * H_drive
    
    # Create open system with time-dependent Hamiltonian
    sys = OpenQuantumSystem(H, 1.0, [1.0]; time_dependent=true)
    
    N = 10
    ρ_init = ComplexF64[1.0 0.0; 0.0 0.0]  # |0⟩⟨0|
    ρ_goal = ComplexF64[0.0 0.0; 0.0 1.0]  # |1⟩⟨1|
    
    # Test with time-dependent system (automatic time storage)
    qtraj = DensityTrajectory(sys, ρ_init, ρ_goal, N)
    @test qtraj isa DensityTrajectory
    @test haskey(qtraj.components, :t)
    @test size(qtraj[:t], 2) == N
    @test qtraj[:t][1] ≈ 0.0
    @test qtraj.initial[:t][1] ≈ 0.0
    
    # Verify time values are cumulative sums of Δt
    Δt_cumsum = [0.0; cumsum(qtraj[:Δt][:])[1:end-1]]
    @test qtraj[:t][:] ≈ Δt_cumsum
    
    # Test that time is included in components (but not controls)
    @test :t ∈ keys(qtraj.components)
    @test :t ∉ qtraj.control_names
end

@testitem "Multiple drives with time-dependent Hamiltonians" begin
    using PiccoloQuantumObjects
    using NamedTrajectories
    using LinearAlgebra
    
    # Create time-dependent Hamiltonian with multiple drives
    ω1 = 2π * 5.0
    ω2 = 2π * 3.0
    H_drift = GATES[:Z]
    H_drives = [GATES[:X], GATES[:Y]]
    
    H(u, t) = H_drift + u[1] * cos(ω1 * t) * H_drives[1] + u[2] * cos(ω2 * t) * H_drives[2]
    
    # Create system with multiple drives
    sys = QuantumSystem(H, 1.0, [1.0, 1.0]; time_dependent=true)
    
    N = 10
    U_goal = GATES[:H]
    
    # Test with multiple drives (automatic time storage)
    qtraj = UnitaryTrajectory(sys, U_goal, N)
    @test qtraj isa UnitaryTrajectory
    @test size(qtraj[:u], 1) == 2  # 2 drives
    @test haskey(qtraj.components, :t)
    @test size(qtraj[:t], 2) == N
    
    # Test initial and final control constraints
    @test all(qtraj[:u][:, 1] .== 0.0)  # Initial controls are zero
    @test all(qtraj[:u][:, end] .== 0.0)  # Final controls are zero
    
    # Test bounds on multiple drives
    @test qtraj.bounds[:u][1] == [-1.0, -1.0]  # Lower bounds
    @test qtraj.bounds[:u][2] == [1.0, 1.0]    # Upper bounds
end

@testitem "SamplingTrajectory basic construction" begin
    using PiccoloQuantumObjects
    using NamedTrajectories
    
    # Create systems (different drift Hamiltonians for robust optimization)
    sys1 = QuantumSystem(GATES[:Z], [GATES[:X]], 1.0, [1.0])
    sys2 = QuantumSystem(1.1 * GATES[:Z], [GATES[:X]], 1.0, [1.0])
    systems = [sys1, sys2]
    
    # Create base trajectory
    U_goal = GATES[:H]
    N = 10
    base_qtraj = UnitaryTrajectory(sys1, U_goal, N)
    
    # Create sampling trajectory for robust optimization
    sampling_qtraj = SamplingTrajectory(base_qtraj, systems)
    
    @test sampling_qtraj isa SamplingTrajectory{UnitaryTrajectory}
    @test sampling_qtraj isa AbstractQuantumTrajectory
    
    # Test accessors
    @test get_system(sampling_qtraj) === sys1  # Nominal system
    @test get_systems(sampling_qtraj) === systems
    @test get_goal(sampling_qtraj) === U_goal
    @test get_state_name(sampling_qtraj) == :Ũ⃗
    @test get_control_name(sampling_qtraj) == :u
    @test get_weights(sampling_qtraj) == [1.0, 1.0]
    @test get_ensemble_state_names(sampling_qtraj) == [:Ũ⃗_sample_1, :Ũ⃗_sample_2]
    
    # Test property forwarding
    @test sampling_qtraj.base_trajectory === base_qtraj
    @test sampling_qtraj.systems === systems
    @test size(sampling_qtraj[:Ũ⃗], 2) == N  # Forwarded to base trajectory
    @test size(sampling_qtraj[:u], 2) == N
end

@testitem "SamplingTrajectory with custom weights" begin
    using PiccoloQuantumObjects
    using NamedTrajectories
    
    # Create systems with different perturbations
    sys1 = QuantumSystem(GATES[:Z], [GATES[:X]], 1.0, [1.0])
    sys2 = QuantumSystem(1.1 * GATES[:Z], [GATES[:X]], 1.0, [1.0])
    sys3 = QuantumSystem(0.9 * GATES[:Z], [GATES[:X]], 1.0, [1.0])
    systems = [sys1, sys2, sys3]
    
    # Custom weights (nominal gets higher weight)
    weights = [0.5, 0.25, 0.25]
    
    # Create base trajectory
    base_qtraj = UnitaryTrajectory(sys1, GATES[:X], 10)
    
    # Create sampling trajectory with weights
    sampling_qtraj = SamplingTrajectory(base_qtraj, systems; weights=weights)
    
    @test get_weights(sampling_qtraj) == weights
    @test length(get_ensemble_state_names(sampling_qtraj)) == 3
    @test get_ensemble_state_names(sampling_qtraj) == [:Ũ⃗_sample_1, :Ũ⃗_sample_2, :Ũ⃗_sample_3]
end

@testitem "SamplingTrajectory with KetTrajectory" begin
    using PiccoloQuantumObjects
    using NamedTrajectories
    
    # Create systems
    sys1 = QuantumSystem(GATES[:Z], [GATES[:X]], 1.0, [1.0])
    sys2 = QuantumSystem(1.1 * GATES[:Z], [GATES[:X]], 1.0, [1.0])
    systems = [sys1, sys2]
    
    # Create base ket trajectory
    ψ_init = ComplexF64[1.0, 0.0]
    ψ_goal = ComplexF64[0.0, 1.0]
    base_qtraj = KetTrajectory(sys1, ψ_init, ψ_goal, 10)
    
    # Create sampling trajectory
    sampling_qtraj = SamplingTrajectory(base_qtraj, systems)
    
    @test sampling_qtraj isa SamplingTrajectory{KetTrajectory}
    @test get_state_name(sampling_qtraj) == :ψ̃
    @test get_ensemble_state_names(sampling_qtraj) == [:ψ̃_sample_1, :ψ̃_sample_2]
    @test get_goal(sampling_qtraj) == ψ_goal
end

@testitem "EnsembleTrajectory basic construction" begin
    using PiccoloQuantumObjects
    using NamedTrajectories
    
    # Create shared system
    sys = QuantumSystem(GATES[:Z], [GATES[:X]], 1.0, [1.0])
    
    # Create trajectories with different initial/goal states (same system)
    ψ0 = ComplexF64[1.0, 0.0]
    ψ1 = ComplexF64[0.0, 1.0]
    
    qtraj1 = KetTrajectory(sys, ψ0, ψ1, 10)  # |0⟩ → |1⟩
    qtraj2 = KetTrajectory(sys, ψ1, ψ0, 10)  # |1⟩ → |0⟩
    
    # Create ensemble trajectory
    ensemble_qtraj = EnsembleTrajectory([qtraj1, qtraj2])
    
    @test ensemble_qtraj isa EnsembleTrajectory{KetTrajectory}
    @test ensemble_qtraj isa AbstractQuantumTrajectory
    
    # Test accessors
    @test get_system(ensemble_qtraj) === sys  # Shared system
    @test get_state_name(ensemble_qtraj) == :ψ̃
    @test get_control_name(ensemble_qtraj) == :u
    @test get_weights(ensemble_qtraj) == [1.0, 1.0]
    @test get_ensemble_state_names(ensemble_qtraj) == [:ψ̃_init_1, :ψ̃_init_2]
    
    # Test goals are from all trajectories
    goals = get_goal(ensemble_qtraj)
    @test goals == [ψ1, ψ0]
    
    # Test property forwarding
    @test ensemble_qtraj.trajectories == [qtraj1, qtraj2]
    @test size(ensemble_qtraj[:ψ̃], 2) == 10  # Forwarded to first trajectory
    @test size(ensemble_qtraj[:u], 2) == 10
end

@testitem "EnsembleTrajectory with custom weights" begin
    using PiccoloQuantumObjects
    using NamedTrajectories
    
    # Create shared system
    sys = QuantumSystem(GATES[:Z], [GATES[:X]], 1.0, [1.0])
    
    # Create 3 trajectories with different state transfers
    ψ0 = ComplexF64[1.0, 0.0]
    ψ1 = ComplexF64[0.0, 1.0]
    ψ_plus = ComplexF64[1.0, 1.0] / sqrt(2)
    
    qtraj1 = KetTrajectory(sys, ψ0, ψ1, 10)
    qtraj2 = KetTrajectory(sys, ψ1, ψ0, 10)
    qtraj3 = KetTrajectory(sys, ψ0, ψ_plus, 10)
    
    # Custom weights
    weights = [0.5, 0.3, 0.2]
    
    # Create ensemble trajectory with weights
    ensemble_qtraj = EnsembleTrajectory([qtraj1, qtraj2, qtraj3]; weights=weights)
    
    @test get_weights(ensemble_qtraj) == weights
    @test length(get_ensemble_state_names(ensemble_qtraj)) == 3
    @test get_ensemble_state_names(ensemble_qtraj) == [:ψ̃_init_1, :ψ̃_init_2, :ψ̃_init_3]
end

@testitem "EnsembleTrajectory with UnitaryTrajectory" begin
    using PiccoloQuantumObjects
    using NamedTrajectories
    
    # Create shared system
    sys = QuantumSystem(GATES[:Z], [GATES[:X]], 1.0, [1.0])
    
    # Create trajectories for different target gates
    qtraj1 = UnitaryTrajectory(sys, GATES[:X], 10)
    qtraj2 = UnitaryTrajectory(sys, GATES[:H], 10)
    
    # Create ensemble trajectory
    ensemble_qtraj = EnsembleTrajectory([qtraj1, qtraj2])
    
    @test ensemble_qtraj isa EnsembleTrajectory{UnitaryTrajectory}
    @test get_state_name(ensemble_qtraj) == :Ũ⃗
    @test get_ensemble_state_names(ensemble_qtraj) == [:Ũ⃗_init_1, :Ũ⃗_init_2]
    
    # Goals are both target unitaries
    goals = get_goal(ensemble_qtraj)
    @test goals == [GATES[:X], GATES[:H]]
end

@testitem "build_sampling_trajectory utility" begin
    using PiccoloQuantumObjects
    using NamedTrajectories
    
    # Create a base trajectory
    sys = QuantumSystem(GATES[:Z], [GATES[:X]], 1.0, [1.0])
    base_qtraj = UnitaryTrajectory(sys, GATES[:H], 10)
    base_traj = get_trajectory(base_qtraj)
    
    # Build sampling trajectory
    n_samples = 3
    new_traj, state_names = build_sampling_trajectory(base_traj, :Ũ⃗, n_samples)
    
    # Check state names use _sample_ suffix
    @test state_names == [:Ũ⃗_sample_1, :Ũ⃗_sample_2, :Ũ⃗_sample_3]
    
    # Check that new trajectory has sample state components
    @test haskey(new_traj.components, :Ũ⃗_sample_1)
    @test haskey(new_traj.components, :Ũ⃗_sample_2)
    @test haskey(new_traj.components, :Ũ⃗_sample_3)
    
    # Check that original state is NOT in new trajectory
    @test !haskey(new_traj.components, :Ũ⃗)
    
    # Check that controls are preserved
    @test haskey(new_traj.components, :u)
    @test haskey(new_traj.components, :Δt)
    
    # Check initial conditions for each sample state
    @test haskey(new_traj.initial, :Ũ⃗_sample_1)
    @test haskey(new_traj.initial, :Ũ⃗_sample_2)
    @test haskey(new_traj.initial, :Ũ⃗_sample_3)
    @test new_traj.initial[:Ũ⃗_sample_1] == base_traj.initial[:Ũ⃗]
    
    # Check goal conditions for each sample state
    @test haskey(new_traj.goal, :Ũ⃗_sample_1)
    @test haskey(new_traj.goal, :Ũ⃗_sample_2)
    @test haskey(new_traj.goal, :Ũ⃗_sample_3)
    @test new_traj.goal[:Ũ⃗_sample_1] == base_traj.goal[:Ũ⃗]
end

@testitem "build_ensemble_trajectory_from_trajectories utility" begin
    using PiccoloQuantumObjects
    using NamedTrajectories
    
    # Create shared system
    sys = QuantumSystem(GATES[:Z], [GATES[:X]], 1.0, [1.0])
    
    # Create trajectories with different initial/goal states
    ψ0 = ComplexF64[1.0, 0.0]
    ψ1 = ComplexF64[0.0, 1.0]
    
    qtraj1 = KetTrajectory(sys, ψ0, ψ1, 10)
    qtraj2 = KetTrajectory(sys, ψ1, ψ0, 10)
    
    # Build ensemble trajectory
    new_traj, state_names = build_ensemble_trajectory_from_trajectories([qtraj1, qtraj2])
    
    # Check state names use _init_ suffix
    @test state_names == [:ψ̃_init_1, :ψ̃_init_2]
    
    # Check that new trajectory has ensemble state components
    @test haskey(new_traj.components, :ψ̃_init_1)
    @test haskey(new_traj.components, :ψ̃_init_2)
    
    # Check that original state is NOT in new trajectory
    @test !haskey(new_traj.components, :ψ̃)
    
    # Check initial conditions are from each trajectory
    @test new_traj.initial[:ψ̃_init_1] == get_trajectory(qtraj1).initial[:ψ̃]
    @test new_traj.initial[:ψ̃_init_2] == get_trajectory(qtraj2).initial[:ψ̃]
    
    # Check goal conditions are from each trajectory
    @test new_traj.goal[:ψ̃_init_1] == get_trajectory(qtraj1).goal[:ψ̃]
    @test new_traj.goal[:ψ̃_init_2] == get_trajectory(qtraj2).goal[:ψ̃]
end

@testitem "update_base_trajectory utility for SamplingTrajectory" begin
    using PiccoloQuantumObjects
    using NamedTrajectories
    
    # Create systems
    sys1 = QuantumSystem(GATES[:Z], [GATES[:X]], 1.0, [1.0])
    sys2 = QuantumSystem(1.1 * GATES[:Z], [GATES[:X]], 1.0, [1.0])
    systems = [sys1, sys2]
    weights = [0.6, 0.4]
    
    # Create sampling trajectory
    base_qtraj = UnitaryTrajectory(sys1, GATES[:H], 10)
    sampling_qtraj = SamplingTrajectory(base_qtraj, systems; weights=weights)
    
    # Create a new base trajectory (different goal)
    new_base_qtraj = UnitaryTrajectory(sys1, GATES[:X], 10)
    
    # Update the sampling trajectory
    updated_sampling = update_base_trajectory(sampling_qtraj, new_base_qtraj)
    
    # Check that systems, weights, and state names are preserved
    @test get_systems(updated_sampling) === systems
    @test get_weights(updated_sampling) == weights
    @test get_ensemble_state_names(updated_sampling) == get_ensemble_state_names(sampling_qtraj)
    
    # Check that base trajectory is updated
    @test updated_sampling.base_trajectory === new_base_qtraj
    @test get_goal(updated_sampling) === GATES[:X]
end

end  # module QuantumTrajectories
