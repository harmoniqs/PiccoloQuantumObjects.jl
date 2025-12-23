"""
Sampling trajectory type for robust optimization over system ensembles.

Provides `SamplingTrajectory` for optimization over multiple systems with shared controls.
"""

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
# Helper functions
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
