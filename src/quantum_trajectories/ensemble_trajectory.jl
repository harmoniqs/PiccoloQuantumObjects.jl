"""
Ensemble trajectory type for multi-state optimization with shared system.

Provides `EnsembleTrajectory` for optimizing over multiple initial/goal states with shared controls.
"""

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

"""
    get_combined_trajectory(qtraj::EnsembleTrajectory) -> NamedTrajectory

Return the combined trajectory with all ensemble states.

This builds a NamedTrajectory with separate state variables for each ensemble member.
For efficiency, consider caching this result if calling multiple times.
"""
function get_combined_trajectory(qtraj::EnsembleTrajectory)
    traj, _ = build_ensemble_trajectory(qtraj)
    return traj
end

# ============================================================================= #
# Helper functions
# ============================================================================= #

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
