"""
Abstract base type and interface for quantum trajectories.

Defines `AbstractQuantumTrajectory` and default implementations for:
- Accessor functions (`get_trajectory`, `get_system`, `get_state_name`, etc.)
- Delegation to underlying `NamedTrajectory`
- Convenience accessors (`state`, `controls`)
"""

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
