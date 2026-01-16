# ============================================================================ #
# Extract Pulse from Optimized Controls
# ============================================================================ #

export extract_pulse

"""
    extract_pulse(qtraj::AbstractQuantumTrajectory, traj::NamedTrajectory)

Extract an optimized pulse from a NamedTrajectory.

This function extracts the control values from the optimized trajectory and creates
a new pulse object of the same type as the original pulse in `qtraj`.

The extraction process depends on the pulse type:
- `ZeroOrderPulse`, `LinearSplinePulse`: Extracts `u` (drive variable)
- `CubicSplinePulse`: Extracts both `u` and `du` (derivative variable)

# Arguments
- `qtraj`: Original quantum trajectory (provides pulse type and drive names)
- `traj`: Optimized NamedTrajectory with new control values

# Returns
A new pulse of the same type as `qtraj.pulse` with optimized control values.

# Example
```julia
# After optimization
solve!(prob)
new_pulse = extract_pulse(qtraj, prob.trajectory)
rollout!(qtraj, new_pulse)
```
"""
function extract_pulse end

# Dispatch on pulse type
function extract_pulse(
    qtraj::AbstractQuantumTrajectory{<:Union{ZeroOrderPulse, LinearSplinePulse}},
    traj::NamedTrajectory
)
    times = collect(get_times(traj))
    u_name = drive_name(qtraj)
    u = Matrix(traj[u_name])
    return _rebuild_pulse(qtraj.pulse, u, times)
end

function extract_pulse(
    qtraj::AbstractQuantumTrajectory{<:CubicSplinePulse},
    traj::NamedTrajectory
)
    times = collect(get_times(traj))
    u_name = drive_name(qtraj)
    du_name = Symbol(:d, u_name)
    u = Matrix(traj[u_name])
    du = Matrix(traj[du_name])
    return CubicSplinePulse(u, du, times; drive_name=u_name)
end

# SamplingTrajectory delegates to base_trajectory
function extract_pulse(
    qtraj::SamplingTrajectory,
    traj::NamedTrajectory
)
    return extract_pulse(qtraj.base_trajectory, traj)
end

# Helper functions for pulse reconstruction
function _rebuild_pulse(p::ZeroOrderPulse, u::Matrix, times::Vector)
    return ZeroOrderPulse(u, times; drive_name=p.drive_name)
end

function _rebuild_pulse(p::LinearSplinePulse, u::Matrix, times::Vector)
    return LinearSplinePulse(u, times; drive_name=p.drive_name)
end
