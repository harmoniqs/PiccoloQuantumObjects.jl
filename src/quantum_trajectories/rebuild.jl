# ============================================================================ #
# Rebuild Trajectories from Optimized Controls
# ============================================================================ #

export rebuild

"""
    rebuild(qtraj::AbstractQuantumTrajectory, traj::NamedTrajectory; kwargs...)

Create a new quantum trajectory from optimized control values.

After optimization, the NamedTrajectory contains updated control values. This function:
1. Extracts the optimized controls and times from the NamedTrajectory
2. Creates a new pulse with those controls (dispatches on pulse type)
3. Re-solves the ODE to get the new quantum evolution
4. Returns a new quantum trajectory with the updated pulse and solution

The reconstruction process depends on the pulse type:
- `ZeroOrderPulse`, `LinearSplinePulse`: Extracts `u` (drive variable)
- `CubicSplinePulse`: Extracts both `u` and `du` (derivative variable)

# Arguments
- `qtraj`: Original quantum trajectory (provides system, initial/goal states)
- `traj`: Optimized NamedTrajectory with new control values

# Keyword Arguments
- `algorithm`: ODE solver algorithm (default: MagnusGL4())

# Returns
A new quantum trajectory of the same type as `qtraj` with updated pulse and solution.

# Example
```julia
# After optimization
solve!(prob)
new_qtraj = rebuild(qtraj, prob.trajectory)
fidelity(new_qtraj)  # Check fidelity with updated controls
```
"""
function rebuild end

# Dispatch on pulse type for each trajectory type
function rebuild(
    qtraj::UnitaryTrajectory{<:Union{ZeroOrderPulse, LinearSplinePulse}}, 
    traj::NamedTrajectory;
    algorithm=MagnusGL4()
)
    times = collect(get_times(traj))
    u_name = drive_name(qtraj)
    u = Matrix(traj[u_name])
    pulse = _rebuild_pulse(qtraj.pulse, u, times)
    return UnitaryTrajectory(qtraj.system, pulse, qtraj.goal; algorithm)
end

function rebuild(
    qtraj::UnitaryTrajectory{<:CubicSplinePulse}, 
    traj::NamedTrajectory;
    algorithm=MagnusGL4()
)
    times = collect(get_times(traj))
    u_name = drive_name(qtraj)
    du_name = Symbol(:d, u_name)
    u = Matrix(traj[u_name])
    du = Matrix(traj[du_name])
    pulse = CubicSplinePulse(u, du, times; drive_name=u_name)
    return UnitaryTrajectory(qtraj.system, pulse, qtraj.goal; algorithm)
end

function rebuild(
    qtraj::KetTrajectory{<:Union{ZeroOrderPulse, LinearSplinePulse}}, 
    traj::NamedTrajectory;
    algorithm=MagnusGL4()
)
    times = collect(get_times(traj))
    u_name = drive_name(qtraj)
    u = Matrix(traj[u_name])
    pulse = _rebuild_pulse(qtraj.pulse, u, times)
    return KetTrajectory(qtraj.system, pulse, qtraj.initial, qtraj.goal; algorithm)
end

function rebuild(
    qtraj::KetTrajectory{<:CubicSplinePulse}, 
    traj::NamedTrajectory;
    algorithm=MagnusGL4()
)
    times = collect(get_times(traj))
    u_name = drive_name(qtraj)
    du_name = Symbol(:d, u_name)
    u = Matrix(traj[u_name])
    du = Matrix(traj[du_name])
    pulse = CubicSplinePulse(u, du, times; drive_name=u_name)
    return KetTrajectory(qtraj.system, pulse, qtraj.initial, qtraj.goal; algorithm)
end

function rebuild(
    qtraj::EnsembleKetTrajectory{<:Union{ZeroOrderPulse, LinearSplinePulse}}, 
    traj::NamedTrajectory;
    algorithm=MagnusGL4()
)
    times = collect(get_times(traj))
    u_name = drive_name(qtraj)
    u = Matrix(traj[u_name])
    pulse = _rebuild_pulse(qtraj.pulse, u, times)
    return EnsembleKetTrajectory(
        qtraj.system, pulse, qtraj.initials, qtraj.goals;
        weights=qtraj.weights, algorithm
    )
end

function rebuild(
    qtraj::EnsembleKetTrajectory{<:CubicSplinePulse}, 
    traj::NamedTrajectory;
    algorithm=MagnusGL4()
)
    times = collect(get_times(traj))
    u_name = drive_name(qtraj)
    du_name = Symbol(:d, u_name)
    u = Matrix(traj[u_name])
    du = Matrix(traj[du_name])
    pulse = CubicSplinePulse(u, du, times; drive_name=u_name)
    return EnsembleKetTrajectory(
        qtraj.system, pulse, qtraj.initials, qtraj.goals;
        weights=qtraj.weights, algorithm
    )
end

function rebuild(
    qtraj::DensityTrajectory{<:Union{ZeroOrderPulse, LinearSplinePulse}}, 
    traj::NamedTrajectory;
    algorithm=Tsit5()
)
    times = collect(get_times(traj))
    u_name = drive_name(qtraj)
    u = Matrix(traj[u_name])
    pulse = _rebuild_pulse(qtraj.pulse, u, times)
    return DensityTrajectory(qtraj.system, pulse, qtraj.initial, qtraj.goal; algorithm)
end

function rebuild(
    qtraj::DensityTrajectory{<:CubicSplinePulse}, 
    traj::NamedTrajectory;
    algorithm=Tsit5()
)
    times = collect(get_times(traj))
    u_name = drive_name(qtraj)
    du_name = Symbol(:d, u_name)
    u = Matrix(traj[u_name])
    du = Matrix(traj[du_name])
    pulse = CubicSplinePulse(u, du, times; drive_name=u_name)
    return DensityTrajectory(qtraj.system, pulse, qtraj.initial, qtraj.goal; algorithm)
end

"""
    _rebuild_pulse(original_pulse, controls, times)

Create a new pulse of the same type as `original_pulse` with new control values.
"""
function _rebuild_pulse(p::ZeroOrderPulse, u::Matrix, times::Vector)
    return ZeroOrderPulse(u, times; drive_name=p.drive_name)
end

function _rebuild_pulse(p::LinearSplinePulse, u::Matrix, times::Vector)
    return LinearSplinePulse(u, times; drive_name=p.drive_name)
end
