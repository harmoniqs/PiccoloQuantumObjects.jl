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

# ============================================================================ #
# Tests
# ============================================================================ #

@testitem "rebuild UnitaryTrajectory" begin
    using LinearAlgebra
    using NamedTrajectories
    
    # System setup
    system = QuantumSystem(PAULIS.Z, [PAULIS.X], [1.0])
    
    # Create trajectory
    T = 1.0
    X_gate = ComplexF64[0 1; 1 0]
    qtraj = UnitaryTrajectory(system, X_gate, T)
    
    # Convert to NamedTrajectory and modify controls
    N = 11
    traj = NamedTrajectory(qtraj, N)
    traj.u .= 0.5 * randn(1, N)
    
    # Rebuild
    qtraj2 = rebuild(qtraj, traj)
    
    @test qtraj2 isa UnitaryTrajectory
    @test qtraj2.system === system
    @test qtraj2.goal === X_gate
    @test qtraj2.initial ≈ qtraj.initial
    
    # Verify the pulse was updated
    @test qtraj2.pulse !== qtraj.pulse
end

@testitem "rebuild KetTrajectory" begin
    using LinearAlgebra
    using NamedTrajectories
    
    system = QuantumSystem(PAULIS.Z, [PAULIS.X], [1.0])
    
    T = 1.0
    ψ0 = ComplexF64[1.0, 0.0]
    ψg = ComplexF64[0.0, 1.0]
    qtraj = KetTrajectory(system, ψ0, ψg, T)
    
    N = 11
    traj = NamedTrajectory(qtraj, N)
    traj.u .= 0.5 * randn(1, N)
    
    qtraj2 = rebuild(qtraj, traj)
    
    @test qtraj2 isa KetTrajectory
    @test qtraj2.system === system
    @test qtraj2.initial ≈ ψ0
    @test qtraj2.goal ≈ ψg
end

@testitem "rebuild EnsembleKetTrajectory" begin
    using LinearAlgebra
    using NamedTrajectories
    
    system = QuantumSystem(PAULIS.Z, [PAULIS.X], [1.0])
    
    T = 1.0
    initials = [ComplexF64[1.0, 0.0], ComplexF64[0.0, 1.0]]
    goals = [ComplexF64[0.0, 1.0], ComplexF64[1.0, 0.0]]
    weights = [0.6, 0.4]
    
    qtraj = EnsembleKetTrajectory(system, initials, goals, T; weights=weights)
    
    N = 11
    traj = NamedTrajectory(qtraj, N)
    traj.u .= 0.5 * randn(1, N)
    
    qtraj2 = rebuild(qtraj, traj)
    
    @test qtraj2 isa EnsembleKetTrajectory
    @test qtraj2.system === system
    @test qtraj2.weights ≈ weights
    @test length(qtraj2.initials) == 2
end

@testitem "rebuild DensityTrajectory" begin
    using LinearAlgebra
    using NamedTrajectories
    
    L = ComplexF64[0.1 0.0; 0.0 0.0]
    system = OpenQuantumSystem(PAULIS.Z, [PAULIS.X], [1.0]; dissipation_operators=[L])
    
    T = 1.0
    ρ0 = ComplexF64[1.0 0.0; 0.0 0.0]
    ρg = ComplexF64[0.0 0.0; 0.0 1.0]
    
    qtraj = DensityTrajectory(system, ρ0, ρg, T)
    
    N = 11
    traj = NamedTrajectory(qtraj, N)
    traj.u .= 0.5 * randn(1, N)
    
    qtraj2 = rebuild(qtraj, traj)
    
    @test qtraj2 isa DensityTrajectory
    @test qtraj2.system === system
    @test qtraj2.initial ≈ ρ0
    @test qtraj2.goal ≈ ρg
end

@testitem "rebuild with CubicSplinePulse" begin
    using LinearAlgebra
    using NamedTrajectories
    
    system = QuantumSystem(PAULIS.Z, [PAULIS.X], [1.0])
    
    # Create with CubicSplinePulse
    T = 1.0
    times = collect(range(0.0, T, length=11))
    u = 0.1 * randn(1, 11)
    du = zeros(1, 11)
    pulse = CubicSplinePulse(u, du, times)
    
    X_gate = ComplexF64[0 1; 1 0]
    qtraj = UnitaryTrajectory(system, pulse, X_gate)
    
    # Convert and modify
    traj = NamedTrajectory(qtraj, times)
    traj.u .= 0.5 * randn(1, 11)
    traj.du .= 0.1 * randn(1, 11)
    
    # Rebuild
    qtraj2 = rebuild(qtraj, traj)
    
    @test qtraj2 isa UnitaryTrajectory
    @test qtraj2.pulse isa CubicSplinePulse
end