# ============================================================================ #
# DensityTrajectory
# ============================================================================ #

"""
    DensityTrajectory{P<:AbstractPulse, S<:ODESolution} <: AbstractQuantumTrajectory{P}

Trajectory for open quantum systems (Lindblad dynamics).

# Fields
- `system::OpenQuantumSystem`: The open quantum system
- `pulse::P`: The control pulse
- `initial::Matrix{ComplexF64}`: Initial density matrix ρ₀
- `goal::Matrix{ComplexF64}`: Target density matrix ρ_goal
- `solution::S`: Pre-computed ODE solution

# Callable
`traj(t)` returns the density matrix at time `t`.
"""
struct DensityTrajectory{P<:AbstractPulse, S<:ODESolution} <: AbstractQuantumTrajectory{P}
    system::OpenQuantumSystem
    pulse::P
    initial::Matrix{ComplexF64}
    goal::Matrix{ComplexF64}
    solution::S
end

"""
    DensityTrajectory(system, pulse, initial, goal; algorithm=Tsit5())

Create a density matrix trajectory by solving the Lindblad master equation.

# Arguments
- `system::OpenQuantumSystem`: The open quantum system
- `pulse::AbstractPulse`: The control pulse
- `initial::Matrix`: Initial density matrix ρ₀
- `goal::Matrix`: Target density matrix ρ_goal

# Keyword Arguments
- `algorithm`: ODE solver algorithm (default: Tsit5())
"""
function DensityTrajectory(
    system::OpenQuantumSystem,
    pulse::AbstractPulse,
    initial::AbstractMatrix{<:Number},
    goal::AbstractMatrix{<:Number};
    algorithm=Tsit5(),
)
    @assert n_drives(pulse) == system.n_drives "Pulse has $(n_drives(pulse)) drives, system has $(system.n_drives)"
    
    ρ0 = Matrix{ComplexF64}(initial)
    ρg = Matrix{ComplexF64}(goal)
    times = collect(range(0.0, duration(pulse), length=101))
    prob = DensityODEProblem(system, pulse, ρ0, times)
    sol = solve(prob, algorithm; saveat=times)
    
    return DensityTrajectory{typeof(pulse), typeof(sol)}(system, pulse, ρ0, ρg, sol)
end

"""
    DensityTrajectory(system, initial, goal, T::Real; drive_name=:u, algorithm=Tsit5())

Convenience constructor that creates a zero pulse of duration T.
"""
function DensityTrajectory(
    system::OpenQuantumSystem,
    initial::AbstractMatrix{<:Number},
    goal::AbstractMatrix{<:Number},
    T::Real;
    drive_name::Symbol=:u,
    algorithm=Tsit5(),
)
    times = [0.0, T]
    controls = zeros(system.n_drives, 2)
    pulse = ZeroOrderPulse(controls, times; drive_name)
    return DensityTrajectory(system, pulse, initial, goal; algorithm)
end

# Callable: sample solution at any time
(traj::DensityTrajectory)(t::Real) = traj.solution(t)
