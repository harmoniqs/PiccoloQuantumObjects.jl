# ============================================================================ #
# KetTrajectory
# ============================================================================ #

"""
    KetTrajectory{P<:AbstractPulse, S<:ODESolution} <: AbstractQuantumTrajectory{P}

Trajectory for quantum state transfer. The ODE solution is computed at construction.

# Fields
- `system::QuantumSystem`: The quantum system
- `pulse::P`: The control pulse
- `initial::Vector{ComplexF64}`: Initial state |ψ₀⟩
- `goal::Vector{ComplexF64}`: Target state |ψ_goal⟩
- `solution::S`: Pre-computed ODE solution

# Callable
`traj(t)` returns the state at time `t` by interpolating the solution.
"""
struct KetTrajectory{P<:AbstractPulse, S<:ODESolution} <: AbstractQuantumTrajectory{P}
    system::QuantumSystem
    pulse::P
    initial::Vector{ComplexF64}
    goal::Vector{ComplexF64}
    solution::S
end

"""
    KetTrajectory(system, pulse, initial, goal; algorithm=MagnusGL4())

Create a ket trajectory by solving the Schrödinger equation.

# Arguments
- `system::QuantumSystem`: The quantum system
- `pulse::AbstractPulse`: The control pulse
- `initial::Vector`: Initial state |ψ₀⟩
- `goal::Vector`: Target state |ψ_goal⟩

# Keyword Arguments
- `algorithm`: ODE solver algorithm (default: MagnusGL4())
"""
function KetTrajectory(
    system::QuantumSystem,
    pulse::AbstractPulse,
    initial::AbstractVector{<:Number},
    goal::AbstractVector{<:Number};
    algorithm=MagnusGL4(),
)
    @assert n_drives(pulse) == system.n_drives "Pulse has $(n_drives(pulse)) drives, system has $(system.n_drives)"
    
    ψ0 = Vector{ComplexF64}(initial)
    ψg = Vector{ComplexF64}(goal)
    times = collect(range(0.0, duration(pulse), length=101))
    prob = KetOperatorODEProblem(system, pulse, ψ0, times)
    sol = solve(prob, algorithm; saveat=times)
    
    return KetTrajectory{typeof(pulse), typeof(sol)}(system, pulse, ψ0, ψg, sol)
end

"""
    KetTrajectory(system, initial, goal, T::Real; drive_name=:u, algorithm=MagnusGL4())

Convenience constructor that creates a zero pulse of duration T.

# Arguments
- `system::QuantumSystem`: The quantum system
- `initial::Vector`: Initial state |ψ₀⟩
- `goal::Vector`: Target state |ψ_goal⟩
- `T::Real`: Duration of the pulse

# Keyword Arguments
- `drive_name::Symbol`: Name of the drive variable (default: `:u`)
- `algorithm`: ODE solver algorithm (default: MagnusGL4())
"""
function KetTrajectory(
    system::QuantumSystem,
    initial::AbstractVector{<:Number},
    goal::AbstractVector{<:Number},
    T::Real;
    drive_name::Symbol=:u,
    algorithm=MagnusGL4(),
)
    times = [0.0, T]
    controls = zeros(system.n_drives, 2)
    pulse = ZeroOrderPulse(controls, times; drive_name)
    return KetTrajectory(system, pulse, initial, goal; algorithm)
end

# Callable: sample solution at any time
(traj::KetTrajectory)(t::Real) = traj.solution(t)
