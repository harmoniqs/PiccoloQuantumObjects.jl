# ============================================================================ #
# UnitaryTrajectory
# ============================================================================ #

"""
    UnitaryTrajectory{P<:AbstractPulse, S<:ODESolution, G} <: AbstractQuantumTrajectory{P}

Trajectory for unitary gate synthesis. The ODE solution is computed at construction.

# Fields
- `system::QuantumSystem`: The quantum system
- `pulse::P`: The control pulse (stores drive_name)
- `initial::Matrix{ComplexF64}`: Initial unitary (default: identity)
- `goal::G`: Target unitary operator (AbstractPiccoloOperator or Matrix)
- `solution::S`: Pre-computed ODE solution

# Callable
`traj(t)` returns the unitary at time `t` by interpolating the solution.

# Conversion to NamedTrajectory
Use `NamedTrajectory(traj, N)` or `NamedTrajectory(traj, times)` for optimization.
"""
struct UnitaryTrajectory{P<:AbstractPulse, S<:ODESolution, G} <: AbstractQuantumTrajectory{P}
    system::QuantumSystem
    pulse::P
    initial::Matrix{ComplexF64}
    goal::G
    solution::S
end

"""
    UnitaryTrajectory(system, pulse, goal; initial=I, algorithm=MagnusGL4())

Create a unitary trajectory by solving the Schrödinger equation.

# Arguments
- `system::QuantumSystem`: The quantum system
- `pulse::AbstractPulse`: The control pulse
- `goal`: Target unitary (Matrix or AbstractPiccoloOperator)

# Keyword Arguments
- `initial`: Initial unitary (default: identity matrix)
- `algorithm`: ODE solver algorithm (default: MagnusGL4())
"""
function UnitaryTrajectory(
    system::QuantumSystem,
    pulse::AbstractPulse,
    goal::G;
    initial::AbstractMatrix{<:Number}=Matrix{ComplexF64}(I, system.levels, system.levels),
    algorithm=MagnusGL4(),
) where G
    @assert n_drives(pulse) == system.n_drives "Pulse has $(n_drives(pulse)) drives, system has $(system.n_drives)"
    
    U0 = Matrix{ComplexF64}(initial)
    times = collect(range(0.0, duration(pulse), length=101))
    prob = UnitaryOperatorODEProblem(system, pulse, times; U0=U0)
    sol = solve(prob, algorithm; saveat=times)
    
    return UnitaryTrajectory{typeof(pulse), typeof(sol), G}(system, pulse, U0, goal, sol)
end

"""
    UnitaryTrajectory(system, goal, T::Real; drive_name=:u, algorithm=MagnusGL4())

Convenience constructor that creates a zero pulse of duration T.

# Arguments
- `system::QuantumSystem`: The quantum system
- `goal`: Target unitary (Matrix or AbstractPiccoloOperator)
- `T::Real`: Duration of the pulse

# Keyword Arguments
- `drive_name::Symbol`: Name of the drive variable (default: `:u`)
- `algorithm`: ODE solver algorithm (default: MagnusGL4())
"""
function UnitaryTrajectory(
    system::QuantumSystem,
    goal::G,
    T::Real;
    drive_name::Symbol=:u,
    algorithm=MagnusGL4(),
) where G
    times = [0.0, T]
    controls = zeros(system.n_drives, 2)
    pulse = ZeroOrderPulse(controls, times; drive_name)
    return UnitaryTrajectory(system, pulse, goal; algorithm)
end

# Callable: sample solution at any time
(traj::UnitaryTrajectory)(t::Real) = traj.solution(t)
