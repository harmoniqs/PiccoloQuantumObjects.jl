module QuantumTrajectories

using ..Pulses: AbstractPulse, AbstractSplinePulse, ZeroOrderPulse, LinearSplinePulse, CubicSplinePulse
using ..Pulses: duration, drive_name

"""
Quantum trajectory types with ODESolution and Pulse integration.

These types provide a clean interface for quantum control simulations:
1. Uses `AbstractPulse` for control representation
2. Stores `ODESolution` computed at construction
3. Is callable: `traj(t)` samples the solution at any time

Trajectory types:
- `UnitaryTrajectory`: For unitary gate synthesis
- `KetTrajectory`: For single quantum state transfer  
- `EnsembleKetTrajectory`: For multi-state transfer with shared pulse
- `DensityTrajectory`: For open quantum systems

NamedTrajectory conversion:
- `NamedTrajectory(traj, N)` or `NamedTrajectory(traj, times)` for optimization
"""

# ============================================================================ #
# Abstract Type
# ============================================================================ #

"""
    AbstractQuantumTrajectory{P<:AbstractPulse}

Abstract type for quantum trajectories that wrap physics (system, pulse, solution, goal).
Parametric on pulse type `P` to enable dispatch in problem templates.

All concrete subtypes should implement:
- `state_name(traj)` - Get the state variable symbol (fixed per type)
- `drive_name(traj)` - Get the drive variable symbol (from pulse)
- `time_name(traj)` - Get the time variable symbol (fixed `:t`)
- `timestep_name(traj)` - Get the timestep variable symbol (fixed `:Δt`)
- `duration(traj)` - Get the duration (from pulse)
"""
abstract type AbstractQuantumTrajectory{P<:AbstractPulse} end

export AbstractQuantumTrajectory
export UnitaryTrajectory, KetTrajectory, EnsembleKetTrajectory, DensityTrajectory
export state_name, state_names, drive_name, time_name, timestep_name
export get_system, get_pulse, get_initial, get_goal, get_solution

using LinearAlgebra
using SciMLBase: ODESolution, solve, remake, EnsembleProblem
using OrdinaryDiffEqLinear: MagnusGL4
using OrdinaryDiffEqTsit5: Tsit5
using TestItems

using ..QuantumSystems: AbstractQuantumSystem, QuantumSystem, OpenQuantumSystem
using ..Pulses: AbstractPulse, ZeroOrderPulse, LinearSplinePulse, CubicSplinePulse, GaussianPulse, n_drives
import ..Pulses: duration, drive_name
import ..Rollouts
using ..Rollouts: UnitaryODEProblem, UnitaryOperatorODEProblem, KetODEProblem, KetOperatorODEProblem, DensityODEProblem
using ..Rollouts: unitary_fidelity
using ..EmbeddedOperators: AbstractPiccoloOperator, EmbeddedOperator
using ..Isomorphisms: operator_to_iso_vec, ket_to_iso, iso_to_ket, iso_vec_to_operator

import NamedTrajectories: NamedTrajectory, get_times


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

# ============================================================================ #
# EnsembleKetTrajectory
# ============================================================================ #

"""
    EnsembleKetTrajectory{P<:AbstractPulse, S} <: AbstractQuantumTrajectory{P}

Trajectory for multi-state transfer with a shared pulse. Useful for state-to-state
problems with multiple initial/goal pairs.

# Fields
- `system::QuantumSystem`: The quantum system
- `pulse::P`: The shared control pulse
- `initials::Vector{Vector{ComplexF64}}`: Initial states
- `goals::Vector{Vector{ComplexF64}}`: Target states
- `weights::Vector{Float64}`: Weights for fidelity calculation
- `solution::S`: Pre-computed ensemble solution

# Callable
`traj(t)` returns a vector of states at time `t`.
`traj[i]` returns the i-th trajectory's solution.
"""
struct EnsembleKetTrajectory{P<:AbstractPulse, S} <: AbstractQuantumTrajectory{P}
    system::QuantumSystem
    pulse::P
    initials::Vector{Vector{ComplexF64}}
    goals::Vector{Vector{ComplexF64}}
    weights::Vector{Float64}
    solution::S
end

"""
    EnsembleKetTrajectory(system, pulse, initials, goals; weights=..., algorithm=MagnusGL4())

Create an ensemble ket trajectory by solving multiple Schrödinger equations.

# Arguments
- `system::QuantumSystem`: The quantum system
- `pulse::AbstractPulse`: The shared control pulse
- `initials::Vector{Vector}`: Initial states
- `goals::Vector{Vector}`: Target states

# Keyword Arguments
- `weights`: Weights for fidelity (default: uniform)
- `algorithm`: ODE solver algorithm (default: MagnusGL4())
"""
function EnsembleKetTrajectory(
    system::QuantumSystem,
    pulse::AbstractPulse,
    initials::Vector{<:AbstractVector{<:Number}},
    goals::Vector{<:AbstractVector{<:Number}};
    weights::AbstractVector{<:Real}=fill(1.0/length(initials), length(initials)),
    algorithm=MagnusGL4(),
)
    @assert n_drives(pulse) == system.n_drives "Pulse has $(n_drives(pulse)) drives, system has $(system.n_drives)"
    @assert length(initials) == length(goals) == length(weights) "initials, goals, and weights must have same length"
    
    ψ0s = [Vector{ComplexF64}(ψ) for ψ in initials]
    ψgs = [Vector{ComplexF64}(ψ) for ψ in goals]
    ws = Vector{Float64}(weights)
    
    times = collect(range(0.0, duration(pulse), length=101))
    
    # Build ensemble problem
    dummy = zeros(ComplexF64, system.levels)
    base_prob = KetOperatorODEProblem(system, pulse, dummy, times)
    prob_func(prob, i, repeat) = remake(prob, u0=ψ0s[i])
    ensemble_prob = EnsembleProblem(base_prob; prob_func=prob_func)
    sol = solve(ensemble_prob, algorithm; trajectories=length(initials), saveat=times)
    
    return EnsembleKetTrajectory{typeof(pulse), typeof(sol)}(system, pulse, ψ0s, ψgs, ws, sol)
end

"""
    EnsembleKetTrajectory(system, initials, goals, T::Real; weights=..., drive_name=:u, algorithm=MagnusGL4())

Convenience constructor that creates a zero pulse of duration T.

# Arguments
- `system::QuantumSystem`: The quantum system
- `initials::Vector{Vector}`: Initial states
- `goals::Vector{Vector}`: Target states
- `T::Real`: Duration of the pulse

# Keyword Arguments
- `weights`: Weights for fidelity (default: uniform)
- `drive_name::Symbol`: Name of the drive variable (default: `:u`)
- `algorithm`: ODE solver algorithm (default: MagnusGL4())
"""
function EnsembleKetTrajectory(
    system::QuantumSystem,
    initials::Vector{<:AbstractVector{<:Number}},
    goals::Vector{<:AbstractVector{<:Number}},
    T::Real;
    weights::AbstractVector{<:Real}=fill(1.0/length(initials), length(initials)),
    drive_name::Symbol=:u,
    algorithm=MagnusGL4(),
)
    times = [0.0, T]
    controls = zeros(system.n_drives, 2)
    pulse = ZeroOrderPulse(controls, times; drive_name)
    return EnsembleKetTrajectory(system, pulse, initials, goals; weights, algorithm)
end

# Callable: sample all solutions at time t
(traj::EnsembleKetTrajectory)(t::Real) = [sol(t) for sol in traj.solution]

# Indexing: get individual trajectory solution
Base.getindex(traj::EnsembleKetTrajectory, i::Int) = traj.solution[i]
Base.length(traj::EnsembleKetTrajectory) = length(traj.initials)

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

# ============================================================================ #
# Common Interface
# ============================================================================ #

"""
    get_system(traj)

Get the quantum system from a trajectory.
"""
get_system(traj::AbstractQuantumTrajectory) = traj.system

"""
    get_pulse(traj)

Get the control pulse from a trajectory.
"""
get_pulse(traj::AbstractQuantumTrajectory) = traj.pulse

"""
    get_initial(traj)

Get the initial state/operator from a trajectory.
"""
get_initial(traj::UnitaryTrajectory) = traj.initial
get_initial(traj::KetTrajectory) = traj.initial
get_initial(traj::EnsembleKetTrajectory) = traj.initials
get_initial(traj::DensityTrajectory) = traj.initial

"""
    get_goal(traj)

Get the goal state/operator from a trajectory.
"""
get_goal(traj::UnitaryTrajectory) = traj.goal
get_goal(traj::KetTrajectory) = traj.goal
get_goal(traj::EnsembleKetTrajectory) = traj.goals
get_goal(traj::DensityTrajectory) = traj.goal

"""
    get_solution(traj)

Get the ODE solution from a trajectory.
"""
get_solution(traj::AbstractQuantumTrajectory) = traj.solution

# ============================================================================ #
# Fixed Name Accessors (for NamedTrajectory conversion)
# ============================================================================ #

"""
    state_name(::AbstractQuantumTrajectory)

Get the fixed state variable name for a trajectory type.
- `UnitaryTrajectory` → `:Ũ⃗`
- `KetTrajectory` → `:ψ̃`
- `EnsembleKetTrajectory` → `:ψ̃` (with index appended: `:ψ̃1`, `:ψ̃2`, etc.)
- `DensityTrajectory` → `:ρ⃗̃`
"""
state_name(::UnitaryTrajectory) = :Ũ⃗
state_name(::KetTrajectory) = :ψ̃
state_name(::EnsembleKetTrajectory) = :ψ̃  # prefix for :ψ̃1, :ψ̃2, etc.
state_name(::DensityTrajectory) = :ρ⃗̃

"""
    state_names(traj::EnsembleKetTrajectory)

Get all state names for an ensemble trajectory (`:ψ̃1`, `:ψ̃2`, etc.)
"""
function state_names(traj::EnsembleKetTrajectory)
    prefix = state_name(traj)
    return [Symbol(prefix, i) for i in 1:length(traj.initials)]
end

"""
    drive_name(traj::AbstractQuantumTrajectory)

Get the drive/control variable name from the trajectory's pulse.
"""
drive_name(traj::AbstractQuantumTrajectory) = drive_name(traj.pulse)

"""
    time_name(::AbstractQuantumTrajectory)

Get the time variable name (always `:t`).
"""
time_name(::AbstractQuantumTrajectory) = :t

"""
    timestep_name(::AbstractQuantumTrajectory)

Get the timestep variable name (always `:Δt`).
"""
timestep_name(::AbstractQuantumTrajectory) = :Δt

"""
    duration(traj)

Get the duration of a trajectory (from its pulse).
"""
duration(traj::AbstractQuantumTrajectory) = duration(traj.pulse)

# ============================================================================ #
# Fidelity (extending Rollouts.fidelity)
# ============================================================================ #

"""
    fidelity(traj::UnitaryTrajectory; subspace=nothing)

Compute the fidelity between the final unitary and the goal.
"""
function Rollouts.fidelity(traj::UnitaryTrajectory; subspace::Union{Nothing, AbstractVector{Int}}=nothing)
    U_final = traj.solution.u[end]
    U_goal = traj.goal isa EmbeddedOperator ? traj.goal.operator : traj.goal
    if isnothing(subspace)
        return unitary_fidelity(U_final, U_goal)
    else
        return unitary_fidelity(U_final, U_goal; subspace=subspace)
    end
end

"""
    fidelity(traj::KetTrajectory)

Compute the fidelity between the final state and the goal.
"""
function Rollouts.fidelity(traj::KetTrajectory)
    ψ_final = traj.solution.u[end]
    return abs2(ψ_final' * traj.goal)
end

"""
    fidelity(traj::EnsembleKetTrajectory)

Compute the weighted average fidelity across all state transfers.
"""
function Rollouts.fidelity(traj::EnsembleKetTrajectory)
    fids = map(zip(traj.solution, traj.goals)) do (sol, goal)
        abs2(sol.u[end]' * goal)
    end
    return sum(traj.weights .* fids)
end

"""
    fidelity(traj::DensityTrajectory)

Compute the fidelity between the final density matrix and the goal.
Uses trace fidelity: F = tr(ρ_final * ρ_goal)
"""
function Rollouts.fidelity(traj::DensityTrajectory)
    ρ_final = traj.solution.u[end]
    return real(tr(ρ_final * traj.goal))
end

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
# NamedTrajectory Conversion
# ============================================================================ #

"""
    _named_tuple(pairs...)

Create a NamedTuple from pairs of (Symbol, value). This is needed when keys are 
dynamic (stored in variables).

Example:
    name = :x
    _named_tuple(name => 1, :y => 2)  # Returns (x = 1, y = 2)
"""
function _named_tuple(pairs::Pair{Symbol}...)
    keys = Tuple(p.first for p in pairs)
    vals = Tuple(p.second for p in pairs)
    return NamedTuple{keys}(vals)
end

"""
    _sample_times(traj, N::Int)

Generate N uniformly spaced times for sampling.
"""
_sample_times(traj, N::Int) = collect(range(0.0, duration(traj), length=N))

"""
    _sample_times(traj, times::AbstractVector)

Return times as a Float64 vector.
"""
_sample_times(traj, times::AbstractVector{<:Real}) = collect(Float64, times)

"""
    _get_drive_bounds(sys::QuantumSystem)

Extract drive bounds from system as tuple of (lower, upper) vectors.
"""
function _get_drive_bounds(sys::AbstractQuantumSystem)
    n = sys.n_drives
    lower = [b[1] for b in sys.drive_bounds]
    upper = [b[2] for b in sys.drive_bounds]
    return (lower, upper)
end

"""
    _get_control_data(pulse::Union{ZeroOrderPulse, LinearSplinePulse}, times, sys)

For ZeroOrderPulse and LinearSplinePulse: return `u` data with system bounds.
Uses the pulse's drive_name to determine variable naming.
"""
function _get_control_data(pulse::Union{ZeroOrderPulse, LinearSplinePulse}, times::AbstractVector, sys::AbstractQuantumSystem)
    u_name = drive_name(pulse)
    u = hcat([pulse(t) for t in times]...)
    u_bounds = _get_drive_bounds(sys)
    return _named_tuple(u_name => u), (u_name,), _named_tuple(u_name => u_bounds)
end

"""
    _get_control_data(pulse::CubicSplinePulse, times, sys)

For CubicSplinePulse: return `u` and `du` data with system bounds.
Uses the pulse's drive_name to determine variable naming.
"""
function _get_control_data(pulse::CubicSplinePulse, times::AbstractVector, sys::AbstractQuantumSystem)
    u_name = drive_name(pulse)
    du_name = Symbol(:d, u_name)
    n = n_drives(pulse)
    T = length(times)
    
    # Sample u values
    u = hcat([pulse(t) for t in times]...)
    
    # Compute du via finite differences
    Δt = diff(times)
    du = zeros(n, T)
    for k in 1:T-1
        du[:, k] = (u[:, k+1] - u[:, k]) / Δt[k]
    end
    du[:, T] = du[:, T-1]  # Extrapolate last point
    
    u_bounds = _get_drive_bounds(sys)
    # du bounds are typically unbounded (controlled by regularization)
    du_bounds = (-Inf * ones(n), Inf * ones(n))
    
    return _named_tuple(u_name => u, du_name => du), (u_name, du_name), _named_tuple(u_name => u_bounds, du_name => du_bounds)
end

"""
    _get_control_data(pulse::GaussianPulse, times, sys)

For GaussianPulse: sample as u values with system bounds.
Uses the pulse's drive_name to determine variable naming.
"""
function _get_control_data(pulse::GaussianPulse, times::AbstractVector, sys::AbstractQuantumSystem)
    u_name = drive_name(pulse)
    u = hcat([pulse(t) for t in times]...)
    u_bounds = _get_drive_bounds(sys)
    return _named_tuple(u_name => u), (u_name,), _named_tuple(u_name => u_bounds)
end

# ============================================================================ #
# Public NamedTrajectory Conversion
# ============================================================================ #

"""
    NamedTrajectory(qtraj::UnitaryTrajectory, N::Int; Δt_bounds=nothing)
    NamedTrajectory(qtraj::UnitaryTrajectory, times::AbstractVector; Δt_bounds=nothing)

Convert a UnitaryTrajectory to a NamedTrajectory for optimization.

The trajectory stores actual times `:t` (not timesteps `:Δt`), which is required
for time-dependent integrators used with `SplinePulseProblem`.

# Stored Variables
- `Ũ⃗`: Isomorphism of unitary (vectorized real representation)
- `u` (or custom drive_name): Control values sampled at times
- `du`: Control derivatives (only for CubicSplinePulse)
- `t`: Times

# Arguments
- `qtraj`: The quantum trajectory to convert
- `N::Int`: Number of uniformly spaced time points, OR
- `times::AbstractVector`: Specific times to sample at

# Keyword Arguments
- `Δt_bounds`: Optional tuple `(lower, upper)` for timestep bounds. If provided,
  enables free-time optimization (minimum-time problems). Default: `nothing` (no bounds).

# Returns
A NamedTrajectory suitable for direct collocation optimization.
"""
function NamedTrajectory(
    qtraj::UnitaryTrajectory,
    N_or_times::Union{Int, AbstractVector{<:Real}};
    Δt_bounds::Union{Nothing, Tuple{Float64, Float64}}=nothing
)
    times = _sample_times(qtraj, N_or_times)
    T = length(times)
    s_name = state_name(qtraj)
    
    # Sample unitary states
    states = [qtraj(t) for t in times]
    Ũ⃗ = hcat([operator_to_iso_vec(U) for U in states]...)
    
    # Get control data based on pulse type
    control_data, control_names, control_bounds = _get_control_data(qtraj.pulse, times, qtraj.system)
    
    # State dimension
    state_dim = size(Ũ⃗, 1)
    
    # Compute Δt from times (pad to length T by repeating last value)
    Δt_diff = diff(times)
    Δt = [Δt_diff; Δt_diff[end]]
    
    # Build data NamedTuple with Δt as timestep and t for reference
    data = merge(
        _named_tuple(s_name => Ũ⃗, :Δt => Δt, :t => collect(times)),
        control_data
    )
    
    # Initial and final conditions
    initial = _named_tuple(s_name => operator_to_iso_vec(qtraj.initial))
    U_goal = qtraj.goal isa EmbeddedOperator ? qtraj.goal.operator : qtraj.goal
    goal_nt = _named_tuple(s_name => operator_to_iso_vec(U_goal))
    
    # Bounds (state bounded, controls bounded by system, optionally timestep bounded)
    bounds = merge(
        _named_tuple(s_name => (-ones(state_dim), ones(state_dim))),
        control_bounds
    )
    # Add Δt bounds if provided
    if !isnothing(Δt_bounds)
        bounds = merge(bounds, (Δt = ([Δt_bounds[1]], [Δt_bounds[2]]),))
    end
    
    return NamedTrajectory(
        data;
        timestep=:Δt,
        controls=(:Δt, control_names...),
        bounds=bounds,
        initial=initial,
        goal=goal_nt
    )
end

"""
    NamedTrajectory(qtraj::KetTrajectory, N::Int; Δt_bounds=nothing)
    NamedTrajectory(qtraj::KetTrajectory, times::AbstractVector; Δt_bounds=nothing)

Convert a KetTrajectory to a NamedTrajectory for optimization.

# Stored Variables
- `ψ̃`: Isomorphism of ket state (real representation)
- `u` (or custom drive_name): Control values sampled at times
- `du`: Control derivatives (only for CubicSplinePulse)
- `t`: Times

# Keyword Arguments
- `Δt_bounds`: Optional tuple `(lower, upper)` for timestep bounds. If provided,
  enables free-time optimization (minimum-time problems). Default: `nothing` (no bounds).
"""
function NamedTrajectory(
    qtraj::KetTrajectory,
    N_or_times::Union{Int, AbstractVector{<:Real}};
    Δt_bounds::Union{Nothing, Tuple{Float64, Float64}}=nothing
)
    times = _sample_times(qtraj, N_or_times)
    T = length(times)
    s_name = state_name(qtraj)
    
    # Sample ket states
    states = [qtraj(t) for t in times]
    ψ̃ = hcat([ket_to_iso(ψ) for ψ in states]...)
    
    # Get control data based on pulse type
    control_data, control_names, control_bounds = _get_control_data(qtraj.pulse, times, qtraj.system)
    
    # State dimension
    state_dim = size(ψ̃, 1)
    
    # Compute Δt from times (pad to length T by repeating last value)
    Δt_diff = diff(times)
    Δt = [Δt_diff; Δt_diff[end]]
    
    # Build data with Δt as timestep and t for reference
    data = merge(
        _named_tuple(s_name => ψ̃, :Δt => Δt, :t => collect(times)),
        control_data
    )
    
    # Initial, goal, bounds
    initial = _named_tuple(s_name => ket_to_iso(qtraj.initial))
    goal_nt = _named_tuple(s_name => ket_to_iso(qtraj.goal))
    bounds = merge(
        _named_tuple(s_name => (-ones(state_dim), ones(state_dim))),
        control_bounds
    )
    # Add Δt bounds if provided
    if !isnothing(Δt_bounds)
        bounds = merge(bounds, (Δt = ([Δt_bounds[1]], [Δt_bounds[2]]),))
    end
    
    return NamedTrajectory(
        data;
        timestep=:Δt,
        controls=(:Δt, control_names...),
        bounds=bounds,
        initial=initial,
        goal=goal_nt
    )
end

"""
    NamedTrajectory(qtraj::EnsembleKetTrajectory, N::Int; Δt_bounds=nothing)
    NamedTrajectory(qtraj::EnsembleKetTrajectory, times::AbstractVector; Δt_bounds=nothing)

Convert an EnsembleKetTrajectory to a NamedTrajectory for optimization.

# Stored Variables
- `ψ̃1`, `ψ̃2`, ...: Isomorphisms of each ket state
- `u` (or custom drive_name): Control values sampled at times
- `du`: Control derivatives (only for CubicSplinePulse)
- `t`: Times

# Keyword Arguments
- `Δt_bounds`: Optional tuple `(lower, upper)` for timestep bounds. If provided,
  enables free-time optimization (minimum-time problems). Default: `nothing` (no bounds).
"""
function NamedTrajectory(
    qtraj::EnsembleKetTrajectory,
    N_or_times::Union{Int, AbstractVector{<:Real}};
    Δt_bounds::Union{Nothing, Tuple{Float64, Float64}}=nothing
)
    times = _sample_times(qtraj, N_or_times)
    T = length(times)
    n_states = length(qtraj)
    state_prefix = state_name(qtraj)
    
    # Sample all ket states
    state_data = NamedTuple()
    initial_nt = NamedTuple()
    goal_nt = NamedTuple()
    bounds = NamedTuple()
    
    for i in 1:n_states
        name = Symbol(state_prefix, i)
        sol = qtraj[i]
        states = [sol(t) for t in times]
        ψ̃ = hcat([ket_to_iso(ψ) for ψ in states]...)
        state_dim = size(ψ̃, 1)
        
        state_data = merge(state_data, _named_tuple(name => ψ̃))
        initial_nt = merge(initial_nt, _named_tuple(name => ket_to_iso(qtraj.initials[i])))
        goal_nt = merge(goal_nt, _named_tuple(name => ket_to_iso(qtraj.goals[i])))
        bounds = merge(bounds, _named_tuple(name => (-ones(state_dim), ones(state_dim))))
    end
    
    # Get control data
    control_data, control_names, control_bounds = _get_control_data(qtraj.pulse, times, qtraj.system)
    
    # Compute Δt from times (pad to length T by repeating last value)
    Δt_diff = diff(times)
    Δt = [Δt_diff; Δt_diff[end]]
    
    # Build data with Δt as timestep and t for reference
    data = merge(state_data, (; Δt = Δt, t = collect(times)), control_data)
    bounds = merge(bounds, control_bounds)
    # Add Δt bounds if provided
    if !isnothing(Δt_bounds)
        bounds = merge(bounds, (Δt = ([Δt_bounds[1]], [Δt_bounds[2]]),))
    end
    
    return NamedTrajectory(
        data;
        timestep=:Δt,
        controls=(:Δt, control_names...),
        bounds=bounds,
        initial=initial_nt,
        goal=goal_nt
    )
end

"""
    NamedTrajectory(qtraj::DensityTrajectory, N::Int; Δt_bounds=nothing)
    NamedTrajectory(qtraj::DensityTrajectory, times::AbstractVector; Δt_bounds=nothing)

Convert a DensityTrajectory to a NamedTrajectory for optimization.

# Stored Variables
- `ρ⃗̃`: Vectorized isomorphism of the density matrix
- `u` (or custom drive_name): Control values sampled at times
- `du`: Control derivatives (only for CubicSplinePulse)
- `t`: Times

# Keyword Arguments
- `Δt_bounds`: Optional tuple `(lower, upper)` for timestep bounds. If provided,
  enables free-time optimization (minimum-time problems). Default: `nothing` (no bounds).
"""
function NamedTrajectory(
    qtraj::DensityTrajectory,
    N_or_times::Union{Int, AbstractVector{<:Real}};
    Δt_bounds::Union{Nothing, Tuple{Float64, Float64}}=nothing
)
    times = _sample_times(qtraj, N_or_times)
    T = length(times)
    sname = state_name(qtraj)
    
    # Sample density matrices and convert to isomorphism (vectorized)
    # Use real-valued representation: [vec(Re(ρ)); vec(Im(ρ))]
    states = [qtraj(t) for t in times]
    ρ̃ = hcat([_density_to_iso(ρ) for ρ in states]...)
    state_dim = size(ρ̃, 1)
    
    # Get control data
    control_data, control_names, control_bounds = _get_control_data(qtraj.pulse, times, qtraj.system)
    
    # Compute Δt from times (pad to length T by repeating last value)
    Δt_diff = diff(times)
    Δt = [Δt_diff; Δt_diff[end]]
    
    # Build data with Δt as timestep and t for reference
    data = merge(
        _named_tuple(sname => ρ̃),
        (; Δt = Δt, t = collect(times)),
        control_data
    )
    
    # Note: Density matrix bounds are trickier (trace=1, positive semidefinite)
    # For now, use generous bounds on the vectorized representation
    bounds = merge(
        _named_tuple(sname => (-ones(state_dim), ones(state_dim))),
        control_bounds
    )
    
    # Add Δt bounds if provided
    if !isnothing(Δt_bounds)
        bounds = merge(bounds, (Δt = ([Δt_bounds[1]], [Δt_bounds[2]]),))
    end
    
    # Initial and goal in isomorphism
    initial = _named_tuple(sname => _density_to_iso(qtraj.initial))
    goal_nt = _named_tuple(sname => _density_to_iso(qtraj.goal))
    
    return NamedTrajectory(
        data;
        timestep=:Δt,
        controls=(:Δt, control_names...),
        bounds=bounds,
        initial=initial,
        goal=goal_nt
    )
end

# Helper: convert density matrix to real-valued isomorphism vector
function _density_to_iso(ρ::AbstractMatrix)
    return vcat(vec(real(ρ)), vec(imag(ρ)))
end

# Helper: convert isomorphism vector back to density matrix
function _iso_to_density(ρ̃::AbstractVector, n::Int)
    len = n^2
    re = reshape(ρ̃[1:len], n, n)
    im = reshape(ρ̃[len+1:end], n, n)
    return complex.(re, im)
end

# ============================================================================ #
# Tests
# ============================================================================ #

@testitem "UnitaryTrajectory" begin
    using PiccoloQuantumObjects
    using LinearAlgebra
    
    # Create system (no T_max needed!)
    H_drift = PAULIS.Z
    H_drives = [PAULIS.X, PAULIS.Y]
    sys = QuantumSystem(H_drift, H_drives, [1.0, 1.0])
    
    # Create pulse
    T = 1.0
    times = range(0, T, length=11)
    controls = zeros(2, 11)
    controls[1, :] = sin.(π .* times ./ T)  # X drive
    pulse = LinearSplinePulse(controls, collect(times))
    
    # Create quantum trajectory
    U_goal = GATES[:X]
    qtraj = UnitaryTrajectory(sys, pulse, U_goal)
    
    @test get_system(qtraj) === sys
    @test get_pulse(qtraj) === pulse
    @test get_initial(qtraj) ≈ I(2)
    @test get_goal(qtraj) === U_goal
    @test duration(qtraj) == T
    
    # Test callable interface
    U_mid = qtraj(T/2)
    @test U_mid isa Matrix{ComplexF64}
    @test size(U_mid) == (2, 2)
    
    # Test fidelity
    F = fidelity(qtraj)
    @test 0.0 ≤ F ≤ 1.0
end

@testitem "KetTrajectory" begin
    using PiccoloQuantumObjects
    using LinearAlgebra
    
    # Create system
    sys = QuantumSystem([PAULIS.X, PAULIS.Y], [1.0, 1.0])
    
    # Create pulse
    T = 1.0
    times = range(0, T, length=11)
    controls = zeros(2, 11)
    pulse = LinearSplinePulse(controls, collect(times))
    
    # Create quantum trajectory
    ψ0 = ComplexF64[1, 0]
    ψg = ComplexF64[0, 1]
    qtraj = KetTrajectory(sys, pulse, ψ0, ψg)
    
    @test get_system(qtraj) === sys
    @test get_initial(qtraj) ≈ ψ0
    @test get_goal(qtraj) ≈ ψg
    
    # Test callable interface
    ψ_mid = qtraj(T/2)
    @test ψ_mid isa Vector{ComplexF64}
    @test length(ψ_mid) == 2
    
    # Test fidelity
    F = fidelity(qtraj)
    @test 0.0 ≤ F ≤ 1.0
end

@testitem "EnsembleKetTrajectory" begin
    using PiccoloQuantumObjects
    using LinearAlgebra
    using SciMLBase: ODESolution
    
    # Create system
    sys = QuantumSystem([PAULIS.X, PAULIS.Y], [1.0, 1.0])
    
    # Create pulse
    T = 1.0
    times = range(0, T, length=11)
    controls = zeros(2, 11)
    pulse = LinearSplinePulse(controls, collect(times))
    
    # Multiple state transfers
    initials = [ComplexF64[1, 0], ComplexF64[0, 1]]
    goals = [ComplexF64[0, 1], ComplexF64[1, 0]]
    
    qtraj = EnsembleKetTrajectory(sys, pulse, initials, goals)
    
    @test get_system(qtraj) === sys
    @test length(qtraj) == 2
    @test get_initial(qtraj) == initials
    @test get_goal(qtraj) == goals
    
    # Test indexing
    @test qtraj[1] isa ODESolution
    @test qtraj[2] isa ODESolution
    
    # Test callable interface
    states = qtraj(T/2)
    @test length(states) == 2
    @test all(ψ -> ψ isa Vector{ComplexF64}, states)
    
    # Test fidelity
    F = fidelity(qtraj)
    @test 0.0 ≤ F ≤ 1.0
end

@testitem "DensityTrajectory" begin
    using PiccoloQuantumObjects
    using LinearAlgebra
    
    # Create open system
    csys = QuantumSystem([PAULIS.X, PAULIS.Y], [1.0, 1.0])
    a = ComplexF64[0 1; 0 0]  # Lowering operator
    sys = OpenQuantumSystem(csys; dissipation_operators=[0.1 * a])
    
    # Create pulse  
    T = 1.0
    times = range(0, T, length=11)
    controls = zeros(2, 11)
    pulse = LinearSplinePulse(controls, collect(times))
    
    # Create quantum trajectory
    ψ0 = ComplexF64[1, 0]
    ρ0 = ψ0 * ψ0'
    ρg = ComplexF64[0 0; 0 1] * ComplexF64[0 0; 0 1]'
    
    qtraj = DensityTrajectory(sys, pulse, ρ0, ρg)
    
    @test get_system(qtraj) === sys
    @test get_initial(qtraj) ≈ ρ0
    @test get_goal(qtraj) ≈ ρg
    
    # Test callable interface
    ρ_mid = qtraj(T/2)
    @test ρ_mid isa Matrix{ComplexF64}
    @test size(ρ_mid) == (2, 2)
    
    # Test fidelity
    F = fidelity(qtraj)
    @test 0.0 ≤ F ≤ 1.0
end

@testitem "GaussianPulse with trajectories" begin
    using PiccoloQuantumObjects
    using LinearAlgebra
    
    # Create system
    sys = QuantumSystem([PAULIS.X, PAULIS.Y], [1.0, 1.0])
    
    # Create Gaussian pulse
    T = 2.0
    pulse = GaussianPulse([0.5, 0.3], 0.2, T)
    
    @test duration(pulse) == T
    @test n_drives(pulse) == 2
    
    # Use with quantum trajectory
    ψ0 = ComplexF64[1, 0]
    ψg = ComplexF64[0, 1]
    qtraj = KetTrajectory(sys, pulse, ψ0, ψg)
    
    @test duration(qtraj) == T
    F = fidelity(qtraj)
    @test 0.0 ≤ F ≤ 1.0
end

@testitem "NamedTrajectory from UnitaryTrajectory" begin
    using PiccoloQuantumObjects
    using LinearAlgebra
    using NamedTrajectories: NamedTrajectory
    
    # Create system and pulse
    sys = QuantumSystem(PAULIS.Z, [PAULIS.X, PAULIS.Y], [1.0, 1.0])
    T = 1.0
    times = range(0, T, length=11)
    controls = zeros(2, 11)
    controls[1, :] = sin.(π .* times ./ T)
    
    # Test with LinearSplinePulse (should have u only)
    pulse = LinearSplinePulse(controls, collect(times))
    U_goal = GATES[:X]
    qtraj = UnitaryTrajectory(sys, pulse, U_goal)
    
    traj = NamedTrajectory(qtraj, 11)
    @test :Ũ⃗ ∈ traj.names
    @test :u ∈ traj.names
    @test :t ∈ traj.names   # times
    @test :Δt ∈ traj.names  # timesteps
    @test traj.timestep == :Δt
    @test :du ∉ traj.names
    @test size(traj.Ũ⃗, 2) == 11
    @test size(traj.u, 1) == 2  # 2 drives
    
    # Test with CubicSplinePulse (should have u and du)
    cubic_pulse = CubicSplinePulse(controls, collect(times))
    qtraj_cubic = UnitaryTrajectory(sys, cubic_pulse, U_goal)
    
    traj_cubic = NamedTrajectory(qtraj_cubic, 11)
    @test :u ∈ traj_cubic.names
    @test :du ∈ traj_cubic.names
    @test size(traj_cubic.du, 1) == 2
    
    # Test with times vector
    traj_times = NamedTrajectory(qtraj, collect(times))
    @test size(traj_times.Ũ⃗, 2) == 11
    
    # Check initial and goal
    @test haskey(traj.initial, :Ũ⃗)
    @test haskey(traj.goal, :Ũ⃗)
end

@testitem "NamedTrajectory from KetTrajectory" begin
    using PiccoloQuantumObjects
    using LinearAlgebra
    using NamedTrajectories: NamedTrajectory
    
    # Create system and pulse
    sys = QuantumSystem([PAULIS.X, PAULIS.Y], [1.0, 1.0])
    T = 1.0
    times = range(0, T, length=11)
    controls = zeros(2, 11)
    
    pulse = LinearSplinePulse(controls, collect(times))
    ψ0 = ComplexF64[1, 0]
    ψg = ComplexF64[0, 1]
    qtraj = KetTrajectory(sys, pulse, ψ0, ψg)
    
    traj = NamedTrajectory(qtraj, 11)
    @test :ψ̃ ∈ traj.names
    @test :u ∈ traj.names
    @test :t ∈ traj.names   # times
    @test :Δt ∈ traj.names  # timesteps
    @test traj.timestep == :Δt
    @test size(traj.ψ̃, 1) == 4  # 2-level system → 4 real components
    @test size(traj.ψ̃, 2) == 11
    
    # Check initial and goal
    @test haskey(traj.initial, :ψ̃)
    @test haskey(traj.goal, :ψ̃)
end

@testitem "NamedTrajectory from EnsembleKetTrajectory" begin
    using PiccoloQuantumObjects
    using LinearAlgebra
    using NamedTrajectories: NamedTrajectory
    
    # Create system and pulse
    sys = QuantumSystem([PAULIS.X, PAULIS.Y], [1.0, 1.0])
    T = 1.0
    times = range(0, T, length=11)
    controls = zeros(2, 11)
    
    pulse = LinearSplinePulse(controls, collect(times))
    initials = [ComplexF64[1, 0], ComplexF64[0, 1]]
    goals = [ComplexF64[0, 1], ComplexF64[1, 0]]
    
    qtraj = EnsembleKetTrajectory(sys, pulse, initials, goals)
    
    traj = NamedTrajectory(qtraj, 11)
    @test :ψ̃1 ∈ traj.names
    @test :ψ̃2 ∈ traj.names
    @test :u ∈ traj.names
    @test :t ∈ traj.names   # times
    @test :Δt ∈ traj.names  # timesteps
    @test traj.timestep == :Δt
    @test size(traj.ψ̃1, 1) == 4
    @test size(traj.ψ̃2, 1) == 4
    
    # Check initial and goal for both states
    @test haskey(traj.initial, :ψ̃1)
    @test haskey(traj.initial, :ψ̃2)
    @test haskey(traj.goal, :ψ̃1)
    @test haskey(traj.goal, :ψ̃2)
end

@testitem "state_name, drive_name, time_name accessors" begin
    using PiccoloQuantumObjects
    using LinearAlgebra
    using NamedTrajectories: NamedTrajectory
    
    # Create system and pulse
    sys = QuantumSystem(PAULIS.Z, [PAULIS.X, PAULIS.Y], [1.0, 1.0])
    T = 1.0
    times = range(0, T, length=11)
    controls = zeros(2, 11)
    pulse = LinearSplinePulse(controls, collect(times))
    
    # Test UnitaryTrajectory
    qtraj_u = UnitaryTrajectory(sys, pulse, GATES[:X])
    @test state_name(qtraj_u) == :Ũ⃗
    @test drive_name(qtraj_u) == :u
    @test time_name(qtraj_u) == :t
    @test timestep_name(qtraj_u) == :Δt
    
    # Test with custom drive_name
    pulse_a = LinearSplinePulse(controls, collect(times); drive_name=:a)
    qtraj_a = UnitaryTrajectory(sys, pulse_a, GATES[:X])
    @test drive_name(qtraj_a) == :a
    
    # Test KetTrajectory
    ψ0 = ComplexF64[1, 0]
    ψg = ComplexF64[0, 1]
    qtraj_k = KetTrajectory(sys, pulse, ψ0, ψg)
    @test state_name(qtraj_k) == :ψ̃
    @test drive_name(qtraj_k) == :u
    @test time_name(qtraj_k) == :t
    @test timestep_name(qtraj_k) == :Δt
    
    # Test EnsembleKetTrajectory
    initials = [ComplexF64[1, 0], ComplexF64[0, 1]]
    goals = [ComplexF64[0, 1], ComplexF64[1, 0]]
    qtraj_e = EnsembleKetTrajectory(sys, pulse, initials, goals)
    @test state_name(qtraj_e) == :ψ̃  # prefix
    @test state_names(qtraj_e) == [:ψ̃1, :ψ̃2]
    @test drive_name(qtraj_e) == :u
    @test time_name(qtraj_e) == :t
    @test timestep_name(qtraj_e) == :Δt
end

@testitem "AbstractQuantumTrajectory type hierarchy" begin
    using PiccoloQuantumObjects
    using LinearAlgebra
    
    sys = QuantumSystem([PAULIS.X], [1.0])
    T = 1.0
    times = range(0, T, length=11)
    controls = zeros(1, 11)
    pulse = LinearSplinePulse(controls, collect(times))
    
    # All trajectory types should be subtypes of AbstractQuantumTrajectory
    qtraj_u = UnitaryTrajectory(sys, pulse, GATES[:X])
    @test qtraj_u isa AbstractQuantumTrajectory
    
    ψ0 = ComplexF64[1, 0]
    ψg = ComplexF64[0, 1]
    qtraj_k = KetTrajectory(sys, pulse, ψ0, ψg)
    @test qtraj_k isa AbstractQuantumTrajectory
    
    initials = [ψ0, ψg]
    goals = [ψg, ψ0]
    qtraj_e = EnsembleKetTrajectory(sys, pulse, initials, goals)
    @test qtraj_e isa AbstractQuantumTrajectory
end

@testitem "rebuild KetTrajectory with new controls" begin
    using PiccoloQuantumObjects
    using LinearAlgebra
    using NamedTrajectories: NamedTrajectory
    
    # Create system with X drive (no drift)
    levels = 2
    H_drift = zeros(ComplexF64, levels, levels)
    σx = ComplexF64[0 1; 1 0]
    T = 5.0
    
    sys = QuantumSystem(H_drift, [σx], [(-2.0, 2.0)])
    
    ψ_init = ComplexF64[1, 0]
    ψ_goal = ComplexF64[0, 1]
    
    # Create initial trajectory with zero pulse
    qtraj = KetTrajectory(sys, ψ_init, ψ_goal, T)
    
    # Verify initial fidelity is low (|0⟩ stays at |0⟩ with no drive)
    initial_fid = fidelity(qtraj)
    @test initial_fid < 0.1
    
    # Get NamedTrajectory and create a modified version with optimal controls
    N = 11
    traj = NamedTrajectory(qtraj, N)
    u_opt = π / (2 * T)  # For a π rotation around X: u = π/(2T) gives |0⟩ → |1⟩
    
    # Create new NamedTrajectory with optimal controls (uses :t)
    new_u = fill(u_opt, size(traj.u))
    new_traj = NamedTrajectory(
        (; ψ̃=traj.ψ̃, t=traj.t, u=new_u);
        timestep=:t,
        controls=(:t, :u),
        bounds=traj.bounds,
        initial=traj.initial,
        goal=traj.goal
    )
    
    # Rebuild trajectory with optimized controls
    new_qtraj = rebuild(qtraj, new_traj)
    
    # Check pulse was updated
    @test all(new_qtraj.pulse.controls.u .≈ u_opt)
    
    # Check ODE was re-solved with much higher fidelity
    new_fid = fidelity(new_qtraj)
    @test new_fid > 0.9
    
    # Original trajectory should be unchanged
    @test all(qtraj.pulse.controls.u .≈ 0.0)
    @test fidelity(qtraj) < 0.1
end

@testitem "rebuild UnitaryTrajectory with new controls" begin
    using PiccoloQuantumObjects
    using LinearAlgebra
    using NamedTrajectories: NamedTrajectory
    
    # Create system with X drive
    levels = 2
    H_drift = zeros(ComplexF64, levels, levels)
    σx = ComplexF64[0 1; 1 0]
    T = 5.0
    
    sys = QuantumSystem(H_drift, [σx], [(-2.0, 2.0)])
    
    U_goal = GATES[:X]
    
    # Create initial trajectory with zero pulse
    qtraj = UnitaryTrajectory(sys, U_goal, T)
    
    # Verify initial fidelity is low
    initial_fid = fidelity(qtraj)
    @test initial_fid < 0.1
    
    # Create NamedTrajectory and modify controls
    N = 11
    traj = NamedTrajectory(qtraj, N)
    u_opt = π / (2 * T)  # π rotation for X gate
    
    new_u = fill(u_opt, size(traj.u))
    new_traj = NamedTrajectory(
        (; Ũ⃗=traj.Ũ⃗, t=traj.t, u=new_u);
        timestep=:t,
        controls=(:t, :u),
        bounds=traj.bounds,
        initial=traj.initial,
        goal=traj.goal
    )
    
    # Rebuild
    new_qtraj = rebuild(qtraj, new_traj)
    
    # Check fidelity improved
    new_fid = fidelity(new_qtraj)
    @test new_fid > 0.9
end

# ============================================================================ #
# Composite Trajectory Types for Multi-System Optimization
# ============================================================================ #

# Export composite types
export SamplingTrajectory
export get_systems, get_weights

"""
    SamplingTrajectory{QT<:AbstractQuantumTrajectory} <: AbstractQuantumTrajectory

Wrapper for robust optimization over multiple systems with shared controls.

Used for sampling-based robust optimization where:
- All systems share the same control pulse
- Each system has different dynamics (e.g., parameter variations)
- Optimization minimizes weighted fidelity across all systems

This type does NOT store a NamedTrajectory - use `NamedTrajectory(sampling, N)` for conversion.

# Fields
- `base_trajectory::QT`: Base quantum trajectory (defines pulse, initial, goal)
- `systems::Vector{<:AbstractQuantumSystem}`: Multiple systems to optimize over
- `weights::Vector{Float64}`: Weights for each system in objective

# Example
```julia
sys_nom = QuantumSystem(...)
sys_variations = [QuantumSystem(...) for _ in 1:3]  # Parameter variations
qtraj = UnitaryTrajectory(sys_nom, pulse, U_goal)
sampling = SamplingTrajectory(qtraj, sys_variations, [0.5, 0.3, 0.2])

# Convert to NamedTrajectory for optimization
traj = NamedTrajectory(sampling, 51)  # Creates :Ũ⃗1, :Ũ⃗2, :Ũ⃗3
```
"""
struct SamplingTrajectory{P<:AbstractPulse, QT<:AbstractQuantumTrajectory{P}} <: AbstractQuantumTrajectory{P}
    base_trajectory::QT
    systems::Vector{<:AbstractQuantumSystem}
    weights::Vector{Float64}
end

"""
    SamplingTrajectory(base_trajectory, systems; weights=nothing)

Create a SamplingTrajectory for robust optimization.

# Arguments
- `base_trajectory`: Base quantum trajectory (defines pulse, initial, goal)
- `systems`: Vector of systems with parameter variations

# Keyword Arguments
- `weights`: Optional weights for each system (default: equal weights)
"""
function SamplingTrajectory(
    base_trajectory::QT,
    systems::Vector{<:AbstractQuantumSystem};
    weights::Union{Nothing, Vector{Float64}}=nothing
) where {P<:AbstractPulse, QT<:AbstractQuantumTrajectory{P}}
    n = length(systems)
    if isnothing(weights)
        weights = fill(1.0 / n, n)
    end
    @assert length(weights) == n "Number of weights must match number of systems"
    return SamplingTrajectory{P, QT}(base_trajectory, systems, weights)
end

# Interface implementations for SamplingTrajectory
get_system(traj::SamplingTrajectory) = get_system(traj.base_trajectory)  # Nominal system
get_pulse(traj::SamplingTrajectory) = get_pulse(traj.base_trajectory)
get_initial(traj::SamplingTrajectory) = get_initial(traj.base_trajectory)
get_goal(traj::SamplingTrajectory) = get_goal(traj.base_trajectory)
get_solution(traj::SamplingTrajectory) = get_solution(traj.base_trajectory)
duration(traj::SamplingTrajectory) = duration(traj.base_trajectory)

# Name accessors
state_name(traj::SamplingTrajectory) = state_name(traj.base_trajectory)
drive_name(traj::SamplingTrajectory) = drive_name(traj.base_trajectory)
time_name(traj::SamplingTrajectory) = time_name(traj.base_trajectory)
timestep_name(traj::SamplingTrajectory) = timestep_name(traj.base_trajectory)

"""
    state_names(sampling::SamplingTrajectory)

Get the state variable names for all systems (e.g., [:Ũ⃗1, :Ũ⃗2, :Ũ⃗3]).
"""
function state_names(traj::SamplingTrajectory)
    base = state_name(traj)
    return [Symbol(base, i) for i in 1:length(traj.systems)]
end

"""
    get_systems(sampling::SamplingTrajectory)

Get all systems in the sampling trajectory.
"""
get_systems(traj::SamplingTrajectory) = traj.systems

"""
    get_weights(sampling::SamplingTrajectory)

Get the weights for each system.
"""
get_weights(traj::SamplingTrajectory) = traj.weights

# Length for iteration
Base.length(traj::SamplingTrajectory) = length(traj.systems)

# Callable - sample base trajectory at time t
(traj::SamplingTrajectory)(t::Real) = traj.base_trajectory(t)

# ============================================================================ #
# SamplingTrajectory NamedTrajectory Conversion
# ============================================================================ #

"""
    NamedTrajectory(sampling::SamplingTrajectory, N::Int)
    NamedTrajectory(sampling::SamplingTrajectory, times::AbstractVector)

Convert a SamplingTrajectory to a NamedTrajectory for optimization.

Creates a trajectory with multiple state variables (one per system), 
all sharing the same control pulse. Each state gets a numeric suffix:
- UnitaryTrajectory base → `:Ũ⃗1`, `:Ũ⃗2`, ...
- KetTrajectory base → `:ψ̃1`, `:ψ̃2`, ...

For robust optimization, each state variable represents the evolution under
a different system (e.g., parameter variations), but all share the same controls.

# Example
```julia
# Create sampling trajectory with 3 system variations
sampling = SamplingTrajectory(base_qtraj, [sys1, sys2, sys3])

# Convert to NamedTrajectory with 51 timesteps
traj = NamedTrajectory(sampling, 51)
# Result has: :Ũ⃗1, :Ũ⃗2, :Ũ⃗3, :u, :Δt, :t
```

# Keyword Arguments
- `Δt_bounds`: Optional tuple `(lower, upper)` for timestep bounds. If provided,
  enables free-time optimization (minimum-time problems). Default: `nothing` (no bounds).
"""
function NamedTrajectory(
    sampling::SamplingTrajectory{P, <:UnitaryTrajectory{P}},
    N_or_times::Union{Int, AbstractVector{<:Real}};
    Δt_bounds::Union{Nothing, Tuple{Float64, Float64}}=nothing
) where {P<:AbstractPulse}
    base = sampling.base_trajectory
    times = _sample_times(base, N_or_times)
    T = length(times)
    n_systems = length(sampling.systems)
    snames = state_names(sampling)
    
    # Sample base trajectory for initial state data
    base_states = [base(t) for t in times]
    Ũ⃗_base = hcat([operator_to_iso_vec(U) for U in base_states]...)
    state_dim = size(Ũ⃗_base, 1)
    
    # Build state data for each system (initially all same, dynamics will differ)
    state_data = NamedTuple()
    initial_nt = NamedTuple()
    goal_nt = NamedTuple()
    bounds = NamedTuple()
    
    # All systems share initial and goal (from base trajectory)
    U_init_iso = operator_to_iso_vec(get_initial(base))
    U_goal_iso = operator_to_iso_vec(get_goal(base))
    
    for (i, name) in enumerate(snames)
        state_data = merge(state_data, _named_tuple(name => copy(Ũ⃗_base)))
        initial_nt = merge(initial_nt, _named_tuple(name => U_init_iso))
        goal_nt = merge(goal_nt, _named_tuple(name => U_goal_iso))
        bounds = merge(bounds, _named_tuple(name => (-ones(state_dim), ones(state_dim))))
    end
    
    # Get control data from base pulse
    control_data, control_names, control_bounds = _get_control_data(get_pulse(base), times, get_system(base))
    
    # Compute Δt
    Δt_diff = diff(times)
    Δt = [Δt_diff; Δt_diff[end]]
    
    # Build data
    data = merge(state_data, (; Δt = Δt, t = collect(times)), control_data)
    bounds = merge(bounds, control_bounds)
    # Add Δt bounds if provided
    if !isnothing(Δt_bounds)
        bounds = merge(bounds, (Δt = ([Δt_bounds[1]], [Δt_bounds[2]]),))
    end
    
    return NamedTrajectory(
        data;
        timestep=:Δt,
        controls=(:Δt, control_names...),
        bounds=bounds,
        initial=initial_nt,
        goal=goal_nt
    )
end

function NamedTrajectory(
    sampling::SamplingTrajectory{P, <:KetTrajectory{P}},
    N_or_times::Union{Int, AbstractVector{<:Real}};
    Δt_bounds::Union{Nothing, Tuple{Float64, Float64}}=nothing
) where {P<:AbstractPulse}
    base = sampling.base_trajectory
    times = _sample_times(base, N_or_times)
    T = length(times)
    n_systems = length(sampling.systems)
    snames = state_names(sampling)
    
    # Sample base trajectory for initial state data
    base_states = [base(t) for t in times]
    ψ̃_base = hcat([ket_to_iso(ψ) for ψ in base_states]...)
    state_dim = size(ψ̃_base, 1)
    
    # Build state data for each system
    state_data = NamedTuple()
    initial_nt = NamedTuple()
    goal_nt = NamedTuple()
    bounds = NamedTuple()
    
    # All systems share initial and goal (from base trajectory)
    ψ_init_iso = ket_to_iso(get_initial(base))
    ψ_goal_iso = ket_to_iso(get_goal(base))
    
    for (i, name) in enumerate(snames)
        state_data = merge(state_data, _named_tuple(name => copy(ψ̃_base)))
        initial_nt = merge(initial_nt, _named_tuple(name => ψ_init_iso))
        goal_nt = merge(goal_nt, _named_tuple(name => ψ_goal_iso))
        bounds = merge(bounds, _named_tuple(name => (-ones(state_dim), ones(state_dim))))
    end
    
    # Get control data from base pulse
    control_data, control_names, control_bounds = _get_control_data(get_pulse(base), times, get_system(base))
    
    # Compute Δt
    Δt_diff = diff(times)
    Δt = [Δt_diff; Δt_diff[end]]
    
    # Build data
    data = merge(state_data, (; Δt = Δt, t = collect(times)), control_data)
    bounds = merge(bounds, control_bounds)
    # Add Δt bounds if provided
    if !isnothing(Δt_bounds)
        bounds = merge(bounds, (Δt = ([Δt_bounds[1]], [Δt_bounds[2]]),))
    end
    
    return NamedTrajectory(
        data;
        timestep=:Δt,
        controls=(:Δt, control_names...),
        bounds=bounds,
        initial=initial_nt,
        goal=goal_nt
    )
end

# ============================================================================ #
# SamplingTrajectory Rebuild
# ============================================================================ #

"""
    rebuild(sampling::SamplingTrajectory, traj::NamedTrajectory)

Rebuild a SamplingTrajectory from an optimized NamedTrajectory.

Creates a new SamplingTrajectory with updated pulse (controls) from the optimized
trajectory. The pulse is rebuilt from the control values in `traj`.

# Example
```julia
# After optimization
optimized_sampling = rebuild(sampling, optimized_traj)
```
"""
function rebuild(sampling::SamplingTrajectory, traj::NamedTrajectory)
    # Rebuild base trajectory first (this updates the pulse)
    new_base = rebuild(sampling.base_trajectory, traj)
    
    # Return new SamplingTrajectory with updated base
    return SamplingTrajectory(new_base, sampling.systems; weights=sampling.weights)
end

# ============================================================================ #
# Tests for SamplingTrajectory
# ============================================================================ #

@testitem "SamplingTrajectory with UnitaryTrajectory" begin
    using PiccoloQuantumObjects
    using PiccoloQuantumObjects: SamplingTrajectory, state_names, get_systems, get_weights
    using LinearAlgebra
    using NamedTrajectories: NamedTrajectory
    
    # Create base system and variations
    sys_nom = QuantumSystem(PAULIS.Z, [PAULIS.X], [1.0])
    sys_var1 = QuantumSystem(0.95 * PAULIS.Z, [PAULIS.X], [1.0])
    sys_var2 = QuantumSystem(1.05 * PAULIS.Z, [PAULIS.X], [1.0])
    
    # Create pulse
    T = 1.0
    times = range(0, T, length=11)
    controls = zeros(1, 11)
    pulse = LinearSplinePulse(controls, collect(times))
    
    # Create base trajectory
    U_goal = GATES[:X]
    base_qtraj = UnitaryTrajectory(sys_nom, pulse, U_goal)
    
    # Create sampling trajectory
    systems = [sys_nom, sys_var1, sys_var2]
    weights = [0.5, 0.25, 0.25]
    
    sampling = SamplingTrajectory(base_qtraj, systems; weights=weights)
    
    # Test type and accessors
    @test sampling isa AbstractQuantumTrajectory
    @test sampling isa SamplingTrajectory{<:AbstractPulse, <:UnitaryTrajectory}
    @test get_system(sampling) === sys_nom
    @test length(sampling) == 3
    @test get_systems(sampling) === systems
    @test get_weights(sampling) == weights
    @test state_names(sampling) == [:Ũ⃗1, :Ũ⃗2, :Ũ⃗3]
    @test state_name(sampling) == :Ũ⃗
    @test drive_name(sampling) == :u
    
    # Test NamedTrajectory conversion
    traj = NamedTrajectory(sampling, 11)
    @test :Ũ⃗1 ∈ traj.names
    @test :Ũ⃗2 ∈ traj.names
    @test :Ũ⃗3 ∈ traj.names
    @test :u ∈ traj.names
    @test :Δt ∈ traj.names
    @test :t ∈ traj.names
    
    # Check initial/goal propagated for each state
    for sn in state_names(sampling)
        @test haskey(traj.initial, sn)
        @test haskey(traj.goal, sn)
        @test haskey(traj.bounds, sn)
    end
end

@testitem "SamplingTrajectory with KetTrajectory" begin
    using PiccoloQuantumObjects
    using PiccoloQuantumObjects: SamplingTrajectory, state_names, get_systems, get_weights
    using LinearAlgebra
    using NamedTrajectories: NamedTrajectory
    
    # Create base system and variations
    sys_nom = QuantumSystem([PAULIS.X, PAULIS.Y], [1.0, 1.0])
    sys_var1 = QuantumSystem([0.95 * PAULIS.X, PAULIS.Y], [1.0, 1.0])
    sys_var2 = QuantumSystem([1.05 * PAULIS.X, PAULIS.Y], [1.0, 1.0])
    
    # Create pulse
    T = 1.0
    times = range(0, T, length=11)
    controls = zeros(2, 11)
    pulse = LinearSplinePulse(controls, collect(times))
    
    # Create base ket trajectory
    ψ_init = ComplexF64[1, 0]
    ψ_goal = ComplexF64[0, 1]
    base_qtraj = KetTrajectory(sys_nom, pulse, ψ_init, ψ_goal)
    
    # Create sampling trajectory with default weights
    systems = [sys_nom, sys_var1, sys_var2]
    sampling = SamplingTrajectory(base_qtraj, systems)
    
    # Test type and accessors
    @test sampling isa SamplingTrajectory{<:AbstractPulse, <:KetTrajectory}
    @test length(sampling) == 3
    @test get_weights(sampling) ≈ [1/3, 1/3, 1/3]  # Default equal weights
    @test state_names(sampling) == [:ψ̃1, :ψ̃2, :ψ̃3]
    @test state_name(sampling) == :ψ̃
    
    # Test NamedTrajectory conversion
    traj = NamedTrajectory(sampling, 11)
    @test :ψ̃1 ∈ traj.names
    @test :ψ̃2 ∈ traj.names
    @test :ψ̃3 ∈ traj.names
    @test :u ∈ traj.names
end

@testitem "SamplingTrajectory rebuild" begin
    using PiccoloQuantumObjects
    using PiccoloQuantumObjects: SamplingTrajectory, state_names, rebuild
    using LinearAlgebra
    using NamedTrajectories: NamedTrajectory
    
    # Create base system and variations
    sys_nom = QuantumSystem(PAULIS.Z, [PAULIS.X], [1.0])
    sys_var = QuantumSystem(0.95 * PAULIS.Z, [PAULIS.X], [1.0])
    
    # Create pulse
    T = 1.0
    times = range(0, T, length=11)
    controls = zeros(1, 11)
    pulse = LinearSplinePulse(controls, collect(times))
    
    # Create sampling trajectory
    base_qtraj = UnitaryTrajectory(sys_nom, pulse, GATES[:X])
    sampling = SamplingTrajectory(base_qtraj, [sys_nom, sys_var])
    
    # Convert to NamedTrajectory, modify, and rebuild
    traj = NamedTrajectory(sampling, 11)
    
    # Modify control values
    new_u = fill(0.5, size(traj.u))
    new_traj = NamedTrajectory(
        (; Ũ⃗1=traj.Ũ⃗1, Ũ⃗2=traj.Ũ⃗2, t=traj.t, Δt=traj.Δt, u=new_u);
        timestep=:Δt,
        controls=(:Δt, :u),
        bounds=traj.bounds,
        initial=traj.initial,
        goal=traj.goal
    )
    
    # Rebuild
    new_sampling = rebuild(sampling, new_traj)
    
    @test new_sampling isa SamplingTrajectory{<:AbstractPulse, <:UnitaryTrajectory}
    @test length(new_sampling) == 2
    @test get_weights(new_sampling) == sampling.weights
    
    # Check pulse was updated
    new_pulse = get_pulse(new_sampling)
    @test new_pulse(0.5)[1] ≈ 0.5
end

end # module
