module QuantumTrajectories

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

export UnitaryTrajectory, KetTrajectory, EnsembleKetTrajectory, DensityTrajectory
export get_system, get_pulse, get_initial, get_goal, get_solution

using LinearAlgebra
using SciMLBase: ODESolution, solve, remake, EnsembleProblem
using OrdinaryDiffEqLinear: MagnusGL4
using OrdinaryDiffEqTsit5: Tsit5
using TestItems

using ..QuantumSystems: AbstractQuantumSystem, QuantumSystem, OpenQuantumSystem
using ..Pulses: AbstractPulse, ZeroOrderPulse, LinearSplinePulse, CubicSplinePulse, GaussianPulse, n_drives
import ..Pulses: duration
import ..Rollouts
using ..Rollouts: UnitaryODEProblem, UnitaryOperatorODEProblem, KetODEProblem, KetOperatorODEProblem, DensityODEProblem
using ..Rollouts: unitary_fidelity
using ..EmbeddedOperators: AbstractPiccoloOperator, EmbeddedOperator
using ..Isomorphisms: operator_to_iso_vec, ket_to_iso

using NamedTrajectories: NamedTrajectory

# ============================================================================ #
# UnitaryTrajectory
# ============================================================================ #

"""
    UnitaryTrajectory{P<:AbstractPulse, S<:ODESolution}

Trajectory for unitary gate synthesis. The ODE solution is computed at construction.

# Fields
- `system::QuantumSystem`: The quantum system (physics only, no T_max)
- `pulse::P`: The control pulse
- `initial::Matrix{ComplexF64}`: Initial unitary (default: identity)
- `goal::AbstractPiccoloOperator`: Target unitary operator
- `solution::S`: Pre-computed ODE solution

# Callable
`traj(t)` returns the unitary at time `t` by interpolating the solution.
"""
struct UnitaryTrajectory{P<:AbstractPulse, S<:ODESolution, G}
    system::QuantumSystem
    pulse::P
    initial::Matrix{ComplexF64}
    goal::G
    solution::S
end

"""
    UnitaryTrajectory(system, pulse, goal; initial=I, times=..., algorithm=MagnusGL4())

Create a unitary trajectory by solving the Schrödinger equation.

# Arguments
- `system::QuantumSystem`: The quantum system
- `pulse::AbstractPulse`: The control pulse
- `goal`: Target unitary (Matrix or AbstractPiccoloOperator)

# Keyword Arguments
- `initial`: Initial unitary (default: identity matrix)
- `times`: Times to save solution at (default: 101 uniform points)
- `algorithm`: ODE solver algorithm (default: MagnusGL4())
"""
function UnitaryTrajectory(
    system::QuantumSystem,
    pulse::AbstractPulse,
    goal::G;
    initial::AbstractMatrix{<:Number}=Matrix{ComplexF64}(I, system.levels, system.levels),
    times::AbstractVector{<:Real}=range(0.0, duration(pulse), length=101),
    algorithm=MagnusGL4()
) where G
    @assert n_drives(pulse) == system.n_drives "Pulse has $(n_drives(pulse)) drives, system has $(system.n_drives)"
    
    U0 = Matrix{ComplexF64}(initial)
    prob = UnitaryOperatorODEProblem(system, pulse, collect(times); U0=U0)
    sol = solve(prob, algorithm; saveat=times)
    
    return UnitaryTrajectory{typeof(pulse), typeof(sol), G}(system, pulse, U0, goal, sol)
end

# Callable: sample solution at any time
(traj::UnitaryTrajectory)(t::Real) = traj.solution(t)

# ============================================================================ #
# KetTrajectory
# ============================================================================ #

"""
    KetTrajectory{P<:AbstractPulse, S<:ODESolution}

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
struct KetTrajectory{P<:AbstractPulse, S<:ODESolution}
    system::QuantumSystem
    pulse::P
    initial::Vector{ComplexF64}
    goal::Vector{ComplexF64}
    solution::S
end

"""
    KetTrajectory(system, pulse, initial, goal; times=..., algorithm=MagnusGL4())

Create a ket trajectory by solving the Schrödinger equation.

# Arguments
- `system::QuantumSystem`: The quantum system
- `pulse::AbstractPulse`: The control pulse
- `initial::Vector`: Initial state |ψ₀⟩
- `goal::Vector`: Target state |ψ_goal⟩

# Keyword Arguments
- `times`: Times to save solution at (default: 101 uniform points)
- `algorithm`: ODE solver algorithm (default: MagnusGL4())
"""
function KetTrajectory(
    system::QuantumSystem,
    pulse::AbstractPulse,
    initial::AbstractVector{<:Number},
    goal::AbstractVector{<:Number};
    times::AbstractVector{<:Real}=range(0.0, duration(pulse), length=101),
    algorithm=MagnusGL4()
)
    @assert n_drives(pulse) == system.n_drives "Pulse has $(n_drives(pulse)) drives, system has $(system.n_drives)"
    
    ψ0 = Vector{ComplexF64}(initial)
    ψg = Vector{ComplexF64}(goal)
    prob = KetOperatorODEProblem(system, pulse, ψ0, collect(times))
    sol = solve(prob, algorithm; saveat=times)
    
    return KetTrajectory(system, pulse, ψ0, ψg, sol)
end

# Callable: sample solution at any time
(traj::KetTrajectory)(t::Real) = traj.solution(t)

# ============================================================================ #
# EnsembleKetTrajectory
# ============================================================================ #

"""
    EnsembleKetTrajectory{P<:AbstractPulse, S<:ODESolution}

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
struct EnsembleKetTrajectory{P<:AbstractPulse, S}
    system::QuantumSystem
    pulse::P
    initials::Vector{Vector{ComplexF64}}
    goals::Vector{Vector{ComplexF64}}
    weights::Vector{Float64}
    solution::S
end

"""
    EnsembleKetTrajectory(system, pulse, initials, goals; weights=..., times=..., algorithm=MagnusGL4())

Create an ensemble ket trajectory by solving multiple Schrödinger equations.

# Arguments
- `system::QuantumSystem`: The quantum system
- `pulse::AbstractPulse`: The shared control pulse
- `initials::Vector{Vector}`: Initial states
- `goals::Vector{Vector}`: Target states

# Keyword Arguments
- `weights`: Weights for fidelity (default: uniform)
- `times`: Times to save solution at (default: 101 uniform points)
- `algorithm`: ODE solver algorithm (default: MagnusGL4())
"""
function EnsembleKetTrajectory(
    system::QuantumSystem,
    pulse::AbstractPulse,
    initials::Vector{<:AbstractVector{<:Number}},
    goals::Vector{<:AbstractVector{<:Number}};
    weights::AbstractVector{<:Real}=fill(1.0/length(initials), length(initials)),
    times::AbstractVector{<:Real}=range(0.0, duration(pulse), length=101),
    algorithm=MagnusGL4()
)
    @assert n_drives(pulse) == system.n_drives "Pulse has $(n_drives(pulse)) drives, system has $(system.n_drives)"
    @assert length(initials) == length(goals) == length(weights) "initials, goals, and weights must have same length"
    
    ψ0s = [Vector{ComplexF64}(ψ) for ψ in initials]
    ψgs = [Vector{ComplexF64}(ψ) for ψ in goals]
    ws = Vector{Float64}(weights)
    
    # Build ensemble problem
    dummy = zeros(ComplexF64, system.levels)
    base_prob = KetOperatorODEProblem(system, pulse, dummy, collect(times))
    prob_func(prob, i, repeat) = remake(prob, u0=ψ0s[i])
    ensemble_prob = EnsembleProblem(base_prob; prob_func=prob_func)
    sol = solve(ensemble_prob, algorithm; trajectories=length(initials), saveat=times)
    
    return EnsembleKetTrajectory(system, pulse, ψ0s, ψgs, ws, sol)
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
    DensityTrajectory{P<:AbstractPulse, S<:ODESolution}

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
struct DensityTrajectory{P<:AbstractPulse, S<:ODESolution}
    system::OpenQuantumSystem
    pulse::P
    initial::Matrix{ComplexF64}
    goal::Matrix{ComplexF64}
    solution::S
end

"""
    DensityTrajectory(system, pulse, initial, goal; times=..., algorithm=Tsit5())

Create a density matrix trajectory by solving the Lindblad master equation.

# Arguments
- `system::OpenQuantumSystem`: The open quantum system
- `pulse::AbstractPulse`: The control pulse
- `initial::Matrix`: Initial density matrix ρ₀
- `goal::Matrix`: Target density matrix ρ_goal

# Keyword Arguments
- `times`: Times to save solution at (default: 101 uniform points)
- `algorithm`: ODE solver algorithm (default: Tsit5())
"""
function DensityTrajectory(
    system::OpenQuantumSystem,
    pulse::AbstractPulse,
    initial::AbstractMatrix{<:Number},
    goal::AbstractMatrix{<:Number};
    times::AbstractVector{<:Real}=range(0.0, duration(pulse), length=101),
    algorithm=Tsit5()
)
    @assert n_drives(pulse) == system.n_drives "Pulse has $(n_drives(pulse)) drives, system has $(system.n_drives)"
    
    ρ0 = Matrix{ComplexF64}(initial)
    ρg = Matrix{ComplexF64}(goal)
    prob = DensityODEProblem(system, pulse, ρ0, collect(times))
    sol = solve(prob, algorithm; saveat=times)
    
    return DensityTrajectory(system, pulse, ρ0, ρg, sol)
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
get_system(traj::UnitaryTrajectory) = traj.system
get_system(traj::KetTrajectory) = traj.system
get_system(traj::EnsembleKetTrajectory) = traj.system
get_system(traj::DensityTrajectory) = traj.system

"""
    get_pulse(traj)

Get the control pulse from a trajectory.
"""
get_pulse(traj::UnitaryTrajectory) = traj.pulse
get_pulse(traj::KetTrajectory) = traj.pulse
get_pulse(traj::EnsembleKetTrajectory) = traj.pulse
get_pulse(traj::DensityTrajectory) = traj.pulse

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
get_solution(traj::UnitaryTrajectory) = traj.solution
get_solution(traj::KetTrajectory) = traj.solution
get_solution(traj::EnsembleKetTrajectory) = traj.solution
get_solution(traj::DensityTrajectory) = traj.solution

"""
    duration(traj)

Get the duration of a trajectory (from its pulse).
"""
duration(traj::UnitaryTrajectory) = duration(traj.pulse)
duration(traj::KetTrajectory) = duration(traj.pulse)
duration(traj::EnsembleKetTrajectory) = duration(traj.pulse)
duration(traj::DensityTrajectory) = duration(traj.pulse)

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

For ZeroOrderPulse and LinearSplinePulse: return only `u` data with system bounds.
"""
function _get_control_data(pulse::Union{ZeroOrderPulse, LinearSplinePulse}, times::AbstractVector, sys::AbstractQuantumSystem)
    u = hcat([pulse(t) for t in times]...)
    u_bounds = _get_drive_bounds(sys)
    return (u = u,), (:u,), (u = u_bounds,)
end

"""
    _get_control_data(pulse::CubicSplinePulse, times, sys)

For CubicSplinePulse: return `u` and `du` data with system bounds.
"""
function _get_control_data(pulse::CubicSplinePulse, times::AbstractVector, sys::AbstractQuantumSystem)
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
    
    return (u = u, du = du), (:u, :du), (u = u_bounds, du = du_bounds)
end

"""
    _get_control_data(pulse::GaussianPulse, times, sys)

For GaussianPulse: sample as u values with system bounds.
"""
function _get_control_data(pulse::GaussianPulse, times::AbstractVector, sys::AbstractQuantumSystem)
    u = hcat([pulse(t) for t in times]...)
    u_bounds = _get_drive_bounds(sys)
    return (u = u,), (:u,), (u = u_bounds,)
end

"""
    NamedTrajectory(qtraj::UnitaryTrajectory, N::Int; kwargs...)
    NamedTrajectory(qtraj::UnitaryTrajectory, times::AbstractVector; kwargs...)

Convert a UnitaryTrajectory to a NamedTrajectory for optimization.

Stores:
- `Ũ⃗`: Isomorphism of unitary (vectorized real representation)
- `u`: Control values (and `du` for CubicSplinePulse)
- `Δt`: Timesteps
- `t`: Times (as global data)

# Keyword Arguments
- `timestep_bounds`: Bounds on timestep (default: auto from times)
- `state_name`: Name for state variable (default: `:Ũ⃗`)
- `free_time`: Whether timesteps are free variables (default: `false`)
"""
function NamedTrajectory(
    qtraj::UnitaryTrajectory,
    N_or_times::Union{Int, AbstractVector{<:Real}};
    timestep_bounds::Union{Nothing, Tuple{<:Real, <:Real}}=nothing,
    state_name::Symbol=:Ũ⃗,
    free_time::Bool=false
)
    times = _sample_times(qtraj, N_or_times)
    T = length(times)
    
    # Sample unitary states
    states = [qtraj(t) for t in times]
    Ũ⃗ = hcat([operator_to_iso_vec(U) for U in states]...)
    
    # Get control data based on pulse type
    control_data, control_names, control_bounds = _get_control_data(qtraj.pulse, times, qtraj.system)
    
    # Compute timesteps
    Δt_vals = diff(times)
    Δt = [Δt_vals; Δt_vals[end]]  # Pad last timestep
    
    # Timestep bounds
    if isnothing(timestep_bounds)
        timestep_bounds = free_time ? (0.5 * minimum(Δt), 2.0 * maximum(Δt)) : (minimum(Δt), maximum(Δt))
    end
    
    # State dimension
    state_dim = size(Ũ⃗, 1)
    
    # Build data NamedTuple
    data = merge(_named_tuple(state_name => Ũ⃗, :Δt => Δt), control_data)
    
    # Initial and final conditions
    initial = _named_tuple(state_name => operator_to_iso_vec(qtraj.initial))
    U_goal = qtraj.goal isa EmbeddedOperator ? qtraj.goal.operator : qtraj.goal
    goal = _named_tuple(state_name => operator_to_iso_vec(U_goal))
    
    # Bounds
    bounds = merge(
        _named_tuple(state_name => (-ones(state_dim), ones(state_dim)), :Δt => timestep_bounds),
        control_bounds
    )
    
    return NamedTrajectory(
        data;
        timestep=:Δt,
        controls=(:Δt, control_names...),
        bounds=bounds,
        initial=initial,
        goal=goal
    )
end

"""
    NamedTrajectory(qtraj::KetTrajectory, N::Int; kwargs...)
    NamedTrajectory(qtraj::KetTrajectory, times::AbstractVector; kwargs...)

Convert a KetTrajectory to a NamedTrajectory for optimization.

Stores:
- `ψ̃`: Isomorphism of ket state (real representation)
- `u`: Control values (and `du` for CubicSplinePulse)
- `Δt`: Timesteps
"""
function NamedTrajectory(
    qtraj::KetTrajectory,
    N_or_times::Union{Int, AbstractVector{<:Real}};
    timestep_bounds::Union{Nothing, Tuple{<:Real, <:Real}}=nothing,
    state_name::Symbol=:ψ̃,
    free_time::Bool=false
)
    times = _sample_times(qtraj, N_or_times)
    T = length(times)
    
    # Sample ket states
    states = [qtraj(t) for t in times]
    ψ̃ = hcat([ket_to_iso(ψ) for ψ in states]...)
    
    # Get control data based on pulse type
    control_data, control_names, control_bounds = _get_control_data(qtraj.pulse, times, qtraj.system)
    
    # Compute timesteps
    Δt_vals = diff(times)
    Δt = [Δt_vals; Δt_vals[end]]
    
    # Timestep bounds
    if isnothing(timestep_bounds)
        timestep_bounds = free_time ? (0.5 * minimum(Δt), 2.0 * maximum(Δt)) : (minimum(Δt), maximum(Δt))
    end
    
    # State dimension
    state_dim = size(ψ̃, 1)
    
    # Build data
    data = merge(_named_tuple(state_name => ψ̃, :Δt => Δt), control_data)
    
    # Initial, goal, bounds
    initial = _named_tuple(state_name => ket_to_iso(qtraj.initial))
    goal = _named_tuple(state_name => ket_to_iso(qtraj.goal))
    bounds = merge(
        _named_tuple(state_name => (-ones(state_dim), ones(state_dim)), :Δt => timestep_bounds),
        control_bounds
    )
    
    return NamedTrajectory(
        data;
        timestep=:Δt,
        controls=(:Δt, control_names...),
        bounds=bounds,
        initial=initial,
        goal=goal
    )
end

"""
    NamedTrajectory(qtraj::EnsembleKetTrajectory, N::Int; kwargs...)
    NamedTrajectory(qtraj::EnsembleKetTrajectory, times::AbstractVector; kwargs...)

Convert an EnsembleKetTrajectory to a NamedTrajectory for optimization.

Stores multiple ket states as `ψ̃1`, `ψ̃2`, etc.
"""
function NamedTrajectory(
    qtraj::EnsembleKetTrajectory,
    N_or_times::Union{Int, AbstractVector{<:Real}};
    timestep_bounds::Union{Nothing, Tuple{<:Real, <:Real}}=nothing,
    state_prefix::Symbol=:ψ̃,
    free_time::Bool=false
)
    times = _sample_times(qtraj, N_or_times)
    T = length(times)
    n_states = length(qtraj)
    
    # Sample all ket states
    state_data = NamedTuple()
    initial = NamedTuple()
    goal = NamedTuple()
    bounds = NamedTuple()
    
    for i in 1:n_states
        name = Symbol(state_prefix, i)
        sol = qtraj[i]
        states = [sol(t) for t in times]
        ψ̃ = hcat([ket_to_iso(ψ) for ψ in states]...)
        state_dim = size(ψ̃, 1)
        
        state_data = merge(state_data, _named_tuple(name => ψ̃))
        initial = merge(initial, _named_tuple(name => ket_to_iso(qtraj.initials[i])))
        goal = merge(goal, _named_tuple(name => ket_to_iso(qtraj.goals[i])))
        bounds = merge(bounds, _named_tuple(name => (-ones(state_dim), ones(state_dim))))
    end
    
    # Get control data
    control_data, control_names, control_bounds = _get_control_data(qtraj.pulse, times, qtraj.system)
    
    # Timesteps
    Δt_vals = diff(times)
    Δt = [Δt_vals; Δt_vals[end]]
    
    if isnothing(timestep_bounds)
        timestep_bounds = free_time ? (0.5 * minimum(Δt), 2.0 * maximum(Δt)) : (minimum(Δt), maximum(Δt))
    end
    
    # Build data
    data = merge(state_data, (; Δt = Δt), control_data)
    bounds = merge(bounds, (; Δt = timestep_bounds), control_bounds)
    
    return NamedTrajectory(
        data;
        timestep=:Δt,
        controls=(:Δt, control_names...),
        bounds=bounds,
        initial=initial,
        goal=goal
    )
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
    @test :Δt ∈ traj.names
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
    @test :Δt ∈ traj.names
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
    @test size(traj.ψ̃1, 1) == 4
    @test size(traj.ψ̃2, 1) == 4
    
    # Check initial and goal for both states
    @test haskey(traj.initial, :ψ̃1)
    @test haskey(traj.initial, :ψ̃2)
    @test haskey(traj.goal, :ψ̃1)
    @test haskey(traj.goal, :ψ̃2)
end

end # module
