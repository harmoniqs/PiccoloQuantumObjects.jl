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
