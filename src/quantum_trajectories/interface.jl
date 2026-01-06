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
