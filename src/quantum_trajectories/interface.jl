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
# Tests
# ============================================================================ #

@testitem "Common interface - getters" begin
    using LinearAlgebra
    
    # UnitaryTrajectory
    system = QuantumSystem(PAULIS.Z, [PAULIS.X], [1.0])
    
    X_gate = ComplexF64[0 1; 1 0]
    qtraj = UnitaryTrajectory(system, X_gate, 1.0)
    
    @test get_system(qtraj) === system
    @test get_pulse(qtraj) isa AbstractPulse
    @test get_initial(qtraj) ≈ Matrix{ComplexF64}(I, 2, 2)
    @test get_goal(qtraj) === X_gate
    @test duration(qtraj) ≈ 1.0
    
    # KetTrajectory
    ψ0 = ComplexF64[1.0, 0.0]
    ψg = ComplexF64[0.0, 1.0]
    qtraj_ket = KetTrajectory(system, ψ0, ψg, 1.0)
    
    @test get_system(qtraj_ket) === system
    @test get_initial(qtraj_ket) ≈ ψ0
    @test get_goal(qtraj_ket) ≈ ψg
    
    # EnsembleKetTrajectory
    initials = [ψ0, ψg]
    goals = [ψg, ψ0]
    qtraj_ens = EnsembleKetTrajectory(system, initials, goals, 1.0)
    
    @test get_system(qtraj_ens) === system
    @test get_initial(qtraj_ens) == qtraj_ens.initials
    @test get_goal(qtraj_ens) == qtraj_ens.goals
end

@testitem "Common interface - name accessors" begin
    using LinearAlgebra
    
    system = QuantumSystem([PAULIS.X], [1.0])
    
    # Test state_name for each trajectory type
    qtraj_u = UnitaryTrajectory(system, ComplexF64[0 1; 1 0], 1.0)
    @test state_name(qtraj_u) == :Ũ⃗
    
    qtraj_k = KetTrajectory(system, ComplexF64[1, 0], ComplexF64[0, 1], 1.0)
    @test state_name(qtraj_k) == :ψ̃
    
    qtraj_e = EnsembleKetTrajectory(system, [ComplexF64[1, 0]], [ComplexF64[0, 1]], 1.0)
    @test state_name(qtraj_e) == :ψ̃
    @test state_names(qtraj_e) == [:ψ̃1]
    
    # Test drive_name propagation
    times = [0.0, 1.0]
    pulse = ZeroOrderPulse(zeros(1, 2), times; drive_name=:control)
    qtraj_custom = UnitaryTrajectory(system, pulse, ComplexF64[0 1; 1 0])
    @test drive_name(qtraj_custom) == :control
    
    # Test time_name and timestep_name (always fixed)
    @test time_name(qtraj_u) == :t
    @test timestep_name(qtraj_u) == :Δt
end

@testitem "Interface - DensityTrajectory getters" begin
    using LinearAlgebra
    
    L = ComplexF64[0.1 0.0; 0.0 0.0]
    system = OpenQuantumSystem(PAULIS.Z, [PAULIS.X], [1.0]; dissipation_operators=[L])
    
    ρ0 = ComplexF64[1.0 0.0; 0.0 0.0]
    ρg = ComplexF64[0.0 0.0; 0.0 1.0]
    qtraj = DensityTrajectory(system, ρ0, ρg, 1.0)
    
    @test get_system(qtraj) === system
    @test get_initial(qtraj) ≈ ρ0
    @test get_goal(qtraj) ≈ ρg
    @test state_name(qtraj) == :ρ⃗̃
    @test duration(qtraj) ≈ 1.0
end