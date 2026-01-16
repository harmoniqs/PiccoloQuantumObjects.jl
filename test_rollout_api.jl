using PiccoloQuantumObjects
using LinearAlgebra

println("Testing dual rollout API...")

# Create a simple 2-level system
H_drift = zeros(2, 2)
H_drive = [0.0 1.0; 1.0 0.0]  # σ_x
sys = QuantumSystem([H_drive], [(-1.0, 1.0)])

# Create initial pulse
times = range(0, 1, 11)
controls = [sin(2π * t) for t in times]
pulse_old = ZeroOrderPulse(reshape(controls, 1, :), times)

# Create quantum trajectory
U_goal = [0.0+0.0im 1.0+0.0im; 1.0+0.0im 0.0+0.0im]  # NOT gate
qtraj = UnitaryTrajectory(sys, pulse_old, U_goal)

println("Initial fidelity: ", fidelity(qtraj))

# Test 1: rollout with new pulse
println("\n=== Test 1: rollout(qtraj, new_pulse) ===")
controls_new = [0.5 * sin(2π * t) for t in times]
pulse_new = ZeroOrderPulse(reshape(controls_new, 1, :), times)
qtraj_new = rollout(qtraj, pulse_new)
println("New pulse fidelity: ", fidelity(qtraj_new))
println("Original qtraj pulse unchanged: ", qtraj.pulse === pulse_old)
println("New qtraj has new pulse: ", qtraj_new.pulse === pulse_new)

# Test 2: rollout with ODE parameters
println("\n=== Test 2: rollout(qtraj; algorithm=...) ===")
using OrdinaryDiffEqLinear: MagnusGL4
using OrdinaryDiffEqTsit5: Tsit5
qtraj_rk = rollout(qtraj; algorithm=Tsit5(), n_points=51)
println("RK fidelity: ", fidelity(qtraj_rk))
println("Same pulse: ", qtraj_rk.pulse === qtraj.pulse)
println("Different solution: ", qtraj_rk.solution !== qtraj.solution)

# Test 3: rollout! with new pulse (mutating)
println("\n=== Test 3: rollout!(qtraj, new_pulse) ===")
qtraj_mut = UnitaryTrajectory(sys, pulse_old, U_goal)
println("Before rollout!: ", fidelity(qtraj_mut))
rollout!(qtraj_mut, pulse_new)
println("After rollout!: ", fidelity(qtraj_mut))
println("Pulse updated: ", qtraj_mut.pulse === pulse_new)

# Test 4: rollout! with ODE parameters (mutating)
println("\n=== Test 4: rollout!(qtraj; algorithm=...) ===")
qtraj_mut2 = UnitaryTrajectory(sys, pulse_old, U_goal)
old_sol = qtraj_mut2.solution
rollout!(qtraj_mut2; algorithm=Tsit5(), n_points=51)
println("Solution updated: ", qtraj_mut2.solution !== old_sol)
println("Pulse unchanged: ", qtraj_mut2.pulse === pulse_old)

println("\n✓ All rollout API tests passed!")
