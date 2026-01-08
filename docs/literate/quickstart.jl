# # Quickstart

# This quickstart guide will help you get up and running with PiccoloQuantumObjects.jl.

# ## Installation

# ```julia
# using Pkg
# Pkg.add("PiccoloQuantumObjects")
# ```

# ## Basic Usage

using PiccoloQuantumObjects
using LinearAlgebra
using SparseArrays
using NamedTrajectories

# ### Creating a Quantum System

# Define Hamiltonian components
H_drift = PAULIS[:Z]
H_drives = [PAULIS[:X], PAULIS[:Y]]

# Specify control bounds
drive_bounds = [(-1.0, 1.0), (-1.0, 1.0)]  # Control bounds for each drive

# Create the quantum system
system = QuantumSystem(H_drift, H_drives, drive_bounds)

# ### Working with Quantum States

# Create quantum states
ψ_ground = ket_from_string("g", [2])  # Ground state

#
ψ_excited = ket_from_string("e", [2])  # Excited state

# Calculate fidelity
fidelity(ψ_ground, ψ_excited)

# ### Simulating Evolution (Rollouts)

# Initial state
ψ_init = ComplexF64[1.0, 0.0]

# Define control sequence
T = 10  # Number of time steps
controls = rand(2, T)  # Random controls for two drives
Δt = fill(0.1, T)  # Time step duration

# Perform rollout
ψ̃_rollout = rollout(ψ_init, controls, Δt, system)

# Check final state fidelity
ψ_goal = ComplexF64[0.0, 1.0]
rollout_fidelity(ψ_init, ψ_goal, controls, Δt, system)

# ### Open Quantum Systems

# Add dissipation operators
a = annihilate(2)
dissipation_operators = [a'a, a]

# Create open quantum system
open_system = OpenQuantumSystem(
    H_drives, 
    drive_bounds, 
    dissipation_operators=dissipation_operators
)

# ### Composite Systems

# Create subsystems
sys1 = QuantumSystem([PAULIS[:X]], [(-1.0, 1.0)])
sys2 = QuantumSystem([PAULIS[:Y]], [(-1.0, 1.0)])

# Define coupling
H_coupling = 0.1 * kron(PAULIS[:Z], PAULIS[:Z])

# Create composite system
composite_sys = CompositeQuantumSystem(
    H_coupling, 
    Matrix{ComplexF64}[], 
    [sys1, sys2], 
    Float64[]
)

# ## Visualization

# PiccoloQuantumObjects.jl integrates with NamedTrajectories.jl for plotting trajectories.

using CairoMakie

# ### Plotting Controls and States

# Create a trajectory with controls and states
T_plot = 50
T_duration = 10.0  # Total evolution time
controls_plot = 0.5 * sin.(2π * (1:T_plot) / T_plot)
Δt_plot = fill(T_duration / T_plot, T_plot)

# Perform a rollout to get the state evolution
ψ̃_traj = rollout(ψ_init, hcat(controls_plot, -controls_plot)', Δt_plot, system)

# Create a NamedTrajectory for plotting
traj = NamedTrajectory(
    (
        ψ̃ = ψ̃_traj,
        a = hcat(controls_plot, -controls_plot)',
        Δt = Δt_plot
    );
    timestep=:Δt,
    controls=:a,
    initial=(ψ̃ = ket_to_iso(ψ_init),),
    goal=(ψ̃ = ket_to_iso(ψ_goal),)
)

# Plot the trajectory 
plot(traj)

# ### Plotting State Populations

# Use transformations to plot populations directly from isomorphic states
plot(
    traj, 
    [:a],
    transformations=[
        :ψ̃ => (ψ̃ -> abs2.(iso_to_ket(ψ̃)))
    ],
    transformation_labels=["Populations"]
)

#=
## Next Steps

- Explore the [Quantum Systems](@ref) manual for detailed system construction
- Learn about [Quantum Objects](@ref) for working with states and operators
- See [Rollouts](@ref) for simulation and fidelity calculations
- Understand [Isomorphisms](@ref) for the underlying mathematical transformations
=#
