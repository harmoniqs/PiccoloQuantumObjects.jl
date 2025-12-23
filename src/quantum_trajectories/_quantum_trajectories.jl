module QuantumTrajectories

"""
Quantum trajectory types and constructors for optimal control.

Provides high-level trajectory types that wrap NamedTrajectory with quantum-specific metadata:
- AbstractQuantumTrajectory: Base type
- UnitaryTrajectory: For unitary gate synthesis
- KetTrajectory: For quantum state transfer
- DensityTrajectory: For open quantum systems
- SamplingTrajectory: For robust optimization over system ensembles
- EnsembleTrajectory: For multi-state optimization with shared system

Also includes helper functions for trajectory initialization:
- unitary_geodesic: Geodesic interpolation on the unitary manifold
- unitary_linear_interpolation: Linear interpolation of unitaries
- linear_interpolation: Generic linear interpolation
"""

export AbstractQuantumTrajectory
export UnitaryTrajectory
export KetTrajectory
export DensityTrajectory
export EnsembleTrajectory
export SamplingTrajectory
export get_trajectory, get_system, get_goal, get_state_name, get_control_name, get_state, get_controls
export get_ensemble_state_names, get_systems, get_weights, get_combined_trajectory
export unitary_geodesic, unitary_linear_interpolation, linear_interpolation
export build_sampling_trajectory, build_ensemble_trajectory, build_ensemble_trajectory_from_trajectories
export update_base_trajectory

using NamedTrajectories
using LinearAlgebra
using TestItems

# Import from other PiccoloQuantumObjects modules
using ..QuantumSystems: AbstractQuantumSystem, QuantumSystem, OpenQuantumSystem, get_drift
using ..Isomorphisms: operator_to_iso_vec, iso_vec_to_operator, ket_to_iso, density_to_iso_vec
using ..EmbeddedOperators: AbstractPiccoloOperator, EmbeddedOperator, unembed, embed

# Include submodules in dependency order
include("initialization.jl")
include("abstract_trajectory.jl")
include("unitary_trajectory.jl")
include("ket_trajectory.jl")
include("density_trajectory.jl")
include("sampling_trajectory.jl")
include("ensemble_trajectory.jl")
include("trajectory_builders.jl")

end  # module QuantumTrajectories
