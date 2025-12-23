"""
Trajectory builder utilities for sampling and ensemble optimization.

Provides utilities to construct combined trajectories for:
- `build_sampling_trajectory`: Duplicate state variables for sampling optimization
- `build_ensemble_trajectory`: Combine multiple trajectories into one
- `build_ensemble_trajectory_from_trajectories`: Build from AbstractQuantumTrajectory instances
"""

"""
    build_sampling_trajectory(
        base_traj::NamedTrajectory,
        state_sym::Symbol,
        n_samples::Int
    ) -> Tuple{NamedTrajectory, Vector{Symbol}}

Create a new trajectory with duplicated state variables for sampling optimization.

This function takes a base trajectory and creates a new one where:
- All non-state components (controls, derivatives, timesteps) are shared
- The state variable is duplicated for each sample with names like `:Ũ⃗_sample_1`, `:Ũ⃗_sample_2`
- Initial conditions are duplicated for each sample's state
- Goals are duplicated for each sample's state

Returns a tuple of (new_trajectory, sample_state_names).

# Arguments
- `base_traj::NamedTrajectory`: The base trajectory to duplicate
- `state_sym::Symbol`: The name of the state variable to duplicate
- `n_samples::Int`: Number of samples in the ensemble

# Returns
- `Tuple{NamedTrajectory, Vector{Symbol}}`: New trajectory and list of sample state names

# Example
```julia
base_traj = get_trajectory(qtraj)
new_traj, state_names = build_sampling_trajectory(base_traj, :Ũ⃗, 3)
# state_names = [:Ũ⃗_sample_1, :Ũ⃗_sample_2, :Ũ⃗_sample_3]
```
"""
function build_sampling_trajectory(
    base_traj::NamedTrajectory,
    state_sym::Symbol,
    n_samples::Int
)
    state_names = _sample_state_names(state_sym, n_samples)
    
    # Shared components (controls, derivatives, etc.)
    shared_components = (
        name => base_traj[name] 
        for name in base_traj.names if name != state_sym
    )
    
    # Duplicated state components for each sample
    sample_components = (
        name => copy(base_traj[state_sym]) 
        for name in state_names
    )
    
    # Merge all components
    new_components = merge(
        NamedTuple(shared_components),
        NamedTuple(sample_components)
    )
    
    # Build initial conditions: copy non-state, duplicate state
    base_init_state = base_traj.initial[state_sym]
    init_shared = (k => v for (k, v) in pairs(base_traj.initial) if k != state_sym)
    init_sample = (name => base_init_state for name in state_names)
    new_initial = merge(NamedTuple(init_shared), NamedTuple(init_sample))
    
    # Build final conditions (without state - handled by objective)
    new_final = NamedTuple(k => v for (k, v) in pairs(base_traj.final) if k != state_sym)
    
    # Build goal conditions: duplicate if present
    new_goal = if haskey(base_traj.goal, state_sym)
        base_goal = base_traj.goal[state_sym]
        NamedTuple(name => base_goal for name in state_names)
    else
        NamedTuple()
    end
    
    new_traj = NamedTrajectory(
        new_components;
        controls=base_traj.control_names,
        timestep=base_traj.timestep,
        initial=new_initial,
        final=new_final,
        goal=new_goal,
        bounds=base_traj.bounds
    )
    
    return new_traj, state_names
end

# Alias for backward compatibility
const build_ensemble_trajectory = build_sampling_trajectory

"""
    build_ensemble_trajectory_from_trajectories(
        trajectories::Vector{<:AbstractQuantumTrajectory}
    ) -> Tuple{NamedTrajectory, Vector{Symbol}}

Create a new trajectory from multiple quantum trajectories with different initial/goal states.

Each trajectory's state is included with names like `:Ũ⃗_init_1`, `:Ũ⃗_init_2`.
Controls and other components are taken from the first trajectory.

# Arguments
- `trajectories::Vector{<:AbstractQuantumTrajectory}`: Trajectories with shared system

# Returns
- `Tuple{NamedTrajectory, Vector{Symbol}}`: New trajectory and list of ensemble state names
"""
function build_ensemble_trajectory_from_trajectories(
    trajectories::Vector{<:AbstractQuantumTrajectory}
)
    @assert length(trajectories) > 0 "Must provide at least one trajectory"
    
    state_sym = get_state_name(trajectories[1])
    state_names = _ensemble_state_names(state_sym, length(trajectories))
    
    base_traj = get_trajectory(trajectories[1])
    
    # Shared components from first trajectory (controls, derivatives, etc.)
    shared_components = (
        name => base_traj[name] 
        for name in base_traj.names if name != state_sym
    )
    
    # State components from each trajectory
    ensemble_components = (
        state_names[i] => copy(get_trajectory(trajectories[i])[state_sym])
        for i in 1:length(trajectories)
    )
    
    # Merge all components
    new_components = merge(
        NamedTuple(shared_components),
        NamedTuple(ensemble_components)
    )
    
    # Build initial conditions from each trajectory
    init_shared = (k => v for (k, v) in pairs(base_traj.initial) if k != state_sym)
    init_ensemble = (
        state_names[i] => get_trajectory(trajectories[i]).initial[state_sym]
        for i in 1:length(trajectories)
    )
    new_initial = merge(NamedTuple(init_shared), NamedTuple(init_ensemble))
    
    # Build final conditions (without state - handled by objective)
    new_final = NamedTuple(k => v for (k, v) in pairs(base_traj.final) if k != state_sym)
    
    # Build goal conditions from each trajectory
    new_goal = if haskey(base_traj.goal, state_sym)
        NamedTuple(
            state_names[i] => get_trajectory(trajectories[i]).goal[state_sym]
            for i in 1:length(trajectories)
        )
    else
        NamedTuple()
    end
    
    new_traj = NamedTrajectory(
        new_components;
        controls=base_traj.control_names,
        timestep=base_traj.timestep,
        initial=new_initial,
        final=new_final,
        goal=new_goal,
        bounds=base_traj.bounds
    )
    
    return new_traj, state_names
end

"""
    build_ensemble_trajectory(qtraj::EnsembleTrajectory) -> Tuple{NamedTrajectory, Vector{Symbol}}

Build a combined NamedTrajectory from an EnsembleTrajectory.

Creates a trajectory with separate state variables for each member (e.g., `:ψ̃_init_1`, `:ψ̃_init_2`).
Controls and other components are shared.

# Returns
- `Tuple{NamedTrajectory, Vector{Symbol}}`: Combined trajectory and list of state names
"""
function build_ensemble_trajectory(qtraj::EnsembleTrajectory)
    return build_ensemble_trajectory_from_trajectories(qtraj.trajectories)
end

@testitem "build_sampling_trajectory utility" begin
    using PiccoloQuantumObjects
    using NamedTrajectories
    
    # Create a base trajectory
    sys = QuantumSystem(GATES[:Z], [GATES[:X]], 1.0, [1.0])
    base_qtraj = UnitaryTrajectory(sys, GATES[:H], 10)
    base_traj = get_trajectory(base_qtraj)
    
    # Build sampling trajectory
    n_samples = 3
    new_traj, state_names = build_sampling_trajectory(base_traj, :Ũ⃗, n_samples)
    
    # Check state names use _sample_ suffix
    @test state_names == [:Ũ⃗_sample_1, :Ũ⃗_sample_2, :Ũ⃗_sample_3]
    
    # Check that new trajectory has sample state components
    @test haskey(new_traj.components, :Ũ⃗_sample_1)
    @test haskey(new_traj.components, :Ũ⃗_sample_2)
    @test haskey(new_traj.components, :Ũ⃗_sample_3)
    
    # Check that original state is NOT in new trajectory
    @test !haskey(new_traj.components, :Ũ⃗)
    
    # Check that controls are preserved
    @test haskey(new_traj.components, :u)
    @test haskey(new_traj.components, :Δt)
    
    # Check initial conditions for each sample state
    @test haskey(new_traj.initial, :Ũ⃗_sample_1)
    @test haskey(new_traj.initial, :Ũ⃗_sample_2)
    @test haskey(new_traj.initial, :Ũ⃗_sample_3)
    @test new_traj.initial[:Ũ⃗_sample_1] == base_traj.initial[:Ũ⃗]
    
    # Check goal conditions for each sample state
    @test haskey(new_traj.goal, :Ũ⃗_sample_1)
    @test haskey(new_traj.goal, :Ũ⃗_sample_2)
    @test haskey(new_traj.goal, :Ũ⃗_sample_3)
    @test new_traj.goal[:Ũ⃗_sample_1] == base_traj.goal[:Ũ⃗]
end

@testitem "build_ensemble_trajectory_from_trajectories utility" begin
    using PiccoloQuantumObjects
    using NamedTrajectories
    
    # Create shared system
    sys = QuantumSystem(GATES[:Z], [GATES[:X]], 1.0, [1.0])
    
    # Create trajectories with different initial/goal states
    ψ0 = ComplexF64[1.0, 0.0]
    ψ1 = ComplexF64[0.0, 1.0]
    
    qtraj1 = KetTrajectory(sys, ψ0, ψ1, 10)
    qtraj2 = KetTrajectory(sys, ψ1, ψ0, 10)
    
    # Build ensemble trajectory
    new_traj, state_names = build_ensemble_trajectory_from_trajectories([qtraj1, qtraj2])
    
    # Check state names use _init_ suffix
    @test state_names == [:ψ̃_init_1, :ψ̃_init_2]
    
    # Check that new trajectory has ensemble state components
    @test haskey(new_traj.components, :ψ̃_init_1)
    @test haskey(new_traj.components, :ψ̃_init_2)
    
    # Check that original state is NOT in new trajectory
    @test !haskey(new_traj.components, :ψ̃)
    
    # Check initial conditions are from each trajectory
    @test new_traj.initial[:ψ̃_init_1] == get_trajectory(qtraj1).initial[:ψ̃]
    @test new_traj.initial[:ψ̃_init_2] == get_trajectory(qtraj2).initial[:ψ̃]
    
    # Check goal conditions are from each trajectory
    @test new_traj.goal[:ψ̃_init_1] == get_trajectory(qtraj1).goal[:ψ̃]
    @test new_traj.goal[:ψ̃_init_2] == get_trajectory(qtraj2).goal[:ψ̃]
end
