"""
Density trajectory type for open quantum system problems.

Provides `DensityTrajectory` for open quantum systems.
"""

"""
    DensityTrajectory <: AbstractQuantumTrajectory

A trajectory for open quantum system problems.

# Fields
- `trajectory::NamedTrajectory`: The underlying trajectory data (stored as a copy)
- `system::OpenQuantumSystem`: The open quantum system
- `state_name::Symbol`: Name of the state variable (typically `:ρ⃗̃`)
- `control_name::Symbol`: Name of the control variable (typically `:u`)
- `goal::AbstractMatrix`: Target density matrix
"""
struct DensityTrajectory <: AbstractQuantumTrajectory
    trajectory::NamedTrajectory
    system::OpenQuantumSystem
    state_name::Symbol
    control_name::Symbol
    goal::AbstractMatrix
    
    function DensityTrajectory(
        sys::OpenQuantumSystem,
        ρ_init::AbstractMatrix,
        ρ_goal::AbstractMatrix,
        N::Int;
        Δt_min::Union{Float64, Nothing}=nothing,
        Δt_max::Union{Float64, Nothing}=nothing,
        Δt_bounds::Union{Tuple{Float64, Float64}, Nothing}=nothing,
        free_time::Bool=true
    )
        Δt = sys.T_max / (N - 1)
        n_drives = sys.n_drives
        
        # Handle Δt_bounds: prioritize Δt_bounds tuple if provided, else use Δt_min/Δt_max
        if !isnothing(Δt_bounds)
            _Δt_min, _Δt_max = Δt_bounds
        else
            _Δt_min = isnothing(Δt_min) ? Δt / 2 : Δt_min
            _Δt_max = isnothing(Δt_max) ? 2 * Δt : Δt_max
        end
        
        # Convert to iso representation
        ρ⃗̃_init = density_to_iso_vec(ρ_init)
        ρ⃗̃_goal = density_to_iso_vec(ρ_goal)
        
        # Linear interpolation of state
        ρ⃗̃ = linear_interpolation(ρ⃗̃_init, ρ⃗̃_goal, N)
        
        # Initialize controls (zero at boundaries)
        u = hcat(
            zeros(n_drives),
            randn(n_drives, N - 2) * 0.01,
            zeros(n_drives)
        )
        
        # Timesteps
        Δt_vec = fill(Δt, N)
        
        # Initial and final constraints
        initial = (ρ⃗̃ = ρ⃗̃_init, u = zeros(n_drives))
        final = (u = zeros(n_drives),)
        goal_constraint = (ρ⃗̃ = ρ⃗̃_goal,)
        
        # Time data (automatic for time-dependent systems)
        if sys.time_dependent
            t_data = [0.0; cumsum(Δt_vec)[1:end-1]]
            initial = merge(initial, (t = [0.0],))
        end
        
        # Bounds - convert drive_bounds from Vector{Tuple} to Tuple of Vectors
        u_lower = [sys.drive_bounds[i][1] for i in 1:n_drives]
        u_upper = [sys.drive_bounds[i][2] for i in 1:n_drives]
        _Δt_bounds = free_time ? (_Δt_min, _Δt_max) : (Δt, Δt)
        bounds = (
            u = (u_lower, u_upper),
            Δt = _Δt_bounds
        )
        
        # Build component data
        comps_data = (ρ⃗̃ = ρ⃗̃, u = u, Δt = reshape(Δt_vec, 1, N))
        
        if sys.time_dependent
            comps_data = merge(comps_data, (t = reshape(t_data, 1, N),))
        end
        
        traj = NamedTrajectory(
            comps_data;
            controls = (:u, :Δt),
            timestep = :Δt,
            initial = initial,
            final = final,
            goal = goal_constraint,
            bounds = bounds
        )
        
        return new(traj, sys, :ρ⃗̃, :u, ρ_goal)
    end
end

@testitem "DensityTrajectory high-level constructor" begin
    using PiccoloQuantumObjects
    using NamedTrajectories
    
    # Create an open quantum system
    sys = OpenQuantumSystem(
        GATES[:Z],              # H_drift
        [GATES[:X], GATES[:Y]], # H_drives
        1.0,                    # T_max
        [1.0, 1.0]             # drive_bounds
    )
    
    N = 10
    ρ_init = ComplexF64[1.0 0.0; 0.0 0.0]  # |0⟩⟨0|
    ρ_goal = ComplexF64[0.0 0.0; 0.0 1.0]  # |1⟩⟨1|
    
    # Test basic constructor
    qtraj = DensityTrajectory(sys, ρ_init, ρ_goal, N)
    @test qtraj isa DensityTrajectory
    @test size(qtraj[:ρ⃗̃], 2) == N
    @test size(qtraj[:u], 2) == N
    @test size(qtraj[:u], 1) == 2  # 2 drives
    @test get_system(qtraj) === sys
    @test get_goal(qtraj) == ρ_goal
    @test get_state_name(qtraj) == :ρ⃗̃
    @test get_control_name(qtraj) == :u
    
    # Test with fixed time
    qtraj3 = DensityTrajectory(sys, ρ_init, ρ_goal, N; free_time=false)
    @test qtraj3 isa DensityTrajectory
    Δt_val = sys.T_max / (N - 1)
    @test qtraj3.bounds[:Δt][1][1] == Δt_val
    @test qtraj3.bounds[:Δt][2][1] == Δt_val
    
    # Test with custom Δt bounds
    qtraj4 = DensityTrajectory(sys, ρ_init, ρ_goal, N; Δt_min=0.05, Δt_max=0.2)
    @test qtraj4 isa DensityTrajectory
    @test qtraj4.bounds[:Δt][1][1] == 0.05
    @test qtraj4.bounds[:Δt][2][1] == 0.2
    
    # Test that time is NOT stored for non-time-dependent systems
    @test !haskey(qtraj.components, :t)
    @test :t ∉ keys(qtraj.components)
end

@testitem "Time-dependent Hamiltonians with DensityTrajectory" begin
    using PiccoloQuantumObjects
    using NamedTrajectories
    using LinearAlgebra
    
    # Create time-dependent Hamiltonian
    ω = 2π * 5.0
    H_drift = GATES[:Z]
    H_drive = GATES[:X]
    
    H(u, t) = H_drift + u[1] * cos(ω * t) * H_drive
    
    # Create open system with time-dependent Hamiltonian
    sys = OpenQuantumSystem(H, 1.0, [1.0]; time_dependent=true)
    
    N = 10
    ρ_init = ComplexF64[1.0 0.0; 0.0 0.0]  # |0⟩⟨0|
    ρ_goal = ComplexF64[0.0 0.0; 0.0 1.0]  # |1⟩⟨1|
    
    # Test with time-dependent system (automatic time storage)
    qtraj = DensityTrajectory(sys, ρ_init, ρ_goal, N)
    @test qtraj isa DensityTrajectory
    @test haskey(qtraj.components, :t)
    @test size(qtraj[:t], 2) == N
    @test qtraj[:t][1] ≈ 0.0
    @test qtraj.initial[:t][1] ≈ 0.0
    
    # Verify time values are cumulative sums of Δt
    Δt_cumsum = [0.0; cumsum(qtraj[:Δt][:])[1:end-1]]
    @test qtraj[:t][:] ≈ Δt_cumsum
    
    # Test that time is included in components (but not controls)
    @test :t ∈ keys(qtraj.components)
    @test :t ∉ qtraj.control_names
end
