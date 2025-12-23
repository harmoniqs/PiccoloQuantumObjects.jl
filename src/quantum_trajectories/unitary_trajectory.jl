"""
Unitary trajectory type for gate synthesis problems.

Provides `UnitaryTrajectory` for synthesizing quantum gates.
"""

"""
    UnitaryTrajectory <: AbstractQuantumTrajectory

A trajectory for unitary gate synthesis problems.

# Fields
- `trajectory::NamedTrajectory`: The underlying trajectory data (stored as a copy)
- `system::QuantumSystem`: The quantum system
- `state_name::Symbol`: Name of the state variable (typically `:Ũ⃗`)
- `control_name::Symbol`: Name of the control variable (typically `:u`)
- `goal::AbstractPiccoloOperator`: Target unitary operator
"""
struct UnitaryTrajectory <: AbstractQuantumTrajectory
    trajectory::NamedTrajectory
    system::QuantumSystem
    state_name::Symbol
    control_name::Symbol
    goal::AbstractPiccoloOperator
    
    function UnitaryTrajectory(
        sys::QuantumSystem,
        U_goal::AbstractMatrix{<:Number},
        N::Int;
        U_init::AbstractMatrix{<:Number}=Matrix{ComplexF64}(I(size(sys.H_drift, 1))),
        Δt_min::Union{Float64, Nothing}=nothing,
        Δt_max::Union{Float64, Nothing}=nothing,
        Δt_bounds::Union{Tuple{Float64, Float64}, Nothing}=nothing,
        free_time::Bool=true,
        geodesic::Bool=true
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
        
        # Initialize unitary trajectory
        if geodesic
            H_drift = Matrix(get_drift(sys))
            Ũ⃗ = unitary_geodesic(U_init, U_goal, N, H_drift=H_drift)
        else
            Ũ⃗ = unitary_linear_interpolation(U_init, U_goal, N)
        end

        u_maxs = [sys.drive_bounds[i][2] for i in 1:n_drives]
        
        # Initialize controls (zero at boundaries)
        u = hcat(
            zeros(n_drives),
            mapslices(u_ -> u_ .* u_maxs, 2rand(n_drives, N - 2) .- 1, dims=1),
            zeros(n_drives)
        )
        
        # Timesteps
        Δt_vec = fill(Δt, N)
        
        # Initial and final constraints
        Ũ⃗_init = operator_to_iso_vec(U_init)
        Ũ⃗_goal = operator_to_iso_vec(U_goal)
        
        initial = (Ũ⃗ = Ũ⃗_init, u = zeros(n_drives))
        final = (u = zeros(n_drives),)
        goal_constraint = (Ũ⃗ = Ũ⃗_goal,)
        
        # Time data (automatic for time-dependent systems)
        if sys.time_dependent
            t_data = cumsum([0.0; Δt_vec[1:end-1]])
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
        comps_data = (Ũ⃗ = Ũ⃗, u = u, Δt = reshape(Δt_vec, 1, N))
        
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
        
        return new(traj, sys, :Ũ⃗, :u, U_goal)
    end
end

@testitem "UnitaryTrajectory high-level constructor" begin
    using PiccoloQuantumObjects
    using NamedTrajectories
    
    # Create a simple quantum system
    sys = QuantumSystem(
        GATES[:Z],              # H_drift
        [GATES[:X], GATES[:Y]], # H_drives
        1.0,                    # T_max
        [1.0, 1.0];             # drive_bounds
        time_dependent=false
    )
    
    N = 10
    U_goal = GATES[:H]
    
    # Test basic constructor
    qtraj = UnitaryTrajectory(sys, U_goal, N)
    @test qtraj isa UnitaryTrajectory
    @test size(qtraj[:Ũ⃗], 2) == N
    @test size(qtraj[:u], 2) == N
    @test size(qtraj[:u], 1) == 2  # 2 drives
    @test get_system(qtraj) === sys
    @test get_goal(qtraj) === U_goal
    @test get_state_name(qtraj) == :Ũ⃗
    @test get_control_name(qtraj) == :u
    
    # Test with custom initial unitary
    U_init = GATES[:I]
    qtraj2 = UnitaryTrajectory(sys, U_goal, N; U_init=U_init)
    @test qtraj2 isa UnitaryTrajectory
    @test size(qtraj2[:Ũ⃗], 2) == N
    
    # Test with fixed time (free_time=false)
    qtraj3 = UnitaryTrajectory(sys, U_goal, N; free_time=false)
    @test qtraj3 isa UnitaryTrajectory
    Δt_val = sys.T_max / (N - 1)
    @test qtraj3.bounds[:Δt][1][1] == Δt_val
    @test qtraj3.bounds[:Δt][2][1] == Δt_val
    
    # Test with custom Δt bounds
    qtraj4 = UnitaryTrajectory(sys, U_goal, N; Δt_min=0.05, Δt_max=0.2)
    @test qtraj4 isa UnitaryTrajectory
    @test qtraj4.bounds[:Δt][1][1] == 0.05
    @test qtraj4.bounds[:Δt][2][1] == 0.2
    
    # Test that time is NOT stored for non-time-dependent systems
    @test !haskey(qtraj.components, :t)
    @test :t ∉ keys(qtraj.components)
    
    # Test with linear interpolation (geodesic=false)
    qtraj6 = UnitaryTrajectory(sys, U_goal, N; geodesic=false)
    @test qtraj6 isa UnitaryTrajectory
    @test size(qtraj6[:Ũ⃗], 2) == N
end

@testitem "Time-dependent Hamiltonians with UnitaryTrajectory" begin
    using PiccoloQuantumObjects
    using NamedTrajectories
    using LinearAlgebra
    
    # Create time-dependent Hamiltonian: H(u, t) = H_drift + u(t) * cos(ω*t) * H_drive
    ω = 2π * 5.0  # Drive frequency
    H_drift = GATES[:Z]
    H_drive = GATES[:X]
    
    H(u, t) = H_drift + u[1] * cos(ω * t) * H_drive
    
    # Create system with time-dependent flag
    sys = QuantumSystem(H, 1.0, [1.0]; time_dependent=true)
    
    N = 10
    U_goal = GATES[:H]
    
    # Test that time storage is automatic for time-dependent systems
    qtraj = UnitaryTrajectory(sys, U_goal, N)
    @test qtraj isa UnitaryTrajectory
    @test haskey(qtraj.components, :t)
    @test size(qtraj[:t], 2) == N
    @test qtraj[:t][1] ≈ 0.0
    @test qtraj.initial[:t][1] ≈ 0.0
    
    # Verify time values are cumulative sums of Δt
    Δt_cumsum = [0.0; cumsum(qtraj[:Δt][:])[1:end-1]]
    @test qtraj[:t][:] ≈ Δt_cumsum
    
    # Test with custom time bounds
    qtraj2 = UnitaryTrajectory(sys, U_goal, N; Δt_min=0.05, Δt_max=0.15)
    @test qtraj2 isa UnitaryTrajectory
    @test haskey(qtraj2.components, :t)
    
    # Test that time is included in components (but not controls)
    @test :t ∈ keys(qtraj.components)
    @test :t ∉ qtraj.control_names
end

@testitem "Multiple drives with time-dependent Hamiltonians" begin
    using PiccoloQuantumObjects
    using NamedTrajectories
    using LinearAlgebra
    
    # Create time-dependent Hamiltonian with multiple drives
    ω1 = 2π * 5.0
    ω2 = 2π * 3.0
    H_drift = GATES[:Z]
    H_drives = [GATES[:X], GATES[:Y]]
    
    H(u, t) = H_drift + u[1] * cos(ω1 * t) * H_drives[1] + u[2] * cos(ω2 * t) * H_drives[2]
    
    # Create system with multiple drives
    sys = QuantumSystem(H, 1.0, [1.0, 1.0]; time_dependent=true)
    
    N = 10
    U_goal = GATES[:H]
    
    # Test with multiple drives (automatic time storage)
    qtraj = UnitaryTrajectory(sys, U_goal, N)
    @test qtraj isa UnitaryTrajectory
    @test size(qtraj[:u], 1) == 2  # 2 drives
    @test haskey(qtraj.components, :t)
    @test size(qtraj[:t], 2) == N
    
    # Test initial and final control constraints
    @test all(qtraj[:u][:, 1] .== 0.0)  # Initial controls are zero
    @test all(qtraj[:u][:, end] .== 0.0)  # Final controls are zero
    
    # Test bounds on multiple drives
    @test qtraj.bounds[:u][1] == [-1.0, -1.0]  # Lower bounds
    @test qtraj.bounds[:u][2] == [1.0, 1.0]    # Upper bounds
end
