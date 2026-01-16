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

For ZeroOrderPulse and LinearSplinePulse: return `u` data with system bounds and boundary conditions.
Uses the pulse's drive_name to determine variable naming.

# Returns
- `data`: NamedTuple with control data
- `control_names`: Tuple of control variable names
- `bounds`: NamedTuple with control bounds
- `initial_constraints`: NamedTuple with initial value constraints
- `final_constraints`: NamedTuple with final value constraints
"""
function _get_control_data(pulse::Union{ZeroOrderPulse, LinearSplinePulse}, times::AbstractVector, sys::AbstractQuantumSystem)
    u_name = drive_name(pulse)
    u = hcat([pulse(t) for t in times]...)
    u_bounds = _get_drive_bounds(sys)
    
    # Extract boundary conditions from pulse (default zeros if not specified)
    initial_u = pulse.initial_value
    final_u = pulse.final_value
    
    initial_constraints = _named_tuple(u_name => initial_u)
    final_constraints = _named_tuple(u_name => final_u)
    
    return _named_tuple(u_name => u), (u_name,), _named_tuple(u_name => u_bounds), initial_constraints, final_constraints
end

"""
    _get_control_data(pulse::CubicSplinePulse, times, sys)

For CubicSplinePulse: return `u` and `du` data with system bounds and boundary conditions.
Uses the pulse's drive_name to determine variable naming.

# Returns
- `data`: NamedTuple with control data
- `control_names`: Tuple of control variable names
- `bounds`: NamedTuple with control bounds
- `initial_constraints`: NamedTuple with initial value constraints
- `final_constraints`: NamedTuple with final value constraints
"""
function _get_control_data(pulse::CubicSplinePulse, times::AbstractVector, sys::AbstractQuantumSystem)
    u_name = drive_name(pulse)
    du_name = Symbol(:d, u_name)
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
    
    # Extract boundary conditions from pulse (default zeros if not specified)
    initial_u = pulse.initial_value
    final_u = pulse.final_value
    # For spline pulses, also constrain derivatives to zero at boundaries
    initial_du = zeros(n)
    final_du = zeros(n)
    
    initial_constraints = _named_tuple(u_name => initial_u, du_name => initial_du)
    final_constraints = _named_tuple(u_name => final_u, du_name => final_du)
    
    return _named_tuple(u_name => u, du_name => du), (u_name, du_name), _named_tuple(u_name => u_bounds, du_name => du_bounds), initial_constraints, final_constraints
end

"""
    _get_control_data(pulse::GaussianPulse, times, sys)

For GaussianPulse: sample as u values with system bounds and boundary conditions.
Uses the pulse's drive_name to determine variable naming.

# Returns
- `data`: NamedTuple with control data
- `control_names`: Tuple of control variable names
- `bounds`: NamedTuple with control bounds
- `initial_constraints`: NamedTuple with initial value constraints (empty for parametric pulses)
- `final_constraints`: NamedTuple with final value constraints (empty for parametric pulses)
"""
function _get_control_data(pulse::GaussianPulse, times::AbstractVector, sys::AbstractQuantumSystem)
    u_name = drive_name(pulse)
    u = hcat([pulse(t) for t in times]...)
    u_bounds = _get_drive_bounds(sys)
    # GaussianPulse is parametric, no boundary conditions enforced
    return _named_tuple(u_name => u), (u_name,), _named_tuple(u_name => u_bounds), NamedTuple(), NamedTuple()
end

# ============================================================================ #
# Public NamedTrajectory Conversion
# ============================================================================ #

"""
    NamedTrajectory(qtraj::UnitaryTrajectory, N::Int; Δt_bounds=nothing)
    NamedTrajectory(qtraj::UnitaryTrajectory, times::AbstractVector; Δt_bounds=nothing)

Convert a UnitaryTrajectory to a NamedTrajectory for optimization.

The trajectory stores actual times `:t` (not timesteps `:Δt`), which is required
for time-dependent integrators used with `SplinePulseProblem`.

# Stored Variables
- `Ũ⃗`: Isomorphism of unitary (vectorized real representation)
- `u` (or custom drive_name): Control values sampled at times
- `du`: Control derivatives (only for CubicSplinePulse)
- `t`: Times

# Arguments
- `qtraj`: The quantum trajectory to convert
- `N::Int`: Number of uniformly spaced time points, OR
- `times::AbstractVector`: Specific times to sample at

# Keyword Arguments
- `Δt_bounds`: Optional tuple `(lower, upper)` for timestep bounds. If provided,
  enables free-time optimization (minimum-time problems). Default: `nothing` (no bounds).

# Returns
A NamedTrajectory suitable for direct collocation optimization.
"""
function NamedTrajectory(
    qtraj::UnitaryTrajectory,
    N_or_times::Union{Int, AbstractVector{<:Real}};
    Δt_bounds::Union{Nothing, Tuple{Float64, Float64}}=nothing
)
    times = _sample_times(qtraj, N_or_times)
    T = length(times)
    s_name = state_name(qtraj)
    
    # Sample unitary states
    states = [qtraj(t) for t in times]
    Ũ⃗ = hcat([operator_to_iso_vec(U) for U in states]...)
    
    # Get control data based on pulse type
    control_data, control_names, control_bounds, control_initial, control_final = _get_control_data(qtraj.pulse, times, qtraj.system)
    
    # State dimension
    state_dim = size(Ũ⃗, 1)
    
    # Compute Δt from times (pad to length T by repeating last value)
    Δt_diff = diff(times)
    Δt = [Δt_diff; Δt_diff[end]]
    
    # Build data NamedTuple with Δt as timestep and t for reference
    data = merge(
        _named_tuple(s_name => Ũ⃗, :Δt => Δt, :t => collect(times)),
        control_data
    )
    
    # Initial and final conditions (merge state and control boundaries)
    initial = merge(_named_tuple(s_name => operator_to_iso_vec(qtraj.initial)), control_initial)
    final_nt = merge(control_final)  # Control boundaries go to final (hard constraints)
    U_goal = qtraj.goal isa EmbeddedOperator ? qtraj.goal.operator : qtraj.goal
    goal_nt = _named_tuple(s_name => operator_to_iso_vec(U_goal))  # State goal (soft constraint)
    
    # Bounds (state bounded, controls bounded by system, optionally timestep bounded)
    bounds = merge(
        _named_tuple(s_name => (-ones(state_dim), ones(state_dim))),
        control_bounds
    )
    # Add Δt bounds if provided
    if !isnothing(Δt_bounds)
        bounds = merge(bounds, (Δt = ([Δt_bounds[1]], [Δt_bounds[2]]),))
    end
    
    return NamedTrajectory(
        data;
        timestep=:Δt,
        controls=(:Δt, control_names...),
        bounds=bounds,
        initial=initial,
        final=final_nt,
        goal=goal_nt
    )
end

"""
    NamedTrajectory(qtraj::KetTrajectory, N::Int; Δt_bounds=nothing)
    NamedTrajectory(qtraj::KetTrajectory, times::AbstractVector; Δt_bounds=nothing)

Convert a KetTrajectory to a NamedTrajectory for optimization.

# Stored Variables
- `ψ̃`: Isomorphism of ket state (real representation)
- `u` (or custom drive_name): Control values sampled at times
- `du`: Control derivatives (only for CubicSplinePulse)
- `t`: Times

# Keyword Arguments
- `Δt_bounds`: Optional tuple `(lower, upper)` for timestep bounds. If provided,
  enables free-time optimization (minimum-time problems). Default: `nothing` (no bounds).
"""
function NamedTrajectory(
    qtraj::KetTrajectory,
    N_or_times::Union{Int, AbstractVector{<:Real}};
    Δt_bounds::Union{Nothing, Tuple{Float64, Float64}}=nothing
)
    times = _sample_times(qtraj, N_or_times)
    T = length(times)
    s_name = state_name(qtraj)
    
    # Sample ket states
    states = [qtraj(t) for t in times]
    ψ̃ = hcat([ket_to_iso(ψ) for ψ in states]...)
    
    # Get control data based on pulse type
    control_data, control_names, control_bounds, control_initial, control_final = _get_control_data(qtraj.pulse, times, qtraj.system)
    
    # State dimension
    state_dim = size(ψ̃, 1)
    
    # Compute Δt from times (pad to length T by repeating last value)
    Δt_diff = diff(times)
    Δt = [Δt_diff; Δt_diff[end]]
    
    # Build data with Δt as timestep and t for reference
    data = merge(
        _named_tuple(s_name => ψ̃, :Δt => Δt, :t => collect(times)),
        control_data
    )
    
    # Initial, goal, bounds (merge state and control boundaries)
    initial = merge(_named_tuple(s_name => ket_to_iso(qtraj.initial)), control_initial)
    final_nt = merge(control_final)  # Control boundaries go to final (hard constraints)
    goal_nt = _named_tuple(s_name => ket_to_iso(qtraj.goal))  # State goal (soft constraint)
    bounds = merge(
        _named_tuple(s_name => (-ones(state_dim), ones(state_dim))),
        control_bounds
    )
    # Add Δt bounds if provided
    if !isnothing(Δt_bounds)
        bounds = merge(bounds, (Δt = ([Δt_bounds[1]], [Δt_bounds[2]]),))
    end
    
    return NamedTrajectory(
        data;
        timestep=:Δt,
        controls=(:Δt, control_names...),
        bounds=bounds,
        initial=initial,
        final=final_nt,
        goal=goal_nt
    )
end

"""
    NamedTrajectory(qtraj::MultiKetTrajectory, N::Int; Δt_bounds=nothing)
    NamedTrajectory(qtraj::MultiKetTrajectory, times::AbstractVector; Δt_bounds=nothing)

Convert an MultiKetTrajectory to a NamedTrajectory for optimization.

# Stored Variables
- `ψ̃1`, `ψ̃2`, ...: Isomorphisms of each ket state
- `u` (or custom drive_name): Control values sampled at times
- `du`: Control derivatives (only for CubicSplinePulse)
- `t`: Times

# Keyword Arguments
- `Δt_bounds`: Optional tuple `(lower, upper)` for timestep bounds. If provided,
  enables free-time optimization (minimum-time problems). Default: `nothing` (no bounds).
"""
function NamedTrajectory(
    qtraj::MultiKetTrajectory,
    N_or_times::Union{Int, AbstractVector{<:Real}};
    Δt_bounds::Union{Nothing, Tuple{Float64, Float64}}=nothing
)
    times = _sample_times(qtraj, N_or_times)
    T = length(times)
    n_states = length(qtraj)
    state_prefix = state_name(qtraj)
    
    # Sample all ket states
    state_data = NamedTuple()
    initial_nt = NamedTuple()
    goal_nt = NamedTuple()
    bounds = NamedTuple()
    
    for i in 1:n_states
        name = Symbol(state_prefix, i)
        sol = qtraj[i]
        states = [sol(t) for t in times]
        ψ̃ = hcat([ket_to_iso(ψ) for ψ in states]...)
        state_dim = size(ψ̃, 1)
        
        state_data = merge(state_data, _named_tuple(name => ψ̃))
        initial_nt = merge(initial_nt, _named_tuple(name => ket_to_iso(qtraj.initials[i])))
        goal_nt = merge(goal_nt, _named_tuple(name => ket_to_iso(qtraj.goals[i])))
        bounds = merge(bounds, _named_tuple(name => (-ones(state_dim), ones(state_dim))))
    end
    
    # Get control data
    control_data, control_names, control_bounds, control_initial, control_final = _get_control_data(qtraj.pulse, times, qtraj.system)
    
    # Merge control boundaries with state boundaries
    initial_nt = merge(initial_nt, control_initial)
    final_nt = merge(control_final)  # Control boundaries go to final (hard constraints)
    # goal_nt contains only state goals (soft constraints)
    
    # Compute Δt from times (pad to length T by repeating last value)
    Δt_diff = diff(times)
    Δt = [Δt_diff; Δt_diff[end]]
    
    # Build data with Δt as timestep and t for reference
    data = merge(state_data, (; Δt = Δt, t = collect(times)), control_data)
    bounds = merge(bounds, control_bounds)
    # Add Δt bounds if provided
    if !isnothing(Δt_bounds)
        bounds = merge(bounds, (Δt = ([Δt_bounds[1]], [Δt_bounds[2]]),))
    end
    
    return NamedTrajectory(
        data;
        timestep=:Δt,
        controls=(:Δt, control_names...),
        bounds=bounds,
        initial=initial_nt,
        final=final_nt,
        goal=goal_nt
    )
end

"""
    NamedTrajectory(qtraj::DensityTrajectory, N::Int; Δt_bounds=nothing)
    NamedTrajectory(qtraj::DensityTrajectory, times::AbstractVector; Δt_bounds=nothing)

Convert a DensityTrajectory to a NamedTrajectory for optimization.

# Stored Variables
- `ρ⃗̃`: Vectorized isomorphism of the density matrix
- `u` (or custom drive_name): Control values sampled at times
- `du`: Control derivatives (only for CubicSplinePulse)
- `t`: Times

# Keyword Arguments
- `Δt_bounds`: Optional tuple `(lower, upper)` for timestep bounds. If provided,
  enables free-time optimization (minimum-time problems). Default: `nothing` (no bounds).
"""
function NamedTrajectory(
    qtraj::DensityTrajectory,
    N_or_times::Union{Int, AbstractVector{<:Real}};
    Δt_bounds::Union{Nothing, Tuple{Float64, Float64}}=nothing
)
    times = _sample_times(qtraj, N_or_times)
    T = length(times)
    sname = state_name(qtraj)
    
    # Sample density matrices and convert to isomorphism (vectorized)
    # Use real-valued representation: [vec(Re(ρ)); vec(Im(ρ))]
    states = [qtraj(t) for t in times]
    ρ̃ = hcat([_density_to_iso(ρ) for ρ in states]...)
    state_dim = size(ρ̃, 1)
    
    # Get control data
    control_data, control_names, control_bounds = _get_control_data(qtraj.pulse, times, qtraj.system)
    
    # Compute Δt from times (pad to length T by repeating last value)
    Δt_diff = diff(times)
    Δt = [Δt_diff; Δt_diff[end]]
    
    # Build data with Δt as timestep and t for reference
    data = merge(
        _named_tuple(sname => ρ̃),
        (; Δt = Δt, t = collect(times)),
        control_data
    )
    
    # Note: Density matrix bounds are trickier (trace=1, positive semidefinite)
    # For now, use generous bounds on the vectorized representation
    bounds = merge(
        _named_tuple(sname => (-ones(state_dim), ones(state_dim))),
        control_bounds
    )
    
    # Add Δt bounds if provided
    if !isnothing(Δt_bounds)
        bounds = merge(bounds, (Δt = ([Δt_bounds[1]], [Δt_bounds[2]]),))
    end
    
    # Initial and goal in isomorphism
    initial = _named_tuple(sname => _density_to_iso(qtraj.initial))
    goal_nt = _named_tuple(sname => _density_to_iso(qtraj.goal))
    
    return NamedTrajectory(
        data;
        timestep=:Δt,
        controls=(:Δt, control_names...),
        bounds=bounds,
        initial=initial,
        goal=goal_nt
    )
end

# Helper: convert density matrix to real-valued isomorphism vector
function _density_to_iso(ρ::AbstractMatrix)
    return vcat(vec(real(ρ)), vec(imag(ρ)))
end

# Helper: convert isomorphism vector back to density matrix
function _iso_to_density(ρ̃::AbstractVector, n::Int)
    len = n^2
    re = reshape(ρ̃[1:len], n, n)
    im = reshape(ρ̃[len+1:end], n, n)
    return complex.(re, im)
end

# ============================================================================ #
# Tests
# ============================================================================ #

@testitem "UnitaryTrajectory" begin
    using LinearAlgebra
    using NamedTrajectories
    
    # Simple 2-level system
    system = QuantumSystem(PAULIS.Z, [PAULIS.X], [1.0])
    
    # Create trajectory with zero controls
    T = 1.0
    X_gate = [0.0 1.0; 1.0 0.0]
    qtraj = UnitaryTrajectory(system, X_gate, T)
    
    # Test interface
    @test get_system(qtraj) === system
    @test get_goal(qtraj) === X_gate
    @test state_name(qtraj) == :Ũ⃗
    @test drive_name(qtraj) == :u
    @test time_name(qtraj) == :t
    @test timestep_name(qtraj) == :Δt
    @test duration(qtraj) ≈ T
    
    # Test callable
    U0 = qtraj(0.0)
    @test U0 ≈ Matrix{ComplexF64}(I, 2, 2)
    
    # Convert to NamedTrajectory
    N = 11
    traj = NamedTrajectory(qtraj, N)
    @test length(traj.t) == N
    @test haskey(traj.components, :Ũ⃗)
    @test haskey(traj.components, :u)
end

@testitem "UnitaryTrajectory with EmbeddedOperator" begin
    using LinearAlgebra
    using NamedTrajectories
    
    # Create embedded gate in 3-level system
    X_embedded = EmbeddedOperator(:X, [1, 2], 3)
    
    # 3-level system
    H_drift = diagm(ComplexF64[1.0, 0.0, -1.0])
    H_drive1 = zeros(ComplexF64, 3, 3)
    H_drive2 = zeros(ComplexF64, 3, 3)
    H_drive1[1,2] = H_drive1[2,1] = 1.0
    H_drive2[2,3] = H_drive2[3,2] = 1.0
    system = QuantumSystem(H_drift, [H_drive1, H_drive2], [1.0, 1.0])
    
    # Create trajectory
    T = 1.0
    qtraj = UnitaryTrajectory(system, X_embedded, T)
    
    # Test that goal is preserved
    @test qtraj.goal === X_embedded
    
    # Convert to NamedTrajectory
    N = 11
    traj = NamedTrajectory(qtraj, N)
    @test haskey(traj.components, :Ũ⃗)
end

@testitem "KetTrajectory" begin
    using LinearAlgebra
    using NamedTrajectories
    
    # Simple 2-level system
    system = QuantumSystem(PAULIS.Z, [PAULIS.X], [1.0])
    
    # Create trajectory
    T = 1.0
    ψ0 = [1.0 + 0im, 0.0]
    ψg = [0.0, 1.0 + 0im]
    qtraj = KetTrajectory(system, ψ0, ψg, T)
    
    # Test interface
    @test get_system(qtraj) === system
    @test get_initial(qtraj) == ψ0
    @test get_goal(qtraj) == ψg
    @test state_name(qtraj) == :ψ̃
    @test duration(qtraj) ≈ T
    
    # Convert to NamedTrajectory
    N = 11
    traj = NamedTrajectory(qtraj, N)
    @test length(traj.t) == N
    @test haskey(traj.components, :ψ̃)
end

@testitem "MultiKetTrajectory" begin
    using LinearAlgebra
    using NamedTrajectories
    
    # Simple 2-level system
    system = QuantumSystem(PAULIS.Z, [PAULIS.X], [1.0])
    
    # Create trajectory with multiple states
    T = 1.0
    initials = [[1.0 + 0im, 0.0], [0.0, 1.0 + 0im]]
    goals = [[0.0, 1.0 + 0im], [1.0 + 0im, 0.0]]
    qtraj = MultiKetTrajectory(system, initials, goals, T)
    
    # Test interface
    @test length(qtraj) == 2
    @test state_name(qtraj) == :ψ̃
    @test state_names(qtraj) == [:ψ̃1, :ψ̃2]
    @test get_initial(qtraj) == qtraj.initials
    @test get_goal(qtraj) == qtraj.goals
    
    # Test callable and indexing
    states_0 = qtraj(0.0)
    @test length(states_0) == 2
    
    # Convert to NamedTrajectory
    N = 11
    traj = NamedTrajectory(qtraj, N)
    @test haskey(traj.components, :ψ̃1)
    @test haskey(traj.components, :ψ̃2)
end

@testitem "DensityTrajectory" begin
    using LinearAlgebra
    using NamedTrajectories
    
    # Simple 2-level open system
    L = ComplexF64[0.1 0.0; 0.0 0.0]  # Decay operator
    system = OpenQuantumSystem(PAULIS.Z, [PAULIS.X], [1.0]; dissipation_operators=[L])
    
    # Create trajectory
    T = 1.0
    ρ0 = [1.0 0.0; 0.0 0.0] .+ 0im
    ρg = [0.0 0.0; 0.0 1.0] .+ 0im
    qtraj = DensityTrajectory(system, ρ0, ρg, T)
    
    # Test interface
    @test get_system(qtraj) === system
    @test get_initial(qtraj) == ρ0
    @test get_goal(qtraj) == ρg
    @test state_name(qtraj) == :ρ⃗̃
    @test duration(qtraj) ≈ T
    
    # Convert to NamedTrajectory
    N = 11
    traj = NamedTrajectory(qtraj, N)
    @test haskey(traj.components, :ρ⃗̃)
end

@testitem "NamedTrajectory conversion with specific times" begin
    using LinearAlgebra
    using NamedTrajectories
    
    # System setup
    system = QuantumSystem(PAULIS.Z, [PAULIS.X], [1.0])
    
    # Create trajectory
    T = 1.0
    X_gate = [0.0 1.0; 1.0 0.0]
    qtraj = UnitaryTrajectory(system, X_gate, T)
    
    # Convert with specific times
    times = [0.0, 0.2, 0.5, 0.8, 1.0]
    traj = NamedTrajectory(qtraj, times)
    
    @test length(traj.t) == 5
    @test all(vec(traj.t) .≈ times)
end

@testitem "NamedTrajectory with custom drive_name" begin
    using LinearAlgebra
    using NamedTrajectories
    
    # System setup
    system = QuantumSystem(PAULIS.Z, [PAULIS.X], [1.0])
    
    # Create pulse with custom name
    T = 1.0
    times = [0.0, T]
    controls = zeros(1, 2)
    pulse = ZeroOrderPulse(controls, times; drive_name=:a)
    
    X_gate = [0.0 1.0; 1.0 0.0]
    qtraj = UnitaryTrajectory(system, pulse, X_gate)
    
    # Check that custom name propagates
    @test drive_name(qtraj) == :a
    
    traj = NamedTrajectory(qtraj, 11)
    @test haskey(traj.components, :a)
    @test !haskey(traj.components, :u)
end

@testitem "extract_pulse and rollout trajectory from NamedTrajectory" begin
    using LinearAlgebra
    using NamedTrajectories
    
    # System setup
    system = QuantumSystem(PAULIS.Z, [PAULIS.X], [1.0])
    
    # Create trajectory
    T = 1.0
    X_gate = [0.0 1.0; 1.0 0.0]
    qtraj = UnitaryTrajectory(system, X_gate, T)
    
    # Convert to NamedTrajectory
    N = 11
    traj = NamedTrajectory(qtraj, N)
    
    # Modify controls
    traj.u .= 0.5 * randn(1, N)
    
    # Extract pulse and rollout
    pulse = extract_pulse(qtraj, traj)
    qtraj2 = rollout(qtraj, pulse)
    
    @test qtraj2 isa UnitaryTrajectory
    @test qtraj2.system === system
    @test qtraj2.goal === X_gate
end

@testitem "NamedTrajectory with Δt_bounds" begin
    using LinearAlgebra
    using NamedTrajectories
    
    # System setup
    system = QuantumSystem(PAULIS.Z, [PAULIS.X], [1.0])
    
    # Create trajectory
    T = 1.0
    X_gate = [0.0 1.0; 1.0 0.0]
    qtraj = UnitaryTrajectory(system, X_gate, T)
    
    # Convert with Δt bounds
    N = 11
    traj = NamedTrajectory(qtraj, N; Δt_bounds=(0.05, 0.2))
    
    # Check that Δt bounds are set
    @test haskey(traj.bounds, :Δt)
    @test traj.bounds[:Δt][1] == [0.05]
    @test traj.bounds[:Δt][2] == [0.2]
end
