module Rollouts

"""
Rollouts of quantum systems using SciML's DifferentialEquations.jl. 

# Two Ways to Check Fidelity

## 1. Fast fidelity from quantum trajectory (O(1) - recommended)
```julia
qtraj = UnitaryTrajectory(system, pulse, goal)
fid = fidelity(qtraj)  # Uses pre-computed ODE solution
```
**Use this for:** Post-optimization fidelity checks, analysis

## 2. Validate discrete controls with interpolation (O(solve))
```julia
traj = get_trajectory(qcp)  # NamedTrajectory with discrete controls
fid = rollout_fidelity(traj, system; interpolation=:cubic)
```
**Use this for:** Testing interpolation methods, validation against discrete trajectory

# Rolling Out New Pulses

```julia
# Roll out a new pulse through the system (creates new trajectory)
qtraj_new = rollout(qtraj, new_pulse)

# In-place update after optimization
pulse = extract_pulse(qtraj, get_trajectory(qcp))
rollout!(qtraj, pulse)
```

# Provided Functions

Domain-specific language for quantum system rollouts:
- `KetODEProblem`: Ket rollouts
- `UnitaryODEProblem`: Unitary rollouts  
- `DensityODEProblem`: Density matrix rollouts (open systems)

SciML MatrixOperator versions for Lie group integrators (e.g., Magnus expansion):
- `KetOperatorODEProblem`
- `UnitaryOperatorODEProblem`

Fidelity and rollout methods:
- `fidelity(qtraj)`: Fast lookup from quantum trajectory
- `rollout(qtraj, pulse; kwargs...)`: Roll out a new pulse
- `rollout_fidelity(traj, sys; kwargs...)`: Validate discrete NamedTrajectory controls

"""

export fidelity
export unitary_fidelity
export rollout
export rollout!
export rollout_fidelity
export ket_rollout
export ket_rollout_fidelity
export unitary_rollout
export unitary_rollout_fidelity
export open_rollout
export open_rollout_fidelity

export KetODEProblem
export KetOperatorODEProblem
export UnitaryODEProblem
export UnitaryOperatorODEProblem
export DensityODEProblem

using LinearAlgebra
using NamedTrajectories
using DataInterpolations
using OrdinaryDiffEqLinear: MagnusGL4
using SciMLBase
using SymbolicIndexingInterface
const SII = SymbolicIndexingInterface
using TestItems

using ..Isomorphisms
using ..QuantumSystems

# ------------------------------------------------------------ #
# Rollout functions (stubs - extended in quantum_trajectories)
# ------------------------------------------------------------ #

"""
    rollout(qtraj, args...; kwargs...)

Roll out a quantum trajectory with new pulse or ODE parameters.
Extended in quantum_trajectories module for specific trajectory types.
"""
function rollout end

"""
    rollout!(qtraj, args...; kwargs...)

In-place rollout of quantum trajectory with new pulse or ODE parameters.
Extended in quantum_trajectories module for specific trajectory types.
"""
function rollout! end

# ------------------------------------------------------------ #
# Fidelity
# ------------------------------------------------------------ #

"""
    fidelity(ψ::AbstractVector{<:Number}, ψ_goal::AbstractVector{<:Number})

Calculate the fidelity between two quantum states `ψ` and `ψ_goal`.
"""
function fidelity(
    ψ::AbstractVector{<:Number}, 
    ψ_goal::AbstractVector{<:Number}
)
    return abs2(ψ'ψ_goal)
end

"""
    fidelity(ρ::AbstractMatrix{<:Number}, ρ_goal::AbstractMatrix{<:Number})

Calculate the fidelity between two density matrices `ρ` and `ρ_goal`.
"""
function fidelity(ρ::AbstractMatrix{<:Number}, ρ_goal::AbstractMatrix{<:Number})
    return real(tr(ρ * ρ_goal))
end

"""
    unitary_fidelity(U::AbstractMatrix{<:Number}, U_goal::AbstractMatrix{<:Number})

Calculate the fidelity between unitary operators `U` and `U_goal` in the `subspace`.
"""
function unitary_fidelity(
    U::AbstractMatrix{<:Number},
    U_goal::AbstractMatrix{<:Number};
    subspace::AbstractVector{Int}=axes(U, 1)
)
    U = U[subspace, subspace]
    U_goal = U_goal[subspace, subspace]
    N = size(U, 1)
    return abs2(tr(U' * U_goal)) / N^2
end

# ------------------------------------------------------------ #
# DSL for Piccolo
# ------------------------------------------------------------ #

const SymbolIndex = Union{
    Int,
    AbstractVector{Int},
    CartesianIndex{N} where N,
    CartesianIndices{N} where N
}

function _index(name::Symbol, n::Int)
    index = Dict{Symbol, SymbolIndex}()
    for i = 1:n
        index[Symbol(name, :_, i)] = i
    end
    index[name] = 1:n
    return index
end

function _index(name::Symbol, n1::Int, n2::Int)
    idx = Dict{Symbol, SymbolIndex}()
    for i in 1:n1, j in 1:n2
        idx[Symbol(name, :_, i, :_, j)] = CartesianIndex(i, j)
    end
    # block symbol: preserves matrix shape
    idx[name] = CartesianIndices((n1, n2))
    return idx
end

struct PiccoloRolloutSystem{T1 <: SymbolIndex}
    state_index::Dict{Symbol, T1}
    t::Symbol
    defaults::Dict{Symbol, Float64}
end

function PiccoloRolloutSystem(
    state::Pair{Symbol, Int}, 
    timestep_name::Symbol=:t,
    defaults::Dict{Symbol,Float64}=Dict{Symbol,Float64}()
)
    state_name, n_state = state
    state_index = _index(state_name, n_state)
    return PiccoloRolloutSystem(state_index, timestep_name, defaults)
end

function PiccoloRolloutSystem(
    state::Pair{Symbol, Tuple{Int, Int}}, 
    timestep_name::Symbol=:t,
    defaults::Dict{Symbol,Float64}=Dict{Symbol,Float64}()
)
    state_name, (n1, n2) = state
    state_index = _index(state_name, n1, n2)
    return PiccoloRolloutSystem(state_index, timestep_name, defaults)
end

function _construct_operator(sys::AbstractQuantumSystem, u::F) where F
    A0 = zeros(ComplexF64, sys.levels, sys.levels)
    function update!(A, x, p, t)
        Ht = collect(sys.H(u(t), t))
        @. A = -im * Ht
        return nothing
    end
    return SciMLOperators.MatrixOperator(A0; update_func! = update!)
end

function _construct_rhs(sys::AbstractQuantumSystem, u::F) where F 
    function rhs!(dx, x, p, t)
        mul!(dx, sys.H(u(t), t), x, -im, 0.0)
        return nothing
    end
    return rhs!
end

function _construct_rhs(sys::OpenQuantumSystem, u::F) where F
    Ls = sys.dissipation_operators
    Ks = map(L -> adjoint(L) * L, Ls)  # precompute L†L once
    tmp = similar(Matrix{ComplexF64}, (sys.levels, sys.levels))  # buffer

    rhs!(dρ, ρ, p, t) = begin
        Ht = sys.H(u(t), t)

        # dρ = -im*(Hρ - ρH)  (accumulate directly)
        mul!(dρ, Ht, ρ, -im, 0.0)   # dρ = -im*H*ρ
        mul!(dρ, ρ, Ht,  im, 1.0)   # dρ +=  im*ρ*H

        # dρ += Σ [ LρL† - 1/2(Kρ + ρK) ]
        @inbounds for (L, K) in zip(Ls, Ks)
            mul!(tmp, L, ρ)
            mul!(dρ, tmp, adjoint(L), 1.0, 1.0)  # dρ += tmp*L†

            mul!(dρ, K, ρ, -0.5, 1.0)
            mul!(dρ, ρ, K, -0.5, 1.0)
        end

        return nothing
    end

    return rhs!
end

# ------------------------------------------
# Standard, sparse ODE integrators
# ------------------------------------------
# TODO: document solve kwarg defaults
# TODO: states must be vector (not sparse), but could infer eltype (NT eltype?)

function KetODEProblem(
    sys::AbstractQuantumSystem, 
    u::F, 
    ψ0::Vector{ComplexF64}, 
    times::AbstractVector{<:Real}; 
    state_name::Symbol=:ψ,
    control_name::Symbol=:u,
    kwargs...
) where F
	rhs! = _construct_rhs(sys, u)
    sii_sys = PiccoloRolloutSystem(state_name => sys.levels)
	return ODEProblem(
        ODEFunction(rhs!; sys = sii_sys), ψ0, (0, times[end]); 
        tstops=times, 
        saveat=times,
        kwargs...
    )
end

function UnitaryODEProblem(
    sys::AbstractQuantumSystem, 
    u::F, 
    times::AbstractVector{<:Real};
    U0::Matrix{ComplexF64}=Matrix{ComplexF64}(I, sys.levels, sys.levels),
    state_name::Symbol=:U, 
    control_name::Symbol=:u,
    kwargs...
) where F
	rhs! = _construct_rhs(sys, u)
    sii_sys = PiccoloRolloutSystem(state_name => (sys.levels, sys.levels))
	return ODEProblem(
        ODEFunction(rhs!; sys = sii_sys), U0, (0, times[end]);
        tstops=times, 
        saveat=times,
        kwargs...
    )
end

function DensityODEProblem(
    sys::OpenQuantumSystem, 
    u::F, 
    ρ0::Matrix{ComplexF64}, 
    times::AbstractVector{<:Real}; 
    state_name::Symbol=:ρ,
    control_name::Symbol=:u,
    kwargs...
) where F
    n = sys.levels
	rhs! = _construct_rhs(sys, u)
    sii_sys = PiccoloRolloutSystem(state_name => (n, n))
	return ODEProblem(
        ODEFunction(rhs!; sys = sii_sys), ρ0, (0, times[end]);
        tstops=times, 
        saveat=times,
        kwargs...
    )
end

# ------------------------------------------
# Lie Group ODE solvers (e.g., Magnus)
# ------------------------------------------
# TODO: Operator integrator for Density

function KetOperatorODEProblem(
    sys::AbstractQuantumSystem, 
    u::F, 
    ψ0::Vector{ComplexF64}, 
    times::AbstractVector{<:Real}; 
    state_name::Symbol=:ψ,
    control_name::Symbol=:u,
    kwargs...
) where F
    op! = _construct_operator(sys, u)
    sii_sys = PiccoloRolloutSystem(state_name => sys.levels)
	return ODEProblem(
        ODEFunction(op!; sys = sii_sys), 
        ψ0, 
        (0, times[end]);
        tstops=times, 
        saveat=times,
        kwargs...
     )
end

function UnitaryOperatorODEProblem(
    sys::AbstractQuantumSystem, 
    u::F, 
    times::AbstractVector{<:Real}; 
    U0::Matrix{ComplexF64}=Matrix{ComplexF64}(I, sys.levels, sys.levels),
    state_name::Symbol=:U, 
    control_name::Symbol=:u,
    kwargs...
) where F
    op! = _construct_operator(sys, u)
    sii_sys = PiccoloRolloutSystem(state_name => (sys.levels, sys.levels))
	return ODEProblem(
        ODEFunction(op!; sys = sii_sys), 
        U0,
        (0, times[end]);
        tstops=times, 
        saveat=times,
        kwargs...
    )
end

# ------------------------------------------------------------ #
# Rollout fidelity methods
# ------------------------------------------------------------ #
# TODO: These can be extension methods for OrdinaryDiffEq
# TODO: Adapt these methods to use quantum trajectories (only _one_ rollout_fidelity method (remove unitary_rollout_fidelity), have ensemble trajectory for EnsembleProblem, etc.)

function rollout_fidelity(
    traj::NamedTrajectory, 
    sys::AbstractQuantumSystem;
    state_name::Symbol=:ψ̃,
    control_name::Symbol=:u,
    algorithm=MagnusGL4(),
    interpolation::Symbol=:linear,  # :constant, :linear, or :cubic
)
    state_names = [n for n ∈ traj.names if startswith(string(n), string(state_name))]
    isempty(state_names) && error("Trajectory does not contain $(state_name).")

    # Select interpolation method for controls
    if interpolation == :constant
        u = ConstantInterpolation(traj, control_name)
    elseif interpolation == :linear
        u = LinearInterpolation(traj, control_name)
    elseif interpolation == :cubic
        u = CubicSplineInterpolation(traj, control_name)
    else
        error("Unknown interpolation method: $(interpolation). Use :constant, :linear, or :cubic")
    end
    times = get_times(traj)

    # Blank initial state
    tmp0 = zeros(ComplexF64, sys.levels)
    rollout = KetOperatorODEProblem(sys, u, tmp0, times, state_name=state_name)

    # Ensemble over initial states
    prob_func(prob, i, repeat) = remake(prob, u0=iso_to_ket(traj.initial[state_names[i]]))
    ensemble_prob = EnsembleProblem(rollout, prob_func=prob_func)
    ensemble_sol = solve(ensemble_prob, algorithm, trajectories=length(state_names), saveat=[times[end]])
    
    fids = map(zip(ensemble_sol, state_names)) do (sol, name)
        xf = sol[state_name][end]
        xg = iso_to_ket(traj.goal[name])
        fidelity(xf, xg)
    end
    return length(fids) == 1 ? fids[1] : fids
end

function unitary_rollout_fidelity(
    traj::NamedTrajectory, 
    sys::AbstractQuantumSystem;
    state_name::Symbol=:Ũ⃗,
    control_name::Symbol=:u,
    algorithm=MagnusGL4(),
    interpolation::Symbol=:linear,  # :constant, :linear, or :cubic
)
    state_name ∉ traj.names && error("Trajectory does not contain $(state_name).")

    # Select interpolation method for controls
    if interpolation == :constant
        u = ConstantInterpolation(traj, control_name)
    elseif interpolation == :linear
        u = LinearInterpolation(traj, control_name)
    elseif interpolation == :cubic
        u = CubicSplineInterpolation(traj, control_name)
    else
        error("Unknown interpolation method: $(interpolation). Use :constant, :linear, or :cubic")
    end
    times = get_times(traj)

    x0 = iso_vec_to_operator(traj.initial[state_name])
    rollout = UnitaryOperatorODEProblem(sys, u, times, U0=x0, state_name=state_name)
    sol = solve(rollout, algorithm, saveat=[times[end]])
    xf = sol[state_name][end]
    xg = iso_vec_to_operator(traj.goal[state_name])
    return unitary_fidelity(xf, xg)
end

function unitary_rollout(
    traj::NamedTrajectory, 
    sys::AbstractQuantumSystem;
    state_name::Symbol=:Ũ⃗,
    control_name::Symbol=:u,
    algorithm=MagnusGL4(),
    interpolation::Symbol=:linear,  # :constant, :linear, or :cubic
)
    state_name ∉ traj.names && error("Trajectory does not contain $(state_name).")

    # Select interpolation method for controls
    if interpolation == :constant
        u = ConstantInterpolation(traj, control_name)
    elseif interpolation == :linear
        u = LinearInterpolation(traj, control_name)
    elseif interpolation == :cubic
        u = CubicSplineInterpolation(traj, control_name)
    else
        error("Unknown interpolation method: $(interpolation). Use :constant, :linear, or :cubic")
    end
    times = get_times(traj)

    x0 = iso_vec_to_operator(traj.initial[state_name])
    prob = UnitaryOperatorODEProblem(sys, u, times, U0=x0, state_name=state_name)
    sol = solve(prob, algorithm, saveat=times)
    
    # Extract and convert to iso-vec trajectory
    Ũ⃗_traj = hcat([operator_to_iso_vec(sol[state_name][i]) for i in 1:length(times)]...)
    
    return Ũ⃗_traj
end

function ket_rollout_fidelity(
    traj::NamedTrajectory, 
    sys::AbstractQuantumSystem;
    state_name::Symbol=:ψ̃,
    control_name::Symbol=:u,
    algorithm=MagnusGL4(),
    interpolation::Symbol=:linear,  # :constant, :linear, or :cubic
)
    return rollout_fidelity(
        traj, 
        sys; 
        state_name=state_name, 
        control_name=control_name, 
        algorithm=algorithm, 
        interpolation=interpolation
    )
end

function ket_rollout(
    traj::NamedTrajectory, 
    sys::AbstractQuantumSystem;
    state_name::Symbol=:ψ̃,
    control_name::Symbol=:u,
    algorithm=MagnusGL4(),
    interpolation::Symbol=:linear,  # :constant, :linear, or :cubic
)
    state_name ∉ traj.names && error("Trajectory does not contain $(state_name).")

    # Select interpolation method for controls
    if interpolation == :constant
        u = ConstantInterpolation(traj, control_name)
    elseif interpolation == :linear
        u = LinearInterpolation(traj, control_name)
    elseif interpolation == :cubic
        u = CubicSplineInterpolation(traj, control_name)
    else
        error("Unknown interpolation method: $(interpolation). Use :constant, :linear, or :cubic")
    end
    times = get_times(traj)

    ψ0 = iso_to_ket(traj.initial[state_name])
    prob = KetOperatorODEProblem(sys, u, ψ0, times, state_name=state_name)
    sol = solve(prob, algorithm, saveat=times)
    
    # Extract and convert to iso-vec trajectory
    ψ̃_traj = hcat([ket_to_iso(sol[state_name][i]) for i in 1:length(times)]...)
    
    return ψ̃_traj
end

# ------------------------------------------------------------ #
# Minimal interface 
# https://docs.sciml.ai/SymbolicIndexingInterface/
# ------------------------------------------------------------ #

_name(sym::Symbol) = sym   
_name(::Any) = nothing

SII.constant_structure(::PiccoloRolloutSystem) = true
SII.default_values(sys::PiccoloRolloutSystem) = sys.defaults

SII.is_time_dependent(sys::PiccoloRolloutSystem) = true
SII.is_independent_variable(sys::PiccoloRolloutSystem, sym) = _name(sym) === sys.t
SII.independent_variable_symbols(sys::PiccoloRolloutSystem) = [sys.t]

# solved variables (state)
SII.is_variable(sys::PiccoloRolloutSystem, sym) = haskey(sys.state_index, _name(sym))
SII.variable_index(sys::PiccoloRolloutSystem, sym) = get(sys.state_index, _name(sym), nothing)
SII.variable_symbols(sys::PiccoloRolloutSystem) = collect(keys(sys.state_index))

# parameters (none)
SII.is_parameter(::PiccoloRolloutSystem, _) = false
SII.parameter_index(::PiccoloRolloutSystem, _) = nothing
SII.parameter_symbols(::PiccoloRolloutSystem) = Symbol[]

SII.is_observed(sys::PiccoloRolloutSystem, sym) = false

# *************************************************************************** #
# TODO: Test rollout fidelity (after adpating to new interface)

@testitem "Test ket rollout symbolic interface" begin
    using SciMLBase: solve
    using OrdinaryDiffEqTsit5: Tsit5
    
    T, Δt = 1.0, 0.1
    sys = QuantumSystem([PAULIS.X, PAULIS.Y], [1.0, 1.0])
    ψ0 = ComplexF64[1, 0]
    u = t -> [t; 0.0]
    times = 0:Δt:T
    rollout = KetODEProblem(sys, u, ψ0, times)

    # test default
    sol1 = solve(rollout, Tsit5())
    @test sol1[:ψ] ≈ sol1.u

    # test solve kwargs
    sol2 = solve(rollout, Tsit5(), saveat=[times[end]])
    @test length(sol2[:ψ]) == 1
    @test length(sol2[:ψ][1]) == length(ψ0)

    # rename 
    rollout = KetODEProblem(sys, u, ψ0, times, state_name=:x)
    sol = solve(rollout, Tsit5())
    @test sol[:x] ≈ sol.u
end

@testitem "Test unitary rollout symbolic interface" begin
    using SciMLBase: solve
    using OrdinaryDiffEqLinear: MagnusGL4

    T, Δt = 1.0, 0.1
    sys = QuantumSystem([PAULIS.X, PAULIS.Y], [1.0, 1.0])
    u = t -> [t; 0.0]
    times = 0:Δt:T
    rollout = UnitaryOperatorODEProblem(sys, u, times)

    # test default
    sol1 = solve(rollout, MagnusGL4())
    @test sol1[:U] ≈ sol1.u

    # test solve kwargs
    sol2 = solve(rollout, MagnusGL4(), saveat=[times[end]])
    @test length(sol2[:U]) == 1
    @test size(sol2[:U][1]) == (sys.levels, sys.levels)
    
    # rename 
    rollout = UnitaryOperatorODEProblem(sys, u, times, state_name=:X)
    sol = solve(rollout, MagnusGL4())
    @test sol[:X] ≈ sol.u
end

@testitem "Test density rollout symbolic interface" begin
    using SciMLBase: solve
    using OrdinaryDiffEqTsit5: Tsit5

    T, Δt = 1.0, 0.1
    csys = QuantumSystem([PAULIS.X, PAULIS.Y], [1.0, 1.0])
    a = ComplexF64[0 1; 0 0]
    sys = OpenQuantumSystem(csys, dissipation_operators=[1e-3 * a])
    u = t -> [t; 0.0]
    times = 0:Δt:T

    ψ0 = ComplexF64[1, 0]
    ρ0 = ψ0 * ψ0'
    rollout = DensityODEProblem(sys, u, ρ0, times)

    # test default symbolic access
    sol1 = solve(rollout, Tsit5())
    @test sol1[:ρ] ≈ sol1.u

    # test solve kwargs
    sol2 = solve(rollout, Tsit5(), saveat=[times[end]])
    @test length(sol2[:ρ]) == 1
    @test size(sol2[:ρ][1]) == (sys.levels, sys.levels)

    # rename
    rollout = DensityODEProblem(sys, u, ρ0, times, state_name=:X)
    sol = solve(rollout, Tsit5())
    @test sol[:X] ≈ sol.u
end

@testitem "Rollout internal consistency (ket/unitary/density, closed system)" begin
    using SciMLBase: solve
    using OrdinaryDiffEqTsit5: Tsit5
    using OrdinaryDiffEqLinear: MagnusGL4

    T, Δt = 1.0, 0.1
    sys  = QuantumSystem([PAULIS.X, PAULIS.Y], [1.0, 1.0])
    osys = OpenQuantumSystem(sys)

    u = t -> [t; 0.0]
    times = 0:Δt:T
    ψ0 = ComplexF64[1, 0]
    ρ0 = ψ0 * ψ0'

    ket_prob = KetODEProblem(sys,  u, ψ0, times)
    U_prob = UnitaryOperatorODEProblem(sys, u, times)
    rho_prob = DensityODEProblem(osys, u, ρ0, times)

    # Save only final state so comparisons are well-defined
    kw = (dense=false, save_everystep=false, save_start=false, save_end=true)
    ket_sol = solve(ket_prob, Tsit5(); kw...)
    U_sol = solve(U_prob, MagnusGL4(); kw...)
    ρ_sol = solve(rho_prob, Tsit5(); kw...)

    ψT = ket_sol.u[end]
    UT = U_sol.u[end]
    ρT = ρ_sol.u[end]

    @test ψT ≈ UT * ψ0
    @test ρT ≈ ψT * ψT' atol=1e-5
    @test ρT ≈ UT * ρ0 * UT' atol=1e-5
end

@testitem "Rollouts with all Pulse types" begin
    using SciMLBase: solve
    using OrdinaryDiffEqTsit5: Tsit5
    using OrdinaryDiffEqLinear: MagnusGL4

    T, Δt = 1.0, 0.1
    sys = QuantumSystem([PAULIS.X, PAULIS.Y], [1.0, 1.0])
    osys = OpenQuantumSystem(sys)
    times = 0:Δt:T
    n_times = length(times)
    ψ0 = ComplexF64[1, 0]
    ρ0 = ψ0 * ψ0'

    # Generate test control values (smooth ramp)
    controls = [sin(π * t / T) for t in times]
    control_matrix = [controls zeros(n_times)]'  # 2 drives × T timesteps

    # Test all pulse types
    pulse_types = [
        ZeroOrderPulse(control_matrix, times),
        LinearSplinePulse(control_matrix, times),
        CubicSplinePulse(control_matrix, times),
    ]

    for pulse in pulse_types
        # Verify pulse is callable and returns correct shape
        @test length(pulse(0.0)) == 2
        @test pulse(0.0) ≈ [0.0, 0.0]
        
        # KetODEProblem
        ket_prob = KetODEProblem(sys, pulse, ψ0, times)
        ket_sol = solve(ket_prob, Tsit5())
        @test length(ket_sol.u) == n_times
        @test length(ket_sol.u[end]) == 2  # 2-level system
        
        # UnitaryOperatorODEProblem (for MagnusGL4)
        U_prob = UnitaryOperatorODEProblem(sys, pulse, times)
        U_sol = solve(U_prob, MagnusGL4())
        @test length(U_sol.u) == n_times
        @test size(U_sol.u[end]) == (2, 2)
        
        # DensityODEProblem
        rho_prob = DensityODEProblem(osys, pulse, ρ0, times)
        rho_sol = solve(rho_prob, Tsit5())
        @test length(rho_sol.u) == n_times
        @test size(rho_sol.u[end]) == (2, 2)
        
        # Check consistency: ψ_final should equal U_final * ψ0
        # Note: different solvers (Tsit5 vs MagnusGL4) have different accuracy
        ψT = ket_sol.u[end]
        UT = U_sol.u[end]
        @test ψT ≈ UT * ψ0 atol=1e-2
    end
end

@testitem "Rollouts with GaussianPulse" begin
    using SciMLBase: solve
    using OrdinaryDiffEqTsit5: Tsit5
    using OrdinaryDiffEqLinear: MagnusGL4

    T = 1.0
    Δt = 0.1
    sys = QuantumSystem([PAULIS.X, PAULIS.Y], [1.0, 1.0])
    times = 0:Δt:T
    n_times = length(times)
    ψ0 = ComplexF64[1, 0]

    # Create GaussianPulse with 2 drives
    # Constructor: GaussianPulse(amplitudes, sigmas, centers, duration)
    amplitudes = [1.0, 0.5]
    sigmas = [T/4, T/4]
    centers = [T/2, T/2]
    pulse = GaussianPulse(amplitudes, sigmas, centers, T)

    # Verify pulse properties
    @test duration(pulse) == T
    @test n_drives(pulse) == 2
    @test length(pulse(T/2)) == 2
    
    # Peak should be at t = center (T/2)
    @test pulse(T/2)[1] ≈ 1.0 atol=1e-10
    @test pulse(T/2)[2] ≈ 0.5 atol=1e-10
    
    # Should be symmetric around center
    @test pulse(0.25)[1] ≈ pulse(0.75)[1] atol=1e-10

    # KetODEProblem
    ket_prob = KetODEProblem(sys, pulse, ψ0, times)
    ket_sol = solve(ket_prob, Tsit5())
    @test length(ket_sol.u) == n_times

    # UnitaryOperatorODEProblem
    U_prob = UnitaryOperatorODEProblem(sys, pulse, times)
    U_sol = solve(U_prob, MagnusGL4())
    @test length(U_sol.u) == n_times

    # Check consistency
    # Note: different solvers (Tsit5 vs MagnusGL4) have different accuracy
    ψT = ket_sol.u[end]
    UT = U_sol.u[end]
    @test ψT ≈ UT * ψ0 atol=1e-2
end

@testitem "Two ways to check fidelity" begin
    using SciMLBase: solve
    using OrdinaryDiffEqLinear: MagnusGL4
    using NamedTrajectories

    # Setup
    T = 1.0
    sys = QuantumSystem([PAULIS.X, PAULIS.Y], [1.0, 1.0])
    X_gate = ComplexF64[0 1; 1 0]

    # Method 1: Fast fidelity from quantum trajectory (O(1))
    pulse = ZeroOrderPulse([0.5 0.5; 0.1 0.1], [0.0, T])
    qtraj = UnitaryTrajectory(sys, pulse, X_gate)
    fid1 = fidelity(qtraj)  # Uses stored solution - FAST!
    @test fid1 isa Float64
    @test 0.0 <= fid1 <= 1.0

    # Method 2: Validate discrete controls (for NamedTrajectory)
    # This is useful when you have discrete trajectory and want to test interpolation
    I_matrix = ComplexF64[1 0; 0 1]
    traj = NamedTrajectory(
        (Ũ⃗ = randn(8, 11), u = randn(2, 11), Δt = fill(T/10, 11));
        controls = :u,
        timestep = :Δt,
        initial = (Ũ⃗ = operator_to_iso_vec(I_matrix),),
        goal = (Ũ⃗ = operator_to_iso_vec(X_gate),)
    )
    
    # Test different interpolation methods (use unitary_rollout_fidelity for unitaries)
    fid_constant = unitary_rollout_fidelity(traj, sys; state_name=:Ũ⃗, interpolation=:constant)
    fid_linear = unitary_rollout_fidelity(traj, sys; state_name=:Ũ⃗, interpolation=:linear)
    
    @test fid_constant isa Float64
    @test fid_linear isa Float64
    @test 0.0 <= fid_constant <= 1.0
    @test 0.0 <= fid_linear <= 1.0
end

@testitem "rollout with new pulse" begin
    using OrdinaryDiffEqLinear: MagnusGL4
    
    # Setup
    T = 1.0
    sys = QuantumSystem([PAULIS.X, PAULIS.Y], [1.0, 1.0])
    X_gate = ComplexF64[0 1; 1 0]
    
    # Create initial trajectory
    pulse1 = ZeroOrderPulse([0.5 0.5; 0.1 0.1], [0.0, T])
    qtraj1 = UnitaryTrajectory(sys, pulse1, X_gate)
    fid1 = fidelity(qtraj1)
    
    # Roll out a new pulse
    pulse2 = ZeroOrderPulse([0.8 0.8; 0.2 0.2], [0.0, T])
    qtraj2 = rollout(qtraj1, pulse2)
    fid2 = fidelity(qtraj2)
    
    # Should have different fidelities (different pulses)
    @test fid2 != fid1
    @test qtraj2.pulse === pulse2
    @test qtraj2.system === qtraj1.system
    
    # Roll out with custom resolution
    qtraj3 = rollout(qtraj1, pulse2; n_points=501)
    @test length(qtraj3.solution.u) == 501
end


end
