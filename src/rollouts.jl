module Rollouts

"""
Rollouts of quantum systems using SciML's DifferentialEquations.jl. Construct a an `ODEProblem` for your quantum state and call `solve`.

Provides a domain specific language for quantum system rollouts, with functions
- KetODEProblem: Ket rollouts
- UnitaryODEProblem: For unitary gate synthesis

"""
# TODO: Also include helper functions for fidelity.

export KetODEProblem
export UnitaryODEProblem
export DensityODEProblem

using LinearAlgebra
using SciMLBase
using SymbolicIndexingInterface
const SII = SymbolicIndexingInterface
using TestItems

using ..QuantumSystems


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

struct PiccoloRolloutSystem{T1 <: SymbolIndex, T2 <: SymbolIndex}
    state_index::Dict{Symbol, T1}
    control_index::Dict{Symbol, T2}
    t::Union{Symbol, Nothing}
    defaults::Dict{Symbol, Float64}
end

function PiccoloRolloutSystem(
    state::Pair{Symbol, Int}, 
    control::Pair{Symbol, Int};
    timestep_name::Symbol=:t,
    defaults::Dict{Symbol,Float64}=Dict{Symbol,Float64}()
)
    state_name, n_state = state
    control_name, n_control = control
    state_index = _index(state_name, n_state)
    control_index = _index(control_name, n_control)
    return PiccoloRolloutSystem(state_index, control_index, timestep_name, defaults)
end

function PiccoloRolloutSystem(
    state::Pair{Symbol, Tuple{Int, Int}}, 
    control::Pair{Symbol, Int};
    timestep_name::Symbol=:t,
    defaults::Dict{Symbol,Float64}=Dict{Symbol,Float64}()
)
    state_name, (n1, n2) = state
    control_name, n_control = control
    state_index = _index(state_name, n1, n2)
    control_index = _index(control_name, n_control)
    return PiccoloRolloutSystem(state_index, control_index, timestep_name, defaults)
end

struct PiccoloRolloutControl{F}
    u::F
end

function KetODEProblem(
    sys::AbstractQuantumSystem, u::Function, ψ0::Vector{ComplexF64}, T::Real; 
    state_name::Symbol=:ψ,
    control_name::Symbol=:u,
    kwargs...
)
	rhs!(dx, x, p, t) = mul!(dx, sys.H(p.u(t), t), x, -im, 0.0)

    p = PiccoloRolloutControl(u)
    sii_sys = PiccoloRolloutSystem(state_name => sys.levels, control_name => sys.n_drives)
	return ODEProblem(ODEFunction(rhs!; sys = sii_sys), ψ0, (0, T), p; kwargs...)
end

function UnitaryODEProblem(
    sys::AbstractQuantumSystem, u::Function, T::Real; 
    state_name::Symbol=:U, 
    control_name::Symbol=:u,
    kwargs...
)
	rhs!(dx, x, p, t) = mul!(dx, sys.H(p.u(t), t), x, -im, 0.0)

    p = PiccoloRolloutControl(u)
    U0 = Matrix{ComplexF64}(I, sys.levels, sys.levels)
    sii_sys = PiccoloRolloutSystem(
        state_name => (sys.levels, sys.levels), control_name => sys.n_drives)
	return ODEProblem(ODEFunction(rhs!; sys = sii_sys), U0, (0, T), p; kwargs...)
end

function DensityODEProblem(
    sys::OpenQuantumSystem, u::Function, ρ0::Matrix{ComplexF64}, T::Real; 
    state_name::Symbol=:ρ,
    control_name::Symbol=:u,
    kwargs...
)
    Ls = sys.dissipation_operators
    Ks = map(L -> adjoint(L) * L, Ls)  # precompute L†L once
    tmp = similar(ρ0)  # buffer

    rhs!(dρ, ρ, p, t) = begin
        Ht = sys.H(p.u(t), t)

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

    p = PiccoloRolloutControl(u)
    sii_sys = PiccoloRolloutSystem(
        state_name => (sys.levels, sys.levels), control_name => sys.n_drives)
	return ODEProblem(ODEFunction(rhs!; sys = sii_sys), ρ0, (0, T), p; kwargs...)
end


# ------------------------------------------------------------ #
# Minimal interface 
# https://docs.sciml.ai/SymbolicIndexingInterface/
# ------------------------------------------------------------ #

_name(sym::Symbol) = sym   
_name(sym) = nothing

# ------------------------------------------
# States (and parameters)
# ------------------------------------------

SII.constant_structure(::PiccoloRolloutSystem) = true
SII.default_values(sys::PiccoloRolloutSystem) = sys.defaults

SII.is_time_dependent(sys::PiccoloRolloutSystem) = sys.t !== nothing
SII.is_independent_variable(sys::PiccoloRolloutSystem, sym) = sys.t !== nothing && _name(sym) === sys.t
SII.independent_variable_symbols(sys::PiccoloRolloutSystem) = sys.t === nothing ? Symbol[] : [sys.t]

# solved variables (state)
SII.is_variable(sys::PiccoloRolloutSystem, sym) = haskey(sys.state_index, _name(sym))
SII.variable_index(sys::PiccoloRolloutSystem, sym) = get(sys.state_index, _name(sym), nothing)
SII.variable_symbols(sys::PiccoloRolloutSystem) = collect(keys(sys.state_index))

# parameters (none)
SII.is_parameter(::PiccoloRolloutSystem, _) = false
SII.parameter_index(::PiccoloRolloutSystem, _) = nothing
SII.parameter_symbols(::PiccoloRolloutSystem) = Symbol[]

# ------------------------------------------
# Controls
# ------------------------------------------
# “Controls are observed via parameters (p.u(t)), not solved variables, so they should not be in the ODE state vector. That avoids sol.u.u / sol.u.ψ confusion and keeps the ODE state dimension = n_state.”

SII.is_observed(sys::PiccoloRolloutSystem, sym) = haskey(sys.control_index, _name(sym))

# Prefer parameter_observed: control depends only on (p,t), not on state u
function SII.parameter_observed(sys::PiccoloRolloutSystem, sym)
    @assert SII.is_time_dependent(sys)
    s = _name(sym)
    haskey(sys.control_index, s) || return nothing
    idx = sys.control_index[s]
    return (p, t) -> (@inbounds p.u(t)[idx])
end

# Provide observed() too (some tooling expects it); just wrap parameter_observed.
function SII.observed(sys::PiccoloRolloutSystem, sym)
    g = SII.parameter_observed(sys, sym)
    g === nothing && error("Not an observed symbol: $sym")
    return SII.is_time_dependent(sys) ? ((u, p, t) -> g(p, t)) : ((u, p) -> g(p))
end

SII.all_variable_symbols(sys::PiccoloRolloutSystem) =
    vcat(SII.variable_symbols(sys), collect(keys(sys.control_index)))
SII.all_symbols(sys::PiccoloRolloutSystem) =
    vcat(SII.all_variable_symbols(sys), SII.independent_variable_symbols(sys))

# ------------------------------------------
# route SciMLBase problems/sols
# ------------------------------------------
# The generic SII methods for SciMLBase problems forward observed() through SciMLBase, which triggers the :u -> getproperty(sys,:u) recursion route.
# Intercept just ODEProblem (covers ODESolution via forwarding) when sys isa PiccoloRolloutSystem.

function SII.is_observed(prob::SciMLBase.ODEProblem, sym)
    sys = prob.f.sys
    sys isa PiccoloRolloutSystem && return SII.is_observed(sys, sym)
    return invoke(SII.is_observed, Tuple{SciMLBase.AbstractSciMLProblem, Any}, prob, sym)
end

function SII.observed(prob::SciMLBase.ODEProblem, sym)
    sys = prob.f.sys
    sys isa PiccoloRolloutSystem && return SII.observed(sys, sym)
    return invoke(SII.observed, Tuple{SciMLBase.AbstractSciMLProblem, Any}, prob, sym)
end

# *************************************************************************** #

@testitem "Test ket rollout symbolic interface" begin
    using OrdinaryDiffEq: solve
    
    sys = QuantumSystem([PAULIS.X, PAULIS.Y], 1.0, [1.0, 1.0])
    ψ0 = ComplexF64[1, 0]
    u = t -> [t; 0.0]
    rollout = KetODEProblem(sys, u, ψ0, 1.0)

    # test default
    sol1 = solve(rollout)
    @test sol1[:u] ≈ u.(sol1.t)

    # test solve kwargs
    sol2 = solve(rollout, dense=false, save_everystep=false, save_start=false, save_end=true)
    @test length(sol2[:ψ]) == 1
    @test length(sol2[:ψ][1]) == length(ψ0)

    # rename 
    rollout = KetODEProblem(sys, u, ψ0, 1.0, state_name=:x)
    sol = solve(rollout)
    @test sol[:x] ≈ sol.u
end

@testitem "Test unitary rollout symbolic interface" begin
    using OrdinaryDiffEq: solve
    
    sys = QuantumSystem([PAULIS.X, PAULIS.Y], 1.0, [1.0, 1.0])
    u = t -> [t; 0.0]
    rollout = UnitaryODEProblem(sys, u, 1.0)

    # test default
    sol1 = solve(rollout)
    @test sol1[:U] ≈ sol1.u
    @test sol1[:u] ≈ u.(sol1.t)

    # test solve kwargs
    sol2 = solve(rollout, dense=false, save_everystep=false, save_start=false, save_end=true)
    @test length(sol2[:U]) == 1
    @test size(sol2[:U][1]) == (sys.levels, sys.levels)
    
    # rename 
    rollout = UnitaryODEProblem(sys, u, 1.0, state_name=:X)
    sol = solve(rollout)
    @test sol[:X] ≈ sol.u
end

@testitem "Test density rollout symbolic interface" begin
    using OrdinaryDiffEq: solve

    csys = QuantumSystem([PAULIS.X, PAULIS.Y], 1.0, [1.0, 1.0])
    a = ComplexF64[0 1; 0 0]
    sys = OpenQuantumSystem(csys, dissipation_operators=[1e-3 * a])
    u = t -> [t; 0.0]

    ψ0 = ComplexF64[1, 0]
    ρ0 = ψ0 * ψ0'
    rollout = DensityODEProblem(sys, u, ρ0, 1.0)

    # test default symbolic access
    sol1 = solve(rollout)
    @test sol1[:ρ] ≈ sol1.u
    @test sol1[:u] ≈ u.(sol1.t)

    # test solve kwargs
    sol2 = solve(rollout, dense=false, save_everystep=false, save_start=false, save_end=true)
    @test length(sol2[:ρ]) == 1
    @test size(sol2[:ρ][1]) == (sys.levels, sys.levels)

    # rename
    rollout = DensityODEProblem(sys, u, ρ0, 1.0, state_name=:X)
    sol = solve(rollout)
    @test sol[:X] ≈ sol.u
end

@testitem "Rollout internal consistency (ket/unitary/density, closed system)" begin
    using OrdinaryDiffEq: solve

    sys  = QuantumSystem([PAULIS.X, PAULIS.Y], 1.0, [1.0, 1.0])
    osys = OpenQuantumSystem(sys)

    u = t -> [t; 0.0]
    T = 1.0
    ψ0 = ComplexF64[1, 0]
    ρ0 = ψ0 * ψ0'

    ket_prob = KetODEProblem(sys,  u, ψ0, T)
    U_prob = UnitaryODEProblem(sys, u, T)
    rho_prob = DensityODEProblem(osys, u, ρ0, T)

    # Save only final state so comparisons are well-defined
    kw = (dense=false, save_everystep=false, save_start=false, save_end=true)
    ket_sol = solve(ket_prob; kw...)
    U_sol = solve(U_prob; kw...)
    ρ_sol = solve(rho_prob; kw...)

    ψT = ket_sol.u[end]
    UT = U_sol.u[end]
    ρT = ρ_sol.u[end]

    @test ψT ≈ UT * ψ0
    @test ρT ≈ ψT * ψT' atol=1e-5
    @test ρT ≈ UT * ρ0 * UT' atol=1e-5
end

end