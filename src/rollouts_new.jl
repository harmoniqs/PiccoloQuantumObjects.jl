module NewRollouts

module PiccoloSymbolicInterfaces

export KetODEProblem
export UnitaryODEProblem

using .QuantumSystem: AbstractQuantumSystem

using LinearAlgebra
using SciMLBase
using SymbolicIndexingInterface
const SII = SymbolicIndexingInterface

"Scalar symbolic variable (e.g. x_1, u_2)"
struct PSym
    name::Symbol
end

"""
Vector symbolic variable (e.g. x, u) that expands to scalars.

SymbolicIndexingInterface can always reduce “vector symbol” to list of scalar symbols, so downstream SciML code that expects scalars still works.
"""
struct PVecSym
    name::Symbol
    n::Int
end

# Tell SII these are symbolic and have names
SII.symbolic_type(::Type{PSym})    = SII.ScalarSymbolic()
SII.symbolic_type(::Type{PVecSym}) = SII.ArraySymbolic()

SII.hasname(::PSym)    = true
SII.hasname(::PVecSym) = true

SII.getname(s::PSym)    = s.name
SII.getname(s::PVecSym) = s.name

# How vector symbols expand into scalar symbols
Base.collect(s::PVecSym) = [PSym(Symbol(s.name, :_, i)) for i in 1:s.n]

# Convenience so users can write x[1] in the DSL
Base.getindex(s::PVecSym, i::Int) = collect(s)[i]
Base.length(s::PVecSym) = s.n

# ============================================================ #
# Define DSL for Piccolo
# ============================================================ #

const SymbolIndex = Union{Int,AbstractVector{Int}}

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
    x = PVecSym(state_name, n_state)
    control_name, n_control = control
    u = PVecSym(control_name, n_control)

    # Create scalar coordinates and vector coordinate
    state_index = Dict{Symbol, SymbolIndex}()
    for (i, s) in enumerate(collect(x))
        state_index[SII.getname(s)] = i
    end
    state_index[state_name] = 1:n_state

    control_index = Dict{Symbol, SymbolIndex}()
    for (i, s) in enumerate(collect(u))
        control_index[SII.getname(s)] = i
    end
    control_index[control_name] = 1:n_control

    return PiccoloRolloutSystem(state_index, control_index, timestep_name, defaults)
end

struct PiccoloRolloutControl{F}
    u::F
end

# TODO: 
# - UnitaryODEProblem
# - KetODEProblem

"""
Implement a quantum system rollout.
"""
function SciMLBase.ODEProblem(
    sys::AbstractQuantumSystem, u, x0, T; 
    state_name=:x, 
    control_name=:u, 
    kwargs...
)
    p = PiccoloRolloutControl(u)
	rhs!(dx, x, p, t) = mul!(dx, sys.H(p.u(t), t), x, -im, 1.0)
    sii_sys = PiccoloRolloutSystem(state_name => sys.levels, control_name => sys.n_drives)
	return ODEProblem(ODEFunction(rhs!; sys = sii_sys), x0, (0, T), p; kwargs...)
end

function KetODEProblem(
    sys::AbstractQuantumSystem, u, ψ0, T; 
    state_name=:ψ, 
    kwargs...
)
	return ODEProblem(sys, u, ψ0, T, state_name=state_name, kwargs...)
end

# TODO: This doesn't work yet
function UnitaryODEProblem(
    sys::AbstractQuantumSystem, u, T; 
    state_name=:U, 
    kwargs...
)
    U0 = Matrix{ComplexF64}(I, sys.levels, sys.levels)
	return ODEProblem(sys, u, U0, T; state_name=state_name, kwargs...)
end

# ============================================================ #
# Interface 

# https://docs.sciml.ai/SymbolicIndexingInterface/
# ============================================================ #

_name(sym::Symbol) = sym
_name(sym::PSym) = sym.name          
_name(sym::PVecSym) = sym.name       
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


end


end