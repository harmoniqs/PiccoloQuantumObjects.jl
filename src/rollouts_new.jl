module PiccoloSymbolicInterfaces

export KetODEProblem
export UnitaryODEProblem

using .QuantumSystem: AbstractQuantumSystem

using LinearAlgebra
using SciMLBase
using SymbolicIndexingInterface
const SII = SymbolicIndexingInterface

# ============================================================ #
# DSL for Piccolo
# ============================================================ #

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
    sys::AbstractQuantumSystem, u, ψ0, T; 
    state_name=:ψ,
    control_name=:u,
    kwargs...
)
    # TODO: In-place H buffer?
	rhs!(dx, x, p, t) = mul!(dx, sys.H(p.u(t), t), x, -im, 1.0)

    p = PiccoloRolloutControl(u)
    sii_sys = PiccoloRolloutSystem(state_name => sys.levels, control_name => sys.n_drives)
	return ODEProblem(ODEFunction(rhs!; sys = sii_sys), ψ0, (0, T), p; kwargs...)
end

function UnitaryODEProblem(
    sys::AbstractQuantumSystem, u, T; 
    state_name=:U, 
    control_name=:u,
    kwargs...
)
	rhs!(dx, x, p, t) = mul!(dx, sys.H(p.u(t), t), x, -im, 1.0)

    p = PiccoloRolloutControl(u)
    U0 = Matrix{ComplexF64}(I, sys.levels, sys.levels)
    sii_sys = PiccoloRolloutSystem(state_name => (sys.levels, sys.levels), control_name => sys.n_drives)
	return ODEProblem(ODEFunction(rhs!; sys = sii_sys), U0, (0, T), p; kwargs...)
end

# ============================================================ #
# Interface 

# https://docs.sciml.ai/SymbolicIndexingInterface/
# ============================================================ #

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


end