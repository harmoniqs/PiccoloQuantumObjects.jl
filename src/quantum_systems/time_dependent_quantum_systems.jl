export TimeDependentQuantumSystem

# ----------------------------------------------------------------------------- #
# TimeDependentQuantumSystem
# ----------------------------------------------------------------------------- #

# TODO: Open System

"""
    TimeDependentQuantumSystem <: AbstractQuantumSystem

A struct for storing time-dependent quantum dynamics and the appropriate gradients.

# Additional fields
- `H::Function`: The Hamiltonian function with time: a, t -> H(a, t).
- `G::Function`: The isomorphic generator function with time, a, t -> G(a, t).
- `∂G::Function`: The generator jacobian function with time, a, t -> ∂G(a, t).
- `n_drives::Int`: The number of drives in the system.
- `levels::Int`: The number of levels in the system.
- `params::Dict{Symbol, Any}`: A dictionary of parameters.

"""
struct TimeDependentQuantumSystem{F1<:Function, F2<:Function} <: AbstractQuantumSystem
    H::F1
    G::F2
    n_drives::Int
    levels::Int 
    params::Dict{Symbol, Any}

    function TimeDependentQuantumSystem end

    function TimeDependentQuantumSystem(
        H_drift::AbstractMatrix{<:Number},
        H_drives::Vector{<:AbstractMatrix{<:Number}};
        carriers::AbstractVector{<:Number}=zeros(length(H_drives)),
        phases::AbstractVector{<:Number}=zeros(length(H_drives)),
        params::Dict{Symbol, <:Any}=Dict{Symbol, Any}()
    )
        levels = size(H_drift, 1)
        H_drift = sparse(H_drift)
        G_drift = sparse(Isomorphisms.G(H_drift))

        n_drives = length(H_drives)
        H_drives = sparse.(H_drives)
        G_drives = sparse.(Isomorphisms.G.(H_drives))

        H = (a, t) -> H_drift + sum(a[i] * cos(carriers[i] * t + phases[i]) .* H_drives[i] for i in eachindex(H_drives))
        G = (a, t) -> G_drift + sum(a[i] * cos(carriers[i] * t + phases[i]) .* G_drives[i] for i in eachindex(G_drives))

        # save carries and phases
        params[:carriers] = carriers
        params[:phases] = phases

        return new{typeof(H), typeof(G)}(
            H,
            G,
            ∂G,
            n_drives,
            levels,
            params
        )
    end

    function TimeDependentQuantumSystem(H_drives::Vector{<:AbstractMatrix{ℂ}}; kwargs...) where ℂ <: Number
        @assert !isemtpy(H_drives) "At least one drive is required."
        return TimeDependentQuantumSystem(spzeros(ℂ, size(H_drives[1])), H_drives; kwargs...)
    end

    TimeDependentQuantumSystem(H_drift::AbstractMatrix{ℂ}; kwargs...) where ℂ <: Number = 
        TimeDependentQuantumSystem(H_drift, Matrix{ℂ}[]; kwargs...)

    function TimeDependentQuantumSystem(
        H::F,
        n_drives::Int;
        params::Dict{Symbol, <:Any}=Dict{Symbol, Any}()
    ) where F <: Function
        G = a, t -> Isomorphisms.G(sparse(H(a, t)))
        levels = size(H(zeroes(n_drives)), 1)
        return new{F, typeof(G)}(H, G, n_drives, levels, params)
    end
end

get_drift(sys::TimeDependentQuantumSystem) = t -> sys.H(zeros(sys.n_drives), t)

function get_drives(sys::TimeDependentQuantumSystem)
    H_drift = get_drift(sys)
    # Basis vectors for controls will extract drive operators
    return [t -> sys.H(I[1:sys.n_drives, i], t) - H_drift(t) for i ∈ 1:sys.n_drives]
end
