module Rollouts

export free_phase

export fidelity
export unitary_fidelity
export unitary_free_phase_fidelity

export rollout
export variational_rollout
export open_rollout
export unitary_rollout
export variational_unitary_rollout
export lab_frame_unitary_rollout
export lab_frame_unitary_rollout_trajectory

export rollout_fidelity
export unitary_rollout_fidelity
export open_rollout_fidelity

using ..QuantumSystems
using ..EmbeddedOperators
using ..Isomorphisms
using ..DirectSums

using NamedTrajectories

using ExponentialAction
using ForwardDiff
using LinearAlgebra
using ProgressMeter
using TestItems


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

"""
    free_phase(phases::AbstractVector{<:Real}, phase_operators::AbstractVector{<:AbstractMatrix{<:ℂ}})

Rotate the `phase_operators` by the `phases` and return the Kronecker product.
"""
function free_phase(
    phases::AbstractVector{<:Real},
    phase_operators::AbstractVector{<:AbstractMatrix{<:ℂ}}
) where ℂ <: Number
    # NOTE: switch to expv for ForwardDiff
    # return reduce(kron, [exp(im * ϕ * H) for (ϕ, H) ∈ zip(phases, phase_operators)])
    Id = Matrix{ℂ}(I, size(phase_operators[1]))
    return reduce(kron, [expv(im * ϕ, H, Id) for (ϕ, H) ∈ zip(phases, phase_operators)])
end

"""
    unitary_free_phase_fidelity(
        U::AbstractMatrix,
        U_goal::AbstractMatrix,
        phases::AbstractVector{<:Real},
        phase_operators::AbstractVector{<:AbstractMatrix};
        subspace::AbstractVector{Int}=axes(U, 1)
    )

Calculate the fidelity between unitary operators `U` and `U_goal` in the `subspace`,
including the `phase` rotations about the `phase_operators`.
"""
function unitary_free_phase_fidelity(
    U::AbstractMatrix,
    U_goal::AbstractMatrix,
    phases::AbstractVector{<:Real},
    phase_operators::AbstractVector{<:AbstractMatrix};
    subspace::AbstractVector{Int}=axes(U, 1)
)
    R = free_phase(phases, phase_operators)
    return unitary_fidelity(R * U, U_goal; subspace=subspace)
end


# ----------------------------------------------------------------------------- #
# Utilities
# ----------------------------------------------------------------------------- #

"""
    infer_is_evp(integrator::Function)

Infer whether the integrator is a exponential-vector product (EVP) function.

If `true`, the integrator is expected to have a signature like the exponential action,
`expv`. Otherwise, it is expected to have a signature like `exp`.
"""
function infer_is_evp(integrator::Function)
    # name + args
    ns = fieldcount.([m.sig for m ∈ methods(integrator)])
    is_exp = 2 ∈ ns
    is_expv = 4 ∈ ns
    if is_exp && is_expv
        throw(ErrorException("Ambiguous rollout integrator signature. Please specify manually."))
    elseif is_exp
        return false
    elseif is_expv
        return true
    else
        throw(ErrorException("No valid rollout integrator signature found."))
    end
end

# ----------------------------------------------------------------------------- #
# Quantum state rollouts
# ----------------------------------------------------------------------------- #

@doc raw"""
    rollout(
        ψ̃_init::AbstractVector{<:Real},
        controls::AbstractMatrix{<:Real},
        Δt::AbstractVector,
        system::AbstractQuantumSystem
    )
    rollout(
        ψ_init::AbstractVector{<:Complex},
        controls::AbstractMatrix{<:Real},
        Δt::AbstractVector,
        system::AbstractQuantumSystem
    )
    rollout(
        inits::AbstractVector{<:AbstractVector},
        controls::AbstractMatrix{<:Real},
        Δt::AbstractVector,
        system::AbstractQuantumSystem
    )

Rollout a quantum state `ψ̃_init` under the control `controls` for a time `Δt`
using the system `system`.

If `exp_vector_product` is `true`, the integrator is expected to have a signature like
the exponential action, `expv`. Otherwise, it is expected to have a signature like `exp`.

Types should allow for autodifferentiable controls and times.
"""
function rollout end

function rollout(
    ψ̃_init::AbstractVector{<:Real},
    controls::AbstractMatrix{<:Real},
    Δt::AbstractVector,
    system::AbstractQuantumSystem;
    show_progress=false,
    integrator=expv,
    exp_vector_product=infer_is_evp(integrator),
)
    T = size(controls, 2)

    # Enable ForwardDiff
    R = Base.promote_eltype(ψ̃_init, controls, Δt)
    Ψ̃ = zeros(R, length(ψ̃_init), T)

    Ψ̃[:, 1] .= ψ̃_init

    p = Progress(T-1; enabled=show_progress)
    for t = 2:T
        aₜ₋₁ = controls[:, t - 1]
        Gₜ = system.G(aₜ₋₁)
        if exp_vector_product
            Ψ̃[:, t] .= integrator(Δt[t - 1], Gₜ, Ψ̃[:, t - 1])
        else
            Ψ̃[:, t] .= integrator(Matrix(Gₜ) * Δt[t - 1]) * Ψ̃[:, t - 1]
        end
        next!(p)
    end

    return Ψ̃
end

rollout(ψ::Vector{<:Complex}, args...; kwargs...) =
    rollout(ket_to_iso(ψ), args...; kwargs...)

function rollout(
    inits::AbstractVector{<:AbstractVector}, args...; kwargs...
)
    return [rollout(state, args...; kwargs...) for state ∈ inits]
end

function rollout(
    traj::NamedTrajectory,
    system::AbstractQuantumSystem;
    state_name::Symbol=:ψ̃,
    drive_name::Symbol=:a,
    kwargs...
)   
    # Get the initial state names
    state_names = [
        name for name ∈ traj.names if startswith(string(name), string(state_name))
    ]

    return rollout(
        length(state_names) == 1 ? traj.initial[state_name] : [traj.initial[name] for name ∈ state_names],
        traj[drive_name],
        get_timesteps(traj),
        system;
        kwargs...
    )
end

"""
    rollout_fidelity(
        ψ̃_init::AbstractVector{<:Real},
        ψ̃_goal::AbstractVector{<:Real},
        controls::AbstractMatrix{<:Real},
        Δt::AbstractVector,
        system::AbstractQuantumSystem
    )
    rollout_fidelity(
        ψ_init::AbstractVector{<:Complex},
        ψ_goal::AbstractVector{<:Complex},
        controls::AbstractMatrix{<:Real},
        Δt::AbstractVector,
        system::AbstractQuantumSystem
    )
    rollout_fidelity(
        trajectory::NamedTrajectory,
        system::AbstractQuantumSystem
    )

Calculate the fidelity between the final state of a rollout and a goal state.
"""
function rollout_fidelity end

function rollout_fidelity(
    ψ̃_init::AbstractVector{<:Real},
    ψ̃_goal::AbstractVector{<:Real},
    controls::AbstractMatrix{<:Real},
    Δt::AbstractVector,
    system::AbstractQuantumSystem;
    kwargs...
)
    Ψ̃ = rollout(ψ̃_init, controls, Δt, system; kwargs...)
    ψ_final = iso_to_ket(Ψ̃[:, end])
    ψ_goal = iso_to_ket(ψ̃_goal)
    return fidelity(ψ_final, ψ_goal)
end

function rollout_fidelity(
    ψ_init::AbstractVector{<:Complex},
    ψ_goal::AbstractVector{<:Complex},
    controls::AbstractMatrix{<:Real},
    Δt::AbstractVector,
    system::AbstractQuantumSystem;
    kwargs...
)
    return rollout_fidelity(ket_to_iso(ψ_init), ket_to_iso(ψ_goal), controls, Δt, system; kwargs...)
end

function rollout_fidelity(
    trajectory::NamedTrajectory,
    system::AbstractQuantumSystem;
    state_name::Symbol=:ψ̃,
    control_name=:a,
    kwargs...
)
    fids = []
    for name ∈ trajectory.names
        if startswith(string(name), string(state_name))
            controls = trajectory[control_name]
            init = trajectory.initial[name]
            goal = trajectory.goal[name]
            fid = rollout_fidelity(init, goal, controls, get_timesteps(trajectory), system; kwargs...)
            push!(fids, fid)
        end
    end
    return length(fids) == 1 ? fids[1] : fids
end

# ----------------------------------------------------------------------------- #
# Open quantum system rollouts
# ----------------------------------------------------------------------------- #

"""
    open_rollout(
        ρ⃗₁::AbstractVector{<:Complex},
        controls::AbstractMatrix{<:Real},
        Δt::AbstractVector,
        system::OpenQuantumSystem;
        kwargs...
    )

Rollout a quantum state `ρ⃗₁` under the control `controls` for a time `Δt`

# Arguments
- `ρ⃗₁::AbstractVector{<:Complex}`: Initial state vector
- `controls::AbstractMatrix{<:Real}`: Control matrix
- `Δt::AbstractVector`: Time steps
- `system::OpenQuantumSystem`: Quantum system

# Keyword Arguments
- `show_progress::Bool=false`: Show progress bar
- `integrator::Function=expv`: Integrator function
- `exp_vector_product::Bool`: Infer whether the integrator is an exponential-vector product

"""
function open_rollout end

function open_rollout(
    ρ⃗̃_init::AbstractVector{<:Real},
    controls::AbstractMatrix{<:Real},
    Δt::AbstractVector,
    system::OpenQuantumSystem;
    show_progress=false,
    integrator=expv,
    exp_vector_product=infer_is_evp(integrator),
)
    T = size(controls, 2)

    # Enable ForwardDiff
    R = Base.promote_eltype(ρ⃗̃_init, controls, Δt)
    ρ⃗̃ = zeros(R, length(ρ⃗̃_init), T)

    ρ⃗̃[:, 1] = ρ⃗̃_init

    p = Progress(T-1; enabled=show_progress)
    for t = 2:T
        aₜ₋₁ = controls[:, t - 1]
        𝒢ₜ = system.𝒢(aₜ₋₁)
        if exp_vector_product
            ρ⃗̃[:, t] = integrator(Δt[t - 1], 𝒢ₜ, ρ⃗̃[:, t - 1])
        else
            ρ⃗̃[:, t] = integrator(Δt[t - 1], 𝒢ₜ) * ρ⃗̃[:, t - 1]
        end
        next!(p)
    end

    return ρ⃗̃
end

"""
    open_rollout(
        ρ₁::AbstractMatrix{<:Complex},
        controls::AbstractMatrix{<:Real},
        Δt::AbstractVector,
        system::OpenQuantumSystem;
        kwargs...
    )

Rollout a density matrix `ρ₁` under the control `controls` and timesteps `Δt`

"""
function open_rollout(
    ρ_init::AbstractMatrix{<:Complex},
    controls::AbstractMatrix{<:Real},
    Δt::AbstractVector,
    system::OpenQuantumSystem;
    kwargs...
)
    return open_rollout(density_to_iso_vec(ρ_init), controls, Δt, system; kwargs...)
end

"""
    open_rollout_fidelity(
        ρ⃗₁::AbstractVector{<:Complex},
        ρ⃗₂::AbstractVector{<:Complex},
        controls::AbstractMatrix{<:Real},
        Δt::AbstractVector,
        system::OpenQuantumSystem
    )
    open_rollout_fidelity(
        ρ₁::AbstractMatrix{<:Complex},
        ρ₂::AbstractMatrix{<:Complex},
        controls::AbstractMatrix{<:Real},
        Δt::AbstractVector,
        system::OpenQuantumSystem
    )
    open_rollout_fidelity(
        traj::NamedTrajectory,
        system::OpenQuantumSystem;
        state_name::Symbol=:ρ⃗̃,
        control_name::Symbol=:a,
        kwargs...
    )

Calculate the fidelity between the final state of an open quantum system rollout and a goal state.

"""
function open_rollout_fidelity end

function open_rollout_fidelity(
    ρ_init::AbstractMatrix{<:Complex},
    ρ_goal::AbstractMatrix{<:Complex},
    controls::AbstractMatrix{<:Real},
    Δt::AbstractVector,
    system::AbstractQuantumSystem;
    kwargs...
)

    ρ⃗̃_traj = open_rollout(ρ_init, controls, Δt, system; kwargs...)
    ρ_final = iso_vec_to_density(ρ⃗̃_traj[:, end])
    return fidelity(ρ_final, ρ_goal)
end

function open_rollout_fidelity(
    traj::NamedTrajectory,
    system::OpenQuantumSystem;
    state_name::Symbol=:ρ⃗̃,
    control_name::Symbol=:a,
    kwargs...
)
    ρ_goal = iso_vec_to_density(traj.goal[state_name])
    ρ_init = iso_vec_to_density(traj.initial[state_name])
    controls = traj[control_name]
    Δt = get_timesteps(traj)
    return open_rollout_fidelity(ρ_init, ρ_goal, controls, Δt, system; kwargs...)
end


# ----------------------------------------------------------------------------- #
# Unitary rollouts
# ----------------------------------------------------------------------------- #

"""
    unitary_rollout(
        Ũ⃗_init::AbstractVector{<:Real},
        controls::AbstractMatrix{<:Real},
        Δt::AbstractVector,
        system::AbstractQuantumSystem;
        kwargs...
    )

Rollout a isomorphic unitary operator `Ũ⃗_init` under the control `controls` for a time `Δt`
using the system `system`.

# Arguments
- `Ũ⃗_init::AbstractVector{<:Real}`: Initial unitary vector
- `controls::AbstractMatrix{<:Real}`: Control matrix
- `Δt::AbstractVector`: Time steps
- `system::AbstractQuantumSystem`: Quantum system

# Keyword Arguments
- `show_progress::Bool=false`: Show progress bar
- `integrator::Function=expv`: Integrator function
- `exp_vector_product::Bool`: Infer whether the integrator is an exponential-vector product

"""
function unitary_rollout end

function unitary_rollout(
    Ũ⃗_init::AbstractVector{<:Real},
    controls::AbstractMatrix{<:Real},
    Δt::AbstractVector{<:Real},
    system::AbstractQuantumSystem;
    show_progress=false,
    integrator=expv,
    exp_vector_product=infer_is_evp(integrator),
)
    T = size(controls, 2)

    # Enable ForwardDiff
    R = Base.promote_eltype(Ũ⃗_init, controls, Δt)
    Ũ⃗ = zeros(R, length(Ũ⃗_init), T)

    Ũ⃗[:, 1] .= Ũ⃗_init

    p = Progress(T-1; enabled=show_progress)
    for t = 2:T
        aₜ₋₁ = controls[:, t - 1]
        Gₜ = system.G(aₜ₋₁)
        Ũₜ₋₁ = iso_vec_to_iso_operator(Ũ⃗[:, t - 1])
        if exp_vector_product
            Ũₜ = integrator(Δt[t - 1], Gₜ, Ũₜ₋₁)
        else
            Ũₜ = integrator(Matrix(Gₜ) * Δt[t - 1]) * Ũₜ₋₁
        end
        Ũ⃗[:, t] .= iso_operator_to_iso_vec(Ũₜ)
        next!(p)
    end

    return Ũ⃗
end

function unitary_rollout(
    controls::AbstractMatrix{<:Real},
    Δt::AbstractVector,
    system::AbstractQuantumSystem;
    kwargs...
)
    Ĩ⃗ = operator_to_iso_vec(Matrix{ComplexF64}(I(system.levels)))
    return unitary_rollout(Ĩ⃗, controls, Δt, system; kwargs...)
end

function unitary_rollout(
    traj::NamedTrajectory,
    system::AbstractQuantumSystem;
    unitary_name::Symbol=:Ũ⃗,
    drive_name::Symbol=:a,
    kwargs...
)
    return unitary_rollout(
        traj.initial[unitary_name],
        traj[drive_name],
        get_timesteps(traj),
        system;
        kwargs...
    )
end

"""
    unitary_rollout_fidelity(
        Ũ⃗_init::AbstractVector{<:Real},
        Ũ⃗_goal::AbstractVector{<:Real},
        controls::AbstractMatrix{<:Real},
        Δt::AbstractVector,
        system::AbstractQuantumSystem;
        kwargs...
    )
    unitary_rollout_fidelity(
        Ũ⃗_goal::AbstractVector{<:Real},
        controls::AbstractMatrix{<:Real},
        Δt::AbstractVector,
        system::AbstractQuantumSystem;
        kwargs...
    )
    unitary_rollout_fidelity(
        U_init::AbstractMatrix{<:Complex},
        U_goal::AbstractMatrix{<:Complex},
        controls::AbstractMatrix{<:Real},
        Δt::AbstractVector,
        system::AbstractQuantumSystem;
        kwargs...
    )
    unitary_rollout_fidelity(
        U_goal::AbstractMatrix{<:Complex},
        controls::AbstractMatrix{<:Real},
        Δt::AbstractVector,
        system::AbstractQuantumSystem;
        kwargs...
    )
    unitary_rollout_fidelity(
        U_goal::EmbeddedOperator,
        controls::AbstractMatrix{<:Real},
        Δt::AbstractVector,
        system::AbstractQuantumSystem;
        subspace::AbstractVector{Int}=U_goal.subspace,
        kwargs...
    )
    unitary_rollout_fidelity(
        traj::NamedTrajectory,
        sys::AbstractQuantumSystem;
        kwargs...
    )

Calculate the fidelity between the final state of a unitary rollout and a goal state. 
If the initial unitary is not provided, the identity operator is assumed.
If `phases` and `phase_operators` are provided, the free phase unitary fidelity is calculated.

"""
function unitary_rollout_fidelity end

function unitary_rollout_fidelity(
    Ũ⃗_init::AbstractVector{<:Real},
    Ũ⃗_goal::AbstractVector{<:Real},
    controls::AbstractMatrix{<:Real},
    Δt::AbstractVector,
    system::AbstractQuantumSystem;
    subspace::AbstractVector{Int}=axes(iso_vec_to_operator(Ũ⃗_goal), 1),
    phases::Union{Nothing, AbstractVector{<:Real}}=nothing,
    phase_operators::Union{Nothing, AbstractVector{<:AbstractMatrix{<:Complex}}}=nothing,
    kwargs...
)
    Ũ⃗_T = unitary_rollout(Ũ⃗_init, controls, Δt, system; kwargs...)[:, end]
    U_T = iso_vec_to_operator(Ũ⃗_T)
    U_goal = iso_vec_to_operator(Ũ⃗_goal)
    if !isnothing(phases)
        return unitary_free_phase_fidelity(U_T, U_goal, phases, phase_operators; subspace=subspace)
    else
        return unitary_fidelity(U_T, U_goal; subspace=subspace)
    end
end

function unitary_rollout_fidelity(
    Ũ⃗_goal::AbstractVector{<:Real},
    controls::AbstractMatrix{<:Real},
    Δt::AbstractVector,
    system::AbstractQuantumSystem;
    kwargs...
)
    Ĩ⃗ = operator_to_iso_vec(Matrix{ComplexF64}(I(system.levels)))
    return unitary_rollout_fidelity(Ĩ⃗, Ũ⃗_goal, controls, Δt, system; kwargs...)
end

function unitary_rollout_fidelity(
    U_init::AbstractMatrix{<:Complex},
    U_goal::AbstractMatrix{<:Complex},
    controls::AbstractMatrix{<:Real},
    Δt::AbstractVector,
    system::AbstractQuantumSystem;
    kwargs...
)
    Ũ⃗_init = operator_to_iso_vec(U_init)
    Ũ⃗_goal = operator_to_iso_vec(U_goal)
    return unitary_rollout_fidelity(Ũ⃗_init, Ũ⃗_goal, controls, Δt, system; kwargs...)
end

unitary_rollout_fidelity(
    U_goal::AbstractMatrix{<:Complex},
    controls::AbstractMatrix{<:Real},
    Δt::AbstractVector,
    system::AbstractQuantumSystem;
    kwargs...
) = unitary_rollout_fidelity(operator_to_iso_vec(U_goal), controls, Δt, system; kwargs...)

unitary_rollout_fidelity(
    U_goal::EmbeddedOperator,
    controls::AbstractMatrix{<:Real},
    Δt::AbstractVector,
    system::AbstractQuantumSystem;
    subspace::AbstractVector{Int}=U_goal.subspace,
    kwargs...
) = unitary_rollout_fidelity(U_goal.operator, controls, Δt, system; subspace=subspace, kwargs...)

function unitary_rollout_fidelity(
    traj::NamedTrajectory,
    sys::AbstractQuantumSystem;
    unitary_name::Symbol=:Ũ⃗,
    drive_name::Symbol=:a,
    kwargs...
)
    Ũ⃗_init = traj.initial[unitary_name]
    Ũ⃗_goal = traj.goal[unitary_name]
    controls = traj[drive_name]
    Δt = get_timesteps(traj)
    return unitary_rollout_fidelity(Ũ⃗_init, Ũ⃗_goal, controls, Δt, sys; kwargs...)
end


# ----------------------------------------------------------------------------- #
# Variational rollouts
# ----------------------------------------------------------------------------- #

"""
    variational_rollout(
        ψ̃_init::AbstractVector{<:Real},
        controls::AbstractMatrix{<:Real},
        Δt::AbstractVector{<:Real},
        system::VariationalQuantumSystem;
        show_progress::Bool=false,
        integrator::Function=expv,
        exp_vector_product::Bool=infer_is_evp(integrator)
    )
    variational_rollout(ψ::Vector{<:Complex}, args...; kwargs...)
    variational_rollout(inits::AbstractVector{<:AbstractVector}, args...; kwargs...)
    variational_rollout(
        traj::NamedTrajectory, 
        system::AbstractQuantumSystem; 
        state_name::Symbol=:ψ̃,
        drive_name::Symbol=:a,
        kwargs...
    )   

Simulates the variational evolution of a quantum state under a given control trajectory.

# Returns
- `Ψ̃::Matrix{<:Real}`: The evolved quantum state at each timestep.
- `Ψ̃_vars::Vector{<:Matrix{<:Real}}`: The variational derivatives of the 
    quantum state with respect to the variational parameters.

# Notes
This function computes the variational evolution of a quantum state using the 
variational generators of the system. It supports autodifferentiable controls and 
timesteps, making it suitable for optimization tasks. The variational derivatives are 
computed alongside the state evolution, enabling sensitivity analysis and gradient-based 
optimization.
"""
function variational_rollout end

function variational_rollout(
    ψ̃_init::AbstractVector{<:Real},
    controls::AbstractMatrix{<:Real},
    Δt::AbstractVector{<:Real},
    system::VariationalQuantumSystem;
    show_progress=false,
    integrator=expv,
    exp_vector_product=infer_is_evp(integrator),
)
    V = length(system.G_vars)
    N = length(ψ̃_init)
    T = size(controls, 2)

    # Enable ForwardDiff
    R = Base.promote_eltype(ψ̃_init, controls, Δt)
    Ψ̃ = zeros(R, N, T)
    Ψ̃_vars = [zeros(R, N, T) for _ = 1:V]

    # Variational generator
    Ĝ = a -> Isomorphisms.var_G(system.G(a), [G(a) for G in system.G_vars])

    Ψ̃[:, 1] .= ψ̃_init
    Ṽₜ₋₁ = [ψ̃_init; zeros(R, N * V)]

    p = Progress(T-1; enabled=show_progress)
    for t = 2:T
        aₜ₋₁ = controls[:, t - 1]
        Ĝₜ₋₁ = Ĝ(aₜ₋₁)
        if exp_vector_product
            Ṽₜ = integrator(Δt[t - 1], Ĝₜ₋₁, Ṽₜ₋₁)
        else
            Ṽₜ = integrator(Matrix(Ĝₜ₋₁) * Δt[t - 1]) * Ṽₜ₋₁
        end
        Ψ̃[:, t] .= Ṽₜ[1:N]
        for i = 1:V
            Ψ̃_vars[i][:, t] .= Ṽₜ[1 + i * N:(i + 1) * N]
        end
        Ṽₜ₋₁ = Ṽₜ
        next!(p)
    end

    return Ψ̃, Ψ̃_vars
end

variational_rollout(ψ::Vector{<:Complex}, args...; kwargs...) =
    variational_rollout(ket_to_iso(ψ), args...; kwargs...)

function variational_rollout(
    inits::AbstractVector{<:AbstractVector}, args...; kwargs...
)
    N = length(inits)

    # First call
    ψ̃1, ψ̃_vars1 = variational_rollout(inits[1], args...; kwargs...)

    # Preallocate the rest
    ψ̃s = Vector{typeof(ψ̃1)}(undef, N)
    ψ̃_vars = Vector{typeof(ψ̃_vars1)}(undef, N)
    ψ̃s[1] = ψ̃1
    ψ̃_vars[1] = ψ̃_vars1
    for i = 2:N
        ψ̃s[i], ψ̃_vars[i] = variational_rollout(inits[i], args...; kwargs...)
    end
    return ψ̃s, ψ̃_vars
end


function variational_rollout(
    traj::NamedTrajectory,
    system::AbstractQuantumSystem;
    state_name::Symbol=:ψ̃,
    drive_name::Symbol=:a,
    kwargs...
)   
    # Get the initial state names
    state_names = [
        name for name ∈ traj.names if startswith(string(name), string(state_name))
    ]

    return variational_rollout(
        length(state_names) == 1 ? traj.initial[state_name] : [traj.initial[name] for name ∈ state_names],
        traj[drive_name],
        get_timesteps(traj),
        system;
        kwargs...
    )
end


"""
    variational_unitary_rollout(
        Ũ⃗_init::AbstractVector{<:Real},
        controls::AbstractMatrix{<:Real},
        Δt::AbstractVector{<:Real},
        system::VariationalQuantumSystem;
        show_progress::Bool=false,
        integrator::Function=expv,
        exp_vector_product::Bool=infer_is_evp(integrator)
    )
    variational_unitary_rollout(
        controls::AbstractMatrix{<:Real},
        Δt::AbstractVector,
        system::VariationalQuantumSystem;
        kwargs...
    )
    variational_unitary_rollout(
        traj::NamedTrajectory,
        system::VariationalQuantumSystem;
        unitary_name::Symbol=:Ũ⃗,
        drive_name::Symbol=:a,
        kwargs...
    )

Simulates the variational evolution of a quantum state under a given control trajectory.

# Returns
- `Ũ⃗::Matrix{<:Real}`: The evolved unitary at each timestep.
- `Ũ⃗_vars::Vector{<:Matrix{<:Real}}`: The variational derivatives of the  unitary with 
    respect to the variational parameters.

# Notes
This function computes the variational evolution of a unitary using the 
variational generators of the system. It supports autodifferentiable controls and 
timesteps, making it suitable for optimization tasks. The variational derivatives are 
computed alongside the state evolution, enabling sensitivity analysis and gradient-based 
optimization.
"""
function variational_unitary_rollout end

function variational_unitary_rollout(
    Ũ⃗_init::AbstractVector{<:Real},
    controls::AbstractMatrix{<:Real},
    Δt::AbstractVector{<:Real},
    system::VariationalQuantumSystem;
    show_progress=false,
    integrator=expv,
    exp_vector_product=infer_is_evp(integrator),
)
    V = length(system.G_vars)
    N = length(Ũ⃗_init)
    T = size(controls, 2)

    # Enable ForwardDiff
    R = Base.promote_eltype(Ũ⃗_init, controls, Δt)
    Ũ⃗ = zeros(R, N, T)
    Ũ⃗_vars = [zeros(R, N, T) for _ = 1:V]

    # Variational generator
    Ĝ = a -> Isomorphisms.var_G(
        kron(I(system.levels), system.G(a)),
        [kron(I(system.levels), G(a)) for G in system.G_vars]
    )

    Ũ⃗[:, 1] .= Ũ⃗_init
    Ṽ⃗ₜ₋₁ = [Ũ⃗_init; zeros(R, N * V)]

    p = Progress(T - 1; enabled=show_progress)
    for t = 2:T
        aₜ₋₁ = controls[:, t - 1]
        Ĝₜ₋₁ = Ĝ(aₜ₋₁)
        if exp_vector_product
            Ṽ⃗ₜ = integrator(Δt[t - 1], Ĝₜ₋₁, Ṽ⃗ₜ₋₁)
        else
            Ṽ⃗ₜ = integrator(Matrix(Ĝₜ₋₁) * Δt[t - 1]) * Ṽ⃗ₜ₋₁
        end
        Ũ⃗[:, t] .= Ṽ⃗ₜ[1:N]
        for i = 1:V
            Ũ⃗_vars[i][:, t] .= Ṽ⃗ₜ[1 + i * N:(i + 1) * N]
        end
        Ṽ⃗ₜ₋₁ = Ṽ⃗ₜ
        next!(p)
    end

    return Ũ⃗, Ũ⃗_vars
end

function variational_unitary_rollout(
    controls::AbstractMatrix{<:Real},
    Δt::AbstractVector,
    system::VariationalQuantumSystem;
    kwargs...
)
    Ĩ⃗ = operator_to_iso_vec(Matrix{ComplexF64}(I(system.levels)))
    return variational_unitary_rollout(Ĩ⃗, controls, Δt, system; kwargs...)
end

function variational_unitary_rollout(
    traj::NamedTrajectory,
    system::VariationalQuantumSystem;
    unitary_name::Symbol=:Ũ⃗,
    drive_name::Symbol=:a,
    kwargs...
)
    return variational_unitary_rollout(
        traj.initial[unitary_name],
        traj[drive_name],
        get_timesteps(traj),
        system;
        kwargs...
    )
end


# ----------------------------------------------------------------------------- #
# Experimental rollouts
# ----------------------------------------------------------------------------- #


# *************************************************************************** #

@testitem "Test rollouts using fidelities" begin
    using ExponentialAction

    include("../test/test_utils.jl")

    traj = named_trajectory_type_1()
    sys = QuantumSystem([PAULIS.X, PAULIS.Y])
    U_goal = GATES.H
    embedded_U_goal = EmbeddedOperator(U_goal, sys)

    ψ = ComplexF64[1, 0]
    ψ_goal = U_goal * ψ
    ψ̃ = ket_to_iso(ψ)
    ψ̃_goal = ket_to_iso(ψ_goal)

    as = traj.a
    Δts = get_timesteps(traj)

    # Default integrator
    # State fidelity
    @test rollout_fidelity(ψ, ψ_goal, as, Δts, sys) > 0

    # Unitary fidelity
    @test unitary_rollout_fidelity(U_goal, as, Δts, sys) > 0
    @test unitary_rollout_fidelity(traj, sys) > 0
    @test unitary_rollout_fidelity(embedded_U_goal, as, Δts, sys) > 0

    # Free phase unitary
    @test unitary_rollout_fidelity(traj, sys;
        phases=[0.0], phase_operators=Matrix{ComplexF64}[PAULIS[:Z]]
    ) > 0

    # Free phase unitary
    @test unitary_rollout_fidelity(traj, sys;
        phases=[0.0],
        phase_operators=[PAULIS[:Z]]
    ) > 0

    # Expv explicit
    # State fidelity
    @test rollout_fidelity(ψ, ψ_goal, as, Δts, sys, integrator=expv) > 0

    # Unitary fidelity
    @test unitary_rollout_fidelity(U_goal, as, Δts, sys, integrator=expv) > 0
    @test unitary_rollout_fidelity(traj, sys, integrator=expv) > 0
    @test unitary_rollout_fidelity(embedded_U_goal, as, Δts, sys, integrator=expv) > 0

    # Exp explicit
    # State fidelity
    @test rollout_fidelity(ψ, ψ_goal, as, Δts, sys, integrator=exp) > 0

    # Unitary fidelity
    @test unitary_rollout_fidelity(U_goal, as, Δts, sys, integrator=exp) > 0
    @test unitary_rollout_fidelity(traj, sys, integrator=exp) > 0
    @test unitary_rollout_fidelity(embedded_U_goal, as, Δts, sys, integrator=exp) > 0

    # Bad integrator
    @test_throws ErrorException unitary_rollout_fidelity(U_goal, as, Δts, sys, integrator=(a,b) -> 1) > 0
end

@testitem "Foward diff rollout" begin
    using ForwardDiff
    using ExponentialAction

    sys = QuantumSystem([PAULIS.X, PAULIS.Y])
    T = 51
    Δt = 0.2
    ts = fill(Δt, T)
    as = collect([π/(T-1)/Δt * sin.(π*(0:T-1)/(T-1)).^2 zeros(T)]')

    # Control derivatives
    ψ = ComplexF64[1, 0]
    result1 = ForwardDiff.jacobian(
        as -> rollout(ψ, as, ts, sys, integrator=expv)[:, end], as
    )
    iso_ket_dim = length(ket_to_iso(ψ))
    @test size(result1) == (iso_ket_dim, T * sys.n_drives)

    result2 = ForwardDiff.jacobian(
        as -> unitary_rollout(as, ts, sys, integrator=expv)[:, end], as
    )
    iso_vec_dim = length(operator_to_iso_vec(sys.H(zeros(sys.n_drives))))
    @test size(result2) == (iso_vec_dim, T * sys.n_drives)

    # Time derivatives
    ψ = ComplexF64[1, 0]
    result1 = ForwardDiff.jacobian(
        ts -> rollout(ψ, as, ts, sys, integrator=expv)[:, end], ts
    )
    iso_ket_dim = length(ket_to_iso(ψ))
    @test size(result1) == (iso_ket_dim, T)

    result2 = ForwardDiff.jacobian(
        ts -> unitary_rollout(as, ts, sys, integrator=expv)[:, end], ts
    )
    iso_vec_dim = length(operator_to_iso_vec(sys.H(zeros(sys.n_drives))))
    @test size(result2) == (iso_vec_dim, T)
end

@testitem "Test variational rollouts" begin
    include("../test/test_utils.jl")
    sys = QuantumSystem([PAULIS.X, PAULIS.Y])
    varsys1 = VariationalQuantumSystem([PAULIS.X, PAULIS.Y], [PAULIS.X])
    varsys2 = VariationalQuantumSystem([PAULIS.X, PAULIS.Y], [PAULIS.X, PAULIS.Y])
    U_goal = GATES.H

    # state rollouts
    traj = named_trajectory_type_2()
    ψ̃s_def = rollout(traj, sys)
    ψ̃s_match = []
    
    dims = size(ψ̃s_def[1])
    for vs in [varsys1, varsys2]
        ψ̃s, ψ̃s_vars = variational_rollout(traj, vs)
        push!(ψ̃s_match, [ψ̃_vars[1] for ψ̃_vars in ψ̃s_vars])
        
        @assert ψ̃s ≈ ψ̃s_def
        @assert length(ψ̃s_vars[1]) == length(vs.G_vars)
        for (i, ψ̃_vars) in enumerate(ψ̃s_vars)
            for ψ̃_var in ψ̃_vars
                @assert size(ψ̃_var) == dims
            end
        end
    end
    # same operator (different system)
    @assert ψ̃s_match[1] ≈ ψ̃s_match[2]

    # unitary rollouts
    traj = named_trajectory_type_1()
    Ũ⃗_def = unitary_rollout(traj, sys)
    Ũ⃗ᵥ1_match = []

    for vs in [varsys1, varsys2]
        Ũ⃗, Ũ⃗_vars = variational_unitary_rollout(traj, vs)
        push!(Ũ⃗ᵥ1_match, Ũ⃗_vars[1])

        @assert Ũ⃗ ≈ Ũ⃗_def
        @assert length(Ũ⃗_vars) == length(vs.G_vars)
        @assert size(Ũ⃗_vars[1]) == size(Ũ⃗_def)
    end
    # same operator (different system)
    @assert Ũ⃗ᵥ1_match[1] ≈ Ũ⃗ᵥ1_match[2]
end


end
