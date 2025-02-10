module Rollouts

export free_phase

export rollout
export open_rollout
export unitary_rollout
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
using LinearAlgebra
using ProgressMeter
using TestItems
using ForwardDiff

function fidelity(ψ::AbstractVector, ψ_goal::AbstractVector)
    return abs(dot(ψ, ψ_goal))^2
end

function unitary_fidelity(
    U::AbstractMatrix,
    U_goal::AbstractMatrix;
    subspace::AbstractVector{Int}=axes(U, 1)
)
    U = U[subspace, subspace]
    U_goal = U_goal[subspace, subspace]
    N = size(U, 1)
    return abs(tr(U' * U_goal))^2 / N^2
end

function free_phase(
    ϕs::AbstractVector,
    Hs::AbstractVector{<:AbstractMatrix}
)
    # NOTE: switch to expv for ForwardDiff
    # return reduce(kron, [exp(im * ϕ * H) for (ϕ, H) ∈ zip(ϕs, Hs)])
    Id = Matrix{eltype(Hs[1])}(I, size(Hs[1]))
    return reduce(kron, [expv(im * ϕ, H, Id) for (ϕ, H) ∈ zip(ϕs, Hs)])
end

function unitary_free_phase_fidelity(
    U::AbstractMatrix,
    U_goal::AbstractMatrix,
    ϕs::AbstractVector{<:Real},
    phase_operators::AbstractVector{<:AbstractMatrix};
    subspace::AbstractVector{Int}=axes(U, 1)
)
    R = free_phase(ϕs, phase_operators)
    return unitary_fidelity(R * U, U_goal; subspace=subspace)
end


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
        ψ̃_init::AbstractVector{<:Float64},
        controls::AbstractMatrix,
        Δt::AbstractVector,
        system::AbstractQuantumSystem
    )

Rollout a quantum state `ψ̃_init` under the control `controls` for a time `Δt`
using the system `system`.

If `exp_vector_product` is `true`, the integrator is expected to have a signature like
the exponential action, `expv`. Otherwise, it is expected to have a signature like `exp`.

Types should allow for autodifferentiable controls and times.
"""
function rollout(
    ψ̃_init::AbstractVector{<:Real},
    controls::AbstractMatrix,
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
    return vcat([rollout(state, args...; kwargs...) for state ∈ inits]...)
end

function rollout_fidelity(
    ψ̃_init::AbstractVector{<:Real},
    ψ̃_goal::AbstractVector{<:Real},
    controls::AbstractMatrix,
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
    controls::AbstractMatrix,
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
        if startswith(name, state_name)
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
        controls::AbstractMatrix,
        Δt::AbstractVector,
        system::AbstractQuantumSystem;
        kwargs...
    )

Rollout a quantum state `ρ⃗₁` under the control `controls` for a time `Δt`

# Arguments
- `ρ⃗₁::AbstractVector{<:Complex}`: Initial state vector
- `controls::AbstractMatrix`: Control matrix
- `Δt::AbstractVector`: Time steps
- `system::AbstractQuantumSystem`: Quantum system

# Keyword Arguments
- `show_progress::Bool=false`: Show progress bar
- `integrator::Function=expv`: Integrator function
- `exp_vector_product::Bool`: Infer whether the integrator is an exponential-vector product

"""
function open_rollout(
    ρ⃗̃_init::AbstractVector{<:Real},
    controls::AbstractMatrix,
    Δt::AbstractVector,
    system::AbstractQuantumSystem;
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
        controls::AbstractMatrix,
        Δt::AbstractVector,
        system::AbstractQuantumSystem;
        kwargs...
    )

Rollout a density matrix `ρ₁` under the control `controls` and timesteps `Δt`

"""
function open_rollout(
    ρ_init::AbstractMatrix{<:Complex},
    controls::AbstractMatrix,
    Δt::AbstractVector,
    system::AbstractQuantumSystem;
    kwargs...
)
    return open_rollout(density_to_iso_vec(ρ_init), controls, Δt, system; kwargs...)
end

function open_rollout_fidelity(
    ρ_init::AbstractMatrix{<:Complex},
    ρ_goal::AbstractMatrix{<:Complex},
    controls::AbstractMatrix,
    Δt::AbstractVector,
    system::AbstractQuantumSystem;
    kwargs...
)

    ρ⃗̃_traj = open_rollout(ρ_init, controls, Δt, system; kwargs...)
    ρ_final = iso_vec_to_density(ρ⃗̃_traj[:, end])
    return real(tr(ρ_goal * ρ_final))
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

function unitary_rollout(
    Ũ⃗_init::AbstractVector{R1},
    controls::AbstractMatrix{R2},
    Δt::AbstractVector{R3},
    system::AbstractQuantumSystem;
    show_progress=false,
    integrator=expv,
    exp_vector_product=infer_is_evp(integrator),
) where {R1 <: Real, R2 <: Real, R3 <: Real}
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
    controls::AbstractMatrix,
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

function unitary_rollout_fidelity(
    Ũ⃗_init::AbstractVector{<:Real},
    Ũ⃗_goal::AbstractVector{<:Real},
    controls::AbstractMatrix,
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
    controls::AbstractMatrix,
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
    controls::AbstractMatrix,
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
    controls::AbstractMatrix,
    Δt::AbstractVector,
    system::AbstractQuantumSystem;
    kwargs...
) = unitary_rollout_fidelity(operator_to_iso_vec(U_goal), controls, Δt, system; kwargs...)

unitary_rollout_fidelity(
    U_goal::EmbeddedOperator,
    controls::AbstractMatrix,
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
# Experimental rollouts
# ----------------------------------------------------------------------------- #

unitary_rollout_fidelity(
    U_goal::EmbeddedOperator,
    controls::AbstractMatrix{Float64},
    Δt::Union{AbstractVector{Float64}, AbstractMatrix{Float64}, Float64},
    sys::AbstractQuantumSystem;
    subspace=U_goal.subspace,
    kwargs...
) = unitary_rollout_fidelity(U_goal.operator, controls, Δt, sys; subspace=subspace, kwargs...)

# *************************************************************************** #

@testitem "Test rollouts using fidelities" begin
    using ExponentialAction

    include("../test/test_utils.jl")

    traj = named_trajectory_type_1()

    sys = QuantumSystem(0 * GATES[:Z], [GATES[:X], GATES[:Y]])

    U_goal = GATES[:H]

    embedded_U_goal = EmbeddedOperator(U_goal, sys)

    # T = 51
    # Δt = 0.2
    # ts = fill(Δt, T)
    # as = collect([π/(T-1)/Δt * sin.(π*(0:T-1)/(T-1)).^2 zeros(T)]')

    # prob = UnitarySmoothPulseProblem(
    #     sys, U_goal, T, Δt, a_guess=as,
    #     ipopt_options=IpoptOptions(print_level=1),
    #     piccolo_options=PiccoloOptions(verbose=false, free_time=false)
    # )

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

    sys = QuantumSystem(0 * GATES[:Z], [GATES[:X], GATES[:Y]])
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
end
