"""
Trajectory initialization helper functions.

Provides utilities for interpolating between quantum states and operators:
- `linear_interpolation`: Generic linear interpolation
- `unitary_linear_interpolation`: Linear interpolation of unitary operators
- `unitary_geodesic`: Geodesic interpolation on the unitary manifold
"""

"""
    linear_interpolation(x, y, n)

Linear interpolation between vectors or matrices.
"""
linear_interpolation(x::AbstractVector, y::AbstractVector, n::Int) = hcat(range(x, y, n)...)
linear_interpolation(X::AbstractMatrix, Y::AbstractMatrix, n::Int) =
    hcat([X + (Y - X) * t for t in range(0, 1, length=n)]...)

"""
    unitary_linear_interpolation(U_init, U_goal, samples)

Compute a linear interpolation of unitary operators with `samples` samples.
"""
function unitary_linear_interpolation(
    U_init::AbstractMatrix{<:Number},
    U_goal::AbstractMatrix{<:Number},
    samples::Int
)
    Ũ⃗_init = operator_to_iso_vec(U_init)
    Ũ⃗_goal = operator_to_iso_vec(U_goal)
    Ũ⃗s = [Ũ⃗_init + (Ũ⃗_goal - Ũ⃗_init) * t for t ∈ range(0, 1, length=samples)]
    Ũ⃗ = hcat(Ũ⃗s...)
    return Ũ⃗
end

function unitary_linear_interpolation(
    U_init::AbstractMatrix{<:Number},
    U_goal::EmbeddedOperator,
    samples::Int
)
    return unitary_linear_interpolation(U_init, U_goal.operator, samples)
end

"""
    unitary_geodesic(U_init, U_goal, times; kwargs...)

Compute the geodesic connecting U_init and U_goal at the specified times.

# Arguments
- `U_init::AbstractMatrix{<:Number}`: The initial unitary operator.
- `U_goal::AbstractMatrix{<:Number}`: The goal unitary operator.
- `times::AbstractVector{<:Number}`: The times at which to evaluate the geodesic.

# Keyword Arguments
- `return_unitary_isos::Bool=true`: If true returns a matrix where each column is a unitary 
    isovec, i.e. vec(vcat(real(U), imag(U))). If false, returns a vector of unitary matrices.
- `return_generator::Bool=false`: If true, returns the effective Hamiltonian generating 
    the geodesic.
- `H_drift::AbstractMatrix{<:Number}=zeros(size(U_init))`: Drift Hamiltonian for time-dependent systems.
"""
function unitary_geodesic(
    U_init::AbstractMatrix{<:Number},
    U_goal::AbstractMatrix{<:Number},
    times::AbstractVector{<:Number};
    return_unitary_isos=true,
    return_generator=false,
    H_drift::AbstractMatrix{<:Number}=zeros(size(U_init)),
)
    t₀ = times[1]
    T = times[end] - t₀

    U_drift(t) = exp(-im * H_drift * t)
    H = im * log(U_drift(T)' * (U_goal * U_init')) / T
    # -im prefactor is not included in H
    U_geo = [U_drift(t) * exp(-im * H * (t - t₀)) * U_init for t ∈ times]

    if !return_unitary_isos
        if return_generator
            return U_geo, H
        else
            return U_geo
        end
    else
        Ũ⃗_geo = stack(operator_to_iso_vec.(U_geo), dims=2)
        if return_generator
            return Ũ⃗_geo, H
        else
            return Ũ⃗_geo
        end
    end
end

function unitary_geodesic(
    U_goal::AbstractPiccoloOperator,
    samples::Int;
    kwargs...
)
    return unitary_geodesic(
        I(size(U_goal, 1)),
        U_goal,
        samples;
        kwargs...
    )
end

function unitary_geodesic(
    U_init::AbstractMatrix{<:Number},
    U_goal::EmbeddedOperator,
    samples::Int;
    H_drift::AbstractMatrix{<:Number}=zeros(size(U_init)),
    kwargs...
)
    H_drift = unembed(H_drift, U_goal)
    U1 = unembed(U_init, U_goal)
    U2 = unembed(U_goal)
    Ũ⃗ = unitary_geodesic(U1, U2, samples; H_drift=H_drift, kwargs...)
    return hcat([
        operator_to_iso_vec(embed(iso_vec_to_operator(Ũ⃗ₜ), U_goal))
        for Ũ⃗ₜ ∈ eachcol(Ũ⃗)
    ]...)
end

function unitary_geodesic(
    U_init::AbstractMatrix{<:Number},
    U_goal::AbstractMatrix{<:Number},
    samples::Int;
    kwargs...
)
    return unitary_geodesic(U_init, U_goal, range(0, 1, samples); kwargs...)
end
