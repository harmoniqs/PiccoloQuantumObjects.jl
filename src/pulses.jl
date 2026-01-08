module Pulses

"""
Pulse types for quantum control.

Provides interpolated and analytic pulse types:
- `ZeroOrderPulse`: Piecewise constant (zero-order hold)
- `LinearSplinePulse`: Linear interpolation between samples
- `CubicSplinePulse`: Cubic spline interpolation
- `GaussianPulse`: Analytic Gaussian envelope

All pulses are callable: `pulse(t)` returns the control vector at time `t`.
"""

export AbstractPulse, AbstractSplinePulse
export ZeroOrderPulse, LinearSplinePulse, CubicSplinePulse, GaussianPulse
export duration, n_drives, sample, drive_name

using DataInterpolations: ConstantInterpolation, LinearInterpolation, CubicHermiteSpline
using TestItems

# ============================================================================ #
# Abstract type
# ============================================================================ #

"""
    AbstractPulse

Abstract type for all pulse types. All pulses are callable: `pulse(t)` returns
the control vector at time `t`.
"""
abstract type AbstractPulse end

"""
    AbstractSplinePulse <: AbstractPulse

Abstract type for spline-based pulses (linear and cubic interpolation).
These pulses use the spline coefficients as optimization variables.
"""
abstract type AbstractSplinePulse <: AbstractPulse end

# Make all pulses callable
(pulse::AbstractPulse)(t::Real) = evaluate(pulse, t)

"""
    duration(pulse::AbstractPulse)

Return the duration of the pulse.
"""
duration(pulse::AbstractPulse) = pulse.duration

"""
    n_drives(pulse::AbstractPulse)

Return the number of control drives in the pulse.
"""
n_drives(pulse::AbstractPulse) = pulse.n_drives

"""
    drive_name(pulse::AbstractPulse)

Return the name of the drive variable for this pulse.
"""
drive_name(pulse::AbstractPulse) = pulse.drive_name

"""
    sample(pulse::AbstractPulse, times::AbstractVector)

Sample the pulse at the given times. Returns a matrix of size `(n_drives, length(times))`.
"""
function sample(pulse::AbstractPulse, times::AbstractVector)
    return hcat([pulse(t) for t in times]...)
end

"""
    sample(pulse::AbstractPulse; n_samples::Int=100)

Sample the pulse uniformly with `n_samples` points. Returns `(controls, times)`.
"""
function sample(pulse::AbstractPulse; n_samples::Int=100)
    times = collect(range(0.0, duration(pulse), length=n_samples))
    return sample(pulse, times), times
end

# ============================================================================ #
# ZeroOrderPulse (piecewise constant / zero-order hold)
# ============================================================================ #

"""
    ZeroOrderPulse{I<:ConstantInterpolation} <: AbstractPulse

Piecewise constant pulse (zero-order hold). The control value at time `t` is
the value at the most recent sample point.

# Fields
- `controls::I`: ConstantInterpolation from DataInterpolations
- `duration::Float64`: Total pulse duration
- `n_drives::Int`: Number of control drives
- `drive_name::Symbol`: Name of the drive variable (default `:u`)
"""
struct ZeroOrderPulse{I<:ConstantInterpolation} <: AbstractPulse
    controls::I
    duration::Float64
    n_drives::Int
    drive_name::Symbol
end

"""
    ZeroOrderPulse(controls::AbstractMatrix, times::AbstractVector; drive_name=:u)

Create a zero-order hold pulse from control samples and times.

# Arguments
- `controls`: Matrix of size `(n_drives, n_times)` with control values
- `times`: Vector of sample times (must start at 0)

# Keyword Arguments
- `drive_name`: Name of the drive variable (default `:u`)
"""
function ZeroOrderPulse(controls::AbstractMatrix, times::AbstractVector; drive_name::Symbol=:u)
    # Materialize to Matrix/Vector to ensure consistent type parameters
    interp = ConstantInterpolation(Matrix(controls), collect(times))
    return ZeroOrderPulse(interp, Float64(times[end]), size(controls, 1), drive_name)
end

evaluate(p::ZeroOrderPulse, t) = p.controls(t)

# ============================================================================ #
# LinearSplinePulse
# ============================================================================ #

"""
    LinearSplinePulse{I<:LinearInterpolation} <: AbstractSplinePulse

Pulse with linear interpolation between sample points.

# Fields
- `controls::I`: LinearInterpolation from DataInterpolations
- `duration::Float64`: Total pulse duration
- `n_drives::Int`: Number of control drives
- `drive_name::Symbol`: Name of the drive variable (default `:u`)
"""
struct LinearSplinePulse{I<:LinearInterpolation} <: AbstractSplinePulse
    controls::I
    duration::Float64
    n_drives::Int
    drive_name::Symbol
end

"""
    LinearSplinePulse(controls::AbstractMatrix, times::AbstractVector; drive_name=:u)

Create a linearly interpolated pulse from control samples and times.

# Arguments
- `controls`: Matrix of size `(n_drives, n_times)` with control values
- `times`: Vector of sample times (must start at 0)

# Keyword Arguments
- `drive_name`: Name of the drive variable (default `:u`)
"""
function LinearSplinePulse(controls::AbstractMatrix, times::AbstractVector; drive_name::Symbol=:u)
    # Materialize to Matrix/Vector to ensure consistent type parameters
    interp = LinearInterpolation(Matrix(controls), collect(times))
    return LinearSplinePulse(interp, Float64(times[end]), size(controls, 1), drive_name)
end

evaluate(p::LinearSplinePulse, t) = p.controls(t)

# ============================================================================ #
# CubicSplinePulse (Hermite spline with explicit derivatives)
# ============================================================================ #

"""
    CubicSplinePulse{I<:CubicHermiteSpline} <: AbstractPulse

Pulse with cubic Hermite spline interpolation. Uses both control values AND 
derivatives for exact reconstruction after optimization.

# Fields
- `controls::I`: CubicHermiteSpline from DataInterpolations
- `duration::Float64`: Total pulse duration
- `n_drives::Int`: Number of control drives
- `drive_name::Symbol`: Name of the drive variable (default `:u`)
"""
struct CubicSplinePulse{I<:CubicHermiteSpline} <: AbstractSplinePulse
    controls::I
    duration::Float64
    n_drives::Int
    drive_name::Symbol
end

"""
    CubicSplinePulse(controls::AbstractMatrix, derivatives::AbstractMatrix, times::AbstractVector; drive_name=:u)

Create a cubic Hermite spline pulse from control values, derivatives, and times.

# Arguments
- `controls`: Matrix of size `(n_drives, n_times)` with control values
- `derivatives`: Matrix of size `(n_drives, n_times)` with control derivatives
- `times`: Vector of sample times (must start at 0)

# Keyword Arguments
- `drive_name`: Name of the drive variable (default `:u`)
"""
function CubicSplinePulse(controls::AbstractMatrix, derivatives::AbstractMatrix, times::AbstractVector; drive_name::Symbol=:u)
    # Materialize to Matrix to ensure consistent type parameters across construction methods
    interp = CubicHermiteSpline(Matrix(derivatives), Matrix(controls), collect(times))
    return CubicSplinePulse(interp, Float64(times[end]), size(controls, 1), drive_name)
end

"""
    CubicSplinePulse(controls::AbstractMatrix, times::AbstractVector; drive_name=:u)

Create a cubic Hermite spline pulse with zero derivatives at all knot points.
Useful for initial guesses where smoothness constraints will be enforced by optimizer.

# Arguments
- `controls`: Matrix of size `(n_drives, n_times)` with control values
- `times`: Vector of sample times (must start at 0)

# Keyword Arguments
- `drive_name`: Name of the drive variable (default `:u`)
"""
function CubicSplinePulse(controls::AbstractMatrix, times::AbstractVector; drive_name::Symbol=:u)
    derivatives = zeros(size(controls))
    return CubicSplinePulse(controls, derivatives, times; drive_name)
end

evaluate(p::CubicSplinePulse, t) = p.controls(t)

# ============================================================================ #
# GaussianPulse (analytic)
# ============================================================================ #

"""
    GaussianPulse{F<:Function} <: AbstractPulse

Analytic Gaussian pulse. Each drive has its own amplitude, width (sigma), and center.

    u_i(t) = amplitudes[i] * exp(-(t - centers[i])² / (2 * sigmas[i]²))

# Fields
- `f::F`: Function that evaluates the pulse
- `amplitudes::Vector{Float64}`: Peak amplitude for each drive
- `sigmas::Vector{Float64}`: Gaussian width for each drive
- `centers::Vector{Float64}`: Center time for each drive
- `duration::Float64`: Total pulse duration
- `n_drives::Int`: Number of control drives
"""
struct GaussianPulse{F<:Function} <: AbstractPulse
    f::F
    amplitudes::Vector{Float64}
    sigmas::Vector{Float64}
    centers::Vector{Float64}
    duration::Float64
    n_drives::Int
end

"""
    GaussianPulse(amplitudes, sigmas, centers, duration)

Create a Gaussian pulse with per-drive parameters.

# Arguments
- `amplitudes`: Peak amplitude for each drive
- `sigmas`: Gaussian width (standard deviation) for each drive
- `centers`: Center time for each drive
- `duration`: Total pulse duration
"""
function GaussianPulse(
    amplitudes::AbstractVector{<:Real},
    sigmas::AbstractVector{<:Real},
    centers::AbstractVector{<:Real},
    duration::Real
)
    n = length(amplitudes)
    @assert length(sigmas) == n "sigmas must have same length as amplitudes"
    @assert length(centers) == n "centers must have same length as amplitudes"
    
    amps = Vector{Float64}(amplitudes)
    sigs = Vector{Float64}(sigmas)
    ctrs = Vector{Float64}(centers)
    dur = Float64(duration)
    
    f = t -> [amps[i] * exp(-(t - ctrs[i])^2 / (2 * sigs[i]^2)) for i in 1:n]
    return GaussianPulse(f, amps, sigs, ctrs, dur, n)
end

"""
    GaussianPulse(amplitudes, sigma, duration; center=duration/2)

Create a Gaussian pulse with shared sigma and center across all drives.

# Arguments
- `amplitudes`: Peak amplitude for each drive
- `sigma`: Shared Gaussian width for all drives
- `duration`: Total pulse duration

# Keyword Arguments
- `center`: Shared center time (default: `duration/2`)
"""
function GaussianPulse(
    amplitudes::AbstractVector{<:Real},
    sigma::Real,
    duration::Real;
    center::Real=duration/2
)
    n = length(amplitudes)
    return GaussianPulse(
        amplitudes, 
        fill(Float64(sigma), n), 
        fill(Float64(center), n), 
        duration
    )
end

evaluate(p::GaussianPulse, t) = p.f(t)

# ============================================================================ #
# NamedTrajectory Constructors
# ============================================================================ #

using NamedTrajectories: NamedTrajectory, get_times

"""
    ZeroOrderPulse(traj::NamedTrajectory; drive_name=:u)

Construct a ZeroOrderPulse from a NamedTrajectory.

# Arguments
- `traj`: NamedTrajectory with control data

# Keyword Arguments
- `drive_name`: Name of the drive component (default: `:u`)
"""
function ZeroOrderPulse(traj::NamedTrajectory; drive_name::Symbol=:u)
    controls = traj[drive_name]
    times = get_times(traj)
    return ZeroOrderPulse(controls, times; drive_name)
end

"""
    LinearSplinePulse(traj::NamedTrajectory; drive_name=:u)

Construct a LinearSplinePulse from a NamedTrajectory.

# Arguments
- `traj`: NamedTrajectory with control data

# Keyword Arguments
- `drive_name`: Name of the drive component (default: `:u`)
"""
function LinearSplinePulse(traj::NamedTrajectory; drive_name::Symbol=:u)
    controls = traj[drive_name]
    times = get_times(traj)
    return LinearSplinePulse(controls, times; drive_name)
end

"""
    CubicSplinePulse(traj::NamedTrajectory; drive_name=:u, derivative_name=:du)

Construct a CubicSplinePulse (Hermite) from a NamedTrajectory using both 
control values and derivatives.

# Arguments
- `traj`: NamedTrajectory with control and derivative data

# Keyword Arguments
- `drive_name`: Name of the drive component (default: `:u`)
- `derivative_name`: Name of the derivative component (default: `:du`)
"""
function CubicSplinePulse(traj::NamedTrajectory; drive_name::Symbol=:u, derivative_name::Symbol=:du)
    controls = traj[drive_name]
    derivatives = traj[derivative_name]
    times = get_times(traj)
    return CubicSplinePulse(controls, derivatives, times; drive_name)
end

# ============================================================================ #
# Tests
# ============================================================================ #

@testitem "ZeroOrderPulse" begin
    using PiccoloQuantumObjects: ZeroOrderPulse, duration, n_drives, sample, drive_name
    
    # Create simple pulse
    controls = [0.0 1.0 0.5 0.0; 0.0 -1.0 -0.5 0.0]
    times = [0.0, 0.25, 0.5, 1.0]
    pulse = ZeroOrderPulse(controls, times)
    
    @test duration(pulse) == 1.0
    @test n_drives(pulse) == 2
    @test drive_name(pulse) == :u  # Default
    
    # Test evaluation (zero-order hold)
    @test pulse(0.0) ≈ [0.0, 0.0]
    @test pulse(0.1) ≈ [0.0, 0.0]  # Before first transition
    @test pulse(0.3) ≈ [1.0, -1.0] # After first transition
    @test pulse(1.0) ≈ [0.0, 0.0]
    
    # Test sampling
    sampled, ts = sample(pulse; n_samples=5)
    @test size(sampled) == (2, 5)
    @test length(ts) == 5
    
    # Test custom drive_name
    pulse_custom = ZeroOrderPulse(controls, times; drive_name=:Ω)
    @test drive_name(pulse_custom) == :Ω
end

@testitem "LinearSplinePulse" begin
    using PiccoloQuantumObjects: LinearSplinePulse, duration, n_drives, sample, drive_name
    
    # Create simple pulse
    controls = [0.0 1.0 0.0; 0.0 -1.0 0.0]
    times = [0.0, 0.5, 1.0]
    pulse = LinearSplinePulse(controls, times)
    
    @test duration(pulse) == 1.0
    @test n_drives(pulse) == 2
    @test drive_name(pulse) == :u  # Default
    
    # Test linear interpolation
    @test pulse(0.0) ≈ [0.0, 0.0]
    @test pulse(0.25) ≈ [0.5, -0.5]  # Midpoint of first segment
    @test pulse(0.5) ≈ [1.0, -1.0]
    @test pulse(0.75) ≈ [0.5, -0.5]  # Midpoint of second segment
    @test pulse(1.0) ≈ [0.0, 0.0]
    
    # Test sampling
    sampled, ts = sample(pulse; n_samples=5)
    @test size(sampled) == (2, 5)
    
    # Test custom drive_name
    pulse_custom = LinearSplinePulse(controls, times; drive_name=:amplitude)
    @test drive_name(pulse_custom) == :amplitude
end

@testitem "CubicSplinePulse" begin
    using PiccoloQuantumObjects: CubicSplinePulse, duration, n_drives, sample, drive_name
    
    # Create pulse with explicit derivatives (Hermite spline)
    controls = [0.0 0.5 1.0 0.5 0.0; 0.0 -0.5 -1.0 -0.5 0.0]
    derivatives = [2.0 2.0 0.0 -2.0 -2.0; -2.0 -2.0 0.0 2.0 2.0]  # Symmetric derivatives
    times = [0.0, 0.25, 0.5, 0.75, 1.0]
    pulse = CubicSplinePulse(controls, derivatives, times)
    
    @test duration(pulse) == 1.0
    @test n_drives(pulse) == 2
    @test drive_name(pulse) == :u  # Default
    
    # Test that endpoints are correct
    @test pulse(0.0) ≈ [0.0, 0.0]
    @test pulse(1.0) ≈ [0.0, 0.0]
    
    # Test that it passes through sample points
    @test pulse(0.5) ≈ [1.0, -1.0]
    
    # Cubic spline should be smooth (no discontinuities)
    @test pulse(0.4) isa Vector{Float64}
    @test pulse(0.6) isa Vector{Float64}
    
    # Test custom drive_name
    pulse_custom = CubicSplinePulse(controls, derivatives, times; drive_name=:a)
    @test drive_name(pulse_custom) == :a
    
    # Test zero-derivative constructor
    pulse_zero_deriv = CubicSplinePulse(controls, times)
    @test pulse_zero_deriv(0.5) ≈ [1.0, -1.0]
end

@testitem "GaussianPulse" begin
    using PiccoloQuantumObjects: GaussianPulse, duration, n_drives, sample
    
    # Test with per-drive parameters
    amplitudes = [1.0, 2.0]
    sigmas = [0.1, 0.2]
    centers = [0.5, 0.6]
    dur = 1.0
    
    pulse = GaussianPulse(amplitudes, sigmas, centers, dur)
    
    @test duration(pulse) == 1.0
    @test n_drives(pulse) == 2
    @test pulse.amplitudes == amplitudes
    @test pulse.sigmas == sigmas
    @test pulse.centers == centers
    
    # Test that peak is at center
    @test pulse(0.5)[1] ≈ 1.0  # First drive peaks at t=0.5
    @test pulse(0.6)[2] ≈ 2.0  # Second drive peaks at t=0.6
    
    # Test Gaussian decay
    @test pulse(0.0)[1] < 0.01  # Should be near zero far from center
    @test pulse(1.0)[1] < 0.01
    
    # Test convenience constructor with shared parameters
    pulse2 = GaussianPulse([1.0, 2.0], 0.1, 1.0)
    @test duration(pulse2) == 1.0
    @test n_drives(pulse2) == 2
    @test pulse2.centers == [0.5, 0.5]  # Default center is duration/2
    @test pulse2.sigmas == [0.1, 0.1]
    
    # Test with custom center
    pulse3 = GaussianPulse([1.0], 0.1, 1.0; center=0.3)
    @test pulse3.centers == [0.3]
    @test pulse3(0.3)[1] ≈ 1.0  # Peak at specified center
end

@testitem "Pulse callability" begin
    using PiccoloQuantumObjects: ZeroOrderPulse, LinearSplinePulse, GaussianPulse
    
    # All pulse types should be callable
    controls = [0.0 1.0 0.0]
    times = [0.0, 0.5, 1.0]
    
    zop = ZeroOrderPulse(controls, times)
    lp = LinearSplinePulse(controls, times)
    gp = GaussianPulse([1.0], 0.1, 1.0)
    
    # Test callable interface
    for pulse in [zop, lp, gp]
        @test pulse(0.5) isa Vector{Float64}
        @test length(pulse(0.5)) == 1
    end
end

@testitem "Pulse from NamedTrajectory" begin
    using PiccoloQuantumObjects: ZeroOrderPulse, LinearSplinePulse, CubicSplinePulse
    using PiccoloQuantumObjects: duration, n_drives
    using NamedTrajectories: NamedTrajectory
    
    # Create a simple NamedTrajectory with controls and derivatives
    T = 5
    times = collect(range(0, 1, length=T))
    Δt = diff(times)
    Δt = [Δt; Δt[end]]
    
    u = [sin(π * t) for t in times]
    du = [π * cos(π * t) for t in times]
    
    traj = NamedTrajectory(
        (u = reshape(u, 1, :), du = reshape(du, 1, :), Δt = Δt);
        timestep=:Δt,
        controls=(:u,)
    )
    
    # Test ZeroOrderPulse from NamedTrajectory
    zop = ZeroOrderPulse(traj)
    @test duration(zop) ≈ 1.0
    @test n_drives(zop) == 1
    @test zop(0.0) ≈ [0.0] atol=1e-10
    
    # Test LinearSplinePulse from NamedTrajectory
    lp = LinearSplinePulse(traj)
    @test duration(lp) ≈ 1.0
    @test n_drives(lp) == 1
    @test lp(0.5) ≈ [1.0] atol=1e-10
    
    # Test CubicSplinePulse from NamedTrajectory (uses both u and du)
    csp = CubicSplinePulse(traj)
    @test duration(csp) ≈ 1.0
    @test n_drives(csp) == 1
    @test csp(0.5) ≈ [1.0] atol=1e-10
    @test csp(0.0) ≈ [0.0] atol=1e-10
end

end # module Pulses
