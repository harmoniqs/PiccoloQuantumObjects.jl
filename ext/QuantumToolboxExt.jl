module QuantumToolboxExt

using NamedTrajectories
using PiccoloQuantumObjects
import PiccoloQuantumObjects: get_c_ops

using LinearAlgebra
using QuantumToolbox
using TestItems


function zoh(times::AbstractVector{<:Real}, controls::AbstractVector{<:Real})
    if isempty(times)
        throw(ArgumentError("Time points cannot be empty"))
    elseif length(times) != length(controls)
        throw(DimensionMismatch("Times and amplitudes must have same length"))
    end
    # no parameters
    return (_, t) -> controls[clamp(searchsortedlast(times, t), 1, length(times))]
end

"""
    hamiltonian_evolution(traj::NamedTrajectory, sys::AbstractQuantumSystem)

Internal function to construct the Hamiltonian part of a `QobjEvo` from a system and trajectory.
"""
function hamiltonian_evolution(
    traj::NamedTrajectory, 
    sys::AbstractQuantumSystem;
    control_name::Symbol = :a
)
    @assert sys.n_drives == size(traj[control_name], 1)
    times = get_times(traj)
    H_drift = Qobj(get_drift(sys))
    H_drives = sum(
        QobjEvo(Qobj(H), zoh(times, traj[control_name][j, :])) 
        for (j, H) in enumerate(get_drives(sys))
    )
    return H_drift + H_drives
end

"""
    QobjEvo(sys::QuantumSystem, traj::NamedTrajectory)

Converts a `QuantumSystem` and a `NamedTrajectory` into a `QobjEvo` object using zero-order hold.

The time-dependent Hamiltonian is constructed as ``\\hat{H}(t) = \\hat{H}_0 + \\sum_j a_j(t) \\hat{H}_j``, where ``\\hat{H}_0`` is the drift Hamiltonian, ``\\hat{H}_j`` are the drive Hamiltonians, and ``a_j(t)`` are the control amplitudes from `traj.a`.
"""
function QuantumToolbox.QobjEvo(
    traj::NamedTrajectory, sys::AbstractQuantumSystem; kwargs...
)
    comps = hamiltonian_evolution(traj, sys; kwargs...)
    return QobjEvo(comps)
end

"""
    get_c_ops(sys::AbstractQuantumSystem)

Returns a vector of `Qobj` dissipation operators.
"""
get_c_ops(::AbstractQuantumSystem) = Qobj[]

get_c_ops(sys::OpenQuantumSystem) = Qobj.(sys.dissipation_operators)

# =========================================================================== #

@testitem "QobjEvo tests" begin
    using QuantumToolbox
    include("../test/test_utils.jl")

    # X gate
    traj = named_trajectory_type_2()

    sys = QuantumSystem([PAULIS.X, PAULIS.Y], 3.92, [(-1.0, 1.0), (-1.0, 1.0)])

    open_sys = OpenQuantumSystem(
        [PAULIS.X, PAULIS.Y], 1.0, [1.0, 1.0], dissipation_operators=[annihilate(2)]
    )

    ψ0 = Qobj([1; 0])
    ψT = Qobj([0; 1])

    @testset "mesolve with QuantumSystem and traj" begin
        H_t = QobjEvo(traj, sys)
        @test H_t isa QuantumToolbox.QobjEvo

        res = mesolve(H_t, ψ0, get_times(traj))
        
        @test abs2(res.states[end]'ψT) ≈ 1.0 atol=1e-2
        @test length(res.states) == traj.T
    end

    @testset "mesolve with OpenQuantumSystem and traj" begin
        H_t = QobjEvo(traj, open_sys)
        @test H_t isa QuantumToolbox.QobjEvo

        res = mesolve(H_t, ψ0, get_times(traj))
        @test size(res.states[end]) == (2,)
        @test length(res.states) == traj.T

        open_res = mesolve(H_t, ψ0, get_times(traj), get_c_ops(open_sys))
        @test size(open_res.states[end]) == (2, 2)
        @test length(open_res.states) == traj.T

        # @test abs2(res.states[end]'ψT) ≈ 1.0 atol=1e-2
    end

end

end
