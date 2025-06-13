module PQOQuantumToolboxExt

using QuantumToolbox
import QuantumToolbox: QobjEvo
using PiccoloQuantumObjects
using NamedTrajectories
using LinearAlgebra

function _zoh(times::AbstractVector{<:Real}, amps::AbstractVector{<:Number})
    isempty(times) && throw(ArgumentError("Time points cannot be empty"))
    length(times) != length(amps) && throw(DimensionMismatch("Times and amplitudes must have same length"))
    return (_, t) -> amps[clamp(searchsortedlast(times, t), 1, length(times))]
end

"""Internal function to construct the Hamiltonian part of a `QobjEvo` from a system and trajectory."""
function _construct_hamiltonian_evolution(sys::Union{QuantumSystem,OpenQuantumSystem}, traj::NamedTrajectory)
    H0 = Qobj(get_drift(sys))
    H_drives = Qobj.(get_drives(sys))
    times = get_times(traj)
    amps_mat = traj.a
    n_controls = size(amps_mat, 1)
    @assert n_controls == length(H_drives) "Number of controls in trajectory ($n_controls) doesn't match number of drive Hamiltonians ($(length(H_drives)))"
    comps = Any[H0]
    for j in 1:n_controls
        push!(comps, (H_drives[j], _zoh(times, @view amps_mat[j, :])))
    end
    return Tuple(comps)
end

"""
    QobjEvo(sys::QuantumSystem, traj::NamedTrajectory)

Converts a `QuantumSystem` and a `NamedTrajectory` into a `QobjEvo` object using zero-order hold.

The time-dependent Hamiltonian is constructed as `\\hat{H}(t) = \\hat{H}_0 + \\sum_j a_j(t) \\hat{H}_j,
where `\\hat{H}_0 is the drift Hamiltonian, \\hat{H}_j are the drive Hamiltonians, and a_j(t) are
the control amplitudes from traj.a."""
function QuantumToolbox.QobjEvo(sys::QuantumSystem, traj::NamedTrajectory)
    comps = _construct_hamiltonian_evolution(sys, traj)
    return QuantumToolbox.QuantumObjectEvolution(comps)
end

"""
    QObjEvo(sys::OpenQuantumSystem, traj::NamedTrajectory)

Converts an `OpenQuantumSystem` and `NamedTrajectory` into:
- A `QObjEvo` for the Hamiltonian part (same as QuantumSystem version)
- A vector of `Qobj` dissipation operators
"""
function QuantumToolbox.QobjEvo(sys::OpenQuantumSystem, traj::NamedTrajectory)
    comps = _construct_hamiltonian_evolution(sys, traj)
    c_ops = Qobj.(sys.dissipation_operators)
    return QuantumToolbox.QuantumObjectEvolution(comps), c_ops
end

end
