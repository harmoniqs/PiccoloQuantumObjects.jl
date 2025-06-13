module PQOQuantumToolboxExt

using QuantumToolbox
import QuantumToolbox: QobjEvo
using PiccoloQuantumObjects
using NamedTrajectories: get_times
import LinearAlgebra

function _zoh(times::AbstractVector{<:Real}, amps::AbstractVector{<:Number})
    isempty(times) && throw(ArgumentError("Time points cannot be empty"))
    length(times) != length(amps) && throw(DimensionMismatch("Times and amplitudes must have same length"))
    return (_, t) -> amps[clamp(searchsortedlast(times, t), 1, length(times))]
end

"""
    QobjEvo(sys::QuantumSystem, traj::NamedTrajectory)

Converts a QuantumSystem and a NamedTrajectory into a QobjEvo object using zero-order hold.

The time-dependent Hamiltonian is constructed as `\\hat{H}(t) = \\hat{H}_0 + \\sum_j a_j(t) \\hat{H}_j,
where `\\hat{H}_0 is the drift Hamiltonian, \\hat{H}_j are the drive Hamiltonians, and a_j(t) are
the control amplitudes from traj.a."""
function QuantumToolbox.QobjEvo(sys::QuantumSystem, traj::NamedTrajectory)
    H0 = Qobj(get_drift(sys))
    H_drives = Qobj.(get_drives(sys))
    times = get_times(traj)
    amps_mat = traj.a
    comps = Any[H0]
    n_controls = size(amps_mat, 1)
    for j in 1:n_controls
        push!(comps, (H_drives[j], _zoh(times, @view amps_mat[j, :])))
    end
    return QuantumToolbox.QuantumObjectEvolution(Tuple(comps))
end

end
