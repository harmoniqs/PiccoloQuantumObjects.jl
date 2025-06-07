module PQOQuantumToolboxExt

using QuantumToolbox
import QuantumToolbox: Qobj, length, size
import PiccolouantumObjects
import PiccolouantumObjects: get_drift, get_drives, QobjEvo
import NamedTrajectories
import NamedTrajectories: get_times
import Interpolations
import Interpolations: LinearInterpolation
import LinearAlgebra

"""
    QobjEvo(sys::QuantumSystem, traj::NamedTrajectory) -> QuantumToolbox.QobjEvo

Converts a `QuantumSystem` and a `NamedTrajectory` into a `QobjEvo` object for a closed quantum system.
"""
function PiccoloQuantumObjects.QobjEvo(sys::QuantumSystem, traj::NamedTrajectory)
    H_drift_matrix = get_drift(sys)
    H_drives_matrices = get_drives(sys)
    H0_qobj = Qobj(H_drift_matrix)
    Hj_qobjs = [Qobj(Hj) for Hj in H_drives_matrices]
    times = get_times(traj)
    control_amplitudes_data = traj.a
    num_drives_sys = length(Hj_qobjs)
    num_drives_traj = size(control_amplitudes_data, 1)
    qobj_evo_components = Vector{Any}(undef, num_drives_sys + 1)
    qobj_evo_components[1] = H0_qobj
    for j in 1:num_drives_sys
        itp = LinearInterpolation(times, control_amplitudes_data[j, :], extrapolation_bc=Interpolations.Flat())
        coef_func = (p, t) -> itp(t)
        qobj_evo_components[j+1] = (Hj_qobjs[j], coef_func)
    end
    return QobjEvo(Tuple(qobj_evo_components))
end

end
