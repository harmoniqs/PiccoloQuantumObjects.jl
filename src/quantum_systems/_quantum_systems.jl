module QuantumSystems

export AbstractQuantumSystem
export QuantumSystem
export OpenQuantumSystem
export VariationalQuantumSystem
export CompositeQuantumSystem
export lift_operator

export get_drift
export get_drives
export get_c_ops

using ..Isomorphisms
using ..QuantumObjectUtils

using LinearAlgebra
using SparseArrays
using TestItems
using ForwardDiff

# ----------------------------------------------------------------------------- #
# AbstractQuantumSystem
# ----------------------------------------------------------------------------- #

"""
    AbstractQuantumSystem

Abstract type for defining systems.
"""
abstract type AbstractQuantumSystem end

# ----------------------------------------------------------------------------- #
# AbstractQuantumSystem methods
# ----------------------------------------------------------------------------- #

"""
    get_drift(sys::AbstractQuantumSystem)

Returns the drift Hamiltonian of the system.
"""
get_drift(sys::AbstractQuantumSystem) = sys.H(zeros(sys.n_drives), 0.0)

"""
    get_drives(sys::AbstractQuantumSystem)

Returns the drive Hamiltonians of the system.
"""
function get_drives(sys::AbstractQuantumSystem)
    H_drift = get_drift(sys)
    # Basis vectors for controls will extract drive operators
    return [sys.H(I[1:sys.n_drives, i], 0.0) - H_drift for i ∈ 1:sys.n_drives]
end

function Base.show(io::IO, sys::AbstractQuantumSystem)
    print(io, "$(nameof(typeof(sys))): levels = $(sys.levels), n_drives = $(sys.n_drives)")
end

# ----------------------------------------------------------------------------- #
# Quantum Toolbox ext
# ----------------------------------------------------------------------------- #

function get_c_ops end

# ----------------------------------------------------------------------------- #
# Quantum System Types
# ----------------------------------------------------------------------------- #

include("quantum_systems.jl")
include("open_quantum_systems.jl")
include("variational_quantum_systems.jl")
include("composite_quantum_systems.jl")

end
