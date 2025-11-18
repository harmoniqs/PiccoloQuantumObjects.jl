module QuantumSystemTemplates

"""
Quantum system templates for common physical systems.

Provides constructors for:
- **TransmonSystem**: Transmon qubits with anharmonicity
- **MultiTransmonSystem**: Multi-transmon systems with couplings
- **RydbergChainSystem**: Rydberg atom chains
- **CatSystem**: Quantum cat states in cavities
"""

using LinearAlgebra
using TestItems

# Import from parent PiccoloQuantumObjects modules
using ..QuantumSystems: QuantumSystem, OpenQuantumSystem, CompositeQuantumSystem, AbstractQuantumSystem, lift_operator
using ..QuantumObjectUtils: annihilate, operator_from_string, PAULIS
using ..Gates: GATES

const ⊗ = kron

include("transmons.jl")
include("rydberg.jl")
include("cats.jl")
include("ions.jl")

end  # module QuantumSystemTemplates
