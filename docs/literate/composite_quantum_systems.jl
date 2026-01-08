# ```@meta
# CollapsedDocStrings = true
# ```

# # `Composite Quantum Systems`

using PiccoloQuantumObjects
using SparseArrays # for visualization
⊗ = kron;

#=
## Composite quantum systems

A [`CompositeQuantumSystem`](@ref) is constructed from a list of subsystems and their 
interactions. The interaction, in the form of drift or drive Hamiltonian, acts on the full
Hilbert space. The subsystems, with their own drift and drive Hamiltonians, are internally
lifted to the full Hilbert space.

=#

system_1 = QuantumSystem([PAULIS[:X]], [(-1.0, 1.0)])
system_2 = QuantumSystem([PAULIS[:Y]], [(-1.0, 1.0)])
H_drift = PAULIS[:Z] ⊗ PAULIS[:Z]
system = CompositeQuantumSystem(H_drift, Matrix{ComplexF64}[], [system_1, system_2], Float64[]);

# _The drift Hamiltonian is the ZZ coupling._
get_drift(system) |> sparse

# _The drives are the X and Y operators on the first and second subsystems._
drives = get_drives(system)
drives[1] |> sparse

#
drives[2] |> sparse

#=
### The `lift_operator` function

To lift operators acting on a subsystem into the full Hilbert space, use [`lift_operator`](@ref).
```@docs; canonical = false
lift_operator
```
=#

# _Create an `a + a'` operator acting on the 1st subsystem of a qutrit and qubit system._
subspace_levels = [3, 2]
lift_operator(create(3) + annihilate(3), 1, subspace_levels) .|> real |> sparse

# _Create IXI operator on the 2nd qubit in a 3-qubit system._
lift_operator(PAULIS[:X], 2, 3) .|> real |> sparse

# _Create an XX operator acting on qubits 3 and 4 in a 4-qubit system._
lift_operator([PAULIS[:X], PAULIS[:X]], [3, 4], 4) .|> real |> sparse

#=
We can also lift an operator that entangles different subspaces by passing the indices
of the entangled subsystems.
=#

#_Here's another way to create an XX operator acting on qubits 3 and 4 in a 4-qubit system._
lift_operator(kron(PAULIS[:X], PAULIS[:X]), [3, 4], 4) .|> real |> sparse

# _Lift a CX gate acting on the 1st and 3rd qubits in a 3-qubit system._
# _The result is independent of the state of the second qubit._
lift_operator(GATES[:CX], [1, 3], 3) .|> real |> sparse
