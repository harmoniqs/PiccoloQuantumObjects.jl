# ```@meta
# CollapsedDocStrings = true
# ```

# # `Abstract Quantum Systems`

using PiccoloQuantumObjects
using SparseArrays # for visualization
⊗ = kron;

#=
```@docs; canonical = false
AbstractQuantumSystem
```
=#

#=

## Quantum Systems

The [`QuantumSystem`](@ref) type is used to represent a quantum system with a drift 
Hamiltonian and a set of drive Hamiltonians,

```math
H(u, t) = H_{\text{drift}} + \sum_i u_i H_{\text{drives}}^{(i)}
```

where ``u`` is the control vector and ``t`` is time.

```@docs; canonical = false
QuantumSystem
```

`QuantumSystem`'s are containers for quantum dynamics. Internally, they compute the
necessary isomorphisms to perform the dynamics in a real vector space. All systems
require explicit specification of `T_max` (maximum time) and `drive_bounds` (control bounds).

=#

H_drift = PAULIS[:Z]
H_drives = [PAULIS[:X], PAULIS[:Y]]
T_max = 10.0
drive_bounds = [(-1.0, 1.0), (-1.0, 1.0)]
system = QuantumSystem(H_drift, H_drives, T_max, drive_bounds)

u_controls = [1.0, 0.0]
t = 0.0
system.H(u_controls, t)

#=
To extract the drift and drive Hamiltonians from a `QuantumSystem`, use the 
[`get_drift`](@ref) and [`get_drives`](@ref) functions. 

=#

get_drift(system) |> sparse

# _Get the X drive._
drives = get_drives(system)
drives[1] |> sparse

# _And the Y drive._
drives[2] |> sparse

#=
!!! note
    We can also construct a `QuantumSystem` directly from a Hamiltonian function.
    The function must accept `(u, t)` arguments where `u` is the control vector and `t` is time.
=#

H(u, t) = PAULIS[:Z] + u[1] * PAULIS[:X] + u[2] * PAULIS[:Y]
system = QuantumSystem(H, 10.0, [(-1.0, 1.0), (-1.0, 1.0)])
system.H([1.0, 0.0], 0.0) |> sparse

#=
## Open quantum systems

We can also construct an [`OpenQuantumSystem`](@ref) with Lindblad dynamics, enabling
a user to pass a list of dissipation operators.

```@docs; canonical = false
OpenQuantumSystem
```
=#

# _Add a dephasing and annihilation error channel._
H_drives = [PAULIS[:X]]
a = annihilate(2)
dissipation_operators = [a'a, a]
T_max = 10.0
drive_bounds = [(-1.0, 1.0)]
system = OpenQuantumSystem(H_drives, T_max, drive_bounds, dissipation_operators=dissipation_operators)
system.dissipation_operators[1] |> sparse

# 
system.dissipation_operators[2] |> sparse

#=
!!! warning
    The Hamiltonian part `system.H` excludes the Lindblad operators. This is also true
    for functions that report properties of `system.H`, such as [`get_drift`](@ref), 
    [`get_drives`](@ref), and [`is_reachable`](@ref).
=#

get_drift(system) |> sparse

#=
## Composite quantum systems

A [`CompositeQuantumSystem`](@ref) is constructed from a list of subsystems and their 
interactions. The interaction, in the form of drift or drive Hamiltonian, acts on the full
Hilbert space. The subsystems, with their own drift and drive Hamiltonians, are internally
lifted to the full Hilbert space.

=#

system_1 = QuantumSystem([PAULIS[:X]], 1.0, [(-1.0, 1.0)])
system_2 = QuantumSystem([PAULIS[:Y]], 1.0, [(-1.0, 1.0)])
H_drift = PAULIS[:Z] ⊗ PAULIS[:Z]
system = CompositeQuantumSystem(H_drift, Matrix{ComplexF64}[], [system_1, system_2], 1.0, Float64[]);

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


#=
# Reachability tests

Whether a quantum system can be used to reach a target state or operator can be tested
by computing the dynamical Lie algebra. Access to this calculation is provided by the 
[`is_reachable`](@ref) function.
```@docs; canonical = false
is_reachable
```
=#

# _Y can be reached by commuting Z and X._
system = QuantumSystem(PAULIS[:Z], [PAULIS[:X]], 1.0, [(-1.0, 1.0)])
is_reachable(PAULIS[:Y], system)

# _Y cannot be reached by X alone._
system = QuantumSystem([PAULIS[:X]], 1.0, [(-1.0, 1.0)])
is_reachable(PAULIS[:Y], system)

#=
# Direct sums

The direct sum of two quantum systems is constructed with the [`direct_sum`](@ref) function.
```@docs; canonical = false
direct_sum
```
=#

# _Create a pair of non-interacting qubits._
system_1 = QuantumSystem(PAULIS[:Z], [PAULIS[:X], PAULIS[:Y]], 1.0, [(-1.0, 1.0), (-1.0, 1.0)])
system_2 = QuantumSystem(PAULIS[:Z], [PAULIS[:X], PAULIS[:Y]], 1.0, [(-1.0, 1.0), (-1.0, 1.0)])
system = direct_sum(system_1, system_2)
get_drift(system) |> sparse
