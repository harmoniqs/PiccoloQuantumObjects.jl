# ```@meta
# CollapsedDocStrings = true
# ```

# # `Quantum Systems`

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
