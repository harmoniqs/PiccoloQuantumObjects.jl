# ```@meta
# CollapsedDocStrings = true
# ```

# # `Open Quantum Systems`

using PiccoloQuantumObjects
using SparseArrays # for visualization

#=

## Open quantum systems

We can construct an [`OpenQuantumSystem`](@ref) with Lindblad dynamics, enabling
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
