# ```@meta
# CollapsedDocStrings = true
# ```

# # `Variational Quantum Systems`

using PiccoloQuantumObjects
using SparseArrays # for visualization

#=

## Variational Quantum Systems

The [`VariationalQuantumSystem`](@ref) type is used for sensitivity and robustness analysis
of quantum control protocols. It allows exploring how the dynamics change under perturbations
to the Hamiltonian.

Variational systems are parameterized by variational operators that represent directions 
of uncertainty or perturbation in the system:

```math
H_{\text{perturbed}}(\alpha) = H(\alpha) + \sum_i \epsilon_i H_{\text{var}}^{(i)}
```

where ``\alpha`` represents the control parameters and ``\epsilon_i`` are perturbation magnitudes.

```@docs; canonical = false
VariationalQuantumSystem
```

=#

# _Create a variational system with X and Y as both drives and variational directions._
H_drift = 0.0 * PAULIS[:Z]  # No drift
H_drives = [PAULIS[:X], PAULIS[:Y]]
H_vars = [PAULIS[:X], PAULIS[:Y]]
T_max = 10.0
drive_bounds = [(-1.0, 1.0), (-1.0, 1.0)]
varsys = VariationalQuantumSystem(H_drift, H_drives, H_vars, T_max, drive_bounds)

# _The system has 2 drives and 2 variational operators._
varsys.n_drives

#
length(varsys.G_vars)

# _Check T_max and drive_bounds._
varsys.T_max

#
varsys.drive_bounds

#=
## Variational operators

Variational systems compute the isomorphic generator `G` along with variational 
generators `G_vars` for sensitivity analysis. These can be used to study how 
the system dynamics change under perturbations.
=#

# _Evaluate the generator at specific control values._
control_params = [1.0, 0.5]
G = varsys.G(control_params)
G |> sparse

# _Get the first variational generator._
G_var_1 = varsys.G_vars[1](control_params)
G_var_1 |> sparse

# _Get the second variational generator._
G_var_2 = varsys.G_vars[2](control_params)
G_var_2 |> sparse

#=
## Use cases

Variational quantum systems are particularly useful for:

- **Sensitivity analysis**: Understanding how control imperfections affect gate fidelities
- **Robustness optimization**: Designing controls that are robust to parameter uncertainties
- **Error modeling**: Incorporating known sources of systematic error in control optimization

=#

#=
!!! note "No drift constructor"
    You can create a variational system with no drift Hamiltonian by omitting `H_drift`:
    ```julia
    varsys = VariationalQuantumSystem(H_drives, H_vars, T_max, drive_bounds)
    ```
=#

# _Create a variational system with only drive and variational operators._
varsys_nodrift = VariationalQuantumSystem([PAULIS[:X], PAULIS[:Y]], [PAULIS[:Z]], 10.0, [1.0, 1.0])
varsys_nodrift.n_drives

#
length(varsys_nodrift.G_vars)

#=
## Functional variational systems

For more complex scenarios, variational systems can be constructed from functions
rather than explicit matrix operators:
=#

# _Create a variational system using Hamiltonian functions._
H_func = a -> a[1] * PAULIS[:X] + a[2] * PAULIS[:Y]
H_var_funcs = [
    a -> a[1] * PAULIS[:X],  # Sensitivity to X scaling
    a -> PAULIS[:Z]           # Sensitivity to Z perturbation
]
varsys_func = VariationalQuantumSystem(H_func, H_var_funcs, 2, 10.0, [1.0, 1.0])

# _The functional system behaves similarly._
varsys_func.n_drives

#
length(varsys_func.G_vars)
