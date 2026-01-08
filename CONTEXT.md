# PiccoloQuantumObjects.jl Context

> AI-friendly context for maintaining consistency. Update this when making significant changes.

## Package Purpose

PiccoloQuantumObjects.jl provides the **quantum physics layer** for the Piccolo ecosystem. It defines:
- **QuantumSystem** - Hamiltonian structure and dynamics
- **Pulses** - Control pulse representations (piecewise constant, spline, analytic)
- **Quantum Trajectories** - Physics-level problem definitions (before optimization)
- **Isomorphisms** - Mappings between complex quantum objects and real vectors
- **Rollouts** - ODE-based simulation of quantum dynamics
- **Quantum System Templates** - Pre-built systems (transmons, ions, Rydberg, etc.)

This package is **independent of optimization** - it just does quantum physics. QuantumCollocation.jl builds optimization problems on top of these abstractions.

## Canonical Workflow

### 1. Create a Quantum System

```julia
using PiccoloQuantumObjects

# From drift + drives (duration comes from pulse)
H_drift = GATES[:Z]  # or PAULIS[:Z]
H_drives = [GATES[:X], GATES[:Y]]
drive_bounds = [1.0, 1.0]  # symmetric bounds → [(-1.0, 1.0), (-1.0, 1.0)]
sys = QuantumSystem(H_drift, H_drives, drive_bounds)

# From Hamiltonian function (for complex/time-dependent systems)
ω = 2π * 1.0
H(u, t) = GATES[:Z] + u[1] * cos(ω * t) * GATES[:X] + u[2] * sin(ω * t) * GATES[:Y]
sys = QuantumSystem(H, drive_bounds; time_dependent=true)

# From template
sys = TransmonSystem(ω_q=5.0, α=-0.2, n_levels=3; drive_bound=0.1)
```

### 2. Create a Pulse

```julia
N = 51  # number of sample points
times = collect(range(0.0, T, length=N))
controls = 0.1 * randn(sys.n_drives, N)

# Piecewise constant (zero-order hold)
pulse = ZeroOrderPulse(controls, times)

# Linear interpolation
pulse = LinearSplinePulse(controls, times)

# Cubic Hermite spline (with derivatives)
derivatives = zeros(sys.n_drives, N)
pulse = CubicSplinePulse(controls, derivatives, times)

# Sample pulse at any time
u_at_5 = pulse(5.0)
```

### 3. Create a Quantum Trajectory

```julia
# Unitary gate synthesis
U_goal = GATES[:H]
qtraj = UnitaryTrajectory(sys, pulse, U_goal)

# Or with auto-generated zero pulse
qtraj = UnitaryTrajectory(sys, U_goal, T)

# State transfer
ψ_init = ComplexF64[1.0, 0.0]
ψ_goal = ComplexF64[0.0, 1.0]
qtraj = KetTrajectory(sys, pulse, ψ_init, ψ_goal)

# Ensemble (multiple states, shared controls)
qtraj = EnsembleKetTrajectory(sys, pulse, [ψ0, ψ1], [ψ1, ψ0])

# Sample trajectory at any time
U_at_5 = qtraj(5.0)  # returns unitary/ket at t=5
```

### 4. Compute Fidelity

```julia
using PiccoloQuantumObjects: fidelity

# Fidelity of current pulse
fid = fidelity(qtraj)

# Direct unitary fidelity
U_final = qtraj(T)
fid = unitary_fidelity(U_final, U_goal)

# Ket fidelity
ψ_final = qtraj(T)
fid = abs2(ψ_final' * ψ_goal)
```

### 5. Convert to NamedTrajectory (for optimization)

```julia
using NamedTrajectories

# Convert quantum trajectory to optimization trajectory
N = 51
traj = NamedTrajectory(qtraj, N)

# Now has components:
# - :Ũ⃗ (or :ψ̃) - isomorphism-vectorized state
# - :u - controls
# - :Δt - timesteps
# - :t - accumulated time
```

## Key Abstractions

### QuantumSystem

```julia
struct QuantumSystem{F1, F2} <: AbstractQuantumSystem
    H::F1                          # Hamiltonian function (u, t) -> H
    G::F2                          # Generator function (u, t) -> -i*H (isomorphic)
    H_drift::SparseMatrixCSC       # Drift Hamiltonian
    H_drives::Vector{SparseMatrix} # Drive Hamiltonians (empty if function-based)
    drive_bounds::Vector{Tuple}    # [(lower, upper), ...]
    n_drives::Int                  # Number of control drives
    levels::Int                    # Hilbert space dimension
    time_dependent::Bool           # Has explicit time dependence
end
```

**Constructors:**
- `QuantumSystem(H_drift, H_drives, drive_bounds)` - Full system with drift and drives
- `QuantumSystem(H_drives, drive_bounds)` - No drift
- `QuantumSystem(H_drift)` - Drift only, no drives
- `QuantumSystem(H::Function, drive_bounds; time_dependent=false)` - Function-based Hamiltonian
- `OpenQuantumSystem(...)` - For Lindbladian dynamics (density matrices)

### Pulses

```
AbstractPulse
├── ZeroOrderPulse        # Piecewise constant (zero-order hold)
├── AbstractSplinePulse
│   ├── LinearSplinePulse  # Linear interpolation
│   └── CubicSplinePulse   # Cubic Hermite spline
└── GaussianPulse          # Analytic Gaussian envelope
```

**Key functions:**
- `pulse(t)` - Sample pulse at time t (all pulses are callable)
- `duration(pulse)` - Total duration
- `n_drives(pulse)` - Number of control channels
- `drive_name(pulse)` - Symbol for control variable (default: `:u`)
- `sample(pulse, times)` - Sample at multiple times

**CubicSplinePulse** stores both values AND derivatives - the `:du` component represents Hermite tangents, which are true degrees of freedom.

### Quantum Trajectories

```
AbstractQuantumTrajectory{P<:AbstractPulse}
├── UnitaryTrajectory{P}       # Full unitary evolution, goal is operator
├── KetTrajectory{P}           # Single state, goal is ket
├── EnsembleKetTrajectory{P}   # Multiple states, shared controls
├── DensityTrajectory{P}       # Density matrix evolution
└── SamplingTrajectory{P,Q}    # Wraps trajectory for robustness
```

**Key functions:**
| Function | Returns | Description |
|----------|---------|-------------|
| `get_system(qtraj)` | `QuantumSystem` | The quantum system |
| `get_pulse(qtraj)` | `AbstractPulse` | The control pulse |
| `get_initial(qtraj)` | `Matrix`/`Vector` | Initial state |
| `get_goal(qtraj)` | `Matrix`/`Vector` | Target state |
| `get_solution(qtraj)` | `ODESolution` | Pre-computed ODE solution |
| `state_name(qtraj)` | `Symbol` | State component name |
| `state_names(qtraj)` | `Vector{Symbol}` | All state names (ensemble) |
| `drive_name(qtraj)` | `Symbol` | Control component name |
| `duration(qtraj)` | `Float64` | Total duration |
| `fidelity(qtraj)` | `Float64` | Fidelity to goal |
| `qtraj(t)` | state | Sample at time t |

### Isomorphisms

Map between complex quantum objects and real vectors for optimization.

| Function | Input | Output | Use |
|----------|-------|--------|-----|
| `ket_to_iso(ψ)` | `Vector{Complex}` | `Vector{Real}` | Vectorize ket |
| `iso_to_ket(ψ̃)` | `Vector{Real}` | `Vector{Complex}` | Reconstruct ket |
| `operator_to_iso_vec(U)` | `Matrix{Complex}` | `Vector{Real}` | Vectorize operator |
| `iso_vec_to_operator(Ũ⃗)` | `Vector{Real}` | `Matrix{Complex}` | Reconstruct operator |

**Convention:** 
- Ket: `ψ̃ = [real(ψ); imag(ψ)]` (length 2n)
- Operator: Column-major with `[Re; Im]` per column (length 2n²)

### State Naming

| Trajectory Type | state_name | state_names |
|-----------------|------------|-------------|
| `UnitaryTrajectory` | `:Ũ⃗` | N/A |
| `KetTrajectory` | `:ψ̃` | N/A |
| `EnsembleKetTrajectory` | `:ψ̃` (prefix) | `[:ψ̃1, :ψ̃2, ...]` |
| `DensityTrajectory` | `:ρ⃗̃` | N/A |
| `SamplingTrajectory` | from base | indexed versions |

### NamedTrajectory Conversion

```julia
traj = NamedTrajectory(qtraj, N)  # N timesteps

# Components created:
# - state_name(qtraj) → isomorphism-vectorized state at each timestep
# - drive_name(qtraj) → control values (default :u)
# - :Δt → timestep durations
# - :t → accumulated time (always present!)
```

**Important:** The `:t` component is ALWAYS added. This makes `time_dependent=true` on QuantumSystem unnecessary in most cases.

## Quantum System Templates

Pre-built systems in `src/quantum_system_templates/`:

| Template | Use Case |
|----------|----------|
| `TransmonSystem` | Superconducting transmon qubit |
| `TwoTransmonSystem` | Two-qubit transmon system |
| `IonChainSystem` | Trapped ion chain (Molmer-Sorensen) |
| `RydbergChainSystem` | Rydberg atom array |
| `CatQubitSystem` | Bosonic cat qubit |

## File Structure

```
src/
├── PiccoloQuantumObjects.jl   # Main module
├── gates.jl                   # GATES, PAULIS constants
├── isomorphisms.jl            # Complex ↔ Real mappings
├── pulses.jl                  # Pulse types
├── rollouts.jl                # ODE problems and solutions
├── quantum_object_utils.jl    # Utility functions
├── quantum_systems/
│   ├── _quantum_systems.jl    # Submodule
│   ├── quantum_systems.jl     # QuantumSystem, OpenQuantumSystem
│   └── variational.jl         # VariationalQuantumSystem
├── quantum_trajectories/
│   ├── _quantum_trajectories.jl  # Submodule
│   ├── abstract_trajectory.jl    # AbstractQuantumTrajectory
│   ├── unitary_trajectory.jl     # UnitaryTrajectory
│   ├── ket_trajectory.jl         # KetTrajectory
│   ├── ensemble_trajectory.jl    # EnsembleKetTrajectory
│   ├── density_trajectory.jl     # DensityTrajectory
│   ├── sampling_trajectory.jl    # SamplingTrajectory
│   ├── interface.jl              # Common interface methods
│   ├── rebuild.jl                # Rebuild from optimized trajectory
│   └── named_trajectory_conversion.jl  # NamedTrajectory(qtraj, N)
├── quantum_system_templates/
│   ├── transmons.jl           # TransmonSystem
│   ├── ions.jl                # IonChainSystem
│   ├── rydberg.jl             # RydbergChainSystem
│   └── cats.jl                # CatQubitSystem
├── embedded_operators.jl      # EmbeddedOperator for subspace gates
├── lifted_operators.jl        # Lift operators to larger Hilbert space
└── direct_sums.jl             # Direct sum operations
```

## Testing Conventions

Tests use `@testitem` blocks in source files:
```julia
@testitem "descriptive name" begin
    using PiccoloQuantumObjects
    using LinearAlgebra
    # ... test code
end
```

## Recent Changes (Update This!)

### January 2026
- Quantum trajectories are parametric: `AbstractQuantumTrajectory{P<:AbstractPulse}`
- `NamedTrajectory(qtraj, N)` always includes `:t` component
- Control name is canonically `:u` (configured via `drive_name` kwarg on pulses)
- `rebuild(qtraj, traj)` reconstructs quantum trajectory from optimized NamedTrajectory

### Key Patterns
- All pulses are callable: `pulse(t)` returns control vector
- All trajectories are callable: `qtraj(t)` returns state at time t
- Trajectories pre-compute ODE solution at construction
- Fidelity is computed via `fidelity(qtraj)` method

## Common Gotchas

1. **Trajectories compute ODE at construction** - Changing the pulse requires creating a new trajectory
2. **Isomorphism convention** - `[real; imag]` stacking, column-major for operators
3. **drive_bounds normalization** - Scalars become symmetric bounds: `1.0 → (-1.0, 1.0)`
4. **time_dependent flag** - Only needed for explicit time dependence like `cos(ωt)` in the Hamiltonian function. The `:t` trajectory component is always present regardless.
5. **EnsembleKetTrajectory state names** - Uses `:ψ̃1`, `:ψ̃2`, etc., not `:ψ̃` directly
6. **CubicSplinePulse** - The `:du` component is a true DOF (Hermite tangent), not a derived quantity
