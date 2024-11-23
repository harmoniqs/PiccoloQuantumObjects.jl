
# ```@meta
# CollapsedDocStrings = true
# ```
using PiccoloQuantumObjects

#=
# Isomorphisms

## States

## Operators

## Density matrices

=#

# Julia uses column-major order.
U = [1 5; 2 6] + im * [3 7; 4 8]
operator_to_iso_vec(U)

#=
# Isomorphisms for dynamics

The isomorphism of a Hamiltonian ``H`` is:
```math
\text{iso}(H) := \widetilde{H} = \mqty(1 & 0 \\ 0 & 1) \otimes \Re(H) + \mqty(0 & -1 \\ 1 & 0) \otimes \Im(H)
```
where ``\Im(H)`` and ``\Re(H)`` are the imaginary and real parts of ``H`` and the tilde 
indicates the standard isomorphism of a complex valued matrix:
```math
\widetilde{H} := \mqty(1 & 0 \\ 0 & 1) \otimes \Re(H) + \mqty(0 & -1 \\ 1 & 0) \otimes \Im(H)
```

Therefore, the generator ``G`` associated to a Hamiltonian ``H`` is:
```math
G(H) := \text{iso}(- i \widetilde{H}) = \mqty(1 & 0 \\ 0 & 1) \otimes \Im(H) - \mqty(0 & -1 \\ 1 & 0) \otimes \Re(H)
```

=#
