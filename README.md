<!--```@raw html-->
<div align="center">
  <a href="https://github.com/harmoniqs/Piccolo.jl">
    <img src="assets/logo.svg" alt="Piccolo.jl" width="25%"/>
  </a> 
</div>

<div align="center">
  <table>
    <tr>
      <td align="center">
        <b>Documentation</b>
        <br>
        <a href="https://docs.harmoniqs.co/PiccoloQuantumObjects/stable/">
          <img src="https://img.shields.io/badge/docs-stable-blue.svg" alt="Stable"/>
        </a>
        <a href="https://docs.harmoniqs.co/PiccoloQuantumObjects/dev/">
          <img src="https://img.shields.io/badge/docs-dev-blue.svg" alt="Dev"/>
        </a>
      </td>
      <td align="center">
        <b>Build Status</b>
        <br>
        <a href="https://github.com/harmoniqs/PiccoloQuantumObjects.jl/actions/workflows/CI.yml?query=branch%3Amain">
          <img src="https://github.com/harmoniqs/PiccoloQuantumObjects.jl/actions/workflows/CI.yml/badge.svg?branch=main" alt="Build Status"/>
        </a>
        <a href="https://codecov.io/gh/harmoniqs/PiccoloQuantumObjects.jl">
          <img src="https://codecov.io/gh/harmoniqs/PiccoloQuantumObjects.jl/branch/main/graph/badge.svg" alt="Coverage"/>
        </a>
      </td>
      <td align="center">
        <b>License</b>
        <br>
        <a href="https://opensource.org/licenses/MIT">
          <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="MIT License"/>
        </a>
      </td>
      <td align="center">
        <b>Support</b>
        <br>
        <a href="https://unitary.fund">
          <img src="https://img.shields.io/badge/Supported%20By-Unitary%20Fund-FFFF00.svg" alt="Unitary Fund"/>
        </a>
      </td>
    </tr>
  </table>
</div>

<div align="center">
<i> Make simple transformation of complex objects for quantum numerics </i>
<br>

</div>
<!--```-->

# PiccoloQuantumObjects

**PiccoloQuantumObjects.jl** is a Julia package for working with quantum objects. It provides tools for constructing and manipulating quantum states and operators. It is designed to be used with other packages in the [Piccolo.jl](https://github.com/harmoniqs/Piccolo.jl) ecosystem, such as [QuantumCollocation.jl](https://github.com/harmoniqs/QuantumCollocation.jl) and [NamedTrajectories.jl](https://github.com/harmoniqs/NamedTrajectories.jl).

### Installation

This package is registered! To install, enter the Julia REPL, type `]` to enter pkg mode, and then run:
```julia
pkg> add PiccoloQuantumObjects
```

### Usage

The following example demonstrates how to create a quantum state, create a quantum operator, and apply the operator to the state:

```Julia
using PiccoloQuantumObjects

# Create a quantum state
state = ket_from_string("g", [2])

# Create a quantum operator
operator = PAULIS.X

# Apply the operator to the state
new_state = operator * state

# Transform the state to its real representation
new_iso_state = ket_to_iso(new_state)

# Transform back
iso_to_ket(new_iso_state)
```
