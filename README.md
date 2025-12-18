# Hierarchical Contextual Global Optimization (HC-GO)

## Mitigating Barren Plateaus in Variational Quantum Algorithms

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Qiskit](https://img.shields.io/badge/qiskit-1.0+-purple.svg)](https://qiskit.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the implementation of **Hierarchical Contextual Global Optimization (HC-GO)**, a novel algorithmic framework for variational quantum algorithms that resolves the barren plateau problem while preserving true global optimization objectives.

---

## Overview

Variational Quantum Algorithms (VQAs) like VQE and QAOA are promising approaches for near-term quantum advantage, but they suffer from the **barren plateau phenomenon**—exponentially vanishing gradients that render large-scale optimization infeasible. HC-GO addresses this fundamental challenge through three key innovations:

1. **Quantum Context Register (QCR)**: A small auxiliary register carrying compressed global correlations through sequential isometric cells.

2. **Operator Push-Through**: Exact transformation of global observables into equivalent compressed-register observables.

3. **Classical Shadow Tomography**: Efficient measurement of the global cost function on polynomially-sized compressed registers.

The result is gradient variances that scale with local subsystem dimensions (polynomial) rather than total system size (exponential).

---

## Key Features

- **Barren Plateau Mitigation**: Gradients scale as O(2^{-(b+k)}) with local block and context dimensions, not O(2^{-N}) with total qubits
- **True Global Cost Preservation**: No approximation—the compressed measurement yields exact global expectation values
- **DMRG-Style Training**: Local sweep optimization avoiding random global initialization
- **Adaptive Bond Growth**: Automatic increase of expressivity only where entanglement demands it
- **Modular Implementation**: Clean separation of circuit construction, operator contraction, shadow measurement, and training logic

---


## Core Components

### 1. BarrenPlateauCircuit

Hardware-efficient ansatz implementation for demonstrating the barren plateau phenomenon.

### 2. HCGOAnsatz

Sequential isometric cell architecture with Quantum Context Register.

**Architecture Details:**
- Each cell acts on (b + k) qubits
- Uses `RealAmplitudes` parameterization from Qiskit
- Small random initialization (0, 0.1) to avoid 2-design behavior

### 3. PauliContraction

Implements the PUSH algorithm for backward operator contraction.

**Algorithm:**
1. Construct total operator on (k + b) qubits
2. Conjugate by cell unitary: Ũ = U†OU
3. Partial trace over physical block to obtain compressed operator

### 4. CompressedShadow

Classical shadow tomography engine for efficient expectation value estimation.

**Shadow Protocol:**
- For each snapshot: apply random X, Y, or Z basis rotation to each QCR qubit
- Measure in computational basis
- Store (basis, outcome) pairs
- Use shadow reconstruction formula with 3^|P| normalization

---

## Mathematical Framework

### Generated State

The HC-GO ansatz prepares states of the form:

```
|Ψ(Θ)⟩ = U_HC(Θ)|0⟩ = ∏_{i=1}^{N_b} U_i(θ^{(i)})|0⟩
```

### Compression Equivalence

For any global observable O_global:

```
E(Θ) = ⟨Ψ(Θ)|O_global|Ψ(Θ)⟩ = Tr(ρ_comp · O_comp)
```

where `ρ_comp` is the reduced state on QCR and `O_comp` is the pushed-through operator.

### Gradient Scaling

For HC-GO with local sweep optimization:

```
Var(∂E/∂θ) ≥ Ω(2^{-(b+k)})
```

independent of total system size N.

---

## License

This project is licensed under the MIT License. See LICENSE file for details.

---

## Contact

**Vitthal Saxena**  
Department of Quantum Information Science  
Indiana University, Bloomington, IN 47405  
Email: visaxen@iu.edu

---
