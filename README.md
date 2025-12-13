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

## Installation

### Prerequisites

- Python 3.8 or higher
- Jupyter Notebook (for running the demonstration)

### Required Packages

```bash
pip install numpy matplotlib qiskit qiskit-aer scipy
```

Alternatively, run the first cell of the notebook:

```python
%pip install numpy matplotlib qiskit qiskit-aer scipy --quiet
```

---

## Repository Structure

```
HC-GO/
├── BarrenPlateaus.ipynb      # Main Jupyter notebook with full implementation
├── README.md                  # This file
├── HC-GO_Research_Paper.docx # Full research paper
└── Barren_plateaus_notes.docx # Detailed technical notes
```

---

## Quick Start

### Running the Notebook

1. Clone or download this repository
2. Open `BarrenPlateaus.ipynb` in Jupyter Notebook or Google Colab
3. Run all cells sequentially

### Basic Usage Example

```python
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp

# Create HC-GO ansatz
ansatz = HCGOAnsatz(
    num_blocks=4,      # Number of physical blocks
    block_size=2,      # Qubits per block
    context_size=2     # QCR qubits
)

# Define local Hamiltonians
hamiltonians = [generate_local_hamiltonian(block_size=2) for _ in range(4)]

# Run training
final_params, gradient_norms = run_hcgo_training(
    global_hamiltonian_terms=hamiltonians,
    num_blocks=4,
    block_size=2,
    context_size=2,
    iterations=5
)
```

---

## Core Components

### 1. BarrenPlateauCircuit

Hardware-efficient ansatz implementation for demonstrating the barren plateau phenomenon.

```python
class BarrenPlateauCircuit:
    def __init__(self, num_qubits, num_layers, entanglement='full'):
        """
        Parameters:
            num_qubits: Number of qubits in the circuit
            num_layers: Depth of the variational circuit
            entanglement: Pattern - 'full', 'linear', or 'circular'
        """
```

**Key Methods:**
- `_build_circuit()`: Constructs parameterized circuit with rotation and entanglement layers
- `get_circuit(params)`: Returns circuit with bound parameters

### 2. HCGOAnsatz

Sequential isometric cell architecture with Quantum Context Register.

```python
class HCGOAnsatz:
    def __init__(self, num_blocks, block_size, context_size):
        """
        Parameters:
            num_blocks: Number of physical blocks (N_b)
            block_size: Qubits per block (b)
            context_size: QCR qubits (k)
        """
```

**Architecture Details:**
- Each cell acts on (b + k) qubits
- Uses `RealAmplitudes` parameterization from Qiskit
- Small random initialization (0, 0.1) to avoid 2-design behavior

### 3. PauliContraction

Implements the PUSH algorithm for backward operator contraction.

```python
class PauliContraction:
    @staticmethod
    def conjugate_and_trace(
        operator_from_future: SparsePauliOp,
        local_hamiltonian: SparsePauliOp,
        unitary_circuit: QuantumCircuit,
        num_qcr: int,
        num_block: int
    ) -> SparsePauliOp:
        """
        Computes O_in = Tr_Block[U†(O_future ⊗ I_block + I_qcr ⊗ H_local)U]
        """
```

**Algorithm:**
1. Construct total operator on (k + b) qubits
2. Conjugate by cell unitary: Ũ = U†OU
3. Partial trace over physical block to obtain compressed operator

### 4. CompressedShadow

Classical shadow tomography engine for efficient expectation value estimation.

```python
class CompressedShadow:
    def __init__(self, k_qubits, num_snapshots):
        """
        Parameters:
            k_qubits: Number of QCR qubits to measure
            num_snapshots: Number of randomized measurements
        """
```

**Key Methods:**
- `collect_shadows(state_circuit)`: Performs random Pauli measurements
- `estimate_expectation(shadow_data, observable)`: Median-of-means estimation

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

## Experiments and Results

### 1. Barren Plateau Demonstration

The notebook demonstrates exponential gradient decay in hardware-efficient ansätze:

```python
# Analyze gradient scaling
results = analyze_barren_plateaus(
    num_qubits=[4, 6, 8],
    num_layers=[2, 4, 8, 12, 16],
    num_samples=50,
    entanglement='linear'
)
```

**Expected Results:**
- Gradient variance decays as ~2^{-αL} with α ≈ 0.3-0.5
- 8 qubits at depth 16: gradient norms < 10^{-3}

### 2. HC-GO Mitigation Analysis

```python
# Analyze HC-GO gradient behavior
hcgo_results = analyze_hcgo_mitigation(
    num_qubits=[4, 6, 8],
    block_size=2,
    context_size=2,
    iterations=5
)
```

**Expected Results:**
- Gradient variance remains O(1) across all system sizes
- No exponential suppression with increasing N
- Scaling determined by 2^{-(b+k)} = 2^{-4} ≈ 0.0625

### 3. Comparative Visualization

```python
plot_comparative_gradients_line_graph(
    barren_plateau_results,
    hcgo_analysis_results,
    num_layers_bp=[2, 4, 8, 12, 16],
    block_size_hcgo=2
)
```

Generates side-by-side comparison showing:
- Exponential decay in HEA (unmitigated)
- Stable gradients in HC-GO (mitigated)

---

## Configuration Parameters

### Circuit Design

| Parameter | Description | Recommended Range |
|-----------|-------------|-------------------|
| `block_size` (b) | Physical qubits per block | 2-4 |
| `context_size` (k) | QCR qubits | 1-4 |
| `num_layers` | Depth per cell | 1-3 |

**Design Constraint:** Keep b + k ≤ 12 for tractable classical push-through.

### Shadow Measurement

| Parameter | Description | Typical Value |
|-----------|-------------|---------------|
| `num_snapshots` | Randomized measurements | 200-2000 |
| `k_means` | Median-of-means partitions | 10 |

**Sample Complexity:** N_snap = O(3^k log(M/δ)/ε²)

### Training

| Parameter | Description | Default |
|-----------|-------------|---------|
| `iterations` | Training epochs | 5-20 |
| `learning_rate` | Gradient step size | 0.1 |
| `S_threshold` | Bond growth threshold | 0.05-0.15 |

---

## Generating Local Hamiltonians

The `generate_local_hamiltonian` function creates 1D Ising-like Hamiltonians:

```python
def generate_local_hamiltonian(block_size: int) -> SparsePauliOp:
    """
    Generates H_local = Σ_i Z_i Z_{i+1} + 0.5 Σ_i X_i
    
    Returns SparsePauliOp for the specified block size.
    """
```

This can be modified for other Hamiltonians (Heisenberg, molecular, etc.).

---

## Extending the Implementation

### Custom Hamiltonians

Replace `generate_local_hamiltonian` with your target:

```python
def my_hamiltonian(block_size):
    # Heisenberg model example
    labels = []
    coeffs = []
    for i in range(block_size - 1):
        for pauli in ['XX', 'YY', 'ZZ']:
            term = ['I'] * block_size
            term[i], term[i+1] = pauli[0], pauli[1]
            labels.append(''.join(term))
            coeffs.append(1.0)
    return SparsePauliOp(labels, coeffs)
```

### Hardware Deployment

For real quantum hardware:

1. Replace `StatevectorSampler` with hardware `Sampler`
2. Add error mitigation (zero-noise extrapolation, etc.)
3. Implement readout calibration on compressed registers
4. Consider native gate decomposition for cells

### Adaptive Bond Growth

Implement Rényi-2 entropy monitoring:

```python
def estimate_renyi2(shadow_data, bond_qubits):
    """
    Estimate S_2 = -log(Tr(ρ²)) using randomized measurements
    """
    # Purity estimation from shadows
    purity = estimate_purity(shadow_data, bond_qubits)
    return -np.log(purity)
```

---

## Troubleshooting

### Common Issues

**1. "Operator dimension mismatch"**
- Ensure `block_size + context_size` matches circuit qubit count
- Verify SparsePauliOp labels have correct length

**2. "Gradient explosion"**
- Reduce learning rate
- Check parameter initialization (should be small, ~0.1)
- Verify PUSH algorithm producing valid operators

**3. "Shadow estimation variance too high"**
- Increase `num_snapshots`
- Reduce observable weight in O_comp
- Check for numerical instabilities in partial trace

### Performance Tips

- Use `simplify()` on SparsePauliOp after contractions
- Cache cell unitaries between gradient computations
- Batch shadow collection for multiple gradient components

---

## Citation

If you use this code in your research, please cite:

```bibtex
@article{saxena2025hcgo,
  title={Hierarchical Contextual Global Optimization: Mitigating Barren Plateaus via Quantum Context Registers and Classical Shadow Tomography},
  author={Saxena, Vitthal},
  journal={arXiv preprint},
  year={2025}
}
```

---

## References

1. McClean et al., "Barren plateaus in quantum neural network training landscapes," Nat. Commun. 9, 4812 (2018)
2. Huang, Kueng, Preskill, "Predicting many properties of a quantum system from very few measurements," Nat. Phys. 16, 1050 (2020)
3. Miao & Barthel, "Isometric tensor network optimization for extensive Hamiltonians is free of barren plateaus," Phys. Rev. A 109, L050402 (2024)
4. Cerezo et al., "Cost function dependent barren plateaus in shallow parametrized quantum circuits," Nat. Commun. 12, 1791 (2021)

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

## Acknowledgments

- IBM Qiskit development team for the quantum computing framework
- The broader quantum computing community for foundational work on variational algorithms and barren plateaus
