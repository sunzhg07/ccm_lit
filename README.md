# Coupled Cluster Method for Nuclear Structure (ccm_lit)

A Python implementation of Coupled Cluster methods (CCSD, CCSDT) and their Lambda equation counterparts ($\Lambda$-CCSD, $\Lambda$-CCSDT) for nuclear structure calculations. This code is designed to read nuclear interaction files in SNT format, perform Hartree-Fock computations, and solve the Coupled Cluster equations to determine ground state energies and properties.

## Features

-   **Input:** Reads standard `.snt` interaction files (e.g., `usdb.snt`).
-   **Mean Field:** Restricted Hartree-Fock (HF) solver for spherical nuclei.
-   **Normal Ordering:** Transformation of the Hamiltonian to the Hartree-Fock basis and normal ordering with respect to the HF reference.
-   **Correlation Methods:**
    -   **MP2:** Møller-Plesset perturbation theory (2nd order).
    -   **CCD:** Coupled Cluster Doubles.
    -   **CCSD:** Coupled Cluster Singles and Doubles.
    -   **CCSDT:** Coupled Cluster Singles, Doubles, and Triples (full iterative solution).
-   **Properties:**
    -   **Lambda Equations:** Solution of the left-hand eigenvector equations ($\Lambda$-CCSD, $\Lambda$-CCSDT).
    -   **Density Matrices:** Computation of 1-body and 2-body density matrices.
    -   **Observables:** Particle number trace verification, one-body expectation values.
-   **Analysis Tools:**
    -   Similarity transformations of operators ($e^{-T} H e^T$).
    -   M-scheme basis generation and decoupling.

## Project Structure

-   `main.py`: The entry point. Orchestrates the workflow from file reading to CCSD/CCSDT execution and property calculation.
-   `read_snt_io.py`: Handles parsing of SNT files and transformation of J-scheme matrix elements to M-scheme.
-   `hf.py`: Implements the Hartree-Fock self-consistent field procedure.
-   `cc.py`: Contains the iterative solvers for CCSD and CCSDT amplitudes ($T_1, T_2, T_3$).
-   `lambda_ccsd.py`: Solves the $\Lambda_1$ and $\Lambda_2$ equations for CCSD properties.
-   `lambda_ccsdt.py`: Solves the $\Lambda$-CCSDT equations including $\Lambda_3$.
-   `similarity_transform.py`: Utilities for computing similarity transformed operators (e.g., $\bar{H}$).

## Theoretical Background and Equations

The code solves the Schrödinger equation $H |\Psi\rangle = E |\Psi\rangle$ using the exponential ansatz:
$$ |\Psi\rangle = e^{\hat{T}} |\Phi_0\rangle $$
where $|\Phi_0\rangle$ is the Hartree-Fock reference determinant.

### 1. The Hamiltonian
The Hamiltonian is normal-ordered with respect to the Fermi vacuum $|\Phi_0\rangle$:
$$ H_N = \sum_{ij} f_{ij} \{c_i^\dagger c_j\} + \frac{1}{4} \sum_{ijkl} \bar{v}_{ijkl} \{c_i^\dagger c_j^\dagger c_l c_k\} $$
where $f_{ij}$ is the Fock matrix and $\bar{v}_{ijkl} = \langle ij | V | kl \rangle_{as}$ are the antisymmetrized two-body integrals.

### 2. Coupled Cluster Energy (CCSD/CCSDT)
The correlation energy is uniquely determined by the singles and doubles amplitudes ($T_1, T_2$), even in CCSDT, via the projection onto the reference:
$$ E_{corr} = \langle \Phi_0 | H_N e^T | \Phi_0 \rangle $$
$$ E_{corr} = \sum_{ia} f_{ia} t_i^a + \frac{1}{4} \sum_{ijab} \bar{v}_{ijab} t_{ij}^{ab} + \frac{1}{2} \sum_{ijab} \bar{v}_{ijab} t_i^a t_j^b $$
*(Indices $i,j,k...$ denote occupied orbitals (holes), and $a,b,c...$ denote virtual orbitals (particles).)*

### 3. Amplitude Equations
The amplitudes are determined by projecting the similarity transformed Hamiltonian $\bar{H} = e^{-T} H e^T$ onto excited determinants:
$$ \langle \Phi_\mu | \bar{H} | \Phi_0 \rangle = 0 $$

#### CCSD Residuals ($R_1, R_2$)
The iterative equations solve for $R_1 = 0$ and $R_2 = 0$.
The solver uses a simplified update step based on Jacobi iteration:
$$ T_{new} = T_{old} + R / D $$
where $D$ denotes the Møller-Plesset energy denominators.

**Singles ($T_1$) Equation:**
$$ R_i^a = f_{ai} + \sum_e f_{ae} t_i^e - \sum_m f_{mi} t_m^a + \sum_{me} \bar{v}_{maie} t_m^e + \dots $$

**Doubles ($T_2$) Equation:**
$$ R_{ij}^{ab} = \bar{v}_{ijab} + P(ab) \sum_e \bar{v}_{abie} t_j^e - P(ij) \sum_m \bar{v}_{mbij} t_m^a + \frac{1}{2} \sum_{ef} \bar{v}_{abef} t_{ij}^{ef} + \frac{1}{2} \sum_{mn} \bar{v}_{mnij} t_{mn}^{ab} + \dots $$

#### CCSDT Residual ($R_3$)
In CCSDT, $T_3$ ($t_{ijk}^{abc}$) is included. The $T_3$ amplitude equation includes contributions from $T_2$ (source term), $H_N$ contraction with $T_3$ (diagonal term), and $T_3$ self-interaction.
The dominant cost scales as $O(N^9)$ or $O(N_o^3 N_v^4)$ depending on implementation.

**$T_3$ contributions:**
$$ R_{ijk}^{abc} \leftarrow P(ijk)P(abc) \left[ \sum_e t_{ij}^{ae} \bar{v}_{ke bc} - \sum_m t_{im}^{ab} \bar{v}_{jk mc} \right] + \dots $$

### 4. Lambda Equations (Analytic Gradients)
To calculate properties other than energy, we solve for the de-excitation operator $\Lambda = \Lambda_1 + \Lambda_2 + \dots$.
The $\Lambda$ amplitudes constitute the left-hand eigenvector of $\bar{H}$:
$$ \langle \Phi_0 | (1+\Lambda) \bar{H} = E \langle \Phi_0 | (1+\Lambda) $$
Equivalently, the stationary condition for the Lagrangian $\mathcal{L} = \langle \Phi_0 | (1+\Lambda) e^{-T} H e^T | \Phi_0 \rangle$.

**Equation form:**
$$ \bar{H}^T \Lambda = E \Lambda $$
The residuals for $\Lambda$ are constructed from the transpose of the Jacobian of the CCSD(T) equations.

### 5. Density Matrices
Using converged $T$ and $\Lambda$, one-body density matrices ($\gamma$) are computed:
$$ \gamma_{qp} = \langle \Psi_L | c_p^\dagger c_q | \Psi_R \rangle $$
-   **Hole-Hole ($\rho_{ki}$):** $\delta_{ki} - \sum_e \lambda_{ke} t_i^e - \frac{1}{2} \sum_{mae} \lambda_{kmae} t_{imae} + \dots$
-   **Particle-Particle ($\rho_{ac}$):** $\sum_m \lambda_{mc} t_m^a + \frac{1}{2} \sum_{mne} \lambda_{mnce} t_{mnae} + \dots$

## Usage

Run the main script to execute the full pipeline:

```bash
python3 main.py
```

Ensure `usdb.snt` is in the working directory or update the path in `main.py`.

## Requirements
-   Python 3.x
-   NumPy
-   opt_einsum
