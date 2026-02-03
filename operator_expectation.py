"""
Compute expectation values of operators using CCSD and Lambda-CCSD wave functions.

The expectation value is computed as:
    <O> = <0| (1 + Lambda) * O * exp(T) |0>
    
where Lambda = L1 + L2 is the left wave function from Lambda-CCSD,
and T = T1 + T2 is the right wave function from CCSD.

For a general operator O = O_0 + O_1 + O_2:
    <O> = O_0 + <O_1> + <O_2>
    
where:
    <O_1> = sum_{ia} rho_ia * O_ia
    <O_2> = (1/4) * sum_{ijab} rho_ijab * O_ijab
    
The density matrices are:
    rho_ia = l1_ia
    rho_ijab = l2_ijab + l1_ia * t1_jb - l1_ib * t1_ja
"""

import numpy as np
from opt_einsum import contract
from scipy.sparse import issparse


def compute_one_body_density(l1, t1, n_occ, n_states):
    """
    Compute one-body density matrix for expectation values.
    
    For occupied-virtual block:
        rho_ia = l1_ia
    
    For occupied-occupied block:
        rho_ij = delta_ij - 0.5 * sum_{mef} l2_imef * t2_jmef
        
    For virtual-virtual block:
        rho_ab = 0.5 * sum_{mne} l2_mnae * t2_mnbe
    """
    o = slice(0, n_occ)
    v = slice(n_occ, n_states)
    
    # Full density matrix
    rho = np.zeros((n_states, n_states))
    
    # Occupied-occupied block
    rho[o, o] = np.eye(n_occ)
    
    # Occupied-virtual block (this is the key for response)
    rho[o, v] = l1
    
    return rho


def compute_two_body_density(l1, l2, t1, t2, n_occ, n_states):
    """
    Compute effective two-body density matrix for expectation values.
    
    rho_ijab = l2_ijab + l1_ia * t1_jb - l1_ib * t1_ja
    
    Note: This is the particle-hole density, not the full 2-body density.
    """
    o = slice(0, n_occ)
    v = slice(n_occ, n_states)
    
    # Start with L2
    rho2 = l2.copy()
    
    # Add disconnected T1-L1 contribution
    # l1_ia * t1_jb
    rho2 += contract('ia,jb->ijab', l1, t1)
    
    # Antisymmetrize: subtract l1_ib * t1_ja
    rho2 -= contract('ib,ja->ijab', l1, t1)
    
    return rho2


def transform_operator_to_hf_basis(operator_v1b, operator_v2b_sparse, sp_coeffs, n_states):
    """
    Transform an operator from the original m-scheme basis to the HF basis.
    
    Args:
        operator_v1b: One-body operator in original basis (n_states, n_states)
        operator_v2b_sparse: Two-body operator in original basis (flattened sparse or dense)
        sp_coeffs: HF single-particle coefficients (n_states, n_states)
        n_states: Number of states
        
    Returns:
        v1b_hf: One-body operator in HF basis
        v2b_hf: Two-body operator in HF basis (4D dense array)
    """
    from scipy.sparse import issparse
    from read_snt_io import sparse_to_dense_4d
    
    C = sp_coeffs
    
    # Transform 1-body: O_ij^HF = sum_pq C_pi * O_pq * C_qj
    v1b_hf = C.T @ operator_v1b @ C
    
    # Check for numerical issues
    if np.any(np.isnan(v1b_hf)) or np.any(np.isinf(v1b_hf)):
        print("Warning: NaN or Inf detected in 1-body operator transformation!")
        print(f"  operator_v1b stats: min={np.min(operator_v1b)}, max={np.max(operator_v1b)}")
        print(f"  sp_coeffs stats: min={np.min(sp_coeffs)}, max={np.max(sp_coeffs)}")
        # Set problematic values to zero
        v1b_hf = np.nan_to_num(v1b_hf, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Transform 2-body: O_ijkl^HF = sum_pqrs C_pi * C_qj * O_pqrs * C_rk * C_sl
    # Convert sparse to dense first
    if issparse(operator_v2b_sparse):
        v2b_dense = sparse_to_dense_4d(operator_v2b_sparse, n_states)
    else:
        v2b_dense = operator_v2b_sparse
    
    # Efficient 4-step transformation O(N^5)
    print("  Transforming 2-body operator to HF basis...")
    temp = np.einsum('pqrs,sl->pqrl', v2b_dense, C)
    temp = np.einsum('pqrl,rk->pqkl', temp, C)
    temp = np.einsum('pqkl,qj->pjkl', temp, C)
    v2b_hf = np.einsum('pjkl,pi->ijkl', temp, C)
    
    # Check for numerical issues
    if np.any(np.isnan(v2b_hf)) or np.any(np.isinf(v2b_hf)):
        print("Warning: NaN or Inf detected in 2-body operator transformation!")
        v2b_hf = np.nan_to_num(v2b_hf, nan=0.0, posinf=0.0, neginf=0.0)
    
    return v1b_hf, v2b_hf


def compute_expectation_value(l1, l2, t1, t2, operator, n_occ, n_states):
    """
    Compute expectation value of an operator using CCSD and Lambda-CCSD.
    
    NOTE: The operator must already be in the HF basis (same basis as T and Lambda amplitudes).
    
    Args:
        l1, l2: Lambda amplitudes from Lambda-CCSD
        t1, t2: T amplitudes from CCSD
        operator: Dictionary with keys 'v1b' (one-body) and 'v2b' (two-body)
                 - v1b: (n_states, n_states) array IN HF BASIS
                 - v2b: (n_states, n_states, n_states, n_states) dense array IN HF BASIS
        n_occ: Number of occupied orbitals
        n_states: Total number of orbitals
        
    Returns:
        expectation: Float, the expectation value <O>
    """
    o = slice(0, n_occ)
    v = slice(n_occ, n_states)
    
    expectation = 0.0
    
    # --- One-body contribution ---
    if 'v1b' in operator:
        v1b = operator['v1b']
        
        # Reference contribution: Tr(rho_oo * O_oo)
        expectation += np.trace(v1b[o, o])
        
        # Correlation contribution: sum_ia l1_ia * O_ia
        # But wait - for a scalar operator, O_ia should be zero in the HF basis
        # because HF eigenstates diagonalize the Fock operator
        # Let me include it anyway for generality
        expectation += contract('ia,ia->', l1, v1b[o, v])
        
        # Additional contributions from l2-t2 contractions
        # rho_ij = delta_ij - 0.5 * sum_{mef} l2_imef * t2_jmef
        rho_oo_corr = -0.5 * contract('imef,jmef->ij', l2, t2)
        expectation += contract('ij,ij->', rho_oo_corr, v1b[o, o])
        
        # rho_ab = 0.5 * sum_{mne} l2_mnae * t2_mnbe
        rho_vv = 0.5 * contract('mnae,mnbe->ab', l2, t2)
        expectation += contract('ab,ab->', rho_vv, v1b[v, v])
    
    # --- Two-body contribution ---
    if 'v2b' in operator:
        v2b = operator['v2b']
        
        # Compute effective density in particle-hole space
        rho2_ph = compute_two_body_density(l1, l2, t1, t2, n_occ, n_states)
        
        # The two-body expectation: (1/4) * sum_{ijab} rho_ijab * V_ijab
        expectation += 0.25 * contract('ijab,ijab->', rho2_ph, v2b[o, o, v, v])
    
    return expectation



def test_j2_expectation():
    """
    Test the expectation value calculation with the J^2 operator.
    """
    from read_snt_io import read_snt, generate_m_scheme, decouple_1b, decouple_2b
    from j2_operator import generate_j2_j_scheme
    from hf import hartree_fock, normal_order
    from cc import ccsd_diis_solver
    from lambda_cc import lambda_ccsd
    import os
    
    print("\n" + "="*70)
    print("Testing J^2 Expectation Value Calculation")
    print("="*70)
    
    # Load interaction
    snt_file = "sd.snt"
    if not os.path.exists(snt_file):
        print(f"Error: {snt_file} not found")
        return
    
    print(f"\n1. Loading interaction from {snt_file}...")
    orbits, potential = read_snt(snt_file)
    
    # Generate m-scheme basis
    print("2. Generating m-scheme basis...")
    m_scheme = generate_m_scheme(orbits)
    print(f"   M-scheme basis size: {len(m_scheme.n)}")
    
    # Decouple interaction to m-scheme
    print("3. Decoupling interaction to m-scheme...")
    v2b_sparse = decouple_2b(potential, m_scheme)
    
    # Run Hartree-Fock (8 nucleons: 4 protons + 4 neutrons -> J=0 ground state)
    print("\n4. Running Hartree-Fock...")
    Z_val = 4  # 4 valence protons
    N_val = 4  # 4 valence neutrons
    n_occ = Z_val + N_val
    
    try:
        hf_energy, sp_energies, rho, sp_coeffs = hartree_fock(
            m_scheme, potential, Z_val, N_val, v2b_sparse=v2b_sparse, max_iter=100, tol=1e-8
        )
        print(f"   HF Energy: {hf_energy:.6f} MeV")
    except Exception as e:
        print(f"   HF failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Build normal-ordered Hamiltonian
    print("\n5. Building normal-ordered Hamiltonian...")
    no_ham = normal_order(m_scheme, potential, hf_energy, sp_energies, rho, sp_coeffs, v2b_sparse=v2b_sparse)
    n_states = len(m_scheme.n)
    
    # Run CCSD
    print("\n6. Running CCSD...")
    try:
        e_corr, t1, t2 = ccsd_diis_solver(no_ham, n_occ, max_iter=100, tol=1e-8)
        print(f"   CCSD Correlation Energy: {e_corr:.6f} MeV")
        print(f"   Total Energy: {hf_energy + e_corr:.6f} MeV")
    except Exception as e:
        print(f"   CCSD failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Run Lambda-CCSD
    print("\n7. Running Lambda-CCSD...")
    try:
        l1, l2 = lambda_ccsd(no_ham, t1, t2, n_occ, max_iter=100, tol=1e-8, alpha=0.5)
    except Exception as e:
        print(f"   Lambda-CCSD failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Generate J^2 operator
    print("\n8. Generating J^2 operator...")
    j2_j_scheme = generate_j2_j_scheme(orbits)
    m_j2_pot = decouple_1b(j2_j_scheme, m_scheme)
    v2b_j2_sparse = decouple_2b(j2_j_scheme, m_scheme)
    
    # Transform J^2 to HF basis
    print("   Transforming J^2 to HF basis...")
    j2_v1b_hf, j2_v2b_hf = transform_operator_to_hf_basis(
        m_j2_pot.v1b, v2b_j2_sparse, sp_coeffs, n_states
    )
    
    operator_j2 = {
        'v1b': j2_v1b_hf,
        'v2b': j2_v2b_hf
    }
    
    # Compute expectation value
    print("\n9. Computing <J^2> expectation value...")
    j2_exp = compute_expectation_value(l1, l2, t1, t2, operator_j2, n_occ, n_states)
    
    print("\n" + "="*70)
    print(f"RESULT: <J^2> = {j2_exp:.6f}")
    print("="*70)
    
    # For J=0 ground state, we expect <J^2> ≈ 0
    # For other spins, <J^2> = J(J+1)
    if j2_exp >= 0:
        J_value = (-0.5 + np.sqrt(0.25 + j2_exp))  # Solve J(J+1) = j2_exp
        print(f"\nCorresponding J value: {J_value:.3f}")
        print(f"(For J=0 ground state, expect J ≈ 0)")
    else:
        print(f"\nWarning: <J^2> is negative, which is unphysical")
    
    return j2_exp


if __name__ == "__main__":
    test_j2_expectation()
