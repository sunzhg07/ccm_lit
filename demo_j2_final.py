"""
Complete demonstration of J^2 expectation value with correlation operator.

Uses the correlation J^2 operator which has zero HF expectation value,
ensuring that the normal-ordered operator has E_0 = 0.
"""

import numpy as np
from read_snt_io import read_snt, generate_m_scheme, decouple_1b, decouple_2b
from j2_operator import generate_j2_j_scheme
from hf import hartree_fock, normal_order
from cc import ccsd_diis_solver
from lambda_cc import lambda_ccsd
from operator_expectation import transform_operator_to_hf_basis, compute_expectation_value
from correlation_j2 import generate_correlation_j2


def main():
    print("\n" + "="*80)
    print("J^2 EXPECTATION VALUE WITH CORRELATION OPERATOR (4p + 4n)")
    print("="*80)
    
    # Configuration
    snt_file = "sd.snt"
    Z_val = 4
    N_val = 4
    n_occ = Z_val + N_val
    
    print(f"\nConfiguration: {Z_val}p + {N_val}n")
    
    # Load
    print("\n" + "-"*80)
    print("Step 1: Loading and setup")
    print("-"*80)
    orbits, potential = read_snt(snt_file)
    m_scheme = generate_m_scheme(orbits)
    n_states = len(m_scheme.n)
    v2b_sparse = decouple_2b(potential, m_scheme)
   
    print(f"  M-scheme states: {n_states}")


    print(f"total m-scheme states: {len(m_scheme.n)}")
    for i in range(len(m_scheme.n)):
        print(f"  {i:2d}: n={m_scheme.n[i]}, l={m_scheme.l[i]}, j={m_scheme.j[i]}, jz={m_scheme.jz[i]:2d}, tz={m_scheme.tz[i]:2d}")

    occ_indices = [0, 1, 2, 3,6,7,8,9] 
    hf_energy, sp_energies, rho, sp_coeffs = hartree_fock(
        m_scheme, potential, Z_val, N_val, v2b_sparse=v2b_sparse, max_iter=100, tol=1e-8
    ,occ_indices=occ_indices)
    print(f"  HF Energy: {hf_energy:.6f} MeV")
    
    
    # Normal order Hamiltonian
    print("\n" + "-"*80)
    print("Step 3: Normal ordering Hamiltonian")
    print("-"*80)
    no_ham = normal_order(m_scheme, potential, hf_energy, sp_energies, rho, sp_coeffs, 
                          v2b_sparse=v2b_sparse)
    
    # CCSD
    print("\n" + "-"*80)
    print("Step 4: CCSD")
    print("-"*80)
    e_corr, t1, t2 = ccsd_diis_solver(no_ham, n_occ, max_iter=100, tol=1e-8)
    print(f"  Correlation energy: {e_corr:.6f} MeV")
    print(f"  Total energy: {hf_energy + e_corr:.6f} MeV")
    
    # Lambda-CCSD
    print("\n" + "-"*80)
    print("Step 5: Lambda-CCSD")
    print("-"*80)
    l1, l2 = lambda_ccsd(no_ham, t1, t2, n_occ, max_iter=100, tol=1e-8, alpha=0.5)
    
    # Generate correlation J^2
    print("\n" + "-"*80)
    print("Step 6: Correlation J^2 operator")
    print("-"*80)
    corr_j2_1b, corr_j2_2b, j2_hf_offset = generate_correlation_j2(orbits, rho, m_scheme)
    
    # Transform to HF basis
    print("\n  Transforming to HF basis...")
    j2_corr_1b_hf, j2_corr_2b_hf = transform_operator_to_hf_basis(
        corr_j2_1b, corr_j2_2b, sp_coeffs, n_states
    )
    
    operator_j2_corr = {
        'v1b': j2_corr_1b_hf,
        'v2b': j2_corr_2b_hf
    }
    
    # Compute expectation value
    print("\n" + "-"*80)
    print("Step 7: Computing <J^2> expectation value")
    print("-"*80)
    
    j2_corr_exp = compute_expectation_value(l1, l2, t1, t2, operator_j2_corr, n_occ, n_states)
    
    # Full J^2 = correlation + HF offset
    j2_full_exp = j2_corr_exp + j2_hf_offset
    
    # Results
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print(f"  <HF|J^2|HF>             = {j2_hf_offset:.6f}")
    print(f"  <CCSD|J^2_corr|CCSD>    = {j2_corr_exp:.6f}")
    print(f"  <CCSD|J^2|CCSD>         = {j2_full_exp:.6f}")
    
    if j2_full_exp >= 0:
        J_value = (-0.5 + np.sqrt(0.25 + j2_full_exp))
        print(f"\n  Corresponding J ≈ {J_value:.3f}")
    
    print("\n  INTERPRETATION:")
    print(f"  • The HF reference contributes {j2_hf_offset:.2f} to <J^2>")
    print(f"  • CCSD correlations {'reduce' if j2_corr_exp < 0 else 'increase'} this by {abs(j2_corr_exp):.2f}")
    
    if abs(j2_corr_exp) < abs(j2_hf_offset):
        print(f"  • Correlations partially restore rotational symmetry")
    
    print("="*80)
    
    return j2_full_exp, j2_corr_exp


if __name__ == "__main__":
    main()
