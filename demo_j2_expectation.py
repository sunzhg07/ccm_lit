"""
Summary and demonstration of J^2 operator expectation value calculation.

This module demonstrates:
1. Generation of J^2 operator in j-scheme
2. Transformation to m-scheme via Clebsch-Gordan coefficients  
3. Transformation to HF basis
4. Computation of expectation value using CCSD and Lambda-CCSD wave functions
"""

import numpy as np
from read_snt_io import read_snt, generate_m_scheme, decouple_1b, decouple_2b
from j2_operator import generate_j2_j_scheme
from hf import hartree_fock, normal_order
from cc import ccsd_diis_solver
from lambda_cc import lambda_ccsd
from operator_expectation import transform_operator_to_hf_basis, compute_expectation_value


def main():
    print("\n" + "="*80)
    print("J^2 OPERATOR EXPECTATION VALUE WITH CCSD/LAMBDA-CCSD")
    print("="*80)
    
    # Configuration
    snt_file = "sd.snt"
    Z_val = 4  # 2 valence protons  
    N_val = 4  # 2 valence neutrons
    n_occ = Z_val + N_val
    
    print(f"\nConfiguration: {Z_val} protons + {N_val} neutrons")
    print(f"Interaction file: {snt_file}")
    
    # Step 1: Load interaction and generate m-scheme
    print("\n" + "-"*80)
    print("STEP 1: Loading interaction and generating m-scheme basis")
    print("-"*80)
    
    orbits, potential = read_snt(snt_file)
    m_scheme = generate_m_scheme(orbits)

    print(f"total m-scheme states: {len(m_scheme.n)}")
    for i in range(len(m_scheme.n)):
        print(f"  {i:2d}: n={m_scheme.n[i]}, l={m_scheme.l[i]}, j={m_scheme.j[i]}, jz={m_scheme.jz[i]:2d}, tz={m_scheme.tz[i]:2d}")

    n_states = len(m_scheme.n)
    
    print(f"  J-scheme orbits: {len(orbits.n)}")
    print(f"  M-scheme states: {n_states}")
    
    v2b_sparse = decouple_2b(potential, m_scheme)
    print(f"  2-body matrix elements: {v2b_sparse.count_nonzero()}")
    
    # Step 2: Hartree-Fock
    print("\n" + "-"*80)
    print("STEP 2: Hartree-Fock calculation")
    print("-"*80)
    
    occ_indices = [0, 1, 2, 3,6,7,8,9] 
    hf_energy, sp_energies, rho, sp_coeffs = hartree_fock(
        m_scheme, potential, Z_val, N_val, v2b_sparse=v2b_sparse, max_iter=100, tol=1e-8
    ,occ_indices=occ_indices)
    print(f"  HF Energy: {hf_energy:.6f} MeV")
    
    # Step 3: Normal ordering
    print("\n" + "-"*80)
    print("STEP 3: Normal ordering Hamiltonian")
    print("-"*80)
    
    no_ham = normal_order(m_scheme, potential, hf_energy, sp_energies, rho, sp_coeffs, 
                          v2b_sparse=v2b_sparse)
    print(f"  0-body term: {no_ham.E0:.6f} MeV")
    print(f"  1-body matrix: {no_ham.f.shape}")
    print(f"  2-body tensor: {no_ham.Gamma.shape}")
    
    # Step 4: CCSD
    print("\n" + "-"*80)
    print("STEP 4: CCSD calculation")
    print("-"*80)
    
    e_corr, t1, t2 = ccsd_diis_solver(no_ham, n_occ, max_iter=100, tol=1e-8)
    print(f"  CCSD correlation energy: {e_corr:.6f} MeV")
    print(f"  Total energy: {hf_energy + e_corr:.6f} MeV")
    print(f"  T1 shape: {t1.shape}, norm: {np.linalg.norm(t1):.6f}")
    print(f"  T2 shape: {t2.shape}, norm: {np.linalg.norm(t2):.6f}")
    
    # Step 5: Lambda-CCSD
    print("\n" + "-"*80)
    print("STEP 5: Lambda-CCSD calculation")
    print("-"*80)
    
    l1, l2 = lambda_ccsd(no_ham, t1, t2, n_occ, max_iter=100, tol=1e-8, alpha=0.5)
    print(f"  L1 shape: {l1.shape}, norm: {np.linalg.norm(l1):.6f}")
    print(f"  L2 shape: {l2.shape}, norm: {np.linalg.norm(l2):.6f}")
    
    # Step 6: Generate and transform J^2 operator
    print("\n" + "-"*80)
    print("STEP 6: J^2 operator generation and transformation")
    print("-"*80)
    
    print("  Generating J^2 in j-scheme...")
    j2_j_scheme = generate_j2_j_scheme(orbits)
    print(f"    1-body elements: {np.count_nonzero(j2_j_scheme.v1b)}")
    print(f"    2-body groups: {len(j2_j_scheme.v2b)}")
    
    print("  Decoupling to m-scheme...")
    m_j2_pot = decouple_1b(j2_j_scheme, m_scheme)
    v2b_j2_sparse = decouple_2b(j2_j_scheme, m_scheme)
    print(f"    M-scheme 2-body elements: {v2b_j2_sparse.count_nonzero()}")
    
    print("  Transforming to HF basis...")
    j2_v1b_hf, j2_v2b_hf = transform_operator_to_hf_basis(
        m_j2_pot.v1b, v2b_j2_sparse, sp_coeffs, n_states
    )
    
    operator_j2 = {'v1b': j2_v1b_hf, 'v2b': j2_v2b_hf}
    
    # Step 7: Compute expectation value
    print("\n" + "-"*80)
    print("STEP 7: Computing <J^2> expectation value")
    print("-"*80)
    
    j2_exp = compute_expectation_value(l1, l2, t1, t2, operator_j2, n_occ, n_states)
    
    # Results
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print(f"  <J^2> = {j2_exp:.6f}")
    
    if j2_exp >= 0:
        J_value = (-0.5 + np.sqrt(0.25 + j2_exp))
        print(f"  Corresponding J ≈ {J_value:.3f}")
        print(f"\n  Note: For a true J=0 state, we expect <J^2> = 0")
        print(f"        The non-zero value indicates angular momentum mixing in the wave function")
    else:
        print(f"  Warning: Negative <J^2> indicates numerical issues")
    
    print("="*80)
    
    return j2_exp


if __name__ == "__main__":
    main()
