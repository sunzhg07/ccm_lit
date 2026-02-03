"""
Check if normal-ordered J^2 has correct zero-body term.

After normal ordering with respect to |HF⟩:
    J^2 = E_0 + F + Gamma
    
where E_0 = <HF|J^2|HF> should equal what we calculated manually.

Then for CCSD:
    <CCSD|J^2|CCSD> = E_0 + <one-body> + <two-body>
    
If the HF state is J=0, then E_0 = 0 and the remaining terms come from correlations.
"""

import numpy as np
from read_snt_io import read_snt, generate_m_scheme, decouple_1b, decouple_2b, sparse_to_dense_4d
from j2_operator import generate_j2_j_scheme
from hf import hartree_fock


def normal_order_j2(m_j2_1b, m_j2_2b_sparse, sp_coeffs, rho, n_states, n_occ):
    """
    Normal order J^2 with respect to the HF reference.
    
    Returns:
        E_0: Zero-body term = <HF|J^2|HF>
        f_J2: One-body part in HF basis
        Gamma_J2: Two-body part in HF basis
    """
    print("\n" + "="*80)
    print("NORMAL ORDERING J^2")
    print("="*80)
    
    # First transform to HF basis
    C = sp_coeffs
    
    # 1-body transformation
    j2_1b_hf = C.T @ m_j2_1b @ C
    
    # 2-body transformation  
    v2b_dense = sparse_to_dense_4d(m_j2_2b_sparse, n_states)
    print("  Transforming 2-body to HF basis...")
    temp = np.einsum('pqrs,sl->pqrl', v2b_dense, C)
    temp = np.einsum('pqrl,rk->pqkl', temp, C)
    temp = np.einsum('pqkl,qj->pjkl', temp, C)
    j2_2b_hf = np.einsum('pjkl,pi->ijkl', temp, C)
    
    o = slice(0, n_occ)
    v = slice(n_occ, n_states)
    
    # Compute zero-body term: <HF|J^2|HF>
    # = sum_i^occ j2_1b_ii + (1/2) sum_ij^occ <ij||ij>
    
    E_0 = 0.0
    
    # 1-body part
    E_0 += np.trace(j2_1b_hf[o, o])
    print(f"  E_0 from 1-body: {np.trace(j2_1b_hf[o, o]):.6f}")
    
    # 2-body part
    j2_2b_contrib = 0.0
    for i in range(n_occ):
        for j in range(n_occ):
            j2_2b_contrib += 0.5 * j2_2b_hf[i, j, i, j]
    E_0 += j2_2b_contrib
    print(f"  E_0 from 2-body: {j2_2b_contrib:.6f}")
    print(f"  Total E_0 = <HF|J^2|HF>: {E_0:.6f}")
    
    # One-body normal-ordered part
    # f = j2_1b + sum_k^occ <ik||jk>
    f_J2 = j2_1b_hf.copy()
    
    # add Hartree-Fock contraction of 2-body
    for a in range(n_states):
        for b in range(n_states):
            for k in range(n_occ):
                f_J2[a, b] += j2_2b_hf[a, k, b, k]
    
    print(f"\n  Normal-ordered  1-body:")
    print(f"    Diagonal (occ): {np.diag(f_J2[o,o])}")
    print(f"    Diagonal (virt): {np.diag(f_J2[v,v])[:4]}...")
    print(f"    Off-diag max: {np.max(np.abs(f_J2[o,v])):.6f}")
    
    # Two-body normal-ordered part  
    # Gamma = j2_2b (already antisymmetrized)
    Gamma_J2 = j2_2b_hf
    
    return E_0, f_J2, Gamma_J2


def main():
    print("="*80)
    print("VERIFYING NORMAL-ORDERED J^2")
    print("="*80)
    
    # Setup
    orbits, potential = read_snt("sd.snt")
    m_scheme = generate_m_scheme(orbits)
    n_states = len(m_scheme.n)
    
    # Run HF
    print("\nRunning HF...")
    v2b_sparse = decouple_2b(potential, m_scheme)
    hf_energy, sp_energies, rho, sp_coeffs = hartree_fock(
        m_scheme, potential, 4, 4, v2b_sparse=v2b_sparse, max_iter=100, tol=1e-8
    )
    print(f"  HF Energy: {hf_energy:.6f} MeV")
    
    n_occ = 8
    
    # Generate J^2
    print("\nGenerating J^2 in m-scheme...")
    j2_pot = generate_j2_j_scheme(orbits)
    m_j2_1b = decouple_1b(j2_pot, m_scheme)
    m_j2_2b = decouple_2b(j2_pot, m_scheme)
    
    # Compute <HF|J^2|HF> directly in m-scheme
    print("\nComputing <HF|J^2|HF> in m-scheme basis...")
    occupied = np.where(np.diag(rho) > 0.5)[0]
    
    j2_hf_1b = np.trace(rho @ m_j2_1b.v1b)
    
    j2_hf_2b = 0.0
    for i in occupied:
        for j in occupied:
            idx_ij = i * n_states + j
            j2_hf_2b += 0.5 * m_j2_2b[idx_ij, idx_ij]
    
    j2_hf_total = j2_hf_1b + j2_hf_2b
    print(f"  <HF|J^2|HF> = {j2_hf_1b:.6f} + {j2_hf_2b:.6f} = {j2_hf_total:.6f}")
    
    # Normal order
    E_0, f_J2, Gamma_J2 = normal_order_j2(m_j2_1b.v1b, m_j2_2b, sp_coeffs, rho, n_states, n_occ)
    
    # Check
    print("\n" + "="*80)
    print("VERIFICATION")
    print("="*80)
    print(f"  <HF|J^2|HF> (direct m-scheme): {j2_hf_total:.6f}")
    print(f"  E_0 (from normal ordering):    {E_0:.6f}")
    print(f"  Difference:                     {abs(j2_hf_total - E_0):.6e}")
    
    if abs(j2_hf_total - E_0) < 1e-6:
        print("\n✓ Normal ordering is correct!")
    else:
        print("\n✗ Discrepancy in normal ordering!")
    
    print(f"\n  For J=0 state, E_0 should be 0")
    print(f"  Actual E_0 = {E_0:.6f}")
    
    if abs(E_0) > 0.01:
        print("\n  ⚠ E_0 ≠ 0: The HF reference is NOT a J=0 eigenstate")
        print("     This will add a constant offset to all J^2 expectation values")
        print("     The CCSD calculation will give:")
        print(f"       <CCSD|J^2|CCSD> = {E_0:.6f} + (correlation corrections)")
    
    return E_0


if __name__ == "__main__":
    main()
