"""
Create a correlation J^2 operator by subtracting the HF reference value.

This ensures that after normal ordering, E_0 = 0, so we measure only
the change in J^2 due to correlations beyond HF.

Correlation J^2 = J^2 - <HF|J^2|HF>
"""

import numpy as np
from read_snt_io import Potential, SingleParticleOrbits, decouple_1b, decouple_2b
from j2_operator import generate_j2_j_scheme


def generate_correlation_j2(orbits, rho, m_scheme):
    """
    Generate correlation J^2 operator = J^2 - <HF|J^2|HF>.
    
    This operator has <HF|J^2_corr|HF> = 0 by construction.
    
    Args:
        orbits: J-scheme orbits
        rho: HF density matrix in m-scheme
        m_scheme: M-scheme basis
        
    Returns:
        j2_corr_potential: Potential object with correlation J^2
    """
    # Get the original J^2
    j2_pot = generate_j2_j_scheme(orbits)
    m_j2_1b = decouple_1b(j2_pot, m_scheme)
    m_j2_2b = decouple_2b(j2_pot, m_scheme)
    
    # Compute <HF|J^2|HF>
    n_states = len(m_scheme.n)
    occupied = np.where(np.diag(rho) > 0.5)[0]
    
    # 1-body contribution
    j2_hf_1b = np.trace(rho @ m_j2_1b.v1b)
    
    # 2-body contribution
    j2_hf_2b = 0.0
    for i in occupied:
        for j in occupied:
            idx_ij = i * n_states + j
            j2_hf_2b += 0.5 * m_j2_2b[idx_ij, idx_ij]
    
    j2_hf_total = j2_hf_1b + j2_hf_2b
    
    print(f"\n<HF|J^2|HF> = {j2_hf_total:.6f}")
    print(f"  1-body: {j2_hf_1b:.6f}")
    print(f"  2-body: {j2_hf_2b:.6f}")
    
    # Create correlation operator by subtracting constant
    # We subtract j2_hf_total / N from each occupied state's 1-body term
    # This distributes the constant evenly
    n_occ = len(occupied)
    
    corr_j2_1b = m_j2_1b.v1b.copy()
    for i in occupied:
        corr_j2_1b[i, i] -= j2_hf_total / n_occ
    
    # The 2-body part remains the same
    # (the constant shift is absorbed in the 1-body part)
    
    # Create potential object
    n_orbits_j = len(orbits.n)
    corr_pot = Potential(n_orbits_j)
    
    # We need to transform back to j-scheme... but this is complex
    # Instead, let's just return the  m-scheme matrices directly
    # and handle them separately
    
    return corr_j2_1b, m_j2_2b, j2_hf_total


def verify_correlation_j2():
    """
    Verify that the correlation J^2 operator has <HF|J^2_corr|HF> = 0.
    """
    from read_snt_io import read_snt, generate_m_scheme, decouple_2b
    from hf import hartree_fock
    
    print("="*80)
    print("VERIFICATION: CORRELATION J^2 OPERATOR")
    print("="*80)
    
    # Setup
    orbits, potential = read_snt("sd.snt")
    m_scheme = generate_m_scheme(orbits)
    n_states = len(m_scheme.n)
    
    # Run HF
    v2b_sparse = decouple_2b(potential, m_scheme)
    hf_energy, sp_energies, rho, sp_coeffs = hartree_fock(
        m_scheme, potential, 4, 4, v2b_sparse=v2b_sparse, max_iter=100, tol=1e-8
    )
    
    print(f"\nHF Energy: {hf_energy:.6f} MeV")
    
    # Generate correlation J^2
    print("\nGenerating correlation J^2...")
    corr_j2_1b, corr_j2_2b, hf_offset = generate_correlation_j2(orbits, rho, m_scheme)
    
    # Verify
    print("\nVerifying <HF|J^2_corr|HF> = 0...")
    occupied = np.where(np.diag(rho) > 0.5)[0]
    
    # 1-body
    corr_1b = np.trace(rho @ corr_j2_1b)
    
    # 2-body
    corr_2b = 0.0
    for i in occupied:
        for j in occupied:
            idx_ij = i * n_states + j
            corr_2b += 0.5 * corr_j2_2b[idx_ij, idx_ij]
    
    corr_total = corr_1b + corr_2b
    
    print(f"  <HF|J^2_corr|HF> = {corr_1b:.6f} + {corr_2b:.6f} = {corr_total:.6f}")
    
    if abs(corr_total) < 1e-6:
        print("\n✓ SUCCESS: <HF|J^2_corr|HF> = 0")
        print(f"\nThe normal-ordered correlation J^2 operator can be used with CCSD.")
        print(f"It measures deviations from the HF reference:")
        print(f"  <CCSD|J^2_corr|CCSD> = <CCSD|J^2|CCSD> - {hf_offset:.6f}")
    else:
        print(f"\n✗ ERROR: <HF|J^2_corr|HF> = {corr_total:.6f} ≠ 0")
    
    print("="*80)


if __name__ == "__main__":
    verify_correlation_j2()
