"""
Diagnostic: Check J^2 expectation value at Hartree-Fock level

For a J=0 Hartree-Fock state, we should have:
    <HF|J^2|HF> = 0

After normal ordering J^2 with respect to |HF>, the zero-body term should be:
    E_0 = <HF|J^2|HF> = 0 for J=0 state
"""

import numpy as np
from read_snt_io import read_snt, generate_m_scheme, decouple_1b, decouple_2b
from j2_operator import generate_j2_j_scheme
from hf import hartree_fock


def compute_j2_at_hf_level(rho, j2_1b, j2_2b_sparse, n_states):
    """
    Compute <HF|J^2|HF> = Tr(rho * j2_1b) + (1/2) Tr(rho^2 * j2_2b)
    
    For the two-body part in antisymmetrized form:
    <HF|V_2b|HF> = (1/2) sum_ij rho_ii rho_jj <ij||ij>
    """
    # One-body contribution
    j2_1b_val = np.trace(rho @ j2_1b)
    
    print(f"  One-body contribution: {j2_1b_val:.6f}")
    
    # Get occupied indices
    occupied = np.where(np.diag(rho) > 0.5)[0]
    print(f"  Occupied states: {occupied}")
    
    # Two-body contribution
    # For antisymmetrized elements: <ij||kl> stored in sparse matrix
    # HF contribution: (1/2) sum_{i,j occupied} <ij||ij>
    j2_2b_val = 0.0
    for i in occupied:
        for j in occupied:
            idx_ij = i * n_states + j
            v_elem = j2_2b_sparse[idx_ij, idx_ij]
            j2_2b_val += 0.5 * v_elem
    
    print(f"  Two-body contribution: {j2_2b_val:.6f}")
    
    j2_total = j2_1b_val + j2_2b_val
    print(f"  Total <HF|J^2|HF>: {j2_total:.6f}")
    
    return j2_1b_val, j2_2b_val, j2_total


def check_j2_for_occupied_states(m_scheme, occupied_indices):
    """
    Check if the occupied states form a J=0 configuration.
    
    For J=0, we need:
    - Total M_J = 0 (sum of m_j values)
    - Proper pairing structure
    """
    print("\n  Occupied state quantum numbers:")
    print(f"  {'Idx':>4} | {'n':>2} {'l':>2} {'2j':>3} {'2m_j':>5} {'2t_z':>5}")
    print("  " + "-"*35)
    
    total_mj = 0
    total_tz = 0
    
    for idx in occupied_indices:
        n = m_scheme.n[idx]
        l = m_scheme.l[idx]
        j = m_scheme.j[idx]
        mj = m_scheme.jz[idx]
        tz = m_scheme.tz[idx]
        
        print(f"  {idx:4d} | {n:2d} {l:2d} {j:3d} {mj:5d} {tz:5d}")
        
        total_mj += mj
        total_tz += tz
    
    print(f"\n  Total M_J = {total_mj/2:.1f} (should be 0 for J=0)")
    print(f"  Total T_Z = {total_tz/2:.1f}")
    
    if total_mj == 0:
        print("  ✓ M_J projection satisfied for J=0")
    else:
        print("  ✗ M_J ≠ 0, state cannot be J=0!")
    
    return total_mj == 0


def main():
    print("\n" + "="*80)
    print("J^2 DIAGNOSTICS AT HARTREE-FOCK LEVEL")
    print("="*80)
    
    # Configuration
    snt_file = "sd.snt"
    Z_val = 4
    N_val = 4
    
    print(f"\nConfiguration: {Z_val}p + {N_val}n")
    
    # Load and setup
    print("\nStep 1: Loading interaction and generating basis...")
    orbits, potential = read_snt(snt_file)
    m_scheme = generate_m_scheme(orbits)
    n_states = len(m_scheme.n)
    print(f"  M-scheme states: {n_states}")
    
    # Run HF
    print("\nStep 2: Running Hartree-Fock...")
    v2b_sparse = decouple_2b(potential, m_scheme)
    hf_energy, sp_energies, rho, sp_coeffs = hartree_fock(
        m_scheme, potential, Z_val, N_val, v2b_sparse=v2b_sparse, max_iter=100, tol=1e-8
    )
    print(f"  HF Energy: {hf_energy:.6f} MeV")
    
    # Check occupied states
    print("\nStep 3: Analyzing HF reference state...")
    occupied = np.where(np.diag(rho) > 0.5)[0]
    is_j0_candidate = check_j2_for_occupied_states(m_scheme, occupied)
    
    # Generate J^2 operator
    print("\nStep 4: Generating J^2 operator...")
    j2_j_scheme = generate_j2_j_scheme(orbits)
    m_j2_1b = decouple_1b(j2_j_scheme, m_scheme)
    m_j2_2b_sparse = decouple_2b(j2_j_scheme, m_scheme)
    
    # Compute J^2 at HF level
    print("\nStep 5: Computing <HF|J^2|HF>...")
    j2_1b_val, j2_2b_val, j2_total = compute_j2_at_hf_level(
        rho, m_j2_1b.v1b, m_j2_2b_sparse, n_states
    )
    
    # Analysis
    print("\n" + "="*80)
    print("ANALYSIS")
    print("="*80)
    
    if is_j0_candidate:
        if abs(j2_total) < 0.01:
            print("✓ <HF|J^2|HF> ≈ 0, consistent with J=0 state")
        else:
            print(f"✗ <HF|J^2|HF> = {j2_total:.6f} ≠ 0")
            print("  This suggests:")
            print("  1. The J^2 operator may be incorrectly constructed, OR")
            print("  2. The HF state is not a pure J=0 eigenstate (has admixtures)")
    else:
        print(f"✗ HF state has M_J ≠ 0, cannot be J=0")
        print(f"  <HF|J^2|HF> = {j2_total:.6f}")
    
    print("="*80)
    
    return j2_total


if __name__ == "__main__":
    main()
