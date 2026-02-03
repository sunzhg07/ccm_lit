"""
Find the optimal occupation indices for HF that gives J^2 = 0.

For 4 protons + 4 neutrons in sd-shell, we need to find which 8 m-scheme states
to occupy such that <HF|J^2|HF> = 0.
"""

import numpy as np
from itertools import combinations
from read_snt_io import read_snt, generate_m_scheme, decouple_1b, decouple_2b
from j2_operator import generate_j2_j_scheme


def compute_j2_for_occupation(occupied_indices, m_j2_1b, m_j2_2b, n_states):
    """Compute <J^2> for a given occupation pattern."""
    # 1-body
    j2_1b = sum(m_j2_1b[i, i] for i in occupied_indices)
    
    # 2-body
    j2_2b = 0.0
    for i in occupied_indices:
        for j in occupied_indices:
            idx_ij = i * n_states + j
            j2_2b += 0.5 * m_j2_2b[idx_ij, idx_ij]
    
    return j2_1b + j2_2b


def find_j0_occupation():
    print("="*80)
    print("FINDING J=0 OCCUPATION PATTERN")
    print("="*80)
    
    # Load
    orbits, _ = read_snt("sd.snt")
    m_scheme = generate_m_scheme(orbits)
    n_states = len(m_scheme.n)
    
    # Generate J^2
    j2_pot = generate_j2_j_scheme(orbits)
    m_j2_1b = decouple_1b(j2_pot, m_scheme)
    m_j2_2b = decouple_2b(j2_pot, m_scheme)
    
    # Show states
    print("\nAvailable m-scheme states:")
    print(f"{'Idx':>3} | {'n':>2} {'l':>2} {'2j':>3} {'2m_j':>5} {'2t_z':>5} | j(j+1)")
    for i in range(n_states):
        ji = m_scheme.j[i] / 2.0
        mji = m_scheme.jz[i] / 2.0
        tzi = m_scheme.tz[i]
        particle = "p" if tzi < 0 else "n"
        print(f"{i:3d} | {m_scheme.n[i]:2d} {m_scheme.l[i]:2d} {m_scheme.j[i]:3d} "
              f"{m_scheme.jz[i]:5d} {tzi:5d} | {ji*(ji+1):.3f} ({particle})")
    
    # Separate protons and neutrons
    p_indices = [i for i in range(n_states) if m_scheme.tz[i] < 0]
    n_indices = [i for i in range(n_states) if m_scheme.tz[i] > 0]
    
    print(f"\nProton states: {p_indices}")
    print(f"Neutron states: {n_indices}")
    
    print(f"\nSearching for J=0 occupation (4p + 4n)...")
    print("This may take a moment...\n")
    
    best_j2 = float('inf')
    best_occ = None
    good_occupations = []
    
    # Try all combinations of 4 protons and 4 neutrons
    for p_occ in combinations(p_indices, 4):
        for n_occ in combinations(n_indices, 4):
            occ = sorted(list(p_occ) + list(n_occ))
            
            # Check M_J conservation
            total_mj = sum(m_scheme.jz[i] for i in occ)
            if total_mj != 0:
                continue
            
            # Compute J^2
            j2_val = compute_j2_for_occupation(occ, m_j2_1b.v1b, m_j2_2b, n_states)
            
            if abs(j2_val) < 0.01:
                good_occupations.append((occ, j2_val))
                print(f"  Found J≈0 occupation: {occ}, <J^2> = {j2_val:.6f}")
            
            if abs(j2_val) < abs(best_j2):
                best_j2 = j2_val
                best_occ = occ
    
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    
    if good_occupations:
        print(f"\nFound {len(good_occupations)} occupation(s) with <J^2> ≈ 0:")
        for occ, j2_val in good_occupations:
            print(f"\n  occ_indices = {occ}")
            print(f"  <J^2> = {j2_val:.8f}")
            
            # Show details
            print("  States:")
            for idx in occ:
                ji = m_scheme.j[idx] / 2.0
                mji = m_scheme.jz[idx] / 2.0
                tzi = m_scheme.tz[idx]
                particle = "p" if tzi < 0 else "n"
                print(f"    {idx:2d}: j={ji:.1f}, m_j={mji:+.1f}, {particle}")
    else:
        print(f"\nNo exact J=0 occupation found.")
        print(f"Best occupation: {best_occ}")
        print(f"  <J^2> = {best_j2:.6f}")
        
        print("\n  States:")
        for idx in best_occ:
            ji = m_scheme.j[idx] / 2.0
            mji = m_scheme.jz[idx] / 2.0
            tzi = m_scheme.tz[idx]
            particle = "p" if tzi < 0 else "n"
            print(f"    {idx:2d}: j={ji:.1f}, m_j={mji:+.1f}, {particle}")
    
    print("\n" + "="*80)
    print("TO USE THIS IN HF:")
    print("="*80)
    if good_occupations:
        occ, _ = good_occupations[0]
        print(f"\nocc_indices = {occ}")
    else:
        print(f"\nocc_indices = {best_occ}")
    print("\nhf_energy, sp_energies, rho, sp_coeffs = hartree_fock(")
    print("    m_scheme, potential, 4, 4,")
    print("    v2b_sparse=v2b_sparse,")
    print("    occ_indices=occ_indices")
    print(")")
    
    return good_occupations if good_occupations else [(best_occ, best_j2)]


if __name__ == "__main__":
    find_j0_occupation()
