"""
Manual calculation of <HF|J^2|HF> with detailed breakdown.

For the HF state with (4p+4n) in sd-shell:
- Occupied: 0d_{3/2} m=±1/2 (p), 0s_{1/2} m=±1/2 (p), 
            0d_{3/2} m=±1/2 (n), 0s_{1/2} m=±1/2 (n)
"""

import numpy as np
from read_snt_io import read_snt, generate_m_scheme, decouple_1b, decouple_2b
from j2_operator import generate_j2_j_scheme
from hf import hartree_fock


def manual_j2_calculation():
    print("="*80)
    print("MANUAL CALCULATION OF <HF|J^2|HF>")
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
    
    occupied = np.where(np.diag(rho) > 0.5)[0]
    print(f"\nOccupied states: {occupied}")
    
    # Generate J^2
    j2_pot = generate_j2_j_scheme(orbits)
    m_j2_1b = decouple_1b(j2_pot, m_scheme)
    m_j2_2b = decouple_2b(j2_pot, m_scheme)
    
    # 1-body contribution
    print("\n" + "-"*80)
    print("1-BODY CONTRIBUTION: sum_i <i|j^2|i>")
    print("-"*80)
    
    onebd_total = 0.0
    j_counts = {}  # Count how many particles in each j-orbit
    
    for idx in occupied:
        j_idx = m_scheme.coupling_map[idx]  # j-scheme orbit index
        j_val = m_scheme.j[idx] / 2.0
        jz_val = m_scheme.jz[idx] / 2.0
        tz_val = m_scheme.tz[idx]
        
        j_j_plus_1 = m_j2_1b.v1b[idx, idx]
        onebd_total += j_j_plus_1
        
        particle_type = "p" if tz_val < 0 else "n"
        j_label = f"j={j_val:.1f}"
        key = (j_val, tz_val)
        j_counts[key] = j_counts.get(key, 0) + 1
        
        print(f"  State {idx:2d}: j={j_val:.1f}, m_j={jz_val:+.1f}, tz={tz_val:+d} ({particle_type}), j(j+1)={j_j_plus_1:.3f}")
    
    print(f"\n  Total 1-body: {onebd_total:.6f}")
    
    print("\n  Occupation summary:")
    for (j_val, tz_val), count in sorted(j_counts.items()):
        particle_type = "p" if tz_val < 0 else "n"
        print(f"    j={j_val:.1f} ({particle_type}): {count} particles, contrib = {count * j_val*(j_val+1):.3f}")
    
    # 2-body contribution
    print("\n" + "-"*80)
    print("2-BODY CONTRIBUTION: (1/2) sum_ij <ij||ij>")
    print("-"*80)
    print("  (antisymmetrized matrix elements)")
    
    twobd_total = 0.0
    pair_contribs = {}
    
    for i in occupied:
        for j in occupied:
            idx_ij = i * n_states + j
            v_elem = m_j2_2b[idx_ij, idx_ij]
            
            if abs(v_elem) > 1e-10:
                ji = m_scheme.j[i] / 2.0
                jj = m_scheme.j[j] / 2.0
                mji = m_scheme.jz[i] / 2.0
                mjj = m_scheme.jz[j] / 2.0
                tzi = m_scheme.tz[i]
                tzj = m_scheme.tz[j]
                
                # Contribution
                contrib = 0.5 * v_elem
                twobd_total += contrib
                
                # Group by pair type
                key = (min(ji,jj), max(ji,jj), min(tzi,tzj), max(tzi,tzj))
                if key not in pair_contribs:
                    pair_contribs[key] = []
                pair_contribs[key].append((i, j, v_elem, contrib))
    
    # Summarize by pair type
    for key in sorted(pair_contribs.keys()):
        j1, j2, tz1, tz2 = key
        pairs = pair_contribs[key]
        total_for_type = sum(p[3] for p in pairs)
        
        p1 = "p" if tz1 < 0 else "n"
        p2 = "p" if tz2 < 0 else "n"
        
        print(f"\n  j={j1:.1f}({p1}) - j={j2:.1f}({p2}): {len(pairs)} pairs, total contrib = {total_for_type:.6f}")
        
        # Show a few examples
        for i, j, v_elem, contrib in pairs[:3]:
            mji = m_scheme.jz[i] / 2.0
            mjj = m_scheme.jz[j] / 2.0
            print(f"    ({i},{j}): m_j=({mji:+.1f},{mjj:+.1f}), <ij||ij>={v_elem:.6f}, contrib={contrib:.6f}")
        
        if len(pairs) > 3:
            print(f"    ... ({len(pairs)-3} more pairs)")
    
    print(f"\n  Total 2-body: {twobd_total:.6f}")
    
    # Total
    print("\n" + "="*80)
    print(f"TOTAL <HF|J^2|HF> = {onebd_total:.6f} + {twobd_total:.6f} = {onebd_total + twobd_total:.6f}")
    print("="*80)
    
    # Analysis
    print("\nEXPECTED VALUE FOR J=0:")
    print("  For a pure J=0 state, <J^2> should be 0")
    print(f"  Actual value: {onebd_total + twobd_total:.6f}")
    print(f"  Deviation: {abs(onebd_total + twobd_total):.6f}")
    
    if abs(onebd_total + twobd_total) > 0.01:
        print("\n⚠ The HF state is NOT a pure J=0 eigenstate!")
        print("  Reason: The m-scheme occupation does not correspond to J=0 coupling")

if __name__ == "__main__":
    manual_j2_calculation()
