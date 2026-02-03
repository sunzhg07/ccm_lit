"""
Detailed check of J^2 two-body matrix elements to find the normalization issue.
"""

import numpy as np
from read_snt_io import read_snt, generate_m_scheme
from j2_operator import generate_j2_j_scheme


def analyze_j2_operator():
    print("="*80)
    print("ANALYZING J^2 OPERATOR CONSTRUCTION")  
    print("="*80)
    
    # Load
    orbits, _ = read_snt("sd.snt")
    
    # Generate J^2
    j2_pot = generate_j2_j_scheme(orbits)
    
    print("\nJ-scheme orbits:")
    for i in range(len(orbits.n)):
        ji = orbits.j[i] / 2.0
        tz = orbits.tz[i]
        particle = "p" if tz < 0 else "n"
        print(f"  Orbit {i}: j={ji:.1f}, tz={tz:+d} ({particle}), j(j+1)={ji*(ji+1):.3f}")
    
    print("\n1-body J^2 elements:")
    for i in range(len(orbits.n)):
        print(f"  <{i}|j^2|{i}> = {j2_pot.v1b[i,i]:.6f}")
    
    print(f"\n2-body J^2 elements (total: {len(j2_pot.v2b)}):")
    print(f"  Format: <rs J | 2*j_r·j_s | rs J> = J(J+1) - j_r(j_r+1) - j_s(j_s+1)")
    print(f"  {'r':>3} {'s':>3} | {'j_r':>5} {'j_s':>5} | {'J':>3} | {'TBME':>10}")
    print("  " + "-"*45)
    
    for entry in j2_pot.v2b[:10]:  # Show first 10
        r, s, t, u, J, val = entry
        jr = orbits.j[r] / 2.0
        js = orbits.j[s] / 2.0
        expected = J*(J+1) - jr*(jr+1) - js*(js+1)
        print(f"  {r:3d} {s:3d} | {jr:5.1f} {js:5.1f} | {J:3d} | {val:10.3f} (expect: {expected:.3f})")
    
    if len(j2_pot.v2b) > 10:
        print(f"  ... ({len(j2_pot.v2b) - 10} more elements)")
    
    # Example: For d3/2 - d3/2 coupling
    print("\n" + "="*80)
    print("EXAMPLE: d_{3/2} - d_{3/2} coupling")
    print("="*80)
    j =1.5
    print(f"  j = {j}")
    print(f"  j(j+1) = {j*(j+1):.3f}")
    print(f"\n  Two d_{3/2} particles can couple to J = 0, 1, 2, 3")
    for J in [0, 1, 2, 3]:
        tbme = J*(J+1) - 2*j*(j+1)
        print(f"  J={J}: TBME = {J}({J}+1) - 2×{j}({j}+1) = {tbme:.3f}")
    
    print("\n  For J=0: TBME = -2×j(j+1) = -2×3.75 = -7.5")
    print("  This represents 2*j·j for the coupled pair")
    
    # Check what we actually have
    print("\n  In our j2_pot.v2b:")
    for entry in j2_pot.v2b:
        r, s, t, u, J, val = entry
        if r == 0 and s == 0:  # d3/2 proton - d3/2 proton
            print(f"    Orbit 0-0, J={J}: {val:.3f}")


if __name__ == "__main__":
    analyze_j2_operator()
