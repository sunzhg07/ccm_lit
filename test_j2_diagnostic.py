"""
Test and diagnose the J^2 operator transformation and expectation value.
"""
import numpy as np
from read_snt_io import read_snt, generate_m_scheme, decouple_1b, decouple_2b
from j2_operator import generate_j2_j_scheme
from hf import hartree_fock, normal_order
from cc import ccsd_diis_solver
from lambda_cc import lambda_ccsd
from operator_expectation import compute_expectation_value


def test_j2_operator_transformation():
    """
    Test that J^2 operator transforms correctly to HF basis.
    """
    print("\n" + "="*70)
    print("Diagnosing J^2 Operator Transformation")
    print("="*70)
    
    # Load interaction
    snt_file = "sd.snt"
    orbits, potential = read_snt(snt_file)
    
    # Generate m-scheme basis
    m_scheme = generate_m_scheme(orbits)
    print(f"\nM-scheme basis size: {len(m_scheme.n)}")
    
    # Check single-particle states
    print("\nSingle-particle states in m-scheme:")
    print(f"{'Idx':>3} | {'n':>2} {'l':>2} {'2j':>3} {'2jz':>4} {'2tz':>4} | j(j+1)")
    for i in range(len(m_scheme.n)):
        ji = m_scheme.j[i] / 2.0
        j_j_plus_1 = ji * (ji + 1)
        print(f"{i:3d} | {m_scheme.n[i]:2d} {m_scheme.l[i]:2d} {m_scheme.j[i]:3d} {m_scheme.jz[i]:4d} {m_scheme.tz[i]:4d} | {j_j_plus_1:.3f}")
    
    # Generate J^2 operator
    j2_j_scheme = generate_j2_j_scheme(orbits)
    m_j2_pot = decouple_1b(j2_j_scheme, m_scheme)
    v2b_j2_sparse = decouple_2b(j2_j_scheme, m_scheme)
    
    print("\n1-body J^2 operator (diagonal elements):")
    print(f"{'Idx':>3} | {'J^2_1b':>10}")
    for i in range(len(m_scheme.n)):
        print(f"{i:3d} | {m_j2_pot.v1b[i,i]:10.3f}")
    
    # Check that off-diagonal elements are small
    off_diag = np.abs(m_j2_pot.v1b - np.diag(np.diag(m_j2_pot.v1b)))
    print(f"\nMax off-diagonal 1-body element: {np.max(off_diag):.6e}")
    
    # Statistics on 2-body part
    print(f"\n2-body J^2 non-zero elements: {v2b_j2_sparse.count_nonzero()}")
    print(f"2-body J^2 matrix shape: {v2b_j2_sparse.shape}")
    print(f"2-body J^2 mean value: {np.mean(v2b_j2_sparse.data):.6f}")
    print(f"2-body J^2 max value: {np.max(np.abs(v2b_j2_sparse.data)):.6f}")
    
    # Now transform to HF basis
    print("\n" + "="*70)
    print("Transforming to HF Basis")
    print("="*70)
    
    v2b_sparse = decouple_2b(potential, m_scheme)
    Z_val = 2
    N_val = 2
    n_occ = Z_val + N_val
    
    hf_energy, sp_energies, rho, sp_coeffs = hartree_fock(
        m_scheme, potential, Z_val, N_val, v2b_sparse=v2b_sparse, max_iter=100, tol=1e-8
    )
    print(f"\nHF Energy: {hf_energy:.6f} MeV")
    
    # Check the J^2 expectation value at HF level
    print("\nJ^2 expectation value at HF level (reference state):")
    j2_hf_ref = np.trace(rho @ m_j2_pot.v1b)
    print(f"  <HF|J^2_1b|HF> = {j2_hf_ref:.6f}")
    
    # Also compute 2-body contribution at HF level
    n_m = len(m_scheme.n)
    j2_hf_2b = 0.0
    for i in range(n_m):
        for j in range(n_m):
            if rho[i, i] < 0.5 or rho[j, j] < 0.5:
                continue
            idx_ij = i * n_m + j
            idx_ji = j * n_m + i
            for k in range(n_m):
                for l in range(n_m):
                    if rho[k, k] < 0.5 or rho[l, l] < 0.5:
                        continue
                    idx_kl = k * n_m + l
                    v_elem = v2b_j2_sparse[idx_ij, idx_kl]
                    # Antisymmetrized: <ij||kl> = <ij|kl> - <ij|lk>
                    j2_hf_2b += 0.25 * v_elem * (1 if (i==k and j==l) else 0)
    
    print(f"  <HF|J^2_2b|HF> = {j2_hf_2b:.6f}")
    print(f"  <HF|J^2_total|HF> = {j2_hf_ref + j2_hf_2b:.6f}")
    
    # The HF reference should have a definite J if it's a good state
    # For sd-shell 4 nucleons, ground state is typically J=0
    
    return m_scheme, m_j2_pot, v2b_j2_sparse


def simple_j2_test():
    """
    Simpler test: compute J^2 for a known state.
    """
    print("\n" + "="*70)
    print("Simple J^2 Test: Single j-shell")
    print("="*70)
    
    # For a single j-shell with j=3/2 (like 0d_{3/2}), which has 4 m-states:
    # m = -3/2, -1/2, +1/2, +3/2
    # If we fill 2 particles in J=0 state, we expect <J^2> = 0
    
    # Create a simple 2-state system
    from read_snt_io import SingleParticleOrbits
    orbits = SingleParticleOrbits(0, 0, 2, 2)
    # Two proton orbits with j=3/2
    orbits.n[0] = 0
    orbits.l[0] = 2  # d-wave
    orbits.j[0] = 3  # 2j = 3
    orbits.tz[0] = -1  # proton
    
    orbits.n[1] = 0
    orbits.l[1] = 2
    orbits.j[1] = 1  # 2j = 1
    orbits.tz[1] = -1
    
    # Two neutron orbits
    orbits.n[2] = 0
    orbits.l[2] = 2
    orbits.j[2] = 3
    orbits.tz[2] = 1  # neutron
    
    orbits.n[3] = 0
    orbits.l[3] = 2
    orbits.j[3] = 1
    orbits.tz[3] = 1
    
    m_scheme = generate_m_scheme(orbits)
    print(f"\nM-scheme basis size: {len(m_scheme.n)}")
    
    j2_j = generate_j2_j_scheme(orbits)
    m_j2 = decouple_1b(j2_j, m_scheme)
    
    print("\nJ^2 1-body diagonal:")
    for i in range(len(m_scheme.n)):
        print(f"  State {i}: j={m_scheme.j[i]/2:.1f}, jz={m_scheme.jz[i]/2:.1f}, J^2={m_j2.v1b[i,i]:.3f}")


if __name__ == "__main__":
    test_j2_operator_transformation()
    simple_j2_test()
