"""
Test CCDQ solver with real nuclear interactions from .snt files
"""

import numpy as np
from read_snt_io import read_snt, generate_m_scheme, decouple_1b, decouple_2b
from hf import hartree_fock, normal_order
from cc import mp2, ccd, ccsd
from ccdq import ccdq


def test_ccdq_nuclear():
    """
    Test CCDQ on a real nuclear system using sd-shell interaction
    """
    print("\n" + "="*80)
    print("  CCDQ Test with Nuclear Interaction")
    print("="*80)
    
    # Use sd-shell (smaller interaction for testing)
    #snt_file = "sd.snt"
    snt_file = "gxpf1a.snt"
    
    print(f"\n--- 1. Reading SNT Interaction: {snt_file} ---")
    orbits, potential = read_snt(snt_file)
    print(orbits)
    
    print("\n--- 2. Generating M-Scheme Basis ---")
    m_basis = generate_m_scheme(orbits)
    n_m_states = len(m_basis.n)
    print(f"Total m-scheme states: {n_m_states}")
    
    # Print basis states
    for i in range(min(n_m_states, 20)):  # Show first 20
        print(f"  {i:2d}: n={m_basis.n[i]}, l={m_basis.l[i]}, 2j={m_basis.j[i]}, "
              f"2jz={m_basis.jz[i]:3d}, 2tz={m_basis.tz[i]:3d}")
    if n_m_states > 20:
        print(f"  ... ({n_m_states - 20} more states)")
    
    print("\n--- 3. Decoupling Interaction to M-Scheme ---")
    v1b_m = decouple_1b(potential, m_basis)
    v2b_m = decouple_2b(potential, m_basis)
    print(f"1-body non-zero elements: {np.count_nonzero(v1b_m.v1b)}")
    print(f"2-body sparse matrix shape: {v2b_m.shape}")
    print(f"2-body non-zero elements: {v2b_m.nnz}")
    
    # Small system: He-4 (2 protons + 2 neutrons above O-16 core)
    # Actually, for sd-shell with O-16 core, let's do a minimal valence system
    Z_val = 4  # 2 valence protons
    N_val = 4  # 2 valence neutrons
    occ_indices=[]
    for i in range(4):
        occ_indices.append(i)
    for i in range(4):
        occ_indices.append(i+6)


    Z_val = 8  # 2 valence protons
    N_val = 8  # 2 valence neutrons
    occ_indices=[]
    for i in range(8):
        occ_indices.append(i)
    for i in range(8):
        occ_indices.append(i+20)
    
    print(f"\n--- 4. Hartree-Fock Solution (Z={Z_val}, N={N_val}) ---")
    hf_energy, sp_energies, rho, sp_coeffs = hartree_fock(
        m_basis, potential, Z_val, N_val, v2b_sparse=v2b_m,occ_indices=occ_indices
    )
    print(f"HF Energy: {hf_energy:.6f} MeV")
    
    n_occ = Z_val + N_val
    print(f"\nSingle-Particle Energies (first {min(12, len(sp_energies))} states):")
    print(f"{'Idx':>3} | {'Type':>4} | {'Energy':>10} | {'Dominant n l 2j 2jz 2tz'}")
    print("-" * 60)
    
    for i in range(min(12, len(sp_energies))):
        status = "OCC" if i < n_occ else "VIR"
        # Find dominant component
        dom_idx = np.argmax(np.abs(sp_coeffs[:, i]))
        n = m_basis.n[dom_idx]
        l = m_basis.l[dom_idx]
        j = m_basis.j[dom_idx]
        jz = m_basis.jz[dom_idx]
        tz = m_basis.tz[dom_idx]
        
        print(f"{i:3d} | {status:4s} | {sp_energies[i]:10.6f} | "
              f"{n:2d} {l:2d} {j:2d} {jz:3d} {tz:3d}")
    
    print("\n--- 5. Normal Ordering Hamiltonian ---")
    no_ham = normal_order(m_basis, potential, hf_energy, sp_energies, rho, sp_coeffs, v2b_sparse=v2b_m)
    print(no_ham)
    
    # Double-check dimensions
    print(f"\nNormal-ordered Hamiltonian dimensions:")
    print(f"  Fock matrix f: {no_ham.f.shape}")
    print(f"  2-body Gamma: {no_ham.Gamma.shape}")
    print(f"  Number of occupied states: {n_occ}")
    print(f"  Number of virtual states: {no_ham.f.shape[0] - n_occ}")
    
    print("\n" + "="*80)
    print("  Correlation Energy Calculations")
    print("="*80)
    
    # MP2 (baseline)
    print("\n--- MP2 (2nd-order perturbation theory) ---")
    e_mp2 = mp2(no_ham, n_occ)
    print(f"MP2 Correlation Energy:  {e_mp2:12.6f} MeV")
    print(f"Total MP2 Energy:        {no_ham.E0 + e_mp2:12.6f} MeV")
    
    # CCD (doubles only)
    print("\n--- CCD (Coupled Cluster Doubles) ---")
    e_ccd, t2_ccd = ccd(no_ham, n_occ, max_iter=30, tol=1e-6, alpha=0.5, use_sparse=True)
    print(f"CCD Correlation Energy:  {e_ccd:12.6f} MeV")
    print(f"Total CCD Energy:        {no_ham.E0 + e_ccd:12.6f} MeV")
    
    # CCSD (singles + doubles)
    print("\n--- CCSD (Coupled Cluster Singles + Doubles) ---")
    e_ccsd, t1_ccsd, t2_ccsd = ccsd(no_ham, n_occ, max_iter=30, tol=1e-6, alpha=0.5, use_sparse=True)
    print(f"CCSD Correlation Energy: {e_ccsd:12.6f} MeV")
    print(f"Total CCSD Energy:       {no_ham.E0 + e_ccsd:12.6f} MeV")
    
    # CCDQ (doubles + quadruples) - THE MAIN TEST!
    print("\n" + "="*80)
    print("--- CCDQ (Coupled Cluster Doubles + Quadruples) ---")
    print("="*80)
    
    try:
        e_ccdq, t2_ccdq, t4_ccdq = ccdq(
            no_ham, 
            n_occ, 
            max_iter=100,          # More iterations for T4 convergence
            tol=1e-6,             # Convergence tolerance
            alpha=0.05,           # VERY conservative damping for stability
            use_sparse=True,      # Enable sparse optimization
            sparse_threshold=1e-10,  # Sparsity threshold
            print_level=1,        # Show iteration details
            initial_t2=t2_ccd     # Initialize with converged CCD amplitudes
        )
        
        print(f"\n{'='*80}")
        print(" CCDQ Results")
        print(f"{'='*80}")
        print(f"CCDQ Correlation Energy: {e_ccdq:12.6f} MeV")
        print(f"Total CCDQ Energy:       {no_ham.E0 + e_ccdq:12.6f} MeV")
        
        # Analysis
        print(f"\n{'='*80}")
        print(" Correlation Energy Comparison")
        print(f"{'='*80}")
        print(f"{'Method':<10} | {'E_corr (MeV)':>14} | {'Total E (MeV)':>14} | {'vs MP2':>10}")
        print("-" * 65)
        print(f"{'MP2':<10} | {e_mp2:14.6f} | {no_ham.E0 + e_mp2:14.6f} | {0.0:10.6f}")
        print(f"{'CCD':<10} | {e_ccd:14.6f} | {no_ham.E0 + e_ccd:14.6f} | {e_ccd - e_mp2:10.6f}")
        print(f"{'CCSD':<10} | {e_ccsd:14.6f} | {no_ham.E0 + e_ccsd:14.6f} | {e_ccsd - e_mp2:10.6f}")
        print(f"{'CCDQ':<10} | {e_ccdq:14.6f} | {no_ham.E0 + e_ccdq:14.6f} | {e_ccdq - e_mp2:10.6f}")
        
        print(f"\n{'='*80}")
        print(" Amplitude Statistics")
        print(f"{'='*80}")
        print(f"T2 (CCDQ) max: {np.max(np.abs(t2_ccdq)):.6f}, norm: {np.linalg.norm(t2_ccdq):.6f}")
        print(f"T4 (CCDQ) max: {np.max(np.abs(t4_ccdq)):.6f}, norm: {np.linalg.norm(t4_ccdq):.6f}")
        
        # Sparsity analysis
        t2_sparse = np.sum(np.abs(t2_ccdq) < 1e-10) / t2_ccdq.size
        t4_sparse = np.sum(np.abs(t4_ccdq) < 1e-10) / t4_ccdq.size
        print(f"\nSparsity:")
        print(f"  T2: {t2_sparse*100:.1f}% elements below threshold")
        print(f"  T4: {t4_sparse*100:.1f}% elements below threshold")
        
        print(f"\n✅ CCDQ TEST PASSED!")
        
    except Exception as e:
        print(f"\n❌ CCDQ TEST FAILED!")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    print(f"\n{'='*80}\n")
    

def test_ccdq_gxpf1a():
    """
    Test CCDQ on gxpf1a interaction (larger system)
    Warning: This will be slow due to larger basis!
    """
    print("\n" + "="*80)
    print("  CCDQ Test with GXPF1A Interaction (Larger System)")
    print("="*80)
    
    snt_file = "gxpf1a.snt"
    
    print(f"\n--- 1. Reading SNT Interaction: {snt_file} ---")
    orbits, potential = read_snt(snt_file)
    print(orbits)
    
    print("\n--- 2. Generating M-Scheme Basis ---")
    m_basis = generate_m_scheme(orbits)
    n_m_states = len(m_basis.n)
    print(f"Total m-scheme states: {n_m_states}")
    print("⚠️  Warning: Large basis will make CCDQ very slow!")
    
    # Very small valence system
    Z_val = 2
    N_val = 2
    
    print(f"\n--- 3-5. Running HF and Normal Ordering ---")
    v1b_m = decouple_1b(potential, m_basis)
    v2b_m = decouple_2b(potential, m_basis)
    
    hf_energy, sp_energies, rho, sp_coeffs = hartree_fock(
        m_basis, potential, Z_val, N_val, v2b_sparse=v2b_m
    )
    print(f"HF Energy: {hf_energy:.6f} MeV")
    
    n_occ = Z_val + N_val
    no_ham = normal_order(m_basis, potential, hf_energy, sp_energies, rho, sp_coeffs, v2b_sparse=v2b_m)
    
    print(f"\n--- CCDQ Calculation ---")
    print(f"Occupied: {n_occ}, Virtual: {no_ham.f.shape[0] - n_occ}")
    print(f"T4 tensor size: O^4 × V^4 = {n_occ**4} × {(no_ham.f.shape[0] - n_occ)**4}")
    
    e_ccdq, t2_ccdq, t4_ccdq = ccdq(
        no_ham, n_occ, max_iter=30, tol=1e-5, alpha=0.3, 
        use_sparse=True, print_level=1
    )
    
    print(f"\nCCDQ Correlation Energy: {e_ccdq:.6f} MeV")
    print(f"Total CCDQ Energy:       {no_ham.E0 + e_ccdq:.6f} MeV")
    print(f"\n✅ GXPF1A CCDQ TEST PASSED!")


if __name__ == "__main__":
    # Run the main test with sd-shell
    test_ccdq_nuclear()
    
    # Uncomment to test larger system (warning: slow!)
    # test_ccdq_gxpf1a()
