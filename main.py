import numpy as np
from read_snt_io import read_snt, generate_m_scheme, decouple_1b, decouple_2b,recouple_2b
from hf import hartree_fock, normal_order
from cc import ccsd, ccd, mp2
from lambda_ccsd import lambda_ccsd, compute_properties

def main():
    snt_file = "usdb.snt"
    
    print("--- 1. Reading SNT Interaction ---")
    orbits, potential = read_snt(snt_file)
    print(orbits)
    
    print("\n--- 2. Generating M-Scheme Basis ---")
    m_basis = generate_m_scheme(orbits)
    print(f"Total m-scheme states: {len(m_basis.n)}")
    for i in range(len(m_basis.n)):
        print(f"  {i:2d}: n={m_basis.n[i]}, l={m_basis.l[i]}, j={m_basis.j[i]}, jz={m_basis.jz[i]:2d}, tz={m_basis.tz[i]:2d}")
        
    print("\n--- 3. Decoupling Interaction to M-Scheme ---")
    # decouple_1b and decouple_2b are used inside hartree_fock too, but let's test them here
    v1b_m = decouple_1b(potential, m_basis)
    v2b_m = decouple_2b(potential, m_basis)
    print(v1b_m.v1b)
    print(f"m-scheme v1b non-zero: {np.count_nonzero(v1b_m.v1b)}")
    print(f"m-scheme v2b shape: {v2b_m.shape}")
    
    # Z=2, N=2 for Ne-20 relative to O16 core
    Z_val = 2
    N_val = 2
    
    # Custom initial occupation (set to None for automatic energy-based)
    # Example: occ_indices = [0, 1, 6, 7] for Z=2, N=2
    #occ_indices = [4,5,6,7,8,9,16,17,18,19,20,21]

    occ_indices = [4,5,16,17]
    
    print(f"\n--- 4. Performing Hartree-Fock (Z={Z_val}, N={N_val}) ---")
    hf_energy, sp_energies, rho, sp_coeffs = hartree_fock(m_basis, potential, Z_val, N_val, 
                                                         v2b_sparse=v2b_m, occ_indices=occ_indices)
    print(f"HF Energy: {hf_energy:.6f} MeV")
    
    print("\nFinal Single Particle Orbits (HF Basis):")
    n_occ = Z_val + N_val
    
    print(f"{'Idx':>3} | {'Type':>4} | {'n':>2} {'l':>2} {'2j':>2} {'2jz':>3} {'2tz':>3} | {'Energy':>10}")
    print("-" * 55)
    for i in range(len(sp_energies)):
        status = "H" if i < n_occ else "P"
        
        # Find the dominant original m-scheme state to labels the HF state
        dom_idx = np.argmax(np.abs(sp_coeffs[:, i]))
        n = m_basis.n[dom_idx]
        l = m_basis.l[dom_idx]
        j = m_basis.j[dom_idx]
        jz = m_basis.jz[dom_idx]
        tz = m_basis.tz[dom_idx]
        
        print(f"{i:3d} | {status:4s} | {n:2d} {l:2d} {j:2d} {jz:3d} {tz:3d} | {sp_energies[i]:10.6f}")
    
    print("\n--- 5. Performing Normal Ordering ---")
    no_ham = normal_order(m_basis, potential, hf_energy, sp_energies, rho, sp_coeffs, v2b_sparse=v2b_m)
    print(no_ham)
    
    print("\n--- 6. Performing MP2 (Verification) ---")
    e_corr_mp2 = mp2(no_ham, n_occ)
    print(f"MP2 Correlation Energy: {e_corr_mp2:.6f} MeV")

    print("\n--- 6a. Performing CCD (Diagnostic) ---")
    e_corr_ccd, t2_ccd = ccd(no_ham, n_occ, max_iter=50)
    print(f"CCD Correlation Energy: {e_corr_ccd:.6f} MeV")

    print("\n--- 6b. Performing CCSD ---")
    e_corr, t1, t2 = ccsd(no_ham, n_occ)
    
    print(f"\nCCSD Correlation Energy: {e_corr:.6f} MeV")
    print(f"Total CCSD Energy: {no_ham.E0 + e_corr:.6f} MeV")

    print("\n--- 6c. Solving Lambda-CCSD Equations ---")
    l1, l2 = lambda_ccsd(no_ham, t1, t2, n_occ, max_iter=100)
    
    print("\n--- 6d. Computing Properties with Lambda Amplitudes ---")
    props = compute_properties(no_ham, t1, t2, l1, l2, n_occ)
    print(f"One-body expectation: {props['f_expectation']:.6f} MeV")
    print(f"Particle number: {props['n_particle']:.6f}")
    print(f"Hole density trace: {np.trace(props['rho_oo']):.6f}")
    print(f"Particle density trace: {np.trace(props['rho_vv']):.6f}")

    print("\n--- 7. Verifying J-Scheme Reconstruction ---")
    print("Recoupling M-scheme interaction back to J-scheme...")
    recoupled = recouple_2b(v2b_m, m_basis, potential)
    
    print(f"{'Orbits (r s t u)':<15} | {'J':<2} | {'Original':<12} | {'Recoupled':<12} | {'Diff':<10}")
    print("-" * 65)
    max_diff = 0
    for i, entry in enumerate(potential.v2b):
        orig = entry[5]
        reco = recoupled[i][5]
        diff = abs(orig - reco)
        max_diff = max(max_diff, diff)
        if i < 10 or diff > 1e-6: # Print first 10 or any with significant difference
            label = f"{entry[0]+1},{entry[1]+1} {entry[2]+1},{entry[3]+1}"
            print(f"{label:<15} | {entry[4]:<2} | {orig:12.6f} | {reco:12.6f} | {diff:10.2e}")
            
    print(f"\nMax Difference in J-scheme reconstruction: {max_diff:.2e}")

if __name__ == "__main__":
    main()
