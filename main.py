import numpy as np
from opt_einsum import contract
from read_snt_io import read_snt, generate_m_scheme, decouple_1b, decouple_2b,recouple_2b
from hf import hartree_fock, normal_order
from cc import ccsd, mp2, ccsd_diis_solver, ccd, ccd_diis_solver, ccsd_ode_solver,ccsdt

def main():
    snt_file = "gxpf1a.snt"
    #snt_file = "sd.snt"
    #snt_file = "p.snt"

    
    print("--- 1. reading snt interaction ---")
    scale_factor=1.0
    if(snt_file=="gxpf1a.snt"):
        scale_factor=(42./56)**(0.30)
    orbits, potential = read_snt(snt_file,scale_factor)
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
   # Z_val = 2
   # N_val = 2
   # occ_indices = [0,1,12,13]
    
    # Custom initial occupation (set to None for automatic energy-based)
    # Example: occ_indices = [0, 1, 6, 7] for Z=2, N=2
    #occ_indices = [4,5,6,7,8,9,16,17,18,19,20,21]
    #occ_indices = [4,5,16,17]
    if(snt_file=="gxpf1a.snt"):
        Z_val = 8
        N_val = 8
        occ_indices=[]
        for i in range(8):
            occ_indices.append(i)
        for i in range(8):
            occ_indices.append(i+20)

    if(snt_file=="p.snt"):
        Z_val = 4
        N_val = 4
        occ_indices=[]
        for i in range(4):
            occ_indices.append(i)
        for i in range(4):
            occ_indices.append(i+6)
    
    print(f"\n--- 4. Performing Hartree-Fock (Z={Z_val}, N={N_val}) ---")
    hf_energy, sp_energies, rho, sp_coeffs = hartree_fock(m_basis, potential, Z_val, N_val, 
                                                         v2b_sparse=v2b_m, occ_indices=occ_indices,mode='deformed')
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
        
        purity = sp_coeffs[dom_idx, i]**2
        print(f"{i:3d} | {status:4s} | {n:2d} {l:2d} {j:2d} {jz:3d} {tz:3d} | {sp_energies[i]:10.6f} | {purity:6.4f}")
    
    print("\n--- 5. Performing Normal Ordering ---")
    no_ham = normal_order(m_basis, potential, hf_energy, sp_energies, rho, sp_coeffs, v2b_sparse=v2b_m)
    print(no_ham)
    
    print("\n--- 6. Performing MP2 (Verification) ---")
    e_corr_mp2 = mp2(no_ham, n_occ)
    print(f"MP2 Correlation Energy: {e_corr_mp2:.6f} MeV")

#    print("\n--- CCD ---")
#    e_ccd, t2_ccd = ccd_diis_solver(no_ham, n_occ, max_iter=50, tol=1e-6 )
#    print(f"CCD correlation energy: {e_ccd:.6f} MeV")
#    print(f"Total CCD energy: {no_ham.E0 + e_ccd:.6f} MeV")
#    
#
#    print("\n--- 6a. Performing CCSD ")
#    e_corr_ccsd, t1_ccsd, t2_ccsd = ccsd_diis_solver(no_ham, n_occ, max_iter=50)
#    print(f"CCSD Correlation Energy: {e_corr_ccsd:.6f} MeV")
#    print(f"Total CCSD Energy: {no_ham.E0 + e_corr_ccsd:.6f} MeV")
#
#
#    print("\n--- 6b. Performing CCSDT ")
#    e_corr_ccsdt, t1_ccsdt, t2_ccsdt, t3_ccsdt = ccsdt(no_ham, n_occ, max_iter=50)
#    print(f"CCSD Correlation Energy: {e_corr_ccsdt:.6f} MeV")
#    print(f"Total CCSD Energy: {no_ham.E0 + e_corr_ccsdt:.6f} MeV")


if __name__ == "__main__":
    main()

#write a function to transform a operator with one body and twobody with \Lambda_1 a deexcitation operator
