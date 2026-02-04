#!/usr/bin/env python3
"""
Test script for CCSDTQ tenpi solver
Demonstrates warm starting from CCSD and CCSDT
"""

import numpy as np
from read_snt_io import read_snt, generate_m_scheme, decouple_2b
from hf import hartree_fock, normal_order
from cc import ccsd_diis_solver
from ccd_tenpi_solver import ccd_tenpi_solver
from ccsd_tenpi_solver import ccsd_tenpi_solver
from ccsdt_tenpi_solver import ccsdt_tenpi_solver
from ccsdtq_tenpi_solver import ccsdtq_tenpi_solver
from ccdtq_tenpi_solver import ccdtq_tenpi_solver
from ccdq_tenpi_solver import ccdq_tenpi_solver



def test_ccsdtq_tenpi():
    """Test the tenpi CCSDTQ and CCDTQ solvers with warm start"""
    
    print("=" * 80)
    print("Testing CCSDTQ & CCDTQ Tenpi Solvers")
    print("=" * 80)
    
    # ... (rest of the setup unchanged) ...
    # Read interaction
    snt_file = "p.snt"
    print(f"\n1. Reading interaction from {snt_file}")
    orbits, potential = read_snt(snt_file)
    
    # Generate m-scheme basis
    print("\n2. Generating m-scheme basis")
    m_basis = generate_m_scheme(orbits)
    print(f"   Total m-scheme states: {len(m_basis.n)}")
    
    # Decouple interaction
    print("\n3. Decoupling interaction to m-scheme")
    v2b_m = decouple_2b(potential, m_basis)
    
    # Set up occupation
    Z_val = 4
    N_val = 4
    occ_indices = []
    for i in range(4):
        occ_indices.append(i)
    for i in range(4):
        occ_indices.append(i + 6)
    
    print(f"\n4. Running Hartree-Fock (Z={Z_val}, N={N_val})")
    hf_energy, sp_energies, rho, sp_coeffs = hartree_fock(
        m_basis, potential, Z_val, N_val, 
        v2b_sparse=v2b_m, occ_indices=occ_indices
    )
    print(f"   HF Energy: {hf_energy:.6f} MeV")
    
    # Normal order
    print("\n5. Performing normal ordering")
    no_ham = normal_order(m_basis, potential, hf_energy, sp_energies, rho, sp_coeffs, v2b_sparse=v2b_m)
    print(f"   {no_ham}")
    
    n_occ = Z_val + N_val
    
    # Run CCSD for initial guess
    print(f"\n6. Running CCSD (for initial guess)")
    print("-" * 80)
    e_ccsd, t1_ccsd, t2_ccsd = ccsd_tenpi_solver(
            no_ham, n_occ, 
            max_iter=100, 
            tol=1e-6)
    print(f"   CCSD Correlation: {e_ccsd:.6f} MeV")
    print(f"   Total CCSD Energy: {no_ham.E0 + e_ccsd:.6f} MeV")
    
    # Run CCSDT for better initial guess
    print(f"\n7. Running CCSDT (for better initial guess)")
    print("-" * 80)
    e_ccsdt, t1_ccsdt, t2_ccsdt, t3_ccsdt = ccsdt_tenpi_solver(
        no_ham, n_occ,
        max_iter=25,
        tol=1e-6,
        diis_size=4,
        diis_start=3,
        initial_t1=t1_ccsd,
        initial_t2=t2_ccsd
    )
    print(f"   CCSDT Correlation: {e_ccsdt:.6f} MeV")
    print(f"   Total CCSDT Energy: {no_ham.E0 + e_ccsdt:.6f} MeV")
    
#    # Run CCDTQ
#    print(f"\n8. Running CCDTQ (tenpi solver)")
#    print("=" * 80)
#    e_ccdtq, t2_ccdtq, t3_ccdtq, t4_ccdtq = ccdtq_tenpi_solver(
#        no_ham, n_occ,
#        max_iter=25,
#        tol=1e-5,
#        diis_size=3,
#        diis_start=2,
#        initial_t2=t2_ccsd,
#        initial_t3=t3_ccsdt  # Warm start from CCSDT for T3!
#    )
#    print(f"   CCDTQ Correlation: {e_ccdtq:.6f} MeV")
#    print(f"   Total CCDTQ Energy: {no_ham.E0 + e_ccdtq:.6f} MeV")
#    
    # Run CCDQ
    print(f"\n9. Running CCDQ (tenpi solver)")
    print("=" * 80)
    e_ccdq, t2_ccdq, t4_ccdq = ccdq_tenpi_solver(
        no_ham, n_occ,
        max_iter=25,
        tol=1e-5,
        diis_size=3,
        diis_start=2,
        initial_t2=t2_ccsd
    )
    print(f"   CCDQ Correlation: {e_ccdq:.6f} MeV")
    print(f"   Total CCDQ Energy: {no_ham.E0 + e_ccdq:.6f} MeV")
#    
#    # Run CCSDTQ
#    print(f"\n10. Running CCSDTQ (tenpi solver) - THIS WILL BE SLOW!")
#    print("=" * 80)
    
#    e_ccsdtq, t1, t2, t3, t4 = ccsdtq_tenpi_solver(
#        no_ham, n_occ,
#        max_iter=25,  # Keep small for testing
#        tol=1e-5,     # Slightly relaxed tolerance
#        diis_size=3,  # Smaller DIIS for memory
#        diis_start=2,
#        initial_t1=t1_ccsd,
#        initial_t2=t2_ccsd,
#        initial_t3=t3_ccsdt  # Warm start from CCSDT!
#    )
    
    print("=" * 80)
    print(f"\nResults Summary:")
    print(f"  HF Energy:            {no_ham.E0:.6f} MeV")
    print(f"  CCSD Correlation:     {e_ccsd:.6f} MeV")
    print(f"  CCSDT Correlation:    {e_ccsdt:.6f} MeV")
    print(f"  CCDTQ Correlation:    {e_ccdtq:.6f} MeV")
    print(f"  CCDQ Correlation:     {e_ccdq:.6f} MeV")
    print(f"  CCSDTQ Correlation:   {e_ccsdtq:.6f} MeV")
    print(f"  Total CCSDTQ Energy:  {no_ham.E0 + e_ccsdtq:.6f} MeV")
    
    print(f"\nHigher-order corrections:")
    print(f"  (T) correction:       {e_ccsdt - e_ccsd:.6f} MeV ({100*abs(e_ccsdt - e_ccsd)/abs(e_ccsdt):.2f}%)")
    print(f"  (Q) correction (CCDTQ): {e_ccdtq - e_ccsdt:.6f} MeV")
    print(f"  (Q) correction (CCSDTQ): {e_ccsdtq - e_ccsdt:.6f} MeV ({100*abs(e_ccsdtq - e_ccsdt)/abs(e_ccsdtq):.2f}%)")
    
    print(f"\nAmplitude Statistics:")
    print(f"  Max |t1|:             {np.max(np.abs(t1)):.6f}")
    print(f"  Max |t2|:             {np.max(np.abs(t2)):.6f}")
    print(f"  Max |t3|:             {np.max(np.abs(t3)):.6f}")
    print(f"  Max |t4|:             {np.max(np.abs(t4)):.6f}")
    print(f"  RMS |t4|:             {np.sqrt(np.mean(t4**2)):.6f}")
    print(f"  Non-zero T4 (>1e-8): {np.count_nonzero(np.abs(t4) > 1e-8):,} / {t4.size:,}")
    
    print("\n" + "=" * 80)
    print("Test completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    test_ccsdtq_tenpi()
