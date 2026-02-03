
import numpy as np
import time
import os
import sys

# Ensure current directory is in path import search
sys.path.append(os.getcwd())

from read_snt_io import read_snt, generate_m_scheme, decouple_2b, decouple_1b
from hf import hartree_fock, normal_order
from cc import mp2, ccd_diis_solver, ccsd_diis_solver, ccdt, ccsdt

def test_ccsdt_nuclear():
    print("="*80)
    print("  Nuclear Coupled Cluster Benchmark: MP2, CCD, CCSD, CCDT, CCSDT")
    print("="*80)
    
    # Check for SNT files
#    snt_file = "sd.snt"
    snt_file = "gxpf1a.snt"
    if not os.path.exists(snt_file):
        snt_file = "gxpf1a.snt"
        
    if not os.path.exists(snt_file):
        print("Error: No .snt interaction files found!")
        return

    print(f"Reading interaction from: {snt_file}")
    try:
        orbits, potential = read_snt(snt_file)
    except Exception as e:
        print(f"Error reading snt file: {e}")
        return

    m_basis = generate_m_scheme(orbits)
    # print(f"Basis size: {len(m_basis)} states")
    
    # Decouple to m-scheme
    print("Decoupling 2-body interaction to m-scheme...")
    v1b = decouple_1b(potential, m_basis)
    v2b = decouple_2b(potential, m_basis)
    
    # Small test case (Be-8/He-4 in sd) or O-16 logic
    if "sd" in snt_file:
        Z_val = 4
        N_val = 4
        # Just pick first available orbitals roughly (sd shell starts at index 0 in sd.snt)
        occ_indices = list(range(4)) + [i+6 for i in range(4)]
        print(f"System: Z={Z_val}, N={N_val} (C12 in p shell)")
    else:
        Z_val = 8
        N_val = 8
        occ_indices = list(range(8)) + [i+20 for i in range(8)]
        print(f"System: Z={Z_val}, N={N_val} (Ni-56 in pf shell)")

    print(f"\nComputing Hartree-Fock...")
    t_hf_start = time.time()
    hf_E, sp_e, rho, sp_c = hartree_fock(m_basis, potential, Z_val, N_val, 
                                         v2b_sparse=v2b, occ_indices=occ_indices)
    print(f"HF Energy: {hf_E:.6f} MeV (Time: {time.time()-t_hf_start:.2f}s)")
    
    # Normal Order
    n_occ = Z_val + N_val
    no_ham = normal_order(m_basis, potential, hf_E, sp_e, rho, sp_c, v2b_sparse=v2b)
    
    results = {}
    
    print("\n" + "-"*80)
    print("Starting Correlation Calculations")
    print("-"*80)
    
    # 1. MP2
    print("Running MP2...")
    t0 = time.time()
    E_mp2 = mp2(no_ham, n_occ)
    results['MP2'] = {'E_corr': E_mp2, 'Total': hf_E + E_mp2, 'Time': time.time()-t0}
    print(f"MP2 Done: {E_mp2:.6f} MeV")

    # 2. CCD
    print("\nRunning CCD (DIIS)...")
    try:
        t0 = time.time()
        # ccd_diis_solver returns e_corr, t2
        E_ccd, t2_ccd = ccd_diis_solver(no_ham, n_occ, max_iter=50, tol=1e-6)
        results['CCD'] = {'E_corr': E_ccd, 'Total': hf_E + E_ccd, 'Time': time.time()-t0}
    except Exception as e:
        print(f"CCD Failed: {e}")
        
    # 3. CCSD
    print("\nRunning CCSD (DIIS)...")
    try:
        t0 = time.time()
        # ccsd_diis_solver returns e_corr, t1, t2
        E_ccsd, t1_ccsd, t2_ccsd = ccsd_diis_solver(no_ham, n_occ, max_iter=50, tol=1e-6)
        results['CCSD'] = {'E_corr': E_ccsd, 'Total': hf_E + E_ccsd, 'Time': time.time()-t0}
    except Exception as e:
        print(f"CCSD Failed: {e}")

    # 4. CCDT
    print("\nRunning CCDT (Doubles+Triples)...")
    try:
        t0 = time.time()
        # ccdt returns e_corr, t2, t3
        E_ccdt, t2_dt, t3_dt = ccdt(no_ham, n_occ, max_iter=30, tol=1e-6)
        results['CCDT'] = {'E_corr': E_ccdt, 'Total': hf_E + E_ccdt, 'Time': time.time()-t0}
    except Exception as e:
        print(f"CCDT Failed: {e}")

    # 5. CCSDT
    print("\nRunning CCSDT (Singles+Doubles+Triples)...")
    try:
        t0 = time.time()
        # ccsdt returns e_corr, t1, t2, t3
        E_ccsdt, t1, t2, t3 = ccsdt(no_ham, n_occ, max_iter=30, tol=1e-6)
        results['CCSDT'] = {'E_corr': E_ccsdt, 'Total': hf_E + E_ccsdt, 'Time': time.time()-t0}
    except Exception as e:
        print(f"CCSDT Failed: {e}")

    # =========================================================================
    # Summary Table
    # =========================================================================
    print("\n" + "="*85)
    print(f"{'Method':<10} | {'E_corr (MeV)':<15} | {'E_total (MeV)':<15} | {'ΔE vs CCSDT':<12} | {'Time (s)':<8}")
    print("-" * 85)
    
    ref_E = results.get('CCSDT', {}).get('E_corr', 0.0)
    if 'CCSDT' not in results:
        # Fallback if CCSDT failed
        ref_E = results.get('CCDT', {}).get('E_corr', 0.0)
    
    methods = ['MP2', 'CCD', 'CCSD', 'CCDT', 'CCSDT']
    
    for m in methods:
        if m in results:
            res = results[m]
            e_corr = res['E_corr']
            e_tot = res['Total']
            dt = res['Time']
            diff = abs(e_corr - ref_E)
            print(f"{m:<10} | {e_corr:15.6f} | {e_tot:15.6f} | {diff:12.6f} | {dt:8.2f}")
    print("="*85)

if __name__ == "__main__":
    test_ccsdt_nuclear()
