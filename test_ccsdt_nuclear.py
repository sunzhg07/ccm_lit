
import numpy as np
import time
import os
import sys

# Ensure current directory is in path import search
sys.path.append(os.getcwd())

from read_snt_io import read_snt, generate_m_scheme, decouple_2b, decouple_1b
from hf import hartree_fock, normal_order
from cc import mp2, ccd_diis_solver, ccsd_diis_solver, ccdt, ccsdt
from ccsdtq_complete import ccsdtq
from ccdtq import ccdtq

def test_ccsdt_nuclear():
    print("="*80)
    print("  Nuclear Coupled Cluster Benchmark: MP2, CCD, CCSD, CCDT, CCSDT, CCDTQ, CCSDTQ")
    print("="*80)
    
    # Check for SNT files
    snt_file = "sd.snt"
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
        print(f"System: Z={Z_val}, N={N_val} (Be-8/24Mg model)")
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
        E_ccd, t2_ccd = ccd_diis_solver(no_ham, n_occ, max_iter=50, tol=1e-6)
        results['CCD'] = {'E_corr': E_ccd, 'Total': hf_E + E_ccd, 'Time': time.time()-t0}
    except Exception as e:
        print(f"CCD Failed: {e}")
        
    # 3. CCSD
    print("\nRunning CCSD (DIIS)...")
    try:
        t0 = time.time()
        E_ccsd, t1_ccsd, t2_ccsd = ccsd_diis_solver(no_ham, n_occ, max_iter=50, tol=1e-6)
        results['CCSD'] = {'E_corr': E_ccsd, 'Total': hf_E + E_ccsd, 'Time': time.time()-t0}
    except Exception as e:
        print(f"CCSD Failed: {e}")

    # 4. CCDT
    print("\nRunning CCDT (Doubles+Triples) [Restart from CCSD]...")
    try:
        t0 = time.time()
        E_ccdt, t2_dt, t3_dt = ccdt(no_ham, n_occ, max_iter=30, tol=1e-6, initial_t2=t2_ccsd)
        results['CCDT'] = {'E_corr': E_ccdt, 'Total': hf_E + E_ccdt, 'Time': time.time()-t0}
    except Exception as e:
        print(f"CCDT Failed: {e}")

#    # 5. CCSDT
#    print("\nRunning CCSDT (Singles+Doubles+Triples) [Restart from CCSD]...")
#    try:
#        t0 = time.time()
#        E_ccsdt, t1, t2, t3 = ccsdt(no_ham, n_occ, max_iter=30, tol=1e-6, initial_t1=t1_ccsd, initial_t2=t2_ccsd)
#        results['CCSDT'] = {'E_corr': E_ccsdt, 'Total': hf_E + E_ccsdt, 'Time': time.time()-t0}
#    except Exception as e:
#        print(f"CCSDT Failed: {e}")

    # 6. CCDTQ (New)
    print("\nRunning CCDTQ (S+D+T+Q, No T1) [Restart from CCSDT]...")
    n_virt = no_ham.f.shape[0] - n_occ
    t4_mem_gb = (n_occ**4 * n_virt**4 * 8) / 1e9
    if t4_mem_gb > 24.0:
        print(f"Skipping CCDTQ: Estimated T4 memory {t4_mem_gb:.2f} GB exceeds limit.")
    else:
        try:
            t0 = time.time()
            i_t2 = locals().get('t2', None)
            i_t3 = locals().get('t3', None)
            E_ccdtq, _, _, _ = ccdtq(no_ham, n_occ, max_iter=30, tol=1e-6, initial_t2=i_t2, initial_t3=i_t3)
            results['CCDTQ'] = {'E_corr': E_ccdtq, 'Total': hf_E + E_ccdtq, 'Time': time.time()-t0}
        except Exception as e:
            print(f"CCDTQ Failed: {e}")

#    # 7. CCSDTQ
#    print("\nRunning CCSDTQ (Full) [Restart from CCSDT]...")
#    if t4_mem_gb > 24.0:
#        print(f"Skipping CCSDTQ: Estimated T4 memory {t4_mem_gb:.2f} GB exceeds limit.")
#    else:
#        try:
#            t0 = time.time()
#            i_t1 = locals().get('t1', None)
#            i_t2 = locals().get('t2', None)
#            i_t3 = locals().get('t3', None)
#
#            E_ccsdtq, _, _, _, _ = ccsdtq(no_ham, n_occ, max_iter=30, tol=1e-6, 
#                                          initial_t1=i_t1, initial_t2=i_t2, initial_t3=i_t3)
#            results['CCSDTQ'] = {'E_corr': E_ccsdtq, 'Total': hf_E + E_ccsdtq, 'Time': time.time()-t0}
#        except Exception as e:
#            print(f"CCSDTQ Failed: {e}")
#
    # =========================================================================
    # Summary Table
    # =========================================================================
    print("\n" + "="*95)
    print(f"{'Method':<10} | {'E_corr (MeV)':<15} | {'E_total (MeV)':<15} | {'ΔE vs Ref':<12} | {'Time (s)':<8}")
    print("-" * 95)
    
    # Establish reference
    ref_method = 'CCSDT'
    if 'CCSDTQ' in results: ref_method = 'CCSDTQ'
    elif 'CCDTQ' in results: ref_method = 'CCDTQ'
    
    ref_E = results.get(ref_method, {}).get('E_corr', 0.0)
    print(f"(Reference method: {ref_method})")
    
    methods = ['MP2', 'CCD', 'CCSD', 'CCDT', 'CCSDT', 'CCDTQ', 'CCSDTQ']
    
    for m in methods:
        if m in results:
            res = results[m]
            e_corr = res['E_corr']
            e_tot = res['Total']
            dt = res['Time']
            diff = abs(e_corr - ref_E)
            print(f"{m:<10} | {e_corr:15.6f} | {e_tot:15.6f} | {diff:12.6f} | {dt:8.2f}")
    print("="*95)

if __name__ == "__main__":
    test_ccsdt_nuclear()
