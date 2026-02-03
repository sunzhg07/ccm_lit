#!/usr/bin/env python3
"""
Test CCSDT with real nuclear Hamiltonian - Compare optimized vs original
"""
import os
import sys
import numpy as np
import time

# Ensure current directory is in path import search
sys.path.append(os.getcwd())

from read_snt_io import read_snt, generate_m_scheme, decouple_2b, decouple_1b
from hf import hartree_fock, normal_order
from ccsdt import ccsdt
from ccsdt_optimized import ccsdt_optimized
from cc import mp2

def test_ccsdt_comparison():
    print("---------------------------------------------------------")
    print(" Comparing CCSDT: Original vs Optimized Implementation ")
    print("---------------------------------------------------------")
    
    # Check for SNT files
    snt_file = "gxpf1a.snt"
    if not os.path.exists(snt_file):
        snt_file = "usdb.snt"
        if not os.path.exists(snt_file):
            snt_file = "sd.snt"
    
    if not os.path.exists(snt_file):
        print(f"Error: No .snt file found!")
        return
    
    print(f"Reading {snt_file}...")
    potential = read_snt(snt_file)
    
    # Generate m-scheme basis
    m_basis = generate_m_scheme(potential)
    print(f"M-scheme basis: {len(m_basis)} states")
    
    # Decouple to m-scheme
    print("Decoupling 2-body interaction...")
    v1b = decouple_1b(potential, m_basis)
    v2b = decouple_2b(potential, m_basis)
    
    Z_val = 8
    N_val = 8
    occ_indices = []
    for i in range(8):
        occ_indices.append(i)
    for i in range(8):
        occ_indices.append(i+20)
    
    print(f"\nComputing Hartree-Fock (Z={Z_val}, N={N_val})...")
    hf_E, sp_e, rho, sp_coeffs = hartree_fock(m_basis, potential, Z_val, N_val, 
                                              v2b_sparse=v2b, occ_indices=occ_indices)
    print(f"HF Energy: {hf_E:.6f} MeV")
    
    # Normal Order
    print(f"Transforming 2-body interaction to HF basis (N^5 optimized)...")
    no_ham = normal_order(sp_e, v2b, rho, n_states=len(m_basis))
    
    n_occ = Z_val + N_val
    print(f"\n--- Correlation Methods ---")
    E_mp2 = mp2(no_ham, n_occ)
    print(f"MP2:   {E_mp2:.6f} MeV")
    
    # ========== Original CCSDT ==========
    print(f"\n========== ORIGINAL CCSDT ==========")
    print(f"Running CCSDT (max_iter=5, alpha=0.5)...")
    t_start = time.time()
    try:
        E_ccsdt_orig, t1_o, t2_o, t3_o = ccsdt(no_ham, n_occ, max_iter=5, tol=1e-5, alpha=0.5)
        t_orig = time.time() - t_start
        print(f"\nOriginal CCSDT Result:")
        print(f"  Corr. Energy: {E_ccsdt_orig:.6f} MeV")
        print(f"  Total Energy: {hf_E + E_ccsdt_orig:.6f} MeV")
        print(f"  Time Taken:   {t_orig:.2f} s")
        print(f"  T3 Norm:      {np.linalg.norm(t3_o):.6f}")
    except Exception as e:
        print(f"Original CCSDT Failed: {e}")
        return
    
    # ========== Optimized CCSDT ==========
    print(f"\n========== OPTIMIZED CCSDT ==========")
    print(f"Running CCSDT-Optimized (max_iter=5, alpha=0.5)...")
    t_start = time.time()
    try:
        E_ccsdt_opt, t1_opt, t2_opt, t3_opt = ccsdt_optimized(no_ham, n_occ, max_iter=5, tol=1e-5, alpha=0.5)
        t_opt = time.time() - t_start
        print(f"\nOptimized CCSDT Result:")
        print(f"  Corr. Energy: {E_ccsdt_opt:.6f} MeV")
        print(f"  Total Energy: {hf_E + E_ccsdt_opt:.6f} MeV")
        print(f"  Time Taken:   {t_opt:.2f} s")
        print(f"  T3 Norm:      {np.linalg.norm(t3_opt):.6f}")
    except Exception as e:
        print(f"Optimized CCSDT Failed: {e}")
        return
    
    # ========== Comparison ==========
    print(f"\n========== COMPARISON ==========")
    print(f"Energy Difference: {abs(E_ccsdt_orig - E_ccsdt_opt):.2e} MeV")
    print(f"T1 Difference:     {np.linalg.norm(t1_o - t1_opt):.2e}")
    print(f"T2 Difference:     {np.linalg.norm(t2_o - t2_opt):.2e}")
    print(f"T3 Difference:     {np.linalg.norm(t3_o - t3_opt):.2e}")
    print(f"Speedup Factor:    {t_orig/t_opt:.2f}x")
    
if __name__ == "__main__":
    test_ccsdt_comparison()
