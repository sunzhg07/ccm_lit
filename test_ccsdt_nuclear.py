
import numpy as np
import time
import os
import sys

# Ensure current directory is in path import search
sys.path.append(os.getcwd())

from read_snt_io import read_snt, generate_m_scheme, decouple_2b,decouple_1b
from hf import hartree_fock, normal_order
from ccsdt import ccsdt
from cc import mp2

def test_ccsdt_nuclear():
    print("---------------------------------------------------------")
    print(" Testing CCSDT with Real Nuclear Hamiltonian (USDB/SD) ")
    print("---------------------------------------------------------")
    
    # Check for SNT files
    snt_file = "gxpf1a.snt"
    if not os.path.exists(snt_file):
        snt_file = "sd.snt"
    
    if not os.path.exists(snt_file):
        print("Error: No .snt interaction files found!")
        return

    print(f"Reading {snt_file}...")
    try:
        orbits, potential = read_snt(snt_file)
    except Exception as e:
        print(f"Error reading snt file: {e}")
        return

    m_basis = generate_m_scheme(orbits)
    
    # Decouple to m-scheme
    print("Decoupling 2-body interaction...")
    v1b = decouple_1b(potential, m_basis)
    v2b = decouple_2b(potential, m_basis)
    

    Z_val = 8
    N_val = 8
    occ_indices=[]
    for i in range(8):
        occ_indices.append(i)
    for i in range(8):
        occ_indices.append(i+20)

    
    print(f"\nComputing Hartree-Fock (Z={Z_val}, N={N_val})...")
    hf_E, sp_e, rho, sp_c = hartree_fock(m_basis, potential, Z_val, N_val, 
                                                         v2b_sparse=v2b, occ_indices=occ_indices)
    print(f"HF Energy: {hf_E:.6f} MeV")
    
    # Normal Order
    n_occ = Z_val + N_val
    no_ham = normal_order(m_basis, potential, hf_E, sp_e, rho, sp_c, v2b_sparse=v2b)
    
    print("\n--- Correlation Methods ---")
    
    # MP2
    E_mp2 = mp2(no_ham, n_occ)
    print(f"MP2:   {E_mp2:.6f} MeV")
    
    # CCSDT
    print("\nRunning CCSDT (max_iter=30, alpha=0.5)...")
    start = time.time()
    try:
        E_ccsdt, t1, t2, t3 = ccsdt(no_ham, n_occ, max_iter=30, tol=1e-5, alpha=0.5)
        end = time.time()
        
        print(f"\nCCSDT Result:")
        print(f"  Corr. Energy: {E_ccsdt:.6f} MeV")
        print(f"  Total Energy: {hf_E + E_ccsdt:.6f} MeV")
        print(f"  Time Taken:   {end - start:.2f} s")
        print(f"  T3 Norm:      {np.linalg.norm(t3):.6f}")
        
    except Exception as e:
        print(f"\nCCSDT Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_ccsdt_nuclear()
