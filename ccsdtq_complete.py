"""
COMPLETE CCSDTQ SOLVER WITH ALL 180 AUTO-GENERATED EQUATIONS
All equations parsed from ccsdtq_equations.tex are now integrated
"""

import numpy as np
from opt_einsum import contract
import time
import sys

# Import complete generated residuals
sys.path.insert(0, '/Users/wolf/work/ccm_lit')
from ccsdtq_residuals_complete import (
    compute_ccsdtq_t1_residual_COMPLETE,
    compute_ccsdtq_t2_residual_COMPLETE,
    compute_ccsdtq_t3_residual_COMPLETE,
    compute_ccsdtq_t4_residual_COMPLETE
)

def ccsdtq_complete(no_ham, n_occ, max_iter=20, tol=1e-8, alpha=0.3, diis_size=8,
           diis_start_iter=3, initial_t1=None, initial_t2=None, initial_t3=None):
    """
    COMPLETE CCSDTQ with ALL 180 AUTO-GENERATED EQUATIONS
    
    Based on equations from ccsdtq_equations.tex:
    - T1: 15 equations (all parsed terms)
    - T2: 38 equations (all parsed terms  
    - T3: 53 equations (all parsed terms)
    - T4: 74 equations (all parsed terms)
    TOTAL: 180 equations auto-generated!
    
    Parameters:
    -----------
    no_ham : Hamiltonian object
    n_occ : number of occupied orbitals
    max_iter : maximum iterations (default: 20 due to high cost)
    tol : convergence tolerance
    alpha : damping (recommended: 0.2-0.5 for stability)
    diis_size : DIIS history size
    
    Returns:
    --------
    e_corr, t1, t2, t3, t4 : correlation energy and amplitudes
    """
    
    print("\n" + "="*70)
    print("CCSDTQ COMPLETE - ALL 180 AUTO-GENERATED EQUATIONS")
    print("="*70)
    
    n_states = no_ham.f.shape[0]
    n_virt = n_states - n_occ
    
    # Memory check
    t4_size = n_occ**4 * n_virt**4 * 8  # bytes (float64)
    t4_gb = t4_size / 1e9
    
    if t4_gb > 100:
        print(f"\n⚠️  WARNING: T4 amplitude tensor will require {t4_gb:.1f} GB!")
        print(f"   This is likely impractical. Consider reducing system size.")
        response = input("   Continue anyway? (yes/no): ")
        if response.lower() != 'yes':
            print("   Aborting.")
            return None, None, None, None, None
    
    o = slice(0, n_occ)
    v = slice(n_occ, n_states)
    
    f = no_ham.f
    Gamma = no_ham.Gamma
    
    # Orbital energies
    eps = np.diag(f)
    
    # Energy denominators
    D1 = eps[o, None] - eps[None, v]
    D2 = (eps[o, None, None, None] + eps[None, o, None, None]
          - eps[None, None, v, None] - eps[None, None, None, v])
    D3 = (eps[o, None, None, None, None, None]
          + eps[None, o, None, None, None, None]
          + eps[None, None, o, None, None, None]
          - eps[None, None, None, v, None, None]
          - eps[None, None, None, None, v, None]
          - eps[None, None, None, None, None, v])
    D4 = (eps[o, None, None, None, None, None, None, None]
          + eps[None, o, None, None, None, None, None, None]
          + eps[None, None, o, None, None, None, None, None]
          + eps[None, None, None, o, None, None, None, None]
          - eps[None, None, None, None, v, None, None, None]
          - eps[None, None, None, None, None, v, None, None]
          - eps[None, None, None, None, None, None, v, None]
          - eps[None, None, None, None, None, None, None, v])
    
    # Initialize amplitudes
    t1 = initial_t1.copy() if initial_t1 is not None else np.zeros((n_occ, n_virt))
    t2 = initial_t2.copy() if initial_t2 is not None else Gamma[o,o,v,v] / D2
    t3 = initial_t3.copy() if initial_t3 is not None else np.zeros((n_occ,n_occ,n_occ,n_virt,n_virt,n_virt))
    t4 = np.zeros((n_occ, n_occ, n_occ, n_occ, n_virt, n_virt, n_virt, n_virt))
    
    print(f"\nAmplitude sizes:")
    print(f"  T1: {t1.shape} = {t1.size:,} elements ({t1.nbytes/1e6:.2f} MB)")
    print(f"  T2: {t2.shape} = {t2.size:,} elements ({t2.nbytes/1e6:.2f} MB)")
    print(f"  T3: {t3.shape} = {t3.size:,} elements ({t3.nbytes/1e9:.3f} GB)")
    print(f"  T4: {t4.shape} = {t4.size:,} elements ({t4.nbytes/1e9:.3f} GB)")
    print(f"  Total: {(t1.nbytes+t2.nbytes+t3.nbytes+t4.nbytes)/1e9:.3f} GB\n")
    
    print("Using ALL 180 auto-generated equations:")
    print("  T1: 15 equations")
    print("  T2: 38 equations")
    print("  T3: 53 equations")
    print("  T4: 74 equations\n")
    
    # DIIS for acceleration (crucial for convergence)
    from cc import DIIS
    diis = DIIS(size=diis_size)
    
    old_e = 0.0
    
    print(f"{'Iter':<5} | {'E_corr':<18} | {'ΔE':<12} | {'Time(s)':<8}")
    print("-"*60)
    
    for iteration in range(max_iter):
        t_start = time.time()
        
        # Compute residuals using COMPLETE auto-generated functions
        r1 = compute_ccsdtq_t1_residual_COMPLETE(f, Gamma, t1, t2, t3, t4, o, v)
        r2 = compute_ccsdtq_t2_residual_COMPLETE(f, Gamma, t1, t2, t3, t4, o, v)
        r3 = compute_ccsdtq_t3_residual_COMPLETE(f, Gamma, t1, t2, t3, t4, o, v)
        r4 = compute_ccsdtq_t4_residual_COMPLETE(f, Gamma, t1, t2, t3, t4, o, v)
        
        # Energy
        tau = t2 + contract('ia,jb->ijab', t1, t1)
        e_corr = np.sum(f[o,v] * t1) + 0.25 * np.sum(Gamma[o,o,v,v] * tau)
        
        delta_e = abs(e_corr - old_e)
        t_iter = time.time() - t_start
        
        print(f"{iteration:<5d} | {e_corr:<18.10f} | {delta_e:<12.4e} | {t_iter:<8.1f}")
        
        if delta_e < tol:
            print(f"\n✓ Converged!")
            return e_corr, t1, t2, t3, t4
        
        old_e = e_corr
        
        # Update amplitudes with damping and DIIS
        step1 = alpha * r1 / D1
        step2 = alpha * r2 / D2
        step3 = alpha * r3 / D3
        step4 = alpha * r4 / D4
        
        # DIIS Acceleration
        extrap_t = None
        if iteration >= diis_start_iter:
            flat_t = np.concatenate([
                (t1 + step1).ravel(),
                (t2 + step2).ravel(),
                (t3 + step3).ravel(),
                (t4 + step4).ravel()
            ])
            flat_e = np.concatenate([step1.ravel(), step2.ravel(), step3.ravel(), step4.ravel()])
            
            diis.update(flat_t, flat_e)
            extrap_t = diis.extrapolate()
        
        if extrap_t is not None:
            i1, i2, i3 = t1.size, t1.size + t2.size, t1.size + t2.size + t3.size
            t1 = extrap_t[:i1].reshape(t1.shape)
            t2 = extrap_t[i1:i2].reshape(t2.shape)
            t3 = extrap_t[i2:i3].reshape(t3.shape)
            t4 = extrap_t[i3:].reshape(t4.shape)
        else:
            t1 += step1
            t2 += step2
            t3 += step3
            t4 += step4
    
    print(f"\n⚠ Maximum iterations reached")
    return e_corr, t1, t2, t3, t4


# Convenience alias
ccsdtq = ccsdtq_complete


if __name__ == "__main__":
    print("\nCCSDTQ COMPLETE Solver")
    print("ALL 180 equations auto-generated and integrated!")
    print("\nTo use:")
    print("  from ccsdtq_complete import ccsdtq")
    print("  e_corr, t1, t2, t3, t4 = ccsdtq(hamiltonian, n_occ=4)")
