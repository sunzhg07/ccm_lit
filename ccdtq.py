"""
CCDTQ Solver (T1 = 0)
Coupled Cluster Doubles, Triples, and Quadruples
Approximation: T1 amplitudes are assumed to be zero (or absorbed into Hamiltonian)
"""

import numpy as np
from opt_einsum import contract
import time
import sys

# Import generated residuals
sys.path.insert(0, '/Users/wolf/work/ccm_lit')
from ccdtq_residuals import (
    compute_ccdtq_t2_residual,
    compute_ccdtq_t3_residual,
    compute_ccdtq_t4_residual
)

def ccdtq(no_ham, n_occ, max_iter=20, tol=1e-8, alpha=0.3, diis_size=8,
          diis_start_iter=3, initial_t2=None, initial_t3=None):
    """
    CCDTQ Solver (T1 excluded)
    
    Equations are filtered to remove all T1-dependent terms.
    This creates a significantly cheaper "Doubles+Triples+Quadruples" method.
    
    Parameters:
    -----------
    no_ham : Hamiltonian object
    n_occ : number of occupied orbitals
    max_iter : maximum iterations
    tol : convergence tolerance
    alpha : damping factor
    diis_size : DIIS history size (default: 8)
    diis_start_iter : iterations before enabling DIIS (default: 3)
    """
    print("\n" + "="*70)
    print("CCDTQ Solver (T1 terms removed) - Accelerated")
    print("="*70)
    
    n_states = no_ham.f.shape[0]
    n_virt = n_states - n_occ
    
    # Memory check
    t4_size = n_occ**4 * n_virt**4 * 8 / 1e9
    if t4_size > 100:
        print(f"\n⚠️  WARNING: T4 requires {t4_size:.1f} GB. Aborting.")
        return None, None, None, None
        
    o = slice(0, n_occ)
    v = slice(n_occ, n_states)
    
    f = no_ham.f
    Gamma = no_ham.Gamma
    
    # Orbital energies
    eps = np.diag(f)
    
    # Denominators
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
    t2 = initial_t2.copy() if initial_t2 is not None else Gamma[o,o,v,v] / D2
    t3 = initial_t3.copy() if initial_t3 is not None else np.zeros((n_occ,n_occ,n_occ,n_virt,n_virt,n_virt))
    t4 = np.zeros((n_occ, n_occ, n_occ, n_occ, n_virt, n_virt, n_virt, n_virt))
    
    print(f"Amplitudes: T2 ({t2.size}), T3 ({t3.size}), T4 ({t4.size})")
    print(f"Equations: T2 (14), T3 (22), T4 (34) kept after filtering T1")
    
    from cc import DIIS
    diis = DIIS(size=diis_size)
    old_e = 0.0
    
    print(f"\n{'Iter':<5} | {'E_corr':<18} | {'ΔE':<12} | {'Time(s)':<8}")
    print("-"*60)
    
    for iteration in range(max_iter):
        t_start = time.time()
        
        # Residuals (passed explicit t2, t3, t4)
        r2 = compute_ccdtq_t2_residual(f, Gamma, t2, t3, t4, o, v)
        r3 = compute_ccdtq_t3_residual(f, Gamma, t2, t3, t4, o, v)
        r4 = compute_ccdtq_t4_residual(f, Gamma, t2, t3, t4, o, v)
        
        # Energy (T1=0)
        e_corr = 0.25 * np.sum(Gamma[o,o,v,v] * t2)
        
        delta_e = abs(e_corr - old_e)
        t_iter = time.time() - t_start
        
        print(f"{iteration:<5d} | {e_corr:<18.10f} | {delta_e:<12.4e} | {t_iter:<8.1f}")
        
        if delta_e < tol:
            print(f"\n✓ Converged!")
            return e_corr, t2, t3, t4
            
        old_e = e_corr
        
        # Update
        step2 = alpha * r2 / D2
        step3 = alpha * r3 / D3
        step4 = alpha * r4 / D4
        
        # DIIS Acceleration
        extrap_t = None
        if iteration >= diis_start_iter:
            flat_t = np.concatenate([t2.ravel(), t3.ravel(), t4.ravel()])
            flat_e = np.concatenate([step2.ravel(), step3.ravel(), step4.ravel()])
            diis.update(flat_t, flat_e)
            extrap_t = diis.extrapolate()
        
        if extrap_t is not None:
            i2 = t2.size
            i3 = t2.size + t3.size
            t2 = extrap_t[:i2].reshape(t2.shape)
            t3 = extrap_t[i2:i3].reshape(t3.shape)
            t4 = extrap_t[i3:].reshape(t4.shape)
        else:
            t2 += step2
            t3 += step3
            t4 += step4
            
    print("\n⚠ Maximum iterations reached")
    return e_corr, t2, t3, t4

if __name__ == "__main__":
    print("\nStandalone usage:")
    print("from ccdtq import ccdtq")
    print("e, t2, t3, t4 = ccdtq(ham, n_occ=4)")
