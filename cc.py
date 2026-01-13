import numpy as np
from opt_einsum import contract

# Enable CCSD debug diagnostics
CCSD_DEBUG = True


def pAB(val):
    """Permutator val(abij) -> val(abij) - val(baij)"""
    return val - np.transpose(val, (1, 0, 2, 3))


def pIJ(val):
    """Permutator val(abij) -> val(abij) - val(abji)"""
    return val - np.transpose(val, (0, 1, 3, 2))


def ccsd(no_ham, n_occ, max_iter=200, tol=1e-8, alpha=0.5):
    """
    Fixed CCSD solver with proper iterative residual updates.
    """
    n_states = no_ham.f.shape[0]
    o = slice(0, n_occ)
    v = slice(n_occ, n_states)
    
    f = no_ham.f
    v2b = no_ham.Gamma
    
    eps = np.diag(f)
    f_off = f - np.diag(eps)
    f_oo = f_off[o, o]
    f_vv = f_off[v, v]
    f_ov = f_off[o, v]
    f_vo = f_off[v, o]
    
    # Denominators: D = eps_occ - eps_virt
    D1 = eps[o, None] - eps[None, v] 
    D2 = eps[o, None, None, None] + eps[None, o, None, None] - eps[None, None, v, None] - eps[None, None, None, v]
    
    # Initial Amplitudes
    t1 = np.zeros((n_occ, n_states - n_occ))
    t2 = v2b[o, o, v, v] / D2
    
    old_e = 0.0
    print(f"\n[CCSD] {'Iter':>4} | {'E_corr':>15} | {'Delta E':>12}")
    print("-" * 45)
    
    for iter in range(max_iter):
        # Current Correlation Energy
        e_corr = np.sum(f_ov * t1) 
        e_corr += 0.25 * np.sum(v2b[o, o, v, v] * t2)
        e_corr += 0.5 * np.sum(v2b[o, o, v, v] * contract('ia,jb->ijab', t1, t1))
        
        delta_e = abs(e_corr - old_e)
        print(f"[CCSD] {iter:4d} | {e_corr:15.8f} | {delta_e:12.4e}")
        if delta_e < tol:
            print("CCSD Converged!")
            return e_corr, t1, t2

        # --- Residual T1 ---
        r1 = f_vo.T.copy() # f_ia
        r1 += contract('ae,ie->ia', f_vv, t1)
        r1 -= contract('mi,ma->ia', f_oo, t1)
        r1 += contract('maie,me->ia', v2b[o, v, o, v], t1)
        r1 += contract('amie,me->ia', v2b[v, o, o, v], t1) # Correct 2b term
        r1 += 0.5 * contract('maef,imef->ia', v2b[o, v, v, v], t2)
        r1 -= 0.5 * contract('mnei,mnea->ia', v2b[o, o, v, o], t2)
        
        # --- Residual T2 ---
        r2 = v2b[o, o, v, v].copy() # <ij||ab>
        
        # Linear Ladder/Ring
        r2 += 0.5 * contract('abef,ijef->ijab', v2b[v, v, v, v], t2)
        r2 += 0.5 * contract('mnij,mnab->ijab', v2b[o, o, o, o], t2)
        term = contract('maie,mjeb->ijab', v2b[o, v, o, v], t2)
        r2 -= pIJ(pAB(term))
        
        # T1 coupling
        r2 += pAB(contract('abie,je->ijab', v2b[v, v, o, v], t1))
        r2 -= pIJ(contract('mbij,ma->ijab', v2b[o, v, o, o], t1))


        # Solve: D*t = R (Jacobi iteration)
        new_t1 = r1 / D1
        new_t2 = r2 / D2
        
        # Damping for stability
        t1 = (1 - alpha) * t1 + alpha * new_t1
        t2 = (1 - alpha) * t2 + alpha * new_t2
        
        old_e = e_corr
        if np.isnan(e_corr) or abs(e_corr) > 1e10:
            print("[CCSD ERROR] Diverged.")
            break
            
    return e_corr, t1, t2
def ccd(no_ham, n_occ, max_iter=200, tol=1e-8, alpha=0.5):
    """
    Simpler CCD (Coupled Cluster Doubles) solver for diagnostics.
    """
    n_states = no_ham.f.shape[0]
    o = slice(0, n_occ)
    v = slice(n_occ, n_states)
    
    f = no_ham.f
    Gamma = no_ham.Gamma
    
    eps = np.diag(f)
    f_oo = f[o, o] - np.diag(eps[o])
    f_vv = f[v, v] - np.diag(eps[v])
    
    # Denominators: D_ijab = eps_i + eps_j - eps_a - eps_b
    D2 = eps[o, None, None, None] + eps[None, o, None, None] - eps[None, None, v, None] - eps[None, None, None, v]
    
    
    # Check for stability issues
    if np.any(D2 >= 0):
        print("[CCD WARNING] Positive or zero energy denominators detected!")
        # Find the indices
        pos_idx = np.where(D2 >= 0)
        for idx in range(min(5, len(pos_idx[0]))):
            i, j, a, b = pos_idx[0][idx], pos_idx[1][idx], pos_idx[2][idx], pos_idx[3][idx]
            print(f"  D[{i},{j},{a},{b}] = {D2[i,j,a,b]:.6f} (eps: {eps[o][i]:.2f}, {eps[o][j]:.2f}, {eps[v][a]:.2f}, {eps[v][b]:.2f})")

    # Initial T2 from First-order perturbation (MP2)
    t2 = Gamma[o, o, v, v] / D2
    
    old_e = 0.0
    print(f"\n[CCD] {'Iter':>4} | {'E_corr':>15} | {'Delta E':>12}")
    print("-" * 45)
    
    for iter in range(max_iter):
        # E_ccd = 0.25 * <ij||ab> t_ijab
        e_corr = 0.25 * np.sum(Gamma[o, o, v, v] * t2)
        
        delta_e = abs(e_corr - old_e)
        print(f"[CCD] {iter:4d} | {e_corr:15.8f} | {delta_e:12.4e}")
        if delta_e < tol:
            print("CCD Converged!")
            return e_corr, t2

        # T2 Residual: R_ijab such that D_ijab * t_ijab = R_ijab
        # R = <ij||ab> + 0.5*<mn||ij>t_mnab + 0.5*<ab||ef>t_ijef - P(ij)P(ab)<ma||ie>t_mjeb
        r2 = Gamma[o, o, v, v].copy()
        r2 += 0.5 * contract('mnij,mnab->ijab', Gamma[o, o, o, o], t2)
        r2 += 0.5 * contract('abef,ijef->ijab', Gamma[v, v, v, v], t2)
        term = contract('maie,mjeb->ijab', Gamma[o, v, o, v], t2)
        r2 -= pIJ(pAB(term))
        
        # Solve: t_new = R / D (Jacobi iteration with damping)
        new_t2 = r2 / D2
        t2 = (1 - alpha) * t2 + alpha * new_t2
        
        old_e = e_corr
        if np.isnan(e_corr):
            print("[CCD ERROR] NaN detected in energy. Diverged.")
            break
            
    return e_corr, t2


def mp2(no_ham, n_occ):
    """
    Computes MP2 correlation energy.
    """
    n_states = no_ham.f.shape[0]
    o = slice(0, n_occ)
    v = slice(n_occ, n_states)
    
    eps = np.diag(no_ham.f)
    v_oovv = no_ham.Gamma[o, o, v, v]
    
    # Denominators: D_ijab = eps_i + eps_j - eps_a - eps_b
    D2 = eps[o, None, None, None] + eps[None, o, None, None] - eps[None, None, v, None] - eps[None, None, None, v]
    
    e_mp2 = 0.25 * np.sum((v_oovv * v_oovv) / D2)
    return e_mp2
