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


def ccsdt(no_ham, n_occ, max_iter=100, tol=1e-8, alpha=0.3):
    """
    CCSDT (Coupled Cluster Singles, Doubles, and Triples) solver.
    
    Includes T1, T2, and T3 amplitudes. The T3 equations are expensive
    and scale as O(N^8) in storage and O(N^9) in computation.
    
    Parameters:
    -----------
    no_ham : Hamiltonian object with .f (Fock) and .Gamma (2-body interaction)
    n_occ : number of occupied orbitals
    max_iter : maximum iterations
    tol : convergence tolerance
    alpha : damping factor for stability (smaller = more stable)
    
    Returns:
    --------
    e_corr : correlation energy
    t1 : singles amplitudes [occ, virt]
    t2 : doubles amplitudes [occ, occ, virt, virt]
    t3 : triples amplitudes [occ, occ, occ, virt, virt, virt]
    """
    n_states = no_ham.f.shape[0]
    o = slice(0, n_occ)
    v = slice(n_occ, n_states)
    n_virt = n_states - n_occ
    
    f = no_ham.f
    v2b = no_ham.Gamma
    
    eps = np.diag(f)
    f_off = f - np.diag(eps)
    f_oo = f_off[o, o]
    f_vv = f_off[v, v]
    f_ov = f_off[o, v]
    f_vo = f_off[v, o]
    
    # Energy denominators
    D1 = eps[o, None] - eps[None, v]
    D2 = eps[o, None, None, None] + eps[None, o, None, None] - eps[None, None, v, None] - eps[None, None, None, v]
    D3 = (eps[o, None, None, None, None, None] + 
          eps[None, o, None, None, None, None] + 
          eps[None, None, o, None, None, None] -
          eps[None, None, None, v, None, None] - 
          eps[None, None, None, None, v, None] - 
          eps[None, None, None, None, None, v])
    
    # Initialize amplitudes
    t1 = np.zeros((n_occ, n_virt))
    t2 = v2b[o, o, v, v] / D2
    t3 = np.zeros((n_occ, n_occ, n_occ, n_virt, n_virt, n_virt))
    
    # Check memory usage
    t3_size_gb = t3.nbytes / 1e9
    print(f"\n[CCSDT] T3 array size: {t3_size_gb:.2f} GB")
    if t3_size_gb > 10.0:
        print(f"[CCSDT WARNING] T3 is very large! Consider using CCSD instead.")
    
    old_e = 0.0
    print(f"\n[CCSDT] {'Iter':>4} | {'E_corr':>15} | {'Delta E':>12} | {'|T3|':>10}")
    print("-" * 60)
    
    for iteration in range(max_iter):
        # Compute correlation energy
        e_corr = np.sum(f_ov * t1)
        e_corr += 0.25 * np.sum(v2b[o, o, v, v] * t2)
        e_corr += 0.5 * np.sum(v2b[o, o, v, v] * contract('ia,jb->ijab', t1, t1))
        
        # T3 contribution to energy:
        # In full CCSDT, the energy is determined fully by T1 and T2 via the standard expression:
        # E = <0| H e^T |0>_C = f_ia t_i^a + 0.25 <ij||ab> t_ij^ab + 0.5 <ij||ab> t_i^a t_j^b
        # T3 affects the energy indirectly by coupling into the T1 and T2 equations.
        # So we do NOT add an explicit T3 energy term here.


        
        delta_e = abs(e_corr - old_e)
        t3_norm = np.linalg.norm(t3)
        print(f"[CCSDT] {iteration:4d} | {e_corr:15.8f} | {delta_e:12.4e} | {t3_norm:10.4e}")
        
        if delta_e < tol:
            print("[CCSDT] Converged!")
            return e_corr, t1, t2, t3
        
        # --- T1 Residual ---
        r1 = f_vo.T.copy().astype(np.float64)
        r1 += contract('ae,ie->ia', f_vv, t1)
        r1 -= contract('mi,ma->ia', f_oo, t1)
        r1 += contract('maie,me->ia', v2b[o, v, o, v], t1)
        r1 += contract('amie,me->ia', v2b[v, o, o, v], t1)
        r1 += 0.5 * contract('maef,imef->ia', v2b[o, v, v, v], t2)
        r1 -= 0.5 * contract('mnei,mnea->ia', v2b[o, o, v, o], t2)
        
        # T3 contributions to T1
        r1 += 0.25 * contract('mnef,imnaef->ia', v2b[o, o, v, v], t3)
        
        # --- T2 Residual ---
        r2 = v2b[o, o, v, v].copy()
        r2 += 0.5 * contract('abef,ijef->ijab', v2b[v, v, v, v], t2)
        r2 += 0.5 * contract('mnij,mnab->ijab', v2b[o, o, o, o], t2)
        term = contract('maie,mjeb->ijab', v2b[o, v, o, v], t2)
        r2 -= pIJ(pAB(term))
        
        # T1 contributions
        r2 += pAB(contract('abie,je->ijab', v2b[v, v, o, v], t1))
        r2 -= pIJ(contract('mbij,ma->ijab', v2b[o, v, o, o], t1))
        
        # T3 contributions to T2
        r2 += 0.5 * contract('mnef,ijmnabef->ijab', v2b[o, o, v, v], 
                            contract('ie,jmnabf->ijmnabef', t1, t3))
        
        # --- T3 Residual ---
        
        # Helper: Cyclic permutation 1 - P_ij - P_ik for i/jk
        def permute_1_23(tensor, idx1, idx2, idx3):
            # Returns T - T(swap 1,2) - T(swap 1,3)
            # This generates antisymmetry for index 1 against pair (2,3)
            # assuming (2,3) is already antisymmetric.
            
            # Construct transpose axes
            ndim = tensor.ndim
            axes_swap12 = list(range(ndim))
            axes_swap12[idx1], axes_swap12[idx2] = axes_swap12[idx2], axes_swap12[idx1]
            
            axes_swap13 = list(range(ndim))
            axes_swap13[idx1], axes_swap13[idx3] = axes_swap13[idx3], axes_swap13[idx1]
            
            return tensor - np.transpose(tensor, axes_swap12) - np.transpose(tensor, axes_swap13)

        # 1. Source Terms (involving T2)
        # Term 1: P(ijk) P(abc) sum_e t_ij^ae <ke||bc>
        # t2 is antisym in ij, v2b is antisym in bc.
        # Need P(k/ij) and P(a/bc).
        # k=2, i=0, j=1.  a=3, b=4, c=5.
        src1 = contract('ijae,kebc->ijkabc', t2, v2b[o, v, v, v])
        src1 = permute_1_23(src1, 2, 0, 1) # P(k/ij)
        src1 = permute_1_23(src1, 3, 4, 5) # P(a/bc)
        
        # Term 2: -P(ijk) P(abc) sum_m t_im^ab <jk||mc>
        # t2 antisym in ab, v2b antisym in jk.
        # Need P(c/ab) and P(i/jk).
        # i=0, j=1, k=2. c=5, a=3, b=4.
        src2 = -contract('imab,jkmc->ijkabc', t2, v2b[o, o, o, v])
        src2 = permute_1_23(src2, 0, 1, 2) # P(i/jk)
        src2 = permute_1_23(src2, 5, 3, 4) # P(c/ab)
        
        r3 = src1 + src2
        
        # 2. Linear T3 Terms (Fock contributions)
        # Term 3: -P(i/jk) sum_m t_mjk^abc f_im
        # t3 antisym in jk, abc.
        # Need P(i/jk). i=0.
        term3 = -contract('mjkabc,mi->ijkabc', t3, f_oo)
        r3 += permute_1_23(term3, 0, 1, 2)
        
        # Term 4: P(c/ab) sum_e t_ijk^abe f_ec
        # t3 antisym in ijk, ab.
        # Need P(c/ab). c=5.
        term4 = contract('ijkabe,ec->ijkabc', t3, f_vv)
        r3 += permute_1_23(term4, 5, 3, 4)

        # 3. Leading T3-Interaction Terms (Ladders)
        # Term 5: 0.5 * P(a/bc) sum_ef t_ijk^aef <ef||bc>
        # v2b antisym in bc. t3 antisym in ijk, ef.
        # Need P(a/bc). a=3, b=4, c=5.
        term5 = 0.5 * contract('ijkaef,efbc->ijkabc', t3, v2b[v, v, v, v])
        r3 += permute_1_23(term5, 3, 4, 5)
        
        # Term 6: 0.5 * P(i/jk) sum_mn t_imn^abc <jk||mn>
        # v2b antisym in jk. t3 antisym in abc, mn.
        # Need P(i/jk). i=0, j=1, k=2.
        term6 = 0.5 * contract('imnabc,jkmn->ijkabc', t3, v2b[o, o, o, o])
        r3 += permute_1_23(term6, 0, 1, 2)

        
        # Solve amplitude equations
        new_t1 = r1 / D1
        new_t2 = r2 / D2
        new_t3 = r3 / D3
        
        # Apply damping
        t1 = (1 - alpha) * t1 + alpha * new_t1
        t2 = (1 - alpha) * t2 + alpha * new_t2
        t3 = (1 - alpha) * t3 + alpha * new_t3
        
        old_e = e_corr
        
        if np.isnan(e_corr) or abs(e_corr) > 1e10:
            print("[CCSDT ERROR] Diverged.")
            break
    
    print("[CCSDT] Maximum iterations reached without convergence.")
    return e_corr, t1, t2, t3


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
