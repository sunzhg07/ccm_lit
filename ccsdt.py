
import numpy as np
from opt_einsum import contract
from scipy import sparse

SPARSE_THRESHOLD = 1e-12

def to_sparse_if_beneficial(arr, threshold=SPARSE_THRESHOLD):
    if arr is None or arr.size == 0: return arr
    if np.sum(np.abs(arr) < threshold) / arr.size > 0.5:
        return sparse.csr_matrix(arr.reshape(arr.shape[0], -1))
    return arr

# Permutations
def P_ij(t): return t - t.transpose(1, 0, 2, 3)
def P_ab(t): return t - t.transpose(0, 1, 3, 2)
def P_a_bc(t): return t - t.transpose(0,1,2, 4,3,5) - t.transpose(0,1,2, 5,4,3)
def P_i_jk(t): return t - t.transpose(1,0,2, 3,4,5) - t.transpose(2,1,0, 3,4,5)
def P_ij_k(t): return t - t.transpose(2,1,0, 3,4,5) - t.transpose(0,2,1, 3,4,5)
def P_ab_c(t): return t - t.transpose(0,1,2, 5,4,3) - t.transpose(0,1,2, 3,5,4)
def P_b_ac(t): return t - t.transpose(0,1,2, 4,3,5) - t.transpose(0,1,2, 3,5,4)
def P_c_ab(t): return t - t.transpose(0,1,2, 5,4,3) - t.transpose(0,1,2, 3,5,4)
def P_a_cb(t): return t - t.transpose(0,1,2, 5,4,3) - t.transpose(0,1,2, 4,3,5)
def P_ba_c(t): return t - t.transpose(0,1,2, 3,5,4) - t.transpose(0,1,2, 5,4,3)

def P_ijk_full(t):
    return (t - t.transpose(1,0,2, 3,4,5) - t.transpose(2,1,0, 3,4,5)
              - t.transpose(0,2,1, 3,4,5) + t.transpose(1,2,0, 3,4,5)
              + t.transpose(2,0,1, 3,4,5))

def P_abc_full(t):
    return (t - t.transpose(0,1,2, 4,3,5) - t.transpose(0,1,2, 5,4,3)
              - t.transpose(0,1,2, 3,5,4) + t.transpose(0,1,2, 4,5,3)
              + t.transpose(0,1,2, 5,3,4))

def ccsdt(no_ham, n_occ, max_iter=50, tol=1e-8, alpha=1.0, use_sparse=False):
    n_states = no_ham.f.shape[0]
    n_virt = n_states - n_occ
    o = slice(0, n_occ)
    v = slice(n_occ, n_states)

    f = no_ham.f
    Gamma = no_ham.Gamma
    
    f_oo, f_vv, f_ov, f_vo = f[o, o], f[v, v], f[o, v], f[v, o]
    
    # Interaction blocks
    V_oovv = Gamma[o, o, v, v]
    V_ooov = Gamma[o, o, o, v]
    V_vovv = Gamma[v, o, v, v]
    V_oooo = Gamma[o, o, o, o]
    V_vvvv = Gamma[v, v, v, v]
    V_voov = Gamma[v, o, o, v] # Adjusted: V^{aj}_{ib}
    V_vooo = Gamma[v, o, o, o] # Adjusted: V^{ak}_{ij} (actually minus Gamma[o,v...])
    # V_vooo using global antisymmetry V[v,o,o,o] is fine.
    
    V_vovo = Gamma[v, o, v, o]
    V_vvvo = Gamma[v, v, v, o]
    V_vvov = Gamma[v, v, o, v] # Adjusted: V^{ab}_{ic}
    V_ovvv = Gamma[o, v, v, v]
    V_ovoo = Gamma[o, v, o, o]

    eps = np.diag(f)
    eps_o = eps[o]
    eps_v = eps[v]
    
    D1 = eps_o[:, None] - eps_v[None, :]
    D2 = (eps_o[:, None, None, None] + eps_o[None, :, None, None] 
          - eps_v[None, None, :, None] - eps_v[None, None, None, :])
    D3 = (eps_o[:, None, None, None, None, None] + eps_o[None, :, None, None, None, None] + eps_o[None, None, :, None, None, None]
          - eps_v[None, None, None, :, None, None] - eps_v[None, None, None, None, :, None] - eps_v[None, None, None, None, None, :])

    t1 = np.zeros((n_occ, n_virt))
    t2 = V_oovv / D2
    t3 = np.zeros((n_occ, n_occ, n_occ, n_virt, n_virt, n_virt))
    
    print(f"CCSDT Init. T3 Size: {t3.size/1e6:.2f} M elements")

    old_e = 0.0
    
    for iteration in range(max_iter):
        
        # T1 Update
        r1 = f_vo.copy().T
        r1 -= contract('jkbc,ib,ja,kc->ia', V_oovv, t1, t1, t1)
        r1 -= contract('jb,ib,ja->ia', f_ov, t1, t1)
        r1 += contract('ajbc,ib,jc->ia', V_vovv, t1, t1)
        r1 -= contract('jkib,ja,kb->ia', V_ooov, t1, t1)
        r1 += contract('jkbc,jb,kica->ia', V_oovv, t1, t2)
        r1 += 0.5 * contract('jkbc,ib,jkca->ia', V_oovv, t1, t2)
        r1 += 0.5 * contract('jkbc,ja,kibc->ia', V_oovv, t1, t2)
        r1 += contract('ab,ib->ia', f_vv, t1)
        r1 -= contract('ji,ja->ia', f_oo, t1)
        r1 += contract('ajib,jb->ia', V_voov, t1) # Corrected
        r1 += contract('jb,ijab->ia', f_ov, t2)
        r1 += 0.5 * contract('ajbc,ijbc->ia', V_vovv, t2)
        r1 -= 0.5 * contract('jkib,jkab->ia', V_ooov, t2)
        r1 += 0.25 * contract('jkbc,ijkabc->ia', V_oovv, t3)

        # T2 Update
        r2 = V_oovv.copy()
        term = 0.25 * contract('klcd,ic,jd,ka,lb->ijab', V_oovv, t1, t1, t1, t1)
        r2 += P_ab(P_ij(term))
        term = contract('akcd,ic,jd,kb->ijab', V_vovv, t1, t1, t1)
        r2 -= 0.5 * P_ab(P_ij(term))
        term = contract('klic,jc,ka,lb->ijab', V_ooov, t1, t1, t1)
        r2 += 0.5 * P_ab(P_ij(term))
        r2 += P_ij(contract('klcd,ic,kd,ljab->ijab', V_oovv, t1, t1, t2))
        r2 += P_ab(contract('klcd,ka,lc,ijdb->ijab', V_oovv, t1, t1, t2))
        r2 += 0.25 * P_ij(contract('klcd,ic,jd,klab->ijab', V_oovv, t1, t1, t2))
        r2 -= P_ab(P_ij(contract('klcd,ic,ka,ljdb->ijab', V_oovv, t1, t1, t2)))
        r2 += 0.25 * P_ab(contract('klcd,ka,lb,ijcd->ijab', V_oovv, t1, t1, t2))
        r2 += 0.5 * P_ij(contract('abcd,ic,jd->ijab', V_vvvv, t1, t1))
        r2 += 0.5 * P_ab(contract('klij,ka,lb->ijab', V_oooo, t1, t1))
        
        r2 -= P_ab(P_ij(contract('akic,jc,kb->ijab', V_voov, t1, t1)))
        r2 -= P_ij(contract('kc,ic,kjab->ijab', f_ov, t1, t2))
        r2 -= P_ab(contract('kc,ka,ijcb->ijab', f_ov, t1, t2))
        r2 -= P_ab(contract('akcd,kc,ijdb->ijab', V_vovv, t1, t2))
        r2 += P_ab(P_ij(contract('akcd,ic,kjdb->ijab', V_vovv, t1, t2)))
        r2 -= 0.5 * P_ab(contract('akcd,kb,ijcd->ijab', V_vovv, t1, t2))
        r2 += P_ij(contract('klic,kc,ljab->ijab', V_ooov, t1, t2))
        r2 += 0.5 * P_ij(contract('klic,jc,klab->ijab', V_ooov, t1, t2))
        r2 -= P_ab(P_ij(contract('klic,ka,ljcb->ijab', V_ooov, t1, t2)))
        r2 += contract('klcd,kc,lijdab->ijab', V_oovv, t1, t3)
        r2 += 0.5 * P_ij(contract('klcd,ic,kljdab->ijab', V_oovv, t1, t3))
        r2 += 0.5 * P_ab(contract('klcd,ka,lijcdb->ijab', V_oovv, t1, t3))
        
        r2 += P_ij(contract('abic,jc->ijab', V_vvov, t1))
        
        r2 -= P_ab(contract('akij,kb->ijab', V_vooo, t1))

        r2 += 0.5 * P_ij(contract('klcd,ikab,ljcd->ijab', V_oovv, t2, t2))
        r2 += 0.5 * P_ab(contract('klcd,ijac,kldb->ijab', V_oovv, t2, t2))
        r2 += 0.5 * P_ab(P_ij(contract('klcd,ikac,ljdb->ijab', V_oovv, t2, t2)))
        r2 += 0.25 * contract('klcd,ijcd,klab->ijab', V_oovv, t2, t2)
        r2 += P_ab(contract('ac,jibc->ijab', f_vv, t2))
        r2 -= P_ij(contract('ki,jkba->ijab', f_oo, t2))
        r2 += 0.5 * contract('abcd,ijcd->ijab', V_vvvv, t2)
        r2 += 0.5 * contract('klij,klab->ijab', V_oooo, t2)
        r2 += P_ab(P_ij(contract('akic,jkbc->ijab', V_voov, t2))) # Corrected
        r2 += contract('kc,ijkabc->ijab', f_ov, t3)
        r2 += 0.5 * P_ab(contract('akcd,jikbcd->ijab', V_vovv, t3))
        r2 -= 0.5 * P_ij(contract('klic,jklbac->ijab', V_ooov, t3))
        
        # T3 Update
        r3 = np.zeros_like(t3)
        
        term = contract('lmde,id,je,la,mkbc->ijkabc', V_oovv, t1, t1, t1, t2)
        r3 += 0.5 * P_ijk_full(P_a_bc(term))
        term = contract('lmde,id,la,mb,jkec->ijkabc', V_oovv, t1, t1, t1, t2)
        r3 += 0.5 * P_abc_full(P_i_jk(term))
        term = contract('alde,id,je,lkbc->ijkabc', V_vovv, t1, t1, t2)
        r3 -= 0.5 * P_ijk_full(P_a_bc(term))
        term = contract('alde,id,lb,jkec->ijkabc', V_vovv, t1, t1, t2)
        r3 -= P_abc_full(P_i_jk(term))
        term = contract('lmid,la,mb,jkdc->ijkabc', V_ooov, t1, t1, t2)
        r3 += 0.5 * P_abc_full(P_i_jk(term))
        term = contract('lmid,jd,la,mkbc->ijkabc', V_ooov, t1, t1, t2)
        r3 += P_ijk_full(P_a_bc(term))
        term = contract('lmde,id,le,mjkabc->ijkabc', V_oovv, t1, t1, t3)
        r3 += P_i_jk(term)
        term = contract('lmde,la,md,ijkebc->ijkabc', V_oovv, t1, t1, t3)
        r3 += P_a_bc(term)
        term = contract('lmde,id,je,lmkabc->ijkabc', V_oovv, t1, t1, t3)
        r3 += 0.25 * P_ijk_full(term)
        term = contract('lmde,id,la,mjkebc->ijkabc', V_oovv, t1, t1, t3)
        r3 -= P_a_bc(P_i_jk(term))
        term = contract('lmde,la,mb,ijkdec->ijkabc', V_oovv, t1, t1, t3)
        r3 += 0.25 * P_abc_full(term)
        term = contract('lmde,ld,ijeb,mkac->ijkabc', V_oovv, t1, t2, t2)
        r3 -= P_b_ac(P_ij_k(term))
        term = contract('lmde,la,ijdb,mkec->ijkabc', V_oovv, t1, t2, t2)
        r3 -= P_abc_full(P_ij_k(term))
        term = contract('lmde,id,ljab,mkec->ijkabc', V_oovv, t1, t2, t2)
        r3 -= P_ab_c(P_ijk_full(term))
        term = contract('lmde,la,ijde,mkbc->ijkabc', V_oovv, t1, t2, t2)
        r3 += 0.5 * P_a_bc(P_ij_k(term))
        term = contract('lmde,id,jkec,lmab->ijkabc', V_oovv, t1, t2, t2)
        r3 += 0.5 * P_c_ab(P_i_jk(term))
        term = contract('abde,id,jkec->ijkabc', V_vvvv, t1, t2)
        r3 += P_ab_c(P_i_jk(term))
        term = contract('lmij,la,mkbc->ijkabc', V_oooo, t1, t2)
        r3 += P_a_bc(P_ij_k(term))
        
        term = contract('alid,jd,lkbc->ijkabc', V_voov, t1, t2)
        r3 -= P_ijk_full(P_a_bc(term))
        term = contract('alid,lb,jkdc->ijkabc', V_voov, t1, t2)
        r3 -= P_abc_full(P_i_jk(term))
        
        term = contract('ld,id,ljkabc->ijkabc', f_ov, t1, t3)
        r3 -= P_i_jk(term)
        term = contract('ld,la,ijkdbc->ijkabc', f_ov, t1, t3)
        r3 -= P_a_bc(term)
        term = contract('alde,ld,ijkebc->ijkabc', V_vovv, t1, t3)
        r3 -= P_a_bc(term)
        term = contract('alde,id,ljkebc->ijkabc', V_vovv, t1, t3)
        r3 += P_a_bc(P_i_jk(term))
        term = contract('alde,lb,ijkdec->ijkabc', V_vovv, t1, t3)
        r3 -= 0.5 * P_abc_full(term)
        term = contract('lmid,ld,mjkabc->ijkabc', V_ooov, t1, t3)
        r3 += P_i_jk(term)
        term = contract('lmid,jd,lmkabc->ijkabc', V_ooov, t1, t3)
        r3 += 0.5 * P_ijk_full(term)
        term = contract('lmid,la,mjkdbc->ijkabc', V_ooov, t1, t3)
        r3 -= P_a_bc(P_i_jk(term))
        term = contract('ld,ijad,lkbc->ijkabc', f_ov, t2, t2)
        r3 -= P_a_bc(P_ij_k(term))
        term = contract('alde,jibd,lkec->ijkabc', V_vovv, t2, t2)
        r3 += P_abc_full(P_ij_k(term)) 
        term = contract('alde,ijde,lkbc->ijkabc', V_vovv, t2, t2)
        r3 -= 0.5 * P_a_bc(P_ij_k(term))
        term = contract('lmid,jlba,mkdc->ijkabc', V_ooov, t2, t2)
        r3 -= P_ba_c(P_ijk_full(term))
        term = contract('lmid,jkbd,lmac->ijkabc', V_ooov, t2, t2)
        r3 += 0.5 * P_b_ac(P_i_jk(term))
        term = contract('lmde,ilde,mjkabc->ijkabc', V_oovv, t2, t3)
        r3 += 0.5 * P_i_jk(term)
        term = contract('lmde,lmad,ijkebc->ijkabc', V_oovv, t2, t3)
        r3 += 0.5 * P_a_bc(term)
        term = contract('lmde,ijde,lmkabc->ijkabc', V_oovv, t2, t3)
        r3 += 0.25 * P_ij_k(term)
        term = contract('lmde,ilad,mjkebc->ijkabc', V_oovv, t2, t3)
        r3 += P_a_bc(P_i_jk(term))
        term = contract('lmde,lmab,ijkdec->ijkabc', V_oovv, t2, t3)
        r3 += 0.25 * P_ab_c(term)
        term = contract('lmde,ijad,lmkebc->ijkabc', V_oovv, t2, t3)
        r3 += 0.5 * P_a_bc(P_ij_k(term))
        term = contract('lmde,ilab,mjkdec->ijkabc', V_oovv, t2, t3)
        r3 += 0.5 * P_ab_c(P_i_jk(term))
        
        term = contract('abid,kjcd->ijkabc', V_vvov, t2) # Corrected
        r3 += P_ab_c(P_i_jk(term))
        
        term = - contract('alij,klcb->ijkabc', V_vooo, t2)
        r3 -= P_a_cb(P_ij_k(term))
        
        term = contract('ad,jkibcd->ijkabc', f_vv, t3)
        r3 += P_a_bc(term)
        term = contract('li,jklbca->ijkabc', f_oo, t3)
        r3 -= P_i_jk(term)
        term = contract('abde,kijcde->ijkabc', V_vvvv, t3)
        r3 += 0.5 * P_ab_c(term)
        term = contract('lmij,klmcab->ijkabc', V_oooo, t3)
        r3 += 0.5 * P_ij_k(term)
        term = contract('alid,jklbcd->ijkabc', V_voov, t3) # Corrected
        r3 += P_a_bc(P_i_jk(term))
        
        e_corr = 0.25 * np.sum(V_oovv * t2)
        e_corr += 0.5 * np.sum(contract('ijab,ia,jb->', V_oovv, t1, t1))
        
        delta_e = abs(e_corr - old_e)
        
        print(f"[CCSDT] Iter {iteration:3d} | E_corr {e_corr:18.12f} | dE {delta_e:10.4e}")
        
        if delta_e < tol:
            return e_corr, t1, t2, t3
            
        old_e = e_corr
        
        t1 += alpha * r1 / D1
        t2 += alpha * r2 / D2
        t3 += alpha * r3 / D3

    return e_corr, t1, t2, t3
