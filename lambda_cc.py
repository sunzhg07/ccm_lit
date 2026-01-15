
import numpy as np
from opt_einsum import contract
from similarity_transform import similarity_transform_t1_t2
from scipy import sparse

def pAB(val):
    """Permutator val(abij) -> val(abij) - val(baij)"""
    return val - np.transpose(val, (1, 0, 2, 3))


def pIJ(val):
    """Permutator val(abij) -> val(abij) - val(abji)"""
    return val - np.transpose(val, (0, 1, 3, 2))


def permute_1_23(tensor, idx1, idx2, idx3):
    """Cyclic permutation 1 - P_12 - P_13"""
    ndim = tensor.ndim
    axes_swap12 = list(range(ndim))
    axes_swap12[idx1], axes_swap12[idx2] = axes_swap12[idx2], axes_swap12[idx1]
    
    axes_swap13 = list(range(ndim))
    axes_swap13[idx1], axes_swap13[idx3] = axes_swap13[idx3], axes_swap13[idx1]
    
    return tensor - np.transpose(tensor, axes_swap12) - np.transpose(tensor, axes_swap13)


def lambda_ccsd(no_ham, t1, t2, n_occ, max_iter=200, tol=1e-8, alpha=0.5):
    """
    Solve Lambda-CCSD equations using similarity-transformed Hamiltonian.
    """
    n_states = no_ham.f.shape[0]
    o = slice(0, n_occ)
    v = slice(n_occ, n_states)
    
    # Compute similarity-transformed Hamiltonian (T1+T2)
    H_bar = similarity_transform_t1_t2(no_ham, t1, t2, n_occ)
    f = H_bar.f
    v2b = H_bar.Gamma
    
    eps = np.diag(f)
    f_oo = f[o, o] - np.diag(eps[o])
    f_vv = f[v, v] - np.diag(eps[v])
    
    # Driving terms (de-excitation blocks of H_bar)
    h_ia = f[o, v]
    h_ijab = v2b[o, o, v, v]
    
    # Energy denominators
    D1 = eps[o, None] - eps[None, v]
    D2 = eps[o, None, None, None] + eps[None, o, None, None] - \
         eps[None, None, v, None] - eps[None, None, None, v]
    
    # Initialize Lambda amplitudes (start from T amplitudes)
    l1 = t1.copy()
    l2 = t2.copy()
    
    print(f"\n[Lambda-CCSD] {'Iter':>4} | {'||L1||':>12} | {'||L2||':>12} | {'Delta':>12}")
    print("-" * 60)
    
    old_norm = np.linalg.norm(l1) + np.linalg.norm(l2)
    
    for iteration in range(max_iter):
        # --- Lambda1 Residual ---
        L1 = h_ia.copy()
        L1 += contract('ie,ea->ia', l1, f_vv)
        L1 -= contract('ma,mi->ia', l1, f_oo)
        L1 += contract('me,maie->ia', l1, v2b[o, v, o, v])
        L1 += contract('imef,mnea->ia', l2, v2b[o, o, v, v])
        
        # --- Lambda2 Residual ---
        L2 = h_ijab.copy()
        L2 += pAB(contract('ie,ejab->ijab', l1, v2b[v, o, v, v]))
        L2 -= pIJ(contract('ma,imjb->ijab', l1, v2b[o, o, o, v]))
        L2 += 0.5 * contract('abef,ijef->ijab', v2b[v, v, v, v], l2)
        L2 += 0.5 * contract('mnij,mnab->ijab', v2b[o, o, o, o], l2)
        
        term = contract('mbej,imae->ijab', v2b[o, v, v, o], l2)
        L2 += pIJ(pAB(term))
        
        # Update
        new_l1 = L1 / D1
        new_l2 = L2 / D2
        
        l1 = (1 - alpha) * l1 + alpha * new_l1
        l2 = (1 - alpha) * l2 + alpha * new_l2
        
        norm_l1 = np.linalg.norm(L1)
        norm_l2 = np.linalg.norm(L2)
        new_norm = np.linalg.norm(l1) + np.linalg.norm(l2)
        delta = abs(new_norm - old_norm)
        
        print(f"[Lambda-CCSD] {iteration:4d} | {norm_l1:12.6e} | {norm_l2:12.6e} | {delta:12.6e}")
        
        if delta < tol:
            print("Lambda-CCSD Converged!")
            break
        old_norm = new_norm
        
    return l1, l2


def lambda_ccsdt(no_ham, t1, t2, t3, n_occ, max_iter=100, tol=1e-8, alpha=0.5):
    """
    Solve Lambda-CCSDT equations with sparse T3/L3 support.
    """
    n_states = no_ham.f.shape[0]
    o = slice(0, n_occ)
    v = slice(n_occ, n_states)
    n_virt = n_states - n_occ
    
    # Compute Similarity-transformed Hamiltonian
    H_bar = similarity_transform_t1_t2(no_ham, t1, t2, n_occ)
    f = H_bar.f
    v2b = H_bar.Gamma
    
    eps = np.diag(f)
    f_oo = f[o, o] - np.diag(eps[o])
    f_vv = f[v, v] - np.diag(eps[v])
    
    # Denominators
    D1 = eps[o, None] - eps[None, v]
    D2 = eps[o, None, None, None] + eps[None, o, None, None] - eps[None, None, v, None] - eps[None, None, None, v]
    
    def sp_norm(mat):
        if hasattr(mat, "data"):
            return np.sqrt(np.sum(mat.data**2))
        return np.linalg.norm(mat)
          
    # Initialize L amplitudes
    l1 = t1.copy()
    l2 = t2.copy()
    l3 = t3.copy() # Sparse
    
    print(f"\n[Lambda-CCSDT] {'Iter':>4} | {'||L3||':>12} | {'Delta':>12}")
    print("-" * 60)
    
    old_norm = sp_norm(l3)
    alpha = 0.2
    
    for iteration in range(max_iter):
        # L1 and L2 residuals (same structure as CCSD but driven by H_bar)
        r_l1 = f[o, v].copy() 
        r_l1 += contract('ie,ea->ia', l1, f_vv)
        r_l1 -= contract('ma,mi->ia', l1, f_oo)
        r_l1 += contract('me,maie->ia', l1, v2b[o, v, o, v])
        r_l1 += contract('imef,mnea->ia', l2, v2b[o, o, v, v])
        
        r_l2 = v2b[o, o, v, v].copy() 
        r_l2 += pAB(contract('ie,ejab->ijab', l1, v2b[v, o, v, v]))
        r_l2 -= pIJ(contract('ma,imjb->ijab', l1, v2b[o, o, o, v]))
        r_l2 += 0.5 * contract('abef,ijef->ijab', v2b[v, v, v, v], l2)
        r_l2 += 0.5 * contract('mnij,mnab->ijab', v2b[o, o, o, o], l2)
        r_l2 += pIJ(pAB(contract('mbej,imae->ijab', v2b[o, v, v, o], l2)))
        
        # Simplified L3 residual (driving from T1/T2 part of bar-H)
        r_l3_dense = 0.25 * contract('ia,mnef->imnaef', l1, v2b[o, o, v, v])
        r_l3_dense += 0.5 * contract('ijab,mnef,ie->jmnabf', l2, v2b[o, o, v, v], t1)
        
        # apply D3
        eps_o = eps[o]
        eps_v = eps[v]
        D3 = (eps_o[:,None,None,None,None,None] + eps_o[None,:,None,None,None,None] + eps_o[None,None,:,None,None,None] -
              eps_v[None,None,None,:,None,None] - eps_v[None,None,None,None,:,None] - eps_v[None,None,None,None,None,:])
        
        new_l3 = sparse.csr_matrix((r_l3_dense / D3).reshape(n_occ**3, n_virt**3))
        
        # Updates
        l1 = (1 - alpha) * l1 + alpha * (r_l1 / D1)
        l2 = (1 - alpha) * l2 + alpha * (r_l2 / D2)
        l3 = (1 - alpha) * l3 + alpha * new_l3
        
        new_norm = sp_norm(l3)
        delta = abs(new_norm - old_norm)
        print(f"[Lambda-CCSDT] {iteration:4d} | {new_norm:12.6e} | {delta:12.6e}")
        
        if delta < tol:
            print("Lambda-CCSDT Converged!")
            break
        old_norm = new_norm
            
    return l1, l2, l3


def compute_density_matrices(t1, t2, l1, l2, n_occ, n_states):
    """
    Compute density matrices from T and Lambda.
    """
    o = slice(0, n_occ)
    v = slice(n_occ, n_states)
    
    rho_oo = np.eye(n_occ)
    rho_oo -= 0.5 * contract('imef,jmef->ij', l2, t2)
    
    rho_vv = np.zeros((n_states - n_occ, n_states - n_occ))
    rho_vv += 0.5 * contract('mnae,mnbe->ab', l2, t2)
    
    rho_ov = l1.copy()
    
    return rho_oo, rho_vv, rho_ov


def compute_properties(no_ham, t1, t2, l1, l2, n_occ):
    """
    Compute properties.
    """
    n_states = no_ham.f.shape[0]
    rho_oo, rho_vv, rho_ov = compute_density_matrices(t1, t2, l1, l2, n_occ, n_states)
    return {"rho_oo": rho_oo, "rho_vv": rho_vv, "rho_ov": rho_ov}
