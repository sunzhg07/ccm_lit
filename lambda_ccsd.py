import numpy as np
from opt_einsum import contract


def pAB(val):
    """Permutator val(abij) -> val(abij) - val(baij)"""
    return val - np.transpose(val, (1, 0, 2, 3))


def pIJ(val):
    """Permutator val(abij) -> val(abij) - val(abji)"""
    return val - np.transpose(val, (0, 1, 3, 2))


def lambda_ccsd(no_ham, t1, t2, n_occ, max_iter=200, tol=1e-8, alpha=0.5):
    """
    Solve Lambda-CCSD equations for the left eigenvector.
    
    The Lambda equations are:
    D_ia * l1_ia = L1_ia
    D_ijab * l2_ijab = L2_ijab
    
    where L1 and L2 are the Lambda residuals constructed from the 
    similarity-transformed Hamiltonian H_bar = e^(-T) H e^(T).
    
    Args:
        no_ham: Normal-ordered Hamiltonian object
        t1: Singles amplitudes from CCSD (n_occ, n_virt)
        t2: Doubles amplitudes from CCSD (n_occ, n_occ, n_virt, n_virt)
        n_occ: Number of occupied orbitals
        max_iter: Maximum iterations
        tol: Convergence tolerance
        alpha: Damping factor
        
    Returns:
        l1: Lambda singles (n_occ, n_virt)
        l2: Lambda doubles (n_occ, n_occ, n_virt, n_virt)
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
    
    # Energy denominators (same as T equations)
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
        # Simplified Lambda1 equations (essential terms)
        L1 = f_vo.T.copy()
        L1 += contract('ie,ea->ia', l1, f_vv)
        L1 -= contract('ma,mi->ia', l1, f_oo)
        
        # Key 2-body term
        L1 += 2.0 * contract('imae,me->ia', v2b[o, o, v, v], l1)
        L1 -= contract('imea,me->ia', v2b[o, o, v, v], l1)
        
        # --- Lambda2 Residual ---
        # L2_ijab = <ij||ab> + corrections
        L2 = v2b[o, o, v, v].copy()
        
        # Fock contributions
        L2 += pAB(contract('ie,ejab->ijab', l1, v2b[v, o, v, v]))
        L2 -= pIJ(contract('ma,imjb->ijab', l1, v2b[o, o, o, v]))
        
        # Ladder diagrams
        L2 += 0.5 * contract('abef,ijef->ijab', v2b[v, v, v, v], l2)
        L2 += 0.5 * contract('mnij,mnab->ijab', v2b[o, o, o, o], l2)
        
        # Ring diagram
        term = contract('mbej,imae->ijab', v2b[o, v, v, o], l2)
        L2 += pIJ(pAB(term))
        
        # Solve Lambda equations: D * l = L
        new_l1 = L1 / D1
        new_l2 = L2 / D2
        
        # Damped update
        l1 = (1 - alpha) * l1 + alpha * new_l1
        l2 = (1 - alpha) * l2 + alpha * new_l2
        
        # Check convergence
        new_norm = np.linalg.norm(l1) + np.linalg.norm(l2)
        delta = abs(new_norm - old_norm)
        
        norm_l1 = np.linalg.norm(l1)
        norm_l2 = np.linalg.norm(l2)
        
        print(f"[Lambda-CCSD] {iteration:4d} | {norm_l1:12.6e} | {norm_l2:12.6e} | {delta:12.6e}")
        
        if delta < tol:
            print("Lambda-CCSD Converged!")
            break
            
        old_norm = new_norm
        
        if np.isnan(new_norm):
            print("[Lambda-CCSD ERROR] NaN detected. Diverged.")
            break
    
    return l1, l2


def compute_density_matrices(t1, t2, l1, l2, n_occ, n_states):
    """
    Compute one- and two-body density matrices from T and Lambda amplitudes.
    
    These are needed for expectation values and properties:
    <O> = <0| (1 + Lambda) O_bar |0>
    
    where O_bar = e^(-T) O e^(T) is the similarity-transformed operator.
    
    Args:
        t1, t2: T amplitudes
        l1, l2: Lambda amplitudes
        n_occ: Number of occupied orbitals
        n_states: Total number of states
        
    Returns:
        rho_oo: Hole density matrix
        rho_vv: Particle density matrix
        rho_ov: Hole-particle density matrix
    """
    o = slice(0, n_occ)
    v = slice(n_occ, n_states)
    
    # One-body density matrices
    # rho_ij = delta_ij + corrections from Lambda and T
    rho_oo = np.eye(n_occ)
    rho_oo -= 0.5 * contract('imef,jmef->ij', l2, t2)
    
    # rho_ab = corrections from Lambda and T
    rho_vv = np.zeros((n_states - n_occ, n_states - n_occ))
    rho_vv += 0.5 * contract('mnae,mnbe->ab', l2, t2)
    
    # rho_ia (hole-particle)
    rho_ov = l1.copy()
    
    return rho_oo, rho_vv, rho_ov


def compute_properties(no_ham, t1, t2, l1, l2, n_occ):
    """
    Compute various properties using Lambda amplitudes.
    
    Args:
        no_ham: Normal-ordered Hamiltonian
        t1, t2: T amplitudes
        l1, l2: Lambda amplitudes
        n_occ: Number of occupied orbitals
        
    Returns:
        Dictionary of computed properties
    """
    n_states = no_ham.f.shape[0]
    o = slice(0, n_occ)
    v = slice(n_occ, n_states)
    
    # Compute density matrices
    rho_oo, rho_vv, rho_ov = compute_density_matrices(t1, t2, l1, l2, n_occ, n_states)
    
    # One-body expectation value: <F> = sum_ia f_ia rho_ia
    f_expectation = np.sum(no_ham.f[o, v] * rho_ov)
    
    # Particle number fluctuation
    n_particle = n_occ + np.trace(rho_vv) - np.trace(rho_oo - np.eye(n_occ))
    
    properties = {
        'rho_oo': rho_oo,
        'rho_vv': rho_vv,
        'rho_ov': rho_ov,
        'f_expectation': f_expectation,
        'n_particle': n_particle,
    }
    
    return properties
