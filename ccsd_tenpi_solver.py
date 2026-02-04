import numpy as np
from numpy_tenpi_ccsd import numpy_tenpi_ccsd

def ccsd_tenpi_solver(no_ham, n_occ, max_iter=50, tol=1e-6, damping=0.0, 
                      diis_size=6, diis_start=3, initial_t1=None, initial_t2=None):
    """
    CCSD solver using the tenpi-generated numpy_tenpi_ccsd function.
    
    Parameters:
    -----------
    no_ham : NormalOrderedHamiltonian
        Normal ordered Hamiltonian
    n_occ : int
        Number of occupied orbitals
    max_iter : int
        Maximum iterations
    tol : float
        Convergence tolerance
    damping : float
        Damping factor
        
    Returns:
    --------
    e_corr : float
        Correlation energy
    t1 : ndarray
        Converged T1 amplitudes [nvir, nocc]
    t2 : ndarray
        Converged T2 amplitudes [nvir, nvir, nocc, nocc]
    """
    
    # Extract orbital dimensions
    n_total = no_ham.f.shape[0]
    n_vir = n_total - n_occ
    nocc = n_occ
    nvir = n_vir
    
    print(f"CCSD Solver (tenpi): nocc={nocc}, nvir={nvir}")
    print(f"DIIS settings: size={diis_size}, start={diis_start}, damping={damping}")
    
    # Partition Fock matrix
    f = no_ham.f
    F_oo = f[:nocc, :nocc].copy()       # F[0]: hole-hole
    F_vo = f[nocc:, :nocc].copy()       # F[1]: particle-hole
    F_ov = f[:nocc, nocc:].copy()       # F[2]: hole-particle
    F_vv = f[nocc:, nocc:].copy()       # F[3]: particle-particle
    
    F = [F_oo, F_vo, F_ov, F_vv]
    
    # Partition two-body interaction
    Gamma = no_ham.Gamma
    
    V_oooo = Gamma[:nocc, :nocc, :nocc, :nocc].copy()           # V[0]
    V_vooo = Gamma[nocc:, :nocc, :nocc, :nocc].copy()           # V[1]
    V_vvoo = Gamma[nocc:, nocc:, :nocc, :nocc].copy()           # V[2]
    V_ooov = Gamma[:nocc, :nocc, :nocc, nocc:].copy()           # V[3]
    V_voov = Gamma[nocc:, :nocc, :nocc, nocc:].copy()           # V[4]
    V_vvov = Gamma[nocc:, nocc:, :nocc, nocc:].copy()           # V[5]
    V_oovv = Gamma[:nocc, :nocc, nocc:, nocc:].copy()           # V[6]
    V_vovv = Gamma[nocc:, :nocc, nocc:, nocc:].copy()           # V[7]
    V_vvvv = Gamma[nocc:, nocc:, nocc:, nocc:].copy()           # V[8]
    
    V = [V_oooo, V_vooo, V_vvoo, V_ooov, V_voov, V_vvov, V_oovv, V_vovv, V_vvvv]
    
    # Compute orbital energy denominators
    eps_o = np.diag(F_oo)
    eps_v = np.diag(F_vv)
    
    # Initialize amplitudes
    if initial_t1 is not None:
        # CCSD returns T1 as [nocc, nvir], but tenpi expects [nvir, nocc]
        if initial_t1.shape[0] == nocc:
            t1 = initial_t1.T.copy()
        else:
            t1 = initial_t1.copy()
    else:
        t1 = np.zeros((nvir, nocc), dtype=np.float64)
        
    if initial_t2 is not None:
        # CCSD returns T2 as [nocc, nocc, nvir, nvir], but tenpi expects [nvir, nvir, nocc, nocc]
        if initial_t2.shape[0] == nocc:
            t2 = initial_t2.transpose(2, 3, 0, 1).copy()
        else:
            t2 = initial_t2.copy()
    else:
        # Initialize T2 with MP2 guess
        t2 = np.zeros((nvir, nvir, nocc, nocc), dtype=np.float64)
        for a in range(nvir):
            for b in range(nvir):
                for i in range(nocc):
                    for j in range(nocc):
                        denom = eps_o[i] + eps_o[j] - eps_v[a] - eps_v[b]
                        if abs(denom) > 1e-12:
                            # Note: V_vvoo is V[2]. Wait, MP2 uses <ab||ij> or <ij||ab>?
                            # Usually <ab||ij> / D_ij^ab.
                            # V_vvoo is Gamma[nocc:, nocc:, :nocc, :nocc] -> <ab||ij>.
                            t2[a, b, i, j] = V_vvoo[a, b, i, j] / denom
    
    T = [t1, t2]
    
    # DIIS storage
    diis_t1 = []
    diis_t2 = []
    diis_res1 = []
    diis_res2 = []
    
    print(f"\n{'Iter':>4} {'E_corr':>15} {'Delta_E':>12} {'|R1|_rms':>12} {'|R2|_rms':>12}")
    print("-" * 65)
    
    e_corr_old = 0.0
    
    for iteration in range(max_iter):
        
        result = numpy_tenpi_ccsd(nocc, nvir, F, V, T, type_=np.float64)
        e_corr = result[0]
        r1 = result[1]
        r2 = result[2]
        
        delta_e = abs(e_corr - e_corr_old)
        r1_rms = np.sqrt(np.mean(r1**2))
        r2_rms = np.sqrt(np.mean(r2**2))
        
        print(f"{iteration:4d} {e_corr:15.8f} {delta_e:12.4e} {r1_rms:12.4e} {r2_rms:12.4e}")
        
        if delta_e < tol and r1_rms < tol and r2_rms < tol:
            print("\nConverged!")
            break
            
        # Update T1
        t1_new = t1.copy()
        for a in range(nvir):
            for i in range(nocc):
                denom = eps_o[i] - eps_v[a]
                if abs(denom) > 1e-12:
                    t1_new[a, i] = t1[a, i] + r1[a, i] / denom
                    
        # Update T2
        t2_new = t2.copy()
        for a in range(nvir):
            for b in range(nvir):
                for i in range(nocc):
                    for j in range(nocc):
                        denom = eps_o[i] + eps_o[j] - eps_v[a] - eps_v[b]
                        if abs(denom) > 1e-12:
                            t2_new[a, b, i, j] = t2[a, b, i, j] + r2[a, b, i, j] / denom
                            
        if damping > 0:
            t1_new = damping * t1 + (1.0 - damping) * t1_new
            t2_new = damping * t2 + (1.0 - damping) * t2_new
            
        # DIIS
        if iteration >= diis_start:
            diis_t1.append(t1_new.copy())
            diis_t2.append(t2_new.copy())
            diis_res1.append(r1.copy())
            diis_res2.append(r2.copy())
            
            if len(diis_t1) > diis_size:
                diis_t1.pop(0)
                diis_t2.pop(0)
                diis_res1.pop(0)
                diis_res2.pop(0)
                
            if len(diis_t1) >= 2:
                t1_new, t2_new = diis_extrapolate_ccsd(diis_t1, diis_t2, diis_res1, diis_res2)
                
        t1 = t1_new
        t2 = t2_new
        T = [t1, t2]
        e_corr_old = e_corr
        
    return e_corr, t1, t2


def diis_extrapolate_ccsd(t1_vectors, t2_vectors, r1_vectors, r2_vectors):
    n = len(t1_vectors)
    B = np.zeros((n + 1, n + 1))
    for i in range(n):
        for j in range(n):
            overlap = np.sum(r1_vectors[i] * r1_vectors[j]) + np.sum(r2_vectors[i] * r2_vectors[j])
            B[i, j] = overlap
            
    B[n, :n] = -1.0
    B[:n, n] = -1.0
    B[n, n] = 0.0
    
    rhs = np.zeros(n + 1)
    rhs[n] = -1.0
    
    try:
        coeffs = np.linalg.solve(B, rhs)
    except np.linalg.LinAlgError:
        return t1_vectors[-1].copy(), t2_vectors[-1].copy()
        
    t1_new = np.zeros_like(t1_vectors[0])
    t2_new = np.zeros_like(t2_vectors[0])
    
    for i in range(n):
        t1_new += coeffs[i] * t1_vectors[i]
        t2_new += coeffs[i] * t2_vectors[i]
        
    return t1_new, t2_new
