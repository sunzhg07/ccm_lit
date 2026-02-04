import numpy as np
from numpy_tenpi_ccdtq import numpy_tenpi_ccdtq

def ccdtq_tenpi_solver(no_ham, n_occ, max_iter=20, tol=1e-6, damping=0.0,
                        diis_size=4, diis_start=3,
                        initial_t2=None, initial_t3=None, initial_t4=None):
    """
    CCDTQ solver using the tenpi-generated numpy_tenpi_ccdtq function.
    
    This solver assumes T1 = 0.
    
    ⚠️  **EXTREME MEMORY WARNING** ⚠️
    T4 has size nvir^4 × nocc^4. 
    
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
        CCDTQ correlation energy
    t2 : ndarray
        Converged T2 amplitudes
    t3 : ndarray
        Converged T3 amplitudes
    t4 : ndarray
        Converged T4 amplitudes
    """
    
    # Extract orbital dimensions
    n_total = no_ham.f.shape[0]
    n_vir = n_total - n_occ
    nocc = n_occ
    nvir = n_vir
    
    print(f"CCDTQ Solver (tenpi): nocc={nocc}, nvir={nvir}")
    print(f"DIIS settings: size={diis_size}, start={diis_start}, damping={damping}")
    
    # Partition Fock matrix and Interaction
    f = no_ham.f
    F_oo = f[:nocc, :nocc].copy()
    F_vv = f[nocc:, nocc:].copy()
    # Need full list for tenpi input
    F = [
        f[:nocc, :nocc].copy(),
        f[nocc:, :nocc].copy(), # F_vo
        f[:nocc, nocc:].copy(), # F_ov
        f[nocc:, nocc:].copy()
    ]
    
    Gamma = no_ham.Gamma
    V = [
        Gamma[:nocc, :nocc, :nocc, :nocc].copy(),
        Gamma[nocc:, :nocc, :nocc, :nocc].copy(),
        Gamma[nocc:, nocc:, :nocc, :nocc].copy(),
        Gamma[:nocc, :nocc, :nocc, nocc:].copy(),
        Gamma[nocc:, :nocc, :nocc, nocc:].copy(),
        Gamma[nocc:, nocc:, :nocc, nocc:].copy(),
        Gamma[:nocc, :nocc, nocc:, nocc:].copy(), # V[6] = V_oovv
        Gamma[nocc:, :nocc, nocc:, nocc:].copy(),
        Gamma[nocc:, nocc:, nocc:, nocc:].copy()
    ]
    V_vvoo = V[2]  # Needed for MP2 guess if t2 is None
    
    # Compute orbital energy denominators
    eps_o = np.diag(F_oo)
    eps_v = np.diag(F_vv)
    
    # Initialize amplitudes
    # T1 is implicitly zero. We need to pass a dummy T1 to tenpi func?
    # numpy_tenpi_ccdtq unpacks T[0] as T1.
    t1 = np.zeros((nvir, nocc), dtype=np.float64)
    
    if initial_t2 is not None:
        # CCSD returns T2 as [nocc, nocc, nvir, nvir], but tenpi expects [nvir, nvir, nocc, nocc]
        if initial_t2.shape[0] == nocc and initial_t2.shape[2] == nvir: 
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
                            t2[a, b, i, j] = V_vvoo[a, b, i, j] / denom
    
    if initial_t3 is not None:
        # Standard T3 is [O,O,O,V,V,V], tenpi needs [V,V,V,O,O,O]
        if initial_t3.shape[0] == nocc:
             t3 = initial_t3.transpose(3, 4, 5, 0, 1, 2).copy()
        else:
             t3 = initial_t3.copy()
    else:
        t3 = np.zeros((nvir, nvir, nvir, nocc, nocc, nocc), dtype=np.float64)
    
    if initial_t4 is not None:
        # Standard T4 is [O,O,O,O,V,V,V,V], tenpi needs [V,V,V,V,O,O,O,O]
        if initial_t4.shape[0] == nocc:
             t4 = initial_t4.transpose(4, 5, 6, 7, 0, 1, 2, 3).copy()
        else:
             t4 = initial_t4.copy()
    else:
        t4 = np.zeros((nvir, nvir, nvir, nvir, nocc, nocc, nocc, nocc), dtype=np.float64)
    
    T = [t1, t2, t3, t4]
    
    # DIIS storage
    diis_t2 = []
    diis_t3 = []
    diis_t4 = []
    diis_res2 = []
    diis_res3 = []
    diis_res4 = []
    
    print(f"\n{'Iter':>4} {'E_corr':>15} {'Delta_E':>12} {'|R2|_rms':>12} {'|R3|_rms':>12} {'|R4|_rms':>12}")
    print("-" * 80)
    
    e_corr_old = 0.0
    
    # Debug shapes once
    print(f"Debug V Shapes:")
    for i, v_block in enumerate(V):
        print(f"  V[{i}]: {v_block.shape}")
    print(f"Debug T Shapes:")
    print(f"  T2: {t2.shape}")
    print(f"  T3: {t3.shape}")
    print(f"  T4: {t4.shape}")

    for iteration in range(max_iter):
        print(f"  Iteration {iteration}: Computing residuals...")
        
        # Result = [Z0, DummyZ1, Z2, Z3, Z4]
        result = numpy_tenpi_ccdtq(nocc, nvir, F, V, T, type_=np.float64)
        e_corr = result[0]
        # Skip result[1] which is dummy Z1 or Z2
        r2 = result[2]  # Residual for T2
        r3 = result[3]  # Residual for T3
        r4 = result[4]  # Residual for T4
        
        delta_e = abs(e_corr - e_corr_old)
        r2_rms = np.sqrt(np.mean(r2**2))
        r3_rms = np.sqrt(np.mean(r3**2))
        r4_rms = np.sqrt(np.mean(r4**2))
        
        print(f"{iteration:4d} {e_corr:15.8f} {delta_e:12.4e} {r2_rms:12.4e} {r3_rms:12.4e} {r4_rms:12.4e}")
        
        # Check convergence
        if delta_e < tol and r2_rms < tol and r3_rms < tol and r4_rms < tol:
            print("\nConverged!")
            break
        
        # Update T1? No, T1 stays zero.
        
        # Update T2 amplitudes
        t2_new = t2.copy()
        for a in range(nvir):
            for b in range(nvir):
                for i in range(nocc):
                    for j in range(nocc):
                        denom = eps_o[i] + eps_o[j] - eps_v[a] - eps_v[b]
                        if abs(denom) > 1e-12:
                            t2_new[a, b, i, j] = t2[a, b, i, j] + r2[a, b, i, j] / denom
        
        # Update T3 amplitudes
        t3_new = t3.copy()
        for a in range(nvir):
            for b in range(nvir):
                for c in range(nvir):
                    for i in range(nocc):
                        for j in range(nocc):
                            for k in range(nocc):
                                denom = eps_o[i] + eps_o[j] + eps_o[k] - eps_v[a] - eps_v[b] - eps_v[c]
                                if abs(denom) > 1e-12:
                                    t3_new[a, b, c, i, j, k] = t3[a, b, c, i, j, k] + r3[a, b, c, i, j, k] / denom
        
        # Update T4 amplitudes
        print("    Updating T4...")
        t4_new = t4.copy()
        # Vectorized update for T4 to speed it up?
        # Denom is separable: (e_i+e_j+...) - (e_a+e_b+...)
        # Constructing full denom array is expensive memory wise. 
        # Loop is slow. Vectorize partially.
        # Let's keep loop for now to avoid memory explosion, just like ccsdtq solver.
        for a in range(nvir):
            for b in range(nvir):
                for c in range(nvir):
                    for d in range(nvir):
                        for i in range(nocc):
                           for j in range(nocc):
                                for k in range(nocc):
                                    for l in range(nocc):
                                        denom = (eps_o[i] + eps_o[j] + eps_o[k] + eps_o[l] - 
                                                eps_v[a] - eps_v[b] - eps_v[c] - eps_v[d])
                                        if abs(denom) > 1e-12:
                                            t4_new[a, b, c, d, i, j, k, l] = (t4[a, b, c, d, i, j, k, l] + 
                                                                              r4[a, b, c, d, i, j, k, l] / denom)
        
        if damping > 0:
            t2_new = damping * t2 + (1.0 - damping) * t2_new
            t3_new = damping * t3 + (1.0 - damping) * t3_new
            t4_new = damping * t4 + (1.0 - damping) * t4_new
        
        # DIIS acceleration
        if iteration >= diis_start:
            diis_t2.append(t2_new.copy())
            diis_t3.append(t3_new.copy())
            diis_t4.append(t4_new.copy())
            diis_res2.append(r2.copy())
            diis_res3.append(r3.copy())
            diis_res4.append(r4.copy())
            
            if len(diis_t2) > diis_size:
                diis_t2.pop(0)
                diis_t3.pop(0)
                diis_t4.pop(0)
                diis_res2.pop(0)
                diis_res3.pop(0)
                diis_res4.pop(0)
            
            if len(diis_t2) >= 2:
                print("    Applying DIIS...")
                t2_new, t3_new, t4_new = diis_extrapolate_ccdtq(
                    diis_t2, diis_t3, diis_t4,
                    diis_res2, diis_res3, diis_res4
                )
        
        t2 = t2_new
        t3 = t3_new
        t4 = t4_new
        T = [t1, t2, t3, t4]
        e_corr_old = e_corr
    
    return e_corr, t2, t3, t4


def diis_extrapolate_ccdtq(t2_vectors, t3_vectors, t4_vectors,
                           r2_vectors, r3_vectors, r4_vectors):
    n = len(t2_vectors)
    B = np.zeros((n + 1, n + 1))
    for i in range(n):
        for j in range(n):
            overlap = (np.sum(r2_vectors[i] * r2_vectors[j]) +
                      np.sum(r3_vectors[i] * r3_vectors[j]) +
                      np.sum(r4_vectors[i] * r4_vectors[j]))
            B[i, j] = overlap
    
    B[n, :n] = -1.0
    B[:n, n] = -1.0
    B[n, n] = 0.0
    
    rhs = np.zeros(n + 1)
    rhs[n] = -1.0
    
    try:
        coeffs = np.linalg.solve(B, rhs)
    except np.linalg.LinAlgError:
         return t2_vectors[-1].copy(), t3_vectors[-1].copy(), t4_vectors[-1].copy()
    
    t2_new = np.zeros_like(t2_vectors[0])
    t3_new = np.zeros_like(t3_vectors[0])
    t4_new = np.zeros_like(t4_vectors[0])
    
    for i in range(n):
        t2_new += coeffs[i] * t2_vectors[i]
        t3_new += coeffs[i] * t3_vectors[i]
        t4_new += coeffs[i] * t4_vectors[i]
    
    return t2_new, t3_new, t4_new
