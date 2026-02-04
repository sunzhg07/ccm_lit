import numpy as np
from numpy_tenpi_ccsdtq import numpy_tenpi_ccsdtq

def ccsdtq_tenpi_solver(no_ham, n_occ, max_iter=20, tol=1e-6, damping=0.0,
                        diis_size=4, diis_start=3,
                        initial_t1=None, initial_t2=None, initial_t3=None, initial_t4=None):
    """
    CCSDTQ solver using the tenpi-generated numpy_tenpi_ccsdtq function.
    
    ⚠️  **EXTREME MEMORY WARNING** ⚠️
    T4 has size nvir^4 × nocc^4. For 8 occ and 4 vir: 262,144 elements (~2 MB).
    This scales VERY rapidly! 10 occ, 6 vir = 16.8 million elements (~128 MB).
    
    Parameters:
    -----------
    no_ham : NormalOrderedHamiltonian
        Normal ordered Hamiltonian with E0, f (Fock matrix), and Gamma (two-body interaction)
    n_occ : int
        Number of occupied orbitals
    max_iter : int
        Maximum iterations (default: 20, CCSDTQ is VERY expensive!)
    tol : float
        Convergence tolerance (default: 1e-6)
    damping : float
        Damping factor (default: 0.0)
    diis_size : int
        DIIS vector space size (default: 4, use small for CCSDTQ!)
    diis_start : int
        Iteration to start DIIS (default: 3)
    initial_t1 : ndarray, optional
        Initial T1 amplitudes [nocc, nvir] from CCSD (STRONGLY recommended!)
    initial_t2 : ndarray, optional
        Initial T2 amplitudes [nocc, nocc, nvir, nvir] from CCSD
    initial_t3 : ndarray, optional
        Initial T3 amplitudes [nvir, nvir, nvir, nocc, nocc, nocc] from CCSDT
    initial_t4 : ndarray, optional
        Initial T4 amplitudes [nvir, nvir, nvir, nvir, nocc, nocc, nocc, nocc]
        
    Returns:
    --------
    e_corr : float
        CCSDTQ correlation energy
    t1 : ndarray
        Converged T1 amplitudes [nvir, nocc]
    t2 : ndarray
        Converged T2 amplitudes [nvir, nvir, nocc, nocc]
    t3 : ndarray
        Converged T3 amplitudes [nvir, nvir, nvir, nocc, nocc, nocc]
    t4 : ndarray
        Converged T4 amplitudes [nvir, nvir, nvir, nvir, nocc, nocc, nocc, nocc]
    """
    
    # Extract orbital dimensions
    n_total = no_ham.f.shape[0]
    n_vir = n_total - n_occ
    nocc = n_occ
    nvir = n_vir
    
    # Calculate T4 size
    t4_elements = nvir**4 * nocc**4
    t4_memory_mb = t4_elements * 8 / (1024**2)  # float64 = 8 bytes
    
    print(f"CCSDTQ Solver (tenpi): nocc={nocc}, nvir={nvir}")
    print(f"DIIS settings: size={diis_size}, start={diis_start}, damping={damping}")
    print(f"⚠️  MEMORY WARNING:")
    print(f"  T3 size: [{nvir}x{nvir}x{nvir}x{nocc}x{nocc}x{nocc}] = {nvir**3 * nocc**3:,} elements")
    print(f"  T4 size: [{nvir}x{nvir}x{nvir}x{nvir}x{nocc}x{nocc}x{nocc}x{nocc}] = {t4_elements:,} elements")
    print(f"  T4 memory: ~{t4_memory_mb:.1f} MB")
    
    if t4_memory_mb > 1000:
        print(f"  ⚠️⚠️⚠️  WARNING: T4 requires >{t4_memory_mb/1024:.1f} GB! This may fail!")
    
    # Partition Fock matrix into blocks
    f = no_ham.f
    F_oo = f[:nocc, :nocc].copy()       # F[0]: hole-hole
    F_vo = f[nocc:, :nocc].copy()       # F[1]: particle-hole
    F_ov = f[:nocc, nocc:].copy()       # F[2]: hole-particle
    F_vv = f[nocc:, nocc:].copy()       # F[3]: particle-particle
    
    F = [F_oo, F_vo, F_ov, F_vv]
    
    # Partition two-body interaction into blocks
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
    eps_o = np.diag(F_oo)  # Occupied orbital energies
    eps_v = np.diag(F_vv)  # Virtual orbital energies
    
    # Initialize amplitudes
    # Initialize amplitudes
    if initial_t1 is not None:
        # CCSD returns T1 as [nocc, nvir], but tenpi expects [nvir, nocc]
        if initial_t1.shape == (nocc, nvir):
            t1 = initial_t1.T.copy()
        elif initial_t1.shape == (nvir, nocc):
            t1 = initial_t1.copy()
        else:
             raise ValueError(f"initial_t1 shape {initial_t1.shape} mismatch")
    else:
        t1 = np.zeros((nvir, nocc), dtype=np.float64)
    
    if initial_t2 is not None:
        # CCSD returns T2 as [nocc, nocc, nvir, nvir], but tenpi expects [nvir, nvir, nocc, nocc]
        if initial_t2.shape == (nocc, nocc, nvir, nvir):
            t2 = initial_t2.transpose(2, 3, 0, 1).copy()
        elif initial_t2.shape == (nvir, nvir, nocc, nocc):
            t2 = initial_t2.copy()
        else:
             raise ValueError(f"initial_t2 shape {initial_t2.shape} mismatch")
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
            t3 = initial_t3.copy()  # Already in tenpi format
    else:
        t3 = np.zeros((nvir, nvir, nvir, nocc, nocc, nocc), dtype=np.float64)
    
    if initial_t4 is not None:
        # Standard T4 is [O,O,O,O,V,V,V,V], tenpi needs [V,V,V,V,O,O,O,O]
        if initial_t4.shape[0] == nocc:
             t4 = initial_t4.transpose(4, 5, 6, 7, 0, 1, 2, 3).copy()
        else:
             t4 = initial_t4.copy()
    else:
        # Initialize T4 to zeros (disconnected T2*T2 guess could be used but is complex)
        print("  Initializing T4 to zeros (this may take a moment)...")
        t4 = np.zeros((nvir, nvir, nvir, nvir, nocc, nocc, nocc, nocc), dtype=np.float64)
        print("  T4 initialized.")
    
    T = [t1, t2, t3, t4]
    
    # DIIS storage
    diis_t1 = []
    diis_t2 = []
    diis_t3 = []
    diis_t4 = []
    diis_res1 = []
    diis_res2 = []
    diis_res3 = []
    diis_res4 = []
    
    print(f"\n{'Iter':>4} {'E_corr':>15} {'Delta_E':>12} {'|R1|_rms':>12} {'|R2|_rms':>12} {'|R3|_rms':>12} {'|R4|_rms':>12}")
    print("-" * 95)
    
    e_corr_old = 0.0
    
    # Debug shapes
    print(f"Debug T Shapes:")
    print(f"  T1: {t1.shape}")
    print(f"  T2: {t2.shape}")
    print(f"  T3: {t3.shape}")
    print(f"  T4: {t4.shape}")

    for iteration in range(max_iter):
        # Compute residuals and energy using tenpi function
        print(f"  Iteration {iteration}: Computing residuals (this is VERY slow)...")
        result = numpy_tenpi_ccsdtq(nocc, nvir, F, V, T, type_=np.float64)
        e_corr = result[0]
        r1 = result[1]  # Residual for T1
        r2 = result[2]  # Residual for T2
        r3 = result[3]  # Residual for T3
        r4 = result[4]  # Residual for T4
        
        delta_e = abs(e_corr - e_corr_old)
        r1_rms = np.sqrt(np.mean(r1**2))
        r2_rms = np.sqrt(np.mean(r2**2))
        r3_rms = np.sqrt(np.mean(r3**2))
        r4_rms = np.sqrt(np.mean(r4**2))
        
        print(f"{iteration:4d} {e_corr:15.8f} {delta_e:12.4e} {r1_rms:12.4e} {r2_rms:12.4e} {r3_rms:12.4e} {r4_rms:12.4e}")
        
        # Check convergence
        if delta_e < tol and r1_rms < tol and r2_rms < tol and r3_rms < tol and r4_rms < tol:
            print("\nConverged!")
            break
        
        # Update T1 amplitudes
        t1_new = t1.copy()
        for a in range(nvir):
            for i in range(nocc):
                denom = eps_o[i] - eps_v[a]
                if abs(denom) > 1e-12:
                    t1_new[a, i] = t1[a, i] + r1[a, i] / denom
        
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
        print("    Updating T3...")
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
        
        # Update T4 amplitudes (this is VERY slow)
        print("    Updating T4 (this takes time)...")
        t4_new = t4.copy()
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
        
        # Apply damping
        if damping > 0:
            t1_new = damping * t1 + (1.0 - damping) * t1_new
            t2_new = damping * t2 + (1.0 - damping) * t2_new
            t3_new = damping * t3 + (1.0 - damping) * t3_new
            t4_new = damping * t4 + (1.0 - damping) * t4_new
        
        # DIIS acceleration
        if iteration >= diis_start:
            diis_t1.append(t1_new.copy())
            diis_t2.append(t2_new.copy())
            diis_t3.append(t3_new.copy())
            diis_t4.append(t4_new.copy())
            diis_res1.append(r1.copy())
            diis_res2.append(r2.copy())
            diis_res3.append(r3.copy())
            diis_res4.append(r4.copy())
            
            # Keep only the most recent diis_size vectors
            if len(diis_t1) > diis_size:
                diis_t1.pop(0)
                diis_t2.pop(0)
                diis_t3.pop(0)
                diis_t4.pop(0)
                diis_res1.pop(0)
                diis_res2.pop(0)
                diis_res3.pop(0)
                diis_res4.pop(0)
            
            # Perform DIIS extrapolation
            if len(diis_t1) >= 2:
                print("    Applying DIIS...")
                t1_new, t2_new, t3_new, t4_new = diis_extrapolate_ccsdtq(
                    diis_t1, diis_t2, diis_t3, diis_t4,
                    diis_res1, diis_res2, diis_res3, diis_res4
                )
        
        t1 = t1_new
        t2 = t2_new
        t3 = t3_new
        t4 = t4_new
        T = [t1, t2, t3, t4]
        e_corr_old = e_corr
    
    else:
        print("\nWarning: Maximum iterations reached without convergence!")
    
    return e_corr, t1, t2, t3, t4


def diis_extrapolate_ccsdtq(t1_vectors, t2_vectors, t3_vectors, t4_vectors,
                             r1_vectors, r2_vectors, r3_vectors, r4_vectors):
    """
    DIIS extrapolation for CCSDTQ.
    
    Combines residuals from T1, T2, T3, and T4 into a single DIIS procedure.
    """
    n = len(t1_vectors)
    
    # Build B matrix: B_ij = <r_i | r_j> where r includes r1, r2, r3, and r4
    B = np.zeros((n + 1, n + 1))
    for i in range(n):
        for j in range(n):
            # Combine all residual contributions
            overlap = (np.sum(r1_vectors[i] * r1_vectors[j]) +
                      np.sum(r2_vectors[i] * r2_vectors[j]) +
                      np.sum(r3_vectors[i] * r3_vectors[j]) +
                      np.sum(r4_vectors[i] * r4_vectors[j]))
            B[i, j] = overlap
    
    # Constraint: sum of coefficients = 1
    B[n, :n] = -1.0
    B[:n, n] = -1.0
    B[n, n] = 0.0
    
    # Right-hand side
    rhs = np.zeros(n + 1)
    rhs[n] = -1.0
    
    # Solve for coefficients
    try:
        coeffs = np.linalg.solve(B, rhs)
    except np.linalg.LinAlgError:
        # If singular, just return the most recent vectors
        print("    DIIS: Singular B matrix, using most recent vectors")
        return (t1_vectors[-1].copy(), t2_vectors[-1].copy(), 
                t3_vectors[-1].copy(), t4_vectors[-1].copy())
    
    # Extrapolate all amplitudes
    t1_new = np.zeros_like(t1_vectors[0])
    t2_new = np.zeros_like(t2_vectors[0])
    t3_new = np.zeros_like(t3_vectors[0])
    t4_new = np.zeros_like(t4_vectors[0])
    
    for i in range(n):
        t1_new += coeffs[i] * t1_vectors[i]
        t2_new += coeffs[i] * t2_vectors[i]
        t3_new += coeffs[i] * t3_vectors[i]
        t4_new += coeffs[i] * t4_vectors[i]
    
    return t1_new, t2_new, t3_new, t4_new


if __name__ == "__main__":
    print("CCSDTQ tenpi solver")
    print("Use with NormalOrderedHamiltonian from your HF calculation")
    print("⚠️  WARNING: This is EXTREMELY computationally expensive!")
