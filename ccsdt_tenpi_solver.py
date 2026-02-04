import numpy as np
from numpy_tenpi_ccsdt import numpy_tenpi_ccsdt

def ccsdt_tenpi_solver(no_ham, n_occ, max_iter=30, tol=1e-6, damping=0.0, 
                       diis_size=6, diis_start=3, initial_t1=None, initial_t2=None, initial_t3=None):
    """
    CCSDT solver using the tenpi-generated numpy_tenpi_ccsdt function.
    
    Parameters:
    -----------
    no_ham : NormalOrderedHamiltonian
        Normal ordered Hamiltonian with E0, f (Fock matrix), and Gamma (two-body interaction)
    n_occ : int
        Number of occupied orbitals
    max_iter : int
        Maximum number of iterations
    tol : float
        Convergence tolerance for energy
    damping : float
        Damping factor (0.0 = no damping, 0.5 = half old, half new)
    diis_size : int
        DIIS vector space size
    diis_start : int
        Iteration to start DIIS acceleration
    initial_t1 : ndarray, optional
        Initial T1 amplitudes (default: zeros)
    initial_t2 : ndarray, optional
        Initial T2 amplitudes (default: MP2 guess)
    initial_t3 : ndarray, optional
        Initial T3 amplitudes (default: zeros)
        
    Returns:
    --------
    e_corr : float
        CCSDT correlation energy
    t1 : ndarray
        Converged T1 amplitudes [nvir, nocc]
    t2 : ndarray
        Converged T2 amplitudes [nvir, nvir, nocc, nocc]
    t3 : ndarray
        Converged T3 amplitudes [nvir, nvir, nvir, nocc, nocc, nocc]
    """
    
    # Extract orbital dimensions
    n_total = no_ham.f.shape[0]
    n_vir = n_total - n_occ
    nocc = n_occ
    nvir = n_vir
    
    print(f"CCSDT Solver (tenpi): nocc={nocc}, nvir={nvir}")
    print(f"DIIS settings: size={diis_size}, start={diis_start}, damping={damping}")
    print(f"WARNING: T3 is expensive! Size: [{nvir}x{nvir}x{nvir}x{nocc}x{nocc}x{nocc}] = {nvir**3 * nocc**3} elements")
    
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
    
    # Debug: Print shapes
    print(f"\nDebug - Block shapes:")
    print(f"  F blocks: {[f.shape for f in F]}")
    print(f"  V blocks: {[v.shape for v in V]}")
    print(f"  Expected V[6] (OOVV): [{nocc}, {nocc}, {nvir}, {nvir}] = [8, 8, 4, 4]")
    print(f"  Actual V[6] shape: {V[6].shape}")
    
    # Compute orbital energy denominators
    eps_o = np.diag(F_oo)  # Occupied orbital energies
    eps_v = np.diag(F_vv)  # Virtual orbital energies
    
    # Initialize amplitudes
    if initial_t1 is not None:
        # CCSD returns T1 as [nocc, nvir], but tenpi expects [nvir, nocc]
        if initial_t1.shape == (nocc, nvir):
            t1 = initial_t1.T.copy()
        elif initial_t1.shape == (nvir, nocc):
            t1 = initial_t1.copy()
        else:
            raise ValueError(f"initial_t1 shape {initial_t1.shape} matches neither standard ([{nocc},{nvir}]) nor tenpi ([{nvir},{nocc}])")
    else:
        t1 = np.zeros((nvir, nocc), dtype=np.float64)
    
    if initial_t2 is not None:
        # CCSD returns T2 as [nocc, nocc, nvir, nvir], but tenpi expects [nvir, nvir, nocc, nocc]
        if initial_t2.shape == (nocc, nocc, nvir, nvir):
            t2 = initial_t2.transpose(2, 3, 0, 1).copy()
        elif initial_t2.shape == (nvir, nvir, nocc, nocc):
            t2 = initial_t2.copy()
        else:
             raise ValueError(f"initial_t2 shape {initial_t2.shape} matches neither standard nor tenpi format")
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
        t3 = initial_t3.copy()
    else:
        # Initialize T3 to zeros (could use a better guess but it's expensive)
        t3 = np.zeros((nvir, nvir, nvir, nocc, nocc, nocc), dtype=np.float64)
    
    T = [t1, t2, t3]
    
    # Debug: Print T shapes
    print(f"  T1 shape: {t1.shape} (expected: [{nvir}, {nocc}] = [4, 8])")
    print(f"  T2 shape: {t2.shape} (expected: [{nvir}, {nvir}, {nocc}, {nocc}] = [4, 4, 8, 8])")
    print(f"  T3 shape: {t3.shape} (expected: [{nvir}, {nvir}, {nvir}, {nocc}, {nocc}, {nocc}] = [4, 4, 4, 8, 8, 8])")
    
    # DIIS storage
    diis_t1 = []
    diis_t2 = []
    diis_t3 = []
    diis_res1 = []
    diis_res2 = []
    diis_res3 = []
    
    print(f"\n{'Iter':>4} {'E_corr':>15} {'Delta_E':>12} {'|R1|_rms':>12} {'|R2|_rms':>12} {'|R3|_rms':>12}")
    print("-" * 75)
    
    e_corr_old = 0.0
    
    for iteration in range(max_iter):
        # Compute residuals and energy using tenpi function
        result = numpy_tenpi_ccsdt(nocc, nvir, F, V, T, type_=np.float64)
        e_corr = result[0]
        r1 = result[1]  # Residual for T1
        r2 = result[2]  # Residual for T2
        r3 = result[3]  # Residual for T3
        
        delta_e = abs(e_corr - e_corr_old)
        r1_rms = np.sqrt(np.mean(r1**2))
        r2_rms = np.sqrt(np.mean(r2**2))
        r3_rms = np.sqrt(np.mean(r3**2))
        
        print(f"{iteration:4d} {e_corr:15.8f} {delta_e:12.4e} {r1_rms:12.4e} {r2_rms:12.4e} {r3_rms:12.4e}")
        
        # Check convergence
        if delta_e < tol and r1_rms < tol and r2_rms < tol and r3_rms < tol:
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
        
        # Apply damping
        if damping > 0:
            t1_new = damping * t1 + (1.0 - damping) * t1_new
            t2_new = damping * t2 + (1.0 - damping) * t2_new
            t3_new = damping * t3 + (1.0 - damping) * t3_new
        
        # DIIS acceleration
        if iteration >= diis_start:
            diis_t1.append(t1_new.copy())
            diis_t2.append(t2_new.copy())
            diis_t3.append(t3_new.copy())
            diis_res1.append(r1.copy())
            diis_res2.append(r2.copy())
            diis_res3.append(r3.copy())
            
            # Keep only the most recent diis_size vectors
            if len(diis_t1) > diis_size:
                diis_t1.pop(0)
                diis_t2.pop(0)
                diis_t3.pop(0)
                diis_res1.pop(0)
                diis_res2.pop(0)
                diis_res3.pop(0)
            
            # Perform DIIS extrapolation
            if len(diis_t1) >= 2:
                t1_new, t2_new, t3_new = diis_extrapolate_ccsdt(
                    diis_t1, diis_t2, diis_t3,
                    diis_res1, diis_res2, diis_res3
                )
        
        t1 = t1_new
        t2 = t2_new
        t3 = t3_new
        T = [t1, t2, t3]
        e_corr_old = e_corr
    
    else:
        print("\nWarning: Maximum iterations reached without convergence!")
    
    return e_corr, t1, t2, t3


def diis_extrapolate_ccsdt(t1_vectors, t2_vectors, t3_vectors, r1_vectors, r2_vectors, r3_vectors):
    """
    DIIS extrapolation for CCSDT.
    
    Combines residuals from T1, T2, and T3 into a single DIIS procedure.
    """
    n = len(t1_vectors)
    
    # Build B matrix: B_ij = <r_i | r_j> where r includes r1, r2, and r3
    B = np.zeros((n + 1, n + 1))
    for i in range(n):
        for j in range(n):
            # Combine all residual contributions
            overlap = (np.sum(r1_vectors[i] * r1_vectors[j]) +
                      np.sum(r2_vectors[i] * r2_vectors[j]) +
                      np.sum(r3_vectors[i] * r3_vectors[j]))
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
        return t1_vectors[-1].copy(), t2_vectors[-1].copy(), t3_vectors[-1].copy()
    
    # Extrapolate all amplitudes
    t1_new = np.zeros_like(t1_vectors[0])
    t2_new = np.zeros_like(t2_vectors[0])
    t3_new = np.zeros_like(t3_vectors[0])
    
    for i in range(n):
        t1_new += coeffs[i] * t1_vectors[i]
        t2_new += coeffs[i] * t2_vectors[i]
        t3_new += coeffs[i] * t3_vectors[i]
    
    return t1_new, t2_new, t3_new


if __name__ == "__main__":
    # Test with main.py workflow
    print("Test CCSDT tenpi solver")
    print("Run this via main.py or import and use with NormalOrderedHamiltonian")
