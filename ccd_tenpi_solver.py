import numpy as np
from numpy_tenpi_ccd import numpy_tenpi_ccd

def ccd_tenpi_solver(no_ham, n_occ, max_iter=50, tol=1e-6, damping=0.0, diis_size=6, diis_start=3):
    """
    CCD solver using the tenpi-generated numpy_tenpi_ccd function.
    
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
        
    Returns:
    --------
    e_corr : float
        CCD correlation energy
    t2 : ndarray
        Converged T2 amplitudes [nvir, nvir, nocc, nocc]
    """
    
    # Extract orbital dimensions
    n_total = no_ham.f.shape[0]
    n_vir = n_total - n_occ
    nocc = n_occ
    nvir = n_vir
    
    print(f"CCD Solver (tenpi): nocc={nocc}, nvir={nvir}")
    print(f"DIIS settings: size={diis_size}, start={diis_start}, damping={damping}")
    
    # Partition Fock matrix into blocks
    f = no_ham.f
    F_oo = f[:nocc, :nocc].copy()       # F[0]: hole-hole
    F_vo = f[nocc:, :nocc].copy()       # F[1]: particle-hole
    F_ov = f[:nocc, nocc:].copy()       # F[2]: hole-particle
    F_vv = f[nocc:, nocc:].copy()       # F[3]: particle-particle
    
    F = [F_oo, F_vo, F_ov, F_vv]
    
    # Partition two-body interaction into blocks
    Gamma = no_ham.Gamma
    
    # Extract all 9 blocks in the format expected by numpy_tenpi_ccd
    # Index notation: o = occupied (hole), v = virtual (particle)
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
    
    # Initialize amplitudes
    # T1 is not used in CCD but included for compatibility
    t1 = np.zeros((nvir, nocc), dtype=np.float64)
    
    # Initialize T2 from MP2-like guess
    t2 = np.zeros((nvir, nvir, nocc, nocc), dtype=np.float64)
    
    # Compute orbital energy denominators
    eps_o = np.diag(F_oo)  # Occupied orbital energies
    eps_v = np.diag(F_vv)  # Virtual orbital energies
    
    # Initialize T2 with MP2 guess: t_abij = <ab||ij> / (eps_i + eps_j - eps_a - eps_b)
    for a in range(nvir):
        for b in range(nvir):
            for i in range(nocc):
                for j in range(nocc):
                    denom = eps_o[i] + eps_o[j] - eps_v[a] - eps_v[b]
                    if abs(denom) > 1e-12:
                        t2[a, b, i, j] = V_vvoo[a, b, i, j] / denom
    
    T = [t1, t2]
    
    # DIIS storage
    diis_t2 = []
    diis_res = []
    
    print(f"\n{'Iter':>4} {'E_corr':>15} {'Delta_E':>12} {'|R|_max':>12} {'|R|_rms':>12}")
    print("-" * 62)
    
    e_corr_old = 0.0
    
    for iteration in range(max_iter):
        # Compute residuals and energy using tenpi function
        result = numpy_tenpi_ccd(nocc, nvir, F, V, T, type_=np.float64)
        e_corr = result[0]
        r2 = result[1]  # Residual for T2
        
        delta_e = abs(e_corr - e_corr_old)
        r_max = np.max(np.abs(r2))
        r_rms = np.sqrt(np.mean(r2**2))
        
        print(f"{iteration:4d} {e_corr:15.8f} {delta_e:12.4e} {r_max:12.4e} {r_rms:12.4e}")
        
        # Check convergence
        if delta_e < tol and r_rms < tol:
            print("\nConverged!")
            break
        
        # Update T2 amplitudes: t2_new = t2_old + r2 / D
        t2_new = t2.copy()
        for a in range(nvir):
            for b in range(nvir):
                for i in range(nocc):
                    for j in range(nocc):
                        denom = eps_o[i] + eps_o[j] - eps_v[a] - eps_v[b]
                        if abs(denom) > 1e-12:
                            t2_new[a, b, i, j] = t2[a, b, i, j] + r2[a, b, i, j] / denom
        
        # Apply damping
        if damping > 0:
            t2_new = damping * t2 + (1.0 - damping) * t2_new
        
        # DIIS acceleration
        if iteration >= diis_start:
            diis_t2.append(t2_new.copy())
            diis_res.append(r2.copy())
            
            # Keep only the most recent diis_size vectors
            if len(diis_t2) > diis_size:
                diis_t2.pop(0)
                diis_res.pop(0)
            
            # Perform DIIS extrapolation
            if len(diis_t2) >= 2:
                t2_new = diis_extrapolate(diis_t2, diis_res)
        
        t2 = t2_new
        T = [t1, t2]
        e_corr_old = e_corr
    
    else:
        print("\nWarning: Maximum iterations reached without convergence!")
    
    return e_corr, t2


def diis_extrapolate(t_vectors, r_vectors):
    """
    DIIS extrapolation to accelerate convergence.
    
    Parameters:
    -----------
    t_vectors : list of ndarrays
        List of amplitude vectors from recent iterations
    r_vectors : list of ndarrays
        List of residual vectors from recent iterations
        
    Returns:
    --------
    t_new : ndarray
        Extrapolated amplitude vector
    """
    n = len(t_vectors)
    
    # Build B matrix: B_ij = <r_i | r_j>
    B = np.zeros((n + 1, n + 1))
    for i in range(n):
        for j in range(n):
            B[i, j] = np.sum(r_vectors[i] * r_vectors[j])
    
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
        # If singular, just return the most recent vector
        print("    DIIS: Singular B matrix, using most recent vector")
        return t_vectors[-1].copy()
    
    # Extrapolate
    t_new = np.zeros_like(t_vectors[0])
    for i in range(n):
        t_new += coeffs[i] * t_vectors[i]
    
    return t_new


if __name__ == "__main__":
    # Test with main.py workflow
    print("Test CCD tenpi solver")
    print("Run this via main.py or import and use with NormalOrderedHamiltonian")
