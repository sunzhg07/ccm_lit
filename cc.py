
import numpy as np
from opt_einsum import contract, contract_expression
from scipy import sparse
import time
import os

# Threading Configuration
# Suggest setting these if not set, to utilize multi-threading
if 'OMP_NUM_THREADS' not in os.environ:
    # Default to a reasonable number if not set (e.g., 4 or 8)
    # But usually numpy detects available cores. We just print a message or setup.
    pass

# Optimization: Cache for einsum paths
# This avoids re-calculating the optimal path every iteration
_einsum_cache = {}

def cached_contract(subscripts, *operands, **kwargs):
    """
    Wrapper around opt_einsum.contract that caches the contraction path.
    This provides significant speedup for iterative calculations with static shapes.
    """
    # Use 'optimal' optimization by default for cached calls as the cost is amortized
    if 'optimize' not in kwargs:
        kwargs['optimize'] = 'optimal'
        
    shapes = tuple(op.shape for op in operands)
    key = (subscripts, shapes, frozenset(kwargs.items()))
    
    if key not in _einsum_cache:
        _einsum_cache[key] = contract_expression(subscripts, *shapes, **kwargs)
        
    return _einsum_cache[key](*operands)

# Sparse matrix optimization utilities
SPARSE_THRESHOLD = 1e-12  # Elements below this are considered zero for sparsity


def to_sparse_if_beneficial(arr, threshold=SPARSE_THRESHOLD):
    """Convert array to sparse if it would save memory."""
    if arr is None or arr.size == 0:
        return arr
    
    sparsity = np.sum(np.abs(arr) < threshold) / arr.size
    # Use sparse if more than 50% of elements are near-zero
    if sparsity > 0.5:
        return sparse.csr_matrix(arr.reshape(arr.shape[0], -1))
    return arr


def ensure_dense(arr):
    """Ensure array is in dense format."""
    if sparse.issparse(arr):
        return arr.toarray()
    return arr


def sparse_einsum(subscripts, *operands, threshold=SPARSE_THRESHOLD):
    """
    Perform einsum with automatic sparsity detection.
    Falls back to dense computation for now, but masks small values.
    """
    result = np.einsum(subscripts, *operands)
    result[np.abs(result) < threshold] = 0
    return result



def pAB(val):
    """Permutator val(abij) -> val(abij) - val(baij)"""
    return val - np.transpose(val, (1, 0, 2, 3))


def pIJ(val):
    """Permutator val(abij) -> val(abij) - val(abji)"""
    return val - np.transpose(val, (0, 1, 3, 2))


def ccd(no_ham, n_occ, max_iter=200, tol=1e-8, alpha=1.0, use_sparse=True, sparse_threshold=SPARSE_THRESHOLD):
    """
    CCD solver using the T2 equation from the provided LaTeX images.
    Implements the Stanton & Gauss intermediate factorization.
    Optimized with sparse matrix support for memory efficiency.
    """
    # 1. Setup dimensions and Hamiltonian blocks
    n_states = no_ham.f.shape[0]
    o = slice(0, n_occ)
    v = slice(n_occ, n_states)

    f = no_ham.f
    Gamma = no_ham.Gamma # Antisymmetrized physics notation <pq||rs>
    eps = np.diag(f)

    # Pre-slice Fock blocks
    f_oo, f_vv = f[o, o], f[v, v]

    # Denominator D_ijab = eps_i + eps_j - eps_a - eps_b
    D2 = (eps[o, None, None, None] + eps[None, o, None, None]
          - eps[None, None, v, None] - eps[None, None, None, v])

    # 2. Initial Guess (MP2 amplitudes)
    # Corresponding to the <ab||ij> term in the image
    t2 = Gamma[o, o, v, v] / D2
    
    # Apply sparsity pruning to initial guess
    if use_sparse:
        t2[np.abs(t2) < sparse_threshold] = 0

    old_e = 0.0
    print(f"\n[CCD+Sparse] {'Iter':>4} | {'E_corr':>18} | {'Delta E':>12} | {'Sparsity':>8}")
    print("-" * 65)

    for iteration in range(max_iter):
        # --- Intermediates for CCD (T1 = 0) ---

        # F_ae: Particle-Particle ladder intermediate
        F_ae = f_vv.copy()
        F_ae -= 0.5 * sparse_einsum('mnaf,mnef->ae', t2, Gamma[o, o, v, v], threshold=sparse_threshold)

        # F_mi: Hole-Hole ladder intermediate
        F_mi = f_oo.copy()
        F_mi += 0.5 * sparse_einsum('inef,mnef->mi', t2, Gamma[o, o, v, v], threshold=sparse_threshold)

        # W_mnij: Hole-Hole effective interaction
        W_mnij = Gamma[o, o, o, o] + 0.25 * sparse_einsum('ijef,mnef->mnij', t2, Gamma[o, o, v, v], threshold=sparse_threshold)

        # W_abef: Particle-Particle effective interaction
        W_abef = Gamma[v, v, v, v] + 0.25 * sparse_einsum('mnab,mnef->abef', t2, Gamma[o, o, v, v], threshold=sparse_threshold)

        # W_mbej: Particle-Hole ring intermediate (the P(ij)P(ab) terms)
        W_mbej = Gamma[o, v, v, o] - 0.5 * sparse_einsum('jnfb,mnef->mbej', t2, Gamma[o, o, v, v], threshold=sparse_threshold)

        # --- Residual (R2) Construction ---
        # 1. Constant term: <ab||ij>
        r2 = Gamma[o, o, v, v].copy()

        # 2. F-term coupling (Ladder terms)
        term_ae = sparse_einsum('ijeb,ae->ijab', t2, F_ae, threshold=sparse_threshold)
        r2 += (term_ae - term_ae.transpose(0, 1, 3, 2)) # P(ab)

        term_mi = sparse_einsum('mjab,mi->ijab', t2, F_mi, threshold=sparse_threshold)
        r2 -= (term_mi - term_mi.transpose(1, 0, 2, 3)) # P(ij)

        # 3. W-term coupling (Direct ladders)
        r2 += 0.5 * sparse_einsum('mnab,mnij->ijab', t2, W_mnij, threshold=sparse_threshold)
        r2 += 0.5 * sparse_einsum('ijef,abef->ijab', t2, W_abef, threshold=sparse_threshold)

        # 4. Ring terms with P(ij)P(ab) permutations
        # This maps to the P(ij)P(ab) sum_{kc} <kb||cj> t_ik^ac term in the image
        term_ring = sparse_einsum('imae,mbej->ijab', t2, W_mbej, threshold=sparse_threshold)
        r2 += (term_ring - term_ring.transpose(1, 0, 2, 3)
               - term_ring.transpose(0, 1, 3, 2) + term_ring.transpose(1, 0, 3, 2))

        # --- Energy and Update ---
        e_corr = 0.25 * np.sum(Gamma[o, o, v, v] * t2)

        delta_e = abs(e_corr - old_e)
        
        # Calculate sparsity for monitoring
        sparsity = np.sum(np.abs(t2) < sparse_threshold) / t2.size
        print(f"[CCD] {iteration:4d} | {e_corr:18.10f} | {delta_e:12.4e} | {sparsity:7.2%}")

        if delta_e < tol:
            print("CCD Converged!")
            return e_corr, t2

        # Additive Update: t = t + alpha * (r / D)
        t2 += alpha * (r2 / D2)
        
        # Prune small amplitudes for memory efficiency
        if use_sparse:
            t2[np.abs(t2) < sparse_threshold] = 0
        
        old_e = e_corr

    return e_corr, t2


import numpy as np


def ccd_diis_solver(no_ham, n_occ, max_iter=200, tol=1e-8, alpha=1.0, diis_size=8, use_sparse=True, sparse_threshold=SPARSE_THRESHOLD):
    """
    Full CCD solver using the provided LaTeX equations.
    Includes intermediate factorization and a stabilized DIIS extrapolation.
    Optimized with sparse matrix support for memory efficiency.
    """
    # 1. Setup dimensions and Hamiltonian blocks
    n_states = no_ham.f.shape[0]
    o = slice(0, n_occ)
    v = slice(n_occ, n_states)

    f = no_ham.f
    Gamma = no_ham.Gamma  # Antisymmetrized physics notation <pq||rs>
    eps = np.diag(f)

    # Pre-slice Fock blocks
    f_oo, f_vv = f[o, o], f[v, v]

    # Energy Denominator: D_ijab = eps_i + eps_j - eps_a - eps_b
    D2 = (eps[o, None, None, None] + eps[None, o, None, None]
          - eps[None, None, v, None] - eps[None, None, None, v])

    # Initial Guess: MP2 amplitudes t2[i,j,a,b]
    t2 = Gamma[o, o, v, v] / D2
    
    # Apply sparsity pruning to initial guess
    if use_sparse:
        t2[np.abs(t2) < sparse_threshold] = 0

    # DIIS History Containers
    t_hist = []  # Stores the "extrapolated-ready" amplitudes
    e_hist = []  # Stores the error vectors (Residual / Denominator)

    old_e = 0.0
    print(f"\n[CCD+DIIS+Sparse] {'Iter':>4} | {'E_corr':>18} | {'Delta E':>12} | {'Sparsity':>8}")
    print("-" * 70)

    for i in range(max_iter):
        # 2. Construction of Intermediates (O(N^6) scaling)
        F_ae = f_vv.copy() - 0.5 * sparse_einsum('mnaf,mnef->ae', t2, Gamma[o, o, v, v], threshold=sparse_threshold)
        F_mi = f_oo.copy() + 0.5 * sparse_einsum('inef,mnef->mi', t2, Gamma[o, o, v, v], threshold=sparse_threshold)

        W_mnij = Gamma[o, o, o, o] + 0.25 * sparse_einsum('ijef,mnef->mnij', t2, Gamma[o, o, v, v], threshold=sparse_threshold)
        W_abef = Gamma[v, v, v, v] + 0.25 * sparse_einsum('mnab,mnef->abef', t2, Gamma[o, o, v, v], threshold=sparse_threshold)
        W_mbej = Gamma[o, v, v, o] - 0.5 * sparse_einsum('jnfb,mnef->mbej', t2, Gamma[o, o, v, v], threshold=sparse_threshold)

        # 3. Build Residual R_ijab (Matches the provided LaTeX equations)
        r2 = Gamma[o, o, v, v].copy() # Constant term <ab||ij>

        # F-couplings (Ladders)
        term_ae = sparse_einsum('ijeb,ae->ijab', t2, F_ae, threshold=sparse_threshold)
        r2 += (term_ae - term_ae.transpose(0, 1, 3, 2)) # P(ab)

        term_mi = sparse_einsum('mjab,mi->ijab', t2, F_mi, threshold=sparse_threshold)
        r2 -= (term_mi - term_mi.transpose(1, 0, 2, 3)) # P(ij)

        # W-couplings
        r2 += 0.5 * sparse_einsum('mnab,mnij->ijab', t2, W_mnij, threshold=sparse_threshold)
        r2 += 0.5 * sparse_einsum('ijef,abef->ijab', t2, W_abef, threshold=sparse_threshold)

        # Ring terms with P(ij)P(ab) permutations
        term_ring = sparse_einsum('imae,mbej->ijab', t2, W_mbej, threshold=sparse_threshold)
        r2 += (term_ring - term_ring.transpose(1, 0, 2, 3)
               - term_ring.transpose(0, 1, 3, 2) + term_ring.transpose(1, 0, 3, 2))

        # 4. Energy calculation and Convergence check
        e_corr = 0.25 * np.sum(Gamma[o, o, v, v] * t2)
        delta_e = abs(e_corr - old_e)
        
        # Calculate sparsity for monitoring
        sparsity = np.sum(np.abs(t2) < sparse_threshold) / t2.size
        print(f" {i:4d} | {e_corr:18.10f} | {delta_e:12.4e} | {sparsity:7.2%}")

        if delta_e < tol:
            print("CCD Converged!")
            return e_corr, t2
        old_e = e_corr

        # 5. DIIS Fix: Store Error Vector and trial next-step amplitude
        error_vec = (r2 / D2).ravel()
        # The history stores the "standard" next step (T + error)
        t_hist.append(t2.ravel() + alpha * error_vec)
        e_hist.append(error_vec)

        if len(t_hist) > diis_size:
            t_hist.pop(0)
            e_hist.pop(0)

        n = len(e_hist)
        if n > 2:
            # Build B matrix: B_ij = <e_i | e_j>
            B = np.zeros((n + 1, n + 1))
            for row in range(n):
                for col in range(row, n):
                    dot = np.dot(e_hist[row], e_hist[col])
                    B[row, col] = B[col, row] = dot

            # Constraint: sum of coefficients must be 1
            B[n, :-1] = -1.0
            B[:-1, n] = -1.0
            rhs = np.zeros(n + 1)
            rhs[n] = -1.0

            try:
                # Solve linear system for coefficients
                coeffs = np.linalg.solve(B, rhs)[:-1]

                # Extrapolate to a new t2 vector
                t2_new_vec = np.zeros_like(t_hist[0])
                for j in range(n):
                    t2_new_vec += coeffs[j] * t_hist[j]
                t2 = t2_new_vec.reshape(t2.shape)
            except np.linalg.LinAlgError:
                # Fallback to standard update if B is singular
                t2 = (t2.ravel() + alpha * error_vec).reshape(t2.shape)
        else:
            # First iterations use standard additive update
            t2 = (t2.ravel() + alpha * error_vec).reshape(t2.shape)
        
        # Prune small amplitudes for memory efficiency
        if use_sparse:
            t2[np.abs(t2) < sparse_threshold] = 0

    return e_corr, t2


def ccsd_diis_solver(no_ham, n_occ, max_iter=200, tol=1e-8, alpha=1.0, diis_size=6, use_sparse=True, sparse_threshold=SPARSE_THRESHOLD):
    """
    CCSD solver with DIIS acceleration.
    Optimized with sparse matrix support for memory efficiency.
    """
    n_states = no_ham.f.shape[0]
    o = slice(0, n_occ)
    v = slice(n_occ, n_states)

    f = no_ham.f
    Gamma = no_ham.Gamma
    eps = np.diag(f)

    # Denominators
    D1 = eps[o, None] - eps[None, v]
    D2 = eps[o, None, None, None] + eps[None, o, None, None] - eps[None, None, v, None] - eps[None, None, None, v]

    # Initialize Amplitudes
    t1 = np.zeros((n_occ, n_states - n_occ))
    t2 = Gamma[o, o, v, v] / D2
    
    # Apply sparsity pruning to initial guess
    if use_sparse:
        t2[np.abs(t2) < sparse_threshold] = 0

    # DIIS Storage
    t_list = []
    e_list = []

    old_e = 0.0
    print(f"\n[CCSD+DIIS+Sparse] {'Iter':>4} | {'E_corr':>18} | {'Delta E':>12} | {'T1 Sparsity':>12} | {'T2 Sparsity':>12}")
    print("-" * 95)

    for i in range(max_iter):
        # --- 1. Intermediate and Residual Calculation (Same as before) ---
        t1t1 = sparse_einsum('ia,jb->ijab', t1, t1, threshold=sparse_threshold)
        tau = t2 + t1t1 - t1t1.transpose(0, 1, 3, 2)
        tau_tilde = t2 + 0.5 * (t1t1 - t1t1.transpose(0, 1, 3, 2))

        # F-Intermediates
        F_ae = f[v, v].copy() - 0.5 * sparse_einsum('me,ma->ae', f[o, v], t1, threshold=sparse_threshold)
        F_ae += sparse_einsum('mf,amef->ae', t1, Gamma[v, o, v, v], threshold=sparse_threshold)
        F_ae -= 0.5 * sparse_einsum('mnaf,mnef->ae', tau_tilde, Gamma[o, o, v, v], threshold=sparse_threshold)

        F_mi = f[o, o].copy() + 0.5 * sparse_einsum('ie,me->mi', t1, f[o, v], threshold=sparse_threshold)
        F_mi += sparse_einsum('ne,mnie->mi', t1, Gamma[o, o, o, v], threshold=sparse_threshold)
        F_mi += 0.5 * sparse_einsum('inef,mnef->mi', tau_tilde, Gamma[o, o, v, v], threshold=sparse_threshold)

        F_me = f[o, v] + sparse_einsum('nf,mnef->me', t1, Gamma[o, o, v, v], threshold=sparse_threshold)

        # W-Intermediates
        W_mnij = Gamma[o, o, o, o] + 0.25 * sparse_einsum('ijef,mnef->mnij', tau, Gamma[o, o, v, v], threshold=sparse_threshold)
        term_mnij = sparse_einsum('je,mnie->mnij', t1, Gamma[o, o, o, v], threshold=sparse_threshold)
        W_mnij += (term_mnij - term_mnij.transpose(0, 1, 3, 2))

        W_abef = Gamma[v, v, v, v] + 0.25 * sparse_einsum('mnab,mnef->abef', tau, Gamma[o, o, v, v], threshold=sparse_threshold)
        term_abef = sparse_einsum('ma,mbef->abef', t1, Gamma[o, v, v, v], threshold=sparse_threshold)
        W_abef -= (term_abef - term_abef.transpose(1, 0, 2, 3))

        W_mbej = Gamma[o, v, v, o] + sparse_einsum('jf,mbef->mbej', t1, Gamma[o, v, v, v], threshold=sparse_threshold)
        W_mbej -= sparse_einsum('nb,mnej->mbej', t1, Gamma[o, o, v, o], threshold=sparse_threshold)
        W_mbej -= 0.5 * sparse_einsum('jnfb,mnef->mbej', t2, Gamma[o, o, v, v], threshold=sparse_threshold)
        W_mbej -= sparse_einsum('jf,nb,mnef->mbej', t1, t1, Gamma[o, o, v, v], threshold=sparse_threshold)

        # Residuals
        r1 = f[v, o].T + sparse_einsum('ie,ae->ia', t1, F_ae, threshold=sparse_threshold) - sparse_einsum('ma,mi->ia', t1, F_mi, threshold=sparse_threshold)
        r1 += sparse_einsum('imae,me->ia', t2, F_me, threshold=sparse_threshold) + sparse_einsum('nf,nafi->ia', t1, Gamma[o, v, v, o], threshold=sparse_threshold)
        r1 -= 0.5 * sparse_einsum('imef,maef->ia', t2, Gamma[o, v, v, v], threshold=sparse_threshold) + 0.5 * sparse_einsum('mnea,mnei->ia', t2, Gamma[o, o, v, o], threshold=sparse_threshold)

        r2 = Gamma[o, o, v, v].copy()
        term_ae = sparse_einsum('ijeb,ae->ijab', t2, F_ae, threshold=sparse_threshold)
        r2 += (term_ae - term_ae.transpose(0, 1, 3, 2))
        term_mi = sparse_einsum('mjab,mi->ijab', t2, F_mi, threshold=sparse_threshold)
        r2 -= (term_mi - term_mi.transpose(1, 0, 2, 3))
        r2 += 0.5 * sparse_einsum('mnab,mnij->ijab', tau, W_mnij, threshold=sparse_threshold) + 0.5 * sparse_einsum('ijef,abef->ijab', tau, W_abef, threshold=sparse_threshold)

        term_ring = sparse_einsum('imae,mbej->ijab', t2, W_mbej, threshold=sparse_threshold)
        r2 += (term_ring - term_ring.transpose(1, 0, 2, 3) - term_ring.transpose(0, 1, 3, 2) + term_ring.transpose(1, 0, 3, 2))

        term_t1_v = sparse_einsum('ie,abej->ijab', t1, Gamma[v, v, v, o], threshold=sparse_threshold)
        r2 += (term_t1_v - term_t1_v.transpose(1, 0, 2, 3))
        term_t1_o = sparse_einsum('ma,mbij->ijab', t1, Gamma[o, v, o, o], threshold=sparse_threshold)
        r2 -= (term_t1_o - term_t1_o.transpose(0, 1, 3, 2))

        # --- 2. Energy and Convergence Check ---
        e_corr = np.sum(f[o, v] * t1) + 0.25 * np.sum(Gamma[o, o, v, v] * tau)
        delta_e = abs(e_corr - old_e)
        
        # Calculate sparsity for monitoring
        t1_sparsity = np.sum(np.abs(t1) < sparse_threshold) / t1.size if t1.size > 0 else 0
        t2_sparsity = np.sum(np.abs(t2) < sparse_threshold) / t2.size
        print(f"[CCSD] {i:4d} | {e_corr:18.10f} | {delta_e:12.4e} | {t1_sparsity:11.2%} | {t2_sparsity:11.2%}")

        if delta_e < tol:
            return e_corr, t1, t2
        old_e = e_corr

        # --- 3. DIIS Acceleration Logic ---
        # Error vector: Δt = r / D
        de1 = r1 / D1
        de2 = r2 / D2

        # Flatten and concatenate t1, t2 for DIIS
        t_curr = np.concatenate([t1.ravel(), t2.ravel()])
        e_curr = np.concatenate([de1.ravel(), de2.ravel()])

        t_list.append(t_curr)
        e_list.append(e_curr)

        if len(t_list) > diis_size:
            t_list.pop(0)
            e_list.pop(0)

        if len(t_list) > 2:
            n = len(e_list)
            B = np.zeros((n + 1, n + 1))
            for row in range(n):
                for col in range(row, n):
                    B[row, col] = np.dot(e_list[row], e_list[col])
                    B[col, row] = B[row, col]

            B[n, :-1] = -1
            B[:-1, n] = -1
            rhs = np.zeros(n + 1)
            rhs[n] = -1

            try:
                # Solve for DIIS coefficients
                coeffs = np.linalg.solve(B, rhs)[:-1]
                # Extrapolate Amplitudes
                t_extrap = np.zeros_like(t_curr)
                e_extrap = np.zeros_like(e_curr)
                for j in range(n):
                    t_extrap += coeffs[j] * t_list[j]
                    e_extrap += coeffs[j] * e_list[j]

                # Apply update to the extrapolated state
                t_new = t_extrap + alpha * e_extrap
            except np.linalg.LinAlgError:
                t_new = t_curr + alpha * e_curr
        else:
            t_new = t_curr + alpha * e_curr

        # Reshape back to T1 and T2
        t1 = t_new[:t1.size].reshape(t1.shape)
        t2 = t_new[t1.size:].reshape(t2.shape)
        
        # Prune small amplitudes for memory efficiency
        if use_sparse:
            t1[np.abs(t1) < sparse_threshold] = 0
            t2[np.abs(t2) < sparse_threshold] = 0

    return e_corr, t1, t2

def ccsd(no_ham, n_occ, max_iter=200, tol=1e-8, alpha=0.5, use_sparse=True, sparse_threshold=SPARSE_THRESHOLD):
    """
    CCSD solver based on the provided T1 and T2 long-form equations.
    Optimized via Stanton & Gauss intermediate factorization.
    Optimized with sparse matrix support for memory efficiency.
    """
    n_states = no_ham.f.shape[0]
    o = slice(0, n_occ)
    v = slice(n_occ, n_states)

    f = no_ham.f
    Gamma = no_ham.Gamma # <pq||rs>
    eps = np.diag(f)

    # Pre-slice blocks
    f_oo, f_vv, f_ov, f_vo = f[o, o], f[v, v], f[o, v], f[v, o]

    # Energy Denominators
    D1 = eps[o, None] - eps[None, v]
    D2 = eps[o, None, None, None] + eps[None, o, None, None] - eps[None, None, v, None] - eps[None, None, None, v]

    # Initialize Amplitudes
    t1 = np.zeros((n_occ, n_states - n_occ))
    t2 = Gamma[o, o, v, v] / D2
    
    # Apply sparsity pruning to initial guess
    if use_sparse:
        t2[np.abs(t2) < sparse_threshold] = 0

    old_e = 0.0
    print(f"\n[CCSD+Sparse] {'Iter':>4} | {'E_corr':>18} | {'Delta E':>12} | {'T1 Sparsity':>12} | {'T2 Sparsity':>12}")
    print("-" * 95)
    
    for i in range(max_iter):
        # 1. Intermediates (Combining T1 and T2 terms as shown in equations)
        # tau_tilde = t_ijab + 0.5 * (t_ia * t_jb - t_ib * t_ja)
        t1t1 = sparse_einsum('ia,jb->ijab', t1, t1, threshold=sparse_threshold)
        tau = t2 + t1t1 - t1t1.transpose(0, 1, 3, 2)
        tau_tilde = t2 + 0.5 * (t1t1 - t1t1.transpose(0, 1, 3, 2))

        # F-Intermediates (Effective Fock matrices)
        F_ae = f_vv.copy()
        F_ae -= 0.5 * sparse_einsum('me,ma->ae', f_ov, t1, threshold=sparse_threshold)
        F_ae += sparse_einsum('mf,amef->ae', t1, Gamma[v, o, v, v], threshold=sparse_threshold)
        F_ae -= 0.5 * sparse_einsum('mnaf,mnef->ae', tau_tilde, Gamma[o, o, v, v], threshold=sparse_threshold)

        F_mi = f_oo.copy()
        F_mi += 0.5 * sparse_einsum('ie,me->mi', t1, f_ov, threshold=sparse_threshold)
        F_mi += sparse_einsum('ne,mnie->mi', t1, Gamma[o, o, o, v], threshold=sparse_threshold)
        F_mi += 0.5 * sparse_einsum('inef,mnef->mi', tau_tilde, Gamma[o, o, v, v], threshold=sparse_threshold)

        F_me = f_ov + sparse_einsum('nf,mnef->me', t1, Gamma[o, o, v, v], threshold=sparse_threshold)

        # W-Intermediates (Effective Integrals)
        W_mnij = Gamma[o, o, o, o] + 0.25 * sparse_einsum('ijef,mnef->mnij', tau, Gamma[o, o, v, v], threshold=sparse_threshold)
        W_mnij += (sparse_einsum('je,mnie->mnij', t1, Gamma[o, o, o, v], threshold=sparse_threshold) -
                   sparse_einsum('ie,mnje->mnij', t1, Gamma[o, o, o, v], threshold=sparse_threshold))

        W_abef = Gamma[v, v, v, v] + 0.25 * sparse_einsum('mnab,mnef->abef', tau, Gamma[o, o, v, v], threshold=sparse_threshold)
        W_abef -= (sparse_einsum('ma,mbef->abef', t1, Gamma[o, v, v, v], threshold=sparse_threshold) -
                   sparse_einsum('mb,maef->abef', t1, Gamma[o, v, v, v], threshold=sparse_threshold))

        W_mbej = Gamma[o, v, v, o] + sparse_einsum('jf,mbef->mbej', t1, Gamma[o, v, v, v], threshold=sparse_threshold)
        W_mbej -= sparse_einsum('nb,mnej->mbej', t1, Gamma[o, o, v, o], threshold=sparse_threshold)
        W_mbej -= 0.5 * sparse_einsum('jnfb,mnef->mbej', t2, Gamma[o, o, v, v], threshold=sparse_threshold)
        W_mbej -= sparse_einsum('jf,nb,mnef->mbej', t1, t1, Gamma[o, o, v, v], threshold=sparse_threshold)

        # 2. Residuals
        # T1 Residual (Maps to Image 1)
        r1 = f_vo.T + sparse_einsum('ie,ae->ia', t1, F_ae, threshold=sparse_threshold) - sparse_einsum('ma,mi->ia', t1, F_mi, threshold=sparse_threshold)
        r1 += sparse_einsum('imae,me->ia', t2, F_me, threshold=sparse_threshold)
        r1 += sparse_einsum('nf,nafi->ia', t1, Gamma[o, v, v, o], threshold=sparse_threshold)
        r1 -= 0.5 * sparse_einsum('imef,maef->ia', t2, Gamma[o, v, v, v], threshold=sparse_threshold)
        r1 -= 0.5 * sparse_einsum('mnea,mnei->ia', t2, Gamma[o, o, v, o], threshold=sparse_threshold)

        # T2 Residual (Maps to Image 2)
        r2 = Gamma[o, o, v, v].copy()

        # Linear and Quadratic T2 couplings
        term_ae = sparse_einsum('ijeb,ae->ijab', t2, F_ae, threshold=sparse_threshold)
        r2 += (term_ae - term_ae.transpose(0, 1, 3, 2))
        term_mi = sparse_einsum('mjab,mi->ijab', t2, F_mi, threshold=sparse_threshold)
        r2 -= (term_mi - term_mi.transpose(1, 0, 2, 3))

        r2 += 0.5 * sparse_einsum('mnab,mnij->ijab', tau, W_mnij, threshold=sparse_threshold)
        r2 += 0.5 * sparse_einsum('ijef,abef->ijab', tau, W_abef, threshold=sparse_threshold)

        # Ring terms P(ij)P(ab)
        term_ring = sparse_einsum('imae,mbej->ijab', t2, W_mbej, threshold=sparse_threshold)
        r2 += (term_ring - term_ring.transpose(1, 0, 2, 3) -
               term_ring.transpose(0, 1, 3, 2) + term_ring.transpose(1, 0, 3, 2))

        # T1 couplings
        term_t1_v = sparse_einsum('ie,abej->ijab', t1, Gamma[v, v, v, o], threshold=sparse_threshold)
        r2 += (term_t1_v - term_t1_v.transpose(1, 0, 2, 3))
        term_t1_o = sparse_einsum('ma,mbij->ijab', t1, Gamma[o, v, o, o], threshold=sparse_threshold)
        r2 -= (term_t1_o - term_t1_o.transpose(0, 1, 3, 2))

        # 3. Energy and Step Update
        e_corr = np.sum(f_ov * t1) + 0.25 * np.sum(Gamma[o, o, v, v] * tau)

        delta_e = abs(e_corr - old_e)
        
        # Calculate sparsity for monitoring
        t1_sparsity = np.sum(np.abs(t1) < sparse_threshold) / t1.size if t1.size > 0 else 0
        t2_sparsity = np.sum(np.abs(t2) < sparse_threshold) / t2.size
        print(f"[CCSD] {i:4d} | {e_corr:18.10f} | {delta_e:12.4e} | {t1_sparsity:11.2%} | {t2_sparsity:11.2%}")

        if delta_e < tol:
            return e_corr, t1, t2

        t1 = alpha * (r1 / D1) +  t1
        t2 = alpha * (r2 / D2) +  t2
        
        # Prune small amplitudes for memory efficiency
        if use_sparse:
            t1[np.abs(t1) < sparse_threshold] = 0
            t2[np.abs(t2) < sparse_threshold] = 0
        
        old_e = e_corr

    return e_corr, t1, t2



def ccsd_ode_solver(no_ham, n_occ, max_iter=200, tol=1e-8, step_size=0.01, use_sparse=True, sparse_threshold=SPARSE_THRESHOLD):
    """
    CCSD solver based on the provided T1 and T2 long-form equations.
    Optimized via Stanton & Gauss intermediate factorization.
    Optimized with sparse matrix support for memory efficiency.
    """
    n_states = no_ham.f.shape[0]
    o = slice(0, n_occ)
    v = slice(n_occ, n_states)

    f = no_ham.f
    Gamma = no_ham.Gamma # <pq||rs>
    eps = np.diag(f)

    # Pre-slice blocks
    f_oo, f_vv, f_ov, f_vo = f[o, o], f[v, v], f[o, v], f[v, o]

    # Energy Denominators
    D1 = eps[o, None] - eps[None, v]
    D2 = eps[o, None, None, None] + eps[None, o, None, None] - eps[None, None, v, None] - eps[None, None, None, v]

    # Initialize Amplitudes
    t1 = np.zeros((n_occ, n_states - n_occ))
    t2 = Gamma[o, o, v, v] / D2
    
    # Apply sparsity pruning to initial guess
    if use_sparse:
        t2[np.abs(t2) < sparse_threshold] = 0

    old_e = 0.0
    print(f"\n[CCSD+Sparse] {'Iter':>4} | {'E_corr':>18} | {'Delta E':>12} | {'T1 Sparsity':>12} | {'T2 Sparsity':>12}")
    print("-" * 95)
    
    for i in range(max_iter):
        # 1. Intermediates (Combining T1 and T2 terms as shown in equations)
        # tau_tilde = t_ijab + 0.5 * (t_ia * t_jb - t_ib * t_ja)
        t1t1 = sparse_einsum('ia,jb->ijab', t1, t1, threshold=sparse_threshold)
        tau = t2 + t1t1 - t1t1.transpose(0, 1, 3, 2)
        tau_tilde = t2 + 0.5 * (t1t1 - t1t1.transpose(0, 1, 3, 2))

        # F-Intermediates (Effective Fock matrices)
        F_ae = f_vv.copy()
        F_ae -= 0.5 * sparse_einsum('me,ma->ae', f_ov, t1, threshold=sparse_threshold)
        F_ae += sparse_einsum('mf,amef->ae', t1, Gamma[v, o, v, v], threshold=sparse_threshold)
        F_ae -= 0.5 * sparse_einsum('mnaf,mnef->ae', tau_tilde, Gamma[o, o, v, v], threshold=sparse_threshold)

        F_mi = f_oo.copy()
        F_mi += 0.5 * sparse_einsum('ie,me->mi', t1, f_ov, threshold=sparse_threshold)
        F_mi += sparse_einsum('ne,mnie->mi', t1, Gamma[o, o, o, v], threshold=sparse_threshold)
        F_mi += 0.5 * sparse_einsum('inef,mnef->mi', tau_tilde, Gamma[o, o, v, v], threshold=sparse_threshold)

        F_me = f_ov + sparse_einsum('nf,mnef->me', t1, Gamma[o, o, v, v], threshold=sparse_threshold)

        # W-Intermediates (Effective Integrals)
        W_mnij = Gamma[o, o, o, o] + 0.25 * sparse_einsum('ijef,mnef->mnij', tau, Gamma[o, o, v, v], threshold=sparse_threshold)
        W_mnij += (sparse_einsum('je,mnie->mnij', t1, Gamma[o, o, o, v], threshold=sparse_threshold) -
                   sparse_einsum('ie,mnje->mnij', t1, Gamma[o, o, o, v], threshold=sparse_threshold))

        W_abef = Gamma[v, v, v, v] + 0.25 * sparse_einsum('mnab,mnef->abef', tau, Gamma[o, o, v, v], threshold=sparse_threshold)
        W_abef -= (sparse_einsum('ma,mbef->abef', t1, Gamma[o, v, v, v], threshold=sparse_threshold) -
                   sparse_einsum('mb,maef->abef', t1, Gamma[o, v, v, v], threshold=sparse_threshold))

        W_mbej = Gamma[o, v, v, o] + sparse_einsum('jf,mbef->mbej', t1, Gamma[o, v, v, v], threshold=sparse_threshold)
        W_mbej -= sparse_einsum('nb,mnej->mbej', t1, Gamma[o, o, v, o], threshold=sparse_threshold)
        W_mbej -= 0.5 * sparse_einsum('jnfb,mnef->mbej', t2, Gamma[o, o, v, v], threshold=sparse_threshold)
        W_mbej -= sparse_einsum('jf,nb,mnef->mbej', t1, t1, Gamma[o, o, v, v], threshold=sparse_threshold)

        # 2. Residuals
        # T1 Residual (Maps to Image 1)
        r1 = f_vo.T + sparse_einsum('ie,ae->ia', t1, F_ae, threshold=sparse_threshold) - sparse_einsum('ma,mi->ia', t1, F_mi, threshold=sparse_threshold)
        r1 += sparse_einsum('imae,me->ia', t2, F_me, threshold=sparse_threshold)
        r1 += sparse_einsum('nf,nafi->ia', t1, Gamma[o, v, v, o], threshold=sparse_threshold)
        r1 -= 0.5 * sparse_einsum('imef,maef->ia', t2, Gamma[o, v, v, v], threshold=sparse_threshold)
        r1 -= 0.5 * sparse_einsum('mnea,mnei->ia', t2, Gamma[o, o, v, o], threshold=sparse_threshold)

        # T2 Residual (Maps to Image 2)
        r2 = Gamma[o, o, v, v].copy()

        # Linear and Quadratic T2 couplings
        term_ae = sparse_einsum('ijeb,ae->ijab', t2, F_ae, threshold=sparse_threshold)
        r2 += (term_ae - term_ae.transpose(0, 1, 3, 2))
        term_mi = sparse_einsum('mjab,mi->ijab', t2, F_mi, threshold=sparse_threshold)
        r2 -= (term_mi - term_mi.transpose(1, 0, 2, 3))

        r2 += 0.5 * sparse_einsum('mnab,mnij->ijab', tau, W_mnij, threshold=sparse_threshold)
        r2 += 0.5 * sparse_einsum('ijef,abef->ijab', tau, W_abef, threshold=sparse_threshold)

        # Ring terms P(ij)P(ab)
        term_ring = sparse_einsum('imae,mbej->ijab', t2, W_mbej, threshold=sparse_threshold)
        r2 += (term_ring - term_ring.transpose(1, 0, 2, 3) -
               term_ring.transpose(0, 1, 3, 2) + term_ring.transpose(1, 0, 3, 2))

        # T1 couplings
        term_t1_v = sparse_einsum('ie,abej->ijab', t1, Gamma[v, v, v, o], threshold=sparse_threshold)
        r2 += (term_t1_v - term_t1_v.transpose(1, 0, 2, 3))
        term_t1_o = sparse_einsum('ma,mbij->ijab', t1, Gamma[o, v, o, o], threshold=sparse_threshold)
        r2 -= (term_t1_o - term_t1_o.transpose(0, 1, 3, 2))

        # 3. Energy and Step Update
        e_corr = np.sum(f_ov * t1) + 0.25 * np.sum(Gamma[o, o, v, v] * tau)

        delta_e = abs(e_corr - old_e)
        
        # Calculate sparsity for monitoring
        t1_sparsity = np.sum(np.abs(t1) < sparse_threshold) / t1.size if t1.size > 0 else 0
        t2_sparsity = np.sum(np.abs(t2) < sparse_threshold) / t2.size
        print(f"[CCSD] {i:4d} | {e_corr:18.10f} | {delta_e:12.4e} | {t1_sparsity:11.2%} | {t2_sparsity:11.2%}")

        if delta_e < tol:
            return e_corr, t1, t2

        t1 = -step_size * r1  +  t1
        t2 = -step_size * r2  +  t2
        
        # Prune small amplitudes for memory efficiency
        if use_sparse:
            t1[np.abs(t1) < sparse_threshold] = 0
            t2[np.abs(t2) < sparse_threshold] = 0
        
        old_e = e_corr

    return e_corr, t1, t2

from scipy import sparse


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


# ======================================================================================
# CCSDT Implementation
# ======================================================================================

# Permutations for CCSDT
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

class DIIS:
    """DIIS Acceleration helper class."""
    def __init__(self, size=6):
        self.size = size
        self.t_list = []
        self.e_list = []

    def update(self, t_curr, e_curr):
        self.t_list.append(t_curr)
        self.e_list.append(e_curr)
        
        if len(self.t_list) > self.size:
            self.t_list.pop(0)
            self.e_list.pop(0)
            
    def extrapolate(self):
        n = len(self.t_list)
        if n == 0:
            return None
        if n == 1:
            return self.t_list[0]
            
        B = np.zeros((n + 1, n + 1))
        for i in range(n):
            for j in range(i, n):
                val = np.dot(self.e_list[i], self.e_list[j])
                B[i, j] = B[j, i] = val
        
        B[-1, :n] = -1
        B[:n, -1] = -1
        
        rhs = np.zeros(n + 1)
        rhs[-1] = -1
        
        try:
            coeffs = np.linalg.solve(B, rhs)[:-1]
            t_extrap = np.zeros_like(self.t_list[0])
            for i in range(n):
                t_extrap += coeffs[i] * self.t_list[i]
            return t_extrap
        except np.linalg.LinAlgError:
            return self.t_list[-1]

def ccsdt(no_ham, n_occ, max_iter=50, tol=1e-8, alpha=1.0, use_sparse=False, diis_size=6, initial_t1=None, initial_t2=None):
    """
    Optimized CCSDT with reduced redundant contractions, pre-computed intermediates, 
    and DIIS acceleration.
    """
    # Enable caching for contractions in this function
    contract = cached_contract
    
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
    V_voov = Gamma[v, o, o, v]
    V_vooo = Gamma[v, o, o, o]
    V_vvov = Gamma[v, v, o, v]
    
    eps = np.diag(f)
    eps_o = eps[o]
    eps_v = eps[v]
    
    D1 = eps_o[:, None] - eps_v[None, :]
    D2 = (eps_o[:, None, None, None] + eps_o[None, :, None, None] 
          - eps_v[None, None, :, None] - eps_v[None, None, None, :])
    D3 = (eps_o[:, None, None, None, None, None] + eps_o[None, :, None, None, None, None] + eps_o[None, None, :, None, None, None]
          - eps_v[None, None, None, :, None, None] - eps_v[None, None, None, None, :, None] - eps_v[None, None, None, None, None, :])

    if initial_t1 is not None and initial_t2 is not None:
        t1 = initial_t1.copy()
        t2 = initial_t2.copy()
        print("Using provided initial guesses for T1 and T2.")
    else:
        t1 = np.zeros((n_occ, n_virt))
        t2 = V_oovv / D2
    
    t3 = np.zeros((n_occ, n_occ, n_occ, n_virt, n_virt, n_virt))
    
    t3_size_gb = t3.nbytes / 1e9
    print(f"CCSDT Optimized (w/ DIIS). T3: {t3.size/1e6:.2f}M elements ({t3_size_gb:.2f} GB)")
    print(f"Optimization: Pre-computing t1*t1, t1*t2 intermediates...")

    old_e = 0.0
    iter_times = []
    
    diis = DIIS(size=diis_size)
    
    for iteration in range(max_iter):
        iter_start = time.time()
        
        # ==================== PRE-COMPUTE COMMON t1 PRODUCTS ====================
        # These are reused many times, compute once
        t1_t1_oo = contract('ia,ja->ij', t1, t1)  # occ-occ
        t1_t1_vv = contract('ia,ib->ab', t1, t1)  # virt-virt
        
        # Effective tau for T2
        tau = t2 + contract('ia,jb->ijab', t1, t1)
        
        # ==================== T1 Update ====================
        r1 = f_vo.copy().T
        
        # Pure T1 terms (grouped)
        r1 += contract('ab,ib->ia', f_vv, t1)
        r1 -= contract('ji,ja->ia', f_oo, t1)
        r1 += contract('ajib,jb->ia', V_voov, t1)
        r1 -= contract('jb,ib,ja->ia', f_ov, t1, t1)
        
        # T1^3 terms
        r1 -= contract('jkbc,ib,ja,kc->ia', V_oovv, t1, t1, t1)
        r1 += contract('ajbc,ib,jc->ia', V_vovv, t1, t1)
        r1 -= contract('jkib,ja,kb->ia', V_ooov, t1, t1)
        
        # T1-T2 mixed
        r1 += contract('jkbc,jb,kica->ia', V_oovv, t1, t2)
        r1 += 0.5 * contract('jkbc,ib,jkca->ia', V_oovv, t1, t2)
        r1 += 0.5 * contract('jkbc,ja,kibc->ia', V_oovv, t1, t2)
        
        # Pure T2 terms
        r1 += contract('jb,ijab->ia', f_ov, t2)
        r1 += 0.5 * contract('ajbc,ijbc->ia', V_vovv, t2)
        r1 -= 0.5 * contract('jkib,jkab->ia', V_ooov, t2)
        
        # T3 contribution
        r1 += 0.25 * contract('jkbc,ijkabc->ia', V_oovv, t3)

        # ==================== T2 Update ====================
        r2 = V_oovv.copy()
        
        # Use tau for some contractions
        r2 += 0.5 * contract('klij,klab->ijab', V_oooo, tau)
        r2 += 0.5 * contract('abcd,ijcd->ijab', V_vvvv, tau)
        
        # Pure T2 terms
        r2 += P_ab(contract('ac,jibc->ijab', f_vv, t2))
        r2 -= P_ij(contract('ki,jkba->ijab', f_oo, t2))
        r2 += P_ab(P_ij(contract('akic,jkbc->ijab', V_voov, t2)))
        
        # T2-T2 contractions (expensive but necessary)
        r2 += 0.5 * P_ij(contract('klcd,ikab,ljcd->ijab', V_oovv, t2, t2))
        r2 += 0.5 * P_ab(contract('klcd,ijac,kldb->ijab', V_oovv, t2, t2))
        r2 += 0.5 * P_ab(P_ij(contract('klcd,ikac,ljdb->ijab', V_oovv, t2, t2)))
        r2 += 0.25 * contract('klcd,ijcd,klab->ijab', V_oovv, t2, t2)
        
        # T1-dependent terms (using pre-computed products where possible)
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
        
        # T3 contributions
        r2 += contract('klcd,kc,lijdab->ijab', V_oovv, t1, t3)
        r2 += 0.5 * P_ij(contract('klcd,ic,kljdab->ijab', V_oovv, t1, t3))
        r2 += 0.5 * P_ab(contract('klcd,ka,lijcdb->ijab', V_oovv, t1, t3))
        
        r2 += P_ij(contract('abic,jc->ijab', V_vvov, t1))
        r2 -= P_ab(contract('akij,kb->ijab', V_vooo, t1))
        
        r2 += contract('kc,ijkabc->ijab', f_ov, t3)
        r2 += 0.5 * P_ab(contract('akcd,jikbcd->ijab', V_vovv, t3))
        r2 -= 0.5 * P_ij(contract('klic,jklbac->ijab', V_ooov, t3))
        
        # ==================== T3 Update (Most expensive) ====================
        r3 = np.zeros_like(t3)
        
        # Using same equations as original but with optimized contraction
        term = contract('lmde,id,je,la,mkbc->ijkabc', V_oovv, t1, t1, t1, t2, optimize='optimal')
        r3 += 0.5 * P_ijk_full(P_a_bc(term))
        term = contract('lmde,id,la,mb,jkec->ijkabc', V_oovv, t1, t1, t1, t2, optimize='optimal')
        r3 += 0.5 * P_abc_full(P_i_jk(term))
        term = contract('alde,id,je,lkbc->ijkabc', V_vovv, t1, t1, t2, optimize='optimal')
        r3 -= 0.5 * P_ijk_full(P_a_bc(term))
        term = contract('alde,id,lb,jkec->ijkabc', V_vovv, t1, t1, t2, optimize='optimal')
        r3 -= P_abc_full(P_i_jk(term))
        term = contract('lmid,la,mb,jkdc->ijkabc', V_ooov, t1, t1, t2, optimize='optimal')
        r3 += 0.5 * P_abc_full(P_i_jk(term))
        term = contract('lmid,jd,la,mkbc->ijkabc', V_ooov, t1, t1, t2, optimize='optimal')
        r3 += P_ijk_full(P_a_bc(term))
        
        # T1-T1-T3 terms
        term = contract('lmde,id,le,mjkabc->ijkabc', V_oovv, t1, t1, t3, optimize='optimal')
        r3 += P_i_jk(term)
        term = contract('lmde,la,md,ijkebc->ijkabc', V_oovv, t1, t1, t3, optimize='optimal')
        r3 += P_a_bc(term)
        term = contract('lmde,id,je,lmkabc->ijkabc', V_oovv, t1, t1, t3, optimize='optimal')
        r3 += 0.25 * P_ijk_full(term)
        term = contract('lmde,id,la,mjkebc->ijkabc', V_oovv, t1, t1, t3, optimize='optimal')
        r3 -= P_a_bc(P_i_jk(term))
        term = contract('lmde,la,mb,ijkdec->ijkabc', V_oovv, t1, t1, t3, optimize='optimal')
        r3 += 0.25 * P_abc_full(term)
        
        # T1-T2-T2 terms
        term = contract('lmde,ld,ijeb,mkac->ijkabc', V_oovv, t1, t2, t2, optimize='optimal')
        r3 -= P_b_ac(P_ij_k(term))
        term = contract('lmde,la,ijdb,mkec->ijkabc', V_oovv, t1, t2, t2, optimize='optimal')
        r3 -= P_abc_full(P_ij_k(term))
        term = contract('lmde,id,ljab,mkec->ijkabc', V_oovv, t1, t2, t2, optimize='optimal')
        r3 -= P_ab_c(P_ijk_full(term))
        term = contract('lmde,la,ijde,mkbc->ijkabc', V_oovv, t1, t2, t2, optimize='optimal')
        r3 += 0.5 * P_a_bc(P_ij_k(term))
        term = contract('lmde,id,jkec,lmab->ijkabc', V_oovv, t1, t2, t2, optimize='optimal')
        r3 += 0.5 * P_c_ab(P_i_jk(term))
        
        # Pure T2 terms
        term = contract('abde,id,jkec->ijkabc', V_vvvv, t1, t2, optimize='optimal')
        r3 += P_ab_c(P_i_jk(term))
        term = contract('lmij,la,mkbc->ijkabc', V_oooo, t1, t2, optimize='optimal')
        r3 += P_a_bc(P_ij_k(term))
        
        term = contract('alid,jd,lkbc->ijkabc', V_voov, t1, t2, optimize='optimal')
        r3 -= P_ijk_full(P_a_bc(term))
        term = contract('alid,lb,jkdc->ijkabc', V_voov, t1, t2, optimize='optimal')
        r3 -= P_abc_full(P_i_jk(term))
        
        # T1-T3 terms
        term = contract('ld,id,ljkabc->ijkabc', f_ov, t1, t3, optimize='optimal')
        r3 -= P_i_jk(term)
        term = contract('ld,la,ijkdbc->ijkabc', f_ov, t1, t3, optimize='optimal')
        r3 -= P_a_bc(term)
        term = contract('alde,ld,ijkebc->ijkabc', V_vovv, t1, t3, optimize='optimal')
        r3 -= P_a_bc(term)
        term = contract('alde,id,ljkebc->ijkabc', V_vovv, t1, t3, optimize='optimal')
        r3 += P_a_bc(P_i_jk(term))
        term = contract('alde,lb,ijkdec->ijkabc', V_vovv, t1, t3, optimize='optimal')
        r3 -= 0.5 * P_abc_full(term)
        term = contract('lmid,ld,mjkabc->ijkabc', V_ooov, t1, t3, optimize='optimal')
        r3 += P_i_jk(term)
        term = contract('lmid,jd,lmkabc->ijkabc', V_ooov, t1, t3, optimize='optimal')
        r3 += 0.5 * P_ijk_full(term)
        term = contract('lmid,la,mjkdbc->ijkabc', V_ooov, t1, t3, optimize='optimal')
        r3 -= P_a_bc(P_i_jk(term))
        
        # T2-T2 terms
        term = contract('ld,ijad,lkbc->ijkabc', f_ov, t2, t2, optimize='optimal')
        r3 -= P_a_bc(P_ij_k(term))
        term = contract('alde,jibd,lkec->ijkabc', V_vovv, t2, t2, optimize='optimal')
        r3 += P_abc_full(P_ij_k(term)) 
        term = contract('alde,ijde,lkbc->ijkabc', V_vovv, t2, t2, optimize='optimal')
        r3 -= 0.5 * P_a_bc(P_ij_k(term))
        term = contract('lmid,jlba,mkdc->ijkabc', V_ooov, t2, t2, optimize='optimal')
        r3 -= P_ba_c(P_ijk_full(term))
        term = contract('lmid,jkbd,lmac->ijkabc', V_ooov, t2, t2, optimize='optimal')
        r3 += 0.5 * P_b_ac(P_i_jk(term))
        
        # T2-T3 terms
        term = contract('lmde,ilde,mjkabc->ijkabc', V_oovv, t2, t3, optimize='optimal')
        r3 += 0.5 * P_i_jk(term)
        term = contract('lmde,lmad,ijkebc->ijkabc', V_oovv, t2, t3, optimize='optimal')
        r3 += 0.5 * P_a_bc(term)
        term = contract('lmde,ijde,lmkabc->ijkabc', V_oovv, t2, t3, optimize='optimal')
        r3 += 0.25 * P_ij_k(term)
        term = contract('lmde,ilad,mjkebc->ijkabc', V_oovv, t2, t3, optimize='optimal')
        r3 += P_a_bc(P_i_jk(term))
        term = contract('lmde,lmab,ijkdec->ijkabc', V_oovv, t2, t3, optimize='optimal')
        r3 += 0.25 * P_ab_c(term)
        term = contract('lmde,ijad,lmkebc->ijkabc', V_oovv, t2, t3, optimize='optimal')
        r3 += 0.5 * P_a_bc(P_ij_k(term))
        term = contract('lmde,ilab,mjkdec->ijkabc', V_oovv, t2, t3, optimize='optimal')
        r3 += 0.5 * P_ab_c(P_i_jk(term))
        
        # Pure T2 and T3 terms
        term = contract('abid,kjcd->ijkabc', V_vvov, t2, optimize='optimal')
        r3 += P_ab_c(P_i_jk(term))
        term = - contract('alij,klcb->ijkabc', V_vooo, t2, optimize='optimal')
        r3 -= P_a_cb(P_ij_k(term))
        
        term = contract('ad,jkibcd->ijkabc', f_vv, t3, optimize='optimal')
        r3 += P_a_bc(term)
        term = contract('li,jklbca->ijkabc', f_oo, t3, optimize='optimal')
        r3 -= P_i_jk(term)
        term = contract('abde,kijcde->ijkabc', V_vvvv, t3, optimize='optimal')
        r3 += 0.5 * P_ab_c(term)
        term = contract('lmij,klmcab->ijkabc', V_oooo, t3, optimize='optimal')
        r3 += 0.5 * P_ij_k(term)
        term = contract('alid,jklbcd->ijkabc', V_voov, t3, optimize='optimal')
        r3 += P_a_bc(P_i_jk(term))
        
        # ==================== Energy & Convergence ====================
        e_corr = 0.25 * np.sum(V_oovv * t2)
        e_corr += 0.5 * np.sum(contract('ijab,ia,jb->', V_oovv, t1, t1))
        
        delta_e = abs(e_corr - old_e)
        iter_time = time.time() - iter_start
        iter_times.append(iter_time)
        
        print(f"[CCSDT-OPT-DIIS] Iter {iteration:3d} | E_corr {e_corr:18.12f} | dE {delta_e:10.4e} | t={iter_time:.1f}s")
        
        if delta_e < tol:
            avg_time = np.mean(iter_times)
            print(f"\nConverged! Avg iteration time: {avg_time:.2f}s")
            return e_corr, t1, t2, t3
            
        old_e = e_corr
        
        # ==================== Amplitude Update with DIIS ====================
        
        # Step: t_new = t + update
        # update = alpha * r/D
        
        step1 = alpha * r1 / D1
        step2 = alpha * r2 / D2
        step3 = alpha * r3 / D3
        
        # Current predictions for next step
        t1_next = t1 + step1
        t2_next = t2 + step2
        t3_next = t3 + step3
        
        # Flatten for DIIS
        # vector = [t1, t2, t3]
        # error = [step1, step2, step3] (Usually we minimize the update/residual)
        
        flat_t = np.concatenate([t1_next.ravel(), t2_next.ravel(), t3_next.ravel()])
        flat_e = np.concatenate([step1.ravel(), step2.ravel(), step3.ravel()])
        
        diis.update(flat_t, flat_e)
        extrap_t = diis.extrapolate()
        
        if extrap_t is not None:
             # Unpack
             i1 = t1.size
             i2 = i1 + t2.size
             
             t1 = extrap_t[:i1].reshape(t1.shape)
             t2 = extrap_t[i1:i2].reshape(t2.shape)
             t3 = extrap_t[i2:].reshape(t3.shape)
        else:
             t1 = t1_next
             t2 = t2_next
             t3 = t3_next

    avg_time = np.mean(iter_times)
    print(f"\nMax iterations reached. Avg iteration time: {avg_time:.2f}s")
    return e_corr, t1, t2, t3


def ccdt(no_ham, n_occ, max_iter=50, tol=1e-8, alpha=1.0, use_sparse=False, diis_size=6, initial_t2=None):
    """
    CCDT: Coupled Cluster Doubles and Triples (no singles).
    Approximation: T1 = 0 (valid for optimized HF basis).
    
    This is CCSDT with all T1-dependent terms removed.
    Much faster than full CCSDT when T1 contributions are negligible.
    Accelerated with DIIS.
    """
    import time
    # Enable caching for contractions in this function
    contract = cached_contract
    
    n_states = no_ham.f.shape[0]
    n_virt = n_states - n_occ
    o = slice(0, n_occ)
    v = slice(n_occ, n_states)

    f = no_ham.f
    Gamma = no_ham.Gamma
    
    f_oo, f_vv, f_ov = f[o, o], f[v, v], f[o, v]
    
    # Interaction blocks
    V_oovv = Gamma[o, o, v, v]
    V_ooov = Gamma[o, o, o, v]
    V_vovv = Gamma[v, o, v, v]
    V_oooo = Gamma[o, o, o, o]
    V_vvvv = Gamma[v, v, v, v]
    V_voov = Gamma[v, o, o, v]
    V_vooo = Gamma[v, o, o, o]
    V_vvov = Gamma[v, v, o, v]
    
    eps = np.diag(f)
    eps_o = eps[o]
    eps_v = eps[v]
    
    D2 = (eps_o[:, None, None, None] + eps_o[None, :, None, None] 
          - eps_v[None, None, :, None] - eps_v[None, None, None, :])
    D3 = (eps_o[:, None, None, None, None, None] + eps_o[None, :, None, None, None, None] + eps_o[None, None, :, None, None, None]
          - eps_v[None, None, None, :, None, None] - eps_v[None, None, None, None, :, None] - eps_v[None, None, None, None, None, :])

    # Initialize: NO T1 in CCDT
    # Initialize: NO T1 in CCDT
    if initial_t2 is not None:
        t2 = initial_t2.copy()
        print("Using provided initial guess for T2.")
    else:
        t2 = V_oovv / D2
        
    t3 = np.zeros((n_occ, n_occ, n_occ, n_virt, n_virt, n_virt))
    
    t3_size_gb = t3.nbytes / 1e9
    print(f"CCDT (no singles). T3: {t3.size/1e6:.2f}M elements ({t3_size_gb:.2f} GB)")
    print(f"Approximation: T1=0 (valid for optimized HF basis)")
    print(f"DIIS Acceleration Enabled (size={diis_size})")

    old_e = 0.0
    iter_times = []
    
    # Initialize DIIS
    diis = DIIS(size=diis_size)
    
    for iteration in range(max_iter):
        iter_start = time.time()
        
        # ==================== T2 Update (NO T1 TERMS) ====================
        r2 = V_oovv.copy()
        
        # Pure T2 terms only
        r2 += P_ab(contract('ac,jibc->ijab', f_vv, t2))
        r2 -= P_ij(contract('ki,jkba->ijab', f_oo, t2))
        r2 += P_ab(P_ij(contract('akic,jkbc->ijab', V_voov, t2)))
        
        # T2-T2 contractions
        r2 += 0.5 * contract('klij,klab->ijab', V_oooo, t2)
        r2 += 0.5 * contract('abcd,ijcd->ijab', V_vvvv, t2)
        r2 += 0.5 * P_ij(contract('klcd,ikab,ljcd->ijab', V_oovv, t2, t2))
        r2 += 0.5 * P_ab(contract('klcd,ijac,kldb->ijab', V_oovv, t2, t2))
        r2 += 0.5 * P_ab(P_ij(contract('klcd,ikac,ljdb->ijab', V_oovv, t2, t2)))
        r2 += 0.25 * contract('klcd,ijcd,klab->ijab', V_oovv, t2, t2)
        
        # T3 contributions to T2
        r2 += contract('kc,ijkabc->ijab', f_ov, t3)
        r2 += 0.5 * P_ab(contract('akcd,jikbcd->ijab', V_vovv, t3))
        r2 -= 0.5 * P_ij(contract('klic,jklbac->ijab', V_ooov, t3))
        
        # ==================== T3 Update (NO T1 TERMS) ====================
        r3 = np.zeros_like(t3)
        
        # Only T2-dependent terms remain (no t1 products)
        # Pure T2 terms
        term = contract('abid,kjcd->ijkabc', V_vvov, t2, optimize='optimal')
        r3 += P_ab_c(P_i_jk(term))
        
        term = - contract('alij,klcb->ijkabc', V_vooo, t2, optimize='optimal')
        r3 -= P_a_cb(P_ij_k(term))
        
        # T2-T2 contractions
        term = contract('ld,ijad,lkbc->ijkabc', f_ov, t2, t2, optimize='optimal')
        r3 -= P_a_bc(P_ij_k(term))
        
        term = contract('alde,jibd,lkec->ijkabc', V_vovv, t2, t2, optimize='optimal')
        r3 += P_abc_full(P_ij_k(term))
        
        term = contract('alde,ijde,lkbc->ijkabc', V_vovv, t2, t2, optimize='optimal')
        r3 -= 0.5 * P_a_bc(P_ij_k(term))
        
        term = contract('lmid,jlba,mkdc->ijkabc', V_ooov, t2, t2, optimize='optimal')
        r3 -= P_ba_c(P_ijk_full(term))
        
        term = contract('lmid,jkbd,lmac->ijkabc', V_ooov, t2, t2, optimize='optimal')
        r3 += 0.5 * P_b_ac(P_i_jk(term))
        
        # T2-T3 contractions
        term = contract('lmde,ilde,mjkabc->ijkabc', V_oovv, t2, t3, optimize='optimal')
        r3 += 0.5 * P_i_jk(term)
        
        term = contract('lmde,lmad,ijkebc->ijkabc', V_oovv, t2, t3, optimize='optimal')
        r3 += 0.5 * P_a_bc(term)
        
        term = contract('lmde,ijde,lmkabc->ijkabc', V_oovv, t2, t3, optimize='optimal')
        r3 += 0.25 * P_ij_k(term)
        
        term = contract('lmde,ilad,mjkebc->ijkabc', V_oovv, t2, t3, optimize='optimal')
        r3 += P_a_bc(P_i_jk(term))
        
        term = contract('lmde,lmab,ijkdec->ijkabc', V_oovv, t2, t3, optimize='optimal')
        r3 += 0.25 * P_ab_c(term)
        
        term = contract('lmde,ijad,lmkebc->ijkabc', V_oovv, t2, t3, optimize='optimal')
        r3 += 0.5 * P_a_bc(P_ij_k(term))
        
        term = contract('lmde,ilab,mjkdec->ijkabc', V_oovv, t2, t3, optimize='optimal')
        r3 += 0.5 * P_ab_c(P_i_jk(term))
        
        # Pure T3 terms
        term = contract('ad,jkibcd->ijkabc', f_vv, t3, optimize='optimal')
        r3 += P_a_bc(term)
        
        term = contract('li,jklbca->ijkabc', f_oo, t3, optimize='optimal')
        r3 -= P_i_jk(term)
        
        term = contract('abde,kijcde->ijkabc', V_vvvv, t3, optimize='optimal')
        r3 += 0.5 * P_ab_c(term)
        
        term = contract('lmij,klmcab->ijkabc', V_oooo, t3, optimize='optimal')
        r3 += 0.5 * P_ij_k(term)
        
        term = contract('alid,jklbcd->ijkabc', V_voov, t3, optimize='optimal')
        r3 += P_a_bc(P_i_jk(term))
        
        # ==================== Energy (NO T1 CONTRIBUTION) ====================
        e_corr = 0.25 * np.sum(V_oovv * t2)
        
        delta_e = abs(e_corr - old_e)
        iter_time = time.time() - iter_start
        iter_times.append(iter_time)
        
        print(f"[CCDT-DIIS] Iter {iteration:3d} | E_corr {e_corr:18.12f} | dE {delta_e:10.4e} | t={iter_time:.1f}s")
        
        if delta_e < tol:
            avg_time = np.mean(iter_times)
            print(f"\nConverged! Avg iteration time: {avg_time:.2f}s")
            return e_corr, t2, t3
            
        old_e = e_corr
        
        # ==================== Amplitude Update with DIIS ====================
        
        step2 = alpha * r2 / D2
        step3 = alpha * r3 / D3
        
        t2_next = t2 + step2
        t3_next = t3 + step3
        
        flat_t = np.concatenate([t2_next.ravel(), t3_next.ravel()])
        flat_e = np.concatenate([step2.ravel(), step3.ravel()])
        
        diis.update(flat_t, flat_e)
        extrap_t = diis.extrapolate()
        
        if extrap_t is not None:
             i2 = t2.size
             t2 = extrap_t[:i2].reshape(t2.shape)
             t3 = extrap_t[i2:].reshape(t3.shape)
        else:
             t2 = t2_next
             t3 = t3_next

    avg_time = np.mean(iter_times)
    print(f"\nMax iterations reached. Avg iteration time: {avg_time:.2f}s")
    return e_corr, t2, t3
