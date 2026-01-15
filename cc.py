
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

import numpy as np

import numpy as np

import numpy as np
import numpy as np

import numpy as np


import numpy as np

def ccd(no_ham, n_occ, max_iter=200, tol=1e-8, alpha=1.0):
    """
    CCD solver using the T2 equation from the provided LaTeX images.
    Implements the Stanton & Gauss intermediate factorization.
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

    old_e = 0.0
    print(f"\n[CCD] {'Iter':>4} | {'E_corr':>18} | {'Delta E':>12}")
    print("-" * 50)

    for iteration in range(max_iter):
        # --- Intermediates for CCD (T1 = 0) ---

        # F_ae: Particle-Particle ladder intermediate
        F_ae = f_vv.copy()
        F_ae -= 0.5 * np.einsum('mnaf,mnef->ae', t2, Gamma[o, o, v, v])

        # F_mi: Hole-Hole ladder intermediate
        F_mi = f_oo.copy()
        F_mi += 0.5 * np.einsum('inef,mnef->mi', t2, Gamma[o, o, v, v])

        # W_mnij: Hole-Hole effective interaction
        W_mnij = Gamma[o, o, o, o] + 0.25 * np.einsum('ijef,mnef->mnij', t2, Gamma[o, o, v, v])

        # W_abef: Particle-Particle effective interaction
        W_abef = Gamma[v, v, v, v] + 0.25 * np.einsum('mnab,mnef->abef', t2, Gamma[o, o, v, v])

        # W_mbej: Particle-Hole ring intermediate (the P(ij)P(ab) terms)
        W_mbej = Gamma[o, v, v, o] - 0.5 * np.einsum('jnfb,mnef->mbej', t2, Gamma[o, o, v, v])

        # --- Residual (R2) Construction ---
        # 1. Constant term: <ab||ij>
        r2 = Gamma[o, o, v, v].copy()

        # 2. F-term coupling (Ladder terms)
        term_ae = np.einsum('ijeb,ae->ijab', t2, F_ae)
        r2 += (term_ae - term_ae.transpose(0, 1, 3, 2)) # P(ab)

        term_mi = np.einsum('mjab,mi->ijab', t2, F_mi)
        r2 -= (term_mi - term_mi.transpose(1, 0, 2, 3)) # P(ij)

        # 3. W-term coupling (Direct ladders)
        r2 += 0.5 * np.einsum('mnab,mnij->ijab', t2, W_mnij)
        r2 += 0.5 * np.einsum('ijef,abef->ijab', t2, W_abef)

        # 4. Ring terms with P(ij)P(ab) permutations
        # This maps to the P(ij)P(ab) sum_{kc} <kb||cj> t_ik^ac term in the image
        term_ring = np.einsum('imae,mbej->ijab', t2, W_mbej)
        r2 += (term_ring - term_ring.transpose(1, 0, 2, 3)
               - term_ring.transpose(0, 1, 3, 2) + term_ring.transpose(1, 0, 3, 2))

        # --- Energy and Update ---
        e_corr = 0.25 * np.sum(Gamma[o, o, v, v] * t2)

        delta_e = abs(e_corr - old_e)
        print(f"[CCD] {iteration:4d} | {e_corr:18.10f} | {delta_e:12.4e}")

        if delta_e < tol:
            print("CCD Converged!")
            return e_corr, t2

        # Additive Update: t = t + alpha * (r / D)
        t2 += alpha * (r2 / D2)
        old_e = e_corr

    return e_corr, t2


import numpy as np


def ccd_diis_solver(no_ham, n_occ, max_iter=200, tol=1e-8, alpha=1.0, diis_size=8):
    """
    Full CCD solver using the provided LaTeX equations.
    Includes intermediate factorization and a stabilized DIIS extrapolation.
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

    # DIIS History Containers
    t_hist = []  # Stores the "extrapolated-ready" amplitudes
    e_hist = []  # Stores the error vectors (Residual / Denominator)

    old_e = 0.0
    print(f"\n[CCD+DIIS] {'Iter':>4} | {'E_corr':>18} | {'Delta E':>12}")
    print("-" * 55)

    for i in range(max_iter):
        # 2. Construction of Intermediates (O(N^6) scaling)
        F_ae = f_vv.copy() - 0.5 * np.einsum('mnaf,mnef->ae', t2, Gamma[o, o, v, v])
        F_mi = f_oo.copy() + 0.5 * np.einsum('inef,mnef->mi', t2, Gamma[o, o, v, v])

        W_mnij = Gamma[o, o, o, o] + 0.25 * np.einsum('ijef,mnef->mnij', t2, Gamma[o, o, v, v])
        W_abef = Gamma[v, v, v, v] + 0.25 * np.einsum('mnab,mnef->abef', t2, Gamma[o, o, v, v])
        W_mbej = Gamma[o, v, v, o] - 0.5 * np.einsum('jnfb,mnef->mbej', t2, Gamma[o, o, v, v])

        # 3. Build Residual R_ijab (Matches the provided LaTeX equations)
        r2 = Gamma[o, o, v, v].copy() # Constant term <ab||ij>

        # F-couplings (Ladders)
        term_ae = np.einsum('ijeb,ae->ijab', t2, F_ae)
        r2 += (term_ae - term_ae.transpose(0, 1, 3, 2)) # P(ab)

        term_mi = np.einsum('mjab,mi->ijab', t2, F_mi)
        r2 -= (term_mi - term_mi.transpose(1, 0, 2, 3)) # P(ij)

        # W-couplings
        r2 += 0.5 * np.einsum('mnab,mnij->ijab', t2, W_mnij)
        r2 += 0.5 * np.einsum('ijef,abef->ijab', t2, W_abef)

        # Ring terms with P(ij)P(ab) permutations
        term_ring = np.einsum('imae,mbej->ijab', t2, W_mbej)
        r2 += (term_ring - term_ring.transpose(1, 0, 2, 3)
               - term_ring.transpose(0, 1, 3, 2) + term_ring.transpose(1, 0, 3, 2))

        # 4. Energy calculation and Convergence check
        e_corr = 0.25 * np.sum(Gamma[o, o, v, v] * t2)
        delta_e = abs(e_corr - old_e)
        print(f" {i:4d} | {e_corr:18.10f} | {delta_e:12.4e}")

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

    return e_corr, t2


def ccsd_diis_solver(no_ham, n_occ, max_iter=200, tol=1e-8, alpha=1.0, diis_size=6):
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

    # DIIS Storage
    t_list = []
    e_list = []

    old_e = 0.0
    print(f"\n[CCSD+DIIS] {'Iter':>4} | {'E_corr':>18} | {'Delta E':>12}")
    print("-" * 55)

    for i in range(max_iter):
        # --- 1. Intermediate and Residual Calculation (Same as before) ---
        t1t1 = np.einsum('ia,jb->ijab', t1, t1)
        tau = t2 + t1t1 - t1t1.transpose(0, 1, 3, 2)
        tau_tilde = t2 + 0.5 * (t1t1 - t1t1.transpose(0, 1, 3, 2))

        # F-Intermediates
        F_ae = f[v, v].copy() - 0.5 * np.einsum('me,ma->ae', f[o, v], t1)
        F_ae += np.einsum('mf,amef->ae', t1, Gamma[v, o, v, v])
        F_ae -= 0.5 * np.einsum('mnaf,mnef->ae', tau_tilde, Gamma[o, o, v, v])

        F_mi = f[o, o].copy() + 0.5 * np.einsum('ie,me->mi', t1, f[o, v])
        F_mi += np.einsum('ne,mnie->mi', t1, Gamma[o, o, o, v])
        F_mi += 0.5 * np.einsum('inef,mnef->mi', tau_tilde, Gamma[o, o, v, v])

        F_me = f[o, v] + np.einsum('nf,mnef->me', t1, Gamma[o, o, v, v])

        # W-Intermediates
        W_mnij = Gamma[o, o, o, o] + 0.25 * np.einsum('ijef,mnef->mnij', tau, Gamma[o, o, v, v])
        term_mnij = np.einsum('je,mnie->mnij', t1, Gamma[o, o, o, v])
        W_mnij += (term_mnij - term_mnij.transpose(0, 1, 3, 2))

        W_abef = Gamma[v, v, v, v] + 0.25 * np.einsum('mnab,mnef->abef', tau, Gamma[o, o, v, v])
        term_abef = np.einsum('ma,mbef->abef', t1, Gamma[o, v, v, v])
        W_abef -= (term_abef - term_abef.transpose(1, 0, 2, 3))

        W_mbej = Gamma[o, v, v, o] + np.einsum('jf,mbef->mbej', t1, Gamma[o, v, v, v])
        W_mbej -= np.einsum('nb,mnej->mbej', t1, Gamma[o, o, v, o])
        W_mbej -= 0.5 * np.einsum('jnfb,mnef->mbej', t2, Gamma[o, o, v, v])
        W_mbej -= np.einsum('jf,nb,mnef->mbej', t1, t1, Gamma[o, o, v, v])

        # Residuals
        r1 = f[v, o].T + np.einsum('ie,ae->ia', t1, F_ae) - np.einsum('ma,mi->ia', t1, F_mi)
        r1 += np.einsum('imae,me->ia', t2, F_me) + np.einsum('nf,nafi->ia', t1, Gamma[o, v, v, o])
        r1 -= 0.5 * np.einsum('imef,maef->ia', t2, Gamma[o, v, v, v]) + 0.5 * np.einsum('mnea,mnei->ia', t2, Gamma[o, o, v, o])

        r2 = Gamma[o, o, v, v].copy()
        term_ae = np.einsum('ijeb,ae->ijab', t2, F_ae)
        r2 += (term_ae - term_ae.transpose(0, 1, 3, 2))
        term_mi = np.einsum('mjab,mi->ijab', t2, F_mi)
        r2 -= (term_mi - term_mi.transpose(1, 0, 2, 3))
        r2 += 0.5 * np.einsum('mnab,mnij->ijab', tau, W_mnij) + 0.5 * np.einsum('ijef,abef->ijab', tau, W_abef)

        term_ring = np.einsum('imae,mbej->ijab', t2, W_mbej)
        r2 += (term_ring - term_ring.transpose(1, 0, 2, 3) - term_ring.transpose(0, 1, 3, 2) + term_ring.transpose(1, 0, 3, 2))

        term_t1_v = np.einsum('ie,abej->ijab', t1, Gamma[v, v, v, o])
        r2 += (term_t1_v - term_t1_v.transpose(1, 0, 2, 3))
        term_t1_o = np.einsum('ma,mbij->ijab', t1, Gamma[o, v, o, o])
        r2 -= (term_t1_o - term_t1_o.transpose(0, 1, 3, 2))

        # --- 2. Energy and Convergence Check ---
        e_corr = np.sum(f[o, v] * t1) + 0.25 * np.sum(Gamma[o, o, v, v] * tau)
        delta_e = abs(e_corr - old_e)
        print(f"[CCSD] {i:4d} | {e_corr:18.10f} | {delta_e:12.4e}")

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

    return e_corr, t1, t2

def ccsd(no_ham, n_occ, max_iter=200, tol=1e-8, alpha=0.5):
    """
    CCSD solver based on the provided T1 and T2 long-form equations.
    Optimized via Stanton & Gauss intermediate factorization.
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

    old_e = 0.0
    for i in range(max_iter):
        # 1. Intermediates (Combining T1 and T2 terms as shown in equations)
        # tau_tilde = t_ijab + 0.5 * (t_ia * t_jb - t_ib * t_ja)
        t1t1 = np.einsum('ia,jb->ijab', t1, t1)
        tau = t2 + t1t1 - t1t1.transpose(0, 1, 3, 2)
        tau_tilde = t2 + 0.5 * (t1t1 - t1t1.transpose(0, 1, 3, 2))

        # F-Intermediates (Effective Fock matrices)
        F_ae = f_vv.copy()
        F_ae -= 0.5 * np.einsum('me,ma->ae', f_ov, t1)
        F_ae += np.einsum('mf,amef->ae', t1, Gamma[v, o, v, v])
        F_ae -= 0.5 * np.einsum('mnaf,mnef->ae', tau_tilde, Gamma[o, o, v, v])

        F_mi = f_oo.copy()
        F_mi += 0.5 * np.einsum('ie,me->mi', t1, f_ov)
        F_mi += np.einsum('ne,mnie->mi', t1, Gamma[o, o, o, v])
        F_mi += 0.5 * np.einsum('inef,mnef->mi', tau_tilde, Gamma[o, o, v, v])

        F_me = f_ov + np.einsum('nf,mnef->me', t1, Gamma[o, o, v, v])

        # W-Intermediates (Effective Integrals)
        W_mnij = Gamma[o, o, o, o] + 0.25 * np.einsum('ijef,mnef->mnij', tau, Gamma[o, o, v, v])
        W_mnij += (np.einsum('je,mnie->mnij', t1, Gamma[o, o, o, v]) -
                   np.einsum('ie,mnje->mnij', t1, Gamma[o, o, o, v]))

        W_abef = Gamma[v, v, v, v] + 0.25 * np.einsum('mnab,mnef->abef', tau, Gamma[o, o, v, v])
        W_abef -= (np.einsum('ma,mbef->abef', t1, Gamma[o, v, v, v]) -
                   np.einsum('mb,maef->abef', t1, Gamma[o, v, v, v]))

        W_mbej = Gamma[o, v, v, o] + np.einsum('jf,mbef->mbej', t1, Gamma[o, v, v, v])
        W_mbej -= np.einsum('nb,mnej->mbej', t1, Gamma[o, o, v, o])
        W_mbej -= 0.5 * np.einsum('jnfb,mnef->mbej', t2, Gamma[o, o, v, v])
        W_mbej -= np.einsum('jf,nb,mnef->mbej', t1, t1, Gamma[o, o, v, v])

        # 2. Residuals
        # T1 Residual (Maps to Image 1)
        r1 = f_vo.T + np.einsum('ie,ae->ia', t1, F_ae) - np.einsum('ma,mi->ia', t1, F_mi)
        r1 += np.einsum('imae,me->ia', t2, F_me)
        r1 += np.einsum('nf,nafi->ia', t1, Gamma[o, v, v, o])
        r1 -= 0.5 * np.einsum('imef,maef->ia', t2, Gamma[o, v, v, v])
        r1 -= 0.5 * np.einsum('mnea,mnei->ia', t2, Gamma[o, o, v, o])

        # T2 Residual (Maps to Image 2)
        r2 = Gamma[o, o, v, v].copy()

        # Linear and Quadratic T2 couplings
        term_ae = np.einsum('ijeb,ae->ijab', t2, F_ae)
        r2 += (term_ae - term_ae.transpose(0, 1, 3, 2))
        term_mi = np.einsum('mjab,mi->ijab', t2, F_mi)
        r2 -= (term_mi - term_mi.transpose(1, 0, 2, 3))

        r2 += 0.5 * np.einsum('mnab,mnij->ijab', tau, W_mnij)
        r2 += 0.5 * np.einsum('ijef,abef->ijab', tau, W_abef)

        # Ring terms P(ij)P(ab)
        term_ring = np.einsum('imae,mbej->ijab', t2, W_mbej)
        r2 += (term_ring - term_ring.transpose(1, 0, 2, 3) -
               term_ring.transpose(0, 1, 3, 2) + term_ring.transpose(1, 0, 3, 2))

        # T1 couplings
        term_t1_v = np.einsum('ie,abej->ijab', t1, Gamma[v, v, v, o])
        r2 += (term_t1_v - term_t1_v.transpose(1, 0, 2, 3))
        term_t1_o = np.einsum('ma,mbij->ijab', t1, Gamma[o, v, o, o])
        r2 -= (term_t1_o - term_t1_o.transpose(0, 1, 3, 2))

        # 3. Energy and Step Update
        e_corr = np.sum(f_ov * t1) + 0.25 * np.sum(Gamma[o, o, v, v] * tau)

        if abs(e_corr - old_e) < tol:
            return e_corr, t1, t2

        t1 = alpha * (r1 / D1) +  t1
        t2 = alpha * (r2 / D2) +  t2
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
