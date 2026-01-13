
import numpy as np
from opt_einsum import contract

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

def lambda_ccsdt(no_ham, t1, t2, t3, n_occ, max_iter=100, tol=1e-8, alpha=0.5):
    """
    Solve Lambda-CCSDT equations.
    
    Corresponds to the CCSDT solver in cc.py
    """
    n_states = no_ham.f.shape[0]
    o = slice(0, n_occ)
    v = slice(n_occ, n_states)
    n_virt = n_states - n_occ
    
    f = no_ham.f
    v2b = no_ham.Gamma
    
    eps = np.diag(f)
    f_off = f - np.diag(eps)
    f_oo = f_off[o, o]
    f_vv = f_off[v, v]
    f_ov = f_off[o, v]
    f_vo = f_off[v, o]
    
    # Denominators
    D1 = eps[o, None] - eps[None, v]
    D2 = eps[o, None, None, None] + eps[None, o, None, None] - eps[None, None, v, None] - eps[None, None, None, v]
    D3 = (eps[o, None, None, None, None, None] + 
          eps[None, o, None, None, None, None] + 
          eps[None, None, o, None, None, None] -
          eps[None, None, None, v, None, None] - 
          eps[None, None, None, None, v, None] - 
          eps[None, None, None, None, None, v])
          
    # Initialize L amplitudes
    l1 = t1.copy()
    l2 = t2.copy()
    l3 = t3.copy()
    
    print(f"\n[Lambda-CCSDT] {'Iter':>4} | {'||L3||':>12} | {'Delta':>12}")
    print("-" * 60)
    
    # Debug initial norms
    print(f"DEBUG: Initial ||t3|| = {np.linalg.norm(t3):.6e}")
    old_norm = np.linalg.norm(l3)
    print(f"DEBUG: Initial old_norm = {old_norm:.6e}")
    
    # Reduce alpha for stability
    alpha = 0.2
    
    for iteration in range(max_iter):
        
        # --- L1 Residual (Standard CCSD parts) ---
        r_l1 = f_vo.T.copy()
        r_l1 += contract('ie,ea->ia', l1, f_vv)
        r_l1 -= contract('ma,mi->ia', l1, f_oo)
        r_l1 += contract('imae,me->ia', v2b[o, o, v, v], l1) * 2.0
        r_l1 -= contract('imea,me->ia', v2b[o, o, v, v], l1)
        r_l1 += contract('imef,mnea->ia', l2, v2b[o, o, v, v]) # L2 corrections
        
        # --- L2 Residual (Standard CCSD parts) ---
        r_l2 = v2b[o, o, v, v].copy()
        r_l2 += pAB(contract('ie,ejab->ijab', l1, v2b[v, o, v, v]))
        r_l2 -= pIJ(contract('ma,imjb->ijab', l1, v2b[o, o, o, v]))
        r_l2 += 0.5 * contract('abef,ijef->ijab', v2b[v, v, v, v], l2)
        r_l2 += 0.5 * contract('mnij,mnab->ijab', v2b[o, o, o, o], l2)
        r_l2 += pIJ(pAB(contract('mbej,imae->ijab', v2b[o, v, v, o], l2)))
        
        # --- Coupling from L3 to L2 (Adjoint of T3 source terms) ---
        # T3 Source 1: P(ijk) P(abc) contract('ijae,kebc->ijkabc', t2, v2b[o, v, v, v])
        # L2 Correction: Contract L3 with V to match T2 indices (ijae)
        # l3[ijkabc] * v[kebc] -> ijae
        # Indices: k, b, c summed.
        
        # We need to account for the permutations in T3 source.
        # The T3 source had minimal permutations applied.
        # Ideally, we contract the FULL SYMMETRIC L3 with the raw V, then apply T2 symmetries?
        # Or contract raw L3.
        # Given L3 in the loop should converge to the symmetric form, we can just contract.
        # However, term weights matter.
        # T3 eq: R3 += P(...) (T2*V)
        # J_32 = dR3/dT2 = P * V
        # L2 eq: R_L2 += L3 * J_32 = L3 * P * V = (P^T L3) * V
        # Since L3 is symmetric, P^T L3 = Sum_perm sgn(p) L3 ... 
        # Basically, we contract L3 with V.
        # Factor? The T3 source had no explicit factor (coeff 1.0).
        # We just need to sum over the k, b, c contractions.
        # Note: L3 has 3!*3! = 36 equivalent components. 
        # But we perform specific contractions.
        # The "permute_1_23" logic generates 3 terms. (1 - P12 - P13).
        # And P(abc) generates 3 terms. Total 9 terms.
        # So we should probably accumulate 9 contractions?
        # Or, simpler: Contract L3_symm with V_raw and multiply by appropriate factor?
        # Let's do the explicit contractions corresponding to the raw terms in T-eq,
        # exploiting the fact that L3 is antisymmetric.
        
        # Term 1 logic from cc.py: 
        # src1 = contract('ijae,kebc->ijkabc', t2, v2b[o, v, v, v])
        # src1 = permute_1_23(src1, 2, 0, 1) # k/ij
        # src1 = permute_1_23(src1, 3, 4, 5) # a/bc
        
        # Adjoint:
        # L2[ijae] += contract('ijkabc,kebc->ijae', l3, v2b[o,v,v,v])
        # But for the permutations? 
        # If L3 is fully antisymmetric, contracting with any permutation of ijkabc yields same result (with sign).
        # The permutations in T3 eq were sum of 9 terms (3 for ijk * 3 for abc).
        # So effective contribution is 9 * contract(...).
        # BUT, we must be careful with indices.
        # The term was (1 - P_ik - P_jk)... wait permute_1_23 is (1 - P(1,2) - P(1,3)).
        # For k/ij (k=2, i=0, j=1): 1 - P(20) - P(21) = 1 - (ki) - (kj).
        # This sums over 3 permutations of {i,j,k}.
        # Similarly for {a,b,c}.
        # So yes, factor of 9.
        
        r_l2 += 9.0 * contract('ijkabc,kebc->ijae', l3, v2b[o, v, v, v])
        
        # Term 2 logic:
        # src2 = -contract('imab,jkmc->ijkabc', t2, v2b[o, o, o, v])
        # src2 = permute_1_23(src2, 0, 1, 2) # i/jk
        # src2 = permute_1_23(src2, 5, 3, 4) # c/ab
        # Adjoint:
        # L2[imab] -= 9.0 * contract('ijkabc,jkmc->imab', l3, v2b[o,o,o,v])
        r_l2 -= 9.0 * contract('ijkabc,jkmc->imab', l3, v2b[o, o, o, v])
        
        # --- L3 Residual ---
        # 1. Driving terms from L1/L2 (Adjoint of T3 feedback into T1/T2)
        
        # T1 term: r1 += 0.25 * contract('mnef,imnaef->ia', v2b[o, o, v, v], t3)
        # Map: L3[imnaef] += 0.25 * contract('ia,mnef->imnaef', l1, v2b)
        src_l3_from_l1 = contract('ia,mnef->imnaef', l1, v2b[o, o, v, v])
        r_l3 = 0.25 * src_l3_from_l1
        
        # T2 term: r2 += 0.5 * contract('mnef,ijmnabef->ijab', v2b, contract('ie,jmnabf->ijmnabef', t1, t3))
        # Adjoint L3 equation:
        # R_L3_{jmnabf} += d(R2_{ijab}) / d(T3_{jmnabf}) * L2_{ijab}
        # j->0, m->1, n->2, a->3, b->4, f->5.
        term_l2 = contract('ijab,mnef,ie->jmnabf', l2, v2b[o, o, v, v], t1)
        r_l3 += 0.5 * term_l2

        # Note: The indices of r_l3 must match standard ijkabc.
        
        # 2. Diagonal/Ladder terms (Adjoint of T3 self-interactions - F and W)
        
        # F terms in T3:
        r_l3 -= 3.0 * contract('ijkabc,mi->mjkabc', l3, f_oo)
        r_l3 += 3.0 * contract('ijkabc,ec->ijkabe', l3, f_vv)
        
        # Ladder terms in T3:
        # term5 = 0.5 * contract('ijkaef,efbc->ijkabc', t3, v2b[v, v, v, v]) (+ perms)
        # Adjoint: sum_{bc} L[ijkabc] * V[efbc].
        # Target: ijkaef.
        # Perms: 3.
        # r_l3 += 0.5 * 3.0 * contract('ijkabc,efbc->ijkaef', l3, v2b[v, v, v, v])

        # term6 = 0.5 * contract('imnabc,jkmn->ijkabc', t3, v2b[o, o, o, o]) (+ perms)
        # Adjoint: sum_{jk} L[ijkabc] * V[jkmn].
        # Target: imnabc.
        # Perms: 3.
        # r_l3 += 0.5 * 3.0 * contract('ijkabc,jkmn->imnabc', l3, v2b[o, o, o, o])

        
        # Enforce antisymmetry for L3 Residual
        # (Since we just added raw terms, extracting the antisymmetric component)
        # P(ijk): 1 - P12 - P13 - P23 + ...
        # Standard antisymmetrize 
        
        def antisymmetrize_ijk(tensor):
            t = tensor - np.transpose(tensor, (1, 0, 2, 3, 4, 5)) 
            t = t - np.transpose(t, (2, 1, 0, 3, 4, 5)) # P_ik via cycle? No.
            # Use full sum (6 terms) / 6? Or just construct it.
            # Minimal set to ensure symmetry if starting from somewhat symmetric?
            # Safe bet: Full antisymmetrization summation
            t = (tensor 
                   - np.transpose(tensor, (1, 0, 2, 3, 4, 5)) # - ij
                   - np.transpose(tensor, (2, 1, 0, 3, 4, 5)) # - ik? (no this is 210)
                   - np.transpose(tensor, (0, 2, 1, 3, 4, 5)) # - jk
                   + np.transpose(tensor, (1, 2, 0, 3, 4, 5)) # + jki
                   + np.transpose(tensor, (2, 0, 1, 3, 4, 5)) # + kij
                   )
            # What about -ik? 210 vs 021?
            # 012 -> 210 (swap 0,2).
            # The list above has:
            # 012 (ref)
            # 102 (-ij)
            # 210 (-ik)
            # 021 (-jk)
            # 120 (+jki)
            # 201 (+kij)
            return t

        def antisymmetrize_abc(tensor):
             t = (tensor 
                   - np.transpose(tensor, (0, 1, 2, 4, 3, 5)) 
                   - np.transpose(tensor, (0, 1, 2, 5, 4, 3)) 
                   - np.transpose(tensor, (0, 1, 2, 3, 5, 4)) 
                   + np.transpose(tensor, (0, 1, 2, 4, 5, 3)) 
                   + np.transpose(tensor, (0, 1, 2, 5, 3, 4)) 
                   )
             return t

        r_l3 = antisymmetrize_ijk(r_l3)
        r_l3 = antisymmetrize_abc(r_l3)

        # Update Amplitudes
        new_l1 = r_l1 / D1
        new_l2 = r_l2 / D2
        new_l3 = r_l3 / D3
        
        l1 = (1 - alpha) * l1 + alpha * new_l1
        l2 = (1 - alpha) * l2 + alpha * new_l2
        l3 = (1 - alpha) * l3 + alpha * new_l3
        
        new_norm = np.linalg.norm(l3)
        delta = abs(new_norm - old_norm)
        print(f"[Lambda-CCSDT] {iteration:4d} | {new_norm:12.6e} | {delta:12.6e}")
        
        if delta < tol:
            print("Lambda-CCSDT Converged!")
            break
            
        old_norm = new_norm
        
    return l1, l2, l3
