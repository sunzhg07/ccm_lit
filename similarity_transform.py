import numpy as np
from opt_einsum import contract
import copy

def pAB(val):
    """Permutator val(abij) -> val(abij) - val(baij)"""
    return val - np.transpose(val, (1, 0, 2, 3))

def pIJ(val):
    """Permutator val(abij) -> val(abij) - val(abji)"""
    return val - np.transpose(val, (0, 1, 3, 2))

def similarity_transform_t1(no_ham, t1, n_occ):
    """
    Compute the similarity transformed Hamiltonian with T1 only:
    H_bar = e^{-T1} H e^{T1}
    """
    n_states = no_ham.f.shape[0]
    o = slice(0, n_occ)
    v = slice(n_occ, n_states)
    
    # Define T1-transformed Hamiltonian object
    H_bar = copy.deepcopy(no_ham)
    f = H_bar.f
    g = H_bar.Gamma
    
    # Original integrals for reference
    f_old = no_ham.f
    g_old = no_ham.Gamma
    
    # --- 1. Update 2-body Integrals (V_bar) ---
    
    # W_oooo <mn||ij> += P(ij) <mn||ie> t_j^e + 0.5 <mn||ef> t_i^e t_j^f
    term1 = contract('mnie,je->mnij', g_old[o,o,o,v], t1)
    g[o,o,o,o] += (term1 - np.transpose(term1, (0,1,3,2))) # P(ij)
    g[o,o,o,o] += 0.5 * contract('mnef,ie,jf->mnij', g_old[o,o,v,v], t1, t1)
    
    # W_vvvv <ab||cd> -= P(ab) <am||cd> t_m^b + 0.5 <mn||cd> t_m^a t_n^b
    term1 = contract('amef,mb->abef', g_old[v,o,v,v], t1)
    g[v,v,v,v] -= (term1 - np.transpose(term1, (1,0,2,3))) # P(ab)
    g[v,v,v,v] += 0.5 * contract('mnef,ma,nb->abef', g_old[o,o,v,v], t1, t1)
    
    # W_ovvo <mb||ej> += <mb||ef> t_j^f - <mn||ej> t_n^b - <mn||ef> t_n^b t_j^f
    # Check notation <mb||ej> corresponds to [o,v,v,o]? Yes usually.
    # Term 1: <mb||ef> t_j^f
    g[o,v,v,o] += contract('mbef,jf->mbej', g_old[o,v,v,v], t1)
    # Term 2: - <mn||ej> t_n^b (<mn||ej> is [o,o,v,o])
    # [o,o,v,o] usually stored as [o,o,v,o] <mn||ea>. Let's assume indices match.
    # g[mnje] -> m n j e. contract n, b.
    g[o,v,v,o] -= contract('mnje,nb->mbej', g_old[o,o,o,v], t1) # Using mnje from ooov
    # Term 3: - <mn||ef> t_n^b t_j^f
    g[o,v,v,o] -= contract('mnef,nb,jf->mbej', g_old[o,o,v,v], t1, t1)
    
    # W_ooov <mn||ie> += <mn||ef> t_i^f
    g[o,o,o,v] += contract('mnef,if->mnie', g_old[o,o,v,v], t1)
    
    # W_vovv <am||ef> -= <nm||ef> t_n^a
    g[v,o,v,v] -= contract('nmef,na->amef', g_old[o,o,v,v], t1)
    
    # W_ovov <ma||ie> (also appearing in T1 residual)
    # <ma||ie> += <ma||ef> t_i^f - <mn||ie> t_n^a - <mn||ef> t_n^a t_i^f
    g[o,v,o,v] += contract('maef,if->maie', g_old[o,v,v,v], t1)
    g[o,v,o,v] -= contract('mnie,na->maie', g_old[o,o,o,v], t1)
    g[o,v,o,v] -= contract('mnef,na,if->maie', g_old[o,o,v,v], t1, t1)
    
    # --- 2. Update 1-body Integrals (F_bar) ---
    
    # F_oo (mi) = f_mi + f_me t_i^e + <mn||ie> t_n^e + 0.5 <mn||ef> t_i^e t_n^f
    # Note: Using raw integrals implies we must construct full terms.
    f[o,o] += contract('me,ie->mi', f_old[o,v], t1)
    f[o,o] += contract('mnie,ne->mi', g_old[o,o,o,v], t1)
    f[o,o] += 0.5 * contract('mnef,ie,nf->mi', g_old[o,o,v,v], t1, t1)
    
    # F_vv (ae) = f_ae - f_me t_m^a - <am||ef> t_m^f + 0.5 <mn||ef> t_m^a t_n^f
    f[v,v] -= contract('me,ma->ae', f_old[o,v], t1)
    f[v,v] -= contract('amef,mf->ae', g_old[v,o,v,v], t1)
    f[v,v] += 0.5 * contract('mnef,ma,nf->ae', g_old[o,o,v,v], t1, t1)
    
    # F_ov (me) = f_me (unchanged)
    
    # F_vo (ai) = f_ai + f_ae t_i^e - f_mi t_m^a + <am||ie> t_m^e  ... etc
    # The full T1 residual expression gives F_vo.
    # explicit raw expansion:
    f[v,o] += contract('ae,ie->ai', f_old[v,v], t1)
    f[v,o] -= contract('mi,ma->ai', f_old[o,o], t1)
    f[v,o] += contract('maie,me->ai', g_old[o,v,o,v], t1)
    f[v,o] -= contract('mnie,ma,ne->ai', g_old[o,o,o,v], t1, t1)
    f[v,o] -= contract('maef,ie,mf->ai', g_old[o,v,v,v], t1, t1)
    f[v,o] += contract('mnef,ie,ma,nf->ai', g_old[o,o,v,v], t1, t1, t1)
    f[v,o] -= contract('me,ie,ma->ai', f_old[o,v], t1, t1)
    
    return H_bar

def similarity_transform_t1_t2(no_ham, t1, t2, n_occ):
    """
    Compute similarity transform allowing for T2 contributions.
    H_bar = e^{-(T1+T2)} H e^{(T1+T2)}
    
    Construct H_bar_effective (1-body and 2-body parts).
    Includes [V, T2] terms.
    """
    H_t1 = similarity_transform_t1(no_ham, t1, n_occ)
    
    H_bar = copy.deepcopy(H_t1)
    f = H_bar.f
    v2b = H_bar.Gamma
    n_states = f.shape[0]
    o = slice(0, n_occ)
    v = slice(n_occ, n_states)
    
    # 1. Update F with T2
    f[o,o] += 0.5 * contract('mnef,inef->mi', H_t1.Gamma[o,o,v,v], t2)
    f[v,v] -= 0.5 * contract('mnef,mnaf->ae', H_t1.Gamma[o,o,v,v], t2)
    
    f[v,o] += 0.5 * contract('maef,imef->ai', H_t1.Gamma[o,v,v,v], t2)
    f[v,o] += 0.5 * contract('mnie,mnae->ai', H_t1.Gamma[o,o,o,v], t2)
    
    # 2. Update V with T2
    v2b[o,o,o,o] += 0.5 * contract('mnef,ijef->mnij', H_t1.Gamma[o,o,v,v], t2)
    v2b[v,v,v,v] += 0.5 * contract('mnef,mnab->abef', H_t1.Gamma[o,o,v,v], t2)
    
    term = contract('mnef,jnbf->mejb', H_t1.Gamma[o,o,v,v], t2)
    v2b[o,v,v,o] -= contract('mejb->mbej', term)
    
    v2b[o,o,v,v] += 0.5 * contract('mnij,mnab->ijab', H_t1.Gamma[o, o, o, o], t2)
    v2b[o,o,v,v] += 0.5 * contract('abef,ijef->ijab', H_t1.Gamma[v, v, v, v], t2)
    
    term = contract('maie,mjeb->ijab', H_t1.Gamma[o, v, o, v], t2)
    v2b[o,o,v,v] -= pIJ(pAB(term))
    
    v2b[o,o,v,v] -= pIJ(contract('mi,mjab->ijab', H_t1.f[o,o], t2))
    v2b[o,o,v,v] += pAB(contract('ae,ijeb->ijab', H_t1.f[v,v], t2))
    
    return H_bar

def similarity_transform_l1(op, l1, n_occ):
    """
    Transforms an operator (1-body + 2-body) by a de-excitation operator L1.
    New_Op = e^{-L1} Op e^{L1}
    
    Since L1 is de-excitation (i^dag a), the expansion naturally truncates
    and can be evaluated by transforming the basis of the operator:
    c_i  -> c_i + l_ia c_a
    c_a^dag -> c_a^dag - l_ia c_i^dag
    
    Parameters:
    -----------
    op : Object with .f and .Gamma (like Hamiltonian)
    l1 : L1 amplitudes [n_occ, n_virt]
    n_occ : number of occupied orbitals
    
    Returns:
    --------
    transformed_op : New object with transformed .f and .Gamma
    """
    n_states = op.f.shape[0]
    o = slice(0, n_occ)
    v = slice(n_occ, n_states)
    
    # Create transformation matrices
    # U_L for creation operators (rows p)
    # c_p^dag -> sum_q U^L_qp c_q^dag
    # c_i^dag -> c_i^dag (column i has 1 at i)
    # c_a^dag -> c_a^dag - l_ia c_i^dag (column a has 1 at a, -l_ia at i)
    u_l = np.eye(n_states)
    u_l[o, v] -= l1
    
    # U_R for annihilation operators (cols q)
    # c_q -> sum_r U^R_rq c_r
    # c_i -> c_i + l_ia c_a (col i has 1 at i, l_ia at a)
    # c_a -> c_a (col a has 1 at a)
    u_r = np.eye(n_states)
    u_r[v, o] += l1.T
    
    # Transformed op = U_L Op U_R
    # (Since Op_pq p^dag q -> U_rp p^dag ... U_sq q -> U^L_rp Op_pq U^R_sq r^dag s)
    # Matrix mult: U_L * f * U_R
    
    new_op = copy.deepcopy(op)
    
    # 1-body transform
    new_op.f = u_l @ op.f @ u_r
    
    # 2-body transform
    # Gamma_pqrs p^dag q^dag s r
    # Transform p (index 0) with U_L
    # Transform q (index 1) with U_L
    # Transform r (index 3) with U_R
    # Transform s (index 2) with U_R
    # Note: einsum handles this cleanely.
    
    # G_new = U_L(0) U_L(1) G_old U_R(2) U_R(3)
    # But note indices of G are [p,q,r,s] corresponding to <pq||rs> = p^dag q^dag s r.
    # indices 0,1 are creation (U_L).
    # indices 2,3 are annihilation (U_R).
    
    # Optimize contractions:
    # 1. Contract index 0
    temp = contract('ap, pqrs -> aqrs', u_l, op.Gamma)
    # 2. Contract index 1
    temp = contract('bq, aqrs -> abrs', u_l, temp)
    # 3. Contract index 2
    temp = contract('cr, abrs -> abcs', u_r.T, temp) # U_R contracted on second index of U?
    # U_R[r, q] means coeff of basis r for input q.
    # q is the index in G. r is the new index.
    # So we sum over q: U_rq G_...q...
    # U_R is (n_states, n_states).
    # We want (U_R)_cr * G_...r...
    # My U_R construction: row r, col q.
    # So yes, contract U_R with G on index 2.
    # 4. Contract index 3
    new_op.Gamma = contract('ds, abcs -> abcd', u_r.T, temp)
    
    return new_op
