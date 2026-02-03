"""
Permutation operators for CCSDTQ
Follows the convention:
- P(a/bcd): a vs. already-antisymmetric triple (b,c,d) → 4 terms
- P(ab/cd): pair (a,b) vs. pair (c,d) → 3 terms  
- P(abc/d): triple (a,b,c) vs. d → 4 terms
- P(a/b/c/d): Full antisymmetrization → 24 terms
"""

import numpy as np

# ============================================================================
# T2 Permutations (2 indices)
# ============================================================================

def P_ij(t):
    """P(i/j) = 1 - P_ij: Antisymmetrize i,j"""
    return t - t.transpose(1, 0, 2, 3)

def P_ab(t):
    """P(a/b) = 1 - P_ab: Antisymmetrize a,b"""
    return t - t.transpose(0, 1, 3, 2)

# ============================================================================
# T3 Permutations (3 indices)
# ============================================================================

def P_a_bc(t):
    """P(a/bc): a vs. antisymmetric pair (b,c) → 3 terms"""
    return t - t.transpose(0,1,2, 4,3,5) - t.transpose(0,1,2, 5,4,3)

def P_i_jk(t):
    """P(i/jk): i vs. antisymmetric pair (j,k) → 3 terms"""
    return t - t.transpose(1,0,2, 3,4,5) - t.transpose(2,1,0, 3,4,5)

def P_ij_k(t):
    """P(ij/k): pair (i,j) vs. k → 3 terms"""
    return t - t.transpose(2,1,0, 3,4,5) - t.transpose(0,2,1, 3,4,5)

def P_ab_c(t):
    """P(ab/c): pair (a,b) vs. c → 3 terms"""
    return t - t.transpose(0,1,2, 5,4,3) - t.transpose(0,1,2, 3,5,4)

def P_b_ac(t):
    """P(b/ac): b vs. pair (a,c) → 3 terms"""
    return t - t.transpose(0,1,2, 4,3,5) - t.transpose(0,1,2, 3,5,4)

def P_c_ab(t):
    """P(c/ab): c vs. pair (a,b) → 3 terms"""
    return t - t.transpose(0,1,2, 5,4,3) - t.transpose(0,1,2, 3,5,4)

def P_a_cb(t):
    """P(a/cb): a vs. pair (c,b) → 3 terms"""
    return t - t.transpose(0,1,2, 5,4,3) - t.transpose(0,1,2, 4,3,5)

def P_ba_c(t):
    """P(ba/c): pair (b,a) vs. c → 3 terms"""
    return t - t.transpose(0,1,2, 3,5,4) - t.transpose(0,1,2, 5,4,3)

def P_abc_full(t):
    """P(a/b/c): Full antisymmetrization of a,b,c → 6 terms"""
    return (t - t.transpose(0,1,2, 4,3,5) - t.transpose(0,1,2, 5,4,3)
              - t.transpose(0,1,2, 3,5,4) + t.transpose(0,1,2, 4,5,3)
              + t.transpose(0,1,2, 5,3,4))

def P_ijk_full(t):
    """P(i/j/k): Full antisymmetrization of i,j,k → 6 terms"""
    return (t - t.transpose(1,0,2, 3,4,5) - t.transpose(2,1,0, 3,4,5)
              - t.transpose(0,2,1, 3,4,5) + t.transpose(1,2,0, 3,4,5)
              + t.transpose(2,0,1, 3,4,5))

# ============================================================================
# T4 Permutations (4 indices)
# ============================================================================

def P_a_bcd(t):
    """
    P(a/bcd): a vs. already-antisymmetric triple (b,c,d) → 4 terms
    t has shape (i,j,k,l, a,b,c,d)
    """
    return (t 
            - t.transpose(0,1,2,3, 5,4,6,7)  # swap a↔b
            - t.transpose(0,1,2,3, 6,5,4,7)  # swap a↔c  
            - t.transpose(0,1,2,3, 7,5,6,4))  # swap a↔d

def P_i_jkl(t):
    """
    P(i/jkl): i vs. already-antisymmetric triple (j,k,l) → 4 terms
    """
    return (t 
            - t.transpose(1,0,2,3, 4,5,6,7)  # swap i↔j
            - t.transpose(2,1,0,3, 4,5,6,7)  # swap i↔k
            - t.transpose(3,1,2,0, 4,5,6,7))  # swap i↔l

def P_ab_cd(t):
    """
    P(ab/cd): pair (a,b) vs. pair (c,d), both internally antisymmetric → 3 terms
    """
    return (t 
            - t.transpose(0,1,2,3, 6,7,4,5)  # swap pairs (a,b)↔(c,d)
            - t.transpose(0,1,2,3, 4,6,5,7))  # alternative swap pattern

def P_ij_kl(t):
    """
    P(ij/kl): pair (i,j) vs. pair (k,l), both internally antisymmetric → 3 terms
    """
    return (t 
            - t.transpose(2,3,0,1, 4,5,6,7)  # swap pairs (i,j)↔(k,l)
            - t.transpose(0,2,1,3, 4,5,6,7))  # alternative swap pattern

def P_abc_d(t):
    """
    P(abc/d): triple (a,b,c) vs. d, where (a,b,c) is already antisymmetric → 4 terms
    """
    return (t 
            - t.transpose(0,1,2,3, 7,5,6,4)  # swap a↔d
            - t.transpose(0,1,2,3, 4,7,6,5)  # swap b↔d
            - t.transpose(0,1,2,3, 4,5,7,6))  # swap c↔d

def P_ijk_l(t):
    """
    P(ijk/l): triple (i,j,k) vs. l, where (i,j,k) is already antisymmetric → 4 terms
    """
    return (t 
            - t.transpose(3,1,2,0, 4,5,6,7)  # swap i↔l
            - t.transpose(0,3,2,1, 4,5,6,7)  # swap j↔l
            - t.transpose(0,1,3,2, 4,5,6,7))  # swap k↔l

def P_a_b_cd(t):
    """
    P(a/b/cd): Full antisymmetrization of a,b with pair (c,d) → 12 terms
    This is a compound permutation
    """
    # First antisymmetrize (c,d) if not already done
    # Then antisymmetrize a,b fully with the pair
    result = t.copy()
    # All permutations of a,b
    for perm_a, perm_b in [(4,5), (5,4)]:  # swap a and b
        indices = list(range(8))
        indices[4] = perm_a
        indices[5] = perm_b
        result -= t.transpose(0,1,2,3, *indices[4:])
    
    # Permutations involving c,d
    for perm_a, perm_c in [(4,6), (6,4)]:
        indices = [0,1,2,3,4,5,6,7]
        indices[4] = perm_a
        indices[6] = perm_c
        result -= t.transpose(*indices)
    
    for perm_a, perm_d in [(4,7), (7,4)]:
        indices = [0,1,2,3,4,5,6,7]
        indices[4] = perm_a
        indices[7] = perm_d
        result -= t.transpose(*indices)
    
    for perm_b, perm_c in [(5,6), (6,5)]:
        indices = [0,1,2,3,4,5,6,7]
        indices[5] = perm_b
        indices[6] = perm_c
        result -= t.transpose(*indices)
    
    for perm_b, perm_d in [(5,7), (7,5)]:
        indices = [0,1,2,3,4,5,6,7]
        indices[5] = perm_b
        indices[7] = perm_d
        result -= t.transpose(*indices)
    
    return result

def P_a_b_c_d(t):
    """
    P(a/b/c/d): Full antisymmetrization of all 4 virtual indices → 24 terms
    """
    from itertools import permutations
    result = np.zeros_like(t)
    
    # Generate all 24 permutations of (a,b,c,d) = (4,5,6,7) in tensor indices
    count = 0
    for perm_virt in permutations([4,5,6,7]):
        # Determine sign: number of transpositions
        sign = 1
        perm_list = list(perm_virt)
        # Count inversions to determine sign
        for i in range(4):
            for j in range(i+1, 4):
                if (perm_list[i] > perm_list[j]):
                    sign *= -1
        
        indices = [0,1,2,3] + list(perm_virt)
        result += sign * t.transpose(*indices)
        count += 1
    
    return result

def P_i_j_k_l(t):
    """
    P(i/j/k/l): Full antisymmetrization of all 4 occupied indices → 24 terms
    """
    from itertools import permutations
    result = np.zeros_like(t)
    
    # Generate all 24 permutations of (i,j,k,l) = (0,1,2,3) in tensor indices
    for perm_occ in permutations([0,1,2,3]):
        # Determine sign: number of transpositions
        sign = 1
        perm_list = list(perm_occ)
        # Count inversions to determine sign
        for i in range(4):
            for j in range(i+1, 4):
                if (perm_list[i] > perm_list[j]):
                    sign *= -1
        
        indices = list(perm_occ) + [4,5,6,7]
        result += sign * t.transpose(*indices)
    
    return result

# Additional compound permutations used in equations

def P_a_c_bd(t):
    """P(a/c/bd): a and c with pair (b,d)"""
    return (t 
            - t.transpose(0,1,2,3, 6,5,4,7)  # swap a↔c
            - t.transpose(0,1,2,3, 5,4,6,7)  # swap a↔b
            - t.transpose(0,1,2,3, 7,5,6,4))  # swap a↔d

def P_ac_bd(t):
    """P(ac/bd): pair (a,c) vs pair (b,d)"""
    return (t 
            - t.transpose(0,1,2,3, 5,7,6,4)  # one type of swap
            - t.transpose(0,1,2,3, 4,6,5,7))  # another swap

def P_i_jk_l(t):
    """P(i/jk/l): i, pair (j,k), and l"""
    return (t 
            - t.transpose(1,0,2,3, 4,5,6,7)  # swap i↔j
            - t.transpose(2,1,0,3, 4,5,6,7)  # swap i↔k
            - t.transpose(3,1,2,0, 4,5,6,7))  # swap i↔l

# Occupied index counterparts
def P_i_j_kl(t):
    """P(i/j/kl): i and j with pair (k,l)"""
    return (t 
            - t.transpose(1,0,2,3, 4,5,6,7)  # swap i↔j
            - t.transpose(2,1,0,3, 4,5,6,7)  # swap i↔k
            - t.transpose(3,1,2,0, 4,5,6,7))  # swap i↔l

def P_ik_jl(t):
    """P(ik/jl): pair (i,k) vs. pair (j,l)"""
    return (t 
            - t.transpose(1,3,2,0, 4,5,6,7)  # one swap  
            - t.transpose(2,1,0,3, 4,5,6,7))  # another swap

def P_ij_k_l(t):
    """P(ij/k/l): pair (i,j), k, and l separately"""
    return (t 
            - t.transpose(2,1,0,3, 4,5,6,7)  # swap i↔k
            - t.transpose(3,1,2,0, 4,5,6,7)  # swap i↔l
            - t.transpose(0,2,1,3, 4,5,6,7))  # swap j↔k

def P_a_cb_d(t):
    """P(a/cb/d): a, pair (c,b), and d"""
    return (t 
            - t.transpose(0,1,2,3, 6,5,4,7 )  # swap a↔c
            - t.transpose(0,1,2,3, 5,6,4,7)  # swap a↔b
            - t.transpose(0,1,2,3, 7,5,6,4))  # swap a↔d

def P_ba_cd(t):
    """P(ba/cd): pair (b,a) vs. pair (c,d)"""
    return (t 
            - t.transpose(0,1,2,3, 6,7,4,5)  # swap pairs
            - t.transpose(0,1,2,3, 5,6,7,4))  # alternative

def P_a_bc_d(t):
    """P(a/bc/d): a, pair (b,c), and d"""  
    return (t 
            - t.transpose(0,1,2,3, 5,4,6,7)  # swap a↔b
            - t.transpose(0,1,2,3, 6,5,4,7)  # swap a↔c
            - t.transpose(0,1,2,3, 7,5,6,4))  # swap a↔d

def P_b_acd(t):
    """P(b/acd): b vs. triple (a,c,d)"""
    return (t 
            - t.transpose(0,1,2,3, 5,4,6,7)  # swap b↔a
            - t.transpose(0,1,2,3, 4,6,5,7)  # swap b↔c
            - t.transpose(0,1,2,3, 4,7,6,5))  # swap b↔d

def P_c_abd(t):
    """P(c/abd): c vs. triple (a,b,d)"""
    return (t 
            - t.transpose(0,1,2,3, 6,5,4,7)  # swap c↔a
            - t.transpose(0,1,2,3, 4,6,5,7)  # swap c↔b
            - t.transpose(0,1,2,3, 4,5,7,6))  # swap c↔d

def P_ab_c_d(t):
    """P(ab/c/d): pair (a,b), c, and d"""
    return (t 
            - t.transpose(0,1,2,3, 6,5,4,7)  # swap a↔c
            - t.transpose(0,1,2,3, 7,5,6,4)  # swap a↔d
            - t.transpose(0,1,2,3, 4,6,5,7))  # swap b↔c

def P_a_cdb(t):
    """P(a/cdb): a vs. triple (c,d,b)"""
    return (t 
            - t.transpose(0,1,2,3, 6,5,4,7)  # swap a↔c
            - t.transpose(0,1,2,3, 7,6,5,4)  # swap a↔d
            - t.transpose(0,1,2,3, 5,4,6,7))  # swap a↔b

# Analogous occupied permutations
def P_j_ikl(t):
    """P(j/ikl): j vs. triple (i,k,l)"""
    return (t 
            - t.transpose(1,0,2,3, 4,5,6,7)  # swap j↔i
            - t.transpose(2,1,0,3, 4,5,6,7)  # swap j↔k
            - t.transpose(3,1,2,0, 4,5,6,7))  # swap j↔l

def P_ji_kl(t):
    """P(ji/kl): pair (j,i) vs. pair (k,l)"""
    return (t 
            - t.transpose(2,3,0,1, 4,5,6,7)  # swap pairs
            - t.transpose(1,2,3,0, 4,5,6,7))  # alternative

def P_ki_jl(t):
    """P(ki/jl): pair (k,i) vs. pair (j,l)"""
    return (t 
            - t.transpose(1,3,2,0, 4,5,6,7)  # one swap
            - t.transpose(2,1,0,3, 4,5,6,7))  # another

def P_i_klj(t):
    """P(i/klj): i vs. triple (k,l,j)"""
    return (t 
            - t.transpose(2,1,0,3, 4,5,6,7)  # swap i↔k
            - t.transpose(3,1,2,0, 4,5,6,7)  # swap i↔l
            - t.transpose(1,0,2,3, 4,5,6,7))  # swap i↔j

def P_ab_cd_simple(t):
    """
    Simplified P(ab/cd): Just swap the two pairs
    For some equations this is the correct interpretation
    """
    return t - t.transpose(0,1,2,3, 6,7,4,5)

def P_ij_kl_simple(t):
    """
    Simplified P(ij/kl): Just swap the two pairs  
    """
    return t - t.transpose(2,3,0,1, 4,5,6,7)

# ============================================================================
# Aliases for compatibility with generated code
# ============================================================================
P_i_j = P_ij
P_a_b = P_ab
P_a_b_c = P_abc_full
P_i_j_k = P_ijk_full
P_ji_k = P_ij_k
P_i_kj = P_i_jk
P_a_d_bc = P_a_bc_d
P_ca_bd = P_ac_bd
