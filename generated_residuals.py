"""
Complete CCSDTQ Residual Functions
Auto-generated from ccsdtq_equations.tex
"""

import numpy as np
from opt_einsum import contract


def compute_ccsdtq_t1_residual_COMPLETE(f, Gamma, t1, t2, t3, t4, o, v):
    """T1 residual with ALL 44 equations from lines 13-56"""
    r1 = f[v,o].T.copy()  # Base term

    # Line 13: \begin{eqnarray}-V^{jk}_{bc}t^{b}_{i}t^{a}_{j}t^{c}_{k}\end{...
    term = -1.0 * contract('jkbc,ib,ja,kc->ia', Gamma[o,o,v,v], t1, t1, t1)
    r1 += term

    # Line 16: \begin{eqnarray}-f^{j}_{b}t^{b}_{i}t^{a}_{j}\end{eqnarray}
...
    term = -1.0 * contract('jb,ib,ja->ia', f[o,v], t1, t1)
    r1 += term

    # Line 19: \begin{eqnarray}V^{aj}_{bc}t^{b}_{i}t^{c}_{j}\end{eqnarray}
...
    term = 1.0 * contract('ajbc,ib,jc->ia', Gamma[v,o,v,v], t1, t1)
    r1 += term

    # Line 22: \begin{eqnarray}-V^{jk}_{ib}t^{a}_{j}t^{b}_{k}\end{eqnarray}...
    term = -1.0 * contract('jkib,ja,kb->ia', Gamma[o,o,o,v], t1, t1)
    r1 += term

    # Line 25: \begin{eqnarray}V^{jk}_{bc}t^{b}_{j}t^{ca}_{ki}\end{eqnarray...
    term = 1.0 * contract('jkbc,jb,kica->ia', Gamma[o,o,v,v], t1, t2)
    r1 += term

    # Line 28: \begin{eqnarray}\frac{1}{2}V^{jk}_{bc}t^{b}_{i}t^{ca}_{jk}\e...
    term = 0.5 * contract('jkbc,ib,jkca->ia', Gamma[o,o,v,v], t1, t2)
    r1 += term

    # Line 31: \begin{eqnarray}\frac{1}{2}V^{jk}_{bc}t^{a}_{j}t^{bc}_{ki}\e...
    term = 0.5 * contract('jkbc,ja,kibc->ia', Gamma[o,o,v,v], t1, t2)
    r1 += term

    # Line 34: \begin{eqnarray}f^{a}_{b}t^{b}_{i}\end{eqnarray}
...
    term = 1.0 * contract('ab,ib->ia', f[v,v], t1)
    r1 += term

    # Line 37: \begin{eqnarray}-f^{j}_{i}t^{a}_{j}\end{eqnarray}
...
    term = -1.0 * contract('ji,ja->ia', f[o,o], t1)
    r1 += term

    # Line 40: \begin{eqnarray}V^{aj}_{ib}t^{b}_{j}\end{eqnarray}
...
    term = 1.0 * contract('ajib,jb->ia', Gamma[v,o,o,v], t1)
    r1 += term

    # Line 43: \begin{eqnarray}f^{j}_{b}t^{ab}_{ij}\end{eqnarray}
...
    term = 1.0 * contract('jb,ijab->ia', f[o,v], t2)
    r1 += term

    # Line 46: \begin{eqnarray}\frac{1}{2}V^{aj}_{bc}t^{bc}_{ij}\end{eqnarr...
    term = 0.5 * contract('ajbc,ijbc->ia', Gamma[v,o,v,v], t2)
    r1 += term

    # Line 49: \begin{eqnarray}-\frac{1}{2}V^{jk}_{ib}t^{ab}_{jk}\end{eqnar...
    term = -0.5 * contract('jkib,jkab->ia', Gamma[o,o,o,v], t2)
    r1 += term

    # Line 52: \begin{eqnarray}\frac{1}{4}V^{jk}_{bc}t^{abc}_{ijk}\end{eqna...
    term = 0.25 * contract('jkbc,ijkabc->ia', Gamma[o,o,v,v], t3)
    r1 += term

    # Line 55: \begin{eqnarray}f^{a}_{i}\end{eqnarray}
...
    term = 1.0 * contract('ai->ia', f[v,o])
    r1 += term

    return r1


def compute_ccsdtq_t2_residual_COMPLETE(f, Gamma, t1, t2, t3, t4, o, v):
    """T2 residual with ALL 113 equations from lines 60-172"""
    r2 = Gamma[o,o,v,v].copy()  # Base term

    # Line 60: \begin{eqnarray}\frac{1}{4}P(a/b)P(i/j)V^{kl}_{cd}t^{c}_{i}t...
    term = 0.25 * contract('klcd,ic,jd,ka,lb->ijab', Gamma[o,o,v,v], t1, t1, t1, t1)
    r2 += P_a_b(P_i_j(term))

    # Line 63: \begin{eqnarray}-\frac{1}{2}P(a/b)P(i/j)V^{ak}_{cd}t^{c}_{i}...
    term = -0.5 * contract('akcd,ic,jd,kb->ijab', Gamma[v,o,v,v], t1, t1, t1)
    r2 += P_a_b(P_i_j(term))

    # Line 66: \begin{eqnarray}\frac{1}{2}P(a/b)P(i/j)V^{kl}_{ic}t^{c}_{j}t...
    term = 0.5 * contract('klic,jc,ka,lb->ijab', Gamma[o,o,o,v], t1, t1, t1)
    r2 += P_a_b(P_i_j(term))

    # Line 69: \begin{eqnarray}P(i/j)V^{kl}_{cd}t^{c}_{i}t^{d}_{k}t^{ab}_{l...
    term = 1.0 * contract('klcd,ic,kd,ljab->ijab', Gamma[o,o,v,v], t1, t1, t2)
    r2 += P_i_j(term)

    # Line 72: \begin{eqnarray}P(a/b)V^{kl}_{cd}t^{a}_{k}t^{c}_{l}t^{db}_{i...
    term = 1.0 * contract('klcd,ka,lc,ijdb->ijab', Gamma[o,o,v,v], t1, t1, t2)
    r2 += P_a_b(term)

    # Line 75: \begin{eqnarray}\frac{1}{4}P(i/j)V^{kl}_{cd}t^{c}_{i}t^{d}_{...
    term = 0.25 * contract('klcd,ic,jd,klab->ijab', Gamma[o,o,v,v], t1, t1, t2)
    r2 += P_i_j(term)

    # Line 78: \begin{eqnarray}-P(a/b)P(i/j)V^{kl}_{cd}t^{c}_{i}t^{a}_{k}t^...
    term = -1.0 * contract('klcd,ic,ka,ljdb->ijab', Gamma[o,o,v,v], t1, t1, t2)
    r2 += P_a_b(P_i_j(term))

    # Line 81: \begin{eqnarray}\frac{1}{4}P(a/b)V^{kl}_{cd}t^{a}_{k}t^{b}_{...
    term = 0.25 * contract('klcd,ka,lb,ijcd->ijab', Gamma[o,o,v,v], t1, t1, t2)
    r2 += P_a_b(term)

    # Line 84: \begin{eqnarray}\frac{1}{2}P(i/j)V^{ab}_{cd}t^{c}_{i}t^{d}_{...
    term = 0.5 * contract('abcd,ic,jd->ijab', Gamma[v,v,v,v], t1, t1)
    r2 += P_i_j(term)

    # Line 87: \begin{eqnarray}\frac{1}{2}P(a/b)V^{kl}_{ij}t^{a}_{k}t^{b}_{...
    term = 0.5 * contract('klij,ka,lb->ijab', Gamma[o,o,o,o], t1, t1)
    r2 += P_a_b(term)

    # Line 90: \begin{eqnarray}-P(a/b)P(i/j)V^{ak}_{ic}t^{c}_{j}t^{b}_{k}\e...
    term = -1.0 * contract('akic,jc,kb->ijab', Gamma[v,o,o,v], t1, t1)
    r2 += P_a_b(P_i_j(term))

    # Line 93: \begin{eqnarray}-P(i/j)f^{k}_{c}t^{c}_{i}t^{ab}_{kj}\end{eqn...
    term = -1.0 * contract('kc,ic,kjab->ijab', f[o,v], t1, t2)
    r2 += P_i_j(term)

    # Line 96: \begin{eqnarray}-P(a/b)f^{k}_{c}t^{a}_{k}t^{cb}_{ij}\end{eqn...
    term = -1.0 * contract('kc,ka,ijcb->ijab', f[o,v], t1, t2)
    r2 += P_a_b(term)

    # Line 99: \begin{eqnarray}-P(a/b)V^{ak}_{cd}t^{c}_{k}t^{db}_{ij}\end{e...
    term = -1.0 * contract('akcd,kc,ijdb->ijab', Gamma[v,o,v,v], t1, t2)
    r2 += P_a_b(term)

    # Line 102: \begin{eqnarray}P(a/b)P(i/j)V^{ak}_{cd}t^{c}_{i}t^{db}_{kj}\...
    term = 1.0 * contract('akcd,ic,kjdb->ijab', Gamma[v,o,v,v], t1, t2)
    r2 += P_a_b(P_i_j(term))

    # Line 105: \begin{eqnarray}-\frac{1}{2}P(a/b)V^{ak}_{cd}t^{b}_{k}t^{cd}...
    term = -0.5 * contract('akcd,kb,ijcd->ijab', Gamma[v,o,v,v], t1, t2)
    r2 += P_a_b(term)

    # Line 108: \begin{eqnarray}P(i/j)V^{kl}_{ic}t^{c}_{k}t^{ab}_{lj}\end{eq...
    term = 1.0 * contract('klic,kc,ljab->ijab', Gamma[o,o,o,v], t1, t2)
    r2 += P_i_j(term)

    # Line 111: \begin{eqnarray}\frac{1}{2}P(i/j)V^{kl}_{ic}t^{c}_{j}t^{ab}_...
    term = 0.5 * contract('klic,jc,klab->ijab', Gamma[o,o,o,v], t1, t2)
    r2 += P_i_j(term)

    # Line 114: \begin{eqnarray}-P(a/b)P(i/j)V^{kl}_{ic}t^{a}_{k}t^{cb}_{lj}...
    term = -1.0 * contract('klic,ka,ljcb->ijab', Gamma[o,o,o,v], t1, t2)
    r2 += P_a_b(P_i_j(term))

    # Line 117: \begin{eqnarray}V^{kl}_{cd}t^{c}_{k}t^{dab}_{lij}\end{eqnarr...
    term = 1.0 * contract('klcd,kc,lijdab->ijab', Gamma[o,o,v,v], t1, t3)
    r2 += term

    # Line 120: \begin{eqnarray}\frac{1}{2}P(i/j)V^{kl}_{cd}t^{c}_{i}t^{dab}...
    term = 0.5 * contract('klcd,ic,kljdab->ijab', Gamma[o,o,v,v], t1, t3)
    r2 += P_i_j(term)

    # Line 123: \begin{eqnarray}\frac{1}{2}P(a/b)V^{kl}_{cd}t^{a}_{k}t^{cdb}...
    term = 0.5 * contract('klcd,ka,lijcdb->ijab', Gamma[o,o,v,v], t1, t3)
    r2 += P_a_b(term)

    # Line 126: \begin{eqnarray}P(i/j)V^{ab}_{ic}t^{c}_{j}\end{eqnarray}
...
    term = 1.0 * contract('abic,jc->ijab', Gamma[v,v,o,v], t1)
    r2 += P_i_j(term)

    # Line 129: \begin{eqnarray}-P(a/b)V^{ak}_{ij}t^{b}_{k}\end{eqnarray}
...
    term = -1.0 * contract('akij,kb->ijab', Gamma[v,o,o,o], t1)
    r2 += P_a_b(term)

    # Line 132: \begin{eqnarray}\frac{1}{2}P(i/j)V^{kl}_{cd}t^{ab}_{ik}t^{cd...
    term = 0.5 * contract('klcd,ikab,ljcd->ijab', Gamma[o,o,v,v], t2, t2)
    r2 += P_i_j(term)

    # Line 135: \begin{eqnarray}\frac{1}{2}P(a/b)V^{kl}_{cd}t^{ac}_{ij}t^{db...
    term = 0.5 * contract('klcd,ijac,kldb->ijab', Gamma[o,o,v,v], t2, t2)
    r2 += P_a_b(term)

    # Line 138: \begin{eqnarray}\frac{1}{2}P(a/b)P(i/j)V^{kl}_{cd}t^{ac}_{ik...
    term = 0.5 * contract('klcd,ikac,ljdb->ijab', Gamma[o,o,v,v], t2, t2)
    r2 += P_a_b(P_i_j(term))

    # Line 141: \begin{eqnarray}\frac{1}{4}V^{kl}_{cd}t^{cd}_{ij}t^{ab}_{kl}...
    term = 0.25 * contract('klcd,ijcd,klab->ijab', Gamma[o,o,v,v], t2, t2)
    r2 += term

    # Line 144: \begin{eqnarray}P(a/b)f^{a}_{c}t^{bc}_{ji}\end{eqnarray}
...
    term = 1.0 * contract('ac,jibc->ijab', f[v,v], t2)
    r2 += P_a_b(term)

    # Line 147: \begin{eqnarray}-P(i/j)f^{k}_{i}t^{ba}_{jk}\end{eqnarray}
...
    term = -1.0 * contract('ki,jkba->ijab', f[o,o], t2)
    r2 += P_i_j(term)

    # Line 150: \begin{eqnarray}\frac{1}{2}V^{ab}_{cd}t^{cd}_{ij}\end{eqnarr...
    term = 0.5 * contract('abcd,ijcd->ijab', Gamma[v,v,v,v], t2)
    r2 += term

    # Line 153: \begin{eqnarray}\frac{1}{2}V^{kl}_{ij}t^{ab}_{kl}\end{eqnarr...
    term = 0.5 * contract('klij,klab->ijab', Gamma[o,o,o,o], t2)
    r2 += term

    # Line 156: \begin{eqnarray}P(a/b)P(i/j)V^{ak}_{ic}t^{bc}_{jk}\end{eqnar...
    term = 1.0 * contract('akic,jkbc->ijab', Gamma[v,o,o,v], t2)
    r2 += P_a_b(P_i_j(term))

    # Line 159: \begin{eqnarray}f^{k}_{c}t^{abc}_{ijk}\end{eqnarray}
...
    term = 1.0 * contract('kc,ijkabc->ijab', f[o,v], t3)
    r2 += term

    # Line 162: \begin{eqnarray}\frac{1}{2}P(a/b)V^{ak}_{cd}t^{bcd}_{jik}\en...
    term = 0.5 * contract('akcd,jikbcd->ijab', Gamma[v,o,v,v], t3)
    r2 += P_a_b(term)

    # Line 165: \begin{eqnarray}-\frac{1}{2}P(i/j)V^{kl}_{ic}t^{bac}_{jkl}\e...
    term = -0.5 * contract('klic,jklbac->ijab', Gamma[o,o,o,v], t3)
    r2 += P_i_j(term)

    # Line 168: \begin{eqnarray}\frac{1}{4}V^{kl}_{cd}t^{abcd}_{ijkl}\end{eq...
    term = 0.25 * contract('klcd,ijklabcd->ijab', Gamma[o,o,v,v], t4)
    r2 += term

    # Line 171: \begin{eqnarray}V^{ab}_{ij}\end{eqnarray}
...
    term = 1.0 * contract('abij->ijab', Gamma[v,v,o,o])
    r2 += term

    return r2


def compute_ccsdtq_t3_residual_COMPLETE(f, Gamma, t1, t2, t3, t4, o, v):
    """T3 residual with ALL 158 equations from lines 177-334"""
    r3 = np.zeros_like(t3)

    # Line 177: \begin{eqnarray}\frac{1}{2}P(a/bc)P(i/j/k)V^{lm}_{de}t^{d}_{...
    term = 0.5 * contract('lmde,id,je,la,mkbc->ijkabc', Gamma[o,o,v,v], t1, t1, t1, t2)
    r3 += P_a_bc(P_i_j_k(term))

    # Line 180: \begin{eqnarray}\frac{1}{2}P(a/b/c)P(i/jk)V^{lm}_{de}t^{d}_{...
    term = 0.5 * contract('lmde,id,la,mb,jkec->ijkabc', Gamma[o,o,v,v], t1, t1, t1, t2)
    r3 += P_a_b_c(P_i_jk(term))

    # Line 183: \begin{eqnarray}-\frac{1}{2}P(a/bc)P(i/j/k)V^{al}_{de}t^{d}_...
    term = -0.5 * contract('alde,id,je,lkbc->ijkabc', Gamma[v,o,v,v], t1, t1, t2)
    r3 += P_a_bc(P_i_j_k(term))

    # Line 186: \begin{eqnarray}-P(a/b/c)P(i/jk)V^{al}_{de}t^{d}_{i}t^{b}_{l...
    term = -1.0 * contract('alde,id,lb,jkec->ijkabc', Gamma[v,o,v,v], t1, t1, t2)
    r3 += P_a_b_c(P_i_jk(term))

    # Line 189: \begin{eqnarray}\frac{1}{2}P(a/b/c)P(i/jk)V^{lm}_{id}t^{a}_{...
    term = 0.5 * contract('lmid,la,mb,jkdc->ijkabc', Gamma[o,o,o,v], t1, t1, t2)
    r3 += P_a_b_c(P_i_jk(term))

    # Line 192: \begin{eqnarray}P(a/bc)P(i/j/k)V^{lm}_{id}t^{d}_{j}t^{a}_{l}...
    term = 1.0 * contract('lmid,jd,la,mkbc->ijkabc', Gamma[o,o,o,v], t1, t1, t2)
    r3 += P_a_bc(P_i_j_k(term))

    # Line 195: \begin{eqnarray}P(i/jk)V^{lm}_{de}t^{d}_{i}t^{e}_{l}t^{abc}_...
    term = 1.0 * contract('lmde,id,le,mjkabc->ijkabc', Gamma[o,o,v,v], t1, t1, t3)
    r3 += P_i_jk(term)

    # Line 198: \begin{eqnarray}P(a/bc)V^{lm}_{de}t^{a}_{l}t^{d}_{m}t^{ebc}_...
    term = 1.0 * contract('lmde,la,md,ijkebc->ijkabc', Gamma[o,o,v,v], t1, t1, t3)
    r3 += P_a_bc(term)

    # Line 201: \begin{eqnarray}\frac{1}{4}P(i/j/k)V^{lm}_{de}t^{d}_{i}t^{e}...
    term = 0.25 * contract('lmde,id,je,lmkabc->ijkabc', Gamma[o,o,v,v], t1, t1, t3)
    r3 += P_i_j_k(term)

    # Line 204: \begin{eqnarray}-P(a/bc)P(i/jk)V^{lm}_{de}t^{d}_{i}t^{a}_{l}...
    term = -1.0 * contract('lmde,id,la,mjkebc->ijkabc', Gamma[o,o,v,v], t1, t1, t3)
    r3 += P_a_bc(P_i_jk(term))

    # Line 207: \begin{eqnarray}\frac{1}{4}P(a/b/c)V^{lm}_{de}t^{a}_{l}t^{b}...
    term = 0.25 * contract('lmde,la,mb,ijkdec->ijkabc', Gamma[o,o,v,v], t1, t1, t3)
    r3 += P_a_b_c(term)

    # Line 210: \begin{eqnarray}-P(b/ac)P(ij/k)V^{lm}_{de}t^{d}_{l}t^{eb}_{i...
    term = -1.0 * contract('lmde,ld,ijeb,mkac->ijkabc', Gamma[o,o,v,v], t1, t2, t2)
    r3 += P_b_ac(P_ij_k(term))

    # Line 213: \begin{eqnarray}-P(a/b/c)P(ij/k)V^{lm}_{de}t^{a}_{l}t^{db}_{...
    term = -1.0 * contract('lmde,la,ijdb,mkec->ijkabc', Gamma[o,o,v,v], t1, t2, t2)
    r3 += P_a_b_c(P_ij_k(term))

    # Line 216: \begin{eqnarray}-P(ab/c)P(i/j/k)V^{lm}_{de}t^{d}_{i}t^{ab}_{...
    term = -1.0 * contract('lmde,id,ljab,mkec->ijkabc', Gamma[o,o,v,v], t1, t2, t2)
    r3 += P_ab_c(P_i_j_k(term))

    # Line 219: \begin{eqnarray}\frac{1}{2}P(a/bc)P(ij/k)V^{lm}_{de}t^{a}_{l...
    term = 0.5 * contract('lmde,la,ijde,mkbc->ijkabc', Gamma[o,o,v,v], t1, t2, t2)
    r3 += P_a_bc(P_ij_k(term))

    # Line 222: \begin{eqnarray}\frac{1}{2}P(c/ab)P(i/jk)V^{lm}_{de}t^{d}_{i...
    term = 0.5 * contract('lmde,id,jkec,lmab->ijkabc', Gamma[o,o,v,v], t1, t2, t2)
    r3 += P_c_ab(P_i_jk(term))

    # Line 225: \begin{eqnarray}P(ab/c)P(i/jk)V^{ab}_{de}t^{d}_{i}t^{ec}_{jk...
    term = 1.0 * contract('abde,id,jkec->ijkabc', Gamma[v,v,v,v], t1, t2)
    r3 += P_ab_c(P_i_jk(term))

    # Line 228: \begin{eqnarray}P(a/bc)P(ij/k)V^{lm}_{ij}t^{a}_{l}t^{bc}_{mk...
    term = 1.0 * contract('lmij,la,mkbc->ijkabc', Gamma[o,o,o,o], t1, t2)
    r3 += P_a_bc(P_ij_k(term))

    # Line 231: \begin{eqnarray}-P(a/bc)P(i/j/k)V^{al}_{id}t^{d}_{j}t^{bc}_{...
    term = -1.0 * contract('alid,jd,lkbc->ijkabc', Gamma[v,o,o,v], t1, t2)
    r3 += P_a_bc(P_i_j_k(term))

    # Line 234: \begin{eqnarray}-P(a/b/c)P(i/jk)V^{al}_{id}t^{b}_{l}t^{dc}_{...
    term = -1.0 * contract('alid,lb,jkdc->ijkabc', Gamma[v,o,o,v], t1, t2)
    r3 += P_a_b_c(P_i_jk(term))

    # Line 237: \begin{eqnarray}-P(i/jk)f^{l}_{d}t^{d}_{i}t^{abc}_{ljk}\end{...
    term = -1.0 * contract('ld,id,ljkabc->ijkabc', f[o,v], t1, t3)
    r3 += P_i_jk(term)

    # Line 240: \begin{eqnarray}-P(a/bc)f^{l}_{d}t^{a}_{l}t^{dbc}_{ijk}\end{...
    term = -1.0 * contract('ld,la,ijkdbc->ijkabc', f[o,v], t1, t3)
    r3 += P_a_bc(term)

    # Line 243: \begin{eqnarray}-P(a/bc)V^{al}_{de}t^{d}_{l}t^{ebc}_{ijk}\en...
    term = -1.0 * contract('alde,ld,ijkebc->ijkabc', Gamma[v,o,v,v], t1, t3)
    r3 += P_a_bc(term)

    # Line 246: \begin{eqnarray}P(a/bc)P(i/jk)V^{al}_{de}t^{d}_{i}t^{ebc}_{l...
    term = 1.0 * contract('alde,id,ljkebc->ijkabc', Gamma[v,o,v,v], t1, t3)
    r3 += P_a_bc(P_i_jk(term))

    # Line 249: \begin{eqnarray}-\frac{1}{2}P(a/b/c)V^{al}_{de}t^{b}_{l}t^{d...
    term = -0.5 * contract('alde,lb,ijkdec->ijkabc', Gamma[v,o,v,v], t1, t3)
    r3 += P_a_b_c(term)

    # Line 252: \begin{eqnarray}P(i/jk)V^{lm}_{id}t^{d}_{l}t^{abc}_{mjk}\end...
    term = 1.0 * contract('lmid,ld,mjkabc->ijkabc', Gamma[o,o,o,v], t1, t3)
    r3 += P_i_jk(term)

    # Line 255: \begin{eqnarray}\frac{1}{2}P(i/j/k)V^{lm}_{id}t^{d}_{j}t^{ab...
    term = 0.5 * contract('lmid,jd,lmkabc->ijkabc', Gamma[o,o,o,v], t1, t3)
    r3 += P_i_j_k(term)

    # Line 258: \begin{eqnarray}-P(a/bc)P(i/jk)V^{lm}_{id}t^{a}_{l}t^{dbc}_{...
    term = -1.0 * contract('lmid,la,mjkdbc->ijkabc', Gamma[o,o,o,v], t1, t3)
    r3 += P_a_bc(P_i_jk(term))

    # Line 261: \begin{eqnarray}V^{lm}_{de}t^{d}_{l}t^{eabc}_{mijk}\end{eqna...
    term = 1.0 * contract('lmde,ld,mijkeabc->ijkabc', Gamma[o,o,v,v], t1, t4)
    r3 += term

    # Line 264: \begin{eqnarray}\frac{1}{2}P(i/jk)V^{lm}_{de}t^{d}_{i}t^{eab...
    term = 0.5 * contract('lmde,id,lmjkeabc->ijkabc', Gamma[o,o,v,v], t1, t4)
    r3 += P_i_jk(term)

    # Line 267: \begin{eqnarray}\frac{1}{2}P(a/bc)V^{lm}_{de}t^{a}_{l}t^{deb...
    term = 0.5 * contract('lmde,la,mijkdebc->ijkabc', Gamma[o,o,v,v], t1, t4)
    r3 += P_a_bc(term)

    # Line 270: \begin{eqnarray}-P(a/bc)P(ij/k)f^{l}_{d}t^{ad}_{ij}t^{bc}_{l...
    term = -1.0 * contract('ld,ijad,lkbc->ijkabc', f[o,v], t2, t2)
    r3 += P_a_bc(P_ij_k(term))

    # Line 273: \begin{eqnarray}P(a/b/c)P(ji/k)V^{al}_{de}t^{bd}_{ji}t^{ec}_...
    term = 1.0 * contract('alde,jibd,lkec->ijkabc', Gamma[v,o,v,v], t2, t2)
    r3 += P_a_b_c(P_ji_k(term))

    # Line 276: \begin{eqnarray}-\frac{1}{2}P(a/bc)P(ij/k)V^{al}_{de}t^{de}_...
    term = -0.5 * contract('alde,ijde,lkbc->ijkabc', Gamma[v,o,v,v], t2, t2)
    r3 += P_a_bc(P_ij_k(term))

    # Line 279: \begin{eqnarray}-P(ba/c)P(i/j/k)V^{lm}_{id}t^{ba}_{jl}t^{dc}...
    term = -1.0 * contract('lmid,jlba,mkdc->ijkabc', Gamma[o,o,o,v], t2, t2)
    r3 += P_ba_c(P_i_j_k(term))

    # Line 282: \begin{eqnarray}\frac{1}{2}P(b/ac)P(i/jk)V^{lm}_{id}t^{bd}_{...
    term = 0.5 * contract('lmid,jkbd,lmac->ijkabc', Gamma[o,o,o,v], t2, t2)
    r3 += P_b_ac(P_i_jk(term))

    # Line 285: \begin{eqnarray}\frac{1}{2}P(i/jk)V^{lm}_{de}t^{de}_{il}t^{a...
    term = 0.5 * contract('lmde,ilde,mjkabc->ijkabc', Gamma[o,o,v,v], t2, t3)
    r3 += P_i_jk(term)

    # Line 288: \begin{eqnarray}\frac{1}{2}P(a/bc)V^{lm}_{de}t^{ad}_{lm}t^{e...
    term = 0.5 * contract('lmde,lmad,ijkebc->ijkabc', Gamma[o,o,v,v], t2, t3)
    r3 += P_a_bc(term)

    # Line 291: \begin{eqnarray}\frac{1}{4}P(ij/k)V^{lm}_{de}t^{de}_{ij}t^{a...
    term = 0.25 * contract('lmde,ijde,lmkabc->ijkabc', Gamma[o,o,v,v], t2, t3)
    r3 += P_ij_k(term)

    # Line 294: \begin{eqnarray}P(a/bc)P(i/jk)V^{lm}_{de}t^{ad}_{il}t^{ebc}_...
    term = 1.0 * contract('lmde,ilad,mjkebc->ijkabc', Gamma[o,o,v,v], t2, t3)
    r3 += P_a_bc(P_i_jk(term))

    # Line 297: \begin{eqnarray}\frac{1}{4}P(ab/c)V^{lm}_{de}t^{ab}_{lm}t^{d...
    term = 0.25 * contract('lmde,lmab,ijkdec->ijkabc', Gamma[o,o,v,v], t2, t3)
    r3 += P_ab_c(term)

    # Line 300: \begin{eqnarray}\frac{1}{2}P(a/bc)P(ij/k)V^{lm}_{de}t^{ad}_{...
    term = 0.5 * contract('lmde,ijad,lmkebc->ijkabc', Gamma[o,o,v,v], t2, t3)
    r3 += P_a_bc(P_ij_k(term))

    # Line 303: \begin{eqnarray}\frac{1}{2}P(ab/c)P(i/jk)V^{lm}_{de}t^{ab}_{...
    term = 0.5 * contract('lmde,ilab,mjkdec->ijkabc', Gamma[o,o,v,v], t2, t3)
    r3 += P_ab_c(P_i_jk(term))

    # Line 306: \begin{eqnarray}P(ab/c)P(i/kj)V^{ab}_{id}t^{cd}_{kj}\end{eqn...
    term = 1.0 * contract('abid,kjcd->ijkabc', Gamma[v,v,o,v], t2)
    r3 += P_ab_c(P_i_kj(term))

    # Line 309: \begin{eqnarray}-P(a/cb)P(ij/k)V^{al}_{ij}t^{cb}_{kl}\end{eq...
    term = -1.0 * contract('alij,klcb->ijkabc', Gamma[v,o,o,o], t2)
    r3 += P_a_cb(P_ij_k(term))

    # Line 312: \begin{eqnarray}P(a/bc)f^{a}_{d}t^{bcd}_{jki}\end{eqnarray}
...
    term = 1.0 * contract('ad,jkibcd->ijkabc', f[v,v], t3)
    r3 += P_a_bc(term)

    # Line 315: \begin{eqnarray}-P(i/jk)f^{l}_{i}t^{bca}_{jkl}\end{eqnarray}...
    term = -1.0 * contract('li,jklbca->ijkabc', f[o,o], t3)
    r3 += P_i_jk(term)

    # Line 318: \begin{eqnarray}\frac{1}{2}P(ab/c)V^{ab}_{de}t^{cde}_{kij}\e...
    term = 0.5 * contract('abde,kijcde->ijkabc', Gamma[v,v,v,v], t3)
    r3 += P_ab_c(term)

    # Line 321: \begin{eqnarray}\frac{1}{2}P(ij/k)V^{lm}_{ij}t^{cab}_{klm}\e...
    term = 0.5 * contract('lmij,klmcab->ijkabc', Gamma[o,o,o,o], t3)
    r3 += P_ij_k(term)

    # Line 324: \begin{eqnarray}P(a/bc)P(i/jk)V^{al}_{id}t^{bcd}_{jkl}\end{e...
    term = 1.0 * contract('alid,jklbcd->ijkabc', Gamma[v,o,o,v], t3)
    r3 += P_a_bc(P_i_jk(term))

    # Line 327: \begin{eqnarray}f^{l}_{d}t^{abcd}_{ijkl}\end{eqnarray}
...
    term = 1.0 * contract('ld,ijklabcd->ijkabc', f[o,v], t4)
    r3 += term

    # Line 330: \begin{eqnarray}\frac{1}{2}P(a/bc)V^{al}_{de}t^{bcde}_{jkil}...
    term = 0.5 * contract('alde,jkilbcde->ijkabc', Gamma[v,o,v,v], t4)
    r3 += P_a_bc(term)

    # Line 333: \begin{eqnarray}-\frac{1}{2}P(i/jk)V^{lm}_{id}t^{bcad}_{jklm...
    term = -0.5 * contract('lmid,jklmbcad->ijkabc', Gamma[o,o,o,v], t4)
    r3 += P_i_jk(term)

    return r3


def compute_ccsdtq_t4_residual_COMPLETE(f, Gamma, t1, t2, t3, t4, o, v):
    """T4 residual with ALL 222 equations from lines 339-560"""
    r4 = np.zeros_like(t4)

    # Line 339: \begin{eqnarray}\frac{1}{2}P(a/bcd)P(i/j/kl)V^{mn}_{ef}t^{e}...
    term = 0.5 * contract('mnef,ie,jf,ma,nklbcd->ijklabcd', Gamma[o,o,v,v], t1, t1, t1, t3)
    r4 += P_a_bcd(P_i_j_kl(term))

    # Line 342: \begin{eqnarray}\frac{1}{2}P(a/b/cd)P(i/jkl)V^{mn}_{ef}t^{e}...
    term = 0.5 * contract('mnef,ie,ma,nb,jklfcd->ijklabcd', Gamma[o,o,v,v], t1, t1, t1, t3)
    r4 += P_a_b_cd(P_i_jkl(term))

    # Line 345: \begin{eqnarray}P(a/c/bd)P(i/jk/l)V^{mn}_{ef}t^{e}_{i}t^{a}_...
    term = 1.0 * contract('mnef,ie,ma,jkfc,nlbd->ijklabcd', Gamma[o,o,v,v], t1, t1, t2, t2)
    r4 += P_a_c_bd(P_i_jk_l(term))

    # Line 348: \begin{eqnarray}\frac{1}{4}P(ac/bd)P(i/j/k/l)V^{mn}_{ef}t^{e...
    term = 0.25 * contract('mnef,ie,jf,mkac,nlbd->ijklabcd', Gamma[o,o,v,v], t1, t1, t2, t2)
    r4 += P_ac_bd(P_i_j_k_l(term))

    # Line 351: \begin{eqnarray}\frac{1}{4}P(a/b/c/d)P(ik/jl)V^{mn}_{ef}t^{a...
    term = 0.25 * contract('mnef,ma,nb,ikec,jlfd->ijklabcd', Gamma[o,o,v,v], t1, t1, t2, t2)
    r4 += P_a_b_c_d(P_ik_jl(term))

    # Line 354: \begin{eqnarray}-\frac{1}{2}P(a/bcd)P(i/j/kl)V^{am}_{ef}t^{e...
    term = -0.5 * contract('amef,ie,jf,mklbcd->ijklabcd', Gamma[v,o,v,v], t1, t1, t3)
    r4 += P_a_bcd(P_i_j_kl(term))

    # Line 357: \begin{eqnarray}-P(a/b/cd)P(i/jkl)V^{am}_{ef}t^{e}_{i}t^{b}_...
    term = -1.0 * contract('amef,ie,mb,jklfcd->ijklabcd', Gamma[v,o,v,v], t1, t1, t3)
    r4 += P_a_b_cd(P_i_jkl(term))

    # Line 360: \begin{eqnarray}\frac{1}{2}P(a/b/cd)P(i/jkl)V^{mn}_{ie}t^{a}...
    term = 0.5 * contract('mnie,ma,nb,jklecd->ijklabcd', Gamma[o,o,o,v], t1, t1, t3)
    r4 += P_a_b_cd(P_i_jkl(term))

    # Line 363: \begin{eqnarray}P(a/bcd)P(i/j/kl)V^{mn}_{ie}t^{e}_{j}t^{a}_{...
    term = 1.0 * contract('mnie,je,ma,nklbcd->ijklabcd', Gamma[o,o,o,v], t1, t1, t3)
    r4 += P_a_bcd(P_i_j_kl(term))

    # Line 366: \begin{eqnarray}P(i/jkl)V^{mn}_{ef}t^{e}_{i}t^{f}_{m}t^{abcd...
    term = 1.0 * contract('mnef,ie,mf,njklabcd->ijklabcd', Gamma[o,o,v,v], t1, t1, t4)
    r4 += P_i_jkl(term)

    # Line 369: \begin{eqnarray}P(a/bcd)V^{mn}_{ef}t^{a}_{m}t^{e}_{n}t^{fbcd...
    term = 1.0 * contract('mnef,ma,ne,ijklfbcd->ijklabcd', Gamma[o,o,v,v], t1, t1, t4)
    r4 += P_a_bcd(term)

    # Line 372: \begin{eqnarray}\frac{1}{4}P(i/j/kl)V^{mn}_{ef}t^{e}_{i}t^{f...
    term = 0.25 * contract('mnef,ie,jf,mnklabcd->ijklabcd', Gamma[o,o,v,v], t1, t1, t4)
    r4 += P_i_j_kl(term)

    # Line 375: \begin{eqnarray}-P(a/bcd)P(i/jkl)V^{mn}_{ef}t^{e}_{i}t^{a}_{...
    term = -1.0 * contract('mnef,ie,ma,njklfbcd->ijklabcd', Gamma[o,o,v,v], t1, t1, t4)
    r4 += P_a_bcd(P_i_jkl(term))

    # Line 378: \begin{eqnarray}\frac{1}{4}P(a/b/cd)V^{mn}_{ef}t^{a}_{m}t^{b...
    term = 0.25 * contract('mnef,ma,nb,ijklefcd->ijklabcd', Gamma[o,o,v,v], t1, t1, t4)
    r4 += P_a_b_cd(term)

    # Line 381: \begin{eqnarray}-P(a/c/bd)P(i/jk/l)V^{am}_{ef}t^{e}_{i}t^{fc...
    term = -1.0 * contract('amef,ie,jkfc,mlbd->ijklabcd', Gamma[v,o,v,v], t1, t2, t2)
    r4 += P_a_c_bd(P_i_jk_l(term))

    # Line 384: \begin{eqnarray}-\frac{1}{2}P(a/b/c/d)P(ik/jl)V^{am}_{ef}t^{...
    term = -0.5 * contract('amef,mb,ikec,jlfd->ijklabcd', Gamma[v,o,v,v], t1, t2, t2)
    r4 += P_a_b_c_d(P_ik_jl(term))

    # Line 387: \begin{eqnarray}\frac{1}{2}P(ac/bd)P(i/j/k/l)V^{mn}_{ie}t^{e...
    term = 0.5 * contract('mnie,je,mkac,nlbd->ijklabcd', Gamma[o,o,o,v], t1, t2, t2)
    r4 += P_ac_bd(P_i_j_k_l(term))

    # Line 390: \begin{eqnarray}P(a/c/bd)P(i/jk/l)V^{mn}_{ie}t^{a}_{m}t^{ec}...
    term = 1.0 * contract('mnie,ma,jkec,nlbd->ijklabcd', Gamma[o,o,o,v], t1, t2, t2)
    r4 += P_a_c_bd(P_i_jk_l(term))

    # Line 393: \begin{eqnarray}-P(b/acd)P(ij/kl)V^{mn}_{ef}t^{e}_{m}t^{fb}_...
    term = -1.0 * contract('mnef,me,ijfb,nklacd->ijklabcd', Gamma[o,o,v,v], t1, t2, t3)
    r4 += P_b_acd(P_ij_kl(term))

    # Line 396: \begin{eqnarray}-P(ba/cd)P(i/jkl)V^{mn}_{ef}t^{e}_{m}t^{ba}_...
    term = -1.0 * contract('mnef,me,niba,jklfcd->ijklabcd', Gamma[o,o,v,v], t1, t2, t3)
    r4 += P_ba_cd(P_i_jkl(term))

    # Line 399: \begin{eqnarray}P(a/b/cd)P(j/ikl)V^{mn}_{ef}t^{a}_{m}t^{eb}_...
    term = 1.0 * contract('mnef,ma,njeb,iklfcd->ijklabcd', Gamma[o,o,v,v], t1, t2, t3)
    r4 += P_a_b_cd(P_j_ikl(term))

    # Line 402: \begin{eqnarray}P(b/acd)P(i/j/kl)V^{mn}_{ef}t^{e}_{i}t^{fb}_...
    term = 1.0 * contract('mnef,ie,mjfb,nklacd->ijklabcd', Gamma[o,o,v,v], t1, t2, t3)
    r4 += P_b_acd(P_i_j_kl(term))

    # Line 405: \begin{eqnarray}\frac{1}{2}P(a/bcd)P(ij/kl)V^{mn}_{ef}t^{a}_...
    term = 0.5 * contract('mnef,ma,ijef,nklbcd->ijklabcd', Gamma[o,o,v,v], t1, t2, t3)
    r4 += P_a_bcd(P_ij_kl(term))

    # Line 408: \begin{eqnarray}\frac{1}{2}P(ab/cd)P(i/jkl)V^{mn}_{ef}t^{e}_...
    term = 0.5 * contract('mnef,ie,mnab,jklfcd->ijklabcd', Gamma[o,o,v,v], t1, t2, t3)
    r4 += P_ab_cd(P_i_jkl(term))

    # Line 411: \begin{eqnarray}\frac{1}{2}P(c/abd)P(i/jk/l)V^{mn}_{ef}t^{e}...
    term = 0.5 * contract('mnef,ie,jkfc,mnlabd->ijklabcd', Gamma[o,o,v,v], t1, t2, t3)
    r4 += P_c_abd(P_i_jk_l(term))

    # Line 414: \begin{eqnarray}-P(a/b/cd)P(ij/kl)V^{mn}_{ef}t^{a}_{m}t^{eb}...
    term = -1.0 * contract('mnef,ma,ijeb,nklfcd->ijklabcd', Gamma[o,o,v,v], t1, t2, t3)
    r4 += P_a_b_cd(P_ij_kl(term))

    # Line 417: \begin{eqnarray}-P(ab/cd)P(i/j/kl)V^{mn}_{ef}t^{e}_{i}t^{ab}...
    term = -1.0 * contract('mnef,ie,mjab,nklfcd->ijklabcd', Gamma[o,o,v,v], t1, t2, t3)
    r4 += P_ab_cd(P_i_j_kl(term))

    # Line 420: \begin{eqnarray}\frac{1}{2}P(a/cb/d)P(j/ikl)V^{mn}_{ef}t^{a}...
    term = 0.5 * contract('mnef,ma,njcb,iklefd->ijklabcd', Gamma[o,o,v,v], t1, t2, t3)
    r4 += P_a_cb_d(P_j_ikl(term))

    # Line 423: \begin{eqnarray}P(ab/cd)P(i/jkl)V^{ab}_{ef}t^{e}_{i}t^{fcd}_...
    term = 1.0 * contract('abef,ie,jklfcd->ijklabcd', Gamma[v,v,v,v], t1, t3)
    r4 += P_ab_cd(P_i_jkl(term))

    # Line 426: \begin{eqnarray}P(a/bcd)P(ij/kl)V^{mn}_{ij}t^{a}_{m}t^{bcd}_...
    term = 1.0 * contract('mnij,ma,nklbcd->ijklabcd', Gamma[o,o,o,o], t1, t3)
    r4 += P_a_bcd(P_ij_kl(term))

    # Line 429: \begin{eqnarray}-P(a/bcd)P(i/j/kl)V^{am}_{ie}t^{e}_{j}t^{bcd...
    term = -1.0 * contract('amie,je,mklbcd->ijklabcd', Gamma[v,o,o,v], t1, t3)
    r4 += P_a_bcd(P_i_j_kl(term))

    # Line 432: \begin{eqnarray}-P(a/b/cd)P(i/jkl)V^{am}_{ie}t^{b}_{m}t^{ecd...
    term = -1.0 * contract('amie,mb,jklecd->ijklabcd', Gamma[v,o,o,v], t1, t3)
    r4 += P_a_b_cd(P_i_jkl(term))

    # Line 435: \begin{eqnarray}-P(i/jkl)f^{m}_{e}t^{e}_{i}t^{abcd}_{mjkl}\{...
    term = -1.0 * contract('me,ie,mjklabcd->ijklabcd', f[o,v], t1, t4)
    r4 += P_i_jkl(term)

    # Line 438: \begin{eqnarray}-P(a/bcd)f^{m}_{e}t^{a}_{m}t^{ebcd}_{ijkl}\{...
    term = -1.0 * contract('me,ma,ijklebcd->ijklabcd', f[o,v], t1, t4)
    r4 += P_a_bcd(term)

    # Line 441: \begin{eqnarray}-P(a/bcd)V^{am}_{ef}t^{e}_{m}t^{fbcd}_{ijkl}...
    term = -1.0 * contract('amef,me,ijklfbcd->ijklabcd', Gamma[v,o,v,v], t1, t4)
    r4 += P_a_bcd(term)

    # Line 444: \begin{eqnarray}P(a/bcd)P(i/jkl)V^{am}_{ef}t^{e}_{i}t^{fbcd}...
    term = 1.0 * contract('amef,ie,mjklfbcd->ijklabcd', Gamma[v,o,v,v], t1, t4)
    r4 += P_a_bcd(P_i_jkl(term))

    # Line 447: \begin{eqnarray}-\frac{1}{2}P(a/b/cd)V^{am}_{ef}t^{b}_{m}t^{...
    term = -0.5 * contract('amef,mb,ijklefcd->ijklabcd', Gamma[v,o,v,v], t1, t4)
    r4 += P_a_b_cd(term)

    # Line 450: \begin{eqnarray}P(i/jkl)V^{mn}_{ie}t^{e}_{m}t^{abcd}_{njkl}\...
    term = 1.0 * contract('mnie,me,njklabcd->ijklabcd', Gamma[o,o,o,v], t1, t4)
    r4 += P_i_jkl(term)

    # Line 453: \begin{eqnarray}\frac{1}{2}P(i/j/kl)V^{mn}_{ie}t^{e}_{j}t^{a...
    term = 0.5 * contract('mnie,je,mnklabcd->ijklabcd', Gamma[o,o,o,v], t1, t4)
    r4 += P_i_j_kl(term)

    # Line 456: \begin{eqnarray}-P(a/bcd)P(i/jkl)V^{mn}_{ie}t^{a}_{m}t^{ebcd...
    term = -1.0 * contract('mnie,ma,njklebcd->ijklabcd', Gamma[o,o,o,v], t1, t4)
    r4 += P_a_bcd(P_i_jkl(term))

    # Line 459: \begin{eqnarray}\frac{1}{4}P(ac/bd)P(ij/k/l)V^{mn}_{ef}t^{ef...
    term = 0.25 * contract('mnef,ijef,mkac,nlbd->ijklabcd', Gamma[o,o,v,v], t2, t2, t2)
    r4 += P_ac_bd(P_ij_k_l(term))

    # Line 462: \begin{eqnarray}-P(a/bc/d)P(ij/k/l)V^{mn}_{ef}t^{ae}_{ij}t^{...
    term = -1.0 * contract('mnef,ijae,mkbc,nlfd->ijklabcd', Gamma[o,o,v,v], t2, t2, t2)
    r4 += P_a_bc_d(P_ij_k_l(term))

    # Line 465: \begin{eqnarray}\frac{1}{4}P(a/d/bc)P(ij/kl)V^{mn}_{ef}t^{ae...
    term = 0.25 * contract('mnef,ijae,klfd,mnbc->ijklabcd', Gamma[o,o,v,v], t2, t2, t2)
    r4 += P_a_d_bc(P_ij_kl(term))

    # Line 468: \begin{eqnarray}\frac{1}{2}P(ab/c/d)P(ki/jl)V^{ab}_{ef}t^{ce...
    term = 0.5 * contract('abef,kice,jlfd->ijklabcd', Gamma[v,v,v,v], t2, t2)
    r4 += P_ab_c_d(P_ki_jl(term))

    # Line 471: \begin{eqnarray}\frac{1}{2}P(ca/bd)P(ij/k/l)V^{mn}_{ij}t^{ca...
    term = 0.5 * contract('mnij,kmca,nlbd->ijklabcd', Gamma[o,o,o,o], t2, t2)
    r4 += P_ca_bd(P_ij_k_l(term))

    # Line 474: \begin{eqnarray}-P(a/b/cd)P(i/jk/l)V^{am}_{ie}t^{be}_{jk}t^{...
    term = -1.0 * contract('amie,jkbe,mlcd->ijklabcd', Gamma[v,o,o,v], t2, t2)
    r4 += P_a_b_cd(P_i_jk_l(term))

    # Line 477: \begin{eqnarray}-P(a/bcd)P(ij/kl)f^{m}_{e}t^{ae}_{ij}t^{bcd}...
    term = -1.0 * contract('me,ijae,mklbcd->ijklabcd', f[o,v], t2, t3)
    r4 += P_a_bcd(P_ij_kl(term))

    # Line 480: \begin{eqnarray}-P(ab/cd)P(i/jkl)f^{m}_{e}t^{ab}_{im}t^{ecd}...
    term = -1.0 * contract('me,imab,jklecd->ijklabcd', f[o,v], t2, t3)
    r4 += P_ab_cd(P_i_jkl(term))

    # Line 483: \begin{eqnarray}-P(a/b/cd)P(j/ikl)V^{am}_{ef}t^{be}_{jm}t^{f...
    term = -1.0 * contract('amef,jmbe,iklfcd->ijklabcd', Gamma[v,o,v,v], t2, t3)
    r4 += P_a_b_cd(P_j_ikl(term))

    # Line 486: \begin{eqnarray}-\frac{1}{2}P(a/bcd)P(ij/kl)V^{am}_{ef}t^{ef...
    term = -0.5 * contract('amef,ijef,mklbcd->ijklabcd', Gamma[v,o,v,v], t2, t3)
    r4 += P_a_bcd(P_ij_kl(term))

    # Line 489: \begin{eqnarray}P(a/b/cd)P(ji/kl)V^{am}_{ef}t^{be}_{ji}t^{fc...
    term = 1.0 * contract('amef,jibe,mklfcd->ijklabcd', Gamma[v,o,v,v], t2, t3)
    r4 += P_a_b_cd(P_ji_kl(term))

    # Line 492: \begin{eqnarray}-\frac{1}{2}P(a/bc/d)P(j/ikl)V^{am}_{ef}t^{b...
    term = -0.5 * contract('amef,jmbc,iklefd->ijklabcd', Gamma[v,o,v,v], t2, t3)
    r4 += P_a_bc_d(P_j_ikl(term))

    # Line 495: \begin{eqnarray}P(b/acd)P(i/j/kl)V^{mn}_{ie}t^{be}_{jm}t^{ac...
    term = 1.0 * contract('mnie,jmbe,nklacd->ijklabcd', Gamma[o,o,o,v], t2, t3)
    r4 += P_b_acd(P_i_j_kl(term))

    # Line 498: \begin{eqnarray}\frac{1}{2}P(ab/cd)P(i/jkl)V^{mn}_{ie}t^{ab}...
    term = 0.5 * contract('mnie,mnab,jklecd->ijklabcd', Gamma[o,o,o,v], t2, t3)
    r4 += P_ab_cd(P_i_jkl(term))

    # Line 501: \begin{eqnarray}\frac{1}{2}P(b/acd)P(i/jk/l)V^{mn}_{ie}t^{be...
    term = 0.5 * contract('mnie,jkbe,mnlacd->ijklabcd', Gamma[o,o,o,v], t2, t3)
    r4 += P_b_acd(P_i_jk_l(term))

    # Line 504: \begin{eqnarray}-P(ba/cd)P(i/j/kl)V^{mn}_{ie}t^{ba}_{jm}t^{e...
    term = -1.0 * contract('mnie,jmba,nklecd->ijklabcd', Gamma[o,o,o,v], t2, t3)
    r4 += P_ba_cd(P_i_j_kl(term))

    # Line 507: \begin{eqnarray}\frac{1}{2}P(i/jkl)V^{mn}_{ef}t^{ef}_{im}t^{...
    term = 0.5 * contract('mnef,imef,njklabcd->ijklabcd', Gamma[o,o,v,v], t2, t4)
    r4 += P_i_jkl(term)

    # Line 510: \begin{eqnarray}\frac{1}{2}P(a/bcd)V^{mn}_{ef}t^{ae}_{mn}t^{...
    term = 0.5 * contract('mnef,mnae,ijklfbcd->ijklabcd', Gamma[o,o,v,v], t2, t4)
    r4 += P_a_bcd(term)

    # Line 513: \begin{eqnarray}\frac{1}{4}P(ij/kl)V^{mn}_{ef}t^{ef}_{ij}t^{...
    term = 0.25 * contract('mnef,ijef,mnklabcd->ijklabcd', Gamma[o,o,v,v], t2, t4)
    r4 += P_ij_kl(term)

    # Line 516: \begin{eqnarray}P(a/bcd)P(i/jkl)V^{mn}_{ef}t^{ae}_{im}t^{fbc...
    term = 1.0 * contract('mnef,imae,njklfbcd->ijklabcd', Gamma[o,o,v,v], t2, t4)
    r4 += P_a_bcd(P_i_jkl(term))

    # Line 519: \begin{eqnarray}\frac{1}{4}P(ab/cd)V^{mn}_{ef}t^{ab}_{mn}t^{...
    term = 0.25 * contract('mnef,mnab,ijklefcd->ijklabcd', Gamma[o,o,v,v], t2, t4)
    r4 += P_ab_cd(term)

    # Line 522: \begin{eqnarray}\frac{1}{2}P(a/bcd)P(ij/kl)V^{mn}_{ef}t^{ae}...
    term = 0.5 * contract('mnef,ijae,mnklfbcd->ijklabcd', Gamma[o,o,v,v], t2, t4)
    r4 += P_a_bcd(P_ij_kl(term))

    # Line 525: \begin{eqnarray}\frac{1}{2}P(ab/cd)P(i/jkl)V^{mn}_{ef}t^{ab}...
    term = 0.5 * contract('mnef,imab,njklefcd->ijklabcd', Gamma[o,o,v,v], t2, t4)
    r4 += P_ab_cd(P_i_jkl(term))

    # Line 528: \begin{eqnarray}\frac{1}{2}P(abc/d)P(ij/kl)V^{mn}_{ef}t^{abc...
    term = 0.5 * contract('mnef,ijmabc,nklefd->ijklabcd', Gamma[o,o,v,v], t3, t3)
    r4 += P_abc_d(P_ij_kl(term))

    # Line 531: \begin{eqnarray}\frac{1}{2}P(ab/cd)P(ijk/l)V^{mn}_{ef}t^{abe...
    term = 0.5 * contract('mnef,ijkabe,mnlfcd->ijklabcd', Gamma[o,o,v,v], t3, t3)
    r4 += P_ab_cd(P_ijk_l(term))

    # Line 534: \begin{eqnarray}\frac{1}{2}P(ab/cd)P(ij/kl)V^{mn}_{ef}t^{abe...
    term = 0.5 * contract('mnef,ijmabe,nklfcd->ijklabcd', Gamma[o,o,v,v], t3, t3)
    r4 += P_ab_cd(P_ij_kl(term))

    # Line 537: \begin{eqnarray}\frac{1}{4}P(a/bcd)P(ijk/l)V^{mn}_{ef}t^{aef...
    term = 0.25 * contract('mnef,ijkaef,mnlbcd->ijklabcd', Gamma[o,o,v,v], t3, t3)
    r4 += P_a_bcd(P_ijk_l(term))

    # Line 540: \begin{eqnarray}P(ab/cd)P(i/klj)V^{ab}_{ie}t^{cde}_{klj}\{a^...
    term = 1.0 * contract('abie,kljcde->ijklabcd', Gamma[v,v,o,v], t3)
    r4 += P_ab_cd(P_i_klj(term))

    # Line 543: \begin{eqnarray}-P(a/cdb)P(ij/kl)V^{am}_{ij}t^{cdb}_{klm}\{a...
    term = -1.0 * contract('amij,klmcdb->ijklabcd', Gamma[v,o,o,o], t3)
    r4 += P_a_cdb(P_ij_kl(term))

    # Line 546: \begin{eqnarray}P(a/bcd)f^{a}_{e}t^{bcde}_{jkli}\{a^\dagger_...
    term = 1.0 * contract('ae,jklibcde->ijklabcd', f[v,v], t4)
    r4 += P_a_bcd(term)

    # Line 549: \begin{eqnarray}-P(i/jkl)f^{m}_{i}t^{bcda}_{jklm}\{a^\dagger...
    term = -1.0 * contract('mi,jklmbcda->ijklabcd', f[o,o], t4)
    r4 += P_i_jkl(term)

    # Line 552: \begin{eqnarray}\frac{1}{2}P(ab/cd)V^{ab}_{ef}t^{cdef}_{klij...
    term = 0.5 * contract('abef,klijcdef->ijklabcd', Gamma[v,v,v,v], t4)
    r4 += P_ab_cd(term)

    # Line 555: \begin{eqnarray}\frac{1}{2}P(ij/kl)V^{mn}_{ij}t^{cdab}_{klmn...
    term = 0.5 * contract('mnij,klmncdab->ijklabcd', Gamma[o,o,o,o], t4)
    r4 += P_ij_kl(term)

    # Line 558: \begin{eqnarray}P(a/bcd)P(i/jkl)V^{am}_{ie}t^{bcde}_{jklm}\{...
    term = 1.0 * contract('amie,jklmbcde->ijklabcd', Gamma[v,o,o,v], t4)
    r4 += P_a_bcd(P_i_jkl(term))

    return r4
