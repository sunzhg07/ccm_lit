"""
CCDTQ Residuals (T1 = 0)
Auto-generated from ccsdtq_equations.tex
"""

import numpy as np
from opt_einsum import contract
from ccsdtq_permutations import *


def compute_ccdtq_t2_residual(f, Gamma, t2, t3, t4, o, v):
    """T2 residual (T1 terms removed)"""
    r2 = Gamma[o,o,v,v].copy()

    # Line 132
    term = 0.5 * contract('klcd,ikab,ljcd->ijab', Gamma[o,o,v,v], t2, t2)
    r2 += P_i_j(term)

    # Line 135
    term = 0.5 * contract('klcd,ijac,kldb->ijab', Gamma[o,o,v,v], t2, t2)
    r2 += P_a_b(term)

    # Line 138
    term = 0.5 * contract('klcd,ikac,ljdb->ijab', Gamma[o,o,v,v], t2, t2)
    r2 += P_a_b(P_i_j(term))

    # Line 141
    term = 0.25 * contract('klcd,ijcd,klab->ijab', Gamma[o,o,v,v], t2, t2)
    r2 += term

    # Line 144
    term = 1.0 * contract('ac,jibc->ijab', f[v,v], t2)
    r2 += P_a_b(term)

    # Line 147
    term = -1.0 * contract('ki,jkba->ijab', f[o,o], t2)
    r2 += P_i_j(term)

    # Line 150
    term = 0.5 * contract('abcd,ijcd->ijab', Gamma[v,v,v,v], t2)
    r2 += term

    # Line 153
    term = 0.5 * contract('klij,klab->ijab', Gamma[o,o,o,o], t2)
    r2 += term

    # Line 156
    term = 1.0 * contract('akic,jkbc->ijab', Gamma[v,o,o,v], t2)
    r2 += P_a_b(P_i_j(term))

    # Line 159
    term = 1.0 * contract('kc,ijkabc->ijab', f[o,v], t3)
    r2 += term

    # Line 162
    term = 0.5 * contract('akcd,jikbcd->ijab', Gamma[v,o,v,v], t3)
    r2 += P_a_b(term)

    # Line 165
    term = -0.5 * contract('klic,jklbac->ijab', Gamma[o,o,o,v], t3)
    r2 += P_i_j(term)

    # Line 168
    term = 0.25 * contract('klcd,ijklabcd->ijab', Gamma[o,o,v,v], t4)
    r2 += term

    # Line 171
    term = 1.0 * contract('abij->ijab', Gamma[v,v,o,o])
    r2 += term

    return r2


def compute_ccdtq_t3_residual(f, Gamma, t2, t3, t4, o, v):
    """T3 residual (T1 terms removed)"""
    r3 = np.zeros_like(t3)

    # Line 270
    term = -1.0 * contract('ld,ijad,lkbc->ijkabc', f[o,v], t2, t2)
    r3 += P_a_bc(P_ij_k(term))

    # Line 273
    term = 1.0 * contract('alde,jibd,lkec->ijkabc', Gamma[v,o,v,v], t2, t2)
    r3 += P_a_b_c(P_ji_k(term))

    # Line 276
    term = -0.5 * contract('alde,ijde,lkbc->ijkabc', Gamma[v,o,v,v], t2, t2)
    r3 += P_a_bc(P_ij_k(term))

    # Line 279
    term = -1.0 * contract('lmid,jlba,mkdc->ijkabc', Gamma[o,o,o,v], t2, t2)
    r3 += P_ba_c(P_i_j_k(term))

    # Line 282
    term = 0.5 * contract('lmid,jkbd,lmac->ijkabc', Gamma[o,o,o,v], t2, t2)
    r3 += P_b_ac(P_i_jk(term))

    # Line 285
    term = 0.5 * contract('lmde,ilde,mjkabc->ijkabc', Gamma[o,o,v,v], t2, t3)
    r3 += P_i_jk(term)

    # Line 288
    term = 0.5 * contract('lmde,lmad,ijkebc->ijkabc', Gamma[o,o,v,v], t2, t3)
    r3 += P_a_bc(term)

    # Line 291
    term = 0.25 * contract('lmde,ijde,lmkabc->ijkabc', Gamma[o,o,v,v], t2, t3)
    r3 += P_ij_k(term)

    # Line 294
    term = 1.0 * contract('lmde,ilad,mjkebc->ijkabc', Gamma[o,o,v,v], t2, t3)
    r3 += P_a_bc(P_i_jk(term))

    # Line 297
    term = 0.25 * contract('lmde,lmab,ijkdec->ijkabc', Gamma[o,o,v,v], t2, t3)
    r3 += P_ab_c(term)

    # Line 300
    term = 0.5 * contract('lmde,ijad,lmkebc->ijkabc', Gamma[o,o,v,v], t2, t3)
    r3 += P_a_bc(P_ij_k(term))

    # Line 303
    term = 0.5 * contract('lmde,ilab,mjkdec->ijkabc', Gamma[o,o,v,v], t2, t3)
    r3 += P_ab_c(P_i_jk(term))

    # Line 306
    term = 1.0 * contract('abid,kjcd->ijkabc', Gamma[v,v,o,v], t2)
    r3 += P_ab_c(P_i_kj(term))

    # Line 309
    term = -1.0 * contract('alij,klcb->ijkabc', Gamma[v,o,o,o], t2)
    r3 += P_a_cb(P_ij_k(term))

    # Line 312
    term = 1.0 * contract('ad,jkibcd->ijkabc', f[v,v], t3)
    r3 += P_a_bc(term)

    # Line 315
    term = -1.0 * contract('li,jklbca->ijkabc', f[o,o], t3)
    r3 += P_i_jk(term)

    # Line 318
    term = 0.5 * contract('abde,kijcde->ijkabc', Gamma[v,v,v,v], t3)
    r3 += P_ab_c(term)

    # Line 321
    term = 0.5 * contract('lmij,klmcab->ijkabc', Gamma[o,o,o,o], t3)
    r3 += P_ij_k(term)

    # Line 324
    term = 1.0 * contract('alid,jklbcd->ijkabc', Gamma[v,o,o,v], t3)
    r3 += P_a_bc(P_i_jk(term))

    # Line 327
    term = 1.0 * contract('ld,ijklabcd->ijkabc', f[o,v], t4)
    r3 += term

    # Line 330
    term = 0.5 * contract('alde,jkilbcde->ijkabc', Gamma[v,o,v,v], t4)
    r3 += P_a_bc(term)

    # Line 333
    term = -0.5 * contract('lmid,jklmbcad->ijkabc', Gamma[o,o,o,v], t4)
    r3 += P_i_jk(term)

    return r3


def compute_ccdtq_t4_residual(f, Gamma, t2, t3, t4, o, v):
    """T4 residual (T1 terms removed)"""
    r4 = np.zeros_like(t4)

    # Line 459
    term = 0.25 * contract('mnef,ijef,mkac,nlbd->ijklabcd', Gamma[o,o,v,v], t2, t2, t2)
    r4 += P_ac_bd(P_ij_k_l(term))

    # Line 462
    term = -1.0 * contract('mnef,ijae,mkbc,nlfd->ijklabcd', Gamma[o,o,v,v], t2, t2, t2)
    r4 += P_a_bc_d(P_ij_k_l(term))

    # Line 465
    term = 0.25 * contract('mnef,ijae,klfd,mnbc->ijklabcd', Gamma[o,o,v,v], t2, t2, t2)
    r4 += P_a_d_bc(P_ij_kl(term))

    # Line 468
    term = 0.5 * contract('abef,kice,jlfd->ijklabcd', Gamma[v,v,v,v], t2, t2)
    r4 += P_ab_c_d(P_ki_jl(term))

    # Line 471
    term = 0.5 * contract('mnij,kmca,nlbd->ijklabcd', Gamma[o,o,o,o], t2, t2)
    r4 += P_ca_bd(P_ij_k_l(term))

    # Line 474
    term = -1.0 * contract('amie,jkbe,mlcd->ijklabcd', Gamma[v,o,o,v], t2, t2)
    r4 += P_a_b_cd(P_i_jk_l(term))

    # Line 477
    term = -1.0 * contract('me,ijae,mklbcd->ijklabcd', f[o,v], t2, t3)
    r4 += P_a_bcd(P_ij_kl(term))

    # Line 480
    term = -1.0 * contract('me,imab,jklecd->ijklabcd', f[o,v], t2, t3)
    r4 += P_ab_cd(P_i_jkl(term))

    # Line 483
    term = -1.0 * contract('amef,jmbe,iklfcd->ijklabcd', Gamma[v,o,v,v], t2, t3)
    r4 += P_a_b_cd(P_j_ikl(term))

    # Line 486
    term = -0.5 * contract('amef,ijef,mklbcd->ijklabcd', Gamma[v,o,v,v], t2, t3)
    r4 += P_a_bcd(P_ij_kl(term))

    # Line 489
    term = 1.0 * contract('amef,jibe,mklfcd->ijklabcd', Gamma[v,o,v,v], t2, t3)
    r4 += P_a_b_cd(P_ji_kl(term))

    # Line 492
    term = -0.5 * contract('amef,jmbc,iklefd->ijklabcd', Gamma[v,o,v,v], t2, t3)
    r4 += P_a_bc_d(P_j_ikl(term))

    # Line 495
    term = 1.0 * contract('mnie,jmbe,nklacd->ijklabcd', Gamma[o,o,o,v], t2, t3)
    r4 += P_b_acd(P_i_j_kl(term))

    # Line 498
    term = 0.5 * contract('mnie,mnab,jklecd->ijklabcd', Gamma[o,o,o,v], t2, t3)
    r4 += P_ab_cd(P_i_jkl(term))

    # Line 501
    term = 0.5 * contract('mnie,jkbe,mnlacd->ijklabcd', Gamma[o,o,o,v], t2, t3)
    r4 += P_b_acd(P_i_jk_l(term))

    # Line 504
    term = -1.0 * contract('mnie,jmba,nklecd->ijklabcd', Gamma[o,o,o,v], t2, t3)
    r4 += P_ba_cd(P_i_j_kl(term))

    # Line 507
    term = 0.5 * contract('mnef,imef,njklabcd->ijklabcd', Gamma[o,o,v,v], t2, t4)
    r4 += P_i_jkl(term)

    # Line 510
    term = 0.5 * contract('mnef,mnae,ijklfbcd->ijklabcd', Gamma[o,o,v,v], t2, t4)
    r4 += P_a_bcd(term)

    # Line 513
    term = 0.25 * contract('mnef,ijef,mnklabcd->ijklabcd', Gamma[o,o,v,v], t2, t4)
    r4 += P_ij_kl(term)

    # Line 516
    term = 1.0 * contract('mnef,imae,njklfbcd->ijklabcd', Gamma[o,o,v,v], t2, t4)
    r4 += P_a_bcd(P_i_jkl(term))

    # Line 519
    term = 0.25 * contract('mnef,mnab,ijklefcd->ijklabcd', Gamma[o,o,v,v], t2, t4)
    r4 += P_ab_cd(term)

    # Line 522
    term = 0.5 * contract('mnef,ijae,mnklfbcd->ijklabcd', Gamma[o,o,v,v], t2, t4)
    r4 += P_a_bcd(P_ij_kl(term))

    # Line 525
    term = 0.5 * contract('mnef,imab,njklefcd->ijklabcd', Gamma[o,o,v,v], t2, t4)
    r4 += P_ab_cd(P_i_jkl(term))

    # Line 528
    term = 0.5 * contract('mnef,ijmabc,nklefd->ijklabcd', Gamma[o,o,v,v], t3, t3)
    r4 += P_abc_d(P_ij_kl(term))

    # Line 531
    term = 0.5 * contract('mnef,ijkabe,mnlfcd->ijklabcd', Gamma[o,o,v,v], t3, t3)
    r4 += P_ab_cd(P_ijk_l(term))

    # Line 534
    term = 0.5 * contract('mnef,ijmabe,nklfcd->ijklabcd', Gamma[o,o,v,v], t3, t3)
    r4 += P_ab_cd(P_ij_kl(term))

    # Line 537
    term = 0.25 * contract('mnef,ijkaef,mnlbcd->ijklabcd', Gamma[o,o,v,v], t3, t3)
    r4 += P_a_bcd(P_ijk_l(term))

    # Line 540
    term = 1.0 * contract('abie,kljcde->ijklabcd', Gamma[v,v,o,v], t3)
    r4 += P_ab_cd(P_i_klj(term))

    # Line 543
    term = -1.0 * contract('amij,klmcdb->ijklabcd', Gamma[v,o,o,o], t3)
    r4 += P_a_cdb(P_ij_kl(term))

    # Line 546
    term = 1.0 * contract('ae,jklibcde->ijklabcd', f[v,v], t4)
    r4 += P_a_bcd(term)

    # Line 549
    term = -1.0 * contract('mi,jklmbcda->ijklabcd', f[o,o], t4)
    r4 += P_i_jkl(term)

    # Line 552
    term = 0.5 * contract('abef,klijcdef->ijklabcd', Gamma[v,v,v,v], t4)
    r4 += P_ab_cd(term)

    # Line 555
    term = 0.5 * contract('mnij,klmncdab->ijklabcd', Gamma[o,o,o,o], t4)
    r4 += P_ij_kl(term)

    # Line 558
    term = 1.0 * contract('amie,jklmbcde->ijklabcd', Gamma[v,o,o,v], t4)
    r4 += P_a_bcd(P_i_jkl(term))

    return r4
