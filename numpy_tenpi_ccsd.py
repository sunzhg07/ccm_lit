#!/usr/bin/python
# -*- coding: utf-8 -*-

# CCSD_T6 for simple use with numpy

import numpy as np #numpy arrays
import sys


# returns a list of np.ndarrays ordered by growing rank: [energy, residual_vo, residual_vvoo, ...] except the scalars are returned as scalars.
def numpy_tenpi_ccsd(nocc: int, nvir: int, F: list, V: list, T: list, type_: type = np.complex128) -> list:
    
    F1 = F[0]
    F2 = F[1]
    F3 = F[2]
    F4 = F[3]
    V1 = V[0]
    V2 = V[1]
    V3 = V[2]
    V4 = V[3]
    V5 = V[4]
    V6 = V[5]
    V7 = V[6]
    V8 = V[7]
    V9 = V[8]
    T1 = T[0]
    T2 = T[1]
    
    # F1 = np.array([nocc, nocc], dtype=type_)     #       OO
    # F2 = np.array([nvir, nocc], dtype=type_)     #       VO
    # F3 = np.array([nocc, nvir], dtype=type_)     #       OV
    # F4 = np.array([nvir, nvir], dtype=type_)     #       VV
    # V1 = np.array([nocc, nocc, nocc, nocc], dtype=type_)     #       OOOO
    # V2 = np.array([nvir, nocc, nocc, nocc], dtype=type_)     #       VOOO
    # V3 = np.array([nvir, nvir, nocc, nocc], dtype=type_)     #       VVOO
    # V4 = np.array([nocc, nocc, nocc, nvir], dtype=type_)     #       OOOV
    # V5 = np.array([nvir, nocc, nocc, nvir], dtype=type_)     #       VOOV
    # V6 = np.array([nvir, nvir, nocc, nvir], dtype=type_)     #       VVOV
    # V7 = np.array([nocc, nocc, nvir, nvir], dtype=type_)     #       OOVV
    # V8 = np.array([nvir, nocc, nvir, nvir], dtype=type_)     #       VOVV
    # V9 = np.array([nvir, nvir, nvir, nvir], dtype=type_)     #       VVVV
    # T1 = np.array([nvir, nocc], dtype=type_)     #       VO
    # T2 = np.array([nvir, nvir, nocc, nocc], dtype=type_)     #       VVOO
    Z0 = np.zeros([1], dtype=type_)     #       scalar
    Z1 = np.zeros([nvir, nocc], dtype=type_)     #       VO
    Z2 = np.zeros([nvir, nvir, nocc, nocc], dtype=type_)     #       VVOO
    # E1 = np.array([nocc, nvir], dtype=type_)     #       OV
    # J1 = np.array([nocc, nvir, nvir, nocc], dtype=type_)     #       OVVO
    # M1 = np.array([nocc, nvir, nvir, nocc], dtype=type_)     #       OVVO
    # X1 = np.array([nocc, nocc, nvir, nocc], dtype=type_)     #       OOVO
    # A2 = np.array([nocc, nocc], dtype=type_)     #       OO
    # I3 = np.array([nvir, nvir], dtype=type_)     #       VV
    # E2 = np.array([nocc, nocc, nvir, nocc], dtype=type_)     #       OOVO
    # G2 = np.array([nocc, nocc], dtype=type_)     #       OO
    # A3 = np.array([nocc, nocc], dtype=type_)     #       OO
    # E3 = np.array([nvir, nvir], dtype=type_)     #       VV
    # M3 = np.array([nocc, nocc, nocc, nocc], dtype=type_)     #       OOOO
    # X3 = np.array([nocc, nocc, nocc, nocc], dtype=type_)     #       OOOO
    # Y3 = np.array([nocc, nvir], dtype=type_)     #       OV
    # A4 = np.array([nvir, nvir, nocc, nocc], dtype=type_)     #       VVOO
    # D4 = np.array([nvir, nvir, nocc, nocc], dtype=type_)     #       VVOO
    # E4 = np.array([nvir, nocc, nvir, nocc], dtype=type_)     #       VOVO
    # G4 = np.array([nvir, nocc, nocc, nocc], dtype=type_)     #       VOOO
    # I4 = np.array([nvir, nvir, nocc, nocc], dtype=type_)     #       VVOO
    # J4 = np.array([nvir, nocc, nocc, nocc], dtype=type_)     #       VOOO
    # M4 = np.array([nvir, nocc, nvir, nocc], dtype=type_)     #       VOVO
    # X4 = np.array([nvir, nocc, nocc, nocc], dtype=type_)     #       VOOO
    # Y4 = np.array([nvir, nvir, nocc, nocc], dtype=type_)     #       VVOO
    # M6 = np.array([nocc, nocc, nocc, nocc], dtype=type_)     #       OOOO
    # D5 = np.array([nvir, nvir], dtype=type_)     #       VV
    # E5 = np.array([nvir, nvir, nocc, nocc], dtype=type_)     #       VVOO
    # G5 = np.array([nvir, nvir, nocc, nocc], dtype=type_)     #       VVOO
    # I5 = np.array([nvir, nocc, nocc, nocc], dtype=type_)     #       VOOO
    # J5 = np.array([nocc, nvir, nocc, nocc], dtype=type_)     #       OVOO
    # M5 = np.array([nvir, nvir, nvir, nocc], dtype=type_)     #       VVVO
    # X5 = np.array([nvir, nvir, nocc, nocc], dtype=type_)     #       VVOO
    # Y5 = np.array([nvir, nvir, nocc, nocc], dtype=type_)     #       VVOO
    # A6 = np.array([nocc, nocc, nocc, nocc], dtype=type_)     #       OOOO
    # X10 = np.array([nocc, nocc, nocc, nocc], dtype=type_)     #       OOOO
    # A11 = np.array([nocc, nocc, nocc, nocc], dtype=type_)     #       OOOO
    # X6 = np.array([nocc, nvir, nocc, nocc], dtype=type_)     #       OVOO
    # E10 = np.array([nvir, nocc, nocc, nocc], dtype=type_)     #       VOOO
    # A7 = np.array([nvir, nvir, nocc, nocc], dtype=type_)     #       VVOO
    # M11 = np.array([nvir, nocc, nocc, nocc], dtype=type_)     #       VOOO
    # G7 = np.array([nvir, nvir, nocc, nocc], dtype=type_)     #       VVOO
    # J7 = np.array([nocc, nocc], dtype=type_)     #       OO
    # M7 = np.array([nvir, nvir, nocc, nocc], dtype=type_)     #       VVOO
    # X7 = np.array([nvir, nvir, nocc, nocc], dtype=type_)     #       VVOO
    # Y7 = np.array([nocc, nocc], dtype=type_)     #       OO
    # A8 = np.array([nvir, nvir, nocc, nocc], dtype=type_)     #       VVOO
    # D8 = np.array([nocc, nocc, nocc, nocc], dtype=type_)     #       OOOO
    # J8 = np.array([nvir, nocc, nocc, nocc], dtype=type_)     #       VOOO
    # I10 = np.array([nocc, nocc, nocc, nocc], dtype=type_)     #       OOOO
    
    print('    Running code generated by tenpi.   ')
    
    #Contraction 1; Tree Level  0; Scaling  4/ 4 Result_size  0/ 0
    Z0 += 0.25 * np.einsum('ijab,abij->',V7,T2, optimize='optimal')
    
    E1 = np.zeros([nocc, nvir], dtype=type_)
    
    #Contraction 2; Tree Level  2; Scaling  3/ 3 Result_size  1/ 1
    E1 += np.einsum('ijab,ai->jb',V7,T1, optimize='optimal')
    
    #Contraction 3; Tree Level  2; Scaling  1/ 1 Result_size  1/ 1
    E1 += 2.0 * np.einsum('jb->jb',F3)
    
    #Contraction 4; Tree Level  1; Scaling  2/ 2 Result_size  0/ 0
    Z0 += 0.5 * np.einsum('bj,jb->',T1,E1, optimize='optimal')
    
    del E1
    
    #Contraction 5; Tree Level  0; Scaling  3/ 3 Result_size  1/ 1
    Z1 += np.einsum('jb,abij->ai',F3,T2, optimize='optimal')
    
    J1 = np.zeros([nocc, nvir, nvir, nocc], dtype=type_)
    
    #Contraction 6; Tree Level  3; Scaling  4/ 4 Result_size  2/ 2
    J1 += np.einsum('klcd,dblj->kbcj',V7,T2, optimize='optimal')
    
    M1 = np.zeros([nocc, nvir, nvir, nocc], dtype=type_)
    
    #Contraction 7; Tree Level  2; Scaling  2/ 2 Result_size  2/ 2
    M1 += np.einsum('jabi->jabi',J1)
    
    #Contraction 8; Tree Level  2; Scaling  2/ 2 Result_size  2/ 2
    M1 += np.einsum('ajib->jabi',V5)
    
    X1 = np.zeros([nocc, nocc, nvir, nocc], dtype=type_)
    
    #Contraction 9; Tree Level  4; Scaling  3/ 3 Result_size  3/ 1
    X1 += np.einsum('jkbc,bi->jkci',V7,T1, optimize='optimal')
    
    #Contraction 10; Tree Level  1; Scaling  5/ 3 Result_size  1/ 1
    Z1 += -0.5 * np.einsum('jkib,abjk->ai',V4,T2, optimize='optimal')
    
    A2 = np.zeros([nocc, nocc], dtype=type_)
    
    #Contraction 11; Tree Level  3; Scaling  4/ 4 Result_size  2/ 0
    A2 += np.einsum('jkbc,bcki->ji',V7,T2, optimize='optimal')
    
    I3 = np.zeros([nvir, nvir], dtype=type_)
    
    #Contraction 12; Tree Level  2; Scaling  2/ 4 Result_size  0/ 2
    I3 += 2.0 * np.einsum('ajbc,cj->ab',V8,T1, optimize='optimal')
    
    E2 = np.zeros([nocc, nocc, nvir, nocc], dtype=type_)
    
    #Contraction 13; Tree Level  3; Scaling  3/ 1 Result_size  3/ 1
    E2 += np.einsum('jkci->jkci',X1)
    
    #Contraction 14; Tree Level  3; Scaling  3/ 1 Result_size  3/ 1
    E2 += np.einsum('jkic->jkci',V4)
    
    G2 = np.zeros([nocc, nocc], dtype=type_)
    
    #Contraction 15; Tree Level  3; Scaling  2/ 2 Result_size  2/ 0
    G2 += np.einsum('jb,bi->ji',F3,T1, optimize='optimal')
    
    A3 = np.zeros([nocc, nocc], dtype=type_)
    
    #Contraction 16; Tree Level  2; Scaling  2/ 0 Result_size  2/ 0
    A3 += -2.0 * np.einsum('ji->ji',G2)
    
    #Contraction 17; Tree Level  2; Scaling  2/ 0 Result_size  2/ 0
    A3 += -2.0 * np.einsum('ji->ji',F1)
    
    #Contraction 18; Tree Level  1; Scaling  3/ 3 Result_size  1/ 1
    Z1 += np.einsum('bj,jabi->ai',T1,M1, optimize='optimal')
    
    del M1
    
    #Contraction 19; Tree Level  1; Scaling  3/ 5 Result_size  1/ 1
    Z1 += 0.5 * np.einsum('ajbc,bcij->ai',V8,T2, optimize='optimal')
    
    #Contraction 20; Tree Level  2; Scaling  4/ 2 Result_size  2/ 0
    A3 += -2.0 * np.einsum('ck,jkci->ji',T1,E2, optimize='optimal')
    
    del E2
    
    #Contraction 21; Tree Level  2; Scaling  2/ 0 Result_size  2/ 0
    A3 += np.einsum('ji->ji',A2)
    
    #Contraction 22; Tree Level  1; Scaling  3/ 1 Result_size  1/ 1
    Z1 += 0.5 * np.einsum('aj,ji->ai',T1,A3, optimize='optimal')
    
    del A3
    
    E3 = np.zeros([nvir, nvir], dtype=type_)
    
    #Contraction 23; Tree Level  3; Scaling  4/ 4 Result_size  0/ 2
    E3 += np.einsum('klcd,dbkl->bc',V7,T2, optimize='optimal')
    
    #Contraction 24; Tree Level  2; Scaling  0/ 2 Result_size  0/ 2
    I3 += np.einsum('ab->ab',E3)
    
    #Contraction 25; Tree Level  2; Scaling  0/ 2 Result_size  0/ 2
    I3 += 2.0 * np.einsum('ab->ab',F4)
    
    #Contraction 26; Tree Level  1; Scaling  1/ 3 Result_size  1/ 1
    Z1 += 0.5 * np.einsum('bi,ab->ai',T1,I3, optimize='optimal')
    
    del I3
    
    #Contraction 27; Tree Level  1; Scaling  1/ 1 Result_size  1/ 1
    Z1 += np.einsum('ai->ai',F2)
    
    #del F2
    
    M3 = np.zeros([nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 28; Tree Level  4; Scaling  4/ 2 Result_size  4/ 0
    M3 += np.einsum('dj,kldi->klji',T1,X1, optimize='optimal')
    
    X3 = np.zeros([nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 29; Tree Level  4; Scaling  4/ 4 Result_size  4/ 0
    X3 += np.einsum('klcd,cdij->klij',V7,T2, optimize='optimal')
    
    Y3 = np.zeros([nocc, nvir], dtype=type_)
    
    #Contraction 30; Tree Level  4; Scaling  3/ 3 Result_size  1/ 1
    Y3 += np.einsum('klcd,cl->kd',V7,T1, optimize='optimal')
    
    #del V7
    
    A4 = np.zeros([nvir, nvir, nocc, nocc], dtype=type_)
    
    #Contraction 31; Tree Level  2; Scaling  4/ 2 Result_size  2/ 2
    A4 += np.einsum('ki,bajk->baij',F1,T2, optimize='optimal')
    
    #del F1
    
    #Contraction 32; Tree Level  0; Scaling  2/ 2 Result_size  2/ 2
    Z2 += -1.0 * np.einsum('baij->abij',A4)
    
    #Contraction 33; Tree Level  1; Scaling  2/ 2 Result_size  2/ 2
    Z2 += np.einsum('baji->abij',A4)
    
    del A4
    
    D4 = np.zeros([nvir, nvir, nocc, nocc], dtype=type_)
    
    #Contraction 34; Tree Level  2; Scaling  4/ 2 Result_size  2/ 2
    D4 += np.einsum('abkj,ki->abji',T2,G2, optimize='optimal')
    
    del G2
    
    #Contraction 35; Tree Level  1; Scaling  2/ 2 Result_size  2/ 2
    Z2 += -1.0 * np.einsum('abji->abij',D4)
    
    #Contraction 36; Tree Level  1; Scaling  2/ 2 Result_size  2/ 2
    Z2 += np.einsum('abij->abij',D4)
    
    del D4
    
    E4 = np.zeros([nvir, nocc, nvir, nocc], dtype=type_)
    
    #Contraction 37; Tree Level  4; Scaling  2/ 4 Result_size  2/ 2
    E4 += np.einsum('akcd,dj->akcj',V8,T1, optimize='optimal')
    
    G4 = np.zeros([nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 38; Tree Level  3; Scaling  3/ 5 Result_size  3/ 1
    G4 += np.einsum('akcd,cdij->akij',V8,T2, optimize='optimal')
    
    I4 = np.zeros([nvir, nvir, nocc, nocc], dtype=type_)
    
    #Contraction 39; Tree Level  2; Scaling  2/ 4 Result_size  2/ 2
    I4 += np.einsum('abic,cj->abij',V6,T1, optimize='optimal')
    
    #del V6
    
    #Contraction 40; Tree Level  1; Scaling  2/ 2 Result_size  2/ 2
    Z2 += np.einsum('abij->abij',I4)
    
    #Contraction 41; Tree Level  1; Scaling  2/ 2 Result_size  2/ 2
    Z2 += -1.0 * np.einsum('abji->abij',I4)
    
    del I4
    
    J4 = np.zeros([nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 42; Tree Level  3; Scaling  3/ 3 Result_size  3/ 1
    J4 += np.einsum('akic,cj->akij',V5,T1, optimize='optimal')
    
    M4 = np.zeros([nvir, nocc, nvir, nocc], dtype=type_)
    
    #Contraction 43; Tree Level  3; Scaling  2/ 4 Result_size  2/ 2
    M4 += np.einsum('akcd,ci->akdi',V8,T1, optimize='optimal')
    
    X4 = np.zeros([nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 44; Tree Level  3; Scaling  3/ 3 Result_size  3/ 1
    X4 += np.einsum('dbij,kd->bkij',T2,Y3, optimize='optimal')
    
    del Y3
    
    Y4 = np.zeros([nvir, nvir, nocc, nocc], dtype=type_)
    
    #Contraction 45; Tree Level  2; Scaling  2/ 4 Result_size  2/ 2
    Y4 += np.einsum('acij,bc->abij',T2,E3, optimize='optimal')
    
    del E3
    
    #Contraction 46; Tree Level  1; Scaling  2/ 2 Result_size  2/ 2
    Z2 += 0.5 * np.einsum('abij->abij',Y4)
    
    #Contraction 47; Tree Level  1; Scaling  2/ 2 Result_size  2/ 2
    Z2 += -0.5 * np.einsum('baij->abij',Y4)
    
    del Y4
    
    M6 = np.zeros([nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 48; Tree Level  3; Scaling  4/ 0 Result_size  4/ 0
    M6 += np.einsum('klij->lkji',V1)
    
    #Contraction 49; Tree Level  3; Scaling  4/ 0 Result_size  4/ 0
    M6 += -1.0 * np.einsum('lkij->lkji',V1)
    
    D5 = np.zeros([nvir, nvir], dtype=type_)
    
    #Contraction 50; Tree Level  3; Scaling  2/ 4 Result_size  0/ 2
    D5 += np.einsum('akcd,ck->ad',V8,T1, optimize='optimal')
    
    #del V8
    
    E5 = np.zeros([nvir, nvir, nocc, nocc], dtype=type_)
    
    #Contraction 51; Tree Level  2; Scaling  2/ 4 Result_size  2/ 2
    E5 += np.einsum('ac,bcji->abji',F4,T2, optimize='optimal')
    
    #del F4
    
    #Contraction 52; Tree Level  1; Scaling  2/ 2 Result_size  2/ 2
    Z2 += np.einsum('abji->abij',E5)
    
    #Contraction 53; Tree Level  1; Scaling  2/ 2 Result_size  2/ 2
    Z2 += -1.0 * np.einsum('baji->abij',E5)
    
    del E5
    
    G5 = np.zeros([nvir, nvir, nocc, nocc], dtype=type_)
    
    #Contraction 54; Tree Level  2; Scaling  4/ 4 Result_size  2/ 2
    G5 += np.einsum('dbkj,akdi->baji',T2,M4, optimize='optimal')
    
    del M4
    
    #Contraction 55; Tree Level  1; Scaling  2/ 2 Result_size  2/ 2
    Z2 += np.einsum('baji->abij',G5)
    
    #Contraction 56; Tree Level  1; Scaling  2/ 2 Result_size  2/ 2
    Z2 += -1.0 * np.einsum('abji->abij',G5)
    
    #Contraction 57; Tree Level  1; Scaling  2/ 2 Result_size  2/ 2
    Z2 += -1.0 * np.einsum('baij->abij',G5)
    
    #Contraction 58; Tree Level  1; Scaling  2/ 2 Result_size  2/ 2
    Z2 += np.einsum('abij->abij',G5)
    
    del G5
    
    I5 = np.zeros([nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 59; Tree Level  3; Scaling  5/ 3 Result_size  3/ 1
    I5 += np.einsum('dblj,kldi->bkji',T2,X1, optimize='optimal')
    
    J5 = np.zeros([nocc, nvir, nocc, nocc], dtype=type_)
    
    #Contraction 60; Tree Level  3; Scaling  3/ 3 Result_size  3/ 1
    J5 += np.einsum('kc,cbij->kbij',F3,T2, optimize='optimal')
    
    #del F3
    
    M5 = np.zeros([nvir, nvir, nvir, nocc], dtype=type_)
    
    #Contraction 61; Tree Level  3; Scaling  1/ 5 Result_size  1/ 3
    M5 += np.einsum('abcd,dj->abcj',V9,T1, optimize='optimal')
    
    X5 = np.zeros([nvir, nvir, nocc, nocc], dtype=type_)
    
    #Contraction 62; Tree Level  2; Scaling  2/ 4 Result_size  2/ 2
    X5 += np.einsum('ci,abcj->abij',T1,M5, optimize='optimal')
    
    del M5
    
    #Contraction 63; Tree Level  1; Scaling  2/ 2 Result_size  2/ 2
    Z2 += 0.5 * np.einsum('abij->abij',X5)
    
    #Contraction 64; Tree Level  1; Scaling  2/ 2 Result_size  2/ 2
    Z2 += -0.5 * np.einsum('abji->abij',X5)
    
    del X5
    
    Y5 = np.zeros([nvir, nvir, nocc, nocc], dtype=type_)
    
    #Contraction 65; Tree Level  2; Scaling  4/ 4 Result_size  2/ 2
    Y5 += np.einsum('akic,bcjk->abij',V5,T2, optimize='optimal')
    
    #del V5
    
    #Contraction 66; Tree Level  1; Scaling  2/ 2 Result_size  2/ 2
    Z2 += np.einsum('abij->abij',Y5)
    
    #Contraction 67; Tree Level  1; Scaling  2/ 2 Result_size  2/ 2
    Z2 += -1.0 * np.einsum('baij->abij',Y5)
    
    #Contraction 68; Tree Level  1; Scaling  2/ 2 Result_size  2/ 2
    Z2 += -1.0 * np.einsum('abji->abij',Y5)
    
    #Contraction 69; Tree Level  1; Scaling  2/ 2 Result_size  2/ 2
    Z2 += np.einsum('baji->abij',Y5)
    
    del Y5
    
    A6 = np.zeros([nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 70; Tree Level  4; Scaling  4/ 2 Result_size  4/ 0
    A6 += np.einsum('klic,cj->klij',V4,T1, optimize='optimal')
    
    X10 = np.zeros([nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 71; Tree Level  2; Scaling  4/ 0 Result_size  4/ 0
    X10 += 2.0 * np.einsum('klij->klij',A6)
    
    #Contraction 72; Tree Level  2; Scaling  4/ 0 Result_size  4/ 0
    X10 += -2.0 * np.einsum('klji->klij',A6)
    
    A11 = np.zeros([nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 73; Tree Level  3; Scaling  4/ 0 Result_size  4/ 0
    A11 += -1.0 * np.einsum('klji->lkji',M3)
    
    #Contraction 74; Tree Level  3; Scaling  4/ 0 Result_size  4/ 0
    A11 += -2.0 * np.einsum('klij->lkji',A6)
    
    #Contraction 75; Tree Level  3; Scaling  4/ 0 Result_size  4/ 0
    A11 += np.einsum('klij->lkji',M3)
    
    #Contraction 76; Tree Level  3; Scaling  4/ 0 Result_size  4/ 0
    M6 += -1.0 * np.einsum('lkij->lkji',A6)
    
    #Contraction 77; Tree Level  3; Scaling  4/ 0 Result_size  4/ 0
    M6 += -1.0 * np.einsum('klji->lkji',A6)
    
    #Contraction 78; Tree Level  3; Scaling  4/ 0 Result_size  4/ 0
    M6 += np.einsum('lkji->lkji',A6)
    
    del A6
    
    X6 = np.zeros([nocc, nvir, nocc, nocc], dtype=type_)
    
    #Contraction 79; Tree Level  3; Scaling  5/ 3 Result_size  3/ 1
    X6 += np.einsum('klic,cblj->kbij',V4,T2, optimize='optimal')
    
    E10 = np.zeros([nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 80; Tree Level  2; Scaling  3/ 1 Result_size  3/ 1
    E10 += -4.0 * np.einsum('kaij->akij',X6)
    
    #Contraction 81; Tree Level  2; Scaling  3/ 1 Result_size  3/ 1
    E10 += 4.0 * np.einsum('kaji->akij',X6)
    
    A7 = np.zeros([nvir, nvir, nocc, nocc], dtype=type_)
    
    #Contraction 82; Tree Level  2; Scaling  4/ 4 Result_size  2/ 2
    A7 += np.einsum('acik,kbcj->abij',T2,J1, optimize='optimal')
    
    del J1
    
    #Contraction 83; Tree Level  1; Scaling  2/ 2 Result_size  2/ 2
    Z2 += 0.5 * np.einsum('abij->abij',A7)
    
    #Contraction 84; Tree Level  1; Scaling  2/ 2 Result_size  2/ 2
    Z2 += -0.5 * np.einsum('baij->abij',A7)
    
    #Contraction 85; Tree Level  1; Scaling  2/ 2 Result_size  2/ 2
    Z2 += -0.5 * np.einsum('abji->abij',A7)
    
    #Contraction 86; Tree Level  1; Scaling  2/ 2 Result_size  2/ 2
    Z2 += 0.5 * np.einsum('baji->abij',A7)
    
    del A7
    
    M11 = np.zeros([nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 87; Tree Level  2; Scaling  3/ 1 Result_size  3/ 1
    M11 += 4.0 * np.einsum('bkij->bkij',J4)
    
    #Contraction 88; Tree Level  2; Scaling  3/ 1 Result_size  3/ 1
    M11 += -4.0 * np.einsum('bkji->bkij',J4)
    
    #Contraction 89; Tree Level  2; Scaling  3/ 1 Result_size  3/ 1
    E10 += -4.0 * np.einsum('kaij->akij',J5)
    
    #Contraction 90; Tree Level  2; Scaling  3/ 1 Result_size  3/ 1
    E10 += 4.0 * np.einsum('akij->akij',X4)
    
    G7 = np.zeros([nvir, nvir, nocc, nocc], dtype=type_)
    
    #Contraction 91; Tree Level  2; Scaling  4/ 2 Result_size  2/ 2
    G7 += np.einsum('abik,kj->abij',T2,A2, optimize='optimal')
    
    del A2
    
    #Contraction 92; Tree Level  1; Scaling  2/ 2 Result_size  2/ 2
    Z2 += 0.5 * np.einsum('abij->abij',G7)
    
    #Contraction 93; Tree Level  1; Scaling  2/ 2 Result_size  2/ 2
    Z2 += -0.5 * np.einsum('abji->abij',G7)
    
    del G7
    
    #Contraction 94; Tree Level  2; Scaling  3/ 1 Result_size  3/ 1
    E10 += -4.0 * np.einsum('akji->akij',I5)
    
    #Contraction 95; Tree Level  2; Scaling  3/ 1 Result_size  3/ 1
    E10 += 4.0 * np.einsum('akij->akij',I5)
    
    J7 = np.zeros([nocc, nocc], dtype=type_)
    
    #Contraction 96; Tree Level  3; Scaling  4/ 2 Result_size  2/ 0
    J7 += np.einsum('dk,kldi->li',T1,X1, optimize='optimal')
    
    del X1
    
    M7 = np.zeros([nvir, nvir, nocc, nocc], dtype=type_)
    
    #Contraction 97; Tree Level  2; Scaling  4/ 2 Result_size  2/ 2
    M7 += np.einsum('ablj,li->abji',T2,J7, optimize='optimal')
    
    del J7
    
    #Contraction 98; Tree Level  1; Scaling  2/ 2 Result_size  2/ 2
    Z2 += np.einsum('abji->abij',M7)
    
    #Contraction 99; Tree Level  1; Scaling  2/ 2 Result_size  2/ 2
    Z2 += -1.0 * np.einsum('abij->abij',M7)
    
    del M7
    
    X7 = np.zeros([nvir, nvir, nocc, nocc], dtype=type_)
    
    #Contraction 100; Tree Level  2; Scaling  2/ 4 Result_size  2/ 2
    X7 += np.einsum('dbij,ad->baij',T2,D5, optimize='optimal')
    
    del D5
    
    #Contraction 101; Tree Level  1; Scaling  2/ 2 Result_size  2/ 2
    Z2 += -1.0 * np.einsum('baij->abij',X7)
    
    #Contraction 102; Tree Level  1; Scaling  2/ 2 Result_size  2/ 2
    Z2 += np.einsum('abij->abij',X7)
    
    del X7
    
    Y7 = np.zeros([nocc, nocc], dtype=type_)
    
    #Contraction 103; Tree Level  3; Scaling  4/ 2 Result_size  2/ 0
    Y7 += np.einsum('klic,ck->li',V4,T1, optimize='optimal')
    
    #del V4
    
    A8 = np.zeros([nvir, nvir, nocc, nocc], dtype=type_)
    
    #Contraction 104; Tree Level  2; Scaling  4/ 2 Result_size  2/ 2
    A8 += np.einsum('ablj,li->abji',T2,Y7, optimize='optimal')
    
    del Y7
    
    #Contraction 105; Tree Level  1; Scaling  2/ 2 Result_size  2/ 2
    Z2 += np.einsum('abji->abij',A8)
    
    #Contraction 106; Tree Level  1; Scaling  2/ 2 Result_size  2/ 2
    Z2 += -1.0 * np.einsum('abij->abij',A8)
    
    del A8
    
    D8 = np.zeros([nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 107; Tree Level  3; Scaling  4/ 0 Result_size  4/ 0
    D8 += np.einsum('klij->klij',X3)
    
    #Contraction 108; Tree Level  3; Scaling  4/ 0 Result_size  4/ 0
    D8 += -1.0 * np.einsum('lkij->klij',X3)
    
    #Contraction 109; Tree Level  2; Scaling  5/ 1 Result_size  3/ 1
    E10 += -1.0 * np.einsum('al,lkij->akij',T1,D8, optimize='optimal')
    
    del D8
    
    #Contraction 110; Tree Level  2; Scaling  3/ 1 Result_size  3/ 1
    E10 += 4.0 * np.einsum('akij->akij',J4)
    
    #Contraction 111; Tree Level  2; Scaling  3/ 1 Result_size  3/ 1
    E10 += -4.0 * np.einsum('akji->akij',J4)
    
    del J4
    
    J8 = np.zeros([nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 112; Tree Level  3; Scaling  3/ 3 Result_size  3/ 1
    J8 += np.einsum('ci,akcj->akij',T1,E4, optimize='optimal')
    
    del E4
    
    #Contraction 113; Tree Level  2; Scaling  3/ 1 Result_size  3/ 1
    E10 += 2.0 * np.einsum('akij->akij',J8)
    
    #Contraction 114; Tree Level  2; Scaling  3/ 1 Result_size  3/ 1
    E10 += -2.0 * np.einsum('akji->akij',J8)
    
    #Contraction 115; Tree Level  2; Scaling  3/ 1 Result_size  3/ 1
    E10 += 2.0 * np.einsum('akij->akij',G4)
    
    #Contraction 116; Tree Level  2; Scaling  3/ 1 Result_size  3/ 1
    M11 += 2.0 * np.einsum('bkij->bkij',J8)
    
    #Contraction 117; Tree Level  2; Scaling  3/ 1 Result_size  3/ 1
    M11 += -2.0 * np.einsum('bkji->bkij',J8)
    
    del J8
    
    #Contraction 118; Tree Level  2; Scaling  3/ 1 Result_size  3/ 1
    M11 += -4.0 * np.einsum('bkji->bkij',I5)
    
    #Contraction 119; Tree Level  2; Scaling  3/ 1 Result_size  3/ 1
    M11 += 4.0 * np.einsum('bkij->bkij',I5)
    
    del I5
    
    #Contraction 120; Tree Level  2; Scaling  3/ 1 Result_size  3/ 1
    M11 += -4.0 * np.einsum('kbij->bkij',X6)
    
    #Contraction 121; Tree Level  2; Scaling  3/ 1 Result_size  3/ 1
    M11 += 4.0 * np.einsum('kbji->bkij',X6)
    
    del X6
    
    #Contraction 122; Tree Level  2; Scaling  3/ 1 Result_size  3/ 1
    M11 += 2.0 * np.einsum('bkij->bkij',G4)
    
    del G4
    
    #Contraction 123; Tree Level  2; Scaling  5/ 1 Result_size  3/ 1
    E10 += -2.0 * np.einsum('al,klji->akij',T1,M6, optimize='optimal')
    
    del M6
    
    #Contraction 124; Tree Level  2; Scaling  3/ 1 Result_size  3/ 1
    E10 += 4.0 * np.einsum('akij->akij',V2)
    
    #Contraction 125; Tree Level  1; Scaling  4/ 2 Result_size  2/ 2
    Z2 += -0.25 * np.einsum('bk,akij->abij',T1,E10, optimize='optimal')
    
    del E10
    
    I10 = np.zeros([nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 126; Tree Level  3; Scaling  4/ 0 Result_size  4/ 0
    I10 += np.einsum('klji->klji',M3)
    
    #Contraction 127; Tree Level  3; Scaling  4/ 0 Result_size  4/ 0
    I10 += -1.0 * np.einsum('klij->klji',M3)
    
    del M3
    
    #Contraction 128; Tree Level  2; Scaling  4/ 0 Result_size  4/ 0
    X10 += np.einsum('klji->klij',I10)
    
    #Contraction 129; Tree Level  2; Scaling  4/ 0 Result_size  4/ 0
    X10 += np.einsum('klij->klij',X3)
    
    del X3
    
    #Contraction 130; Tree Level  2; Scaling  4/ 0 Result_size  4/ 0
    X10 += 2.0 * np.einsum('klij->klij',V1)
    
    #del V1
    
    #Contraction 131; Tree Level  1; Scaling  6/ 2 Result_size  2/ 2
    Z2 += 0.25 * np.einsum('abkl,klij->abij',T2,X10, optimize='optimal')
    
    del X10
    
    #Contraction 132; Tree Level  3; Scaling  4/ 0 Result_size  4/ 0
    A11 += np.einsum('lkji->lkji',I10)
    
    del I10
    
    #Contraction 133; Tree Level  2; Scaling  5/ 1 Result_size  3/ 1
    M11 += -1.0 * np.einsum('bl,lkji->bkij',T1,A11, optimize='optimal')
    
    del A11
    
    #Contraction 134; Tree Level  2; Scaling  3/ 1 Result_size  3/ 1
    M11 += 4.0 * np.einsum('bkij->bkij',X4)
    
    del X4
    
    #Contraction 135; Tree Level  2; Scaling  3/ 1 Result_size  3/ 1
    M11 += -4.0 * np.einsum('kbij->bkij',J5)
    
    del J5
    
    #Contraction 136; Tree Level  2; Scaling  3/ 1 Result_size  3/ 1
    M11 += 4.0 * np.einsum('bkij->bkij',V2)
    
    #del V2
    
    #Contraction 137; Tree Level  1; Scaling  4/ 2 Result_size  2/ 2
    Z2 += 0.25 * np.einsum('ak,bkij->abij',T1,M11, optimize='optimal')
    
    del M11
    
    #del T1
    
    #Contraction 138; Tree Level  1; Scaling  2/ 6 Result_size  2/ 2
    Z2 += 0.5 * np.einsum('abcd,cdij->abij',V9,T2, optimize='optimal')
    
    #del T2
    
    #del V9
    
    #Contraction 139; Tree Level  1; Scaling  2/ 2 Result_size  2/ 2
    Z2 += np.einsum('abij->abij',V3)
    
    #del V3
    
    return([Z0[0], Z1, Z2])
    
# end of numpy_tenpi_ccsd
