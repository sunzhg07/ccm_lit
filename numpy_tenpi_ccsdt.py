#!/usr/bin/python
# -*- coding: utf-8 -*-

# CCSDT_T6 for simple use with numpy

import numpy as np #numpy arrays
import opt_einsum as oe
import sys


# returns a list of np.ndarrays ordered by growing rank: [energy, residual_vo, residual_vvoo, ...] except the scalars are returned as scalars.
def numpy_tenpi_ccsdt(nocc: int, nvir: int, F: list, V: list, T: list, type_: type = np.complex128) -> list:
    
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
    T3 = T[2]
    
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
    # T3 = np.array([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)     #       VVVOOO
    Z0 = np.zeros([1], dtype=type_)     #       scalar
    Z1 = np.zeros([nvir, nocc], dtype=type_)     #       VO
    Z2 = np.zeros([nvir, nvir, nocc, nocc], dtype=type_)     #       VVOO
    Z3 = np.zeros([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)     #       VVVOOO
    # A1 = np.array([nocc, nvir], dtype=type_)     #       OV
    # D1 = np.array([nocc, nvir], dtype=type_)     #       OV
    # M3 = np.array([nvir, nvir], dtype=type_)     #       VV
    # X1 = np.array([nocc, nvir, nvir, nocc], dtype=type_)     #       OVVO
    # Y1 = np.array([nocc, nvir, nvir, nocc], dtype=type_)     #       OVVO
    # D2 = np.array([nocc, nocc], dtype=type_)     #       OO
    # G2 = np.array([nocc, nocc, nvir, nocc], dtype=type_)     #       OOVO
    # D3 = np.array([nocc, nocc], dtype=type_)     #       OO
    # J2 = np.array([nocc, nocc], dtype=type_)     #       OO
    # I3 = np.array([nvir, nvir], dtype=type_)     #       VV
    # A4 = np.array([nvir, nvir, nocc, nocc], dtype=type_)     #       VVOO
    # D4 = np.array([nvir, nvir, nocc, nocc], dtype=type_)     #       VVOO
    # E4 = np.array([nvir, nvir, nocc, nocc], dtype=type_)     #       VVOO
    # G4 = np.array([nocc, nocc, nocc, nocc], dtype=type_)     #       OOOO
    # I4 = np.array([nvir, nvir, nocc, nocc], dtype=type_)     #       VVOO
    # J4 = np.array([nvir, nocc, nvir, nocc], dtype=type_)     #       VOVO
    # M4 = np.array([nvir, nvir, nocc, nocc], dtype=type_)     #       VVOO
    # X4 = np.array([nvir, nvir, nocc, nocc], dtype=type_)     #       VVOO
    # Y4 = np.array([nvir, nvir, nocc, nocc], dtype=type_)     #       VVOO
    # A5 = np.array([nocc, nvir, nocc, nocc], dtype=type_)     #       OVOO
    # A10 = np.array([nocc, nocc, nocc, nocc], dtype=type_)     #       OOOO
    # E5 = np.array([nvir, nocc, nvir, nocc], dtype=type_)     #       VOVO
    # G5 = np.array([nocc, nvir, nocc, nocc], dtype=type_)     #       OVOO
    # J5 = np.array([nocc, nocc], dtype=type_)     #       OO
    # M5 = np.array([nvir, nvir, nocc, nocc], dtype=type_)     #       VVOO
    # X5 = np.array([nocc, nvir, nocc, nocc], dtype=type_)     #       OVOO
    # Y5 = np.array([nvir, nocc, nocc, nocc], dtype=type_)     #       VOOO
    # A13 = np.array([nvir, nocc, nocc, nocc], dtype=type_)     #       VOOO
    # D6 = np.array([nvir, nvir], dtype=type_)     #       VV
    # E6 = np.array([nvir, nvir, nocc, nocc], dtype=type_)     #       VVOO
    # I6 = np.array([nocc, nvir], dtype=type_)     #       OV
    # J6 = np.array([nvir, nocc, nocc, nocc], dtype=type_)     #       VOOO
    # M6 = np.array([nvir, nvir, nocc, nocc], dtype=type_)     #       VVOO
    # X6 = np.array([nvir, nvir, nocc, nocc], dtype=type_)     #       VVOO
    # Y6 = np.array([nvir, nocc, nocc, nocc], dtype=type_)     #       VOOO
    # A12 = np.array([nocc, nvir, nocc, nocc], dtype=type_)     #       OVOO
    # I7 = np.array([nocc, nocc], dtype=type_)     #       OO
    # J7 = np.array([nvir, nvir, nocc, nocc], dtype=type_)     #       VVOO
    # M7 = np.array([nocc, nocc, nocc, nocc], dtype=type_)     #       OOOO
    # J8 = np.array([nocc, nocc, nocc, nocc], dtype=type_)     #       OOOO
    # Y7 = np.array([nvir, nvir, nvir, nocc], dtype=type_)     #       VVVO
    # A8 = np.array([nvir, nvir, nocc, nocc], dtype=type_)     #       VVOO
    # D8 = np.array([nocc, nocc, nocc, nocc], dtype=type_)     #       OOOO
    # X8 = np.array([nvir, nocc, nocc, nocc], dtype=type_)     #       VOOO
    # Y8 = np.array([nocc, nocc, nocc, nocc], dtype=type_)     #       OOOO
    # D9 = np.array([nvir, nocc, nocc, nocc], dtype=type_)     #       VOOO
    # A11 = np.array([nvir, nocc, nocc, nocc], dtype=type_)     #       VOOO
    # E12 = np.array([nvir, nvir, nocc, nocc], dtype=type_)     #       VVOO
    # G12 = np.array([nvir, nvir, nocc, nocc], dtype=type_)     #       VVOO
    # I12 = np.array([nvir, nvir, nocc, nocc], dtype=type_)     #       VVOO
    # E13 = np.array([nvir, nocc, nocc, nocc], dtype=type_)     #       VOOO
    # G13 = np.array([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)     #       VVVOOO
    # I13 = np.array([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)     #       VVVOOO
    # J13 = np.array([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)     #       VVVOOO
    # M13 = np.array([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)     #       VVVOOO
    # X13 = np.array([nvir, nocc, nocc, nocc, nocc, nocc], dtype=type_)     #       VOOOOO
    # Y13 = np.array([nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)     #       VVOOOO
    # A14 = np.array([nvir, nocc, nocc, nocc], dtype=type_)     #       VOOO
    # D14 = np.array([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)     #       VVVOOO
    # E14 = np.array([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)     #       VVVOOO
    # G14 = np.array([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)     #       VVVOOO
    # I14 = np.array([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)     #       VVVOOO
    # J14 = np.array([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)     #       VVVOOO
    # M14 = np.array([nocc, nvir, nvir, nocc, nocc, nocc], dtype=type_)     #       OVVOOO
    # X14 = np.array([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)     #       VVVOOO
    # Y14 = np.array([nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)     #       VVOOOO
    # A15 = np.array([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)     #       VVVOOO
    # D15 = np.array([nocc, nvir, nocc, nocc], dtype=type_)     #       OVOO
    # E15 = np.array([nocc, nocc, nvir, nocc, nocc, nocc], dtype=type_)     #       OOVOOO
    # G15 = np.array([nvir, nocc, nvir, nocc, nocc, nocc], dtype=type_)     #       VOVOOO
    # I15 = np.array([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)     #       VVVOOO
    # J15 = np.array([nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)     #       VVOOOO
    # M15 = np.array([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)     #       VVVOOO
    # X15 = np.array([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)     #       VVVOOO
    # Y15 = np.array([nvir, nocc, nvir, nocc, nocc, nocc], dtype=type_)     #       VOVOOO
    # A16 = np.array([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)     #       VVVOOO
    # D16 = np.array([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)     #       VVVOOO
    # E16 = np.array([nvir, nvir, nvir, nocc], dtype=type_)     #       VVVO
    # G16 = np.array([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)     #       VVVOOO
    # I16 = np.array([nocc, nvir, nocc, nocc], dtype=type_)     #       OVOO
    # J16 = np.array([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)     #       VVVOOO
    # M16 = np.array([nocc, nocc, nvir, nocc, nocc, nocc], dtype=type_)     #       OOVOOO
    # X16 = np.array([nvir, nocc, nvir, nocc, nocc, nocc], dtype=type_)     #       VOVOOO
    # Y16 = np.array([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)     #       VVVOOO
    # A17 = np.array([nocc, nocc], dtype=type_)     #       OO
    # D17 = np.array([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)     #       VVVOOO
    # E17 = np.array([nvir, nvir, nvir, nocc], dtype=type_)     #       VVVO
    # G17 = np.array([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)     #       VVVOOO
    # I17 = np.array([nvir, nvir, nvir, nocc], dtype=type_)     #       VVVO
    # J17 = np.array([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)     #       VVVOOO
    # M17 = np.array([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)     #       VVVOOO
    # X17 = np.array([nvir, nocc, nocc, nocc], dtype=type_)     #       VOOO
    # Y17 = np.array([nocc, nvir, nvir, nocc, nocc, nocc], dtype=type_)     #       OVVOOO
    # A18 = np.array([nocc, nvir, nvir, nocc], dtype=type_)     #       OVVO
    # D18 = np.array([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)     #       VVVOOO
    # E18 = np.array([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)     #       VVVOOO
    # G18 = np.array([nvir, nvir, nocc, nvir], dtype=type_)     #       VVOV
    # I18 = np.array([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)     #       VVVOOO
    # J18 = np.array([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)     #       VVVOOO
    # M18 = np.array([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)     #       VVVOOO
    # X18 = np.array([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)     #       VVVOOO
    # Y18 = np.array([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)     #       VVVOOO
    # A19 = np.array([nvir, nocc, nvir, nocc, nocc, nocc], dtype=type_)     #       VOVOOO
    # D19 = np.array([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)     #       VVVOOO
    # E19 = np.array([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)     #       VVVOOO
    # G19 = np.array([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)     #       VVVOOO
    # I19 = np.array([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)     #       VVVOOO
    # J19 = np.array([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)     #       VVVOOO
    # M19 = np.array([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)     #       VVVOOO
    # X19 = np.array([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)     #       VVVOOO
    # Y19 = np.array([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)     #       VVVOOO
    # A20 = np.array([nvir, nocc, nvir, nocc, nocc, nocc], dtype=type_)     #       VOVOOO
    # D20 = np.array([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)     #       VVVOOO
    # E20 = np.array([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)     #       VVVOOO
    # G20 = np.array([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)     #       VVVOOO
    # I20 = np.array([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)     #       VVVOOO
    # J20 = np.array([nvir, nvir], dtype=type_)     #       VV
    # M20 = np.array([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)     #       VVVOOO
    # X20 = np.array([nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)     #       VVOOOO
    # Y20 = np.array([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)     #       VVVOOO
    # A21 = np.array([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)     #       VVVOOO
    # D21 = np.array([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)     #       VVVOOO
    # E21 = np.array([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)     #       VVVOOO
    
    print('    Running code generated by tenpi.   ')
    
    A1 = np.zeros([nocc, nvir], dtype=type_)
    
    #Contraction 1; Tree Level  3; Scaling  3/ 3 Result_size  1/ 1
    A1 += oe.contract('ijab,ai->jb',V7,T1, optimize='optimal')
    
    D1 = np.zeros([nocc, nvir], dtype=type_)
    
    #Contraction 2; Tree Level  2; Scaling  1/ 1 Result_size  1/ 1
    D1 += oe.contract('jb->jb',A1)
    
    #Contraction 3; Tree Level  2; Scaling  1/ 1 Result_size  1/ 1
    D1 += 2.0 * oe.contract('jb->jb',F3)
    
    #Contraction 4; Tree Level  0; Scaling  2/ 2 Result_size  0/ 0
    Z0 += 0.5 * oe.contract('bj,jb->',T1,D1, optimize='optimal')
    
    del D1
    
    #Contraction 5; Tree Level  1; Scaling  4/ 4 Result_size  0/ 0
    Z0 += 0.25 * oe.contract('ijab,abij->',V7,T2, optimize='optimal')
    
    #Contraction 6; Tree Level  0; Scaling  5/ 5 Result_size  1/ 1
    Z1 += 0.25 * oe.contract('jkbc,abcijk->ai',V7,T3, optimize='optimal')
    
    M3 = np.zeros([nvir, nvir], dtype=type_)
    
    #Contraction 7; Tree Level  2; Scaling  2/ 4 Result_size  0/ 2
    M3 += 2.0 * oe.contract('ajbc,cj->ab',V8,T1, optimize='optimal')
    
    #Contraction 8; Tree Level  1; Scaling  3/ 5 Result_size  1/ 1
    Z1 += 0.5 * oe.contract('ajbc,bcij->ai',V8,T2, optimize='optimal')
    
    X1 = np.zeros([nocc, nvir, nvir, nocc], dtype=type_)
    
    #Contraction 9; Tree Level  3; Scaling  4/ 4 Result_size  2/ 2
    X1 += oe.contract('klcd,dblj->kbcj',V7,T2, optimize='optimal')
    
    Y1 = np.zeros([nocc, nvir, nvir, nocc], dtype=type_)
    
    #Contraction 10; Tree Level  2; Scaling  2/ 2 Result_size  2/ 2
    Y1 += oe.contract('jabi->jabi',X1)
    
    #Contraction 11; Tree Level  2; Scaling  2/ 2 Result_size  2/ 2
    Y1 += oe.contract('ajib->jabi',V5)
    
    #Contraction 12; Tree Level  1; Scaling  3/ 3 Result_size  1/ 1
    Z1 += oe.contract('bj,jabi->ai',T1,Y1, optimize='optimal')
    
    del Y1
    
    D2 = np.zeros([nocc, nocc], dtype=type_)
    
    #Contraction 13; Tree Level  3; Scaling  4/ 4 Result_size  2/ 0
    D2 += oe.contract('jkbc,bcki->ji',V7,T2, optimize='optimal')
    
    #Contraction 14; Tree Level  1; Scaling  5/ 3 Result_size  1/ 1
    Z1 += -0.5 * oe.contract('jkib,abjk->ai',V4,T2, optimize='optimal')
    
    G2 = np.zeros([nocc, nocc, nvir, nocc], dtype=type_)
    
    #Contraction 15; Tree Level  3; Scaling  3/ 3 Result_size  3/ 1
    G2 += oe.contract('jkbc,bi->jkci',V7,T1, optimize='optimal')
    
    D3 = np.zeros([nocc, nocc], dtype=type_)
    
    #Contraction 16; Tree Level  2; Scaling  4/ 2 Result_size  2/ 0
    D3 += -2.0 * oe.contract('ck,jkci->ji',T1,G2, optimize='optimal')
    
    J2 = np.zeros([nocc, nocc], dtype=type_)
    
    #Contraction 17; Tree Level  3; Scaling  2/ 2 Result_size  2/ 0
    J2 += oe.contract('jb,bi->ji',F3,T1, optimize='optimal')
    
    #Contraction 18; Tree Level  2; Scaling  2/ 0 Result_size  2/ 0
    D3 += oe.contract('ji->ji',D2)
    
    #Contraction 19; Tree Level  2; Scaling  4/ 2 Result_size  2/ 0
    D3 += -2.0 * oe.contract('jkib,bk->ji',V4,T1, optimize='optimal')
    
    #Contraction 20; Tree Level  2; Scaling  2/ 0 Result_size  2/ 0
    D3 += -2.0 * oe.contract('ji->ji',J2)
    
    #Contraction 21; Tree Level  2; Scaling  2/ 0 Result_size  2/ 0
    D3 += -2.0 * oe.contract('ji->ji',F1)
    
    #Contraction 22; Tree Level  1; Scaling  3/ 1 Result_size  1/ 1
    Z1 += 0.5 * oe.contract('aj,ji->ai',T1,D3, optimize='optimal')
    
    del D3
    
    #Contraction 23; Tree Level  1; Scaling  3/ 3 Result_size  1/ 1
    Z1 += oe.contract('jb,abij->ai',F3,T2, optimize='optimal')
    
    I3 = np.zeros([nvir, nvir], dtype=type_)
    
    #Contraction 24; Tree Level  3; Scaling  4/ 4 Result_size  0/ 2
    I3 += oe.contract('klcd,dbkl->bc',V7,T2, optimize='optimal')
    
    #Contraction 25; Tree Level  2; Scaling  0/ 2 Result_size  0/ 2
    M3 += oe.contract('ab->ab',I3)
    
    #Contraction 26; Tree Level  2; Scaling  0/ 2 Result_size  0/ 2
    M3 += 2.0 * oe.contract('ab->ab',F4)
    
    #Contraction 27; Tree Level  1; Scaling  1/ 3 Result_size  1/ 1
    Z1 += 0.5 * oe.contract('bi,ab->ai',T1,M3, optimize='optimal')
    
    del M3
    
    #Contraction 28; Tree Level  1; Scaling  1/ 1 Result_size  1/ 1
    Z1 += oe.contract('ai->ai',F2)
    
    #del F2
    
    #Contraction 29; Tree Level  0; Scaling  4/ 4 Result_size  2/ 2
    Z2 += oe.contract('dablij,ld->abij',T3,A1, optimize='optimal')
    
    A4 = np.zeros([nvir, nvir, nocc, nocc], dtype=type_)
    
    #Contraction 30; Tree Level  2; Scaling  2/ 4 Result_size  2/ 2
    A4 += oe.contract('ac,bcji->abji',F4,T2, optimize='optimal')
    
    #Contraction 31; Tree Level  1; Scaling  2/ 2 Result_size  2/ 2
    Z2 += oe.contract('abji->abij',A4)
    
    #Contraction 32; Tree Level  1; Scaling  2/ 2 Result_size  2/ 2
    Z2 += -1.0 * oe.contract('baji->abij',A4)
    
    del A4
    
    D4 = np.zeros([nvir, nvir, nocc, nocc], dtype=type_)
    
    #Contraction 33; Tree Level  2; Scaling  4/ 4 Result_size  2/ 2
    D4 += oe.contract('acik,kbcj->abij',T2,X1, optimize='optimal')
    
    #Contraction 34; Tree Level  1; Scaling  2/ 2 Result_size  2/ 2
    Z2 += 0.5 * oe.contract('abij->abij',D4)
    
    #Contraction 35; Tree Level  1; Scaling  2/ 2 Result_size  2/ 2
    Z2 += -0.5 * oe.contract('baij->abij',D4)
    
    #Contraction 36; Tree Level  1; Scaling  2/ 2 Result_size  2/ 2
    Z2 += -0.5 * oe.contract('abji->abij',D4)
    
    #Contraction 37; Tree Level  1; Scaling  2/ 2 Result_size  2/ 2
    Z2 += 0.5 * oe.contract('baji->abij',D4)
    
    del D4
    
    E4 = np.zeros([nvir, nvir, nocc, nocc], dtype=type_)
    
    #Contraction 38; Tree Level  2; Scaling  6/ 4 Result_size  2/ 2
    E4 += oe.contract('klic,bacjkl->baij',V4,T3, optimize='optimal')
    
    #Contraction 39; Tree Level  1; Scaling  2/ 2 Result_size  2/ 2
    Z2 += -0.5 * oe.contract('baij->abij',E4)
    
    #Contraction 40; Tree Level  1; Scaling  2/ 2 Result_size  2/ 2
    Z2 += 0.5 * oe.contract('baji->abij',E4)
    
    del E4
    
    G4 = np.zeros([nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 41; Tree Level  3; Scaling  4/ 4 Result_size  4/ 0
    G4 += oe.contract('klcd,cdij->klij',V7,T2, optimize='optimal')
    
    I4 = np.zeros([nvir, nvir, nocc, nocc], dtype=type_)
    
    #Contraction 42; Tree Level  2; Scaling  2/ 4 Result_size  2/ 2
    I4 += oe.contract('abic,cj->abij',V6,T1, optimize='optimal')
    
    #Contraction 43; Tree Level  1; Scaling  2/ 2 Result_size  2/ 2
    Z2 += oe.contract('abij->abij',I4)
    
    #Contraction 44; Tree Level  1; Scaling  2/ 2 Result_size  2/ 2
    Z2 += -1.0 * oe.contract('abji->abij',I4)
    
    del I4
    
    J4 = np.zeros([nvir, nocc, nvir, nocc], dtype=type_)
    
    #Contraction 45; Tree Level  3; Scaling  2/ 4 Result_size  2/ 2
    J4 += oe.contract('akcd,ci->akdi',V8,T1, optimize='optimal')
    
    M4 = np.zeros([nvir, nvir, nocc, nocc], dtype=type_)
    
    #Contraction 46; Tree Level  2; Scaling  4/ 4 Result_size  2/ 2
    M4 += oe.contract('dbkj,akdi->baji',T2,J4, optimize='optimal')
    
    #Contraction 47; Tree Level  1; Scaling  2/ 2 Result_size  2/ 2
    Z2 += oe.contract('baji->abij',M4)
    
    #Contraction 48; Tree Level  1; Scaling  2/ 2 Result_size  2/ 2
    Z2 += -1.0 * oe.contract('abji->abij',M4)
    
    #Contraction 49; Tree Level  1; Scaling  2/ 2 Result_size  2/ 2
    Z2 += -1.0 * oe.contract('baij->abij',M4)
    
    #Contraction 50; Tree Level  1; Scaling  2/ 2 Result_size  2/ 2
    Z2 += oe.contract('abij->abij',M4)
    
    del M4
    
    X4 = np.zeros([nvir, nvir, nocc, nocc], dtype=type_)
    
    #Contraction 51; Tree Level  2; Scaling  4/ 6 Result_size  2/ 2
    X4 += oe.contract('akcd,bcdjik->abji',V8,T3, optimize='optimal')
    
    #Contraction 52; Tree Level  1; Scaling  2/ 2 Result_size  2/ 2
    Z2 += 0.5 * oe.contract('abji->abij',X4)
    
    #Contraction 53; Tree Level  1; Scaling  2/ 2 Result_size  2/ 2
    Z2 += -0.5 * oe.contract('baji->abij',X4)
    
    del X4
    
    Y4 = np.zeros([nvir, nvir, nocc, nocc], dtype=type_)
    
    #Contraction 54; Tree Level  2; Scaling  6/ 4 Result_size  2/ 2
    Y4 += oe.contract('dabklj,kldi->abji',T3,G2, optimize='optimal')
    
    #Contraction 55; Tree Level  1; Scaling  2/ 2 Result_size  2/ 2
    Z2 += 0.5 * oe.contract('abji->abij',Y4)
    
    #Contraction 56; Tree Level  1; Scaling  2/ 2 Result_size  2/ 2
    Z2 += -0.5 * oe.contract('abij->abij',Y4)
    
    del Y4
    
    A5 = np.zeros([nocc, nvir, nocc, nocc], dtype=type_)
    
    #Contraction 57; Tree Level  3; Scaling  5/ 5 Result_size  3/ 1
    A5 += oe.contract('klcd,cdblij->kbij',V7,T3, optimize='optimal')
    
    A10 = np.zeros([nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 58; Tree Level  3; Scaling  4/ 0 Result_size  4/ 0
    A10 += oe.contract('klij->klij',V1)
    
    #Contraction 59; Tree Level  3; Scaling  4/ 0 Result_size  4/ 0
    A10 += -1.0 * oe.contract('lkij->klij',V1)
    
    E5 = np.zeros([nvir, nocc, nvir, nocc], dtype=type_)
    
    #Contraction 60; Tree Level  4; Scaling  2/ 4 Result_size  2/ 2
    E5 += oe.contract('akcd,dj->akcj',V8,T1, optimize='optimal')
    
    G5 = np.zeros([nocc, nvir, nocc, nocc], dtype=type_)
    
    #Contraction 61; Tree Level  3; Scaling  5/ 3 Result_size  3/ 1
    G5 += oe.contract('klic,cblj->kbij',V4,T2, optimize='optimal')
    
    #Contraction 62; Tree Level  1; Scaling  2/ 6 Result_size  2/ 2
    Z2 += 0.5 * oe.contract('abcd,cdij->abij',V9,T2, optimize='optimal')
    
    J5 = np.zeros([nocc, nocc], dtype=type_)
    
    #Contraction 63; Tree Level  3; Scaling  4/ 2 Result_size  2/ 0
    J5 += oe.contract('klic,ck->li',V4,T1, optimize='optimal')
    
    M5 = np.zeros([nvir, nvir, nocc, nocc], dtype=type_)
    
    #Contraction 64; Tree Level  2; Scaling  4/ 2 Result_size  2/ 2
    M5 += oe.contract('ablj,li->abji',T2,J5, optimize='optimal')
    
    #Contraction 65; Tree Level  1; Scaling  2/ 2 Result_size  2/ 2
    Z2 += oe.contract('abji->abij',M5)
    
    #Contraction 66; Tree Level  1; Scaling  2/ 2 Result_size  2/ 2
    Z2 += -1.0 * oe.contract('abij->abij',M5)
    
    del M5
    
    X5 = np.zeros([nocc, nvir, nocc, nocc], dtype=type_)
    
    #Contraction 67; Tree Level  3; Scaling  3/ 3 Result_size  3/ 1
    X5 += oe.contract('kc,cbij->kbij',F3,T2, optimize='optimal')
    
    Y5 = np.zeros([nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 68; Tree Level  3; Scaling  5/ 3 Result_size  3/ 1
    Y5 += oe.contract('dblj,kldi->bkji',T2,G2, optimize='optimal')
    
    A13 = np.zeros([nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 69; Tree Level  2; Scaling  3/ 1 Result_size  3/ 1
    A13 += -4.0 * oe.contract('akji->akij',Y5)
    
    #Contraction 70; Tree Level  2; Scaling  3/ 1 Result_size  3/ 1
    A13 += 4.0 * oe.contract('akij->akij',Y5)
    
    D6 = np.zeros([nvir, nvir], dtype=type_)
    
    #Contraction 71; Tree Level  3; Scaling  2/ 4 Result_size  0/ 2
    D6 += oe.contract('akcd,ck->ad',V8,T1, optimize='optimal')
    
    E6 = np.zeros([nvir, nvir, nocc, nocc], dtype=type_)
    
    #Contraction 72; Tree Level  2; Scaling  2/ 4 Result_size  2/ 2
    E6 += oe.contract('dbij,ad->baij',T2,D6, optimize='optimal')
    
    #Contraction 73; Tree Level  1; Scaling  2/ 2 Result_size  2/ 2
    Z2 += -1.0 * oe.contract('baij->abij',E6)
    
    #Contraction 74; Tree Level  1; Scaling  2/ 2 Result_size  2/ 2
    Z2 += oe.contract('abij->abij',E6)
    
    del E6
    
    #Contraction 75; Tree Level  2; Scaling  3/ 1 Result_size  3/ 1
    A13 += -4.0 * oe.contract('kaij->akij',G5)
    
    #Contraction 76; Tree Level  2; Scaling  3/ 1 Result_size  3/ 1
    A13 += 4.0 * oe.contract('kaji->akij',G5)
    
    I6 = np.zeros([nocc, nvir], dtype=type_)
    
    #Contraction 77; Tree Level  4; Scaling  3/ 3 Result_size  1/ 1
    I6 += oe.contract('klcd,cl->kd',V7,T1, optimize='optimal')
    
    J6 = np.zeros([nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 78; Tree Level  3; Scaling  3/ 3 Result_size  3/ 1
    J6 += oe.contract('dbij,kd->bkij',T2,I6, optimize='optimal')
    
    M6 = np.zeros([nvir, nvir, nocc, nocc], dtype=type_)
    
    #Contraction 79; Tree Level  2; Scaling  2/ 4 Result_size  2/ 2
    M6 += oe.contract('acij,bc->abij',T2,I3, optimize='optimal')
    
    del I3
    
    #Contraction 80; Tree Level  1; Scaling  2/ 2 Result_size  2/ 2
    Z2 += 0.5 * oe.contract('abij->abij',M6)
    
    #Contraction 81; Tree Level  1; Scaling  2/ 2 Result_size  2/ 2
    Z2 += -0.5 * oe.contract('baij->abij',M6)
    
    del M6
    
    X6 = np.zeros([nvir, nvir, nocc, nocc], dtype=type_)
    
    #Contraction 82; Tree Level  2; Scaling  4/ 2 Result_size  2/ 2
    X6 += oe.contract('abik,kj->abij',T2,D2, optimize='optimal')
    
    del D2
    
    #Contraction 83; Tree Level  1; Scaling  2/ 2 Result_size  2/ 2
    Z2 += 0.5 * oe.contract('abij->abij',X6)
    
    #Contraction 84; Tree Level  1; Scaling  2/ 2 Result_size  2/ 2
    Z2 += -0.5 * oe.contract('abji->abij',X6)
    
    del X6
    
    Y6 = np.zeros([nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 85; Tree Level  3; Scaling  3/ 3 Result_size  3/ 1
    Y6 += oe.contract('akic,cj->akij',V5,T1, optimize='optimal')
    
    A12 = np.zeros([nocc, nvir, nocc, nocc], dtype=type_)
    
    #Contraction 86; Tree Level  2; Scaling  3/ 1 Result_size  3/ 1
    A12 += 4.0 * oe.contract('bkij->kbij',Y6)
    
    #Contraction 87; Tree Level  2; Scaling  3/ 1 Result_size  3/ 1
    A12 += -4.0 * oe.contract('kbij->kbij',X5)
    
    #Contraction 88; Tree Level  2; Scaling  3/ 1 Result_size  3/ 1
    A12 += -4.0 * oe.contract('bkji->kbij',Y6)
    
    #Contraction 89; Tree Level  2; Scaling  3/ 1 Result_size  3/ 1
    A13 += 4.0 * oe.contract('akij->akij',Y6)
    
    #Contraction 90; Tree Level  2; Scaling  3/ 1 Result_size  3/ 1
    A13 += -4.0 * oe.contract('akji->akij',Y6)
    
    #Contraction 91; Tree Level  2; Scaling  3/ 1 Result_size  3/ 1
    A12 += 4.0 * oe.contract('bkij->kbij',J6)
    
    I7 = np.zeros([nocc, nocc], dtype=type_)
    
    #Contraction 92; Tree Level  3; Scaling  4/ 2 Result_size  2/ 0
    I7 += oe.contract('dk,kldi->li',T1,G2, optimize='optimal')
    
    J7 = np.zeros([nvir, nvir, nocc, nocc], dtype=type_)
    
    #Contraction 93; Tree Level  2; Scaling  4/ 2 Result_size  2/ 2
    J7 += oe.contract('ablj,li->abji',T2,I7, optimize='optimal')
    
    #Contraction 94; Tree Level  1; Scaling  2/ 2 Result_size  2/ 2
    Z2 += oe.contract('abji->abij',J7)
    
    #Contraction 95; Tree Level  1; Scaling  2/ 2 Result_size  2/ 2
    Z2 += -1.0 * oe.contract('abij->abij',J7)
    
    del J7
    
    M7 = np.zeros([nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 96; Tree Level  3; Scaling  4/ 2 Result_size  4/ 0
    M7 += oe.contract('klic,cj->klij',V4,T1, optimize='optimal')
    
    J8 = np.zeros([nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 97; Tree Level  2; Scaling  4/ 0 Result_size  4/ 0
    J8 += 2.0 * oe.contract('klij->klij',M7)
    
    #Contraction 98; Tree Level  2; Scaling  4/ 0 Result_size  4/ 0
    J8 += -2.0 * oe.contract('klji->klij',M7)
    
    Y7 = np.zeros([nvir, nvir, nvir, nocc], dtype=type_)
    
    #Contraction 99; Tree Level  3; Scaling  1/ 5 Result_size  1/ 3
    Y7 += oe.contract('abcd,dj->abcj',V9,T1, optimize='optimal')
    
    A8 = np.zeros([nvir, nvir, nocc, nocc], dtype=type_)
    
    #Contraction 100; Tree Level  2; Scaling  2/ 4 Result_size  2/ 2
    A8 += oe.contract('ci,abcj->abij',T1,Y7, optimize='optimal')
    
    del Y7
    
    #Contraction 101; Tree Level  1; Scaling  2/ 2 Result_size  2/ 2
    Z2 += 0.5 * oe.contract('abij->abij',A8)
    
    #Contraction 102; Tree Level  1; Scaling  2/ 2 Result_size  2/ 2
    Z2 += -0.5 * oe.contract('abji->abij',A8)
    
    del A8
    
    D8 = np.zeros([nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 103; Tree Level  3; Scaling  4/ 2 Result_size  4/ 0
    D8 += oe.contract('dj,kldi->klji',T1,G2, optimize='optimal')
    
    #Contraction 104; Tree Level  2; Scaling  4/ 0 Result_size  4/ 0
    J8 += oe.contract('klji->klij',D8)
    
    #Contraction 105; Tree Level  2; Scaling  4/ 0 Result_size  4/ 0
    J8 += -1.0 * oe.contract('klij->klij',D8)
    
    #Contraction 106; Tree Level  2; Scaling  4/ 0 Result_size  4/ 0
    J8 += 2.0 * oe.contract('klij->klij',V1)
    
    #Contraction 107; Tree Level  2; Scaling  4/ 0 Result_size  4/ 0
    J8 += oe.contract('klij->klij',G4)
    
    #Contraction 108; Tree Level  1; Scaling  6/ 2 Result_size  2/ 2
    Z2 += 0.25 * oe.contract('abkl,klij->abij',T2,J8, optimize='optimal')
    
    del J8
    
    X8 = np.zeros([nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 109; Tree Level  3; Scaling  3/ 5 Result_size  3/ 1
    X8 += oe.contract('akcd,cdij->akij',V8,T2, optimize='optimal')
    
    Y8 = np.zeros([nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 110; Tree Level  3; Scaling  4/ 0 Result_size  4/ 0
    Y8 += oe.contract('klij->klij',G4)
    
    #Contraction 111; Tree Level  3; Scaling  4/ 0 Result_size  4/ 0
    Y8 += -1.0 * oe.contract('lkij->klij',G4)
    
    #Contraction 112; Tree Level  1; Scaling  4/ 4 Result_size  2/ 2
    Z2 += oe.contract('kc,abcijk->abij',F3,T3, optimize='optimal')
    
    D9 = np.zeros([nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 113; Tree Level  3; Scaling  5/ 1 Result_size  3/ 1
    D9 += oe.contract('al,lmji->amji',T1,D8, optimize='optimal')
    
    #Contraction 114; Tree Level  2; Scaling  3/ 1 Result_size  3/ 1
    A12 += -1.0 * oe.contract('bkji->kbij',D9)
    
    #Contraction 115; Tree Level  2; Scaling  3/ 1 Result_size  3/ 1
    A12 += oe.contract('bkij->kbij',D9)
    
    #Contraction 116; Tree Level  2; Scaling  3/ 1 Result_size  3/ 1
    A12 += -4.0 * oe.contract('kbij->kbij',G5)
    
    #Contraction 117; Tree Level  2; Scaling  3/ 1 Result_size  3/ 1
    A12 += 4.0 * oe.contract('kbji->kbij',G5)
    
    #Contraction 118; Tree Level  3; Scaling  4/ 0 Result_size  4/ 0
    A10 += -1.0 * oe.contract('lkij->klij',M7)
    
    #Contraction 119; Tree Level  3; Scaling  4/ 0 Result_size  4/ 0
    A10 += -1.0 * oe.contract('klji->klij',M7)
    
    #Contraction 120; Tree Level  3; Scaling  4/ 0 Result_size  4/ 0
    A10 += oe.contract('lkji->klij',M7)
    
    #Contraction 121; Tree Level  3; Scaling  4/ 0 Result_size  4/ 0
    A10 += oe.contract('klij->klij',M7)
    
    #Contraction 122; Tree Level  2; Scaling  5/ 1 Result_size  3/ 1
    A13 += -2.0 * oe.contract('al,lkij->akij',T1,A10, optimize='optimal')
    
    del A10
    
    #Contraction 123; Tree Level  2; Scaling  3/ 1 Result_size  3/ 1
    A13 += 4.0 * oe.contract('akij->akij',J6)
    
    del J6
    
    #Contraction 124; Tree Level  2; Scaling  3/ 1 Result_size  3/ 1
    A13 += -4.0 * oe.contract('kaij->akij',X5)
    
    del X5
    
    #Contraction 125; Tree Level  2; Scaling  3/ 1 Result_size  3/ 1
    A13 += -1.0 * oe.contract('akji->akij',D9)
    
    #Contraction 126; Tree Level  2; Scaling  3/ 1 Result_size  3/ 1
    A13 += oe.contract('akij->akij',D9)
    
    #Contraction 127; Tree Level  2; Scaling  5/ 1 Result_size  3/ 1
    A13 += -1.0 * oe.contract('al,lkij->akij',T1,Y8, optimize='optimal')
    
    del Y8
    
    A11 = np.zeros([nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 128; Tree Level  3; Scaling  3/ 3 Result_size  3/ 1
    A11 += oe.contract('ci,akcj->akij',T1,E5, optimize='optimal')
    
    del E5
    
    #Contraction 129; Tree Level  2; Scaling  3/ 1 Result_size  3/ 1
    A12 += 2.0 * oe.contract('bkij->kbij',A11)
    
    #Contraction 130; Tree Level  2; Scaling  3/ 1 Result_size  3/ 1
    A12 += -2.0 * oe.contract('bkji->kbij',A11)
    
    #Contraction 131; Tree Level  2; Scaling  3/ 1 Result_size  3/ 1
    A12 += -4.0 * oe.contract('bkji->kbij',Y5)
    
    #Contraction 132; Tree Level  2; Scaling  3/ 1 Result_size  3/ 1
    A12 += 4.0 * oe.contract('bkij->kbij',Y5)
    
    #Contraction 133; Tree Level  2; Scaling  3/ 1 Result_size  3/ 1
    A13 += 2.0 * oe.contract('akij->akij',A11)
    
    #Contraction 134; Tree Level  2; Scaling  3/ 1 Result_size  3/ 1
    A13 += -2.0 * oe.contract('akji->akij',A11)
    
    #Contraction 135; Tree Level  2; Scaling  3/ 1 Result_size  3/ 1
    A12 += 2.0 * oe.contract('bkij->kbij',X8)
    
    #Contraction 136; Tree Level  2; Scaling  3/ 1 Result_size  3/ 1
    A12 += 2.0 * oe.contract('kbij->kbij',A5)
    
    #Contraction 137; Tree Level  2; Scaling  3/ 1 Result_size  3/ 1
    A12 += 4.0 * oe.contract('bkij->kbij',V2)
    
    #Contraction 138; Tree Level  1; Scaling  4/ 2 Result_size  2/ 2
    Z2 += 0.25 * oe.contract('ak,kbij->abij',T1,A12, optimize='optimal')
    
    del A12
    
    E12 = np.zeros([nvir, nvir, nocc, nocc], dtype=type_)
    
    #Contraction 139; Tree Level  2; Scaling  4/ 4 Result_size  2/ 2
    E12 += oe.contract('akic,bcjk->abij',V5,T2, optimize='optimal')
    
    #Contraction 140; Tree Level  1; Scaling  2/ 2 Result_size  2/ 2
    Z2 += oe.contract('abij->abij',E12)
    
    #Contraction 141; Tree Level  1; Scaling  2/ 2 Result_size  2/ 2
    Z2 += -1.0 * oe.contract('baij->abij',E12)
    
    #Contraction 142; Tree Level  1; Scaling  2/ 2 Result_size  2/ 2
    Z2 += -1.0 * oe.contract('abji->abij',E12)
    
    #Contraction 143; Tree Level  1; Scaling  2/ 2 Result_size  2/ 2
    Z2 += oe.contract('baji->abij',E12)
    
    del E12
    
    G12 = np.zeros([nvir, nvir, nocc, nocc], dtype=type_)
    
    #Contraction 144; Tree Level  2; Scaling  4/ 2 Result_size  2/ 2
    G12 += oe.contract('abkj,ki->abji',T2,J2, optimize='optimal')
    
    #Contraction 145; Tree Level  1; Scaling  2/ 2 Result_size  2/ 2
    Z2 += -1.0 * oe.contract('abji->abij',G12)
    
    #Contraction 146; Tree Level  1; Scaling  2/ 2 Result_size  2/ 2
    Z2 += oe.contract('abij->abij',G12)
    
    del G12
    
    I12 = np.zeros([nvir, nvir, nocc, nocc], dtype=type_)
    
    #Contraction 147; Tree Level  2; Scaling  4/ 2 Result_size  2/ 2
    I12 += oe.contract('ki,bajk->baij',F1,T2, optimize='optimal')
    
    #Contraction 148; Tree Level  1; Scaling  2/ 2 Result_size  2/ 2
    Z2 += -1.0 * oe.contract('baij->abij',I12)
    
    #Contraction 149; Tree Level  1; Scaling  2/ 2 Result_size  2/ 2
    Z2 += oe.contract('baji->abij',I12)
    
    del I12
    
    #Contraction 150; Tree Level  2; Scaling  3/ 1 Result_size  3/ 1
    A13 += 2.0 * oe.contract('akij->akij',X8)
    
    #Contraction 151; Tree Level  2; Scaling  3/ 1 Result_size  3/ 1
    A13 += 2.0 * oe.contract('kaij->akij',A5)
    
    #Contraction 152; Tree Level  2; Scaling  3/ 1 Result_size  3/ 1
    A13 += 4.0 * oe.contract('akij->akij',V2)
    
    #Contraction 153; Tree Level  1; Scaling  4/ 2 Result_size  2/ 2
    Z2 += -0.25 * oe.contract('bk,akij->abij',T1,A13, optimize='optimal')
    
    del A13
    
    #Contraction 154; Tree Level  1; Scaling  2/ 2 Result_size  2/ 2
    Z2 += oe.contract('abij->abij',V3)
    
    #del V3
    
    E13 = np.zeros([nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 155; Tree Level  3; Scaling  5/ 1 Result_size  3/ 1
    E13 += oe.contract('al,lmij->amij',T1,M7, optimize='optimal')
    
    G13 = np.zeros([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 156; Tree Level  2; Scaling  7/ 3 Result_size  3/ 3
    G13 += oe.contract('abclmk,lmij->abckij',T3,G4, optimize='optimal')
    
    #Contraction 157; Tree Level  0; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.25 * oe.contract('abckij->abcijk',G13)
    
    #Contraction 158; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.25 * oe.contract('abcjik->abcijk',G13)
    
    #Contraction 159; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.25 * oe.contract('abcijk->abcijk',G13)
    
    del G13
    
    I13 = np.zeros([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 160; Tree Level  2; Scaling  7/ 3 Result_size  3/ 3
    I13 += oe.contract('abclmk,lmji->abckji',T3,D8, optimize='optimal')
    
    del D8
    
    #Contraction 161; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.25 * oe.contract('abckji->abcijk',I13)
    
    #Contraction 162; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.25 * oe.contract('abcjki->abcijk',I13)
    
    #Contraction 163; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.25 * oe.contract('abckij->abcijk',I13)
    
    #Contraction 164; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.25 * oe.contract('abcikj->abcijk',I13)
    
    #Contraction 165; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.25 * oe.contract('abcjik->abcijk',I13)
    
    #Contraction 166; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.25 * oe.contract('abcijk->abcijk',I13)
    
    del I13
    
    J13 = np.zeros([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 167; Tree Level  2; Scaling  5/ 3 Result_size  3/ 3
    J13 += oe.contract('bclk,alij->bcakij',T2,Y6, optimize='optimal')
    
    del Y6
    
    #Contraction 168; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('bcakij->abcijk',J13)
    
    #Contraction 169; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('acbkij->abcijk',J13)
    
    #Contraction 170; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('abckij->abcijk',J13)
    
    #Contraction 171; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('bcajik->abcijk',J13)
    
    #Contraction 172; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('acbjik->abcijk',J13)
    
    #Contraction 173; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('abcjik->abcijk',J13)
    
    #Contraction 174; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('bcakji->abcijk',J13)
    
    #Contraction 175; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('acbkji->abcijk',J13)
    
    #Contraction 176; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('abckji->abcijk',J13)
    
    #Contraction 177; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('bcaijk->abcijk',J13)
    
    #Contraction 178; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('acbijk->abcijk',J13)
    
    #Contraction 179; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('abcijk->abcijk',J13)
    
    #Contraction 180; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('bcajki->abcijk',J13)
    
    #Contraction 181; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('acbjki->abcijk',J13)
    
    #Contraction 182; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('abcjki->abcijk',J13)
    
    #Contraction 183; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('bcaikj->abcijk',J13)
    
    #Contraction 184; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('acbikj->abcijk',J13)
    
    #Contraction 185; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('abcikj->abcijk',J13)
    
    del J13
    
    M13 = np.zeros([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 186; Tree Level  2; Scaling  5/ 5 Result_size  3/ 3
    M13 += oe.contract('ebcljk,alei->bcajki',T3,J4, optimize='optimal')
    
    #Contraction 187; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('bcajki->abcijk',M13)
    
    #Contraction 188; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('acbjki->abcijk',M13)
    
    #Contraction 189; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('abcjki->abcijk',M13)
    
    #Contraction 190; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('bcaikj->abcijk',M13)
    
    #Contraction 191; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('acbikj->abcijk',M13)
    
    #Contraction 192; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('abcikj->abcijk',M13)
    
    #Contraction 193; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('bcaijk->abcijk',M13)
    
    #Contraction 194; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('acbijk->abcijk',M13)
    
    #Contraction 195; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('abcijk->abcijk',M13)
    
    del M13
    
    X13 = np.zeros([nvir, nocc, nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 196; Tree Level  4; Scaling  5/ 3 Result_size  5/ 1
    X13 += oe.contract('ecjk,lmei->clmjki',T2,G2, optimize='optimal')
    
    Y13 = np.zeros([nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 197; Tree Level  3; Scaling  6/ 2 Result_size  4/ 2
    Y13 += oe.contract('bm,clmjki->bcljki',T1,X13, optimize='optimal')
    
    A14 = np.zeros([nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 198; Tree Level  3; Scaling  3/ 3 Result_size  3/ 1
    A14 += oe.contract('ebij,me->bmij',T2,A1, optimize='optimal')
    
    del A1
    
    D14 = np.zeros([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 199; Tree Level  2; Scaling  5/ 3 Result_size  3/ 3
    D14 += oe.contract('bcmk,amij->bcakij',T2,E13, optimize='optimal')
    
    del E13
    
    #Contraction 200; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('bcakij->abcijk',D14)
    
    #Contraction 201; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('acbkij->abcijk',D14)
    
    #Contraction 202; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('abckij->abcijk',D14)
    
    #Contraction 203; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('bcajik->abcijk',D14)
    
    #Contraction 204; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('acbjik->abcijk',D14)
    
    #Contraction 205; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('abcjik->abcijk',D14)
    
    #Contraction 206; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('bcakji->abcijk',D14)
    
    #Contraction 207; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('acbkji->abcijk',D14)
    
    #Contraction 208; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('abckji->abcijk',D14)
    
    #Contraction 209; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('bcaijk->abcijk',D14)
    
    #Contraction 210; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('acbijk->abcijk',D14)
    
    #Contraction 211; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('abcijk->abcijk',D14)
    
    #Contraction 212; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('bcajki->abcijk',D14)
    
    #Contraction 213; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('acbjki->abcijk',D14)
    
    #Contraction 214; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('abcjki->abcijk',D14)
    
    #Contraction 215; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('bcaikj->abcijk',D14)
    
    #Contraction 216; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('acbikj->abcijk',D14)
    
    #Contraction 217; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('abcikj->abcijk',D14)
    
    del D14
    
    E14 = np.zeros([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 218; Tree Level  2; Scaling  3/ 5 Result_size  3/ 3
    E14 += oe.contract('ad,bcdjki->abcjki',F4,T3, optimize='optimal')
    
    #del F4
    
    #Contraction 219; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('abcjki->abcijk',E14)
    
    #Contraction 220; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('bacjki->abcijk',E14)
    
    #Contraction 221; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('cabjki->abcijk',E14)
    
    del E14
    
    G14 = np.zeros([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 222; Tree Level  2; Scaling  5/ 3 Result_size  3/ 3
    G14 += oe.contract('al,bcljki->abcjki',T1,Y13, optimize='optimal')
    
    del Y13
    
    #Contraction 223; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * oe.contract('abcjki->abcijk',G14)
    
    #Contraction 224; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.5 * oe.contract('acbjki->abcijk',G14)
    
    #Contraction 225; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.5 * oe.contract('bacjki->abcijk',G14)
    
    #Contraction 226; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * oe.contract('bcajki->abcijk',G14)
    
    #Contraction 227; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * oe.contract('cabjki->abcijk',G14)
    
    #Contraction 228; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.5 * oe.contract('cbajki->abcijk',G14)
    
    #Contraction 229; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.5 * oe.contract('abcikj->abcijk',G14)
    
    #Contraction 230; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * oe.contract('acbikj->abcijk',G14)
    
    #Contraction 231; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * oe.contract('bacikj->abcijk',G14)
    
    #Contraction 232; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.5 * oe.contract('bcaikj->abcijk',G14)
    
    #Contraction 233; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.5 * oe.contract('cabikj->abcijk',G14)
    
    #Contraction 234; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * oe.contract('cbaikj->abcijk',G14)
    
    #Contraction 235; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * oe.contract('abcijk->abcijk',G14)
    
    #Contraction 236; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.5 * oe.contract('acbijk->abcijk',G14)
    
    #Contraction 237; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.5 * oe.contract('bacijk->abcijk',G14)
    
    #Contraction 238; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * oe.contract('bcaijk->abcijk',G14)
    
    #Contraction 239; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * oe.contract('cabijk->abcijk',G14)
    
    #Contraction 240; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.5 * oe.contract('cbaijk->abcijk',G14)
    
    del G14
    
    I14 = np.zeros([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 241; Tree Level  2; Scaling  5/ 3 Result_size  3/ 3
    I14 += oe.contract('bclk,alij->bcakij',T2,X8, optimize='optimal')
    
    del X8
    
    #Contraction 242; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.5 * oe.contract('bcakij->abcijk',I14)
    
    #Contraction 243; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * oe.contract('acbkij->abcijk',I14)
    
    #Contraction 244; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.5 * oe.contract('abckij->abcijk',I14)
    
    #Contraction 245; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * oe.contract('bcajik->abcijk',I14)
    
    #Contraction 246; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.5 * oe.contract('acbjik->abcijk',I14)
    
    #Contraction 247; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * oe.contract('abcjik->abcijk',I14)
    
    #Contraction 248; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.5 * oe.contract('bcaijk->abcijk',I14)
    
    #Contraction 249; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * oe.contract('acbijk->abcijk',I14)
    
    #Contraction 250; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.5 * oe.contract('abcijk->abcijk',I14)
    
    del I14
    
    J14 = np.zeros([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 251; Tree Level  2; Scaling  5/ 3 Result_size  3/ 3
    J14 += oe.contract('abcmjk,mi->abcjki',T3,I7, optimize='optimal')
    
    del I7
    
    #Contraction 252; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('abcjki->abcijk',J14)
    
    #Contraction 253; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('abcikj->abcijk',J14)
    
    #Contraction 254; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('abcijk->abcijk',J14)
    
    del J14
    
    M14 = np.zeros([nocc, nvir, nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 255; Tree Level  3; Scaling  6/ 4 Result_size  4/ 2
    M14 += oe.contract('lmid,dbcmjk->lbcijk',V4,T3, optimize='optimal')
    
    X14 = np.zeros([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 256; Tree Level  2; Scaling  5/ 3 Result_size  3/ 3
    X14 += oe.contract('al,lbcijk->abcijk',T1,M14, optimize='optimal')
    
    del M14
    
    #Contraction 257; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('abcijk->abcijk',X14)
    
    #Contraction 258; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('bacijk->abcijk',X14)
    
    #Contraction 259; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('cabijk->abcijk',X14)
    
    #Contraction 260; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('abcjik->abcijk',X14)
    
    #Contraction 261; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('bacjik->abcijk',X14)
    
    #Contraction 262; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('cabjik->abcijk',X14)
    
    #Contraction 263; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('abckij->abcijk',X14)
    
    #Contraction 264; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('backij->abcijk',X14)
    
    #Contraction 265; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('cabkij->abcijk',X14)
    
    del X14
    
    Y14 = np.zeros([nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 266; Tree Level  3; Scaling  4/ 4 Result_size  4/ 2
    Y14 += oe.contract('ebcijk,le->bclijk',T3,I6, optimize='optimal')
    
    del I6
    
    A15 = np.zeros([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 267; Tree Level  2; Scaling  5/ 3 Result_size  3/ 3
    A15 += oe.contract('ablj,clki->abcjki',T2,Y5, optimize='optimal')
    
    del Y5
    
    #Contraction 268; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('abcjki->abcijk',A15)
    
    #Contraction 269; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('acbjki->abcijk',A15)
    
    #Contraction 270; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('bcajki->abcijk',A15)
    
    #Contraction 271; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('abckji->abcijk',A15)
    
    #Contraction 272; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('acbkji->abcijk',A15)
    
    #Contraction 273; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('bcakji->abcijk',A15)
    
    #Contraction 274; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('abcikj->abcijk',A15)
    
    #Contraction 275; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('acbikj->abcijk',A15)
    
    #Contraction 276; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('bcaikj->abcijk',A15)
    
    #Contraction 277; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('abckij->abcijk',A15)
    
    #Contraction 278; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('acbkij->abcijk',A15)
    
    #Contraction 279; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('bcakij->abcijk',A15)
    
    #Contraction 280; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('abcijk->abcijk',A15)
    
    #Contraction 281; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('acbijk->abcijk',A15)
    
    #Contraction 282; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('bcaijk->abcijk',A15)
    
    #Contraction 283; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('abcjik->abcijk',A15)
    
    #Contraction 284; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('acbjik->abcijk',A15)
    
    #Contraction 285; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('bcajik->abcijk',A15)
    
    del A15
    
    D15 = np.zeros([nocc, nvir, nocc, nocc], dtype=type_)
    
    #Contraction 286; Tree Level  3; Scaling  5/ 1 Result_size  3/ 1
    D15 += oe.contract('lmij,al->maij',V1,T1, optimize='optimal')
    
    E15 = np.zeros([nocc, nocc, nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 287; Tree Level  4; Scaling  5/ 3 Result_size  5/ 1
    E15 += oe.contract('lmid,dcjk->lmcijk',V4,T2, optimize='optimal')
    
    G15 = np.zeros([nvir, nocc, nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 288; Tree Level  3; Scaling  6/ 2 Result_size  4/ 2
    G15 += oe.contract('bm,lmcijk->blcijk',T1,E15, optimize='optimal')
    
    del E15
    
    I15 = np.zeros([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 289; Tree Level  2; Scaling  5/ 3 Result_size  3/ 3
    I15 += oe.contract('al,blcijk->abcijk',T1,G15, optimize='optimal')
    
    del G15
    
    #Contraction 290; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * oe.contract('abcijk->abcijk',I15)
    
    #Contraction 291; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.5 * oe.contract('acbijk->abcijk',I15)
    
    #Contraction 292; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.5 * oe.contract('bacijk->abcijk',I15)
    
    #Contraction 293; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * oe.contract('bcaijk->abcijk',I15)
    
    #Contraction 294; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * oe.contract('cabijk->abcijk',I15)
    
    #Contraction 295; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.5 * oe.contract('cbaijk->abcijk',I15)
    
    #Contraction 296; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.5 * oe.contract('abcjik->abcijk',I15)
    
    #Contraction 297; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * oe.contract('acbjik->abcijk',I15)
    
    #Contraction 298; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * oe.contract('bacjik->abcijk',I15)
    
    #Contraction 299; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.5 * oe.contract('bcajik->abcijk',I15)
    
    #Contraction 300; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.5 * oe.contract('cabjik->abcijk',I15)
    
    #Contraction 301; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * oe.contract('cbajik->abcijk',I15)
    
    #Contraction 302; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * oe.contract('abckij->abcijk',I15)
    
    #Contraction 303; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.5 * oe.contract('acbkij->abcijk',I15)
    
    #Contraction 304; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.5 * oe.contract('backij->abcijk',I15)
    
    #Contraction 305; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * oe.contract('bcakij->abcijk',I15)
    
    #Contraction 306; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * oe.contract('cabkij->abcijk',I15)
    
    #Contraction 307; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.5 * oe.contract('cbakij->abcijk',I15)
    
    del I15
    
    J15 = np.zeros([nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 308; Tree Level  3; Scaling  4/ 4 Result_size  4/ 2
    J15 += oe.contract('ecjk,alei->caljki',T2,J4, optimize='optimal')
    
    del J4
    
    M15 = np.zeros([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 309; Tree Level  2; Scaling  5/ 3 Result_size  3/ 3
    M15 += oe.contract('alij,cbkl->acbijk',V2,T2, optimize='optimal')
    
    #del V2
    
    #Contraction 310; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('acbijk->abcijk',M15)
    
    #Contraction 311; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('cabijk->abcijk',M15)
    
    #Contraction 312; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('bacijk->abcijk',M15)
    
    #Contraction 313; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('acbikj->abcijk',M15)
    
    #Contraction 314; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('cabikj->abcijk',M15)
    
    #Contraction 315; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('bacikj->abcijk',M15)
    
    #Contraction 316; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('acbjki->abcijk',M15)
    
    #Contraction 317; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('cabjki->abcijk',M15)
    
    #Contraction 318; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('bacjki->abcijk',M15)
    
    del M15
    
    X15 = np.zeros([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 319; Tree Level  2; Scaling  5/ 3 Result_size  3/ 3
    X15 += oe.contract('abil,lcjk->abcijk',T2,A5, optimize='optimal')
    
    del A5
    
    #Contraction 320; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * oe.contract('abcijk->abcijk',X15)
    
    #Contraction 321; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.5 * oe.contract('acbijk->abcijk',X15)
    
    #Contraction 322; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * oe.contract('bcaijk->abcijk',X15)
    
    #Contraction 323; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.5 * oe.contract('abcjik->abcijk',X15)
    
    #Contraction 324; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * oe.contract('acbjik->abcijk',X15)
    
    #Contraction 325; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.5 * oe.contract('bcajik->abcijk',X15)
    
    #Contraction 326; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * oe.contract('abckij->abcijk',X15)
    
    #Contraction 327; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.5 * oe.contract('acbkij->abcijk',X15)
    
    #Contraction 328; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * oe.contract('bcakij->abcijk',X15)
    
    del X15
    
    Y15 = np.zeros([nvir, nocc, nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 329; Tree Level  3; Scaling  4/ 4 Result_size  4/ 2
    Y15 += oe.contract('alid,dcjk->alcijk',V5,T2, optimize='optimal')
    
    A16 = np.zeros([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 330; Tree Level  2; Scaling  5/ 3 Result_size  3/ 3
    A16 += oe.contract('bcmk,maij->bcakij',T2,D15, optimize='optimal')
    
    del D15
    
    #Contraction 331; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('bcakij->abcijk',A16)
    
    #Contraction 332; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('acbkij->abcijk',A16)
    
    #Contraction 333; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('abckij->abcijk',A16)
    
    #Contraction 334; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('bcajik->abcijk',A16)
    
    #Contraction 335; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('acbjik->abcijk',A16)
    
    #Contraction 336; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('abcjik->abcijk',A16)
    
    #Contraction 337; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('bcaijk->abcijk',A16)
    
    #Contraction 338; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('acbijk->abcijk',A16)
    
    #Contraction 339; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('abcijk->abcijk',A16)
    
    del A16
    
    D16 = np.zeros([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 340; Tree Level  2; Scaling  3/ 5 Result_size  3/ 3
    D16 += oe.contract('ebcijk,ae->bcaijk',T3,D6, optimize='optimal')
    
    del D6
    
    #Contraction 341; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('bcaijk->abcijk',D16)
    
    #Contraction 342; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('acbijk->abcijk',D16)
    
    #Contraction 343; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('abcijk->abcijk',D16)
    
    del D16
    
    E16 = np.zeros([nvir, nvir, nvir, nocc], dtype=type_)
    
    #Contraction 344; Tree Level  3; Scaling  3/ 5 Result_size  1/ 3
    E16 += oe.contract('alde,eclk->acdk',V8,T2, optimize='optimal')
    
    G16 = np.zeros([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 345; Tree Level  2; Scaling  3/ 5 Result_size  3/ 3
    G16 += oe.contract('bdji,acdk->bacjik',T2,E16, optimize='optimal')
    
    del E16
    
    #Contraction 346; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('bacjik->abcijk',G16)
    
    #Contraction 347; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('cabjik->abcijk',G16)
    
    #Contraction 348; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('abcjik->abcijk',G16)
    
    #Contraction 349; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('cbajik->abcijk',G16)
    
    #Contraction 350; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('acbjik->abcijk',G16)
    
    #Contraction 351; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('bcajik->abcijk',G16)
    
    #Contraction 352; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('bacjki->abcijk',G16)
    
    #Contraction 353; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('cabjki->abcijk',G16)
    
    #Contraction 354; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('abcjki->abcijk',G16)
    
    #Contraction 355; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('cbajki->abcijk',G16)
    
    #Contraction 356; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('acbjki->abcijk',G16)
    
    #Contraction 357; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('bcajki->abcijk',G16)
    
    #Contraction 358; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('bacikj->abcijk',G16)
    
    #Contraction 359; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('cabikj->abcijk',G16)
    
    #Contraction 360; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('abcikj->abcijk',G16)
    
    #Contraction 361; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('cbaikj->abcijk',G16)
    
    #Contraction 362; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('acbikj->abcijk',G16)
    
    #Contraction 363; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('bcaikj->abcijk',G16)
    
    del G16
    
    I16 = np.zeros([nocc, nvir, nocc, nocc], dtype=type_)
    
    #Contraction 364; Tree Level  3; Scaling  3/ 3 Result_size  3/ 1
    I16 += oe.contract('ld,adij->laij',F3,T2, optimize='optimal')
    
    J16 = np.zeros([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 365; Tree Level  2; Scaling  5/ 3 Result_size  3/ 3
    J16 += oe.contract('bclk,laij->bcakij',T2,I16, optimize='optimal')
    
    del I16
    
    #Contraction 366; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('bcakij->abcijk',J16)
    
    #Contraction 367; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('acbkij->abcijk',J16)
    
    #Contraction 368; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('abckij->abcijk',J16)
    
    #Contraction 369; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('bcajik->abcijk',J16)
    
    #Contraction 370; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('acbjik->abcijk',J16)
    
    #Contraction 371; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('abcjik->abcijk',J16)
    
    #Contraction 372; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('bcaijk->abcijk',J16)
    
    #Contraction 373; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('acbijk->abcijk',J16)
    
    #Contraction 374; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('abcijk->abcijk',J16)
    
    del J16
    
    M16 = np.zeros([nocc, nocc, nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 375; Tree Level  4; Scaling  5/ 5 Result_size  5/ 1
    M16 += oe.contract('lmde,decijk->lmcijk',V7,T3, optimize='optimal')
    
    X16 = np.zeros([nvir, nocc, nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 376; Tree Level  3; Scaling  6/ 2 Result_size  4/ 2
    X16 += oe.contract('bm,lmcijk->blcijk',T1,M16, optimize='optimal')
    
    Y16 = np.zeros([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 377; Tree Level  2; Scaling  5/ 3 Result_size  3/ 3
    Y16 += oe.contract('al,blcijk->abcijk',T1,X16, optimize='optimal')
    
    del X16
    
    #Contraction 378; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.25 * oe.contract('abcijk->abcijk',Y16)
    
    #Contraction 379; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.25 * oe.contract('acbijk->abcijk',Y16)
    
    #Contraction 380; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.25 * oe.contract('bacijk->abcijk',Y16)
    
    #Contraction 381; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.25 * oe.contract('bcaijk->abcijk',Y16)
    
    #Contraction 382; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.25 * oe.contract('cabijk->abcijk',Y16)
    
    #Contraction 383; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.25 * oe.contract('cbaijk->abcijk',Y16)
    
    del Y16
    
    A17 = np.zeros([nocc, nocc], dtype=type_)
    
    #Contraction 384; Tree Level  3; Scaling  4/ 4 Result_size  2/ 0
    A17 += oe.contract('lmde,deil->mi',V7,T2, optimize='optimal')
    
    D17 = np.zeros([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 385; Tree Level  2; Scaling  5/ 3 Result_size  3/ 3
    D17 += oe.contract('li,bcajkl->bcaijk',F1,T3, optimize='optimal')
    
    #del F1
    
    #Contraction 386; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('bcaijk->abcijk',D17)
    
    #Contraction 387; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('bcajik->abcijk',D17)
    
    #Contraction 388; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('bcakij->abcijk',D17)
    
    del D17
    
    E17 = np.zeros([nvir, nvir, nvir, nocc], dtype=type_)
    
    #Contraction 389; Tree Level  3; Scaling  1/ 5 Result_size  1/ 3
    E17 += oe.contract('abde,di->abei',V9,T1, optimize='optimal')
    
    G17 = np.zeros([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 390; Tree Level  2; Scaling  3/ 5 Result_size  3/ 3
    G17 += oe.contract('ecjk,abei->cabjki',T2,E17, optimize='optimal')
    
    del E17
    
    #Contraction 391; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('cabjki->abcijk',G17)
    
    #Contraction 392; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('bacjki->abcijk',G17)
    
    #Contraction 393; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('abcjki->abcijk',G17)
    
    #Contraction 394; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('cabikj->abcijk',G17)
    
    #Contraction 395; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('bacikj->abcijk',G17)
    
    #Contraction 396; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('abcikj->abcijk',G17)
    
    #Contraction 397; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('cabijk->abcijk',G17)
    
    #Contraction 398; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('bacijk->abcijk',G17)
    
    #Contraction 399; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('abcijk->abcijk',G17)
    
    del G17
    
    I17 = np.zeros([nvir, nvir, nvir, nocc], dtype=type_)
    
    #Contraction 400; Tree Level  3; Scaling  5/ 5 Result_size  1/ 3
    I17 += oe.contract('lmde,ebclmk->bcdk',V7,T3, optimize='optimal')
    
    J17 = np.zeros([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 401; Tree Level  2; Scaling  3/ 5 Result_size  3/ 3
    J17 += oe.contract('adij,bcdk->abcijk',T2,I17, optimize='optimal')
    
    del I17
    
    #Contraction 402; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * oe.contract('abcijk->abcijk',J17)
    
    #Contraction 403; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.5 * oe.contract('bacijk->abcijk',J17)
    
    #Contraction 404; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * oe.contract('cabijk->abcijk',J17)
    
    #Contraction 405; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.5 * oe.contract('abcikj->abcijk',J17)
    
    #Contraction 406; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * oe.contract('bacikj->abcijk',J17)
    
    #Contraction 407; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.5 * oe.contract('cabikj->abcijk',J17)
    
    #Contraction 408; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * oe.contract('abcjki->abcijk',J17)
    
    #Contraction 409; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.5 * oe.contract('bacjki->abcijk',J17)
    
    #Contraction 410; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * oe.contract('cabjki->abcijk',J17)
    
    del J17
    
    M17 = np.zeros([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 411; Tree Level  2; Scaling  7/ 3 Result_size  3/ 3
    M17 += oe.contract('lmij,cabklm->cabijk',V1,T3, optimize='optimal')
    
    #del V1
    
    #Contraction 412; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * oe.contract('cabijk->abcijk',M17)
    
    #Contraction 413; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.5 * oe.contract('cabikj->abcijk',M17)
    
    #Contraction 414; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * oe.contract('cabjki->abcijk',M17)
    
    del M17
    
    X17 = np.zeros([nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 415; Tree Level  3; Scaling  5/ 1 Result_size  3/ 1
    X17 += oe.contract('al,lmij->amij',T1,G4, optimize='optimal')
    
    del G4
    
    Y17 = np.zeros([nocc, nvir, nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 416; Tree Level  3; Scaling  4/ 4 Result_size  4/ 2
    Y17 += oe.contract('ld,dbcijk->lbcijk',F3,T3, optimize='optimal')
    
    #del F3
    
    A18 = np.zeros([nocc, nvir, nvir, nocc], dtype=type_)
    
    #Contraction 417; Tree Level  3; Scaling  4/ 4 Result_size  2/ 2
    A18 += oe.contract('lmde,adil->maei',V7,T2, optimize='optimal')
    
    D18 = np.zeros([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 418; Tree Level  2; Scaling  5/ 5 Result_size  3/ 3
    D18 += oe.contract('ebcmjk,maei->bcajki',T3,A18, optimize='optimal')
    
    del A18
    
    #Contraction 419; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('bcajki->abcijk',D18)
    
    #Contraction 420; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('acbjki->abcijk',D18)
    
    #Contraction 421; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('abcjki->abcijk',D18)
    
    #Contraction 422; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('bcaikj->abcijk',D18)
    
    #Contraction 423; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('acbikj->abcijk',D18)
    
    #Contraction 424; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('abcikj->abcijk',D18)
    
    #Contraction 425; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('bcaijk->abcijk',D18)
    
    #Contraction 426; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('acbijk->abcijk',D18)
    
    #Contraction 427; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('abcijk->abcijk',D18)
    
    del D18
    
    E18 = np.zeros([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 428; Tree Level  2; Scaling  3/ 7 Result_size  3/ 3
    E18 += oe.contract('abde,cdekij->abckij',V9,T3, optimize='optimal')
    
    #del V9
    
    #Contraction 429; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * oe.contract('abckij->abcijk',E18)
    
    #Contraction 430; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.5 * oe.contract('acbkij->abcijk',E18)
    
    #Contraction 431; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * oe.contract('bcakij->abcijk',E18)
    
    del E18
    
    G18 = np.zeros([nvir, nvir, nocc, nvir], dtype=type_)
    
    #Contraction 432; Tree Level  3; Scaling  5/ 3 Result_size  1/ 3
    G18 += oe.contract('lmid,aclm->acid',V4,T2, optimize='optimal')
    
    #del V4
    
    I18 = np.zeros([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 433; Tree Level  2; Scaling  3/ 5 Result_size  3/ 3
    I18 += oe.contract('bdjk,acid->bacjki',T2,G18, optimize='optimal')
    
    del G18
    
    #Contraction 434; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * oe.contract('bacjki->abcijk',I18)
    
    #Contraction 435; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.5 * oe.contract('abcjki->abcijk',I18)
    
    #Contraction 436; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * oe.contract('cbajki->abcijk',I18)
    
    #Contraction 437; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.5 * oe.contract('bacikj->abcijk',I18)
    
    #Contraction 438; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * oe.contract('abcikj->abcijk',I18)
    
    #Contraction 439; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.5 * oe.contract('cbaikj->abcijk',I18)
    
    #Contraction 440; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * oe.contract('bacijk->abcijk',I18)
    
    #Contraction 441; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.5 * oe.contract('abcijk->abcijk',I18)
    
    #Contraction 442; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * oe.contract('cbaijk->abcijk',I18)
    
    del I18
    
    J18 = np.zeros([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 443; Tree Level  2; Scaling  5/ 3 Result_size  3/ 3
    J18 += oe.contract('bcmk,amji->bcakji',T2,D9, optimize='optimal')
    
    del D9
    
    #Contraction 444; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * oe.contract('bcakji->abcijk',J18)
    
    #Contraction 445; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.5 * oe.contract('acbkji->abcijk',J18)
    
    #Contraction 446; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * oe.contract('abckji->abcijk',J18)
    
    #Contraction 447; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.5 * oe.contract('bcajki->abcijk',J18)
    
    #Contraction 448; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * oe.contract('acbjki->abcijk',J18)
    
    #Contraction 449; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.5 * oe.contract('abcjki->abcijk',J18)
    
    #Contraction 450; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.5 * oe.contract('bcakij->abcijk',J18)
    
    #Contraction 451; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * oe.contract('acbkij->abcijk',J18)
    
    #Contraction 452; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.5 * oe.contract('abckij->abcijk',J18)
    
    #Contraction 453; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * oe.contract('bcaikj->abcijk',J18)
    
    #Contraction 454; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.5 * oe.contract('acbikj->abcijk',J18)
    
    #Contraction 455; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * oe.contract('abcikj->abcijk',J18)
    
    #Contraction 456; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * oe.contract('bcajik->abcijk',J18)
    
    #Contraction 457; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.5 * oe.contract('acbjik->abcijk',J18)
    
    #Contraction 458; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * oe.contract('abcjik->abcijk',J18)
    
    #Contraction 459; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.5 * oe.contract('bcaijk->abcijk',J18)
    
    #Contraction 460; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * oe.contract('acbijk->abcijk',J18)
    
    #Contraction 461; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.5 * oe.contract('abcijk->abcijk',J18)
    
    del J18
    
    M18 = np.zeros([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 462; Tree Level  2; Scaling  5/ 3 Result_size  3/ 3
    M18 += oe.contract('acmk,bmij->acbkij',T2,A14, optimize='optimal')
    
    del A14
    
    #Contraction 463; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('acbkij->abcijk',M18)
    
    #Contraction 464; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('bcakij->abcijk',M18)
    
    #Contraction 465; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('backij->abcijk',M18)
    
    #Contraction 466; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('acbjik->abcijk',M18)
    
    #Contraction 467; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('bcajik->abcijk',M18)
    
    #Contraction 468; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('bacjik->abcijk',M18)
    
    #Contraction 469; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('acbijk->abcijk',M18)
    
    #Contraction 470; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('bcaijk->abcijk',M18)
    
    #Contraction 471; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('bacijk->abcijk',M18)
    
    del M18
    
    X18 = np.zeros([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 472; Tree Level  2; Scaling  7/ 3 Result_size  3/ 3
    X18 += oe.contract('ablm,lmcijk->abcijk',T2,M16, optimize='optimal')
    
    del M16
    
    #Contraction 473; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.25 * oe.contract('abcijk->abcijk',X18)
    
    #Contraction 474; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.25 * oe.contract('acbijk->abcijk',X18)
    
    #Contraction 475; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.25 * oe.contract('bcaijk->abcijk',X18)
    
    del X18
    
    Y18 = np.zeros([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 476; Tree Level  2; Scaling  5/ 3 Result_size  3/ 3
    Y18 += oe.contract('bl,caljki->bcajki',T1,J15, optimize='optimal')
    
    del J15
    
    #Contraction 477; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('bcajki->abcijk',Y18)
    
    #Contraction 478; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('cbajki->abcijk',Y18)
    
    #Contraction 479; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('acbjki->abcijk',Y18)
    
    #Contraction 480; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('cabjki->abcijk',Y18)
    
    #Contraction 481; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('abcjki->abcijk',Y18)
    
    #Contraction 482; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('bacjki->abcijk',Y18)
    
    #Contraction 483; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('bcaikj->abcijk',Y18)
    
    #Contraction 484; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('cbaikj->abcijk',Y18)
    
    #Contraction 485; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('acbikj->abcijk',Y18)
    
    #Contraction 486; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('cabikj->abcijk',Y18)
    
    #Contraction 487; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('abcikj->abcijk',Y18)
    
    #Contraction 488; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('bacikj->abcijk',Y18)
    
    #Contraction 489; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('bcaijk->abcijk',Y18)
    
    #Contraction 490; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('cbaijk->abcijk',Y18)
    
    #Contraction 491; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('acbijk->abcijk',Y18)
    
    #Contraction 492; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('cabijk->abcijk',Y18)
    
    #Contraction 493; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('abcijk->abcijk',Y18)
    
    #Contraction 494; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('bacijk->abcijk',Y18)
    
    del Y18
    
    A19 = np.zeros([nvir, nocc, nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 495; Tree Level  3; Scaling  4/ 6 Result_size  4/ 2
    A19 += oe.contract('alde,decijk->alcijk',V8,T3, optimize='optimal')
    
    #del V8
    
    D19 = np.zeros([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 496; Tree Level  2; Scaling  5/ 3 Result_size  3/ 3
    D19 += oe.contract('bl,alcijk->bacijk',T1,A19, optimize='optimal')
    
    del A19
    
    #Contraction 497; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.5 * oe.contract('bacijk->abcijk',D19)
    
    #Contraction 498; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * oe.contract('cabijk->abcijk',D19)
    
    #Contraction 499; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * oe.contract('abcijk->abcijk',D19)
    
    #Contraction 500; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.5 * oe.contract('cbaijk->abcijk',D19)
    
    #Contraction 501; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.5 * oe.contract('acbijk->abcijk',D19)
    
    #Contraction 502; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * oe.contract('bcaijk->abcijk',D19)
    
    del D19
    
    E19 = np.zeros([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 503; Tree Level  2; Scaling  5/ 3 Result_size  3/ 3
    E19 += oe.contract('abcmjk,mi->abcjki',T3,J5, optimize='optimal')
    
    del J5
    
    #Contraction 504; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('abcjki->abcijk',E19)
    
    #Contraction 505; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('abcikj->abcijk',E19)
    
    #Contraction 506; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('abcijk->abcijk',E19)
    
    del E19
    
    G19 = np.zeros([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 507; Tree Level  2; Scaling  5/ 3 Result_size  3/ 3
    G19 += oe.contract('bcmk,amij->bcakij',T2,X17, optimize='optimal')
    
    del X17
    
    #Contraction 508; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * oe.contract('bcakij->abcijk',G19)
    
    #Contraction 509; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.5 * oe.contract('acbkij->abcijk',G19)
    
    #Contraction 510; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * oe.contract('abckij->abcijk',G19)
    
    #Contraction 511; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.5 * oe.contract('bcajik->abcijk',G19)
    
    #Contraction 512; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * oe.contract('acbjik->abcijk',G19)
    
    #Contraction 513; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.5 * oe.contract('abcjik->abcijk',G19)
    
    #Contraction 514; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * oe.contract('bcaijk->abcijk',G19)
    
    #Contraction 515; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.5 * oe.contract('acbijk->abcijk',G19)
    
    #Contraction 516; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * oe.contract('abcijk->abcijk',G19)
    
    del G19
    
    I19 = np.zeros([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 517; Tree Level  2; Scaling  5/ 3 Result_size  3/ 3
    I19 += oe.contract('al,lbcijk->abcijk',T1,Y17, optimize='optimal')
    
    del Y17
    
    #Contraction 518; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('abcijk->abcijk',I19)
    
    #Contraction 519; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('bacijk->abcijk',I19)
    
    #Contraction 520; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('cabijk->abcijk',I19)
    
    del I19
    
    J19 = np.zeros([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 521; Tree Level  2; Scaling  5/ 3 Result_size  3/ 3
    J19 += oe.contract('bajl,lcik->bacjik',T2,G5, optimize='optimal')
    
    del G5
    
    #Contraction 522; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('bacjik->abcijk',J19)
    
    #Contraction 523; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('bcajik->abcijk',J19)
    
    #Contraction 524; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('acbjik->abcijk',J19)
    
    #Contraction 525; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('backij->abcijk',J19)
    
    #Contraction 526; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('bcakij->abcijk',J19)
    
    #Contraction 527; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('acbkij->abcijk',J19)
    
    #Contraction 528; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('bacijk->abcijk',J19)
    
    #Contraction 529; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('bcaijk->abcijk',J19)
    
    #Contraction 530; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('acbijk->abcijk',J19)
    
    #Contraction 531; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('backji->abcijk',J19)
    
    #Contraction 532; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('bcakji->abcijk',J19)
    
    #Contraction 533; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('acbkji->abcijk',J19)
    
    #Contraction 534; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('bacikj->abcijk',J19)
    
    #Contraction 535; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('bcaikj->abcijk',J19)
    
    #Contraction 536; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('acbikj->abcijk',J19)
    
    #Contraction 537; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('bacjki->abcijk',J19)
    
    #Contraction 538; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('bcajki->abcijk',J19)
    
    #Contraction 539; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('acbjki->abcijk',J19)
    
    del J19
    
    M19 = np.zeros([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 540; Tree Level  2; Scaling  5/ 3 Result_size  3/ 3
    M19 += oe.contract('bl,alcijk->bacijk',T1,Y15, optimize='optimal')
    
    del Y15
    
    #Contraction 541; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('bacijk->abcijk',M19)
    
    #Contraction 542; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('cabijk->abcijk',M19)
    
    #Contraction 543; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('abcijk->abcijk',M19)
    
    #Contraction 544; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('cbaijk->abcijk',M19)
    
    #Contraction 545; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('acbijk->abcijk',M19)
    
    #Contraction 546; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('bcaijk->abcijk',M19)
    
    #Contraction 547; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('bacjik->abcijk',M19)
    
    #Contraction 548; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('cabjik->abcijk',M19)
    
    #Contraction 549; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('abcjik->abcijk',M19)
    
    #Contraction 550; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('cbajik->abcijk',M19)
    
    #Contraction 551; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('acbjik->abcijk',M19)
    
    #Contraction 552; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('bcajik->abcijk',M19)
    
    #Contraction 553; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('backij->abcijk',M19)
    
    #Contraction 554; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('cabkij->abcijk',M19)
    
    #Contraction 555; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('abckij->abcijk',M19)
    
    #Contraction 556; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('cbakij->abcijk',M19)
    
    #Contraction 557; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('acbkij->abcijk',M19)
    
    #Contraction 558; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('bcakij->abcijk',M19)
    
    del M19
    
    X19 = np.zeros([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 559; Tree Level  2; Scaling  5/ 3 Result_size  3/ 3
    X19 += oe.contract('al,bclijk->abcijk',T1,Y14, optimize='optimal')
    
    del Y14
    
    #Contraction 560; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('abcijk->abcijk',X19)
    
    #Contraction 561; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('bacijk->abcijk',X19)
    
    #Contraction 562; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('cabijk->abcijk',X19)
    
    del X19
    
    Y19 = np.zeros([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 563; Tree Level  2; Scaling  3/ 5 Result_size  3/ 3
    Y19 += oe.contract('abid,cdkj->abcikj',V6,T2, optimize='optimal')
    
    #del V6
    
    #Contraction 564; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('abcikj->abcijk',Y19)
    
    #Contraction 565; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('acbikj->abcijk',Y19)
    
    #Contraction 566; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('bcaikj->abcijk',Y19)
    
    #Contraction 567; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('abckij->abcijk',Y19)
    
    #Contraction 568; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('acbkij->abcijk',Y19)
    
    #Contraction 569; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('bcakij->abcijk',Y19)
    
    #Contraction 570; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('abcjik->abcijk',Y19)
    
    #Contraction 571; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('acbjik->abcijk',Y19)
    
    #Contraction 572; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('bcajik->abcijk',Y19)
    
    del Y19
    
    A20 = np.zeros([nvir, nocc, nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 573; Tree Level  3; Scaling  4/ 4 Result_size  4/ 2
    A20 += oe.contract('dbij,lcdk->blcijk',T2,X1, optimize='optimal')
    
    del X1
    
    D20 = np.zeros([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 574; Tree Level  2; Scaling  5/ 3 Result_size  3/ 3
    D20 += oe.contract('al,blcijk->abcijk',T1,A20, optimize='optimal')
    
    del A20
    
    #Contraction 575; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('abcijk->abcijk',D20)
    
    #Contraction 576; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('acbijk->abcijk',D20)
    
    #Contraction 577; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('bacijk->abcijk',D20)
    
    #Contraction 578; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('bcaijk->abcijk',D20)
    
    #Contraction 579; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('cabijk->abcijk',D20)
    
    #Contraction 580; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('cbaijk->abcijk',D20)
    
    #Contraction 581; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('abcikj->abcijk',D20)
    
    #Contraction 582; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('acbikj->abcijk',D20)
    
    #Contraction 583; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('bacikj->abcijk',D20)
    
    #Contraction 584; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('bcaikj->abcijk',D20)
    
    #Contraction 585; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('cabikj->abcijk',D20)
    
    #Contraction 586; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('cbaikj->abcijk',D20)
    
    #Contraction 587; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('abcjki->abcijk',D20)
    
    #Contraction 588; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('acbjki->abcijk',D20)
    
    #Contraction 589; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('bacjki->abcijk',D20)
    
    #Contraction 590; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('bcajki->abcijk',D20)
    
    #Contraction 591; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('cabjki->abcijk',D20)
    
    #Contraction 592; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('cbajki->abcijk',D20)
    
    del D20
    
    E20 = np.zeros([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 593; Tree Level  2; Scaling  5/ 3 Result_size  3/ 3
    E20 += oe.contract('abcljk,li->abcjki',T3,J2, optimize='optimal')
    
    del J2
    
    #Contraction 594; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('abcjki->abcijk',E20)
    
    #Contraction 595; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('abcikj->abcijk',E20)
    
    #Contraction 596; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('abcijk->abcijk',E20)
    
    del E20
    
    G20 = np.zeros([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 597; Tree Level  2; Scaling  5/ 3 Result_size  3/ 3
    G20 += oe.contract('abcmjk,mi->abcjki',T3,A17, optimize='optimal')
    
    del A17
    
    #Contraction 598; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * oe.contract('abcjki->abcijk',G20)
    
    #Contraction 599; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.5 * oe.contract('abcikj->abcijk',G20)
    
    #Contraction 600; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * oe.contract('abcijk->abcijk',G20)
    
    del G20
    
    I20 = np.zeros([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 601; Tree Level  2; Scaling  7/ 3 Result_size  3/ 3
    I20 += oe.contract('abclmk,lmij->abckij',T3,M7, optimize='optimal')
    
    del M7
    
    #Contraction 602; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * oe.contract('abckij->abcijk',I20)
    
    #Contraction 603; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.5 * oe.contract('abcjik->abcijk',I20)
    
    #Contraction 604; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.5 * oe.contract('abckji->abcijk',I20)
    
    #Contraction 605; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * oe.contract('abcijk->abcijk',I20)
    
    #Contraction 606; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * oe.contract('abcjki->abcijk',I20)
    
    #Contraction 607; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.5 * oe.contract('abcikj->abcijk',I20)
    
    del I20
    
    J20 = np.zeros([nvir, nvir], dtype=type_)
    
    #Contraction 608; Tree Level  3; Scaling  4/ 4 Result_size  0/ 2
    J20 += oe.contract('lmde,adlm->ae',V7,T2, optimize='optimal')
    
    #del V7
    
    M20 = np.zeros([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 609; Tree Level  2; Scaling  3/ 5 Result_size  3/ 3
    M20 += oe.contract('ebcijk,ae->bcaijk',T3,J20, optimize='optimal')
    
    del J20
    
    #Contraction 610; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * oe.contract('bcaijk->abcijk',M20)
    
    #Contraction 611; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.5 * oe.contract('acbijk->abcijk',M20)
    
    #Contraction 612; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * oe.contract('abcijk->abcijk',M20)
    
    del M20
    
    X20 = np.zeros([nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 613; Tree Level  3; Scaling  6/ 4 Result_size  4/ 2
    X20 += oe.contract('ebcmjk,lmei->bcljki',T3,G2, optimize='optimal')
    
    del G2
    
    Y20 = np.zeros([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 614; Tree Level  2; Scaling  5/ 3 Result_size  3/ 3
    Y20 += oe.contract('al,bcljki->abcjki',T1,X20, optimize='optimal')
    
    del X20
    
    #del T1
    
    #Contraction 615; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('abcjki->abcijk',Y20)
    
    #Contraction 616; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('bacjki->abcijk',Y20)
    
    #Contraction 617; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('cabjki->abcijk',Y20)
    
    #Contraction 618; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('abcikj->abcijk',Y20)
    
    #Contraction 619; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('bacikj->abcijk',Y20)
    
    #Contraction 620; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('cabikj->abcijk',Y20)
    
    #Contraction 621; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('abcijk->abcijk',Y20)
    
    #Contraction 622; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('bacijk->abcijk',Y20)
    
    #Contraction 623; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('cabijk->abcijk',Y20)
    
    del Y20
    
    A21 = np.zeros([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 624; Tree Level  2; Scaling  5/ 5 Result_size  3/ 3
    A21 += oe.contract('alid,bcdjkl->abcijk',V5,T3, optimize='optimal')
    
    #del T3
    
    #del V5
    
    #Contraction 625; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('abcijk->abcijk',A21)
    
    #Contraction 626; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('bacijk->abcijk',A21)
    
    #Contraction 627; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('cabijk->abcijk',A21)
    
    #Contraction 628; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('abcjik->abcijk',A21)
    
    #Contraction 629; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('bacjik->abcijk',A21)
    
    #Contraction 630; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('cabjik->abcijk',A21)
    
    #Contraction 631; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('abckij->abcijk',A21)
    
    #Contraction 632; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('backij->abcijk',A21)
    
    #Contraction 633; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('cabkij->abcijk',A21)
    
    del A21
    
    D21 = np.zeros([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 634; Tree Level  2; Scaling  5/ 3 Result_size  3/ 3
    D21 += oe.contract('bclk,alij->bcakij',T2,A11, optimize='optimal')
    
    del A11
    
    #Contraction 635; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.5 * oe.contract('bcakij->abcijk',D21)
    
    #Contraction 636; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * oe.contract('acbkij->abcijk',D21)
    
    #Contraction 637; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.5 * oe.contract('abckij->abcijk',D21)
    
    #Contraction 638; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * oe.contract('bcajik->abcijk',D21)
    
    #Contraction 639; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.5 * oe.contract('acbjik->abcijk',D21)
    
    #Contraction 640; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * oe.contract('abcjik->abcijk',D21)
    
    #Contraction 641; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * oe.contract('bcakji->abcijk',D21)
    
    #Contraction 642; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.5 * oe.contract('acbkji->abcijk',D21)
    
    #Contraction 643; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * oe.contract('abckji->abcijk',D21)
    
    #Contraction 644; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.5 * oe.contract('bcaijk->abcijk',D21)
    
    #Contraction 645; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * oe.contract('acbijk->abcijk',D21)
    
    #Contraction 646; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.5 * oe.contract('abcijk->abcijk',D21)
    
    #Contraction 647; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.5 * oe.contract('bcajki->abcijk',D21)
    
    #Contraction 648; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * oe.contract('acbjki->abcijk',D21)
    
    #Contraction 649; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.5 * oe.contract('abcjki->abcijk',D21)
    
    #Contraction 650; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * oe.contract('bcaikj->abcijk',D21)
    
    #Contraction 651; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.5 * oe.contract('acbikj->abcijk',D21)
    
    #Contraction 652; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * oe.contract('abcikj->abcijk',D21)
    
    del D21
    
    E21 = np.zeros([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 653; Tree Level  2; Scaling  7/ 3 Result_size  3/ 3
    E21 += oe.contract('ablm,clmjki->abcjki',T2,X13, optimize='optimal')
    
    del X13
    
    #del T2
    
    #Contraction 654; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * oe.contract('abcjki->abcijk',E21)
    
    #Contraction 655; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.5 * oe.contract('cbajki->abcijk',E21)
    
    #Contraction 656; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * oe.contract('cabjki->abcijk',E21)
    
    #Contraction 657; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.5 * oe.contract('abcikj->abcijk',E21)
    
    #Contraction 658; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * oe.contract('cbaikj->abcijk',E21)
    
    #Contraction 659; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.5 * oe.contract('cabikj->abcijk',E21)
    
    #Contraction 660; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * oe.contract('abcijk->abcijk',E21)
    
    #Contraction 661; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.5 * oe.contract('cbaijk->abcijk',E21)
    
    #Contraction 662; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * oe.contract('cabijk->abcijk',E21)
    
    del E21
    
    return([Z0[0], Z1, Z2, Z3])
    
# end of numpy_tenpi_ccsdt
