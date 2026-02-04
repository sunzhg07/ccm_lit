#!/usr/bin/python
# -*- coding: utf-8 -*-

# CCSDTQ_T6 for simple use with numpy

import numpy as np #numpy arrays
import sys


# returns a list of np.ndarrays ordered by growing rank: [energy, residual_vo, residual_vvoo, ...] except the scalars are returned as scalars.
def numpy_tenpi_ccsdtq(nocc: int, nvir: int, F: list, V: list, T: list, type_: type = np.complex128) -> list:
    
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
    T4 = T[3]
    
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
    # T4 = np.array([nvir, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)     #       VVVVOOOO
    Z0 = np.zeros([1], dtype=type_)     #       scalar
    Z1 = np.zeros([nvir, nocc], dtype=type_)     #       VO
    Z2 = np.zeros([nvir, nvir, nocc, nocc], dtype=type_)     #       VVOO
    Z3 = np.zeros([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)     #       VVVOOO
    Z4 = np.zeros([nvir, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)     #       VVVVOOOO
    # A1 = np.array([nocc, nvir], dtype=type_)     #       OV
    # I1 = np.array([nocc, nocc, nvir, nocc], dtype=type_)     #       OOVO
    # I3 = np.array([nocc, nocc], dtype=type_)     #       OO
    # M1 = np.array([nocc, nocc], dtype=type_)     #       OO
    # X1 = np.array([nvir, nvir], dtype=type_)     #       VV
    # Y1 = np.array([nocc, nocc], dtype=type_)     #       OO
    # G2 = np.array([nocc, nvir, nvir, nocc], dtype=type_)     #       OVVO
    # I2 = np.array([nocc, nvir, nvir, nocc], dtype=type_)     #       OVVO
    # X2 = np.array([nvir, nvir], dtype=type_)     #       VV
    # Y3 = np.array([nocc, nvir, nocc, nocc], dtype=type_)     #       OVOO
    # A4 = np.array([nvir, nvir, nocc, nocc], dtype=type_)     #       VVOO
    # E4 = np.array([nvir, nvir, nocc, nocc], dtype=type_)     #       VVOO
    # G4 = np.array([nvir, nvir, nocc, nocc], dtype=type_)     #       VVOO
    # J4 = np.array([nvir, nocc, nocc, nocc], dtype=type_)     #       VOOO
    # M4 = np.array([nvir, nocc, nvir, nocc], dtype=type_)     #       VOVO
    # X4 = np.array([nvir, nvir, nocc, nocc], dtype=type_)     #       VVOO
    # Y4 = np.array([nvir, nvir, nocc, nocc], dtype=type_)     #       VVOO
    # A5 = np.array([nocc, nocc], dtype=type_)     #       OO
    # D5 = np.array([nvir, nvir, nocc, nocc], dtype=type_)     #       VVOO
    # E5 = np.array([nocc, nvir, nocc, nocc], dtype=type_)     #       OVOO
    # G5 = np.array([nvir, nocc, nvir, nocc], dtype=type_)     #       VOVO
    # I5 = np.array([nocc, nocc, nocc, nocc], dtype=type_)     #       OOOO
    # J5 = np.array([nvir, nocc, nocc, nocc], dtype=type_)     #       VOOO
    # M5 = np.array([nvir, nvir, nvir, nocc], dtype=type_)     #       VVVO
    # X5 = np.array([nvir, nvir, nocc, nocc], dtype=type_)     #       VVOO
    # Y5 = np.array([nvir, nocc, nocc, nocc], dtype=type_)     #       VOOO
    # A6 = np.array([nvir, nvir, nocc, nocc], dtype=type_)     #       VVOO
    # D6 = np.array([nocc, nvir], dtype=type_)     #       OV
    # E6 = np.array([nvir, nocc, nocc, nocc], dtype=type_)     #       VOOO
    # G6 = np.array([nvir, nvir, nocc, nocc], dtype=type_)     #       VVOO
    # J6 = np.array([nvir, nocc, nocc, nocc], dtype=type_)     #       VOOO
    # M6 = np.array([nvir, nvir, nocc, nocc], dtype=type_)     #       VVOO
    # G12 = np.array([nocc, nvir, nocc, nocc], dtype=type_)     #       OVOO
    # Y6 = np.array([nocc, nocc, nocc, nocc], dtype=type_)     #       OOOO
    # G8 = np.array([nocc, nocc, nocc, nocc], dtype=type_)     #       OOOO
    # A11 = np.array([nocc, nocc, nocc, nocc], dtype=type_)     #       OOOO
    # E7 = np.array([nocc, nocc, nocc, nocc], dtype=type_)     #       OOOO
    # I7 = np.array([nvir, nocc, nocc, nocc], dtype=type_)     #       VOOO
    # A12 = np.array([nvir, nocc, nocc, nocc], dtype=type_)     #       VOOO
    # A8 = np.array([nvir, nvir, nocc, nocc], dtype=type_)     #       VVOO
    # D8 = np.array([nvir, nocc, nocc, nocc], dtype=type_)     #       VOOO
    # J8 = np.array([nocc, nocc], dtype=type_)     #       OO
    # M8 = np.array([nvir, nvir, nocc, nocc], dtype=type_)     #       VVOO
    # X8 = np.array([nvir, nvir, nocc, nocc], dtype=type_)     #       VVOO
    # Y8 = np.array([nvir, nocc, nocc, nocc], dtype=type_)     #       VOOO
    # A9 = np.array([nvir, nvir, nocc, nocc], dtype=type_)     #       VVOO
    # D9 = np.array([nvir, nvir, nocc, nocc], dtype=type_)     #       VVOO
    # E9 = np.array([nvir, nvir, nocc, nocc], dtype=type_)     #       VVOO
    # G9 = np.array([nvir, nvir, nocc, nocc], dtype=type_)     #       VVOO
    # I9 = np.array([nvir, nocc, nocc, nocc], dtype=type_)     #       VOOO
    # J9 = np.array([nvir, nvir, nocc, nocc], dtype=type_)     #       VVOO
    # M9 = np.array([nocc, nvir, nocc, nocc], dtype=type_)     #       OVOO
    # X9 = np.array([nvir, nvir, nocc, nocc], dtype=type_)     #       VVOO
    # E10 = np.array([nvir, nvir, nocc, nocc], dtype=type_)     #       VVOO
    # I10 = np.array([nvir, nvir], dtype=type_)     #       VV
    # J10 = np.array([nvir, nvir, nocc, nocc], dtype=type_)     #       VVOO
    # J12 = np.array([nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)     #       VVOOOO
    # M12 = np.array([nocc, nocc, nvir, nocc, nocc, nocc], dtype=type_)     #       OOVOOO
    # X12 = np.array([nocc, nvir, nvir, nocc, nocc, nocc], dtype=type_)     #       OVVOOO
    # Y12 = np.array([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)     #       VVVOOO
    # A13 = np.array([nvir, nocc, nocc, nocc], dtype=type_)     #       VOOO
    # D13 = np.array([nvir, nocc, nvir, nocc, nocc, nocc], dtype=type_)     #       VOVOOO
    # E13 = np.array([nocc, nvir, nocc, nocc], dtype=type_)     #       OVOO
    # G13 = np.array([nocc, nocc, nvir, nocc, nocc, nocc], dtype=type_)     #       OOVOOO
    # I13 = np.array([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)     #       VVVOOO
    # J13 = np.array([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)     #       VVVOOO
    # M13 = np.array([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)     #       VVVOOO
    # X13 = np.array([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)     #       VVVOOO
    # Y13 = np.array([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)     #       VVVOOO
    # A14 = np.array([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)     #       VVVOOO
    # D14 = np.array([nvir, nocc, nvir, nocc, nocc, nocc], dtype=type_)     #       VOVOOO
    # E14 = np.array([nocc, nvir, nvir, nocc, nocc, nocc], dtype=type_)     #       OVVOOO
    # G14 = np.array([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)     #       VVVOOO
    # I14 = np.array([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)     #       VVVOOO
    # J14 = np.array([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)     #       VVVOOO
    # M14 = np.array([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)     #       VVVOOO
    # X14 = np.array([nvir, nocc, nvir, nocc, nocc, nocc], dtype=type_)     #       VOVOOO
    # Y14 = np.array([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)     #       VVVOOO
    # A15 = np.array([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)     #       VVVOOO
    # D15 = np.array([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)     #       VVVOOO
    # E15 = np.array([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)     #       VVVOOO
    # G15 = np.array([nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)     #       VVOOOO
    # I15 = np.array([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)     #       VVVOOO
    # J15 = np.array([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)     #       VVVOOO
    # M15 = np.array([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)     #       VVVOOO
    # X15 = np.array([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)     #       VVVOOO
    # Y15 = np.array([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)     #       VVVOOO
    # A16 = np.array([nvir, nvir], dtype=type_)     #       VV
    # D16 = np.array([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)     #       VVVOOO
    # E16 = np.array([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)     #       VVVOOO
    # G16 = np.array([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)     #       VVVOOO
    # I16 = np.array([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)     #       VVVOOO
    # J16 = np.array([nvir, nvir, nvir, nocc], dtype=type_)     #       VVVO
    # M16 = np.array([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)     #       VVVOOO
    # X16 = np.array([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)     #       VVVOOO
    # Y16 = np.array([nocc, nocc], dtype=type_)     #       OO
    # A17 = np.array([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)     #       VVVOOO
    # D17 = np.array([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)     #       VVVOOO
    # E17 = np.array([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)     #       VVVOOO
    # G17 = np.array([nocc, nvir, nvir, nocc], dtype=type_)     #       OVVO
    # I17 = np.array([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)     #       VVVOOO
    # J17 = np.array([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)     #       VVVOOO
    # M17 = np.array([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)     #       VVVOOO
    # X17 = np.array([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)     #       VVVOOO
    # Y17 = np.array([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)     #       VVVOOO
    # A18 = np.array([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)     #       VVVOOO
    # D18 = np.array([nvir, nocc, nvir, nocc, nocc, nocc], dtype=type_)     #       VOVOOO
    # E18 = np.array([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)     #       VVVOOO
    # G18 = np.array([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)     #       VVVOOO
    # I18 = np.array([nocc, nvir, nocc, nocc], dtype=type_)     #       OVOO
    # J18 = np.array([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)     #       VVVOOO
    # M18 = np.array([nvir, nvir, nocc, nvir], dtype=type_)     #       VVOV
    # X18 = np.array([nvir, nocc, nocc, nocc, nocc, nocc], dtype=type_)     #       VOOOOO
    # Y18 = np.array([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)     #       VVVOOO
    # A19 = np.array([nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)     #       VVOOOO
    # D19 = np.array([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)     #       VVVOOO
    # E19 = np.array([nvir, nocc, nvir, nocc, nocc, nocc], dtype=type_)     #       VOVOOO
    # G19 = np.array([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)     #       VVVOOO
    # I19 = np.array([nvir, nvir, nvir, nocc], dtype=type_)     #       VVVO
    # J19 = np.array([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)     #       VVVOOO
    # M19 = np.array([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)     #       VVVOOO
    # X19 = np.array([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)     #       VVVOOO
    # Y19 = np.array([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)     #       VVVOOO
    # A20 = np.array([nvir, nvir, nvir, nocc], dtype=type_)     #       VVVO
    # D20 = np.array([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)     #       VVVOOO
    # I20 = np.array([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)     #       VVVOOO
    # J20 = np.array([nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)     #       VVOOOO
    # M20 = np.array([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)     #       VVVOOO
    # X20 = np.array([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)     #       VVVOOO
    # Y20 = np.array([nocc, nvir, nvir, nocc, nocc, nocc], dtype=type_)     #       OVVOOO
    # A21 = np.array([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)     #       VVVOOO
    # D21 = np.array([nvir, nocc, nocc, nocc], dtype=type_)     #       VOOO
    # E21 = np.array([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)     #       VVVOOO
    # G21 = np.array([nvir, nvir, nvir, nocc, nocc, nocc, nocc, nocc], dtype=type_)     #       VVVOOOOO
    # I21 = np.array([nocc, nvir, nvir, nvir, nocc, nocc], dtype=type_)     #       OVVVOO
    # J21 = np.array([nvir, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)     #       VVVVOOOO
    # M21 = np.array([nvir, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)     #       VVVVOOOO
    # X21 = np.array([nocc, nvir, nvir, nocc], dtype=type_)     #       OVVO
    # Y21 = np.array([nvir, nocc, nocc, nocc], dtype=type_)     #       VOOO
    # A22 = np.array([nvir, nocc, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)     #       VOVVOOOO
    # D22 = np.array([nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)     #       VVOOOO
    # E22 = np.array([nvir, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)     #       VVVVOOOO
    # G22 = np.array([nvir, nocc, nvir, nocc, nocc, nocc], dtype=type_)     #       VOVOOO
    # I22 = np.array([nvir, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)     #       VVVVOOOO
    # J22 = np.array([nocc, nocc, nvir, nocc, nocc, nocc], dtype=type_)     #       OOVOOO
    # M22 = np.array([nvir, nocc, nvir, nocc, nocc, nocc], dtype=type_)     #       VOVOOO
    # X22 = np.array([nvir, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)     #       VVVVOOOO
    # Y22 = np.array([nvir, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)     #       VVVVOOOO
    # A23 = np.array([nvir, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)     #       VVVVOOOO
    # D23 = np.array([nvir, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)     #       VVVVOOOO
    # E23 = np.array([nvir, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)     #       VVVVOOOO
    # G23 = np.array([nvir, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)     #       VVVVOOOO
    # I23 = np.array([nvir, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)     #       VVVVOOOO
    # J23 = np.array([nvir, nvir, nvir, nvir, nocc, nocc], dtype=type_)     #       VVVVOO
    # M23 = np.array([nvir, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)     #       VVVVOOOO
    # X23 = np.array([nvir, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)     #       VVVVOOOO
    # Y23 = np.array([nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)     #       VVOOOO
    # A24 = np.array([nvir, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)     #       VVVVOOOO
    # D24 = np.array([nvir, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)     #       VVVVOOOO
    # E24 = np.array([nvir, nocc, nvir, nocc, nocc, nocc], dtype=type_)     #       VOVOOO
    # G24 = np.array([nvir, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)     #       VVVVOOOO
    # I24 = np.array([nvir, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)     #       VVVVOOOO
    # J24 = np.array([nocc, nocc, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)     #       OOVVOOOO
    # M24 = np.array([nvir, nocc, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)     #       VOVVOOOO
    # X24 = np.array([nvir, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)     #       VVVVOOOO
    # Y24 = np.array([nvir, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)     #       VVVVOOOO
    # A25 = np.array([nvir, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)     #       VVVVOOOO
    # D25 = np.array([nvir, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)     #       VVVVOOOO
    # E25 = np.array([nvir, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)     #       VVVVOOOO
    # G25 = np.array([nvir, nocc, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)     #       VOVVOOOO
    # I25 = np.array([nocc, nvir, nocc, nocc], dtype=type_)     #       OVOO
    # J25 = np.array([nvir, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)     #       VVVVOOOO
    # M25 = np.array([nvir, nocc, nvir, nvir, nocc, nocc], dtype=type_)     #       VOVVOO
    # X25 = np.array([nvir, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)     #       VVVVOOOO
    # Y25 = np.array([nvir, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)     #       VVVVOOOO
    # A26 = np.array([nvir, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)     #       VVVVOOOO
    # D26 = np.array([nvir, nvir, nocc, nvir, nocc, nocc, nocc, nocc], dtype=type_)     #       VVOVOOOO
    # E26 = np.array([nvir, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)     #       VVVVOOOO
    # G26 = np.array([nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)     #       VVOOOO
    # I26 = np.array([nvir, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)     #       VVVVOOOO
    # J26 = np.array([nocc, nvir, nvir, nocc, nocc, nocc], dtype=type_)     #       OVVOOO
    # M26 = np.array([nvir, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)     #       VVVVOOOO
    # X26 = np.array([nvir, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)     #       VVVVOOOO
    # Y26 = np.array([nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)     #       VVOOOO
    # A27 = np.array([nvir, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)     #       VVVVOOOO
    # D27 = np.array([nvir, nvir, nvir, nocc, nocc, nocc, nocc, nocc], dtype=type_)     #       VVVOOOOO
    # E27 = np.array([nvir, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)     #       VVVVOOOO
    # G27 = np.array([nvir, nvir, nocc, nocc, nocc, nocc, nocc, nocc], dtype=type_)     #       VVOOOOOO
    # I27 = np.array([nvir, nvir, nvir, nocc, nocc, nocc, nocc, nocc], dtype=type_)     #       VVVOOOOO
    # J27 = np.array([nvir, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)     #       VVVVOOOO
    # M27 = np.array([nvir, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)     #       VVVVOOOO
    # X27 = np.array([nvir, nocc, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)     #       VOVVOOOO
    # Y27 = np.array([nvir, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)     #       VVVVOOOO
    # A28 = np.array([nvir, nvir, nvir, nocc, nocc, nocc, nocc, nocc], dtype=type_)     #       VVVOOOOO
    # D28 = np.array([nvir, nocc, nvir, nocc, nocc, nocc], dtype=type_)     #       VOVOOO
    # E28 = np.array([nvir, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)     #       VVVVOOOO
    # G28 = np.array([nvir, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)     #       VVVVOOOO
    # I28 = np.array([nocc, nocc, nvir, nvir, nocc, nocc], dtype=type_)     #       OOVVOO
    # J28 = np.array([nvir, nocc, nocc, nvir, nocc, nocc, nocc, nocc], dtype=type_)     #       VOOVOOOO
    # M28 = np.array([nvir, nvir, nocc, nvir, nocc, nocc, nocc, nocc], dtype=type_)     #       VVOVOOOO
    # X28 = np.array([nvir, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)     #       VVVVOOOO
    # Y28 = np.array([nvir, nvir, nvir, nvir, nocc, nocc], dtype=type_)     #       VVVVOO
    # A29 = np.array([nvir, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)     #       VVVVOOOO
    # D29 = np.array([nvir, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)     #       VVVVOOOO
    # E29 = np.array([nocc, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)     #       OVVVOOOO
    # G29 = np.array([nvir, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)     #       VVVVOOOO
    # I29 = np.array([nvir, nvir, nvir, nvir, nocc, nocc], dtype=type_)     #       VVVVOO
    # J29 = np.array([nvir, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)     #       VVVVOOOO
    # M29 = np.array([nocc, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)     #       OVVVOOOO
    # X29 = np.array([nvir, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)     #       VVVVOOOO
    # Y29 = np.array([nvir, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)     #       VVVVOOOO
    # A30 = np.array([nvir, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)     #       VVVVOOOO
    # D30 = np.array([nvir, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)     #       VVVVOOOO
    # E30 = np.array([nvir, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)     #       VVVVOOOO
    # G30 = np.array([nvir, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)     #       VVVVOOOO
    # I30 = np.array([nvir, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)     #       VVVVOOOO
    # J30 = np.array([nvir, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)     #       VVVVOOOO
    # M30 = np.array([nvir, nvir, nocc, nvir, nocc, nocc, nocc, nocc], dtype=type_)     #       VVOVOOOO
    # X30 = np.array([nvir, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)     #       VVVVOOOO
    # Y30 = np.array([nocc, nocc, nvir, nocc, nocc, nocc], dtype=type_)     #       OOVOOO
    # A31 = np.array([nvir, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)     #       VVVVOOOO
    # D31 = np.array([nvir, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)     #       VVVVOOOO
    # E31 = np.array([nvir, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)     #       VVVVOOOO
    # G31 = np.array([nvir, nvir, nvir, nocc], dtype=type_)     #       VVVO
    # I31 = np.array([nvir, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)     #       VVVVOOOO
    # J31 = np.array([nvir, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)     #       VVVVOOOO
    # M31 = np.array([nvir, nvir, nvir, nvir, nocc, nocc], dtype=type_)     #       VVVVOO
    # X31 = np.array([nvir, nvir, nvir, nocc], dtype=type_)     #       VVVO
    # Y31 = np.array([nvir, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)     #       VVVVOOOO
    # A32 = np.array([nvir, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)     #       VVVVOOOO
    # D32 = np.array([nvir, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)     #       VVVVOOOO
    # E32 = np.array([nvir, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)     #       VVVVOOOO
    # G32 = np.array([nvir, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)     #       VVVVOOOO
    # I32 = np.array([nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)     #       VVOOOO
    # J32 = np.array([nvir, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)     #       VVVVOOOO
    # M32 = np.array([nvir, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)     #       VVVVOOOO
    # X32 = np.array([nvir, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)     #       VVVVOOOO
    # Y32 = np.array([nvir, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)     #       VVVVOOOO
    # A33 = np.array([nvir, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)     #       VVVVOOOO
    # D33 = np.array([nvir, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)     #       VVVVOOOO
    # E33 = np.array([nvir, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)     #       VVVVOOOO
    # G33 = np.array([nvir, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)     #       VVVVOOOO
    # I33 = np.array([nocc, nocc, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)     #       OOVVOOOO
    # J33 = np.array([nvir, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)     #       VVVVOOOO
    # M33 = np.array([nvir, nocc, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)     #       VOVVOOOO
    # X33 = np.array([nvir, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)     #       VVVVOOOO
    # Y33 = np.array([nvir, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)     #       VVVVOOOO
    # A34 = np.array([nvir, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)     #       VVVVOOOO
    # D34 = np.array([nvir, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)     #       VVVVOOOO
    
    print('    Running code generated by tenpi.   ')
    
    A1 = np.zeros([nocc, nvir], dtype=type_)
    
    #Contraction 1; Tree Level  2; Scaling  3/ 3 Result_size  1/ 1
    A1 += np.einsum('ijab,ai->jb',V7,T1, optimize='optimal')
    
    #Contraction 2; Tree Level  0; Scaling  2/ 2 Result_size  0/ 0
    Z0 += np.einsum('ia,ai->',F3,T1, optimize='optimal')
    
    #Contraction 3; Tree Level  1; Scaling  2/ 2 Result_size  0/ 0
    Z0 += 0.5 * np.einsum('bj,jb->',T1,A1, optimize='optimal')
    
    #Contraction 4; Tree Level  1; Scaling  4/ 4 Result_size  0/ 0
    Z0 += 0.25 * np.einsum('ijab,abij->',V7,T2, optimize='optimal')
    
    I1 = np.zeros([nocc, nocc, nvir, nocc], dtype=type_)
    
    #Contraction 5; Tree Level  3; Scaling  3/ 3 Result_size  3/ 1
    I1 += np.einsum('jkbc,bi->jkci',V7,T1, optimize='optimal')
    
    I3 = np.zeros([nocc, nocc], dtype=type_)
    
    #Contraction 6; Tree Level  2; Scaling  4/ 2 Result_size  2/ 0
    I3 += -2.0 * np.einsum('jkib,bk->ji',V4,T1, optimize='optimal')
    
    M1 = np.zeros([nocc, nocc], dtype=type_)
    
    #Contraction 7; Tree Level  3; Scaling  4/ 4 Result_size  2/ 0
    M1 += np.einsum('jkbc,bcki->ji',V7,T2, optimize='optimal')
    
    X1 = np.zeros([nvir, nvir], dtype=type_)
    
    #Contraction 8; Tree Level  3; Scaling  4/ 4 Result_size  0/ 2
    X1 += np.einsum('klcd,dbkl->bc',V7,T2, optimize='optimal')
    
    Y1 = np.zeros([nocc, nocc], dtype=type_)
    
    #Contraction 9; Tree Level  3; Scaling  2/ 2 Result_size  2/ 0
    Y1 += np.einsum('jb,bi->ji',F3,T1, optimize='optimal')
    
    #Contraction 10; Tree Level  2; Scaling  2/ 0 Result_size  2/ 0
    I3 += -2.0 * np.einsum('ji->ji',Y1)
    
    #Contraction 11; Tree Level  0; Scaling  5/ 3 Result_size  1/ 1
    Z1 += -0.5 * np.einsum('jkib,abjk->ai',V4,T2, optimize='optimal')
    
    #Contraction 12; Tree Level  1; Scaling  5/ 5 Result_size  1/ 1
    Z1 += 0.25 * np.einsum('jkbc,abcijk->ai',V7,T3, optimize='optimal')
    
    G2 = np.zeros([nocc, nvir, nvir, nocc], dtype=type_)
    
    #Contraction 13; Tree Level  3; Scaling  4/ 4 Result_size  2/ 2
    G2 += np.einsum('klcd,dblj->kbcj',V7,T2, optimize='optimal')
    
    I2 = np.zeros([nocc, nvir, nvir, nocc], dtype=type_)
    
    #Contraction 14; Tree Level  2; Scaling  2/ 2 Result_size  2/ 2
    I2 += np.einsum('jabi->jabi',G2)
    
    #Contraction 15; Tree Level  2; Scaling  2/ 2 Result_size  2/ 2
    I2 += np.einsum('ajib->jabi',V5)
    
    X2 = np.zeros([nvir, nvir], dtype=type_)
    
    #Contraction 16; Tree Level  2; Scaling  2/ 4 Result_size  0/ 2
    X2 += 2.0 * np.einsum('ajbc,cj->ab',V8,T1, optimize='optimal')
    
    #Contraction 17; Tree Level  2; Scaling  0/ 2 Result_size  0/ 2
    X2 += np.einsum('ab->ab',X1)
    
    #Contraction 18; Tree Level  2; Scaling  0/ 2 Result_size  0/ 2
    X2 += 2.0 * np.einsum('ab->ab',F4)
    
    #Contraction 19; Tree Level  1; Scaling  1/ 3 Result_size  1/ 1
    Z1 += 0.5 * np.einsum('bi,ab->ai',T1,X2, optimize='optimal')
    
    del X2
    
    #Contraction 20; Tree Level  2; Scaling  2/ 0 Result_size  2/ 0
    I3 += -2.0 * np.einsum('ji->ji',F1)
    
    #Contraction 21; Tree Level  1; Scaling  3/ 3 Result_size  1/ 1
    Z1 += np.einsum('jb,abij->ai',F3,T2, optimize='optimal')
    
    #Contraction 22; Tree Level  2; Scaling  4/ 2 Result_size  2/ 0
    I3 += -2.0 * np.einsum('ck,jkci->ji',T1,I1, optimize='optimal')
    
    #Contraction 23; Tree Level  2; Scaling  2/ 0 Result_size  2/ 0
    I3 += np.einsum('ji->ji',M1)
    
    #Contraction 24; Tree Level  1; Scaling  3/ 1 Result_size  1/ 1
    Z1 += 0.5 * np.einsum('aj,ji->ai',T1,I3, optimize='optimal')
    
    del I3
    
    #Contraction 25; Tree Level  1; Scaling  3/ 5 Result_size  1/ 1
    Z1 += 0.5 * np.einsum('ajbc,bcij->ai',V8,T2, optimize='optimal')
    
    #Contraction 26; Tree Level  1; Scaling  3/ 3 Result_size  1/ 1
    Z1 += np.einsum('bj,jabi->ai',T1,I2, optimize='optimal')
    
    del I2
    
    #Contraction 27; Tree Level  1; Scaling  1/ 1 Result_size  1/ 1
    Z1 += np.einsum('ai->ai',F2)
    
    #del F2
    
    Y3 = np.zeros([nocc, nvir, nocc, nocc], dtype=type_)
    
    #Contraction 28; Tree Level  3; Scaling  5/ 3 Result_size  3/ 1
    Y3 += np.einsum('klic,cblj->kbij',V4,T2, optimize='optimal')
    
    A4 = np.zeros([nvir, nvir, nocc, nocc], dtype=type_)
    
    #Contraction 29; Tree Level  2; Scaling  4/ 6 Result_size  2/ 2
    A4 += np.einsum('akcd,bcdjik->abji',V8,T3, optimize='optimal')
    
    #Contraction 30; Tree Level  0; Scaling  2/ 2 Result_size  2/ 2
    Z2 += 0.5 * np.einsum('abji->abij',A4)
    
    #Contraction 31; Tree Level  1; Scaling  2/ 2 Result_size  2/ 2
    Z2 += -0.5 * np.einsum('baji->abij',A4)
    
    del A4
    
    #Contraction 32; Tree Level  1; Scaling  4/ 4 Result_size  2/ 2
    Z2 += np.einsum('dablij,ld->abij',T3,A1, optimize='optimal')
    
    E4 = np.zeros([nvir, nvir, nocc, nocc], dtype=type_)
    
    #Contraction 33; Tree Level  2; Scaling  4/ 2 Result_size  2/ 2
    E4 += np.einsum('abkj,ki->abji',T2,Y1, optimize='optimal')
    
    #Contraction 34; Tree Level  1; Scaling  2/ 2 Result_size  2/ 2
    Z2 += -1.0 * np.einsum('abji->abij',E4)
    
    #Contraction 35; Tree Level  1; Scaling  2/ 2 Result_size  2/ 2
    Z2 += np.einsum('abij->abij',E4)
    
    del E4
    
    G4 = np.zeros([nvir, nvir, nocc, nocc], dtype=type_)
    
    #Contraction 36; Tree Level  2; Scaling  4/ 2 Result_size  2/ 2
    G4 += np.einsum('ki,bajk->baij',F1,T2, optimize='optimal')
    
    #Contraction 37; Tree Level  1; Scaling  2/ 2 Result_size  2/ 2
    Z2 += -1.0 * np.einsum('baij->abij',G4)
    
    #Contraction 38; Tree Level  1; Scaling  2/ 2 Result_size  2/ 2
    Z2 += np.einsum('baji->abij',G4)
    
    del G4
    
    #Contraction 39; Tree Level  1; Scaling  6/ 6 Result_size  2/ 2
    Z2 += 0.25 * np.einsum('klcd,abcdijkl->abij',V7,T4, optimize='optimal')
    
    J4 = np.zeros([nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 40; Tree Level  3; Scaling  3/ 3 Result_size  3/ 1
    J4 += np.einsum('akic,cj->akij',V5,T1, optimize='optimal')
    
    M4 = np.zeros([nvir, nocc, nvir, nocc], dtype=type_)
    
    #Contraction 41; Tree Level  4; Scaling  2/ 4 Result_size  2/ 2
    M4 += np.einsum('akcd,dj->akcj',V8,T1, optimize='optimal')
    
    X4 = np.zeros([nvir, nvir, nocc, nocc], dtype=type_)
    
    #Contraction 42; Tree Level  2; Scaling  4/ 4 Result_size  2/ 2
    X4 += np.einsum('akic,bcjk->abij',V5,T2, optimize='optimal')
    
    #Contraction 43; Tree Level  1; Scaling  2/ 2 Result_size  2/ 2
    Z2 += np.einsum('abij->abij',X4)
    
    #Contraction 44; Tree Level  1; Scaling  2/ 2 Result_size  2/ 2
    Z2 += -1.0 * np.einsum('baij->abij',X4)
    
    #Contraction 45; Tree Level  1; Scaling  2/ 2 Result_size  2/ 2
    Z2 += -1.0 * np.einsum('abji->abij',X4)
    
    #Contraction 46; Tree Level  1; Scaling  2/ 2 Result_size  2/ 2
    Z2 += np.einsum('baji->abij',X4)
    
    del X4
    
    Y4 = np.zeros([nvir, nvir, nocc, nocc], dtype=type_)
    
    #Contraction 47; Tree Level  2; Scaling  4/ 2 Result_size  2/ 2
    Y4 += np.einsum('akij,bk->abij',V2,T1, optimize='optimal')
    
    #Contraction 48; Tree Level  1; Scaling  2/ 2 Result_size  2/ 2
    Z2 += -1.0 * np.einsum('abij->abij',Y4)
    
    #Contraction 49; Tree Level  1; Scaling  2/ 2 Result_size  2/ 2
    Z2 += np.einsum('baij->abij',Y4)
    
    del Y4
    
    A5 = np.zeros([nocc, nocc], dtype=type_)
    
    #Contraction 50; Tree Level  3; Scaling  4/ 2 Result_size  2/ 0
    A5 += np.einsum('dk,kldi->li',T1,I1, optimize='optimal')
    
    D5 = np.zeros([nvir, nvir, nocc, nocc], dtype=type_)
    
    #Contraction 51; Tree Level  2; Scaling  2/ 4 Result_size  2/ 2
    D5 += np.einsum('abic,cj->abij',V6,T1, optimize='optimal')
    
    #Contraction 52; Tree Level  1; Scaling  2/ 2 Result_size  2/ 2
    Z2 += np.einsum('abij->abij',D5)
    
    #Contraction 53; Tree Level  1; Scaling  2/ 2 Result_size  2/ 2
    Z2 += -1.0 * np.einsum('abji->abij',D5)
    
    del D5
    
    E5 = np.zeros([nocc, nvir, nocc, nocc], dtype=type_)
    
    #Contraction 54; Tree Level  3; Scaling  3/ 3 Result_size  3/ 1
    E5 += np.einsum('kc,cbij->kbij',F3,T2, optimize='optimal')
    
    G5 = np.zeros([nvir, nocc, nvir, nocc], dtype=type_)
    
    #Contraction 55; Tree Level  3; Scaling  2/ 4 Result_size  2/ 2
    G5 += np.einsum('akcd,ci->akdi',V8,T1, optimize='optimal')
    
    I5 = np.zeros([nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 56; Tree Level  2; Scaling  4/ 4 Result_size  4/ 0
    I5 += np.einsum('klcd,cdij->klij',V7,T2, optimize='optimal')
    
    J5 = np.zeros([nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 57; Tree Level  2; Scaling  5/ 1 Result_size  3/ 1
    J5 += np.einsum('bl,klij->bkij',T1,I5, optimize='optimal')
    
    M5 = np.zeros([nvir, nvir, nvir, nocc], dtype=type_)
    
    #Contraction 58; Tree Level  3; Scaling  1/ 5 Result_size  1/ 3
    M5 += np.einsum('abcd,dj->abcj',V9,T1, optimize='optimal')
    
    X5 = np.zeros([nvir, nvir, nocc, nocc], dtype=type_)
    
    #Contraction 59; Tree Level  2; Scaling  2/ 4 Result_size  2/ 2
    X5 += np.einsum('acij,bc->abij',T2,X1, optimize='optimal')
    
    del X1
    
    #Contraction 60; Tree Level  1; Scaling  2/ 2 Result_size  2/ 2
    Z2 += 0.5 * np.einsum('abij->abij',X5)
    
    #Contraction 61; Tree Level  1; Scaling  2/ 2 Result_size  2/ 2
    Z2 += -0.5 * np.einsum('baij->abij',X5)
    
    del X5
    
    Y5 = np.zeros([nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 62; Tree Level  3; Scaling  5/ 3 Result_size  3/ 1
    Y5 += np.einsum('dblj,kldi->bkji',T2,I1, optimize='optimal')
    
    A6 = np.zeros([nvir, nvir, nocc, nocc], dtype=type_)
    
    #Contraction 63; Tree Level  2; Scaling  2/ 4 Result_size  2/ 2
    A6 += np.einsum('ac,bcji->abji',F4,T2, optimize='optimal')
    
    #Contraction 64; Tree Level  1; Scaling  2/ 2 Result_size  2/ 2
    Z2 += np.einsum('abji->abij',A6)
    
    #Contraction 65; Tree Level  1; Scaling  2/ 2 Result_size  2/ 2
    Z2 += -1.0 * np.einsum('baji->abij',A6)
    
    del A6
    
    D6 = np.zeros([nocc, nvir], dtype=type_)
    
    #Contraction 66; Tree Level  4; Scaling  3/ 3 Result_size  1/ 1
    D6 += np.einsum('klcd,cl->kd',V7,T1, optimize='optimal')
    
    E6 = np.zeros([nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 67; Tree Level  3; Scaling  3/ 3 Result_size  3/ 1
    E6 += np.einsum('dbij,kd->bkij',T2,D6, optimize='optimal')
    
    G6 = np.zeros([nvir, nvir, nocc, nocc], dtype=type_)
    
    #Contraction 68; Tree Level  2; Scaling  2/ 4 Result_size  2/ 2
    G6 += np.einsum('ci,abcj->abij',T1,M5, optimize='optimal')
    
    del M5
    
    #Contraction 69; Tree Level  1; Scaling  2/ 2 Result_size  2/ 2
    Z2 += 0.5 * np.einsum('abij->abij',G6)
    
    #Contraction 70; Tree Level  1; Scaling  2/ 2 Result_size  2/ 2
    Z2 += -0.5 * np.einsum('abji->abij',G6)
    
    del G6
    
    #Contraction 71; Tree Level  1; Scaling  6/ 2 Result_size  2/ 2
    Z2 += 0.25 * np.einsum('abkl,klij->abij',T2,I5, optimize='optimal')
    
    J6 = np.zeros([nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 72; Tree Level  3; Scaling  3/ 5 Result_size  3/ 1
    J6 += np.einsum('akcd,cdij->akij',V8,T2, optimize='optimal')
    
    M6 = np.zeros([nvir, nvir, nocc, nocc], dtype=type_)
    
    #Contraction 73; Tree Level  2; Scaling  4/ 2 Result_size  2/ 2
    M6 += np.einsum('bk,akij->baij',T1,J6, optimize='optimal')
    
    #Contraction 74; Tree Level  1; Scaling  2/ 2 Result_size  2/ 2
    Z2 += -0.5 * np.einsum('baij->abij',M6)
    
    #Contraction 75; Tree Level  1; Scaling  2/ 2 Result_size  2/ 2
    Z2 += 0.5 * np.einsum('abij->abij',M6)
    
    del M6
    
    G12 = np.zeros([nocc, nvir, nocc, nocc], dtype=type_)
    
    #Contraction 76; Tree Level  2; Scaling  3/ 1 Result_size  3/ 1
    G12 += -1.0 * np.einsum('kaij->kaji',E5)
    
    #Contraction 77; Tree Level  2; Scaling  3/ 1 Result_size  3/ 1
    G12 += np.einsum('akij->kaji',E6)
    
    Y6 = np.zeros([nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 78; Tree Level  3; Scaling  4/ 2 Result_size  4/ 0
    Y6 += np.einsum('klic,cj->klij',V4,T1, optimize='optimal')
    
    G8 = np.zeros([nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 79; Tree Level  2; Scaling  4/ 0 Result_size  4/ 0
    G8 += -2.0 * np.einsum('klji->klij',Y6)
    
    #Contraction 80; Tree Level  2; Scaling  4/ 0 Result_size  4/ 0
    G8 += 2.0 * np.einsum('klij->klij',V1)
    
    A11 = np.zeros([nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 81; Tree Level  3; Scaling  4/ 0 Result_size  4/ 0
    A11 += np.einsum('klji->lkij',Y6)
    
    #Contraction 82; Tree Level  3; Scaling  4/ 0 Result_size  4/ 0
    A11 += -1.0 * np.einsum('lkji->lkij',Y6)
    
    E7 = np.zeros([nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 83; Tree Level  3; Scaling  4/ 2 Result_size  4/ 0
    E7 += np.einsum('dj,kldi->klji',T1,I1, optimize='optimal')
    
    #Contraction 84; Tree Level  2; Scaling  4/ 0 Result_size  4/ 0
    G8 += np.einsum('klji->klij',E7)
    
    #Contraction 85; Tree Level  2; Scaling  4/ 0 Result_size  4/ 0
    G8 += -1.0 * np.einsum('klij->klij',E7)
    
    I7 = np.zeros([nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 86; Tree Level  3; Scaling  5/ 1 Result_size  3/ 1
    I7 += np.einsum('al,lmji->amji',T1,E7, optimize='optimal')
    
    #Contraction 87; Tree Level  2; Scaling  4/ 0 Result_size  4/ 0
    G8 += 2.0 * np.einsum('klij->klij',Y6)
    
    #Contraction 88; Tree Level  2; Scaling  3/ 1 Result_size  3/ 1
    G12 += np.einsum('akij->kaji',J4)
    
    #Contraction 89; Tree Level  2; Scaling  3/ 1 Result_size  3/ 1
    G12 += -1.0 * np.einsum('akji->kaji',J4)
    
    A12 = np.zeros([nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 90; Tree Level  2; Scaling  3/ 1 Result_size  3/ 1
    A12 += 4.0 * np.einsum('kbij->bkji',Y3)
    
    #Contraction 91; Tree Level  2; Scaling  3/ 1 Result_size  3/ 1
    A12 += -4.0 * np.einsum('kbji->bkji',Y3)
    
    A8 = np.zeros([nvir, nvir, nocc, nocc], dtype=type_)
    
    #Contraction 92; Tree Level  2; Scaling  4/ 4 Result_size  2/ 2
    A8 += np.einsum('dbkj,akdi->baji',T2,G5, optimize='optimal')
    
    #Contraction 93; Tree Level  1; Scaling  2/ 2 Result_size  2/ 2
    Z2 += np.einsum('baji->abij',A8)
    
    #Contraction 94; Tree Level  1; Scaling  2/ 2 Result_size  2/ 2
    Z2 += -1.0 * np.einsum('abji->abij',A8)
    
    #Contraction 95; Tree Level  1; Scaling  2/ 2 Result_size  2/ 2
    Z2 += -1.0 * np.einsum('baij->abij',A8)
    
    #Contraction 96; Tree Level  1; Scaling  2/ 2 Result_size  2/ 2
    Z2 += np.einsum('abij->abij',A8)
    
    del A8
    
    D8 = np.zeros([nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 97; Tree Level  3; Scaling  5/ 1 Result_size  3/ 1
    D8 += np.einsum('bl,klji->bkji',T1,E7, optimize='optimal')
    
    #Contraction 98; Tree Level  2; Scaling  3/ 1 Result_size  3/ 1
    A12 += np.einsum('bkij->bkji',D8)
    
    #Contraction 99; Tree Level  2; Scaling  3/ 1 Result_size  3/ 1
    A12 += -1.0 * np.einsum('bkij->bkji',I7)
    
    #Contraction 100; Tree Level  1; Scaling  4/ 4 Result_size  2/ 2
    Z2 += np.einsum('kc,abcijk->abij',F3,T3, optimize='optimal')
    
    J8 = np.zeros([nocc, nocc], dtype=type_)
    
    #Contraction 101; Tree Level  3; Scaling  4/ 2 Result_size  2/ 0
    J8 += np.einsum('klic,ck->li',V4,T1, optimize='optimal')
    
    M8 = np.zeros([nvir, nvir, nocc, nocc], dtype=type_)
    
    #Contraction 102; Tree Level  2; Scaling  4/ 4 Result_size  2/ 2
    M8 += np.einsum('acik,kbcj->abij',T2,G2, optimize='optimal')
    
    #Contraction 103; Tree Level  1; Scaling  2/ 2 Result_size  2/ 2
    Z2 += 0.5 * np.einsum('abij->abij',M8)
    
    #Contraction 104; Tree Level  1; Scaling  2/ 2 Result_size  2/ 2
    Z2 += -0.5 * np.einsum('baij->abij',M8)
    
    #Contraction 105; Tree Level  1; Scaling  2/ 2 Result_size  2/ 2
    Z2 += -0.5 * np.einsum('abji->abij',M8)
    
    #Contraction 106; Tree Level  1; Scaling  2/ 2 Result_size  2/ 2
    Z2 += 0.5 * np.einsum('baji->abij',M8)
    
    del M8
    
    X8 = np.zeros([nvir, nvir, nocc, nocc], dtype=type_)
    
    #Contraction 107; Tree Level  2; Scaling  4/ 2 Result_size  2/ 2
    X8 += np.einsum('ablj,li->abji',T2,A5, optimize='optimal')
    
    #Contraction 108; Tree Level  1; Scaling  2/ 2 Result_size  2/ 2
    Z2 += np.einsum('abji->abij',X8)
    
    #Contraction 109; Tree Level  1; Scaling  2/ 2 Result_size  2/ 2
    Z2 += -1.0 * np.einsum('abij->abij',X8)
    
    del X8
    
    Y8 = np.zeros([nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 110; Tree Level  3; Scaling  5/ 1 Result_size  3/ 1
    Y8 += np.einsum('al,lmij->amij',T1,I5, optimize='optimal')
    
    A9 = np.zeros([nvir, nvir, nocc, nocc], dtype=type_)
    
    #Contraction 111; Tree Level  2; Scaling  6/ 4 Result_size  2/ 2
    A9 += np.einsum('klic,bacjkl->baij',V4,T3, optimize='optimal')
    
    #Contraction 112; Tree Level  1; Scaling  2/ 2 Result_size  2/ 2
    Z2 += -0.5 * np.einsum('baij->abij',A9)
    
    #Contraction 113; Tree Level  1; Scaling  2/ 2 Result_size  2/ 2
    Z2 += 0.5 * np.einsum('baji->abij',A9)
    
    del A9
    
    D9 = np.zeros([nvir, nvir, nocc, nocc], dtype=type_)
    
    #Contraction 114; Tree Level  2; Scaling  4/ 2 Result_size  2/ 2
    D9 += np.einsum('ak,bkji->abji',T1,Y5, optimize='optimal')
    
    #Contraction 115; Tree Level  1; Scaling  2/ 2 Result_size  2/ 2
    Z2 += -1.0 * np.einsum('abji->abij',D9)
    
    #Contraction 116; Tree Level  1; Scaling  2/ 2 Result_size  2/ 2
    Z2 += np.einsum('baji->abij',D9)
    
    #Contraction 117; Tree Level  1; Scaling  2/ 2 Result_size  2/ 2
    Z2 += np.einsum('abij->abij',D9)
    
    #Contraction 118; Tree Level  1; Scaling  2/ 2 Result_size  2/ 2
    Z2 += -1.0 * np.einsum('baij->abij',D9)
    
    del D9
    
    E9 = np.zeros([nvir, nvir, nocc, nocc], dtype=type_)
    
    #Contraction 119; Tree Level  2; Scaling  6/ 4 Result_size  2/ 2
    E9 += np.einsum('dabklj,kldi->abji',T3,I1, optimize='optimal')
    
    #Contraction 120; Tree Level  1; Scaling  2/ 2 Result_size  2/ 2
    Z2 += 0.5 * np.einsum('abji->abij',E9)
    
    #Contraction 121; Tree Level  1; Scaling  2/ 2 Result_size  2/ 2
    Z2 += -0.5 * np.einsum('abij->abij',E9)
    
    del E9
    
    G9 = np.zeros([nvir, nvir, nocc, nocc], dtype=type_)
    
    #Contraction 122; Tree Level  2; Scaling  4/ 2 Result_size  2/ 2
    G9 += np.einsum('abik,kj->abij',T2,M1, optimize='optimal')
    
    del M1
    
    #Contraction 123; Tree Level  1; Scaling  2/ 2 Result_size  2/ 2
    Z2 += 0.5 * np.einsum('abij->abij',G9)
    
    #Contraction 124; Tree Level  1; Scaling  2/ 2 Result_size  2/ 2
    Z2 += -0.5 * np.einsum('abji->abij',G9)
    
    del G9
    
    I9 = np.zeros([nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 125; Tree Level  3; Scaling  3/ 3 Result_size  3/ 1
    I9 += np.einsum('ci,akcj->akij',T1,M4, optimize='optimal')
    
    del M4
    
    J9 = np.zeros([nvir, nvir, nocc, nocc], dtype=type_)
    
    #Contraction 126; Tree Level  2; Scaling  4/ 2 Result_size  2/ 2
    J9 += np.einsum('bk,akij->baij',T1,I9, optimize='optimal')
    
    #Contraction 127; Tree Level  1; Scaling  2/ 2 Result_size  2/ 2
    Z2 += -0.5 * np.einsum('baij->abij',J9)
    
    #Contraction 128; Tree Level  1; Scaling  2/ 2 Result_size  2/ 2
    Z2 += 0.5 * np.einsum('abij->abij',J9)
    
    #Contraction 129; Tree Level  1; Scaling  2/ 2 Result_size  2/ 2
    Z2 += 0.5 * np.einsum('baji->abij',J9)
    
    #Contraction 130; Tree Level  1; Scaling  2/ 2 Result_size  2/ 2
    Z2 += -0.5 * np.einsum('abji->abij',J9)
    
    del J9
    
    M9 = np.zeros([nocc, nvir, nocc, nocc], dtype=type_)
    
    #Contraction 131; Tree Level  3; Scaling  5/ 5 Result_size  3/ 1
    M9 += np.einsum('klcd,cdblij->kbij',V7,T3, optimize='optimal')
    
    X9 = np.zeros([nvir, nvir, nocc, nocc], dtype=type_)
    
    #Contraction 132; Tree Level  2; Scaling  4/ 2 Result_size  2/ 2
    X9 += np.einsum('ak,kbij->abij',T1,M9, optimize='optimal')
    
    #Contraction 133; Tree Level  1; Scaling  2/ 2 Result_size  2/ 2
    Z2 += 0.5 * np.einsum('abij->abij',X9)
    
    #Contraction 134; Tree Level  1; Scaling  2/ 2 Result_size  2/ 2
    Z2 += -0.5 * np.einsum('baij->abij',X9)
    
    del X9
    
    #Contraction 135; Tree Level  1; Scaling  6/ 2 Result_size  2/ 2
    Z2 += 0.25 * np.einsum('abkl,klij->abij',T2,G8, optimize='optimal')
    
    del G8
    
    #Contraction 136; Tree Level  1; Scaling  2/ 6 Result_size  2/ 2
    Z2 += 0.5 * np.einsum('abcd,cdij->abij',V9,T2, optimize='optimal')
    
    #Contraction 137; Tree Level  1; Scaling  4/ 2 Result_size  2/ 2
    Z2 += 0.25 * np.einsum('ak,bkij->abij',T1,J5, optimize='optimal')
    
    del J5
    
    E10 = np.zeros([nvir, nvir, nocc, nocc], dtype=type_)
    
    #Contraction 138; Tree Level  2; Scaling  4/ 2 Result_size  2/ 2
    E10 += np.einsum('ablj,li->abji',T2,J8, optimize='optimal')
    
    #Contraction 139; Tree Level  1; Scaling  2/ 2 Result_size  2/ 2
    Z2 += np.einsum('abji->abij',E10)
    
    #Contraction 140; Tree Level  1; Scaling  2/ 2 Result_size  2/ 2
    Z2 += -1.0 * np.einsum('abij->abij',E10)
    
    del E10
    
    #Contraction 141; Tree Level  3; Scaling  4/ 0 Result_size  4/ 0
    A11 += -1.0 * np.einsum('klij->lkij',V1)
    
    #Contraction 142; Tree Level  3; Scaling  4/ 0 Result_size  4/ 0
    A11 += np.einsum('lkij->lkij',V1)
    
    I10 = np.zeros([nvir, nvir], dtype=type_)
    
    #Contraction 143; Tree Level  3; Scaling  2/ 4 Result_size  0/ 2
    I10 += np.einsum('akcd,ck->ad',V8,T1, optimize='optimal')
    
    J10 = np.zeros([nvir, nvir, nocc, nocc], dtype=type_)
    
    #Contraction 144; Tree Level  2; Scaling  2/ 4 Result_size  2/ 2
    J10 += np.einsum('dbij,ad->baij',T2,I10, optimize='optimal')
    
    #Contraction 145; Tree Level  1; Scaling  2/ 2 Result_size  2/ 2
    Z2 += -1.0 * np.einsum('baij->abij',J10)
    
    #Contraction 146; Tree Level  1; Scaling  2/ 2 Result_size  2/ 2
    Z2 += np.einsum('abij->abij',J10)
    
    del J10
    
    #Contraction 147; Tree Level  3; Scaling  4/ 0 Result_size  4/ 0
    A11 += -1.0 * np.einsum('klij->lkij',Y6)
    
    #Contraction 148; Tree Level  2; Scaling  3/ 1 Result_size  3/ 1
    A12 += -4.0 * np.einsum('bkij->bkji',J4)
    
    #Contraction 149; Tree Level  2; Scaling  3/ 1 Result_size  3/ 1
    A12 += 4.0 * np.einsum('bkji->bkji',J4)
    
    #Contraction 150; Tree Level  3; Scaling  4/ 0 Result_size  4/ 0
    A11 += np.einsum('lkij->lkij',Y6)
    
    #Contraction 151; Tree Level  2; Scaling  5/ 1 Result_size  3/ 1
    A12 += 2.0 * np.einsum('bl,lkij->bkji',T1,A11, optimize='optimal')
    
    del A11
    
    #Contraction 152; Tree Level  2; Scaling  3/ 1 Result_size  3/ 1
    A12 += -4.0 * np.einsum('bkij->bkji',E6)
    
    del E6
    
    #Contraction 153; Tree Level  2; Scaling  3/ 1 Result_size  3/ 1
    A12 += 4.0 * np.einsum('kbij->bkji',E5)
    
    del E5
    
    #Contraction 154; Tree Level  2; Scaling  3/ 1 Result_size  3/ 1
    A12 += np.einsum('bkij->bkji',Y8)
    
    #Contraction 155; Tree Level  2; Scaling  3/ 1 Result_size  3/ 1
    A12 += -1.0 * np.einsum('bkji->bkji',D8)
    
    del D8
    
    #Contraction 156; Tree Level  2; Scaling  3/ 1 Result_size  3/ 1
    A12 += np.einsum('bkji->bkji',I7)
    
    #Contraction 157; Tree Level  1; Scaling  4/ 2 Result_size  2/ 2
    Z2 += -0.25 * np.einsum('al,blji->abij',T1,A12, optimize='optimal')
    
    del A12
    
    #Contraction 158; Tree Level  2; Scaling  3/ 1 Result_size  3/ 1
    G12 += -1.0 * np.einsum('kaij->kaji',Y3)
    
    #Contraction 159; Tree Level  2; Scaling  3/ 1 Result_size  3/ 1
    G12 += np.einsum('kaji->kaji',Y3)
    
    #Contraction 160; Tree Level  1; Scaling  4/ 2 Result_size  2/ 2
    Z2 += -1.0 * np.einsum('bk,kaji->abij',T1,G12, optimize='optimal')
    
    del G12
    
    #Contraction 161; Tree Level  1; Scaling  2/ 2 Result_size  2/ 2
    Z2 += np.einsum('abij->abij',V3)
    
    #del V3
    
    J12 = np.zeros([nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 162; Tree Level  3; Scaling  6/ 4 Result_size  4/ 2
    J12 += np.einsum('ebcmjk,lmei->bcljki',T3,I1, optimize='optimal')
    
    M12 = np.zeros([nocc, nocc, nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 163; Tree Level  4; Scaling  5/ 3 Result_size  5/ 1
    M12 += np.einsum('lmid,dcjk->lmcijk',V4,T2, optimize='optimal')
    
    X12 = np.zeros([nocc, nvir, nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 164; Tree Level  3; Scaling  6/ 4 Result_size  4/ 2
    X12 += np.einsum('lmid,dbcmjk->lbcijk',V4,T3, optimize='optimal')
    
    Y12 = np.zeros([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 165; Tree Level  2; Scaling  5/ 3 Result_size  3/ 3
    Y12 += np.einsum('abil,lcjk->abcijk',T2,M9, optimize='optimal')
    
    #Contraction 166; Tree Level  0; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * np.einsum('abcijk->abcijk',Y12)
    
    #Contraction 167; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.5 * np.einsum('acbijk->abcijk',Y12)
    
    #Contraction 168; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * np.einsum('bcaijk->abcijk',Y12)
    
    #Contraction 169; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.5 * np.einsum('abcjik->abcijk',Y12)
    
    #Contraction 170; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * np.einsum('acbjik->abcijk',Y12)
    
    #Contraction 171; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.5 * np.einsum('bcajik->abcijk',Y12)
    
    #Contraction 172; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * np.einsum('abckij->abcijk',Y12)
    
    #Contraction 173; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.5 * np.einsum('acbkij->abcijk',Y12)
    
    #Contraction 174; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * np.einsum('bcakij->abcijk',Y12)
    
    del Y12
    
    A13 = np.zeros([nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 175; Tree Level  3; Scaling  5/ 1 Result_size  3/ 1
    A13 += np.einsum('al,lmij->amij',T1,Y6, optimize='optimal')
    
    D13 = np.zeros([nvir, nocc, nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 176; Tree Level  3; Scaling  6/ 2 Result_size  4/ 2
    D13 += np.einsum('bm,lmcijk->blcijk',T1,M12, optimize='optimal')
    
    E13 = np.zeros([nocc, nvir, nocc, nocc], dtype=type_)
    
    #Contraction 177; Tree Level  3; Scaling  5/ 1 Result_size  3/ 1
    E13 += np.einsum('lmij,al->maij',V1,T1, optimize='optimal')
    
    G13 = np.zeros([nocc, nocc, nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 178; Tree Level  4; Scaling  5/ 5 Result_size  5/ 1
    G13 += np.einsum('lmde,decijk->lmcijk',V7,T3, optimize='optimal')
    
    I13 = np.zeros([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 179; Tree Level  2; Scaling  5/ 3 Result_size  3/ 3
    I13 += np.einsum('abcmjk,mi->abcjki',T3,J8, optimize='optimal')
    
    #Contraction 180; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += np.einsum('abcjki->abcijk',I13)
    
    #Contraction 181; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * np.einsum('abcikj->abcijk',I13)
    
    #Contraction 182; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += np.einsum('abcijk->abcijk',I13)
    
    del I13
    
    J13 = np.zeros([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 183; Tree Level  2; Scaling  5/ 3 Result_size  3/ 3
    J13 += np.einsum('bcmk,amij->bcakij',T2,A13, optimize='optimal')
    
    #Contraction 184; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += np.einsum('bcakij->abcijk',J13)
    
    #Contraction 185; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * np.einsum('acbkij->abcijk',J13)
    
    #Contraction 186; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += np.einsum('abckij->abcijk',J13)
    
    #Contraction 187; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * np.einsum('bcajik->abcijk',J13)
    
    #Contraction 188; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += np.einsum('acbjik->abcijk',J13)
    
    #Contraction 189; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * np.einsum('abcjik->abcijk',J13)
    
    #Contraction 190; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * np.einsum('bcakji->abcijk',J13)
    
    #Contraction 191; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += np.einsum('acbkji->abcijk',J13)
    
    #Contraction 192; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * np.einsum('abckji->abcijk',J13)
    
    #Contraction 193; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += np.einsum('bcaijk->abcijk',J13)
    
    #Contraction 194; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * np.einsum('acbijk->abcijk',J13)
    
    #Contraction 195; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += np.einsum('abcijk->abcijk',J13)
    
    #Contraction 196; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += np.einsum('bcajki->abcijk',J13)
    
    #Contraction 197; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * np.einsum('acbjki->abcijk',J13)
    
    #Contraction 198; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += np.einsum('abcjki->abcijk',J13)
    
    #Contraction 199; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * np.einsum('bcaikj->abcijk',J13)
    
    #Contraction 200; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += np.einsum('acbikj->abcijk',J13)
    
    #Contraction 201; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * np.einsum('abcikj->abcijk',J13)
    
    del J13
    
    M13 = np.zeros([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 202; Tree Level  2; Scaling  7/ 5 Result_size  3/ 3
    M13 += np.einsum('eabclmjk,lmei->abcjki',T4,I1, optimize='optimal')
    
    #Contraction 203; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * np.einsum('abcjki->abcijk',M13)
    
    #Contraction 204; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.5 * np.einsum('abcikj->abcijk',M13)
    
    #Contraction 205; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * np.einsum('abcijk->abcijk',M13)
    
    del M13
    
    X13 = np.zeros([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 206; Tree Level  2; Scaling  3/ 5 Result_size  3/ 3
    X13 += np.einsum('abid,cdkj->abcikj',V6,T2, optimize='optimal')
    
    #Contraction 207; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += np.einsum('abcikj->abcijk',X13)
    
    #Contraction 208; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * np.einsum('acbikj->abcijk',X13)
    
    #Contraction 209; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += np.einsum('bcaikj->abcijk',X13)
    
    #Contraction 210; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * np.einsum('abckij->abcijk',X13)
    
    #Contraction 211; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += np.einsum('acbkij->abcijk',X13)
    
    #Contraction 212; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * np.einsum('bcakij->abcijk',X13)
    
    #Contraction 213; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += np.einsum('abcjik->abcijk',X13)
    
    #Contraction 214; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * np.einsum('acbjik->abcijk',X13)
    
    #Contraction 215; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += np.einsum('bcajik->abcijk',X13)
    
    del X13
    
    Y13 = np.zeros([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 216; Tree Level  2; Scaling  5/ 7 Result_size  3/ 3
    Y13 += np.einsum('alde,bcdejkil->abcjki',V8,T4, optimize='optimal')
    
    #Contraction 217; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * np.einsum('abcjki->abcijk',Y13)
    
    #Contraction 218; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.5 * np.einsum('bacjki->abcijk',Y13)
    
    #Contraction 219; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * np.einsum('cabjki->abcijk',Y13)
    
    del Y13
    
    A14 = np.zeros([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 220; Tree Level  2; Scaling  5/ 3 Result_size  3/ 3
    A14 += np.einsum('bclk,alij->bcakij',T2,J4, optimize='optimal')
    
    #Contraction 221; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * np.einsum('bcakij->abcijk',A14)
    
    #Contraction 222; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += np.einsum('acbkij->abcijk',A14)
    
    #Contraction 223; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * np.einsum('abckij->abcijk',A14)
    
    #Contraction 224; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += np.einsum('bcajik->abcijk',A14)
    
    #Contraction 225; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * np.einsum('acbjik->abcijk',A14)
    
    #Contraction 226; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += np.einsum('abcjik->abcijk',A14)
    
    #Contraction 227; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += np.einsum('bcakji->abcijk',A14)
    
    #Contraction 228; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * np.einsum('acbkji->abcijk',A14)
    
    #Contraction 229; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += np.einsum('abckji->abcijk',A14)
    
    #Contraction 230; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * np.einsum('bcaijk->abcijk',A14)
    
    #Contraction 231; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += np.einsum('acbijk->abcijk',A14)
    
    #Contraction 232; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * np.einsum('abcijk->abcijk',A14)
    
    #Contraction 233; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * np.einsum('bcajki->abcijk',A14)
    
    #Contraction 234; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += np.einsum('acbjki->abcijk',A14)
    
    #Contraction 235; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * np.einsum('abcjki->abcijk',A14)
    
    #Contraction 236; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += np.einsum('bcaikj->abcijk',A14)
    
    #Contraction 237; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * np.einsum('acbikj->abcijk',A14)
    
    #Contraction 238; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += np.einsum('abcikj->abcijk',A14)
    
    del A14
    
    D14 = np.zeros([nvir, nocc, nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 239; Tree Level  3; Scaling  4/ 6 Result_size  4/ 2
    D14 += np.einsum('alde,decijk->alcijk',V8,T3, optimize='optimal')
    
    E14 = np.zeros([nocc, nvir, nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 240; Tree Level  3; Scaling  6/ 6 Result_size  4/ 2
    E14 += np.einsum('lmde,debcmijk->lbcijk',V7,T4, optimize='optimal')
    
    G14 = np.zeros([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 241; Tree Level  2; Scaling  5/ 3 Result_size  3/ 3
    G14 += np.einsum('al,lbcijk->abcijk',T1,E14, optimize='optimal')
    
    #Contraction 242; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * np.einsum('abcijk->abcijk',G14)
    
    #Contraction 243; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.5 * np.einsum('bacijk->abcijk',G14)
    
    #Contraction 244; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * np.einsum('cabijk->abcijk',G14)
    
    del G14
    
    I14 = np.zeros([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 245; Tree Level  2; Scaling  5/ 3 Result_size  3/ 3
    I14 += np.einsum('bclk,alij->bcakij',T2,J6, optimize='optimal')
    
    #Contraction 246; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.5 * np.einsum('bcakij->abcijk',I14)
    
    #Contraction 247; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * np.einsum('acbkij->abcijk',I14)
    
    #Contraction 248; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.5 * np.einsum('abckij->abcijk',I14)
    
    #Contraction 249; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * np.einsum('bcajik->abcijk',I14)
    
    #Contraction 250; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.5 * np.einsum('acbjik->abcijk',I14)
    
    #Contraction 251; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * np.einsum('abcjik->abcijk',I14)
    
    #Contraction 252; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.5 * np.einsum('bcaijk->abcijk',I14)
    
    #Contraction 253; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * np.einsum('acbijk->abcijk',I14)
    
    #Contraction 254; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.5 * np.einsum('abcijk->abcijk',I14)
    
    del I14
    
    J14 = np.zeros([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 255; Tree Level  2; Scaling  5/ 3 Result_size  3/ 3
    J14 += np.einsum('al,bcljki->abcjki',T1,J12, optimize='optimal')
    
    #Contraction 256; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * np.einsum('abcjki->abcijk',J14)
    
    #Contraction 257; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += np.einsum('bacjki->abcijk',J14)
    
    #Contraction 258; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * np.einsum('cabjki->abcijk',J14)
    
    #Contraction 259; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += np.einsum('abcikj->abcijk',J14)
    
    #Contraction 260; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * np.einsum('bacikj->abcijk',J14)
    
    #Contraction 261; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += np.einsum('cabikj->abcijk',J14)
    
    #Contraction 262; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * np.einsum('abcijk->abcijk',J14)
    
    #Contraction 263; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += np.einsum('bacijk->abcijk',J14)
    
    #Contraction 264; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * np.einsum('cabijk->abcijk',J14)
    
    del J14
    
    M14 = np.zeros([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 265; Tree Level  2; Scaling  7/ 3 Result_size  3/ 3
    M14 += np.einsum('lmij,cabklm->cabijk',V1,T3, optimize='optimal')
    
    #Contraction 266; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * np.einsum('cabijk->abcijk',M14)
    
    #Contraction 267; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.5 * np.einsum('cabikj->abcijk',M14)
    
    #Contraction 268; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * np.einsum('cabjki->abcijk',M14)
    
    del M14
    
    X14 = np.zeros([nvir, nocc, nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 269; Tree Level  3; Scaling  6/ 2 Result_size  4/ 2
    X14 += np.einsum('bm,lmcijk->blcijk',T1,G13, optimize='optimal')
    
    Y14 = np.zeros([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 270; Tree Level  2; Scaling  5/ 3 Result_size  3/ 3
    Y14 += np.einsum('al,blcijk->abcijk',T1,X14, optimize='optimal')
    
    del X14
    
    #Contraction 271; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.25 * np.einsum('abcijk->abcijk',Y14)
    
    #Contraction 272; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.25 * np.einsum('acbijk->abcijk',Y14)
    
    #Contraction 273; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.25 * np.einsum('bacijk->abcijk',Y14)
    
    #Contraction 274; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.25 * np.einsum('bcaijk->abcijk',Y14)
    
    #Contraction 275; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.25 * np.einsum('cabijk->abcijk',Y14)
    
    #Contraction 276; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.25 * np.einsum('cbaijk->abcijk',Y14)
    
    del Y14
    
    A15 = np.zeros([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 277; Tree Level  2; Scaling  5/ 3 Result_size  3/ 3
    A15 += np.einsum('li,bcajkl->bcaijk',F1,T3, optimize='optimal')
    
    #Contraction 278; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * np.einsum('bcaijk->abcijk',A15)
    
    #Contraction 279; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += np.einsum('bcajik->abcijk',A15)
    
    #Contraction 280; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * np.einsum('bcakij->abcijk',A15)
    
    del A15
    
    D15 = np.zeros([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 281; Tree Level  2; Scaling  5/ 5 Result_size  3/ 3
    D15 += np.einsum('ebcljk,alei->bcajki',T3,G5, optimize='optimal')
    
    #Contraction 282; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += np.einsum('bcajki->abcijk',D15)
    
    #Contraction 283; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * np.einsum('acbjki->abcijk',D15)
    
    #Contraction 284; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += np.einsum('abcjki->abcijk',D15)
    
    #Contraction 285; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * np.einsum('bcaikj->abcijk',D15)
    
    #Contraction 286; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += np.einsum('acbikj->abcijk',D15)
    
    #Contraction 287; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * np.einsum('abcikj->abcijk',D15)
    
    #Contraction 288; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += np.einsum('bcaijk->abcijk',D15)
    
    #Contraction 289; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * np.einsum('acbijk->abcijk',D15)
    
    #Contraction 290; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += np.einsum('abcijk->abcijk',D15)
    
    del D15
    
    E15 = np.zeros([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 291; Tree Level  2; Scaling  5/ 3 Result_size  3/ 3
    E15 += np.einsum('ablj,clki->abcjki',T2,Y5, optimize='optimal')
    
    del Y5
    
    #Contraction 292; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * np.einsum('abcjki->abcijk',E15)
    
    #Contraction 293; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += np.einsum('acbjki->abcijk',E15)
    
    #Contraction 294; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * np.einsum('bcajki->abcijk',E15)
    
    #Contraction 295; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += np.einsum('abckji->abcijk',E15)
    
    #Contraction 296; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * np.einsum('acbkji->abcijk',E15)
    
    #Contraction 297; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += np.einsum('bcakji->abcijk',E15)
    
    #Contraction 298; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += np.einsum('abcikj->abcijk',E15)
    
    #Contraction 299; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * np.einsum('acbikj->abcijk',E15)
    
    #Contraction 300; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += np.einsum('bcaikj->abcijk',E15)
    
    #Contraction 301; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * np.einsum('abckij->abcijk',E15)
    
    #Contraction 302; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += np.einsum('acbkij->abcijk',E15)
    
    #Contraction 303; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * np.einsum('bcakij->abcijk',E15)
    
    #Contraction 304; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * np.einsum('abcijk->abcijk',E15)
    
    #Contraction 305; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += np.einsum('acbijk->abcijk',E15)
    
    #Contraction 306; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * np.einsum('bcaijk->abcijk',E15)
    
    #Contraction 307; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += np.einsum('abcjik->abcijk',E15)
    
    #Contraction 308; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * np.einsum('acbjik->abcijk',E15)
    
    #Contraction 309; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += np.einsum('bcajik->abcijk',E15)
    
    del E15
    
    G15 = np.zeros([nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 310; Tree Level  3; Scaling  4/ 4 Result_size  4/ 2
    G15 += np.einsum('ebcijk,le->bclijk',T3,D6, optimize='optimal')
    
    I15 = np.zeros([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 311; Tree Level  2; Scaling  3/ 5 Result_size  3/ 3
    I15 += np.einsum('ebcijk,ae->bcaijk',T3,I10, optimize='optimal')
    
    #Contraction 312; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * np.einsum('bcaijk->abcijk',I15)
    
    #Contraction 313; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += np.einsum('acbijk->abcijk',I15)
    
    #Contraction 314; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * np.einsum('abcijk->abcijk',I15)
    
    del I15
    
    J15 = np.zeros([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 315; Tree Level  2; Scaling  5/ 3 Result_size  3/ 3
    J15 += np.einsum('bclk,alij->bcakij',T2,I9, optimize='optimal')
    
    #Contraction 316; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.5 * np.einsum('bcakij->abcijk',J15)
    
    #Contraction 317; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * np.einsum('acbkij->abcijk',J15)
    
    #Contraction 318; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.5 * np.einsum('abckij->abcijk',J15)
    
    #Contraction 319; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * np.einsum('bcajik->abcijk',J15)
    
    #Contraction 320; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.5 * np.einsum('acbjik->abcijk',J15)
    
    #Contraction 321; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * np.einsum('abcjik->abcijk',J15)
    
    #Contraction 322; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * np.einsum('bcakji->abcijk',J15)
    
    #Contraction 323; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.5 * np.einsum('acbkji->abcijk',J15)
    
    #Contraction 324; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * np.einsum('abckji->abcijk',J15)
    
    #Contraction 325; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.5 * np.einsum('bcaijk->abcijk',J15)
    
    #Contraction 326; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * np.einsum('acbijk->abcijk',J15)
    
    #Contraction 327; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.5 * np.einsum('abcijk->abcijk',J15)
    
    #Contraction 328; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.5 * np.einsum('bcajki->abcijk',J15)
    
    #Contraction 329; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * np.einsum('acbjki->abcijk',J15)
    
    #Contraction 330; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.5 * np.einsum('abcjki->abcijk',J15)
    
    #Contraction 331; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * np.einsum('bcaikj->abcijk',J15)
    
    #Contraction 332; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.5 * np.einsum('acbikj->abcijk',J15)
    
    #Contraction 333; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * np.einsum('abcikj->abcijk',J15)
    
    del J15
    
    M15 = np.zeros([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 334; Tree Level  2; Scaling  5/ 3 Result_size  3/ 3
    M15 += np.einsum('alij,cbkl->acbijk',V2,T2, optimize='optimal')
    
    #Contraction 335; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * np.einsum('acbijk->abcijk',M15)
    
    #Contraction 336; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += np.einsum('cabijk->abcijk',M15)
    
    #Contraction 337; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * np.einsum('bacijk->abcijk',M15)
    
    #Contraction 338; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += np.einsum('acbikj->abcijk',M15)
    
    #Contraction 339; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * np.einsum('cabikj->abcijk',M15)
    
    #Contraction 340; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += np.einsum('bacikj->abcijk',M15)
    
    #Contraction 341; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * np.einsum('acbjki->abcijk',M15)
    
    #Contraction 342; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += np.einsum('cabjki->abcijk',M15)
    
    #Contraction 343; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * np.einsum('bacjki->abcijk',M15)
    
    del M15
    
    X15 = np.zeros([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 344; Tree Level  2; Scaling  5/ 5 Result_size  3/ 3
    X15 += np.einsum('alid,bcdjkl->abcijk',V5,T3, optimize='optimal')
    
    #Contraction 345; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += np.einsum('abcijk->abcijk',X15)
    
    #Contraction 346; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * np.einsum('bacijk->abcijk',X15)
    
    #Contraction 347; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += np.einsum('cabijk->abcijk',X15)
    
    #Contraction 348; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * np.einsum('abcjik->abcijk',X15)
    
    #Contraction 349; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += np.einsum('bacjik->abcijk',X15)
    
    #Contraction 350; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * np.einsum('cabjik->abcijk',X15)
    
    #Contraction 351; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += np.einsum('abckij->abcijk',X15)
    
    #Contraction 352; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * np.einsum('backij->abcijk',X15)
    
    #Contraction 353; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += np.einsum('cabkij->abcijk',X15)
    
    del X15
    
    Y15 = np.zeros([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 354; Tree Level  2; Scaling  5/ 3 Result_size  3/ 3
    Y15 += np.einsum('bl,alcijk->bacijk',T1,D14, optimize='optimal')
    
    #Contraction 355; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.5 * np.einsum('bacijk->abcijk',Y15)
    
    #Contraction 356; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * np.einsum('cabijk->abcijk',Y15)
    
    #Contraction 357; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * np.einsum('abcijk->abcijk',Y15)
    
    #Contraction 358; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.5 * np.einsum('cbaijk->abcijk',Y15)
    
    #Contraction 359; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.5 * np.einsum('acbijk->abcijk',Y15)
    
    #Contraction 360; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * np.einsum('bcaijk->abcijk',Y15)
    
    del Y15
    
    A16 = np.zeros([nvir, nvir], dtype=type_)
    
    #Contraction 361; Tree Level  3; Scaling  4/ 4 Result_size  0/ 2
    A16 += np.einsum('lmde,adlm->ae',V7,T2, optimize='optimal')
    
    D16 = np.zeros([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 362; Tree Level  2; Scaling  5/ 3 Result_size  3/ 3
    D16 += np.einsum('abcljk,li->abcjki',T3,Y1, optimize='optimal')
    
    #Contraction 363; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * np.einsum('abcjki->abcijk',D16)
    
    #Contraction 364; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += np.einsum('abcikj->abcijk',D16)
    
    #Contraction 365; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * np.einsum('abcijk->abcijk',D16)
    
    del D16
    
    E16 = np.zeros([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 366; Tree Level  2; Scaling  5/ 3 Result_size  3/ 3
    E16 += np.einsum('al,lbcijk->abcijk',T1,X12, optimize='optimal')
    
    #Contraction 367; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * np.einsum('abcijk->abcijk',E16)
    
    #Contraction 368; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += np.einsum('bacijk->abcijk',E16)
    
    #Contraction 369; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * np.einsum('cabijk->abcijk',E16)
    
    #Contraction 370; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += np.einsum('abcjik->abcijk',E16)
    
    #Contraction 371; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * np.einsum('bacjik->abcijk',E16)
    
    #Contraction 372; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += np.einsum('cabjik->abcijk',E16)
    
    #Contraction 373; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * np.einsum('abckij->abcijk',E16)
    
    #Contraction 374; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += np.einsum('backij->abcijk',E16)
    
    #Contraction 375; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * np.einsum('cabkij->abcijk',E16)
    
    del E16
    
    G16 = np.zeros([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 376; Tree Level  2; Scaling  5/ 3 Result_size  3/ 3
    G16 += np.einsum('abcmjk,mi->abcjki',T3,A5, optimize='optimal')
    
    #Contraction 377; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += np.einsum('abcjki->abcijk',G16)
    
    #Contraction 378; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * np.einsum('abcikj->abcijk',G16)
    
    #Contraction 379; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += np.einsum('abcijk->abcijk',G16)
    
    del G16
    
    I16 = np.zeros([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 380; Tree Level  2; Scaling  7/ 3 Result_size  3/ 3
    I16 += np.einsum('abclmk,lmij->abckij',T3,Y6, optimize='optimal')
    
    #Contraction 381; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * np.einsum('abckij->abcijk',I16)
    
    #Contraction 382; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.5 * np.einsum('abcjik->abcijk',I16)
    
    #Contraction 383; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.5 * np.einsum('abckji->abcijk',I16)
    
    #Contraction 384; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * np.einsum('abcijk->abcijk',I16)
    
    #Contraction 385; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * np.einsum('abcjki->abcijk',I16)
    
    #Contraction 386; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.5 * np.einsum('abcikj->abcijk',I16)
    
    del I16
    
    J16 = np.zeros([nvir, nvir, nvir, nocc], dtype=type_)
    
    #Contraction 387; Tree Level  3; Scaling  3/ 5 Result_size  1/ 3
    J16 += np.einsum('alde,eclk->acdk',V8,T2, optimize='optimal')
    
    M16 = np.zeros([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 388; Tree Level  2; Scaling  5/ 3 Result_size  3/ 3
    M16 += np.einsum('al,bclijk->abcijk',T1,G15, optimize='optimal')
    
    del G15
    
    #Contraction 389; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += np.einsum('abcijk->abcijk',M16)
    
    #Contraction 390; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * np.einsum('bacijk->abcijk',M16)
    
    #Contraction 391; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += np.einsum('cabijk->abcijk',M16)
    
    del M16
    
    X16 = np.zeros([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 392; Tree Level  2; Scaling  3/ 5 Result_size  3/ 3
    X16 += np.einsum('ad,bcdjki->abcjki',F4,T3, optimize='optimal')
    
    #Contraction 393; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += np.einsum('abcjki->abcijk',X16)
    
    #Contraction 394; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * np.einsum('bacjki->abcijk',X16)
    
    #Contraction 395; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += np.einsum('cabjki->abcijk',X16)
    
    del X16
    
    Y16 = np.zeros([nocc, nocc], dtype=type_)
    
    #Contraction 396; Tree Level  3; Scaling  4/ 4 Result_size  2/ 0
    Y16 += np.einsum('lmde,deil->mi',V7,T2, optimize='optimal')
    
    A17 = np.zeros([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 397; Tree Level  2; Scaling  5/ 3 Result_size  3/ 3
    A17 += np.einsum('abcmjk,mi->abcjki',T3,Y16, optimize='optimal')
    
    #Contraction 398; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * np.einsum('abcjki->abcijk',A17)
    
    #Contraction 399; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.5 * np.einsum('abcikj->abcijk',A17)
    
    #Contraction 400; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * np.einsum('abcijk->abcijk',A17)
    
    del A17
    
    D17 = np.zeros([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 401; Tree Level  2; Scaling  5/ 3 Result_size  3/ 3
    D17 += np.einsum('bcmk,amij->bcakij',T2,Y8, optimize='optimal')
    
    #Contraction 402; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * np.einsum('bcakij->abcijk',D17)
    
    #Contraction 403; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.5 * np.einsum('acbkij->abcijk',D17)
    
    #Contraction 404; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * np.einsum('abckij->abcijk',D17)
    
    #Contraction 405; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.5 * np.einsum('bcajik->abcijk',D17)
    
    #Contraction 406; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * np.einsum('acbjik->abcijk',D17)
    
    #Contraction 407; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.5 * np.einsum('abcjik->abcijk',D17)
    
    #Contraction 408; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * np.einsum('bcaijk->abcijk',D17)
    
    #Contraction 409; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.5 * np.einsum('acbijk->abcijk',D17)
    
    #Contraction 410; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * np.einsum('abcijk->abcijk',D17)
    
    del D17
    
    E17 = np.zeros([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 411; Tree Level  2; Scaling  5/ 3 Result_size  3/ 3
    E17 += np.einsum('al,blcijk->abcijk',T1,D13, optimize='optimal')
    
    del D13
    
    #Contraction 412; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * np.einsum('abcijk->abcijk',E17)
    
    #Contraction 413; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.5 * np.einsum('acbijk->abcijk',E17)
    
    #Contraction 414; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.5 * np.einsum('bacijk->abcijk',E17)
    
    #Contraction 415; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * np.einsum('bcaijk->abcijk',E17)
    
    #Contraction 416; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * np.einsum('cabijk->abcijk',E17)
    
    #Contraction 417; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.5 * np.einsum('cbaijk->abcijk',E17)
    
    #Contraction 418; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.5 * np.einsum('abcjik->abcijk',E17)
    
    #Contraction 419; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * np.einsum('acbjik->abcijk',E17)
    
    #Contraction 420; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * np.einsum('bacjik->abcijk',E17)
    
    #Contraction 421; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.5 * np.einsum('bcajik->abcijk',E17)
    
    #Contraction 422; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.5 * np.einsum('cabjik->abcijk',E17)
    
    #Contraction 423; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * np.einsum('cbajik->abcijk',E17)
    
    #Contraction 424; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * np.einsum('abckij->abcijk',E17)
    
    #Contraction 425; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.5 * np.einsum('acbkij->abcijk',E17)
    
    #Contraction 426; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.5 * np.einsum('backij->abcijk',E17)
    
    #Contraction 427; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * np.einsum('bcakij->abcijk',E17)
    
    #Contraction 428; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * np.einsum('cabkij->abcijk',E17)
    
    #Contraction 429; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.5 * np.einsum('cbakij->abcijk',E17)
    
    del E17
    
    G17 = np.zeros([nocc, nvir, nvir, nocc], dtype=type_)
    
    #Contraction 430; Tree Level  3; Scaling  4/ 4 Result_size  2/ 2
    G17 += np.einsum('lmde,adil->maei',V7,T2, optimize='optimal')
    
    I17 = np.zeros([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 431; Tree Level  2; Scaling  5/ 5 Result_size  3/ 3
    I17 += np.einsum('ebcmjk,maei->bcajki',T3,G17, optimize='optimal')
    
    #Contraction 432; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += np.einsum('bcajki->abcijk',I17)
    
    #Contraction 433; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * np.einsum('acbjki->abcijk',I17)
    
    #Contraction 434; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += np.einsum('abcjki->abcijk',I17)
    
    #Contraction 435; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * np.einsum('bcaikj->abcijk',I17)
    
    #Contraction 436; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += np.einsum('acbikj->abcijk',I17)
    
    #Contraction 437; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * np.einsum('abcikj->abcijk',I17)
    
    #Contraction 438; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += np.einsum('bcaijk->abcijk',I17)
    
    #Contraction 439; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * np.einsum('acbijk->abcijk',I17)
    
    #Contraction 440; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += np.einsum('abcijk->abcijk',I17)
    
    del I17
    
    J17 = np.zeros([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 441; Tree Level  2; Scaling  7/ 5 Result_size  3/ 3
    J17 += np.einsum('lmid,bcadjklm->bcaijk',V4,T4, optimize='optimal')
    
    #Contraction 442; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.5 * np.einsum('bcaijk->abcijk',J17)
    
    #Contraction 443; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * np.einsum('bcajik->abcijk',J17)
    
    #Contraction 444; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.5 * np.einsum('bcakij->abcijk',J17)
    
    del J17
    
    M17 = np.zeros([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 445; Tree Level  2; Scaling  5/ 3 Result_size  3/ 3
    M17 += np.einsum('bcmk,maij->bcakij',T2,E13, optimize='optimal')
    
    #Contraction 446; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += np.einsum('bcakij->abcijk',M17)
    
    #Contraction 447; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * np.einsum('acbkij->abcijk',M17)
    
    #Contraction 448; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += np.einsum('abckij->abcijk',M17)
    
    #Contraction 449; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * np.einsum('bcajik->abcijk',M17)
    
    #Contraction 450; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += np.einsum('acbjik->abcijk',M17)
    
    #Contraction 451; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * np.einsum('abcjik->abcijk',M17)
    
    #Contraction 452; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += np.einsum('bcaijk->abcijk',M17)
    
    #Contraction 453; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * np.einsum('acbijk->abcijk',M17)
    
    #Contraction 454; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += np.einsum('abcijk->abcijk',M17)
    
    del M17
    
    X17 = np.zeros([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 455; Tree Level  2; Scaling  3/ 5 Result_size  3/ 3
    X17 += np.einsum('bdji,acdk->bacjik',T2,J16, optimize='optimal')
    
    del J16
    
    #Contraction 456; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += np.einsum('bacjik->abcijk',X17)
    
    #Contraction 457; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * np.einsum('cabjik->abcijk',X17)
    
    #Contraction 458; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * np.einsum('abcjik->abcijk',X17)
    
    #Contraction 459; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += np.einsum('cbajik->abcijk',X17)
    
    #Contraction 460; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += np.einsum('acbjik->abcijk',X17)
    
    #Contraction 461; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * np.einsum('bcajik->abcijk',X17)
    
    #Contraction 462; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * np.einsum('bacjki->abcijk',X17)
    
    #Contraction 463; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += np.einsum('cabjki->abcijk',X17)
    
    #Contraction 464; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += np.einsum('abcjki->abcijk',X17)
    
    #Contraction 465; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * np.einsum('cbajki->abcijk',X17)
    
    #Contraction 466; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * np.einsum('acbjki->abcijk',X17)
    
    #Contraction 467; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += np.einsum('bcajki->abcijk',X17)
    
    #Contraction 468; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += np.einsum('bacikj->abcijk',X17)
    
    #Contraction 469; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * np.einsum('cabikj->abcijk',X17)
    
    #Contraction 470; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * np.einsum('abcikj->abcijk',X17)
    
    #Contraction 471; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += np.einsum('cbaikj->abcijk',X17)
    
    #Contraction 472; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += np.einsum('acbikj->abcijk',X17)
    
    #Contraction 473; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * np.einsum('bcaikj->abcijk',X17)
    
    del X17
    
    Y17 = np.zeros([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 474; Tree Level  2; Scaling  7/ 3 Result_size  3/ 3
    Y17 += np.einsum('abclmk,lmji->abckji',T3,E7, optimize='optimal')
    
    #Contraction 475; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.25 * np.einsum('abckji->abcijk',Y17)
    
    #Contraction 476; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.25 * np.einsum('abcjki->abcijk',Y17)
    
    #Contraction 477; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.25 * np.einsum('abckij->abcijk',Y17)
    
    #Contraction 478; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.25 * np.einsum('abcikj->abcijk',Y17)
    
    #Contraction 479; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.25 * np.einsum('abcjik->abcijk',Y17)
    
    #Contraction 480; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.25 * np.einsum('abcijk->abcijk',Y17)
    
    del Y17
    
    A18 = np.zeros([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 481; Tree Level  2; Scaling  5/ 3 Result_size  3/ 3
    A18 += np.einsum('bajl,lcik->bacjik',T2,Y3, optimize='optimal')
    
    del Y3
    
    #Contraction 482; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * np.einsum('bacjik->abcijk',A18)
    
    #Contraction 483; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += np.einsum('bcajik->abcijk',A18)
    
    #Contraction 484; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * np.einsum('acbjik->abcijk',A18)
    
    #Contraction 485; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += np.einsum('backij->abcijk',A18)
    
    #Contraction 486; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * np.einsum('bcakij->abcijk',A18)
    
    #Contraction 487; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += np.einsum('acbkij->abcijk',A18)
    
    #Contraction 488; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += np.einsum('bacijk->abcijk',A18)
    
    #Contraction 489; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * np.einsum('bcaijk->abcijk',A18)
    
    #Contraction 490; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += np.einsum('acbijk->abcijk',A18)
    
    #Contraction 491; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * np.einsum('backji->abcijk',A18)
    
    #Contraction 492; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += np.einsum('bcakji->abcijk',A18)
    
    #Contraction 493; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * np.einsum('acbkji->abcijk',A18)
    
    #Contraction 494; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * np.einsum('bacikj->abcijk',A18)
    
    #Contraction 495; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += np.einsum('bcaikj->abcijk',A18)
    
    #Contraction 496; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * np.einsum('acbikj->abcijk',A18)
    
    #Contraction 497; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += np.einsum('bacjki->abcijk',A18)
    
    #Contraction 498; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * np.einsum('bcajki->abcijk',A18)
    
    #Contraction 499; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += np.einsum('acbjki->abcijk',A18)
    
    del A18
    
    D18 = np.zeros([nvir, nocc, nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 500; Tree Level  3; Scaling  4/ 4 Result_size  4/ 2
    D18 += np.einsum('alid,dcjk->alcijk',V5,T2, optimize='optimal')
    
    E18 = np.zeros([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 501; Tree Level  2; Scaling  5/ 3 Result_size  3/ 3
    E18 += np.einsum('bl,alcijk->bacijk',T1,D18, optimize='optimal')
    
    del D18
    
    #Contraction 502; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * np.einsum('bacijk->abcijk',E18)
    
    #Contraction 503; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += np.einsum('cabijk->abcijk',E18)
    
    #Contraction 504; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += np.einsum('abcijk->abcijk',E18)
    
    #Contraction 505; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * np.einsum('cbaijk->abcijk',E18)
    
    #Contraction 506; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * np.einsum('acbijk->abcijk',E18)
    
    #Contraction 507; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += np.einsum('bcaijk->abcijk',E18)
    
    #Contraction 508; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += np.einsum('bacjik->abcijk',E18)
    
    #Contraction 509; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * np.einsum('cabjik->abcijk',E18)
    
    #Contraction 510; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * np.einsum('abcjik->abcijk',E18)
    
    #Contraction 511; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += np.einsum('cbajik->abcijk',E18)
    
    #Contraction 512; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += np.einsum('acbjik->abcijk',E18)
    
    #Contraction 513; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * np.einsum('bcajik->abcijk',E18)
    
    #Contraction 514; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * np.einsum('backij->abcijk',E18)
    
    #Contraction 515; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += np.einsum('cabkij->abcijk',E18)
    
    #Contraction 516; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += np.einsum('abckij->abcijk',E18)
    
    #Contraction 517; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * np.einsum('cbakij->abcijk',E18)
    
    #Contraction 518; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * np.einsum('acbkij->abcijk',E18)
    
    #Contraction 519; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += np.einsum('bcakij->abcijk',E18)
    
    del E18
    
    G18 = np.zeros([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 520; Tree Level  2; Scaling  5/ 3 Result_size  3/ 3
    G18 += np.einsum('bcmk,amji->bcakji',T2,I7, optimize='optimal')
    
    #Contraction 521; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * np.einsum('bcakji->abcijk',G18)
    
    #Contraction 522; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.5 * np.einsum('acbkji->abcijk',G18)
    
    #Contraction 523; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * np.einsum('abckji->abcijk',G18)
    
    #Contraction 524; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.5 * np.einsum('bcajki->abcijk',G18)
    
    #Contraction 525; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * np.einsum('acbjki->abcijk',G18)
    
    #Contraction 526; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.5 * np.einsum('abcjki->abcijk',G18)
    
    #Contraction 527; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.5 * np.einsum('bcakij->abcijk',G18)
    
    #Contraction 528; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * np.einsum('acbkij->abcijk',G18)
    
    #Contraction 529; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.5 * np.einsum('abckij->abcijk',G18)
    
    #Contraction 530; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * np.einsum('bcaikj->abcijk',G18)
    
    #Contraction 531; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.5 * np.einsum('acbikj->abcijk',G18)
    
    #Contraction 532; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * np.einsum('abcikj->abcijk',G18)
    
    #Contraction 533; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * np.einsum('bcajik->abcijk',G18)
    
    #Contraction 534; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.5 * np.einsum('acbjik->abcijk',G18)
    
    #Contraction 535; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * np.einsum('abcjik->abcijk',G18)
    
    #Contraction 536; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.5 * np.einsum('bcaijk->abcijk',G18)
    
    #Contraction 537; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * np.einsum('acbijk->abcijk',G18)
    
    #Contraction 538; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.5 * np.einsum('abcijk->abcijk',G18)
    
    del G18
    
    I18 = np.zeros([nocc, nvir, nocc, nocc], dtype=type_)
    
    #Contraction 539; Tree Level  3; Scaling  3/ 3 Result_size  3/ 1
    I18 += np.einsum('ld,adij->laij',F3,T2, optimize='optimal')
    
    J18 = np.zeros([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 540; Tree Level  2; Scaling  5/ 3 Result_size  3/ 3
    J18 += np.einsum('bclk,laij->bcakij',T2,I18, optimize='optimal')
    
    #Contraction 541; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * np.einsum('bcakij->abcijk',J18)
    
    #Contraction 542; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += np.einsum('acbkij->abcijk',J18)
    
    #Contraction 543; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * np.einsum('abckij->abcijk',J18)
    
    #Contraction 544; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += np.einsum('bcajik->abcijk',J18)
    
    #Contraction 545; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * np.einsum('acbjik->abcijk',J18)
    
    #Contraction 546; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += np.einsum('abcjik->abcijk',J18)
    
    #Contraction 547; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * np.einsum('bcaijk->abcijk',J18)
    
    #Contraction 548; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += np.einsum('acbijk->abcijk',J18)
    
    #Contraction 549; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * np.einsum('abcijk->abcijk',J18)
    
    del J18
    
    M18 = np.zeros([nvir, nvir, nocc, nvir], dtype=type_)
    
    #Contraction 550; Tree Level  3; Scaling  5/ 3 Result_size  1/ 3
    M18 += np.einsum('lmid,aclm->acid',V4,T2, optimize='optimal')
    
    X18 = np.zeros([nvir, nocc, nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 551; Tree Level  3; Scaling  5/ 3 Result_size  5/ 1
    X18 += np.einsum('ecjk,lmei->clmjki',T2,I1, optimize='optimal')
    
    Y18 = np.zeros([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 552; Tree Level  2; Scaling  7/ 3 Result_size  3/ 3
    Y18 += np.einsum('ablm,clmjki->abcjki',T2,X18, optimize='optimal')
    
    #Contraction 553; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * np.einsum('abcjki->abcijk',Y18)
    
    #Contraction 554; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.5 * np.einsum('cbajki->abcijk',Y18)
    
    #Contraction 555; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * np.einsum('cabjki->abcijk',Y18)
    
    #Contraction 556; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.5 * np.einsum('abcikj->abcijk',Y18)
    
    #Contraction 557; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * np.einsum('cbaikj->abcijk',Y18)
    
    #Contraction 558; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.5 * np.einsum('cabikj->abcijk',Y18)
    
    #Contraction 559; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * np.einsum('abcijk->abcijk',Y18)
    
    #Contraction 560; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.5 * np.einsum('cbaijk->abcijk',Y18)
    
    #Contraction 561; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * np.einsum('cabijk->abcijk',Y18)
    
    del Y18
    
    A19 = np.zeros([nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 562; Tree Level  3; Scaling  6/ 2 Result_size  4/ 2
    A19 += np.einsum('bm,clmjki->bcljki',T1,X18, optimize='optimal')
    
    D19 = np.zeros([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 563; Tree Level  2; Scaling  5/ 3 Result_size  3/ 3
    D19 += np.einsum('al,bcljki->abcjki',T1,A19, optimize='optimal')
    
    del A19
    
    #Contraction 564; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * np.einsum('abcjki->abcijk',D19)
    
    #Contraction 565; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.5 * np.einsum('acbjki->abcijk',D19)
    
    #Contraction 566; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.5 * np.einsum('bacjki->abcijk',D19)
    
    #Contraction 567; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * np.einsum('bcajki->abcijk',D19)
    
    #Contraction 568; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * np.einsum('cabjki->abcijk',D19)
    
    #Contraction 569; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.5 * np.einsum('cbajki->abcijk',D19)
    
    #Contraction 570; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.5 * np.einsum('abcikj->abcijk',D19)
    
    #Contraction 571; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * np.einsum('acbikj->abcijk',D19)
    
    #Contraction 572; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * np.einsum('bacikj->abcijk',D19)
    
    #Contraction 573; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.5 * np.einsum('bcaikj->abcijk',D19)
    
    #Contraction 574; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.5 * np.einsum('cabikj->abcijk',D19)
    
    #Contraction 575; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * np.einsum('cbaikj->abcijk',D19)
    
    #Contraction 576; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * np.einsum('abcijk->abcijk',D19)
    
    #Contraction 577; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.5 * np.einsum('acbijk->abcijk',D19)
    
    #Contraction 578; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.5 * np.einsum('bacijk->abcijk',D19)
    
    #Contraction 579; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * np.einsum('bcaijk->abcijk',D19)
    
    #Contraction 580; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * np.einsum('cabijk->abcijk',D19)
    
    #Contraction 581; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.5 * np.einsum('cbaijk->abcijk',D19)
    
    del D19
    
    E19 = np.zeros([nvir, nocc, nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 582; Tree Level  3; Scaling  4/ 4 Result_size  4/ 2
    E19 += np.einsum('dbij,lcdk->blcijk',T2,G2, optimize='optimal')
    
    G19 = np.zeros([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 583; Tree Level  2; Scaling  5/ 3 Result_size  3/ 3
    G19 += np.einsum('al,blcijk->abcijk',T1,E19, optimize='optimal')
    
    del E19
    
    #Contraction 584; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * np.einsum('abcijk->abcijk',G19)
    
    #Contraction 585; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += np.einsum('acbijk->abcijk',G19)
    
    #Contraction 586; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += np.einsum('bacijk->abcijk',G19)
    
    #Contraction 587; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * np.einsum('bcaijk->abcijk',G19)
    
    #Contraction 588; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * np.einsum('cabijk->abcijk',G19)
    
    #Contraction 589; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += np.einsum('cbaijk->abcijk',G19)
    
    #Contraction 590; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += np.einsum('abcikj->abcijk',G19)
    
    #Contraction 591; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * np.einsum('acbikj->abcijk',G19)
    
    #Contraction 592; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * np.einsum('bacikj->abcijk',G19)
    
    #Contraction 593; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += np.einsum('bcaikj->abcijk',G19)
    
    #Contraction 594; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += np.einsum('cabikj->abcijk',G19)
    
    #Contraction 595; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * np.einsum('cbaikj->abcijk',G19)
    
    #Contraction 596; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * np.einsum('abcjki->abcijk',G19)
    
    #Contraction 597; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += np.einsum('acbjki->abcijk',G19)
    
    #Contraction 598; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += np.einsum('bacjki->abcijk',G19)
    
    #Contraction 599; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * np.einsum('bcajki->abcijk',G19)
    
    #Contraction 600; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * np.einsum('cabjki->abcijk',G19)
    
    #Contraction 601; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += np.einsum('cbajki->abcijk',G19)
    
    del G19
    
    I19 = np.zeros([nvir, nvir, nvir, nocc], dtype=type_)
    
    #Contraction 602; Tree Level  3; Scaling  5/ 5 Result_size  1/ 3
    I19 += np.einsum('lmde,ebclmk->bcdk',V7,T3, optimize='optimal')
    
    J19 = np.zeros([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 603; Tree Level  2; Scaling  3/ 5 Result_size  3/ 3
    J19 += np.einsum('adij,bcdk->abcijk',T2,I19, optimize='optimal')
    
    #Contraction 604; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * np.einsum('abcijk->abcijk',J19)
    
    #Contraction 605; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.5 * np.einsum('bacijk->abcijk',J19)
    
    #Contraction 606; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * np.einsum('cabijk->abcijk',J19)
    
    #Contraction 607; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.5 * np.einsum('abcikj->abcijk',J19)
    
    #Contraction 608; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * np.einsum('bacikj->abcijk',J19)
    
    #Contraction 609; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.5 * np.einsum('cabikj->abcijk',J19)
    
    #Contraction 610; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * np.einsum('abcjki->abcijk',J19)
    
    #Contraction 611; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.5 * np.einsum('bacjki->abcijk',J19)
    
    #Contraction 612; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * np.einsum('cabjki->abcijk',J19)
    
    del J19
    
    M19 = np.zeros([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 613; Tree Level  2; Scaling  3/ 5 Result_size  3/ 3
    M19 += np.einsum('ebcijk,ae->bcaijk',T3,A16, optimize='optimal')
    
    #Contraction 614; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * np.einsum('bcaijk->abcijk',M19)
    
    #Contraction 615; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.5 * np.einsum('acbijk->abcijk',M19)
    
    #Contraction 616; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * np.einsum('abcijk->abcijk',M19)
    
    del M19
    
    X19 = np.zeros([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 617; Tree Level  2; Scaling  3/ 5 Result_size  3/ 3
    X19 += np.einsum('bdjk,acid->bacjki',T2,M18, optimize='optimal')
    
    #Contraction 618; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * np.einsum('bacjki->abcijk',X19)
    
    #Contraction 619; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.5 * np.einsum('abcjki->abcijk',X19)
    
    #Contraction 620; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * np.einsum('cbajki->abcijk',X19)
    
    #Contraction 621; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.5 * np.einsum('bacikj->abcijk',X19)
    
    #Contraction 622; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * np.einsum('abcikj->abcijk',X19)
    
    #Contraction 623; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.5 * np.einsum('cbaikj->abcijk',X19)
    
    #Contraction 624; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * np.einsum('bacijk->abcijk',X19)
    
    #Contraction 625; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.5 * np.einsum('abcijk->abcijk',X19)
    
    #Contraction 626; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * np.einsum('cbaijk->abcijk',X19)
    
    del X19
    
    Y19 = np.zeros([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 627; Tree Level  2; Scaling  7/ 3 Result_size  3/ 3
    Y19 += np.einsum('abclmk,lmij->abckij',T3,I5, optimize='optimal')
    
    #Contraction 628; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.25 * np.einsum('abckij->abcijk',Y19)
    
    #Contraction 629; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.25 * np.einsum('abcjik->abcijk',Y19)
    
    #Contraction 630; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.25 * np.einsum('abcijk->abcijk',Y19)
    
    del Y19
    
    A20 = np.zeros([nvir, nvir, nvir, nocc], dtype=type_)
    
    #Contraction 631; Tree Level  3; Scaling  1/ 5 Result_size  1/ 3
    A20 += np.einsum('abde,di->abei',V9,T1, optimize='optimal')
    
    D20 = np.zeros([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 632; Tree Level  2; Scaling  3/ 5 Result_size  3/ 3
    D20 += np.einsum('ecjk,abei->cabjki',T2,A20, optimize='optimal')
    
    #Contraction 633; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += np.einsum('cabjki->abcijk',D20)
    
    #Contraction 634; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * np.einsum('bacjki->abcijk',D20)
    
    #Contraction 635; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += np.einsum('abcjki->abcijk',D20)
    
    #Contraction 636; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * np.einsum('cabikj->abcijk',D20)
    
    #Contraction 637; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += np.einsum('bacikj->abcijk',D20)
    
    #Contraction 638; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * np.einsum('abcikj->abcijk',D20)
    
    #Contraction 639; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += np.einsum('cabijk->abcijk',D20)
    
    #Contraction 640; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * np.einsum('bacijk->abcijk',D20)
    
    #Contraction 641; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += np.einsum('abcijk->abcijk',D20)
    
    del D20
    
    #Contraction 642; Tree Level  1; Scaling  5/ 5 Result_size  3/ 3
    Z3 += np.einsum('ld,abcdijkl->abcijk',F3,T4, optimize='optimal')
    
    #Contraction 643; Tree Level  1; Scaling  5/ 5 Result_size  3/ 3
    Z3 += np.einsum('eabcmijk,me->abcijk',T4,A1, optimize='optimal')
    
    I20 = np.zeros([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 644; Tree Level  2; Scaling  7/ 3 Result_size  3/ 3
    I20 += np.einsum('ablm,lmcijk->abcijk',T2,G13, optimize='optimal')
    
    #Contraction 645; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.25 * np.einsum('abcijk->abcijk',I20)
    
    #Contraction 646; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.25 * np.einsum('acbijk->abcijk',I20)
    
    #Contraction 647; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.25 * np.einsum('bcaijk->abcijk',I20)
    
    del I20
    
    J20 = np.zeros([nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 648; Tree Level  3; Scaling  4/ 4 Result_size  4/ 2
    J20 += np.einsum('ecjk,alei->caljki',T2,G5, optimize='optimal')
    
    M20 = np.zeros([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 649; Tree Level  2; Scaling  5/ 3 Result_size  3/ 3
    M20 += np.einsum('bl,caljki->bcajki',T1,J20, optimize='optimal')
    
    #Contraction 650; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * np.einsum('bcajki->abcijk',M20)
    
    #Contraction 651; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += np.einsum('cbajki->abcijk',M20)
    
    #Contraction 652; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += np.einsum('acbjki->abcijk',M20)
    
    #Contraction 653; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * np.einsum('cabjki->abcijk',M20)
    
    #Contraction 654; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * np.einsum('abcjki->abcijk',M20)
    
    #Contraction 655; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += np.einsum('bacjki->abcijk',M20)
    
    #Contraction 656; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += np.einsum('bcaikj->abcijk',M20)
    
    #Contraction 657; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * np.einsum('cbaikj->abcijk',M20)
    
    #Contraction 658; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * np.einsum('acbikj->abcijk',M20)
    
    #Contraction 659; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += np.einsum('cabikj->abcijk',M20)
    
    #Contraction 660; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += np.einsum('abcikj->abcijk',M20)
    
    #Contraction 661; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * np.einsum('bacikj->abcijk',M20)
    
    #Contraction 662; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * np.einsum('bcaijk->abcijk',M20)
    
    #Contraction 663; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += np.einsum('cbaijk->abcijk',M20)
    
    #Contraction 664; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += np.einsum('acbijk->abcijk',M20)
    
    #Contraction 665; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * np.einsum('cabijk->abcijk',M20)
    
    #Contraction 666; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * np.einsum('abcijk->abcijk',M20)
    
    #Contraction 667; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += np.einsum('bacijk->abcijk',M20)
    
    del M20
    
    X20 = np.zeros([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 668; Tree Level  2; Scaling  3/ 7 Result_size  3/ 3
    X20 += np.einsum('abde,cdekij->abckij',V9,T3, optimize='optimal')
    
    #Contraction 669; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * np.einsum('abckij->abcijk',X20)
    
    #Contraction 670; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.5 * np.einsum('acbkij->abcijk',X20)
    
    #Contraction 671; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * np.einsum('bcakij->abcijk',X20)
    
    del X20
    
    Y20 = np.zeros([nocc, nvir, nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 672; Tree Level  3; Scaling  4/ 4 Result_size  4/ 2
    Y20 += np.einsum('ld,dbcijk->lbcijk',F3,T3, optimize='optimal')
    
    A21 = np.zeros([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 673; Tree Level  2; Scaling  5/ 3 Result_size  3/ 3
    A21 += np.einsum('al,lbcijk->abcijk',T1,Y20, optimize='optimal')
    
    #Contraction 674; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * np.einsum('abcijk->abcijk',A21)
    
    #Contraction 675; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += np.einsum('bacijk->abcijk',A21)
    
    #Contraction 676; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * np.einsum('cabijk->abcijk',A21)
    
    del A21
    
    D21 = np.zeros([nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 677; Tree Level  3; Scaling  3/ 3 Result_size  3/ 1
    D21 += np.einsum('ebij,me->bmij',T2,A1, optimize='optimal')
    
    E21 = np.zeros([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 678; Tree Level  2; Scaling  5/ 3 Result_size  3/ 3
    E21 += np.einsum('acmk,bmij->acbkij',T2,D21, optimize='optimal')
    
    #Contraction 679; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * np.einsum('acbkij->abcijk',E21)
    
    #Contraction 680; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += np.einsum('bcakij->abcijk',E21)
    
    #Contraction 681; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * np.einsum('backij->abcijk',E21)
    
    #Contraction 682; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += np.einsum('acbjik->abcijk',E21)
    
    #Contraction 683; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * np.einsum('bcajik->abcijk',E21)
    
    #Contraction 684; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += np.einsum('bacjik->abcijk',E21)
    
    #Contraction 685; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * np.einsum('acbijk->abcijk',E21)
    
    #Contraction 686; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += np.einsum('bcaijk->abcijk',E21)
    
    #Contraction 687; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * np.einsum('bacijk->abcijk',E21)
    
    del E21
    
    G21 = np.zeros([nvir, nvir, nvir, nocc, nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 688; Tree Level  3; Scaling  5/ 5 Result_size  5/ 3
    G21 += np.einsum('fcdjkl,amfi->cdamjkli',T3,G5, optimize='optimal')
    
    I21 = np.zeros([nocc, nvir, nvir, nvir, nocc, nocc], dtype=type_)
    
    #Contraction 689; Tree Level  4; Scaling  5/ 5 Result_size  3/ 3
    I21 += np.einsum('mnef,fcdnkl->mcdekl',V7,T3, optimize='optimal')
    
    J21 = np.zeros([nvir, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 690; Tree Level  2; Scaling  4/ 6 Result_size  4/ 4
    J21 += np.einsum('abie,cdeklj->abcdiklj',V6,T3, optimize='optimal')
    
    #del V6
    
    #Contraction 691; Tree Level  0; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('abcdiklj->abcdijkl',J21)
    
    #Contraction 692; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('acbdiklj->abcdijkl',J21)
    
    #Contraction 693; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('adbciklj->abcdijkl',J21)
    
    #Contraction 694; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bcadiklj->abcdijkl',J21)
    
    #Contraction 695; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bdaciklj->abcdijkl',J21)
    
    #Contraction 696; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('cdabiklj->abcdijkl',J21)
    
    #Contraction 697; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('abcdkilj->abcdijkl',J21)
    
    #Contraction 698; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('acbdkilj->abcdijkl',J21)
    
    #Contraction 699; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('adbckilj->abcdijkl',J21)
    
    #Contraction 700; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bcadkilj->abcdijkl',J21)
    
    #Contraction 701; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bdackilj->abcdijkl',J21)
    
    #Contraction 702; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('cdabkilj->abcdijkl',J21)
    
    #Contraction 703; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('abcdlikj->abcdijkl',J21)
    
    #Contraction 704; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('acbdlikj->abcdijkl',J21)
    
    #Contraction 705; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('adbclikj->abcdijkl',J21)
    
    #Contraction 706; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bcadlikj->abcdijkl',J21)
    
    #Contraction 707; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bdaclikj->abcdijkl',J21)
    
    #Contraction 708; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('cdablikj->abcdijkl',J21)
    
    #Contraction 709; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('abcdjikl->abcdijkl',J21)
    
    #Contraction 710; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('acbdjikl->abcdijkl',J21)
    
    #Contraction 711; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('adbcjikl->abcdijkl',J21)
    
    #Contraction 712; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bcadjikl->abcdijkl',J21)
    
    #Contraction 713; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bdacjikl->abcdijkl',J21)
    
    #Contraction 714; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('cdabjikl->abcdijkl',J21)
    
    del J21
    
    M21 = np.zeros([nvir, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 715; Tree Level  2; Scaling  6/ 4 Result_size  4/ 4
    M21 += np.einsum('bcdmkl,maij->bcdaklij',T3,I18, optimize='optimal')
    
    del I18
    
    #Contraction 716; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bcdaklij->abcdijkl',M21)
    
    #Contraction 717; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('acdbklij->abcdijkl',M21)
    
    #Contraction 718; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('abdcklij->abcdijkl',M21)
    
    #Contraction 719; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('abcdklij->abcdijkl',M21)
    
    #Contraction 720; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bcdajlik->abcdijkl',M21)
    
    #Contraction 721; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('acdbjlik->abcdijkl',M21)
    
    #Contraction 722; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('abdcjlik->abcdijkl',M21)
    
    #Contraction 723; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('abcdjlik->abcdijkl',M21)
    
    #Contraction 724; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bcdajkil->abcdijkl',M21)
    
    #Contraction 725; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('acdbjkil->abcdijkl',M21)
    
    #Contraction 726; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('abdcjkil->abcdijkl',M21)
    
    #Contraction 727; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('abcdjkil->abcdijkl',M21)
    
    #Contraction 728; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bcdailjk->abcdijkl',M21)
    
    #Contraction 729; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('acdbiljk->abcdijkl',M21)
    
    #Contraction 730; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('abdciljk->abcdijkl',M21)
    
    #Contraction 731; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('abcdiljk->abcdijkl',M21)
    
    #Contraction 732; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bcdaikjl->abcdijkl',M21)
    
    #Contraction 733; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('acdbikjl->abcdijkl',M21)
    
    #Contraction 734; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('abdcikjl->abcdijkl',M21)
    
    #Contraction 735; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('abcdikjl->abcdijkl',M21)
    
    #Contraction 736; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bcdaijkl->abcdijkl',M21)
    
    #Contraction 737; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('acdbijkl->abcdijkl',M21)
    
    #Contraction 738; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('abdcijkl->abcdijkl',M21)
    
    #Contraction 739; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('abcdijkl->abcdijkl',M21)
    
    del M21
    
    X21 = np.zeros([nocc, nvir, nvir, nocc], dtype=type_)
    
    #Contraction 740; Tree Level  4; Scaling  4/ 4 Result_size  2/ 2
    X21 += np.einsum('mnef,ebnj->mbfj',V7,T2, optimize='optimal')
    
    Y21 = np.zeros([nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 741; Tree Level  3; Scaling  5/ 3 Result_size  3/ 1
    Y21 += np.einsum('fbmj,mnfi->bnji',T2,I1, optimize='optimal')
    
    A22 = np.zeros([nvir, nocc, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 742; Tree Level  3; Scaling  5/ 5 Result_size  5/ 3
    A22 += np.einsum('ebij,mcdekl->bmcdijkl',T2,I21, optimize='optimal')
    
    D22 = np.zeros([nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 743; Tree Level  3; Scaling  4/ 4 Result_size  4/ 2
    D22 += np.einsum('fcdjkl,nf->cdnjkl',T3,A1, optimize='optimal')
    
    del A1
    
    E22 = np.zeros([nvir, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 744; Tree Level  2; Scaling  8/ 4 Result_size  4/ 4
    E22 += np.einsum('abdmnl,cmnjki->abdcljki',T3,X18, optimize='optimal')
    
    #Contraction 745; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('abdcljki->abcdijkl',E22)
    
    #Contraction 746; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('cbdaljki->abcdijkl',E22)
    
    #Contraction 747; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('cadbljki->abcdijkl',E22)
    
    #Contraction 748; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('cabdljki->abcdijkl',E22)
    
    #Contraction 749; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('abdckjli->abcdijkl',E22)
    
    #Contraction 750; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('cbdakjli->abcdijkl',E22)
    
    #Contraction 751; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('cadbkjli->abcdijkl',E22)
    
    #Contraction 752; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('cabdkjli->abcdijkl',E22)
    
    #Contraction 753; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('abdcjkli->abcdijkl',E22)
    
    #Contraction 754; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('cbdajkli->abcdijkl',E22)
    
    #Contraction 755; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('cadbjkli->abcdijkl',E22)
    
    #Contraction 756; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('cabdjkli->abcdijkl',E22)
    
    #Contraction 757; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('abdclikj->abcdijkl',E22)
    
    #Contraction 758; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('cbdalikj->abcdijkl',E22)
    
    #Contraction 759; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('cadblikj->abcdijkl',E22)
    
    #Contraction 760; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('cabdlikj->abcdijkl',E22)
    
    #Contraction 761; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('abdckilj->abcdijkl',E22)
    
    #Contraction 762; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('cbdakilj->abcdijkl',E22)
    
    #Contraction 763; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('cadbkilj->abcdijkl',E22)
    
    #Contraction 764; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('cabdkilj->abcdijkl',E22)
    
    #Contraction 765; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('abdciklj->abcdijkl',E22)
    
    #Contraction 766; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('cbdaiklj->abcdijkl',E22)
    
    #Contraction 767; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('cadbiklj->abcdijkl',E22)
    
    #Contraction 768; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('cabdiklj->abcdijkl',E22)
    
    #Contraction 769; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('abdclijk->abcdijkl',E22)
    
    #Contraction 770; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('cbdalijk->abcdijkl',E22)
    
    #Contraction 771; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('cadblijk->abcdijkl',E22)
    
    #Contraction 772; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('cabdlijk->abcdijkl',E22)
    
    #Contraction 773; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('abdcjilk->abcdijkl',E22)
    
    #Contraction 774; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('cbdajilk->abcdijkl',E22)
    
    #Contraction 775; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('cadbjilk->abcdijkl',E22)
    
    #Contraction 776; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('cabdjilk->abcdijkl',E22)
    
    #Contraction 777; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('abdcijlk->abcdijkl',E22)
    
    #Contraction 778; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('cbdaijlk->abcdijkl',E22)
    
    #Contraction 779; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('cadbijlk->abcdijkl',E22)
    
    #Contraction 780; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('cabdijlk->abcdijkl',E22)
    
    #Contraction 781; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('abdckijl->abcdijkl',E22)
    
    #Contraction 782; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('cbdakijl->abcdijkl',E22)
    
    #Contraction 783; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('cadbkijl->abcdijkl',E22)
    
    #Contraction 784; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('cabdkijl->abcdijkl',E22)
    
    #Contraction 785; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('abdcjikl->abcdijkl',E22)
    
    #Contraction 786; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('cbdajikl->abcdijkl',E22)
    
    #Contraction 787; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('cadbjikl->abcdijkl',E22)
    
    #Contraction 788; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('cabdjikl->abcdijkl',E22)
    
    #Contraction 789; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('abdcijkl->abcdijkl',E22)
    
    #Contraction 790; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('cbdaijkl->abcdijkl',E22)
    
    #Contraction 791; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('cadbijkl->abcdijkl',E22)
    
    #Contraction 792; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('cabdijkl->abcdijkl',E22)
    
    del E22
    
    G22 = np.zeros([nvir, nocc, nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 793; Tree Level  3; Scaling  6/ 2 Result_size  4/ 2
    G22 += np.einsum('am,mncijk->ancijk',T1,M12, optimize='optimal')
    
    del M12
    
    I22 = np.zeros([nvir, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 794; Tree Level  2; Scaling  6/ 4 Result_size  4/ 4
    I22 += np.einsum('bajm,mcdikl->bacdjikl',T2,X12, optimize='optimal')
    
    del X12
    
    #Contraction 795; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bacdjikl->abcdijkl',I22)
    
    #Contraction 796; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bcadjikl->abcdijkl',I22)
    
    #Contraction 797; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bdacjikl->abcdijkl',I22)
    
    #Contraction 798; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('acbdjikl->abcdijkl',I22)
    
    #Contraction 799; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('adbcjikl->abcdijkl',I22)
    
    #Contraction 800; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('cdbajikl->abcdijkl',I22)
    
    #Contraction 801; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bacdkijl->abcdijkl',I22)
    
    #Contraction 802; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bcadkijl->abcdijkl',I22)
    
    #Contraction 803; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bdackijl->abcdijkl',I22)
    
    #Contraction 804; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('acbdkijl->abcdijkl',I22)
    
    #Contraction 805; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('adbckijl->abcdijkl',I22)
    
    #Contraction 806; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('cdbakijl->abcdijkl',I22)
    
    #Contraction 807; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bacdlijk->abcdijkl',I22)
    
    #Contraction 808; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bcadlijk->abcdijkl',I22)
    
    #Contraction 809; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bdaclijk->abcdijkl',I22)
    
    #Contraction 810; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('acbdlijk->abcdijkl',I22)
    
    #Contraction 811; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('adbclijk->abcdijkl',I22)
    
    #Contraction 812; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('cdbalijk->abcdijkl',I22)
    
    #Contraction 813; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bacdijkl->abcdijkl',I22)
    
    #Contraction 814; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bcadijkl->abcdijkl',I22)
    
    #Contraction 815; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bdacijkl->abcdijkl',I22)
    
    #Contraction 816; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('acbdijkl->abcdijkl',I22)
    
    #Contraction 817; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('adbcijkl->abcdijkl',I22)
    
    #Contraction 818; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('cdbaijkl->abcdijkl',I22)
    
    #Contraction 819; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bacdkjil->abcdijkl',I22)
    
    #Contraction 820; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bcadkjil->abcdijkl',I22)
    
    #Contraction 821; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bdackjil->abcdijkl',I22)
    
    #Contraction 822; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('acbdkjil->abcdijkl',I22)
    
    #Contraction 823; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('adbckjil->abcdijkl',I22)
    
    #Contraction 824; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('cdbakjil->abcdijkl',I22)
    
    #Contraction 825; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bacdljik->abcdijkl',I22)
    
    #Contraction 826; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bcadljik->abcdijkl',I22)
    
    #Contraction 827; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bdacljik->abcdijkl',I22)
    
    #Contraction 828; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('acbdljik->abcdijkl',I22)
    
    #Contraction 829; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('adbcljik->abcdijkl',I22)
    
    #Contraction 830; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('cdbaljik->abcdijkl',I22)
    
    #Contraction 831; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bacdikjl->abcdijkl',I22)
    
    #Contraction 832; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bcadikjl->abcdijkl',I22)
    
    #Contraction 833; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bdacikjl->abcdijkl',I22)
    
    #Contraction 834; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('acbdikjl->abcdijkl',I22)
    
    #Contraction 835; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('adbcikjl->abcdijkl',I22)
    
    #Contraction 836; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('cdbaikjl->abcdijkl',I22)
    
    #Contraction 837; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bacdjkil->abcdijkl',I22)
    
    #Contraction 838; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bcadjkil->abcdijkl',I22)
    
    #Contraction 839; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bdacjkil->abcdijkl',I22)
    
    #Contraction 840; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('acbdjkil->abcdijkl',I22)
    
    #Contraction 841; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('adbcjkil->abcdijkl',I22)
    
    #Contraction 842; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('cdbajkil->abcdijkl',I22)
    
    #Contraction 843; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bacdlkij->abcdijkl',I22)
    
    #Contraction 844; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bcadlkij->abcdijkl',I22)
    
    #Contraction 845; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bdaclkij->abcdijkl',I22)
    
    #Contraction 846; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('acbdlkij->abcdijkl',I22)
    
    #Contraction 847; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('adbclkij->abcdijkl',I22)
    
    #Contraction 848; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('cdbalkij->abcdijkl',I22)
    
    #Contraction 849; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bacdiljk->abcdijkl',I22)
    
    #Contraction 850; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bcadiljk->abcdijkl',I22)
    
    #Contraction 851; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bdaciljk->abcdijkl',I22)
    
    #Contraction 852; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('acbdiljk->abcdijkl',I22)
    
    #Contraction 853; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('adbciljk->abcdijkl',I22)
    
    #Contraction 854; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('cdbailjk->abcdijkl',I22)
    
    #Contraction 855; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bacdjlik->abcdijkl',I22)
    
    #Contraction 856; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bcadjlik->abcdijkl',I22)
    
    #Contraction 857; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bdacjlik->abcdijkl',I22)
    
    #Contraction 858; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('acbdjlik->abcdijkl',I22)
    
    #Contraction 859; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('adbcjlik->abcdijkl',I22)
    
    #Contraction 860; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('cdbajlik->abcdijkl',I22)
    
    #Contraction 861; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bacdklij->abcdijkl',I22)
    
    #Contraction 862; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bcadklij->abcdijkl',I22)
    
    #Contraction 863; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bdacklij->abcdijkl',I22)
    
    #Contraction 864; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('acbdklij->abcdijkl',I22)
    
    #Contraction 865; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('adbcklij->abcdijkl',I22)
    
    #Contraction 866; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('cdbaklij->abcdijkl',I22)
    
    del I22
    
    J22 = np.zeros([nocc, nocc, nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 867; Tree Level  3; Scaling  5/ 3 Result_size  5/ 1
    J22 += np.einsum('mnie,bejk->mnbijk',V4,T2, optimize='optimal')
    
    M22 = np.zeros([nvir, nocc, nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 868; Tree Level  3; Scaling  4/ 4 Result_size  4/ 2
    M22 += np.einsum('aeij,mdel->amdijl',T2,G2, optimize='optimal')
    
    del G2
    
    X22 = np.zeros([nvir, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 869; Tree Level  2; Scaling  6/ 4 Result_size  4/ 4
    X22 += np.einsum('bcdnkl,anij->bcdaklij',T3,A13, optimize='optimal')
    
    del A13
    
    #Contraction 870; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bcdaklij->abcdijkl',X22)
    
    #Contraction 871; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('acdbklij->abcdijkl',X22)
    
    #Contraction 872; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('abdcklij->abcdijkl',X22)
    
    #Contraction 873; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('abcdklij->abcdijkl',X22)
    
    #Contraction 874; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bcdajlik->abcdijkl',X22)
    
    #Contraction 875; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('acdbjlik->abcdijkl',X22)
    
    #Contraction 876; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('abdcjlik->abcdijkl',X22)
    
    #Contraction 877; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('abcdjlik->abcdijkl',X22)
    
    #Contraction 878; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bcdajkil->abcdijkl',X22)
    
    #Contraction 879; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('acdbjkil->abcdijkl',X22)
    
    #Contraction 880; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('abdcjkil->abcdijkl',X22)
    
    #Contraction 881; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('abcdjkil->abcdijkl',X22)
    
    #Contraction 882; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bcdaklji->abcdijkl',X22)
    
    #Contraction 883; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('acdbklji->abcdijkl',X22)
    
    #Contraction 884; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('abdcklji->abcdijkl',X22)
    
    #Contraction 885; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('abcdklji->abcdijkl',X22)
    
    #Contraction 886; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bcdailjk->abcdijkl',X22)
    
    #Contraction 887; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('acdbiljk->abcdijkl',X22)
    
    #Contraction 888; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('abdciljk->abcdijkl',X22)
    
    #Contraction 889; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('abcdiljk->abcdijkl',X22)
    
    #Contraction 890; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bcdaikjl->abcdijkl',X22)
    
    #Contraction 891; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('acdbikjl->abcdijkl',X22)
    
    #Contraction 892; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('abdcikjl->abcdijkl',X22)
    
    #Contraction 893; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('abcdikjl->abcdijkl',X22)
    
    #Contraction 894; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bcdajlki->abcdijkl',X22)
    
    #Contraction 895; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('acdbjlki->abcdijkl',X22)
    
    #Contraction 896; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('abdcjlki->abcdijkl',X22)
    
    #Contraction 897; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('abcdjlki->abcdijkl',X22)
    
    #Contraction 898; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bcdailkj->abcdijkl',X22)
    
    #Contraction 899; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('acdbilkj->abcdijkl',X22)
    
    #Contraction 900; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('abdcilkj->abcdijkl',X22)
    
    #Contraction 901; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('abcdilkj->abcdijkl',X22)
    
    #Contraction 902; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bcdaijkl->abcdijkl',X22)
    
    #Contraction 903; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('acdbijkl->abcdijkl',X22)
    
    #Contraction 904; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('abdcijkl->abcdijkl',X22)
    
    #Contraction 905; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('abcdijkl->abcdijkl',X22)
    
    #Contraction 906; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bcdajkli->abcdijkl',X22)
    
    #Contraction 907; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('acdbjkli->abcdijkl',X22)
    
    #Contraction 908; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('abdcjkli->abcdijkl',X22)
    
    #Contraction 909; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('abcdjkli->abcdijkl',X22)
    
    #Contraction 910; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bcdaiklj->abcdijkl',X22)
    
    #Contraction 911; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('acdbiklj->abcdijkl',X22)
    
    #Contraction 912; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('abdciklj->abcdijkl',X22)
    
    #Contraction 913; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('abcdiklj->abcdijkl',X22)
    
    #Contraction 914; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bcdaijlk->abcdijkl',X22)
    
    #Contraction 915; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('acdbijlk->abcdijkl',X22)
    
    #Contraction 916; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('abdcijlk->abcdijkl',X22)
    
    #Contraction 917; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('abcdijlk->abcdijkl',X22)
    
    del X22
    
    Y22 = np.zeros([nvir, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 918; Tree Level  2; Scaling  6/ 4 Result_size  4/ 4
    Y22 += np.einsum('bcdnkl,anij->bcdaklij',T3,Y8, optimize='optimal')
    
    del Y8
    
    #Contraction 919; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('bcdaklij->abcdijkl',Y22)
    
    #Contraction 920; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('acdbklij->abcdijkl',Y22)
    
    #Contraction 921; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('abdcklij->abcdijkl',Y22)
    
    #Contraction 922; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('abcdklij->abcdijkl',Y22)
    
    #Contraction 923; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('bcdajlik->abcdijkl',Y22)
    
    #Contraction 924; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('acdbjlik->abcdijkl',Y22)
    
    #Contraction 925; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('abdcjlik->abcdijkl',Y22)
    
    #Contraction 926; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('abcdjlik->abcdijkl',Y22)
    
    #Contraction 927; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('bcdajkil->abcdijkl',Y22)
    
    #Contraction 928; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('acdbjkil->abcdijkl',Y22)
    
    #Contraction 929; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('abdcjkil->abcdijkl',Y22)
    
    #Contraction 930; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('abcdjkil->abcdijkl',Y22)
    
    #Contraction 931; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('bcdailjk->abcdijkl',Y22)
    
    #Contraction 932; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('acdbiljk->abcdijkl',Y22)
    
    #Contraction 933; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('abdciljk->abcdijkl',Y22)
    
    #Contraction 934; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('abcdiljk->abcdijkl',Y22)
    
    #Contraction 935; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('bcdaikjl->abcdijkl',Y22)
    
    #Contraction 936; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('acdbikjl->abcdijkl',Y22)
    
    #Contraction 937; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('abdcikjl->abcdijkl',Y22)
    
    #Contraction 938; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('abcdikjl->abcdijkl',Y22)
    
    #Contraction 939; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('bcdaijkl->abcdijkl',Y22)
    
    #Contraction 940; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('acdbijkl->abcdijkl',Y22)
    
    #Contraction 941; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('abdcijkl->abcdijkl',Y22)
    
    #Contraction 942; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('abcdijkl->abcdijkl',Y22)
    
    del Y22
    
    A23 = np.zeros([nvir, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 943; Tree Level  2; Scaling  4/ 6 Result_size  4/ 4
    A23 += np.einsum('ae,bcdejkli->abcdjkli',F4,T4, optimize='optimal')
    
    #del F4
    
    #Contraction 944; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('abcdjkli->abcdijkl',A23)
    
    #Contraction 945; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bacdjkli->abcdijkl',A23)
    
    #Contraction 946; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('cabdjkli->abcdijkl',A23)
    
    #Contraction 947; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('dabcjkli->abcdijkl',A23)
    
    del A23
    
    D23 = np.zeros([nvir, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 948; Tree Level  2; Scaling  8/ 4 Result_size  4/ 4
    D23 += np.einsum('abcdmnkl,mnij->abcdklij',T4,I5, optimize='optimal')
    
    #Contraction 949; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('abcdklij->abcdijkl',D23)
    
    #Contraction 950; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('abcdjlik->abcdijkl',D23)
    
    #Contraction 951; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('abcdjkil->abcdijkl',D23)
    
    #Contraction 952; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('abcdiljk->abcdijkl',D23)
    
    #Contraction 953; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('abcdikjl->abcdijkl',D23)
    
    #Contraction 954; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('abcdijkl->abcdijkl',D23)
    
    del D23
    
    E23 = np.zeros([nvir, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 955; Tree Level  2; Scaling  6/ 4 Result_size  4/ 4
    E23 += np.einsum('acdnkl,bnij->acdbklij',T3,D21, optimize='optimal')
    
    del D21
    
    #Contraction 956; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('acdbklij->abcdijkl',E23)
    
    #Contraction 957; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bcdaklij->abcdijkl',E23)
    
    #Contraction 958; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('badcklij->abcdijkl',E23)
    
    #Contraction 959; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bacdklij->abcdijkl',E23)
    
    #Contraction 960; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('acdbjlik->abcdijkl',E23)
    
    #Contraction 961; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bcdajlik->abcdijkl',E23)
    
    #Contraction 962; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('badcjlik->abcdijkl',E23)
    
    #Contraction 963; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bacdjlik->abcdijkl',E23)
    
    #Contraction 964; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('acdbjkil->abcdijkl',E23)
    
    #Contraction 965; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bcdajkil->abcdijkl',E23)
    
    #Contraction 966; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('badcjkil->abcdijkl',E23)
    
    #Contraction 967; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bacdjkil->abcdijkl',E23)
    
    #Contraction 968; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('acdbiljk->abcdijkl',E23)
    
    #Contraction 969; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bcdailjk->abcdijkl',E23)
    
    #Contraction 970; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('badciljk->abcdijkl',E23)
    
    #Contraction 971; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bacdiljk->abcdijkl',E23)
    
    #Contraction 972; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('acdbikjl->abcdijkl',E23)
    
    #Contraction 973; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bcdaikjl->abcdijkl',E23)
    
    #Contraction 974; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('badcikjl->abcdijkl',E23)
    
    #Contraction 975; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bacdikjl->abcdijkl',E23)
    
    #Contraction 976; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('acdbijkl->abcdijkl',E23)
    
    #Contraction 977; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bcdaijkl->abcdijkl',E23)
    
    #Contraction 978; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('badcijkl->abcdijkl',E23)
    
    #Contraction 979; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bacdijkl->abcdijkl',E23)
    
    del E23
    
    G23 = np.zeros([nvir, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 980; Tree Level  2; Scaling  6/ 4 Result_size  4/ 4
    G23 += np.einsum('abim,mcdjkl->abcdijkl',T2,Y20, optimize='optimal')
    
    del Y20
    
    #Contraction 981; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('abcdijkl->abcdijkl',G23)
    
    #Contraction 982; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('acbdijkl->abcdijkl',G23)
    
    #Contraction 983; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('adbcijkl->abcdijkl',G23)
    
    #Contraction 984; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bcadijkl->abcdijkl',G23)
    
    #Contraction 985; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bdacijkl->abcdijkl',G23)
    
    #Contraction 986; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('cdabijkl->abcdijkl',G23)
    
    #Contraction 987; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('abcdjikl->abcdijkl',G23)
    
    #Contraction 988; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('acbdjikl->abcdijkl',G23)
    
    #Contraction 989; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('adbcjikl->abcdijkl',G23)
    
    #Contraction 990; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bcadjikl->abcdijkl',G23)
    
    #Contraction 991; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bdacjikl->abcdijkl',G23)
    
    #Contraction 992; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('cdabjikl->abcdijkl',G23)
    
    #Contraction 993; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('abcdkijl->abcdijkl',G23)
    
    #Contraction 994; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('acbdkijl->abcdijkl',G23)
    
    #Contraction 995; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('adbckijl->abcdijkl',G23)
    
    #Contraction 996; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bcadkijl->abcdijkl',G23)
    
    #Contraction 997; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bdackijl->abcdijkl',G23)
    
    #Contraction 998; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('cdabkijl->abcdijkl',G23)
    
    #Contraction 999; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('abcdlijk->abcdijkl',G23)
    
    #Contraction 1000; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('acbdlijk->abcdijkl',G23)
    
    #Contraction 1001; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('adbclijk->abcdijkl',G23)
    
    #Contraction 1002; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bcadlijk->abcdijkl',G23)
    
    #Contraction 1003; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bdaclijk->abcdijkl',G23)
    
    #Contraction 1004; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('cdablijk->abcdijkl',G23)
    
    del G23
    
    I23 = np.zeros([nvir, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 1005; Tree Level  2; Scaling  8/ 4 Result_size  4/ 4
    I23 += np.einsum('abcdmnkl,mnij->abcdklij',T4,Y6, optimize='optimal')
    
    #Contraction 1006; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('abcdklij->abcdijkl',I23)
    
    #Contraction 1007; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('abcdjlik->abcdijkl',I23)
    
    #Contraction 1008; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('abcdjkil->abcdijkl',I23)
    
    #Contraction 1009; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('abcdklji->abcdijkl',I23)
    
    #Contraction 1010; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('abcdiljk->abcdijkl',I23)
    
    #Contraction 1011; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('abcdikjl->abcdijkl',I23)
    
    #Contraction 1012; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('abcdjlki->abcdijkl',I23)
    
    #Contraction 1013; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('abcdilkj->abcdijkl',I23)
    
    #Contraction 1014; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('abcdijkl->abcdijkl',I23)
    
    #Contraction 1015; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('abcdjkli->abcdijkl',I23)
    
    #Contraction 1016; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('abcdiklj->abcdijkl',I23)
    
    #Contraction 1017; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('abcdijlk->abcdijkl',I23)
    
    del I23
    
    J23 = np.zeros([nvir, nvir, nvir, nvir, nocc, nocc], dtype=type_)
    
    #Contraction 1018; Tree Level  3; Scaling  4/ 6 Result_size  2/ 4
    J23 += np.einsum('amef,fcdmkl->acdekl',V8,T3, optimize='optimal')
    
    M23 = np.zeros([nvir, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 1019; Tree Level  2; Scaling  4/ 6 Result_size  4/ 4
    M23 += np.einsum('fbcdijkl,af->bcdaijkl',T4,I10, optimize='optimal')
    
    del I10
    
    #Contraction 1020; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bcdaijkl->abcdijkl',M23)
    
    #Contraction 1021; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('acdbijkl->abcdijkl',M23)
    
    #Contraction 1022; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('abdcijkl->abcdijkl',M23)
    
    #Contraction 1023; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('abcdijkl->abcdijkl',M23)
    
    del M23
    
    X23 = np.zeros([nvir, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 1024; Tree Level  2; Scaling  6/ 4 Result_size  4/ 4
    X23 += np.einsum('bani,cdnjkl->bacdijkl',T2,D22, optimize='optimal')
    
    del D22
    
    #Contraction 1025; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bacdijkl->abcdijkl',X23)
    
    #Contraction 1026; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bcadijkl->abcdijkl',X23)
    
    #Contraction 1027; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bdacijkl->abcdijkl',X23)
    
    #Contraction 1028; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('acbdijkl->abcdijkl',X23)
    
    #Contraction 1029; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('adbcijkl->abcdijkl',X23)
    
    #Contraction 1030; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('cdbaijkl->abcdijkl',X23)
    
    #Contraction 1031; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bacdjikl->abcdijkl',X23)
    
    #Contraction 1032; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bcadjikl->abcdijkl',X23)
    
    #Contraction 1033; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bdacjikl->abcdijkl',X23)
    
    #Contraction 1034; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('acbdjikl->abcdijkl',X23)
    
    #Contraction 1035; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('adbcjikl->abcdijkl',X23)
    
    #Contraction 1036; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('cdbajikl->abcdijkl',X23)
    
    #Contraction 1037; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bacdkijl->abcdijkl',X23)
    
    #Contraction 1038; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bcadkijl->abcdijkl',X23)
    
    #Contraction 1039; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bdackijl->abcdijkl',X23)
    
    #Contraction 1040; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('acbdkijl->abcdijkl',X23)
    
    #Contraction 1041; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('adbckijl->abcdijkl',X23)
    
    #Contraction 1042; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('cdbakijl->abcdijkl',X23)
    
    #Contraction 1043; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bacdlijk->abcdijkl',X23)
    
    #Contraction 1044; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bcadlijk->abcdijkl',X23)
    
    #Contraction 1045; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bdaclijk->abcdijkl',X23)
    
    #Contraction 1046; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('acbdlijk->abcdijkl',X23)
    
    #Contraction 1047; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('adbclijk->abcdijkl',X23)
    
    #Contraction 1048; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('cdbalijk->abcdijkl',X23)
    
    del X23
    
    Y23 = np.zeros([nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 1049; Tree Level  3; Scaling  6/ 2 Result_size  4/ 2
    Y23 += np.einsum('bdnl,mnij->bdmlij',T2,Y6, optimize='optimal')
    
    del Y6
    
    A24 = np.zeros([nvir, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 1050; Tree Level  2; Scaling  6/ 4 Result_size  4/ 4
    A24 += np.einsum('bcdnkl,naij->bcdaklij',T3,E13, optimize='optimal')
    
    del E13
    
    #Contraction 1051; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bcdaklij->abcdijkl',A24)
    
    #Contraction 1052; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('acdbklij->abcdijkl',A24)
    
    #Contraction 1053; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('abdcklij->abcdijkl',A24)
    
    #Contraction 1054; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('abcdklij->abcdijkl',A24)
    
    #Contraction 1055; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bcdajlik->abcdijkl',A24)
    
    #Contraction 1056; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('acdbjlik->abcdijkl',A24)
    
    #Contraction 1057; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('abdcjlik->abcdijkl',A24)
    
    #Contraction 1058; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('abcdjlik->abcdijkl',A24)
    
    #Contraction 1059; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bcdajkil->abcdijkl',A24)
    
    #Contraction 1060; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('acdbjkil->abcdijkl',A24)
    
    #Contraction 1061; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('abdcjkil->abcdijkl',A24)
    
    #Contraction 1062; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('abcdjkil->abcdijkl',A24)
    
    #Contraction 1063; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bcdailjk->abcdijkl',A24)
    
    #Contraction 1064; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('acdbiljk->abcdijkl',A24)
    
    #Contraction 1065; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('abdciljk->abcdijkl',A24)
    
    #Contraction 1066; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('abcdiljk->abcdijkl',A24)
    
    #Contraction 1067; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bcdaikjl->abcdijkl',A24)
    
    #Contraction 1068; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('acdbikjl->abcdijkl',A24)
    
    #Contraction 1069; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('abdcikjl->abcdijkl',A24)
    
    #Contraction 1070; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('abcdikjl->abcdijkl',A24)
    
    #Contraction 1071; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bcdaijkl->abcdijkl',A24)
    
    #Contraction 1072; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('acdbijkl->abcdijkl',A24)
    
    #Contraction 1073; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('abdcijkl->abcdijkl',A24)
    
    #Contraction 1074; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('abcdijkl->abcdijkl',A24)
    
    del A24
    
    D24 = np.zeros([nvir, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 1075; Tree Level  2; Scaling  6/ 4 Result_size  4/ 4
    D24 += np.einsum('bcdmkl,amij->bcdaklij',T3,I9, optimize='optimal')
    
    del I9
    
    #Contraction 1076; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('bcdaklij->abcdijkl',D24)
    
    #Contraction 1077; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('acdbklij->abcdijkl',D24)
    
    #Contraction 1078; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('abdcklij->abcdijkl',D24)
    
    #Contraction 1079; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('abcdklij->abcdijkl',D24)
    
    #Contraction 1080; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('bcdajlik->abcdijkl',D24)
    
    #Contraction 1081; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('acdbjlik->abcdijkl',D24)
    
    #Contraction 1082; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('abdcjlik->abcdijkl',D24)
    
    #Contraction 1083; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('abcdjlik->abcdijkl',D24)
    
    #Contraction 1084; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('bcdajkil->abcdijkl',D24)
    
    #Contraction 1085; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('acdbjkil->abcdijkl',D24)
    
    #Contraction 1086; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('abdcjkil->abcdijkl',D24)
    
    #Contraction 1087; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('abcdjkil->abcdijkl',D24)
    
    #Contraction 1088; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('bcdaklji->abcdijkl',D24)
    
    #Contraction 1089; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('acdbklji->abcdijkl',D24)
    
    #Contraction 1090; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('abdcklji->abcdijkl',D24)
    
    #Contraction 1091; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('abcdklji->abcdijkl',D24)
    
    #Contraction 1092; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('bcdailjk->abcdijkl',D24)
    
    #Contraction 1093; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('acdbiljk->abcdijkl',D24)
    
    #Contraction 1094; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('abdciljk->abcdijkl',D24)
    
    #Contraction 1095; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('abcdiljk->abcdijkl',D24)
    
    #Contraction 1096; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('bcdaikjl->abcdijkl',D24)
    
    #Contraction 1097; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('acdbikjl->abcdijkl',D24)
    
    #Contraction 1098; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('abdcikjl->abcdijkl',D24)
    
    #Contraction 1099; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('abcdikjl->abcdijkl',D24)
    
    #Contraction 1100; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('bcdajlki->abcdijkl',D24)
    
    #Contraction 1101; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('acdbjlki->abcdijkl',D24)
    
    #Contraction 1102; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('abdcjlki->abcdijkl',D24)
    
    #Contraction 1103; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('abcdjlki->abcdijkl',D24)
    
    #Contraction 1104; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('bcdailkj->abcdijkl',D24)
    
    #Contraction 1105; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('acdbilkj->abcdijkl',D24)
    
    #Contraction 1106; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('abdcilkj->abcdijkl',D24)
    
    #Contraction 1107; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('abcdilkj->abcdijkl',D24)
    
    #Contraction 1108; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('bcdaijkl->abcdijkl',D24)
    
    #Contraction 1109; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('acdbijkl->abcdijkl',D24)
    
    #Contraction 1110; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('abdcijkl->abcdijkl',D24)
    
    #Contraction 1111; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('abcdijkl->abcdijkl',D24)
    
    #Contraction 1112; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('bcdajkli->abcdijkl',D24)
    
    #Contraction 1113; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('acdbjkli->abcdijkl',D24)
    
    #Contraction 1114; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('abdcjkli->abcdijkl',D24)
    
    #Contraction 1115; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('abcdjkli->abcdijkl',D24)
    
    #Contraction 1116; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('bcdaiklj->abcdijkl',D24)
    
    #Contraction 1117; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('acdbiklj->abcdijkl',D24)
    
    #Contraction 1118; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('abdciklj->abcdijkl',D24)
    
    #Contraction 1119; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('abcdiklj->abcdijkl',D24)
    
    #Contraction 1120; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('bcdaijlk->abcdijkl',D24)
    
    #Contraction 1121; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('acdbijlk->abcdijkl',D24)
    
    #Contraction 1122; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('abdcijlk->abcdijkl',D24)
    
    #Contraction 1123; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('abcdijlk->abcdijkl',D24)
    
    del D24
    
    E24 = np.zeros([nvir, nocc, nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 1124; Tree Level  3; Scaling  4/ 4 Result_size  4/ 2
    E24 += np.einsum('amie,bejk->ambijk',V5,T2, optimize='optimal')
    
    G24 = np.zeros([nvir, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 1125; Tree Level  2; Scaling  6/ 6 Result_size  4/ 4
    G24 += np.einsum('amie,bcdejklm->abcdijkl',V5,T4, optimize='optimal')
    
    #Contraction 1126; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('abcdijkl->abcdijkl',G24)
    
    #Contraction 1127; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bacdijkl->abcdijkl',G24)
    
    #Contraction 1128; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('cabdijkl->abcdijkl',G24)
    
    #Contraction 1129; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('dabcijkl->abcdijkl',G24)
    
    #Contraction 1130; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('abcdjikl->abcdijkl',G24)
    
    #Contraction 1131; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bacdjikl->abcdijkl',G24)
    
    #Contraction 1132; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('cabdjikl->abcdijkl',G24)
    
    #Contraction 1133; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('dabcjikl->abcdijkl',G24)
    
    #Contraction 1134; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('abcdkijl->abcdijkl',G24)
    
    #Contraction 1135; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bacdkijl->abcdijkl',G24)
    
    #Contraction 1136; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('cabdkijl->abcdijkl',G24)
    
    #Contraction 1137; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('dabckijl->abcdijkl',G24)
    
    #Contraction 1138; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('abcdlijk->abcdijkl',G24)
    
    #Contraction 1139; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bacdlijk->abcdijkl',G24)
    
    #Contraction 1140; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('cabdlijk->abcdijkl',G24)
    
    #Contraction 1141; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('dabclijk->abcdijkl',G24)
    
    del G24
    
    I24 = np.zeros([nvir, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 1142; Tree Level  2; Scaling  6/ 6 Result_size  4/ 4
    I24 += np.einsum('fbcdmjkl,amfi->bcdajkli',T4,G5, optimize='optimal')
    
    del G5
    
    #Contraction 1143; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bcdajkli->abcdijkl',I24)
    
    #Contraction 1144; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('acdbjkli->abcdijkl',I24)
    
    #Contraction 1145; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('abdcjkli->abcdijkl',I24)
    
    #Contraction 1146; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('abcdjkli->abcdijkl',I24)
    
    #Contraction 1147; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bcdaiklj->abcdijkl',I24)
    
    #Contraction 1148; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('acdbiklj->abcdijkl',I24)
    
    #Contraction 1149; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('abdciklj->abcdijkl',I24)
    
    #Contraction 1150; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('abcdiklj->abcdijkl',I24)
    
    #Contraction 1151; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bcdaijlk->abcdijkl',I24)
    
    #Contraction 1152; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('acdbijlk->abcdijkl',I24)
    
    #Contraction 1153; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('abdcijlk->abcdijkl',I24)
    
    #Contraction 1154; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('abcdijlk->abcdijkl',I24)
    
    #Contraction 1155; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bcdaijkl->abcdijkl',I24)
    
    #Contraction 1156; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('acdbijkl->abcdijkl',I24)
    
    #Contraction 1157; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('abdcijkl->abcdijkl',I24)
    
    #Contraction 1158; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('abcdijkl->abcdijkl',I24)
    
    del I24
    
    J24 = np.zeros([nocc, nocc, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 1159; Tree Level  4; Scaling  6/ 4 Result_size  6/ 2
    J24 += np.einsum('mnie,ecdjkl->mncdijkl',V4,T3, optimize='optimal')
    
    M24 = np.zeros([nvir, nocc, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 1160; Tree Level  3; Scaling  7/ 3 Result_size  5/ 3
    M24 += np.einsum('bn,mncdijkl->bmcdijkl',T1,J24, optimize='optimal')
    
    del J24
    
    X24 = np.zeros([nvir, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 1161; Tree Level  2; Scaling  6/ 4 Result_size  4/ 4
    X24 += np.einsum('am,bmcdijkl->abcdijkl',T1,M24, optimize='optimal')
    
    del M24
    
    #Contraction 1162; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('abcdijkl->abcdijkl',X24)
    
    #Contraction 1163; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('acbdijkl->abcdijkl',X24)
    
    #Contraction 1164; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('adbcijkl->abcdijkl',X24)
    
    #Contraction 1165; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('bacdijkl->abcdijkl',X24)
    
    #Contraction 1166; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('bcadijkl->abcdijkl',X24)
    
    #Contraction 1167; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('bdacijkl->abcdijkl',X24)
    
    #Contraction 1168; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('cabdijkl->abcdijkl',X24)
    
    #Contraction 1169; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('cbadijkl->abcdijkl',X24)
    
    #Contraction 1170; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('cdabijkl->abcdijkl',X24)
    
    #Contraction 1171; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('dabcijkl->abcdijkl',X24)
    
    #Contraction 1172; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('dbacijkl->abcdijkl',X24)
    
    #Contraction 1173; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('dcabijkl->abcdijkl',X24)
    
    #Contraction 1174; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('abcdjikl->abcdijkl',X24)
    
    #Contraction 1175; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('acbdjikl->abcdijkl',X24)
    
    #Contraction 1176; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('adbcjikl->abcdijkl',X24)
    
    #Contraction 1177; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('bacdjikl->abcdijkl',X24)
    
    #Contraction 1178; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('bcadjikl->abcdijkl',X24)
    
    #Contraction 1179; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('bdacjikl->abcdijkl',X24)
    
    #Contraction 1180; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('cabdjikl->abcdijkl',X24)
    
    #Contraction 1181; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('cbadjikl->abcdijkl',X24)
    
    #Contraction 1182; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('cdabjikl->abcdijkl',X24)
    
    #Contraction 1183; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('dabcjikl->abcdijkl',X24)
    
    #Contraction 1184; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('dbacjikl->abcdijkl',X24)
    
    #Contraction 1185; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('dcabjikl->abcdijkl',X24)
    
    #Contraction 1186; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('abcdkijl->abcdijkl',X24)
    
    #Contraction 1187; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('acbdkijl->abcdijkl',X24)
    
    #Contraction 1188; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('adbckijl->abcdijkl',X24)
    
    #Contraction 1189; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('bacdkijl->abcdijkl',X24)
    
    #Contraction 1190; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('bcadkijl->abcdijkl',X24)
    
    #Contraction 1191; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('bdackijl->abcdijkl',X24)
    
    #Contraction 1192; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('cabdkijl->abcdijkl',X24)
    
    #Contraction 1193; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('cbadkijl->abcdijkl',X24)
    
    #Contraction 1194; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('cdabkijl->abcdijkl',X24)
    
    #Contraction 1195; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('dabckijl->abcdijkl',X24)
    
    #Contraction 1196; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('dbackijl->abcdijkl',X24)
    
    #Contraction 1197; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('dcabkijl->abcdijkl',X24)
    
    #Contraction 1198; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('abcdlijk->abcdijkl',X24)
    
    #Contraction 1199; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('acbdlijk->abcdijkl',X24)
    
    #Contraction 1200; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('adbclijk->abcdijkl',X24)
    
    #Contraction 1201; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('bacdlijk->abcdijkl',X24)
    
    #Contraction 1202; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('bcadlijk->abcdijkl',X24)
    
    #Contraction 1203; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('bdaclijk->abcdijkl',X24)
    
    #Contraction 1204; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('cabdlijk->abcdijkl',X24)
    
    #Contraction 1205; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('cbadlijk->abcdijkl',X24)
    
    #Contraction 1206; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('cdablijk->abcdijkl',X24)
    
    #Contraction 1207; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('dabclijk->abcdijkl',X24)
    
    #Contraction 1208; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('dbaclijk->abcdijkl',X24)
    
    #Contraction 1209; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('dcablijk->abcdijkl',X24)
    
    del X24
    
    Y24 = np.zeros([nvir, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 1210; Tree Level  2; Scaling  4/ 6 Result_size  4/ 4
    Y24 += np.einsum('ecdjkl,abie->cdabjkli',T3,M18, optimize='optimal')
    
    del M18
    
    #Contraction 1211; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('cdabjkli->abcdijkl',Y24)
    
    #Contraction 1212; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('bdacjkli->abcdijkl',Y24)
    
    #Contraction 1213; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('bcadjkli->abcdijkl',Y24)
    
    #Contraction 1214; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('adbcjkli->abcdijkl',Y24)
    
    #Contraction 1215; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('acbdjkli->abcdijkl',Y24)
    
    #Contraction 1216; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('abcdjkli->abcdijkl',Y24)
    
    #Contraction 1217; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('cdabiklj->abcdijkl',Y24)
    
    #Contraction 1218; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('bdaciklj->abcdijkl',Y24)
    
    #Contraction 1219; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('bcadiklj->abcdijkl',Y24)
    
    #Contraction 1220; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('adbciklj->abcdijkl',Y24)
    
    #Contraction 1221; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('acbdiklj->abcdijkl',Y24)
    
    #Contraction 1222; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('abcdiklj->abcdijkl',Y24)
    
    #Contraction 1223; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('cdabijlk->abcdijkl',Y24)
    
    #Contraction 1224; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('bdacijlk->abcdijkl',Y24)
    
    #Contraction 1225; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('bcadijlk->abcdijkl',Y24)
    
    #Contraction 1226; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('adbcijlk->abcdijkl',Y24)
    
    #Contraction 1227; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('acbdijlk->abcdijkl',Y24)
    
    #Contraction 1228; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('abcdijlk->abcdijkl',Y24)
    
    #Contraction 1229; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('cdabijkl->abcdijkl',Y24)
    
    #Contraction 1230; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('bdacijkl->abcdijkl',Y24)
    
    #Contraction 1231; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('bcadijkl->abcdijkl',Y24)
    
    #Contraction 1232; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('adbcijkl->abcdijkl',Y24)
    
    #Contraction 1233; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('acbdijkl->abcdijkl',Y24)
    
    #Contraction 1234; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('abcdijkl->abcdijkl',Y24)
    
    del Y24
    
    A25 = np.zeros([nvir, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 1235; Tree Level  2; Scaling  8/ 4 Result_size  4/ 4
    A25 += np.einsum('abcdmnkl,mnji->abcdklji',T4,E7, optimize='optimal')
    
    #Contraction 1236; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('abcdklji->abcdijkl',A25)
    
    #Contraction 1237; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('abcdjlki->abcdijkl',A25)
    
    #Contraction 1238; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('abcdjkli->abcdijkl',A25)
    
    #Contraction 1239; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('abcdklij->abcdijkl',A25)
    
    #Contraction 1240; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('abcdilkj->abcdijkl',A25)
    
    #Contraction 1241; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('abcdiklj->abcdijkl',A25)
    
    #Contraction 1242; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('abcdjlik->abcdijkl',A25)
    
    #Contraction 1243; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('abcdiljk->abcdijkl',A25)
    
    #Contraction 1244; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('abcdijlk->abcdijkl',A25)
    
    #Contraction 1245; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('abcdjkil->abcdijkl',A25)
    
    #Contraction 1246; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('abcdikjl->abcdijkl',A25)
    
    #Contraction 1247; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('abcdijkl->abcdijkl',A25)
    
    del A25
    
    D25 = np.zeros([nvir, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 1248; Tree Level  2; Scaling  6/ 4 Result_size  4/ 4
    D25 += np.einsum('am,bmcdijkl->abcdijkl',T1,A22, optimize='optimal')
    
    del A22
    
    #Contraction 1249; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('abcdijkl->abcdijkl',D25)
    
    #Contraction 1250; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('acbdijkl->abcdijkl',D25)
    
    #Contraction 1251; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('adbcijkl->abcdijkl',D25)
    
    #Contraction 1252; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bacdijkl->abcdijkl',D25)
    
    #Contraction 1253; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bcadijkl->abcdijkl',D25)
    
    #Contraction 1254; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bdacijkl->abcdijkl',D25)
    
    #Contraction 1255; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('cabdijkl->abcdijkl',D25)
    
    #Contraction 1256; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('cbadijkl->abcdijkl',D25)
    
    #Contraction 1257; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('cdabijkl->abcdijkl',D25)
    
    #Contraction 1258; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('dabcijkl->abcdijkl',D25)
    
    #Contraction 1259; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('dbacijkl->abcdijkl',D25)
    
    #Contraction 1260; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('dcabijkl->abcdijkl',D25)
    
    #Contraction 1261; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('abcdikjl->abcdijkl',D25)
    
    #Contraction 1262; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('acbdikjl->abcdijkl',D25)
    
    #Contraction 1263; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('adbcikjl->abcdijkl',D25)
    
    #Contraction 1264; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bacdikjl->abcdijkl',D25)
    
    #Contraction 1265; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bcadikjl->abcdijkl',D25)
    
    #Contraction 1266; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bdacikjl->abcdijkl',D25)
    
    #Contraction 1267; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('cabdikjl->abcdijkl',D25)
    
    #Contraction 1268; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('cbadikjl->abcdijkl',D25)
    
    #Contraction 1269; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('cdabikjl->abcdijkl',D25)
    
    #Contraction 1270; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('dabcikjl->abcdijkl',D25)
    
    #Contraction 1271; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('dbacikjl->abcdijkl',D25)
    
    #Contraction 1272; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('dcabikjl->abcdijkl',D25)
    
    #Contraction 1273; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('abcdiljk->abcdijkl',D25)
    
    #Contraction 1274; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('acbdiljk->abcdijkl',D25)
    
    #Contraction 1275; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('adbciljk->abcdijkl',D25)
    
    #Contraction 1276; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bacdiljk->abcdijkl',D25)
    
    #Contraction 1277; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bcadiljk->abcdijkl',D25)
    
    #Contraction 1278; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bdaciljk->abcdijkl',D25)
    
    #Contraction 1279; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('cabdiljk->abcdijkl',D25)
    
    #Contraction 1280; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('cbadiljk->abcdijkl',D25)
    
    #Contraction 1281; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('cdabiljk->abcdijkl',D25)
    
    #Contraction 1282; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('dabciljk->abcdijkl',D25)
    
    #Contraction 1283; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('dbaciljk->abcdijkl',D25)
    
    #Contraction 1284; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('dcabiljk->abcdijkl',D25)
    
    #Contraction 1285; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('abcdjkil->abcdijkl',D25)
    
    #Contraction 1286; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('acbdjkil->abcdijkl',D25)
    
    #Contraction 1287; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('adbcjkil->abcdijkl',D25)
    
    #Contraction 1288; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bacdjkil->abcdijkl',D25)
    
    #Contraction 1289; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bcadjkil->abcdijkl',D25)
    
    #Contraction 1290; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bdacjkil->abcdijkl',D25)
    
    #Contraction 1291; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('cabdjkil->abcdijkl',D25)
    
    #Contraction 1292; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('cbadjkil->abcdijkl',D25)
    
    #Contraction 1293; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('cdabjkil->abcdijkl',D25)
    
    #Contraction 1294; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('dabcjkil->abcdijkl',D25)
    
    #Contraction 1295; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('dbacjkil->abcdijkl',D25)
    
    #Contraction 1296; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('dcabjkil->abcdijkl',D25)
    
    #Contraction 1297; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('abcdjlik->abcdijkl',D25)
    
    #Contraction 1298; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('acbdjlik->abcdijkl',D25)
    
    #Contraction 1299; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('adbcjlik->abcdijkl',D25)
    
    #Contraction 1300; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bacdjlik->abcdijkl',D25)
    
    #Contraction 1301; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bcadjlik->abcdijkl',D25)
    
    #Contraction 1302; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bdacjlik->abcdijkl',D25)
    
    #Contraction 1303; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('cabdjlik->abcdijkl',D25)
    
    #Contraction 1304; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('cbadjlik->abcdijkl',D25)
    
    #Contraction 1305; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('cdabjlik->abcdijkl',D25)
    
    #Contraction 1306; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('dabcjlik->abcdijkl',D25)
    
    #Contraction 1307; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('dbacjlik->abcdijkl',D25)
    
    #Contraction 1308; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('dcabjlik->abcdijkl',D25)
    
    #Contraction 1309; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('abcdklij->abcdijkl',D25)
    
    #Contraction 1310; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('acbdklij->abcdijkl',D25)
    
    #Contraction 1311; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('adbcklij->abcdijkl',D25)
    
    #Contraction 1312; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bacdklij->abcdijkl',D25)
    
    #Contraction 1313; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bcadklij->abcdijkl',D25)
    
    #Contraction 1314; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bdacklij->abcdijkl',D25)
    
    #Contraction 1315; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('cabdklij->abcdijkl',D25)
    
    #Contraction 1316; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('cbadklij->abcdijkl',D25)
    
    #Contraction 1317; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('cdabklij->abcdijkl',D25)
    
    #Contraction 1318; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('dabcklij->abcdijkl',D25)
    
    #Contraction 1319; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('dbacklij->abcdijkl',D25)
    
    #Contraction 1320; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('dcabklij->abcdijkl',D25)
    
    del D25
    
    E25 = np.zeros([nvir, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 1321; Tree Level  2; Scaling  6/ 4 Result_size  4/ 4
    E25 += np.einsum('abim,mcdjkl->abcdijkl',T2,E14, optimize='optimal')
    
    del E14
    
    #Contraction 1322; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('abcdijkl->abcdijkl',E25)
    
    #Contraction 1323; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('acbdijkl->abcdijkl',E25)
    
    #Contraction 1324; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('adbcijkl->abcdijkl',E25)
    
    #Contraction 1325; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('bcadijkl->abcdijkl',E25)
    
    #Contraction 1326; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('bdacijkl->abcdijkl',E25)
    
    #Contraction 1327; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('cdabijkl->abcdijkl',E25)
    
    #Contraction 1328; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('abcdjikl->abcdijkl',E25)
    
    #Contraction 1329; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('acbdjikl->abcdijkl',E25)
    
    #Contraction 1330; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('adbcjikl->abcdijkl',E25)
    
    #Contraction 1331; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('bcadjikl->abcdijkl',E25)
    
    #Contraction 1332; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('bdacjikl->abcdijkl',E25)
    
    #Contraction 1333; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('cdabjikl->abcdijkl',E25)
    
    #Contraction 1334; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('abcdkijl->abcdijkl',E25)
    
    #Contraction 1335; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('acbdkijl->abcdijkl',E25)
    
    #Contraction 1336; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('adbckijl->abcdijkl',E25)
    
    #Contraction 1337; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('bcadkijl->abcdijkl',E25)
    
    #Contraction 1338; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('bdackijl->abcdijkl',E25)
    
    #Contraction 1339; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('cdabkijl->abcdijkl',E25)
    
    #Contraction 1340; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('abcdlijk->abcdijkl',E25)
    
    #Contraction 1341; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('acbdlijk->abcdijkl',E25)
    
    #Contraction 1342; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('adbclijk->abcdijkl',E25)
    
    #Contraction 1343; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('bcadlijk->abcdijkl',E25)
    
    #Contraction 1344; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('bdaclijk->abcdijkl',E25)
    
    #Contraction 1345; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('cdablijk->abcdijkl',E25)
    
    del E25
    
    G25 = np.zeros([nvir, nocc, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 1346; Tree Level  3; Scaling  5/ 5 Result_size  5/ 3
    G25 += np.einsum('amie,ecdjkl->amcdijkl',V5,T3, optimize='optimal')
    
    #del V5
    
    I25 = np.zeros([nocc, nvir, nocc, nocc], dtype=type_)
    
    #Contraction 1347; Tree Level  3; Scaling  5/ 3 Result_size  3/ 1
    I25 += np.einsum('mnie,bejm->nbij',V4,T2, optimize='optimal')
    
    J25 = np.zeros([nvir, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 1348; Tree Level  2; Scaling  6/ 4 Result_size  4/ 4
    J25 += np.einsum('acdnkl,nbij->acdbklij',T3,I25, optimize='optimal')
    
    del I25
    
    #Contraction 1349; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('acdbklij->abcdijkl',J25)
    
    #Contraction 1350; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bcdaklij->abcdijkl',J25)
    
    #Contraction 1351; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('badcklij->abcdijkl',J25)
    
    #Contraction 1352; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bacdklij->abcdijkl',J25)
    
    #Contraction 1353; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('acdbjlik->abcdijkl',J25)
    
    #Contraction 1354; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bcdajlik->abcdijkl',J25)
    
    #Contraction 1355; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('badcjlik->abcdijkl',J25)
    
    #Contraction 1356; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bacdjlik->abcdijkl',J25)
    
    #Contraction 1357; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('acdbjkil->abcdijkl',J25)
    
    #Contraction 1358; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bcdajkil->abcdijkl',J25)
    
    #Contraction 1359; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('badcjkil->abcdijkl',J25)
    
    #Contraction 1360; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bacdjkil->abcdijkl',J25)
    
    #Contraction 1361; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('acdbklji->abcdijkl',J25)
    
    #Contraction 1362; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bcdaklji->abcdijkl',J25)
    
    #Contraction 1363; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('badcklji->abcdijkl',J25)
    
    #Contraction 1364; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bacdklji->abcdijkl',J25)
    
    #Contraction 1365; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('acdbiljk->abcdijkl',J25)
    
    #Contraction 1366; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bcdailjk->abcdijkl',J25)
    
    #Contraction 1367; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('badciljk->abcdijkl',J25)
    
    #Contraction 1368; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bacdiljk->abcdijkl',J25)
    
    #Contraction 1369; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('acdbikjl->abcdijkl',J25)
    
    #Contraction 1370; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bcdaikjl->abcdijkl',J25)
    
    #Contraction 1371; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('badcikjl->abcdijkl',J25)
    
    #Contraction 1372; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bacdikjl->abcdijkl',J25)
    
    #Contraction 1373; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('acdbjlki->abcdijkl',J25)
    
    #Contraction 1374; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bcdajlki->abcdijkl',J25)
    
    #Contraction 1375; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('badcjlki->abcdijkl',J25)
    
    #Contraction 1376; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bacdjlki->abcdijkl',J25)
    
    #Contraction 1377; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('acdbilkj->abcdijkl',J25)
    
    #Contraction 1378; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bcdailkj->abcdijkl',J25)
    
    #Contraction 1379; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('badcilkj->abcdijkl',J25)
    
    #Contraction 1380; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bacdilkj->abcdijkl',J25)
    
    #Contraction 1381; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('acdbijkl->abcdijkl',J25)
    
    #Contraction 1382; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bcdaijkl->abcdijkl',J25)
    
    #Contraction 1383; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('badcijkl->abcdijkl',J25)
    
    #Contraction 1384; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bacdijkl->abcdijkl',J25)
    
    #Contraction 1385; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('acdbjkli->abcdijkl',J25)
    
    #Contraction 1386; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bcdajkli->abcdijkl',J25)
    
    #Contraction 1387; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('badcjkli->abcdijkl',J25)
    
    #Contraction 1388; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bacdjkli->abcdijkl',J25)
    
    #Contraction 1389; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('acdbiklj->abcdijkl',J25)
    
    #Contraction 1390; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bcdaiklj->abcdijkl',J25)
    
    #Contraction 1391; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('badciklj->abcdijkl',J25)
    
    #Contraction 1392; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bacdiklj->abcdijkl',J25)
    
    #Contraction 1393; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('acdbijlk->abcdijkl',J25)
    
    #Contraction 1394; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bcdaijlk->abcdijkl',J25)
    
    #Contraction 1395; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('badcijlk->abcdijkl',J25)
    
    #Contraction 1396; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bacdijlk->abcdijkl',J25)
    
    del J25
    
    M25 = np.zeros([nvir, nocc, nvir, nvir, nocc, nocc], dtype=type_)
    
    #Contraction 1397; Tree Level  4; Scaling  3/ 5 Result_size  3/ 3
    M25 += np.einsum('amef,fdjl->amdejl',V8,T2, optimize='optimal')
    
    X25 = np.zeros([nvir, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 1398; Tree Level  2; Scaling  8/ 4 Result_size  4/ 4
    X25 += np.einsum('acdmnl,mnbijk->acdblijk',T3,J22, optimize='optimal')
    
    del J22
    
    #Contraction 1399; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('acdblijk->abcdijkl',X25)
    
    #Contraction 1400; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('bcdalijk->abcdijkl',X25)
    
    #Contraction 1401; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('badclijk->abcdijkl',X25)
    
    #Contraction 1402; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('bacdlijk->abcdijkl',X25)
    
    #Contraction 1403; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('acdbkijl->abcdijkl',X25)
    
    #Contraction 1404; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('bcdakijl->abcdijkl',X25)
    
    #Contraction 1405; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('badckijl->abcdijkl',X25)
    
    #Contraction 1406; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('bacdkijl->abcdijkl',X25)
    
    #Contraction 1407; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('acdbjikl->abcdijkl',X25)
    
    #Contraction 1408; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('bcdajikl->abcdijkl',X25)
    
    #Contraction 1409; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('badcjikl->abcdijkl',X25)
    
    #Contraction 1410; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('bacdjikl->abcdijkl',X25)
    
    #Contraction 1411; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('acdbljik->abcdijkl',X25)
    
    #Contraction 1412; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('bcdaljik->abcdijkl',X25)
    
    #Contraction 1413; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('badcljik->abcdijkl',X25)
    
    #Contraction 1414; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('bacdljik->abcdijkl',X25)
    
    #Contraction 1415; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('acdbkjil->abcdijkl',X25)
    
    #Contraction 1416; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('bcdakjil->abcdijkl',X25)
    
    #Contraction 1417; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('badckjil->abcdijkl',X25)
    
    #Contraction 1418; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('bacdkjil->abcdijkl',X25)
    
    #Contraction 1419; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('acdbijkl->abcdijkl',X25)
    
    #Contraction 1420; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('bcdaijkl->abcdijkl',X25)
    
    #Contraction 1421; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('badcijkl->abcdijkl',X25)
    
    #Contraction 1422; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('bacdijkl->abcdijkl',X25)
    
    #Contraction 1423; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('acdblkij->abcdijkl',X25)
    
    #Contraction 1424; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('bcdalkij->abcdijkl',X25)
    
    #Contraction 1425; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('badclkij->abcdijkl',X25)
    
    #Contraction 1426; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('bacdlkij->abcdijkl',X25)
    
    #Contraction 1427; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('acdbjkil->abcdijkl',X25)
    
    #Contraction 1428; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('bcdajkil->abcdijkl',X25)
    
    #Contraction 1429; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('badcjkil->abcdijkl',X25)
    
    #Contraction 1430; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('bacdjkil->abcdijkl',X25)
    
    #Contraction 1431; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('acdbikjl->abcdijkl',X25)
    
    #Contraction 1432; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('bcdaikjl->abcdijkl',X25)
    
    #Contraction 1433; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('badcikjl->abcdijkl',X25)
    
    #Contraction 1434; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('bacdikjl->abcdijkl',X25)
    
    #Contraction 1435; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('acdbklij->abcdijkl',X25)
    
    #Contraction 1436; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('bcdaklij->abcdijkl',X25)
    
    #Contraction 1437; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('badcklij->abcdijkl',X25)
    
    #Contraction 1438; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('bacdklij->abcdijkl',X25)
    
    #Contraction 1439; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('acdbjlik->abcdijkl',X25)
    
    #Contraction 1440; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('bcdajlik->abcdijkl',X25)
    
    #Contraction 1441; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('badcjlik->abcdijkl',X25)
    
    #Contraction 1442; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('bacdjlik->abcdijkl',X25)
    
    #Contraction 1443; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('acdbiljk->abcdijkl',X25)
    
    #Contraction 1444; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('bcdailjk->abcdijkl',X25)
    
    #Contraction 1445; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('badciljk->abcdijkl',X25)
    
    #Contraction 1446; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('bacdiljk->abcdijkl',X25)
    
    del X25
    
    Y25 = np.zeros([nvir, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 1447; Tree Level  2; Scaling  6/ 4 Result_size  4/ 4
    Y25 += np.einsum('abcijm,mdkl->abcdijkl',T3,M9, optimize='optimal')
    
    del M9
    
    #Contraction 1448; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('abcdijkl->abcdijkl',Y25)
    
    #Contraction 1449; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('abdcijkl->abcdijkl',Y25)
    
    #Contraction 1450; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('acdbijkl->abcdijkl',Y25)
    
    #Contraction 1451; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('bcdaijkl->abcdijkl',Y25)
    
    #Contraction 1452; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('abcdikjl->abcdijkl',Y25)
    
    #Contraction 1453; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('abdcikjl->abcdijkl',Y25)
    
    #Contraction 1454; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('acdbikjl->abcdijkl',Y25)
    
    #Contraction 1455; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('bcdaikjl->abcdijkl',Y25)
    
    #Contraction 1456; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('abcdiljk->abcdijkl',Y25)
    
    #Contraction 1457; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('abdciljk->abcdijkl',Y25)
    
    #Contraction 1458; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('acdbiljk->abcdijkl',Y25)
    
    #Contraction 1459; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('bcdailjk->abcdijkl',Y25)
    
    #Contraction 1460; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('abcdjkil->abcdijkl',Y25)
    
    #Contraction 1461; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('abdcjkil->abcdijkl',Y25)
    
    #Contraction 1462; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('acdbjkil->abcdijkl',Y25)
    
    #Contraction 1463; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('bcdajkil->abcdijkl',Y25)
    
    #Contraction 1464; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('abcdjlik->abcdijkl',Y25)
    
    #Contraction 1465; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('abdcjlik->abcdijkl',Y25)
    
    #Contraction 1466; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('acdbjlik->abcdijkl',Y25)
    
    #Contraction 1467; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('bcdajlik->abcdijkl',Y25)
    
    #Contraction 1468; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('abcdklij->abcdijkl',Y25)
    
    #Contraction 1469; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('abdcklij->abcdijkl',Y25)
    
    #Contraction 1470; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('acdbklij->abcdijkl',Y25)
    
    #Contraction 1471; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('bcdaklij->abcdijkl',Y25)
    
    del Y25
    
    A26 = np.zeros([nvir, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 1472; Tree Level  2; Scaling  6/ 4 Result_size  4/ 4
    A26 += np.einsum('bcdmkl,amij->bcdaklij',T3,J6, optimize='optimal')
    
    del J6
    
    #Contraction 1473; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('bcdaklij->abcdijkl',A26)
    
    #Contraction 1474; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('acdbklij->abcdijkl',A26)
    
    #Contraction 1475; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('abdcklij->abcdijkl',A26)
    
    #Contraction 1476; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('abcdklij->abcdijkl',A26)
    
    #Contraction 1477; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('bcdajlik->abcdijkl',A26)
    
    #Contraction 1478; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('acdbjlik->abcdijkl',A26)
    
    #Contraction 1479; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('abdcjlik->abcdijkl',A26)
    
    #Contraction 1480; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('abcdjlik->abcdijkl',A26)
    
    #Contraction 1481; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('bcdajkil->abcdijkl',A26)
    
    #Contraction 1482; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('acdbjkil->abcdijkl',A26)
    
    #Contraction 1483; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('abdcjkil->abcdijkl',A26)
    
    #Contraction 1484; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('abcdjkil->abcdijkl',A26)
    
    #Contraction 1485; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('bcdailjk->abcdijkl',A26)
    
    #Contraction 1486; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('acdbiljk->abcdijkl',A26)
    
    #Contraction 1487; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('abdciljk->abcdijkl',A26)
    
    #Contraction 1488; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('abcdiljk->abcdijkl',A26)
    
    #Contraction 1489; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('bcdaikjl->abcdijkl',A26)
    
    #Contraction 1490; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('acdbikjl->abcdijkl',A26)
    
    #Contraction 1491; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('abdcikjl->abcdijkl',A26)
    
    #Contraction 1492; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('abcdikjl->abcdijkl',A26)
    
    #Contraction 1493; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('bcdaijkl->abcdijkl',A26)
    
    #Contraction 1494; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('acdbijkl->abcdijkl',A26)
    
    #Contraction 1495; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('abdcijkl->abcdijkl',A26)
    
    #Contraction 1496; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('abcdijkl->abcdijkl',A26)
    
    del A26
    
    D26 = np.zeros([nvir, nvir, nocc, nvir, nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 1497; Tree Level  3; Scaling  5/ 5 Result_size  5/ 3
    D26 += np.einsum('ecik,amdejl->camdikjl',T2,M25, optimize='optimal')
    
    del M25
    
    E26 = np.zeros([nvir, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 1498; Tree Level  2; Scaling  6/ 4 Result_size  4/ 4
    E26 += np.einsum('bm,camdikjl->bcadikjl',T1,D26, optimize='optimal')
    
    del D26
    
    #Contraction 1499; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('bcadikjl->abcdijkl',E26)
    
    #Contraction 1500; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('bdacikjl->abcdijkl',E26)
    
    #Contraction 1501; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('cbadikjl->abcdijkl',E26)
    
    #Contraction 1502; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('cdabikjl->abcdijkl',E26)
    
    #Contraction 1503; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('dbacikjl->abcdijkl',E26)
    
    #Contraction 1504; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('dcabikjl->abcdijkl',E26)
    
    #Contraction 1505; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('acbdikjl->abcdijkl',E26)
    
    #Contraction 1506; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('adbcikjl->abcdijkl',E26)
    
    #Contraction 1507; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('cabdikjl->abcdijkl',E26)
    
    #Contraction 1508; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('cdbaikjl->abcdijkl',E26)
    
    #Contraction 1509; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('dabcikjl->abcdijkl',E26)
    
    #Contraction 1510; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('dcbaikjl->abcdijkl',E26)
    
    #Contraction 1511; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('abcdikjl->abcdijkl',E26)
    
    #Contraction 1512; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('adcbikjl->abcdijkl',E26)
    
    #Contraction 1513; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('bacdikjl->abcdijkl',E26)
    
    #Contraction 1514; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('bdcaikjl->abcdijkl',E26)
    
    #Contraction 1515; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('dacbikjl->abcdijkl',E26)
    
    #Contraction 1516; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('dbcaikjl->abcdijkl',E26)
    
    #Contraction 1517; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('abdcikjl->abcdijkl',E26)
    
    #Contraction 1518; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('acdbikjl->abcdijkl',E26)
    
    #Contraction 1519; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('badcikjl->abcdijkl',E26)
    
    #Contraction 1520; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('bcdaikjl->abcdijkl',E26)
    
    #Contraction 1521; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('cadbikjl->abcdijkl',E26)
    
    #Contraction 1522; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('cbdaikjl->abcdijkl',E26)
    
    #Contraction 1523; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('bcadijkl->abcdijkl',E26)
    
    #Contraction 1524; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('bdacijkl->abcdijkl',E26)
    
    #Contraction 1525; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('cbadijkl->abcdijkl',E26)
    
    #Contraction 1526; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('cdabijkl->abcdijkl',E26)
    
    #Contraction 1527; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('dbacijkl->abcdijkl',E26)
    
    #Contraction 1528; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('dcabijkl->abcdijkl',E26)
    
    #Contraction 1529; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('acbdijkl->abcdijkl',E26)
    
    #Contraction 1530; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('adbcijkl->abcdijkl',E26)
    
    #Contraction 1531; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('cabdijkl->abcdijkl',E26)
    
    #Contraction 1532; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('cdbaijkl->abcdijkl',E26)
    
    #Contraction 1533; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('dabcijkl->abcdijkl',E26)
    
    #Contraction 1534; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('dcbaijkl->abcdijkl',E26)
    
    #Contraction 1535; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('abcdijkl->abcdijkl',E26)
    
    #Contraction 1536; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('adcbijkl->abcdijkl',E26)
    
    #Contraction 1537; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('bacdijkl->abcdijkl',E26)
    
    #Contraction 1538; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('bdcaijkl->abcdijkl',E26)
    
    #Contraction 1539; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('dacbijkl->abcdijkl',E26)
    
    #Contraction 1540; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('dbcaijkl->abcdijkl',E26)
    
    #Contraction 1541; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('abdcijkl->abcdijkl',E26)
    
    #Contraction 1542; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('acdbijkl->abcdijkl',E26)
    
    #Contraction 1543; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('badcijkl->abcdijkl',E26)
    
    #Contraction 1544; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('bcdaijkl->abcdijkl',E26)
    
    #Contraction 1545; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('cadbijkl->abcdijkl',E26)
    
    #Contraction 1546; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('cbdaijkl->abcdijkl',E26)
    
    #Contraction 1547; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('bcadilkj->abcdijkl',E26)
    
    #Contraction 1548; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('bdacilkj->abcdijkl',E26)
    
    #Contraction 1549; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('cbadilkj->abcdijkl',E26)
    
    #Contraction 1550; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('cdabilkj->abcdijkl',E26)
    
    #Contraction 1551; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('dbacilkj->abcdijkl',E26)
    
    #Contraction 1552; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('dcabilkj->abcdijkl',E26)
    
    #Contraction 1553; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('acbdilkj->abcdijkl',E26)
    
    #Contraction 1554; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('adbcilkj->abcdijkl',E26)
    
    #Contraction 1555; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('cabdilkj->abcdijkl',E26)
    
    #Contraction 1556; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('cdbailkj->abcdijkl',E26)
    
    #Contraction 1557; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('dabcilkj->abcdijkl',E26)
    
    #Contraction 1558; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('dcbailkj->abcdijkl',E26)
    
    #Contraction 1559; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('abcdilkj->abcdijkl',E26)
    
    #Contraction 1560; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('adcbilkj->abcdijkl',E26)
    
    #Contraction 1561; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('bacdilkj->abcdijkl',E26)
    
    #Contraction 1562; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('bdcailkj->abcdijkl',E26)
    
    #Contraction 1563; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('dacbilkj->abcdijkl',E26)
    
    #Contraction 1564; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('dbcailkj->abcdijkl',E26)
    
    #Contraction 1565; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('abdcilkj->abcdijkl',E26)
    
    #Contraction 1566; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('acdbilkj->abcdijkl',E26)
    
    #Contraction 1567; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('badcilkj->abcdijkl',E26)
    
    #Contraction 1568; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('bcdailkj->abcdijkl',E26)
    
    #Contraction 1569; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('cadbilkj->abcdijkl',E26)
    
    #Contraction 1570; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('cbdailkj->abcdijkl',E26)
    
    #Contraction 1571; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('bcadkjil->abcdijkl',E26)
    
    #Contraction 1572; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('bdackjil->abcdijkl',E26)
    
    #Contraction 1573; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('cbadkjil->abcdijkl',E26)
    
    #Contraction 1574; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('cdabkjil->abcdijkl',E26)
    
    #Contraction 1575; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('dbackjil->abcdijkl',E26)
    
    #Contraction 1576; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('dcabkjil->abcdijkl',E26)
    
    #Contraction 1577; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('acbdkjil->abcdijkl',E26)
    
    #Contraction 1578; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('adbckjil->abcdijkl',E26)
    
    #Contraction 1579; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('cabdkjil->abcdijkl',E26)
    
    #Contraction 1580; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('cdbakjil->abcdijkl',E26)
    
    #Contraction 1581; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('dabckjil->abcdijkl',E26)
    
    #Contraction 1582; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('dcbakjil->abcdijkl',E26)
    
    #Contraction 1583; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('abcdkjil->abcdijkl',E26)
    
    #Contraction 1584; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('adcbkjil->abcdijkl',E26)
    
    #Contraction 1585; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('bacdkjil->abcdijkl',E26)
    
    #Contraction 1586; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('bdcakjil->abcdijkl',E26)
    
    #Contraction 1587; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('dacbkjil->abcdijkl',E26)
    
    #Contraction 1588; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('dbcakjil->abcdijkl',E26)
    
    #Contraction 1589; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('abdckjil->abcdijkl',E26)
    
    #Contraction 1590; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('acdbkjil->abcdijkl',E26)
    
    #Contraction 1591; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('badckjil->abcdijkl',E26)
    
    #Contraction 1592; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('bcdakjil->abcdijkl',E26)
    
    #Contraction 1593; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('cadbkjil->abcdijkl',E26)
    
    #Contraction 1594; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('cbdakjil->abcdijkl',E26)
    
    #Contraction 1595; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('bcadklij->abcdijkl',E26)
    
    #Contraction 1596; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('bdacklij->abcdijkl',E26)
    
    #Contraction 1597; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('cbadklij->abcdijkl',E26)
    
    #Contraction 1598; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('cdabklij->abcdijkl',E26)
    
    #Contraction 1599; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('dbacklij->abcdijkl',E26)
    
    #Contraction 1600; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('dcabklij->abcdijkl',E26)
    
    #Contraction 1601; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('acbdklij->abcdijkl',E26)
    
    #Contraction 1602; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('adbcklij->abcdijkl',E26)
    
    #Contraction 1603; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('cabdklij->abcdijkl',E26)
    
    #Contraction 1604; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('cdbaklij->abcdijkl',E26)
    
    #Contraction 1605; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('dabcklij->abcdijkl',E26)
    
    #Contraction 1606; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('dcbaklij->abcdijkl',E26)
    
    #Contraction 1607; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('abcdklij->abcdijkl',E26)
    
    #Contraction 1608; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('adcbklij->abcdijkl',E26)
    
    #Contraction 1609; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('bacdklij->abcdijkl',E26)
    
    #Contraction 1610; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('bdcaklij->abcdijkl',E26)
    
    #Contraction 1611; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('dacbklij->abcdijkl',E26)
    
    #Contraction 1612; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('dbcaklij->abcdijkl',E26)
    
    #Contraction 1613; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('abdcklij->abcdijkl',E26)
    
    #Contraction 1614; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('acdbklij->abcdijkl',E26)
    
    #Contraction 1615; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('badcklij->abcdijkl',E26)
    
    #Contraction 1616; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('bcdaklij->abcdijkl',E26)
    
    #Contraction 1617; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('cadbklij->abcdijkl',E26)
    
    #Contraction 1618; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('cbdaklij->abcdijkl',E26)
    
    #Contraction 1619; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('bcadjlik->abcdijkl',E26)
    
    #Contraction 1620; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('bdacjlik->abcdijkl',E26)
    
    #Contraction 1621; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('cbadjlik->abcdijkl',E26)
    
    #Contraction 1622; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('cdabjlik->abcdijkl',E26)
    
    #Contraction 1623; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('dbacjlik->abcdijkl',E26)
    
    #Contraction 1624; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('dcabjlik->abcdijkl',E26)
    
    #Contraction 1625; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('acbdjlik->abcdijkl',E26)
    
    #Contraction 1626; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('adbcjlik->abcdijkl',E26)
    
    #Contraction 1627; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('cabdjlik->abcdijkl',E26)
    
    #Contraction 1628; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('cdbajlik->abcdijkl',E26)
    
    #Contraction 1629; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('dabcjlik->abcdijkl',E26)
    
    #Contraction 1630; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('dcbajlik->abcdijkl',E26)
    
    #Contraction 1631; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('abcdjlik->abcdijkl',E26)
    
    #Contraction 1632; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('adcbjlik->abcdijkl',E26)
    
    #Contraction 1633; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('bacdjlik->abcdijkl',E26)
    
    #Contraction 1634; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('bdcajlik->abcdijkl',E26)
    
    #Contraction 1635; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('dacbjlik->abcdijkl',E26)
    
    #Contraction 1636; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('dbcajlik->abcdijkl',E26)
    
    #Contraction 1637; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('abdcjlik->abcdijkl',E26)
    
    #Contraction 1638; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('acdbjlik->abcdijkl',E26)
    
    #Contraction 1639; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('badcjlik->abcdijkl',E26)
    
    #Contraction 1640; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('bcdajlik->abcdijkl',E26)
    
    #Contraction 1641; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('cadbjlik->abcdijkl',E26)
    
    #Contraction 1642; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('cbdajlik->abcdijkl',E26)
    
    del E26
    
    G26 = np.zeros([nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 1643; Tree Level  3; Scaling  6/ 2 Result_size  4/ 2
    G26 += np.einsum('bdnl,mnij->bdmlij',T2,I5, optimize='optimal')
    
    del I5
    
    I26 = np.zeros([nvir, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 1644; Tree Level  2; Scaling  6/ 4 Result_size  4/ 4
    I26 += np.einsum('acmk,bdmlij->acbdklij',T2,G26, optimize='optimal')
    
    del G26
    
    #Contraction 1645; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('acbdklij->abcdijkl',I26)
    
    #Contraction 1646; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('abcdklij->abcdijkl',I26)
    
    #Contraction 1647; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('adcbklij->abcdijkl',I26)
    
    #Contraction 1648; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('cbadklij->abcdijkl',I26)
    
    #Contraction 1649; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('cdabklij->abcdijkl',I26)
    
    #Contraction 1650; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('bdacklij->abcdijkl',I26)
    
    #Contraction 1651; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('acbdlkij->abcdijkl',I26)
    
    #Contraction 1652; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('abcdlkij->abcdijkl',I26)
    
    #Contraction 1653; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('adcblkij->abcdijkl',I26)
    
    #Contraction 1654; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('cbadlkij->abcdijkl',I26)
    
    #Contraction 1655; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('cdablkij->abcdijkl',I26)
    
    #Contraction 1656; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('bdaclkij->abcdijkl',I26)
    
    #Contraction 1657; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('acbdjlik->abcdijkl',I26)
    
    #Contraction 1658; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('abcdjlik->abcdijkl',I26)
    
    #Contraction 1659; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('adcbjlik->abcdijkl',I26)
    
    #Contraction 1660; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('cbadjlik->abcdijkl',I26)
    
    #Contraction 1661; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('cdabjlik->abcdijkl',I26)
    
    #Contraction 1662; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('bdacjlik->abcdijkl',I26)
    
    #Contraction 1663; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('acbdljik->abcdijkl',I26)
    
    #Contraction 1664; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('abcdljik->abcdijkl',I26)
    
    #Contraction 1665; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('adcbljik->abcdijkl',I26)
    
    #Contraction 1666; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('cbadljik->abcdijkl',I26)
    
    #Contraction 1667; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('cdabljik->abcdijkl',I26)
    
    #Contraction 1668; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('bdacljik->abcdijkl',I26)
    
    #Contraction 1669; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('acbdjkil->abcdijkl',I26)
    
    #Contraction 1670; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('abcdjkil->abcdijkl',I26)
    
    #Contraction 1671; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('adcbjkil->abcdijkl',I26)
    
    #Contraction 1672; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('cbadjkil->abcdijkl',I26)
    
    #Contraction 1673; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('cdabjkil->abcdijkl',I26)
    
    #Contraction 1674; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('bdacjkil->abcdijkl',I26)
    
    #Contraction 1675; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('acbdkjil->abcdijkl',I26)
    
    #Contraction 1676; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('abcdkjil->abcdijkl',I26)
    
    #Contraction 1677; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('adcbkjil->abcdijkl',I26)
    
    #Contraction 1678; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('cbadkjil->abcdijkl',I26)
    
    #Contraction 1679; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('cdabkjil->abcdijkl',I26)
    
    #Contraction 1680; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('bdackjil->abcdijkl',I26)
    
    #Contraction 1681; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('acbdiljk->abcdijkl',I26)
    
    #Contraction 1682; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('abcdiljk->abcdijkl',I26)
    
    #Contraction 1683; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('adcbiljk->abcdijkl',I26)
    
    #Contraction 1684; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('cbadiljk->abcdijkl',I26)
    
    #Contraction 1685; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('cdabiljk->abcdijkl',I26)
    
    #Contraction 1686; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('bdaciljk->abcdijkl',I26)
    
    #Contraction 1687; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('acbdlijk->abcdijkl',I26)
    
    #Contraction 1688; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('abcdlijk->abcdijkl',I26)
    
    #Contraction 1689; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('adcblijk->abcdijkl',I26)
    
    #Contraction 1690; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('cbadlijk->abcdijkl',I26)
    
    #Contraction 1691; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('cdablijk->abcdijkl',I26)
    
    #Contraction 1692; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('bdaclijk->abcdijkl',I26)
    
    #Contraction 1693; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('acbdikjl->abcdijkl',I26)
    
    #Contraction 1694; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('abcdikjl->abcdijkl',I26)
    
    #Contraction 1695; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('adcbikjl->abcdijkl',I26)
    
    #Contraction 1696; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('cbadikjl->abcdijkl',I26)
    
    #Contraction 1697; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('cdabikjl->abcdijkl',I26)
    
    #Contraction 1698; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('bdacikjl->abcdijkl',I26)
    
    #Contraction 1699; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('acbdkijl->abcdijkl',I26)
    
    #Contraction 1700; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('abcdkijl->abcdijkl',I26)
    
    #Contraction 1701; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('adcbkijl->abcdijkl',I26)
    
    #Contraction 1702; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('cbadkijl->abcdijkl',I26)
    
    #Contraction 1703; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('cdabkijl->abcdijkl',I26)
    
    #Contraction 1704; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('bdackijl->abcdijkl',I26)
    
    #Contraction 1705; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('acbdijkl->abcdijkl',I26)
    
    #Contraction 1706; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('abcdijkl->abcdijkl',I26)
    
    #Contraction 1707; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('adcbijkl->abcdijkl',I26)
    
    #Contraction 1708; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('cbadijkl->abcdijkl',I26)
    
    #Contraction 1709; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('cdabijkl->abcdijkl',I26)
    
    #Contraction 1710; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('bdacijkl->abcdijkl',I26)
    
    #Contraction 1711; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('acbdjikl->abcdijkl',I26)
    
    #Contraction 1712; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('abcdjikl->abcdijkl',I26)
    
    #Contraction 1713; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('adcbjikl->abcdijkl',I26)
    
    #Contraction 1714; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('cbadjikl->abcdijkl',I26)
    
    #Contraction 1715; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('cdabjikl->abcdijkl',I26)
    
    #Contraction 1716; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('bdacjikl->abcdijkl',I26)
    
    del I26
    
    J26 = np.zeros([nocc, nvir, nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 1717; Tree Level  3; Scaling  6/ 2 Result_size  4/ 2
    J26 += np.einsum('mnij,bdnl->mbdijl',V1,T2, optimize='optimal')
    
    M26 = np.zeros([nvir, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 1718; Tree Level  2; Scaling  6/ 4 Result_size  4/ 4
    M26 += np.einsum('cakm,mbdijl->cabdkijl',T2,J26, optimize='optimal')
    
    del J26
    
    #Contraction 1719; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('cabdkijl->abcdijkl',M26)
    
    #Contraction 1720; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('cbadkijl->abcdijkl',M26)
    
    #Contraction 1721; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('cdabkijl->abcdijkl',M26)
    
    #Contraction 1722; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('abcdkijl->abcdijkl',M26)
    
    #Contraction 1723; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('adcbkijl->abcdijkl',M26)
    
    #Contraction 1724; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('bdcakijl->abcdijkl',M26)
    
    #Contraction 1725; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('cabdlijk->abcdijkl',M26)
    
    #Contraction 1726; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('cbadlijk->abcdijkl',M26)
    
    #Contraction 1727; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('cdablijk->abcdijkl',M26)
    
    #Contraction 1728; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('abcdlijk->abcdijkl',M26)
    
    #Contraction 1729; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('adcblijk->abcdijkl',M26)
    
    #Contraction 1730; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('bdcalijk->abcdijkl',M26)
    
    #Contraction 1731; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('cabdjikl->abcdijkl',M26)
    
    #Contraction 1732; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('cbadjikl->abcdijkl',M26)
    
    #Contraction 1733; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('cdabjikl->abcdijkl',M26)
    
    #Contraction 1734; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('abcdjikl->abcdijkl',M26)
    
    #Contraction 1735; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('adcbjikl->abcdijkl',M26)
    
    #Contraction 1736; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('bdcajikl->abcdijkl',M26)
    
    #Contraction 1737; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('cabdlikj->abcdijkl',M26)
    
    #Contraction 1738; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('cbadlikj->abcdijkl',M26)
    
    #Contraction 1739; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('cdablikj->abcdijkl',M26)
    
    #Contraction 1740; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('abcdlikj->abcdijkl',M26)
    
    #Contraction 1741; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('adcblikj->abcdijkl',M26)
    
    #Contraction 1742; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('bdcalikj->abcdijkl',M26)
    
    #Contraction 1743; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('cabdjilk->abcdijkl',M26)
    
    #Contraction 1744; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('cbadjilk->abcdijkl',M26)
    
    #Contraction 1745; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('cdabjilk->abcdijkl',M26)
    
    #Contraction 1746; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('abcdjilk->abcdijkl',M26)
    
    #Contraction 1747; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('adcbjilk->abcdijkl',M26)
    
    #Contraction 1748; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('bdcajilk->abcdijkl',M26)
    
    #Contraction 1749; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('cabdkilj->abcdijkl',M26)
    
    #Contraction 1750; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('cbadkilj->abcdijkl',M26)
    
    #Contraction 1751; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('cdabkilj->abcdijkl',M26)
    
    #Contraction 1752; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('abcdkilj->abcdijkl',M26)
    
    #Contraction 1753; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('adcbkilj->abcdijkl',M26)
    
    #Contraction 1754; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('bdcakilj->abcdijkl',M26)
    
    #Contraction 1755; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('cabdijkl->abcdijkl',M26)
    
    #Contraction 1756; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('cbadijkl->abcdijkl',M26)
    
    #Contraction 1757; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('cdabijkl->abcdijkl',M26)
    
    #Contraction 1758; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('abcdijkl->abcdijkl',M26)
    
    #Contraction 1759; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('adcbijkl->abcdijkl',M26)
    
    #Contraction 1760; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('bdcaijkl->abcdijkl',M26)
    
    #Contraction 1761; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('cabdljki->abcdijkl',M26)
    
    #Contraction 1762; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('cbadljki->abcdijkl',M26)
    
    #Contraction 1763; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('cdabljki->abcdijkl',M26)
    
    #Contraction 1764; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('abcdljki->abcdijkl',M26)
    
    #Contraction 1765; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('adcbljki->abcdijkl',M26)
    
    #Contraction 1766; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('bdcaljki->abcdijkl',M26)
    
    #Contraction 1767; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('cabdijlk->abcdijkl',M26)
    
    #Contraction 1768; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('cbadijlk->abcdijkl',M26)
    
    #Contraction 1769; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('cdabijlk->abcdijkl',M26)
    
    #Contraction 1770; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('abcdijlk->abcdijkl',M26)
    
    #Contraction 1771; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('adcbijlk->abcdijkl',M26)
    
    #Contraction 1772; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('bdcaijlk->abcdijkl',M26)
    
    #Contraction 1773; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('cabdkjli->abcdijkl',M26)
    
    #Contraction 1774; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('cbadkjli->abcdijkl',M26)
    
    #Contraction 1775; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('cdabkjli->abcdijkl',M26)
    
    #Contraction 1776; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('abcdkjli->abcdijkl',M26)
    
    #Contraction 1777; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('adcbkjli->abcdijkl',M26)
    
    #Contraction 1778; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('bdcakjli->abcdijkl',M26)
    
    #Contraction 1779; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('cabdiklj->abcdijkl',M26)
    
    #Contraction 1780; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('cbadiklj->abcdijkl',M26)
    
    #Contraction 1781; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('cdabiklj->abcdijkl',M26)
    
    #Contraction 1782; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('abcdiklj->abcdijkl',M26)
    
    #Contraction 1783; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('adcbiklj->abcdijkl',M26)
    
    #Contraction 1784; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('bdcaiklj->abcdijkl',M26)
    
    #Contraction 1785; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('cabdjkli->abcdijkl',M26)
    
    #Contraction 1786; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('cbadjkli->abcdijkl',M26)
    
    #Contraction 1787; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('cdabjkli->abcdijkl',M26)
    
    #Contraction 1788; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('abcdjkli->abcdijkl',M26)
    
    #Contraction 1789; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('adcbjkli->abcdijkl',M26)
    
    #Contraction 1790; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('bdcajkli->abcdijkl',M26)
    
    del M26
    
    X26 = np.zeros([nvir, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 1791; Tree Level  2; Scaling  6/ 4 Result_size  4/ 4
    X26 += np.einsum('bm,amcdijkl->bacdijkl',T1,G25, optimize='optimal')
    
    del G25
    
    #Contraction 1792; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bacdijkl->abcdijkl',X26)
    
    #Contraction 1793; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('cabdijkl->abcdijkl',X26)
    
    #Contraction 1794; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('dabcijkl->abcdijkl',X26)
    
    #Contraction 1795; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('abcdijkl->abcdijkl',X26)
    
    #Contraction 1796; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('cbadijkl->abcdijkl',X26)
    
    #Contraction 1797; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('dbacijkl->abcdijkl',X26)
    
    #Contraction 1798; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('acbdijkl->abcdijkl',X26)
    
    #Contraction 1799; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bcadijkl->abcdijkl',X26)
    
    #Contraction 1800; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('dcabijkl->abcdijkl',X26)
    
    #Contraction 1801; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('adbcijkl->abcdijkl',X26)
    
    #Contraction 1802; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bdacijkl->abcdijkl',X26)
    
    #Contraction 1803; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('cdabijkl->abcdijkl',X26)
    
    #Contraction 1804; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bacdjikl->abcdijkl',X26)
    
    #Contraction 1805; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('cabdjikl->abcdijkl',X26)
    
    #Contraction 1806; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('dabcjikl->abcdijkl',X26)
    
    #Contraction 1807; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('abcdjikl->abcdijkl',X26)
    
    #Contraction 1808; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('cbadjikl->abcdijkl',X26)
    
    #Contraction 1809; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('dbacjikl->abcdijkl',X26)
    
    #Contraction 1810; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('acbdjikl->abcdijkl',X26)
    
    #Contraction 1811; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bcadjikl->abcdijkl',X26)
    
    #Contraction 1812; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('dcabjikl->abcdijkl',X26)
    
    #Contraction 1813; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('adbcjikl->abcdijkl',X26)
    
    #Contraction 1814; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bdacjikl->abcdijkl',X26)
    
    #Contraction 1815; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('cdabjikl->abcdijkl',X26)
    
    #Contraction 1816; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bacdkijl->abcdijkl',X26)
    
    #Contraction 1817; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('cabdkijl->abcdijkl',X26)
    
    #Contraction 1818; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('dabckijl->abcdijkl',X26)
    
    #Contraction 1819; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('abcdkijl->abcdijkl',X26)
    
    #Contraction 1820; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('cbadkijl->abcdijkl',X26)
    
    #Contraction 1821; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('dbackijl->abcdijkl',X26)
    
    #Contraction 1822; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('acbdkijl->abcdijkl',X26)
    
    #Contraction 1823; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bcadkijl->abcdijkl',X26)
    
    #Contraction 1824; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('dcabkijl->abcdijkl',X26)
    
    #Contraction 1825; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('adbckijl->abcdijkl',X26)
    
    #Contraction 1826; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bdackijl->abcdijkl',X26)
    
    #Contraction 1827; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('cdabkijl->abcdijkl',X26)
    
    #Contraction 1828; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bacdlijk->abcdijkl',X26)
    
    #Contraction 1829; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('cabdlijk->abcdijkl',X26)
    
    #Contraction 1830; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('dabclijk->abcdijkl',X26)
    
    #Contraction 1831; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('abcdlijk->abcdijkl',X26)
    
    #Contraction 1832; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('cbadlijk->abcdijkl',X26)
    
    #Contraction 1833; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('dbaclijk->abcdijkl',X26)
    
    #Contraction 1834; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('acbdlijk->abcdijkl',X26)
    
    #Contraction 1835; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bcadlijk->abcdijkl',X26)
    
    #Contraction 1836; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('dcablijk->abcdijkl',X26)
    
    #Contraction 1837; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('adbclijk->abcdijkl',X26)
    
    #Contraction 1838; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bdaclijk->abcdijkl',X26)
    
    #Contraction 1839; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('cdablijk->abcdijkl',X26)
    
    del X26
    
    Y26 = np.zeros([nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 1840; Tree Level  3; Scaling  6/ 2 Result_size  4/ 2
    Y26 += np.einsum('am,cmnjki->acnjki',T1,X18, optimize='optimal')
    
    del X18
    
    A27 = np.zeros([nvir, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 1841; Tree Level  2; Scaling  6/ 4 Result_size  4/ 4
    A27 += np.einsum('bdnl,acnjki->bdacljki',T2,Y26, optimize='optimal')
    
    del Y26
    
    #Contraction 1842; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bdacljki->abcdijkl',A27)
    
    #Contraction 1843; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('cdabljki->abcdijkl',A27)
    
    #Contraction 1844; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('cbadljki->abcdijkl',A27)
    
    #Contraction 1845; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bdcaljki->abcdijkl',A27)
    
    #Contraction 1846; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('adcbljki->abcdijkl',A27)
    
    #Contraction 1847; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('abcdljki->abcdijkl',A27)
    
    #Contraction 1848; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('cdbaljki->abcdijkl',A27)
    
    #Contraction 1849; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('adbcljki->abcdijkl',A27)
    
    #Contraction 1850; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('acbdljki->abcdijkl',A27)
    
    #Contraction 1851; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('cbdaljki->abcdijkl',A27)
    
    #Contraction 1852; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('abdcljki->abcdijkl',A27)
    
    #Contraction 1853; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('acdbljki->abcdijkl',A27)
    
    #Contraction 1854; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bdackjli->abcdijkl',A27)
    
    #Contraction 1855; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('cdabkjli->abcdijkl',A27)
    
    #Contraction 1856; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('cbadkjli->abcdijkl',A27)
    
    #Contraction 1857; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bdcakjli->abcdijkl',A27)
    
    #Contraction 1858; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('adcbkjli->abcdijkl',A27)
    
    #Contraction 1859; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('abcdkjli->abcdijkl',A27)
    
    #Contraction 1860; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('cdbakjli->abcdijkl',A27)
    
    #Contraction 1861; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('adbckjli->abcdijkl',A27)
    
    #Contraction 1862; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('acbdkjli->abcdijkl',A27)
    
    #Contraction 1863; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('cbdakjli->abcdijkl',A27)
    
    #Contraction 1864; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('abdckjli->abcdijkl',A27)
    
    #Contraction 1865; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('acdbkjli->abcdijkl',A27)
    
    #Contraction 1866; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bdacjkli->abcdijkl',A27)
    
    #Contraction 1867; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('cdabjkli->abcdijkl',A27)
    
    #Contraction 1868; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('cbadjkli->abcdijkl',A27)
    
    #Contraction 1869; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bdcajkli->abcdijkl',A27)
    
    #Contraction 1870; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('adcbjkli->abcdijkl',A27)
    
    #Contraction 1871; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('abcdjkli->abcdijkl',A27)
    
    #Contraction 1872; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('cdbajkli->abcdijkl',A27)
    
    #Contraction 1873; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('adbcjkli->abcdijkl',A27)
    
    #Contraction 1874; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('acbdjkli->abcdijkl',A27)
    
    #Contraction 1875; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('cbdajkli->abcdijkl',A27)
    
    #Contraction 1876; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('abdcjkli->abcdijkl',A27)
    
    #Contraction 1877; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('acdbjkli->abcdijkl',A27)
    
    #Contraction 1878; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bdaclikj->abcdijkl',A27)
    
    #Contraction 1879; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('cdablikj->abcdijkl',A27)
    
    #Contraction 1880; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('cbadlikj->abcdijkl',A27)
    
    #Contraction 1881; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bdcalikj->abcdijkl',A27)
    
    #Contraction 1882; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('adcblikj->abcdijkl',A27)
    
    #Contraction 1883; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('abcdlikj->abcdijkl',A27)
    
    #Contraction 1884; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('cdbalikj->abcdijkl',A27)
    
    #Contraction 1885; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('adbclikj->abcdijkl',A27)
    
    #Contraction 1886; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('acbdlikj->abcdijkl',A27)
    
    #Contraction 1887; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('cbdalikj->abcdijkl',A27)
    
    #Contraction 1888; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('abdclikj->abcdijkl',A27)
    
    #Contraction 1889; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('acdblikj->abcdijkl',A27)
    
    #Contraction 1890; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bdackilj->abcdijkl',A27)
    
    #Contraction 1891; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('cdabkilj->abcdijkl',A27)
    
    #Contraction 1892; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('cbadkilj->abcdijkl',A27)
    
    #Contraction 1893; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bdcakilj->abcdijkl',A27)
    
    #Contraction 1894; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('adcbkilj->abcdijkl',A27)
    
    #Contraction 1895; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('abcdkilj->abcdijkl',A27)
    
    #Contraction 1896; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('cdbakilj->abcdijkl',A27)
    
    #Contraction 1897; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('adbckilj->abcdijkl',A27)
    
    #Contraction 1898; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('acbdkilj->abcdijkl',A27)
    
    #Contraction 1899; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('cbdakilj->abcdijkl',A27)
    
    #Contraction 1900; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('abdckilj->abcdijkl',A27)
    
    #Contraction 1901; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('acdbkilj->abcdijkl',A27)
    
    #Contraction 1902; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bdaciklj->abcdijkl',A27)
    
    #Contraction 1903; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('cdabiklj->abcdijkl',A27)
    
    #Contraction 1904; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('cbadiklj->abcdijkl',A27)
    
    #Contraction 1905; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bdcaiklj->abcdijkl',A27)
    
    #Contraction 1906; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('adcbiklj->abcdijkl',A27)
    
    #Contraction 1907; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('abcdiklj->abcdijkl',A27)
    
    #Contraction 1908; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('cdbaiklj->abcdijkl',A27)
    
    #Contraction 1909; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('adbciklj->abcdijkl',A27)
    
    #Contraction 1910; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('acbdiklj->abcdijkl',A27)
    
    #Contraction 1911; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('cbdaiklj->abcdijkl',A27)
    
    #Contraction 1912; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('abdciklj->abcdijkl',A27)
    
    #Contraction 1913; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('acdbiklj->abcdijkl',A27)
    
    #Contraction 1914; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bdaclijk->abcdijkl',A27)
    
    #Contraction 1915; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('cdablijk->abcdijkl',A27)
    
    #Contraction 1916; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('cbadlijk->abcdijkl',A27)
    
    #Contraction 1917; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bdcalijk->abcdijkl',A27)
    
    #Contraction 1918; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('adcblijk->abcdijkl',A27)
    
    #Contraction 1919; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('abcdlijk->abcdijkl',A27)
    
    #Contraction 1920; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('cdbalijk->abcdijkl',A27)
    
    #Contraction 1921; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('adbclijk->abcdijkl',A27)
    
    #Contraction 1922; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('acbdlijk->abcdijkl',A27)
    
    #Contraction 1923; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('cbdalijk->abcdijkl',A27)
    
    #Contraction 1924; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('abdclijk->abcdijkl',A27)
    
    #Contraction 1925; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('acdblijk->abcdijkl',A27)
    
    #Contraction 1926; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bdacjilk->abcdijkl',A27)
    
    #Contraction 1927; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('cdabjilk->abcdijkl',A27)
    
    #Contraction 1928; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('cbadjilk->abcdijkl',A27)
    
    #Contraction 1929; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bdcajilk->abcdijkl',A27)
    
    #Contraction 1930; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('adcbjilk->abcdijkl',A27)
    
    #Contraction 1931; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('abcdjilk->abcdijkl',A27)
    
    #Contraction 1932; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('cdbajilk->abcdijkl',A27)
    
    #Contraction 1933; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('adbcjilk->abcdijkl',A27)
    
    #Contraction 1934; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('acbdjilk->abcdijkl',A27)
    
    #Contraction 1935; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('cbdajilk->abcdijkl',A27)
    
    #Contraction 1936; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('abdcjilk->abcdijkl',A27)
    
    #Contraction 1937; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('acdbjilk->abcdijkl',A27)
    
    #Contraction 1938; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bdacijlk->abcdijkl',A27)
    
    #Contraction 1939; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('cdabijlk->abcdijkl',A27)
    
    #Contraction 1940; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('cbadijlk->abcdijkl',A27)
    
    #Contraction 1941; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bdcaijlk->abcdijkl',A27)
    
    #Contraction 1942; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('adcbijlk->abcdijkl',A27)
    
    #Contraction 1943; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('abcdijlk->abcdijkl',A27)
    
    #Contraction 1944; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('cdbaijlk->abcdijkl',A27)
    
    #Contraction 1945; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('adbcijlk->abcdijkl',A27)
    
    #Contraction 1946; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('acbdijlk->abcdijkl',A27)
    
    #Contraction 1947; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('cbdaijlk->abcdijkl',A27)
    
    #Contraction 1948; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('abdcijlk->abcdijkl',A27)
    
    #Contraction 1949; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('acdbijlk->abcdijkl',A27)
    
    #Contraction 1950; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bdackijl->abcdijkl',A27)
    
    #Contraction 1951; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('cdabkijl->abcdijkl',A27)
    
    #Contraction 1952; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('cbadkijl->abcdijkl',A27)
    
    #Contraction 1953; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bdcakijl->abcdijkl',A27)
    
    #Contraction 1954; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('adcbkijl->abcdijkl',A27)
    
    #Contraction 1955; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('abcdkijl->abcdijkl',A27)
    
    #Contraction 1956; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('cdbakijl->abcdijkl',A27)
    
    #Contraction 1957; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('adbckijl->abcdijkl',A27)
    
    #Contraction 1958; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('acbdkijl->abcdijkl',A27)
    
    #Contraction 1959; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('cbdakijl->abcdijkl',A27)
    
    #Contraction 1960; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('abdckijl->abcdijkl',A27)
    
    #Contraction 1961; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('acdbkijl->abcdijkl',A27)
    
    #Contraction 1962; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bdacjikl->abcdijkl',A27)
    
    #Contraction 1963; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('cdabjikl->abcdijkl',A27)
    
    #Contraction 1964; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('cbadjikl->abcdijkl',A27)
    
    #Contraction 1965; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bdcajikl->abcdijkl',A27)
    
    #Contraction 1966; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('adcbjikl->abcdijkl',A27)
    
    #Contraction 1967; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('abcdjikl->abcdijkl',A27)
    
    #Contraction 1968; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('cdbajikl->abcdijkl',A27)
    
    #Contraction 1969; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('adbcjikl->abcdijkl',A27)
    
    #Contraction 1970; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('acbdjikl->abcdijkl',A27)
    
    #Contraction 1971; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('cbdajikl->abcdijkl',A27)
    
    #Contraction 1972; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('abdcjikl->abcdijkl',A27)
    
    #Contraction 1973; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('acdbjikl->abcdijkl',A27)
    
    #Contraction 1974; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bdacijkl->abcdijkl',A27)
    
    #Contraction 1975; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('cdabijkl->abcdijkl',A27)
    
    #Contraction 1976; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('cbadijkl->abcdijkl',A27)
    
    #Contraction 1977; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bdcaijkl->abcdijkl',A27)
    
    #Contraction 1978; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('adcbijkl->abcdijkl',A27)
    
    #Contraction 1979; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('abcdijkl->abcdijkl',A27)
    
    #Contraction 1980; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('cdbaijkl->abcdijkl',A27)
    
    #Contraction 1981; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('adbcijkl->abcdijkl',A27)
    
    #Contraction 1982; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('acbdijkl->abcdijkl',A27)
    
    #Contraction 1983; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('cbdaijkl->abcdijkl',A27)
    
    #Contraction 1984; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('abdcijkl->abcdijkl',A27)
    
    #Contraction 1985; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('acdbijkl->abcdijkl',A27)
    
    del A27
    
    D27 = np.zeros([nvir, nvir, nvir, nocc, nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 1986; Tree Level  3; Scaling  7/ 5 Result_size  5/ 3
    D27 += np.einsum('fbcdnjkl,mnfi->bcdmjkli',T4,I1, optimize='optimal')
    
    E27 = np.zeros([nvir, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 1987; Tree Level  2; Scaling  6/ 4 Result_size  4/ 4
    E27 += np.einsum('am,bcdmjkli->abcdjkli',T1,D27, optimize='optimal')
    
    del D27
    
    #Contraction 1988; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('abcdjkli->abcdijkl',E27)
    
    #Contraction 1989; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bacdjkli->abcdijkl',E27)
    
    #Contraction 1990; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('cabdjkli->abcdijkl',E27)
    
    #Contraction 1991; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('dabcjkli->abcdijkl',E27)
    
    #Contraction 1992; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('abcdiklj->abcdijkl',E27)
    
    #Contraction 1993; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bacdiklj->abcdijkl',E27)
    
    #Contraction 1994; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('cabdiklj->abcdijkl',E27)
    
    #Contraction 1995; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('dabciklj->abcdijkl',E27)
    
    #Contraction 1996; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('abcdijlk->abcdijkl',E27)
    
    #Contraction 1997; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bacdijlk->abcdijkl',E27)
    
    #Contraction 1998; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('cabdijlk->abcdijkl',E27)
    
    #Contraction 1999; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('dabcijlk->abcdijkl',E27)
    
    #Contraction 2000; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('abcdijkl->abcdijkl',E27)
    
    #Contraction 2001; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bacdijkl->abcdijkl',E27)
    
    #Contraction 2002; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('cabdijkl->abcdijkl',E27)
    
    #Contraction 2003; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('dabcijkl->abcdijkl',E27)
    
    del E27
    
    G27 = np.zeros([nvir, nvir, nocc, nocc, nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 2004; Tree Level  4; Scaling  6/ 4 Result_size  6/ 2
    G27 += np.einsum('fcdjkl,mnfi->cdmnjkli',T3,I1, optimize='optimal')
    
    I27 = np.zeros([nvir, nvir, nvir, nocc, nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 2005; Tree Level  3; Scaling  7/ 3 Result_size  5/ 3
    I27 += np.einsum('bn,cdmnjkli->bcdmjkli',T1,G27, optimize='optimal')
    
    del G27
    
    J27 = np.zeros([nvir, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 2006; Tree Level  2; Scaling  6/ 4 Result_size  4/ 4
    J27 += np.einsum('am,bcdmjkli->abcdjkli',T1,I27, optimize='optimal')
    
    del I27
    
    #Contraction 2007; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('abcdjkli->abcdijkl',J27)
    
    #Contraction 2008; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('acbdjkli->abcdijkl',J27)
    
    #Contraction 2009; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('adbcjkli->abcdijkl',J27)
    
    #Contraction 2010; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('bacdjkli->abcdijkl',J27)
    
    #Contraction 2011; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('bcadjkli->abcdijkl',J27)
    
    #Contraction 2012; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('bdacjkli->abcdijkl',J27)
    
    #Contraction 2013; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('cabdjkli->abcdijkl',J27)
    
    #Contraction 2014; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('cbadjkli->abcdijkl',J27)
    
    #Contraction 2015; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('cdabjkli->abcdijkl',J27)
    
    #Contraction 2016; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('dabcjkli->abcdijkl',J27)
    
    #Contraction 2017; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('dbacjkli->abcdijkl',J27)
    
    #Contraction 2018; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('dcabjkli->abcdijkl',J27)
    
    #Contraction 2019; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('abcdiklj->abcdijkl',J27)
    
    #Contraction 2020; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('acbdiklj->abcdijkl',J27)
    
    #Contraction 2021; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('adbciklj->abcdijkl',J27)
    
    #Contraction 2022; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('bacdiklj->abcdijkl',J27)
    
    #Contraction 2023; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('bcadiklj->abcdijkl',J27)
    
    #Contraction 2024; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('bdaciklj->abcdijkl',J27)
    
    #Contraction 2025; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('cabdiklj->abcdijkl',J27)
    
    #Contraction 2026; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('cbadiklj->abcdijkl',J27)
    
    #Contraction 2027; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('cdabiklj->abcdijkl',J27)
    
    #Contraction 2028; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('dabciklj->abcdijkl',J27)
    
    #Contraction 2029; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('dbaciklj->abcdijkl',J27)
    
    #Contraction 2030; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('dcabiklj->abcdijkl',J27)
    
    #Contraction 2031; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('abcdijlk->abcdijkl',J27)
    
    #Contraction 2032; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('acbdijlk->abcdijkl',J27)
    
    #Contraction 2033; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('adbcijlk->abcdijkl',J27)
    
    #Contraction 2034; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('bacdijlk->abcdijkl',J27)
    
    #Contraction 2035; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('bcadijlk->abcdijkl',J27)
    
    #Contraction 2036; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('bdacijlk->abcdijkl',J27)
    
    #Contraction 2037; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('cabdijlk->abcdijkl',J27)
    
    #Contraction 2038; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('cbadijlk->abcdijkl',J27)
    
    #Contraction 2039; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('cdabijlk->abcdijkl',J27)
    
    #Contraction 2040; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('dabcijlk->abcdijkl',J27)
    
    #Contraction 2041; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('dbacijlk->abcdijkl',J27)
    
    #Contraction 2042; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('dcabijlk->abcdijkl',J27)
    
    #Contraction 2043; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('abcdijkl->abcdijkl',J27)
    
    #Contraction 2044; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('acbdijkl->abcdijkl',J27)
    
    #Contraction 2045; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('adbcijkl->abcdijkl',J27)
    
    #Contraction 2046; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('bacdijkl->abcdijkl',J27)
    
    #Contraction 2047; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('bcadijkl->abcdijkl',J27)
    
    #Contraction 2048; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('bdacijkl->abcdijkl',J27)
    
    #Contraction 2049; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('cabdijkl->abcdijkl',J27)
    
    #Contraction 2050; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('cbadijkl->abcdijkl',J27)
    
    #Contraction 2051; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('cdabijkl->abcdijkl',J27)
    
    #Contraction 2052; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('dabcijkl->abcdijkl',J27)
    
    #Contraction 2053; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('dbacijkl->abcdijkl',J27)
    
    #Contraction 2054; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('dcabijkl->abcdijkl',J27)
    
    del J27
    
    M27 = np.zeros([nvir, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 2055; Tree Level  2; Scaling  6/ 4 Result_size  4/ 4
    M27 += np.einsum('acmk,bdmlij->acbdklij',T2,Y23, optimize='optimal')
    
    del Y23
    
    #Contraction 2056; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('acbdklij->abcdijkl',M27)
    
    #Contraction 2057; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('abcdklij->abcdijkl',M27)
    
    #Contraction 2058; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('adcbklij->abcdijkl',M27)
    
    #Contraction 2059; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('cbadklij->abcdijkl',M27)
    
    #Contraction 2060; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('cdabklij->abcdijkl',M27)
    
    #Contraction 2061; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('bdacklij->abcdijkl',M27)
    
    #Contraction 2062; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('acbdlkij->abcdijkl',M27)
    
    #Contraction 2063; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('abcdlkij->abcdijkl',M27)
    
    #Contraction 2064; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('adcblkij->abcdijkl',M27)
    
    #Contraction 2065; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('cbadlkij->abcdijkl',M27)
    
    #Contraction 2066; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('cdablkij->abcdijkl',M27)
    
    #Contraction 2067; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('bdaclkij->abcdijkl',M27)
    
    #Contraction 2068; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('acbdjlik->abcdijkl',M27)
    
    #Contraction 2069; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('abcdjlik->abcdijkl',M27)
    
    #Contraction 2070; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('adcbjlik->abcdijkl',M27)
    
    #Contraction 2071; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('cbadjlik->abcdijkl',M27)
    
    #Contraction 2072; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('cdabjlik->abcdijkl',M27)
    
    #Contraction 2073; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('bdacjlik->abcdijkl',M27)
    
    #Contraction 2074; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('acbdljik->abcdijkl',M27)
    
    #Contraction 2075; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('abcdljik->abcdijkl',M27)
    
    #Contraction 2076; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('adcbljik->abcdijkl',M27)
    
    #Contraction 2077; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('cbadljik->abcdijkl',M27)
    
    #Contraction 2078; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('cdabljik->abcdijkl',M27)
    
    #Contraction 2079; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('bdacljik->abcdijkl',M27)
    
    #Contraction 2080; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('acbdjkil->abcdijkl',M27)
    
    #Contraction 2081; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('abcdjkil->abcdijkl',M27)
    
    #Contraction 2082; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('adcbjkil->abcdijkl',M27)
    
    #Contraction 2083; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('cbadjkil->abcdijkl',M27)
    
    #Contraction 2084; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('cdabjkil->abcdijkl',M27)
    
    #Contraction 2085; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('bdacjkil->abcdijkl',M27)
    
    #Contraction 2086; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('acbdkjil->abcdijkl',M27)
    
    #Contraction 2087; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('abcdkjil->abcdijkl',M27)
    
    #Contraction 2088; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('adcbkjil->abcdijkl',M27)
    
    #Contraction 2089; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('cbadkjil->abcdijkl',M27)
    
    #Contraction 2090; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('cdabkjil->abcdijkl',M27)
    
    #Contraction 2091; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('bdackjil->abcdijkl',M27)
    
    #Contraction 2092; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('acbdklji->abcdijkl',M27)
    
    #Contraction 2093; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('abcdklji->abcdijkl',M27)
    
    #Contraction 2094; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('adcbklji->abcdijkl',M27)
    
    #Contraction 2095; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('cbadklji->abcdijkl',M27)
    
    #Contraction 2096; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('cdabklji->abcdijkl',M27)
    
    #Contraction 2097; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('bdacklji->abcdijkl',M27)
    
    #Contraction 2098; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('acbdlkji->abcdijkl',M27)
    
    #Contraction 2099; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('abcdlkji->abcdijkl',M27)
    
    #Contraction 2100; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('adcblkji->abcdijkl',M27)
    
    #Contraction 2101; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('cbadlkji->abcdijkl',M27)
    
    #Contraction 2102; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('cdablkji->abcdijkl',M27)
    
    #Contraction 2103; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('bdaclkji->abcdijkl',M27)
    
    #Contraction 2104; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('acbdiljk->abcdijkl',M27)
    
    #Contraction 2105; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('abcdiljk->abcdijkl',M27)
    
    #Contraction 2106; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('adcbiljk->abcdijkl',M27)
    
    #Contraction 2107; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('cbadiljk->abcdijkl',M27)
    
    #Contraction 2108; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('cdabiljk->abcdijkl',M27)
    
    #Contraction 2109; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('bdaciljk->abcdijkl',M27)
    
    #Contraction 2110; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('acbdlijk->abcdijkl',M27)
    
    #Contraction 2111; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('abcdlijk->abcdijkl',M27)
    
    #Contraction 2112; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('adcblijk->abcdijkl',M27)
    
    #Contraction 2113; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('cbadlijk->abcdijkl',M27)
    
    #Contraction 2114; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('cdablijk->abcdijkl',M27)
    
    #Contraction 2115; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('bdaclijk->abcdijkl',M27)
    
    #Contraction 2116; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('acbdikjl->abcdijkl',M27)
    
    #Contraction 2117; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('abcdikjl->abcdijkl',M27)
    
    #Contraction 2118; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('adcbikjl->abcdijkl',M27)
    
    #Contraction 2119; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('cbadikjl->abcdijkl',M27)
    
    #Contraction 2120; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('cdabikjl->abcdijkl',M27)
    
    #Contraction 2121; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('bdacikjl->abcdijkl',M27)
    
    #Contraction 2122; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('acbdkijl->abcdijkl',M27)
    
    #Contraction 2123; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('abcdkijl->abcdijkl',M27)
    
    #Contraction 2124; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('adcbkijl->abcdijkl',M27)
    
    #Contraction 2125; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('cbadkijl->abcdijkl',M27)
    
    #Contraction 2126; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('cdabkijl->abcdijkl',M27)
    
    #Contraction 2127; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('bdackijl->abcdijkl',M27)
    
    #Contraction 2128; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('acbdjlki->abcdijkl',M27)
    
    #Contraction 2129; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('abcdjlki->abcdijkl',M27)
    
    #Contraction 2130; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('adcbjlki->abcdijkl',M27)
    
    #Contraction 2131; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('cbadjlki->abcdijkl',M27)
    
    #Contraction 2132; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('cdabjlki->abcdijkl',M27)
    
    #Contraction 2133; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('bdacjlki->abcdijkl',M27)
    
    #Contraction 2134; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('acbdljki->abcdijkl',M27)
    
    #Contraction 2135; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('abcdljki->abcdijkl',M27)
    
    #Contraction 2136; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('adcbljki->abcdijkl',M27)
    
    #Contraction 2137; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('cbadljki->abcdijkl',M27)
    
    #Contraction 2138; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('cdabljki->abcdijkl',M27)
    
    #Contraction 2139; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('bdacljki->abcdijkl',M27)
    
    #Contraction 2140; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('acbdilkj->abcdijkl',M27)
    
    #Contraction 2141; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('abcdilkj->abcdijkl',M27)
    
    #Contraction 2142; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('adcbilkj->abcdijkl',M27)
    
    #Contraction 2143; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('cbadilkj->abcdijkl',M27)
    
    #Contraction 2144; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('cdabilkj->abcdijkl',M27)
    
    #Contraction 2145; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('bdacilkj->abcdijkl',M27)
    
    #Contraction 2146; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('acbdlikj->abcdijkl',M27)
    
    #Contraction 2147; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('abcdlikj->abcdijkl',M27)
    
    #Contraction 2148; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('adcblikj->abcdijkl',M27)
    
    #Contraction 2149; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('cbadlikj->abcdijkl',M27)
    
    #Contraction 2150; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('cdablikj->abcdijkl',M27)
    
    #Contraction 2151; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('bdaclikj->abcdijkl',M27)
    
    #Contraction 2152; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('acbdijkl->abcdijkl',M27)
    
    #Contraction 2153; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('abcdijkl->abcdijkl',M27)
    
    #Contraction 2154; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('adcbijkl->abcdijkl',M27)
    
    #Contraction 2155; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('cbadijkl->abcdijkl',M27)
    
    #Contraction 2156; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('cdabijkl->abcdijkl',M27)
    
    #Contraction 2157; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('bdacijkl->abcdijkl',M27)
    
    #Contraction 2158; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('acbdjikl->abcdijkl',M27)
    
    #Contraction 2159; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('abcdjikl->abcdijkl',M27)
    
    #Contraction 2160; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('adcbjikl->abcdijkl',M27)
    
    #Contraction 2161; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('cbadjikl->abcdijkl',M27)
    
    #Contraction 2162; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('cdabjikl->abcdijkl',M27)
    
    #Contraction 2163; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('bdacjikl->abcdijkl',M27)
    
    #Contraction 2164; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('acbdjkli->abcdijkl',M27)
    
    #Contraction 2165; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('abcdjkli->abcdijkl',M27)
    
    #Contraction 2166; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('adcbjkli->abcdijkl',M27)
    
    #Contraction 2167; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('cbadjkli->abcdijkl',M27)
    
    #Contraction 2168; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('cdabjkli->abcdijkl',M27)
    
    #Contraction 2169; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('bdacjkli->abcdijkl',M27)
    
    #Contraction 2170; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('acbdkjli->abcdijkl',M27)
    
    #Contraction 2171; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('abcdkjli->abcdijkl',M27)
    
    #Contraction 2172; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('adcbkjli->abcdijkl',M27)
    
    #Contraction 2173; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('cbadkjli->abcdijkl',M27)
    
    #Contraction 2174; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('cdabkjli->abcdijkl',M27)
    
    #Contraction 2175; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('bdackjli->abcdijkl',M27)
    
    #Contraction 2176; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('acbdiklj->abcdijkl',M27)
    
    #Contraction 2177; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('abcdiklj->abcdijkl',M27)
    
    #Contraction 2178; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('adcbiklj->abcdijkl',M27)
    
    #Contraction 2179; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('cbadiklj->abcdijkl',M27)
    
    #Contraction 2180; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('cdabiklj->abcdijkl',M27)
    
    #Contraction 2181; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('bdaciklj->abcdijkl',M27)
    
    #Contraction 2182; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('acbdkilj->abcdijkl',M27)
    
    #Contraction 2183; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('abcdkilj->abcdijkl',M27)
    
    #Contraction 2184; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('adcbkilj->abcdijkl',M27)
    
    #Contraction 2185; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('cbadkilj->abcdijkl',M27)
    
    #Contraction 2186; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('cdabkilj->abcdijkl',M27)
    
    #Contraction 2187; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('bdackilj->abcdijkl',M27)
    
    #Contraction 2188; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('acbdijlk->abcdijkl',M27)
    
    #Contraction 2189; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('abcdijlk->abcdijkl',M27)
    
    #Contraction 2190; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('adcbijlk->abcdijkl',M27)
    
    #Contraction 2191; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('cbadijlk->abcdijkl',M27)
    
    #Contraction 2192; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('cdabijlk->abcdijkl',M27)
    
    #Contraction 2193; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('bdacijlk->abcdijkl',M27)
    
    #Contraction 2194; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('acbdjilk->abcdijkl',M27)
    
    #Contraction 2195; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('abcdjilk->abcdijkl',M27)
    
    #Contraction 2196; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('adcbjilk->abcdijkl',M27)
    
    #Contraction 2197; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('cbadjilk->abcdijkl',M27)
    
    #Contraction 2198; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('cdabjilk->abcdijkl',M27)
    
    #Contraction 2199; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('bdacjilk->abcdijkl',M27)
    
    del M27
    
    X27 = np.zeros([nvir, nocc, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 2200; Tree Level  3; Scaling  5/ 7 Result_size  5/ 3
    X27 += np.einsum('amef,efcdijkl->amcdijkl',V8,T4, optimize='optimal')
    
    Y27 = np.zeros([nvir, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 2201; Tree Level  2; Scaling  6/ 4 Result_size  4/ 4
    Y27 += np.einsum('abcdnjkl,ni->abcdjkli',T4,A5, optimize='optimal')
    
    del A5
    
    #Contraction 2202; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('abcdjkli->abcdijkl',Y27)
    
    #Contraction 2203; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('abcdiklj->abcdijkl',Y27)
    
    #Contraction 2204; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('abcdijlk->abcdijkl',Y27)
    
    #Contraction 2205; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('abcdijkl->abcdijkl',Y27)
    
    del Y27
    
    A28 = np.zeros([nvir, nvir, nvir, nocc, nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 2206; Tree Level  3; Scaling  5/ 5 Result_size  5/ 3
    A28 += np.einsum('fbcdijkl,mf->bcdmijkl',T4,D6, optimize='optimal')
    
    del D6
    
    D28 = np.zeros([nvir, nocc, nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 2207; Tree Level  3; Scaling  6/ 2 Result_size  4/ 2
    D28 += np.einsum('am,mndikl->andikl',T1,G13, optimize='optimal')
    
    del G13
    
    E28 = np.zeros([nvir, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 2208; Tree Level  2; Scaling  6/ 4 Result_size  4/ 4
    E28 += np.einsum('cbnj,andikl->cbadjikl',T2,D28, optimize='optimal')
    
    del D28
    
    #Contraction 2209; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('cbadjikl->abcdijkl',E28)
    
    #Contraction 2210; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('cdabjikl->abcdijkl',E28)
    
    #Contraction 2211; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('bdacjikl->abcdijkl',E28)
    
    #Contraction 2212; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('abcdjikl->abcdijkl',E28)
    
    #Contraction 2213; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('adcbjikl->abcdijkl',E28)
    
    #Contraction 2214; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('bdcajikl->abcdijkl',E28)
    
    #Contraction 2215; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('acbdjikl->abcdijkl',E28)
    
    #Contraction 2216; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('adbcjikl->abcdijkl',E28)
    
    #Contraction 2217; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('cdbajikl->abcdijkl',E28)
    
    #Contraction 2218; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('acdbjikl->abcdijkl',E28)
    
    #Contraction 2219; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('abdcjikl->abcdijkl',E28)
    
    #Contraction 2220; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('cbdajikl->abcdijkl',E28)
    
    #Contraction 2221; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('cbadijkl->abcdijkl',E28)
    
    #Contraction 2222; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('cdabijkl->abcdijkl',E28)
    
    #Contraction 2223; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('bdacijkl->abcdijkl',E28)
    
    #Contraction 2224; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('abcdijkl->abcdijkl',E28)
    
    #Contraction 2225; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('adcbijkl->abcdijkl',E28)
    
    #Contraction 2226; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('bdcaijkl->abcdijkl',E28)
    
    #Contraction 2227; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('acbdijkl->abcdijkl',E28)
    
    #Contraction 2228; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('adbcijkl->abcdijkl',E28)
    
    #Contraction 2229; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('cdbaijkl->abcdijkl',E28)
    
    #Contraction 2230; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('acdbijkl->abcdijkl',E28)
    
    #Contraction 2231; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('abdcijkl->abcdijkl',E28)
    
    #Contraction 2232; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('cbdaijkl->abcdijkl',E28)
    
    #Contraction 2233; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('cbadkjil->abcdijkl',E28)
    
    #Contraction 2234; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('cdabkjil->abcdijkl',E28)
    
    #Contraction 2235; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('bdackjil->abcdijkl',E28)
    
    #Contraction 2236; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('abcdkjil->abcdijkl',E28)
    
    #Contraction 2237; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('adcbkjil->abcdijkl',E28)
    
    #Contraction 2238; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('bdcakjil->abcdijkl',E28)
    
    #Contraction 2239; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('acbdkjil->abcdijkl',E28)
    
    #Contraction 2240; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('adbckjil->abcdijkl',E28)
    
    #Contraction 2241; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('cdbakjil->abcdijkl',E28)
    
    #Contraction 2242; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('acdbkjil->abcdijkl',E28)
    
    #Contraction 2243; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('abdckjil->abcdijkl',E28)
    
    #Contraction 2244; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('cbdakjil->abcdijkl',E28)
    
    #Contraction 2245; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('cbadljik->abcdijkl',E28)
    
    #Contraction 2246; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('cdabljik->abcdijkl',E28)
    
    #Contraction 2247; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('bdacljik->abcdijkl',E28)
    
    #Contraction 2248; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('abcdljik->abcdijkl',E28)
    
    #Contraction 2249; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('adcbljik->abcdijkl',E28)
    
    #Contraction 2250; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('bdcaljik->abcdijkl',E28)
    
    #Contraction 2251; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('acbdljik->abcdijkl',E28)
    
    #Contraction 2252; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('adbcljik->abcdijkl',E28)
    
    #Contraction 2253; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('cdbaljik->abcdijkl',E28)
    
    #Contraction 2254; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('acdbljik->abcdijkl',E28)
    
    #Contraction 2255; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('abdcljik->abcdijkl',E28)
    
    #Contraction 2256; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('cbdaljik->abcdijkl',E28)
    
    del E28
    
    G28 = np.zeros([nvir, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 2257; Tree Level  2; Scaling  6/ 4 Result_size  4/ 4
    G28 += np.einsum('bcmk,amdijl->bcadkijl',T2,M22, optimize='optimal')
    
    del M22
    
    #Contraction 2258; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bcadkijl->abcdijkl',G28)
    
    #Contraction 2259; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bdackijl->abcdijkl',G28)
    
    #Contraction 2260; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('cdabkijl->abcdijkl',G28)
    
    #Contraction 2261; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('acbdkijl->abcdijkl',G28)
    
    #Contraction 2262; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('adbckijl->abcdijkl',G28)
    
    #Contraction 2263; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('cdbakijl->abcdijkl',G28)
    
    #Contraction 2264; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('abcdkijl->abcdijkl',G28)
    
    #Contraction 2265; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('adcbkijl->abcdijkl',G28)
    
    #Contraction 2266; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bdcakijl->abcdijkl',G28)
    
    #Contraction 2267; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('abdckijl->abcdijkl',G28)
    
    #Contraction 2268; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('acdbkijl->abcdijkl',G28)
    
    #Contraction 2269; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bcdakijl->abcdijkl',G28)
    
    #Contraction 2270; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bcadlijk->abcdijkl',G28)
    
    #Contraction 2271; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bdaclijk->abcdijkl',G28)
    
    #Contraction 2272; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('cdablijk->abcdijkl',G28)
    
    #Contraction 2273; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('acbdlijk->abcdijkl',G28)
    
    #Contraction 2274; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('adbclijk->abcdijkl',G28)
    
    #Contraction 2275; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('cdbalijk->abcdijkl',G28)
    
    #Contraction 2276; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('abcdlijk->abcdijkl',G28)
    
    #Contraction 2277; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('adcblijk->abcdijkl',G28)
    
    #Contraction 2278; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bdcalijk->abcdijkl',G28)
    
    #Contraction 2279; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('abdclijk->abcdijkl',G28)
    
    #Contraction 2280; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('acdblijk->abcdijkl',G28)
    
    #Contraction 2281; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bcdalijk->abcdijkl',G28)
    
    #Contraction 2282; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bcadjikl->abcdijkl',G28)
    
    #Contraction 2283; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bdacjikl->abcdijkl',G28)
    
    #Contraction 2284; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('cdabjikl->abcdijkl',G28)
    
    #Contraction 2285; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('acbdjikl->abcdijkl',G28)
    
    #Contraction 2286; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('adbcjikl->abcdijkl',G28)
    
    #Contraction 2287; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('cdbajikl->abcdijkl',G28)
    
    #Contraction 2288; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('abcdjikl->abcdijkl',G28)
    
    #Contraction 2289; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('adcbjikl->abcdijkl',G28)
    
    #Contraction 2290; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bdcajikl->abcdijkl',G28)
    
    #Contraction 2291; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('abdcjikl->abcdijkl',G28)
    
    #Contraction 2292; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('acdbjikl->abcdijkl',G28)
    
    #Contraction 2293; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bcdajikl->abcdijkl',G28)
    
    #Contraction 2294; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bcadlikj->abcdijkl',G28)
    
    #Contraction 2295; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bdaclikj->abcdijkl',G28)
    
    #Contraction 2296; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('cdablikj->abcdijkl',G28)
    
    #Contraction 2297; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('acbdlikj->abcdijkl',G28)
    
    #Contraction 2298; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('adbclikj->abcdijkl',G28)
    
    #Contraction 2299; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('cdbalikj->abcdijkl',G28)
    
    #Contraction 2300; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('abcdlikj->abcdijkl',G28)
    
    #Contraction 2301; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('adcblikj->abcdijkl',G28)
    
    #Contraction 2302; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bdcalikj->abcdijkl',G28)
    
    #Contraction 2303; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('abdclikj->abcdijkl',G28)
    
    #Contraction 2304; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('acdblikj->abcdijkl',G28)
    
    #Contraction 2305; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bcdalikj->abcdijkl',G28)
    
    #Contraction 2306; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bcadjilk->abcdijkl',G28)
    
    #Contraction 2307; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bdacjilk->abcdijkl',G28)
    
    #Contraction 2308; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('cdabjilk->abcdijkl',G28)
    
    #Contraction 2309; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('acbdjilk->abcdijkl',G28)
    
    #Contraction 2310; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('adbcjilk->abcdijkl',G28)
    
    #Contraction 2311; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('cdbajilk->abcdijkl',G28)
    
    #Contraction 2312; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('abcdjilk->abcdijkl',G28)
    
    #Contraction 2313; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('adcbjilk->abcdijkl',G28)
    
    #Contraction 2314; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bdcajilk->abcdijkl',G28)
    
    #Contraction 2315; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('abdcjilk->abcdijkl',G28)
    
    #Contraction 2316; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('acdbjilk->abcdijkl',G28)
    
    #Contraction 2317; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bcdajilk->abcdijkl',G28)
    
    #Contraction 2318; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bcadkilj->abcdijkl',G28)
    
    #Contraction 2319; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bdackilj->abcdijkl',G28)
    
    #Contraction 2320; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('cdabkilj->abcdijkl',G28)
    
    #Contraction 2321; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('acbdkilj->abcdijkl',G28)
    
    #Contraction 2322; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('adbckilj->abcdijkl',G28)
    
    #Contraction 2323; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('cdbakilj->abcdijkl',G28)
    
    #Contraction 2324; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('abcdkilj->abcdijkl',G28)
    
    #Contraction 2325; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('adcbkilj->abcdijkl',G28)
    
    #Contraction 2326; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bdcakilj->abcdijkl',G28)
    
    #Contraction 2327; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('abdckilj->abcdijkl',G28)
    
    #Contraction 2328; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('acdbkilj->abcdijkl',G28)
    
    #Contraction 2329; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bcdakilj->abcdijkl',G28)
    
    #Contraction 2330; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bcadijkl->abcdijkl',G28)
    
    #Contraction 2331; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bdacijkl->abcdijkl',G28)
    
    #Contraction 2332; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('cdabijkl->abcdijkl',G28)
    
    #Contraction 2333; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('acbdijkl->abcdijkl',G28)
    
    #Contraction 2334; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('adbcijkl->abcdijkl',G28)
    
    #Contraction 2335; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('cdbaijkl->abcdijkl',G28)
    
    #Contraction 2336; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('abcdijkl->abcdijkl',G28)
    
    #Contraction 2337; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('adcbijkl->abcdijkl',G28)
    
    #Contraction 2338; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bdcaijkl->abcdijkl',G28)
    
    #Contraction 2339; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('abdcijkl->abcdijkl',G28)
    
    #Contraction 2340; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('acdbijkl->abcdijkl',G28)
    
    #Contraction 2341; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bcdaijkl->abcdijkl',G28)
    
    #Contraction 2342; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bcadljki->abcdijkl',G28)
    
    #Contraction 2343; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bdacljki->abcdijkl',G28)
    
    #Contraction 2344; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('cdabljki->abcdijkl',G28)
    
    #Contraction 2345; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('acbdljki->abcdijkl',G28)
    
    #Contraction 2346; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('adbcljki->abcdijkl',G28)
    
    #Contraction 2347; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('cdbaljki->abcdijkl',G28)
    
    #Contraction 2348; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('abcdljki->abcdijkl',G28)
    
    #Contraction 2349; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('adcbljki->abcdijkl',G28)
    
    #Contraction 2350; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bdcaljki->abcdijkl',G28)
    
    #Contraction 2351; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('abdcljki->abcdijkl',G28)
    
    #Contraction 2352; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('acdbljki->abcdijkl',G28)
    
    #Contraction 2353; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bcdaljki->abcdijkl',G28)
    
    #Contraction 2354; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bcadijlk->abcdijkl',G28)
    
    #Contraction 2355; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bdacijlk->abcdijkl',G28)
    
    #Contraction 2356; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('cdabijlk->abcdijkl',G28)
    
    #Contraction 2357; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('acbdijlk->abcdijkl',G28)
    
    #Contraction 2358; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('adbcijlk->abcdijkl',G28)
    
    #Contraction 2359; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('cdbaijlk->abcdijkl',G28)
    
    #Contraction 2360; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('abcdijlk->abcdijkl',G28)
    
    #Contraction 2361; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('adcbijlk->abcdijkl',G28)
    
    #Contraction 2362; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bdcaijlk->abcdijkl',G28)
    
    #Contraction 2363; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('abdcijlk->abcdijkl',G28)
    
    #Contraction 2364; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('acdbijlk->abcdijkl',G28)
    
    #Contraction 2365; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bcdaijlk->abcdijkl',G28)
    
    #Contraction 2366; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bcadkjli->abcdijkl',G28)
    
    #Contraction 2367; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bdackjli->abcdijkl',G28)
    
    #Contraction 2368; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('cdabkjli->abcdijkl',G28)
    
    #Contraction 2369; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('acbdkjli->abcdijkl',G28)
    
    #Contraction 2370; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('adbckjli->abcdijkl',G28)
    
    #Contraction 2371; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('cdbakjli->abcdijkl',G28)
    
    #Contraction 2372; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('abcdkjli->abcdijkl',G28)
    
    #Contraction 2373; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('adcbkjli->abcdijkl',G28)
    
    #Contraction 2374; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bdcakjli->abcdijkl',G28)
    
    #Contraction 2375; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('abdckjli->abcdijkl',G28)
    
    #Contraction 2376; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('acdbkjli->abcdijkl',G28)
    
    #Contraction 2377; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bcdakjli->abcdijkl',G28)
    
    #Contraction 2378; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bcadiklj->abcdijkl',G28)
    
    #Contraction 2379; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bdaciklj->abcdijkl',G28)
    
    #Contraction 2380; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('cdabiklj->abcdijkl',G28)
    
    #Contraction 2381; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('acbdiklj->abcdijkl',G28)
    
    #Contraction 2382; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('adbciklj->abcdijkl',G28)
    
    #Contraction 2383; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('cdbaiklj->abcdijkl',G28)
    
    #Contraction 2384; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('abcdiklj->abcdijkl',G28)
    
    #Contraction 2385; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('adcbiklj->abcdijkl',G28)
    
    #Contraction 2386; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bdcaiklj->abcdijkl',G28)
    
    #Contraction 2387; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('abdciklj->abcdijkl',G28)
    
    #Contraction 2388; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('acdbiklj->abcdijkl',G28)
    
    #Contraction 2389; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bcdaiklj->abcdijkl',G28)
    
    #Contraction 2390; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bcadjkli->abcdijkl',G28)
    
    #Contraction 2391; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bdacjkli->abcdijkl',G28)
    
    #Contraction 2392; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('cdabjkli->abcdijkl',G28)
    
    #Contraction 2393; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('acbdjkli->abcdijkl',G28)
    
    #Contraction 2394; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('adbcjkli->abcdijkl',G28)
    
    #Contraction 2395; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('cdbajkli->abcdijkl',G28)
    
    #Contraction 2396; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('abcdjkli->abcdijkl',G28)
    
    #Contraction 2397; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('adcbjkli->abcdijkl',G28)
    
    #Contraction 2398; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bdcajkli->abcdijkl',G28)
    
    #Contraction 2399; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('abdcjkli->abcdijkl',G28)
    
    #Contraction 2400; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('acdbjkli->abcdijkl',G28)
    
    #Contraction 2401; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bcdajkli->abcdijkl',G28)
    
    del G28
    
    I28 = np.zeros([nocc, nocc, nvir, nvir, nocc, nocc], dtype=type_)
    
    #Contraction 2402; Tree Level  5; Scaling  4/ 4 Result_size  4/ 2
    I28 += np.einsum('mnef,fdjl->mndejl',V7,T2, optimize='optimal')
    
    J28 = np.zeros([nvir, nocc, nocc, nvir, nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 2403; Tree Level  4; Scaling  6/ 4 Result_size  6/ 2
    J28 += np.einsum('ecik,mndejl->cmndikjl',T2,I28, optimize='optimal')
    
    M28 = np.zeros([nvir, nvir, nocc, nvir, nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 2404; Tree Level  3; Scaling  7/ 3 Result_size  5/ 3
    M28 += np.einsum('bn,cmndikjl->bcmdikjl',T1,J28, optimize='optimal')
    
    del J28
    
    X28 = np.zeros([nvir, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 2405; Tree Level  2; Scaling  6/ 4 Result_size  4/ 4
    X28 += np.einsum('am,bcmdikjl->abcdikjl',T1,M28, optimize='optimal')
    
    del M28
    
    #Contraction 2406; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('abcdikjl->abcdijkl',X28)
    
    #Contraction 2407; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('abdcikjl->abcdijkl',X28)
    
    #Contraction 2408; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('acbdikjl->abcdijkl',X28)
    
    #Contraction 2409; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('acdbikjl->abcdijkl',X28)
    
    #Contraction 2410; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('adbcikjl->abcdijkl',X28)
    
    #Contraction 2411; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('adcbikjl->abcdijkl',X28)
    
    #Contraction 2412; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('bacdikjl->abcdijkl',X28)
    
    #Contraction 2413; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('badcikjl->abcdijkl',X28)
    
    #Contraction 2414; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('bcadikjl->abcdijkl',X28)
    
    #Contraction 2415; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('bcdaikjl->abcdijkl',X28)
    
    #Contraction 2416; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('bdacikjl->abcdijkl',X28)
    
    #Contraction 2417; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('bdcaikjl->abcdijkl',X28)
    
    #Contraction 2418; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('cabdikjl->abcdijkl',X28)
    
    #Contraction 2419; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('cadbikjl->abcdijkl',X28)
    
    #Contraction 2420; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('cbadikjl->abcdijkl',X28)
    
    #Contraction 2421; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('cbdaikjl->abcdijkl',X28)
    
    #Contraction 2422; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('cdabikjl->abcdijkl',X28)
    
    #Contraction 2423; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('cdbaikjl->abcdijkl',X28)
    
    #Contraction 2424; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('dabcikjl->abcdijkl',X28)
    
    #Contraction 2425; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('dacbikjl->abcdijkl',X28)
    
    #Contraction 2426; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('dbacikjl->abcdijkl',X28)
    
    #Contraction 2427; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('dbcaikjl->abcdijkl',X28)
    
    #Contraction 2428; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('dcabikjl->abcdijkl',X28)
    
    #Contraction 2429; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('dcbaikjl->abcdijkl',X28)
    
    #Contraction 2430; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('abcdijkl->abcdijkl',X28)
    
    #Contraction 2431; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('abdcijkl->abcdijkl',X28)
    
    #Contraction 2432; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('acbdijkl->abcdijkl',X28)
    
    #Contraction 2433; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('acdbijkl->abcdijkl',X28)
    
    #Contraction 2434; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('adbcijkl->abcdijkl',X28)
    
    #Contraction 2435; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('adcbijkl->abcdijkl',X28)
    
    #Contraction 2436; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('bacdijkl->abcdijkl',X28)
    
    #Contraction 2437; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('badcijkl->abcdijkl',X28)
    
    #Contraction 2438; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('bcadijkl->abcdijkl',X28)
    
    #Contraction 2439; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('bcdaijkl->abcdijkl',X28)
    
    #Contraction 2440; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('bdacijkl->abcdijkl',X28)
    
    #Contraction 2441; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('bdcaijkl->abcdijkl',X28)
    
    #Contraction 2442; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('cabdijkl->abcdijkl',X28)
    
    #Contraction 2443; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('cadbijkl->abcdijkl',X28)
    
    #Contraction 2444; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('cbadijkl->abcdijkl',X28)
    
    #Contraction 2445; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('cbdaijkl->abcdijkl',X28)
    
    #Contraction 2446; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('cdabijkl->abcdijkl',X28)
    
    #Contraction 2447; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('cdbaijkl->abcdijkl',X28)
    
    #Contraction 2448; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('dabcijkl->abcdijkl',X28)
    
    #Contraction 2449; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('dacbijkl->abcdijkl',X28)
    
    #Contraction 2450; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('dbacijkl->abcdijkl',X28)
    
    #Contraction 2451; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('dbcaijkl->abcdijkl',X28)
    
    #Contraction 2452; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('dcabijkl->abcdijkl',X28)
    
    #Contraction 2453; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('dcbaijkl->abcdijkl',X28)
    
    #Contraction 2454; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('abcdilkj->abcdijkl',X28)
    
    #Contraction 2455; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('abdcilkj->abcdijkl',X28)
    
    #Contraction 2456; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('acbdilkj->abcdijkl',X28)
    
    #Contraction 2457; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('acdbilkj->abcdijkl',X28)
    
    #Contraction 2458; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('adbcilkj->abcdijkl',X28)
    
    #Contraction 2459; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('adcbilkj->abcdijkl',X28)
    
    #Contraction 2460; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('bacdilkj->abcdijkl',X28)
    
    #Contraction 2461; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('badcilkj->abcdijkl',X28)
    
    #Contraction 2462; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('bcadilkj->abcdijkl',X28)
    
    #Contraction 2463; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('bcdailkj->abcdijkl',X28)
    
    #Contraction 2464; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('bdacilkj->abcdijkl',X28)
    
    #Contraction 2465; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('bdcailkj->abcdijkl',X28)
    
    #Contraction 2466; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('cabdilkj->abcdijkl',X28)
    
    #Contraction 2467; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('cadbilkj->abcdijkl',X28)
    
    #Contraction 2468; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('cbadilkj->abcdijkl',X28)
    
    #Contraction 2469; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('cbdailkj->abcdijkl',X28)
    
    #Contraction 2470; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('cdabilkj->abcdijkl',X28)
    
    #Contraction 2471; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('cdbailkj->abcdijkl',X28)
    
    #Contraction 2472; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('dabcilkj->abcdijkl',X28)
    
    #Contraction 2473; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('dacbilkj->abcdijkl',X28)
    
    #Contraction 2474; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('dbacilkj->abcdijkl',X28)
    
    #Contraction 2475; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('dbcailkj->abcdijkl',X28)
    
    #Contraction 2476; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('dcabilkj->abcdijkl',X28)
    
    #Contraction 2477; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('dcbailkj->abcdijkl',X28)
    
    #Contraction 2478; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('abcdkjil->abcdijkl',X28)
    
    #Contraction 2479; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('abdckjil->abcdijkl',X28)
    
    #Contraction 2480; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('acbdkjil->abcdijkl',X28)
    
    #Contraction 2481; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('acdbkjil->abcdijkl',X28)
    
    #Contraction 2482; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('adbckjil->abcdijkl',X28)
    
    #Contraction 2483; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('adcbkjil->abcdijkl',X28)
    
    #Contraction 2484; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('bacdkjil->abcdijkl',X28)
    
    #Contraction 2485; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('badckjil->abcdijkl',X28)
    
    #Contraction 2486; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('bcadkjil->abcdijkl',X28)
    
    #Contraction 2487; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('bcdakjil->abcdijkl',X28)
    
    #Contraction 2488; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('bdackjil->abcdijkl',X28)
    
    #Contraction 2489; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('bdcakjil->abcdijkl',X28)
    
    #Contraction 2490; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('cabdkjil->abcdijkl',X28)
    
    #Contraction 2491; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('cadbkjil->abcdijkl',X28)
    
    #Contraction 2492; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('cbadkjil->abcdijkl',X28)
    
    #Contraction 2493; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('cbdakjil->abcdijkl',X28)
    
    #Contraction 2494; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('cdabkjil->abcdijkl',X28)
    
    #Contraction 2495; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('cdbakjil->abcdijkl',X28)
    
    #Contraction 2496; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('dabckjil->abcdijkl',X28)
    
    #Contraction 2497; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('dacbkjil->abcdijkl',X28)
    
    #Contraction 2498; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('dbackjil->abcdijkl',X28)
    
    #Contraction 2499; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('dbcakjil->abcdijkl',X28)
    
    #Contraction 2500; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('dcabkjil->abcdijkl',X28)
    
    #Contraction 2501; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('dcbakjil->abcdijkl',X28)
    
    #Contraction 2502; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('abcdklij->abcdijkl',X28)
    
    #Contraction 2503; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('abdcklij->abcdijkl',X28)
    
    #Contraction 2504; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('acbdklij->abcdijkl',X28)
    
    #Contraction 2505; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('acdbklij->abcdijkl',X28)
    
    #Contraction 2506; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('adbcklij->abcdijkl',X28)
    
    #Contraction 2507; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('adcbklij->abcdijkl',X28)
    
    #Contraction 2508; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('bacdklij->abcdijkl',X28)
    
    #Contraction 2509; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('badcklij->abcdijkl',X28)
    
    #Contraction 2510; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('bcadklij->abcdijkl',X28)
    
    #Contraction 2511; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('bcdaklij->abcdijkl',X28)
    
    #Contraction 2512; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('bdacklij->abcdijkl',X28)
    
    #Contraction 2513; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('bdcaklij->abcdijkl',X28)
    
    #Contraction 2514; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('cabdklij->abcdijkl',X28)
    
    #Contraction 2515; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('cadbklij->abcdijkl',X28)
    
    #Contraction 2516; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('cbadklij->abcdijkl',X28)
    
    #Contraction 2517; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('cbdaklij->abcdijkl',X28)
    
    #Contraction 2518; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('cdabklij->abcdijkl',X28)
    
    #Contraction 2519; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('cdbaklij->abcdijkl',X28)
    
    #Contraction 2520; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('dabcklij->abcdijkl',X28)
    
    #Contraction 2521; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('dacbklij->abcdijkl',X28)
    
    #Contraction 2522; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('dbacklij->abcdijkl',X28)
    
    #Contraction 2523; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('dbcaklij->abcdijkl',X28)
    
    #Contraction 2524; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('dcabklij->abcdijkl',X28)
    
    #Contraction 2525; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('dcbaklij->abcdijkl',X28)
    
    #Contraction 2526; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('abcdjlik->abcdijkl',X28)
    
    #Contraction 2527; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('abdcjlik->abcdijkl',X28)
    
    #Contraction 2528; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('acbdjlik->abcdijkl',X28)
    
    #Contraction 2529; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('acdbjlik->abcdijkl',X28)
    
    #Contraction 2530; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('adbcjlik->abcdijkl',X28)
    
    #Contraction 2531; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('adcbjlik->abcdijkl',X28)
    
    #Contraction 2532; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('bacdjlik->abcdijkl',X28)
    
    #Contraction 2533; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('badcjlik->abcdijkl',X28)
    
    #Contraction 2534; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('bcadjlik->abcdijkl',X28)
    
    #Contraction 2535; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('bcdajlik->abcdijkl',X28)
    
    #Contraction 2536; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('bdacjlik->abcdijkl',X28)
    
    #Contraction 2537; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('bdcajlik->abcdijkl',X28)
    
    #Contraction 2538; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('cabdjlik->abcdijkl',X28)
    
    #Contraction 2539; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('cadbjlik->abcdijkl',X28)
    
    #Contraction 2540; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('cbadjlik->abcdijkl',X28)
    
    #Contraction 2541; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('cbdajlik->abcdijkl',X28)
    
    #Contraction 2542; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('cdabjlik->abcdijkl',X28)
    
    #Contraction 2543; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('cdbajlik->abcdijkl',X28)
    
    #Contraction 2544; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('dabcjlik->abcdijkl',X28)
    
    #Contraction 2545; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('dacbjlik->abcdijkl',X28)
    
    #Contraction 2546; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('dbacjlik->abcdijkl',X28)
    
    #Contraction 2547; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('dbcajlik->abcdijkl',X28)
    
    #Contraction 2548; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('dcabjlik->abcdijkl',X28)
    
    #Contraction 2549; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('dcbajlik->abcdijkl',X28)
    
    del X28
    
    Y28 = np.zeros([nvir, nvir, nvir, nvir, nocc, nocc], dtype=type_)
    
    #Contraction 2550; Tree Level  3; Scaling  6/ 4 Result_size  2/ 4
    Y28 += np.einsum('bcmn,mndekl->bcdekl',T2,I28, optimize='optimal')
    
    del I28
    
    A29 = np.zeros([nvir, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 2551; Tree Level  2; Scaling  6/ 4 Result_size  4/ 4
    A29 += np.einsum('am,bcdmijkl->abcdijkl',T1,A28, optimize='optimal')
    
    del A28
    
    #Contraction 2552; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('abcdijkl->abcdijkl',A29)
    
    #Contraction 2553; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bacdijkl->abcdijkl',A29)
    
    #Contraction 2554; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('cabdijkl->abcdijkl',A29)
    
    #Contraction 2555; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('dabcijkl->abcdijkl',A29)
    
    del A29
    
    D29 = np.zeros([nvir, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 2556; Tree Level  2; Scaling  6/ 4 Result_size  4/ 4
    D29 += np.einsum('acdnkl,bnji->acdbklji',T3,Y21, optimize='optimal')
    
    del Y21
    
    #Contraction 2557; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('acdbklji->abcdijkl',D29)
    
    #Contraction 2558; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bcdaklji->abcdijkl',D29)
    
    #Contraction 2559; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('badcklji->abcdijkl',D29)
    
    #Contraction 2560; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bacdklji->abcdijkl',D29)
    
    #Contraction 2561; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('acdbjlki->abcdijkl',D29)
    
    #Contraction 2562; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bcdajlki->abcdijkl',D29)
    
    #Contraction 2563; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('badcjlki->abcdijkl',D29)
    
    #Contraction 2564; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bacdjlki->abcdijkl',D29)
    
    #Contraction 2565; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('acdbjkli->abcdijkl',D29)
    
    #Contraction 2566; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bcdajkli->abcdijkl',D29)
    
    #Contraction 2567; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('badcjkli->abcdijkl',D29)
    
    #Contraction 2568; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bacdjkli->abcdijkl',D29)
    
    #Contraction 2569; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('acdbklij->abcdijkl',D29)
    
    #Contraction 2570; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bcdaklij->abcdijkl',D29)
    
    #Contraction 2571; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('badcklij->abcdijkl',D29)
    
    #Contraction 2572; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bacdklij->abcdijkl',D29)
    
    #Contraction 2573; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('acdbilkj->abcdijkl',D29)
    
    #Contraction 2574; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bcdailkj->abcdijkl',D29)
    
    #Contraction 2575; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('badcilkj->abcdijkl',D29)
    
    #Contraction 2576; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bacdilkj->abcdijkl',D29)
    
    #Contraction 2577; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('acdbiklj->abcdijkl',D29)
    
    #Contraction 2578; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bcdaiklj->abcdijkl',D29)
    
    #Contraction 2579; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('badciklj->abcdijkl',D29)
    
    #Contraction 2580; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bacdiklj->abcdijkl',D29)
    
    #Contraction 2581; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('acdbjlik->abcdijkl',D29)
    
    #Contraction 2582; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bcdajlik->abcdijkl',D29)
    
    #Contraction 2583; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('badcjlik->abcdijkl',D29)
    
    #Contraction 2584; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bacdjlik->abcdijkl',D29)
    
    #Contraction 2585; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('acdbiljk->abcdijkl',D29)
    
    #Contraction 2586; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bcdailjk->abcdijkl',D29)
    
    #Contraction 2587; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('badciljk->abcdijkl',D29)
    
    #Contraction 2588; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bacdiljk->abcdijkl',D29)
    
    #Contraction 2589; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('acdbijlk->abcdijkl',D29)
    
    #Contraction 2590; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bcdaijlk->abcdijkl',D29)
    
    #Contraction 2591; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('badcijlk->abcdijkl',D29)
    
    #Contraction 2592; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bacdijlk->abcdijkl',D29)
    
    #Contraction 2593; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('acdbjkil->abcdijkl',D29)
    
    #Contraction 2594; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bcdajkil->abcdijkl',D29)
    
    #Contraction 2595; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('badcjkil->abcdijkl',D29)
    
    #Contraction 2596; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bacdjkil->abcdijkl',D29)
    
    #Contraction 2597; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('acdbikjl->abcdijkl',D29)
    
    #Contraction 2598; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bcdaikjl->abcdijkl',D29)
    
    #Contraction 2599; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('badcikjl->abcdijkl',D29)
    
    #Contraction 2600; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bacdikjl->abcdijkl',D29)
    
    #Contraction 2601; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('acdbijkl->abcdijkl',D29)
    
    #Contraction 2602; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bcdaijkl->abcdijkl',D29)
    
    #Contraction 2603; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('badcijkl->abcdijkl',D29)
    
    #Contraction 2604; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bacdijkl->abcdijkl',D29)
    
    del D29
    
    E29 = np.zeros([nocc, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 2605; Tree Level  3; Scaling  5/ 5 Result_size  5/ 3
    E29 += np.einsum('me,ebcdijkl->mbcdijkl',F3,T4, optimize='optimal')
    
    #del F3
    
    G29 = np.zeros([nvir, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 2606; Tree Level  2; Scaling  6/ 4 Result_size  4/ 4
    G29 += np.einsum('am,mbcdijkl->abcdijkl',T1,E29, optimize='optimal')
    
    del E29
    
    #Contraction 2607; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('abcdijkl->abcdijkl',G29)
    
    #Contraction 2608; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bacdijkl->abcdijkl',G29)
    
    #Contraction 2609; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('cabdijkl->abcdijkl',G29)
    
    #Contraction 2610; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('dabcijkl->abcdijkl',G29)
    
    del G29
    
    I29 = np.zeros([nvir, nvir, nvir, nvir, nocc, nocc], dtype=type_)
    
    #Contraction 2611; Tree Level  3; Scaling  6/ 6 Result_size  2/ 4
    I29 += np.einsum('mnef,fbcdmnkl->bcdekl',V7,T4, optimize='optimal')
    
    J29 = np.zeros([nvir, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 2612; Tree Level  2; Scaling  4/ 6 Result_size  4/ 4
    J29 += np.einsum('aeij,bcdekl->abcdijkl',T2,I29, optimize='optimal')
    
    del I29
    
    #Contraction 2613; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('abcdijkl->abcdijkl',J29)
    
    #Contraction 2614; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('bacdijkl->abcdijkl',J29)
    
    #Contraction 2615; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('cabdijkl->abcdijkl',J29)
    
    #Contraction 2616; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('dabcijkl->abcdijkl',J29)
    
    #Contraction 2617; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('abcdikjl->abcdijkl',J29)
    
    #Contraction 2618; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('bacdikjl->abcdijkl',J29)
    
    #Contraction 2619; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('cabdikjl->abcdijkl',J29)
    
    #Contraction 2620; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('dabcikjl->abcdijkl',J29)
    
    #Contraction 2621; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('abcdiljk->abcdijkl',J29)
    
    #Contraction 2622; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('bacdiljk->abcdijkl',J29)
    
    #Contraction 2623; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('cabdiljk->abcdijkl',J29)
    
    #Contraction 2624; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('dabciljk->abcdijkl',J29)
    
    #Contraction 2625; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('abcdjkil->abcdijkl',J29)
    
    #Contraction 2626; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('bacdjkil->abcdijkl',J29)
    
    #Contraction 2627; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('cabdjkil->abcdijkl',J29)
    
    #Contraction 2628; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('dabcjkil->abcdijkl',J29)
    
    #Contraction 2629; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('abcdjlik->abcdijkl',J29)
    
    #Contraction 2630; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('bacdjlik->abcdijkl',J29)
    
    #Contraction 2631; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('cabdjlik->abcdijkl',J29)
    
    #Contraction 2632; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('dabcjlik->abcdijkl',J29)
    
    #Contraction 2633; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('abcdklij->abcdijkl',J29)
    
    #Contraction 2634; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('bacdklij->abcdijkl',J29)
    
    #Contraction 2635; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('cabdklij->abcdijkl',J29)
    
    #Contraction 2636; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('dabcklij->abcdijkl',J29)
    
    del J29
    
    M29 = np.zeros([nocc, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 2637; Tree Level  3; Scaling  7/ 5 Result_size  5/ 3
    M29 += np.einsum('mnie,ebcdnjkl->mbcdijkl',V4,T4, optimize='optimal')
    
    #del V4
    
    X29 = np.zeros([nvir, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 2638; Tree Level  2; Scaling  6/ 4 Result_size  4/ 4
    X29 += np.einsum('am,mbcdijkl->abcdijkl',T1,M29, optimize='optimal')
    
    del M29
    
    #Contraction 2639; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('abcdijkl->abcdijkl',X29)
    
    #Contraction 2640; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bacdijkl->abcdijkl',X29)
    
    #Contraction 2641; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('cabdijkl->abcdijkl',X29)
    
    #Contraction 2642; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('dabcijkl->abcdijkl',X29)
    
    #Contraction 2643; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('abcdjikl->abcdijkl',X29)
    
    #Contraction 2644; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bacdjikl->abcdijkl',X29)
    
    #Contraction 2645; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('cabdjikl->abcdijkl',X29)
    
    #Contraction 2646; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('dabcjikl->abcdijkl',X29)
    
    #Contraction 2647; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('abcdkijl->abcdijkl',X29)
    
    #Contraction 2648; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bacdkijl->abcdijkl',X29)
    
    #Contraction 2649; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('cabdkijl->abcdijkl',X29)
    
    #Contraction 2650; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('dabckijl->abcdijkl',X29)
    
    #Contraction 2651; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('abcdlijk->abcdijkl',X29)
    
    #Contraction 2652; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bacdlijk->abcdijkl',X29)
    
    #Contraction 2653; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('cabdlijk->abcdijkl',X29)
    
    #Contraction 2654; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('dabclijk->abcdijkl',X29)
    
    del X29
    
    Y29 = np.zeros([nvir, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 2655; Tree Level  2; Scaling  6/ 6 Result_size  4/ 4
    Y29 += np.einsum('fbcdnjkl,nafi->bcdajkli',T4,G17, optimize='optimal')
    
    del G17
    
    #Contraction 2656; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bcdajkli->abcdijkl',Y29)
    
    #Contraction 2657; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('acdbjkli->abcdijkl',Y29)
    
    #Contraction 2658; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('abdcjkli->abcdijkl',Y29)
    
    #Contraction 2659; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('abcdjkli->abcdijkl',Y29)
    
    #Contraction 2660; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bcdaiklj->abcdijkl',Y29)
    
    #Contraction 2661; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('acdbiklj->abcdijkl',Y29)
    
    #Contraction 2662; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('abdciklj->abcdijkl',Y29)
    
    #Contraction 2663; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('abcdiklj->abcdijkl',Y29)
    
    #Contraction 2664; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bcdaijlk->abcdijkl',Y29)
    
    #Contraction 2665; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('acdbijlk->abcdijkl',Y29)
    
    #Contraction 2666; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('abdcijlk->abcdijkl',Y29)
    
    #Contraction 2667; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('abcdijlk->abcdijkl',Y29)
    
    #Contraction 2668; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bcdaijkl->abcdijkl',Y29)
    
    #Contraction 2669; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('acdbijkl->abcdijkl',Y29)
    
    #Contraction 2670; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('abdcijkl->abcdijkl',Y29)
    
    #Contraction 2671; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('abcdijkl->abcdijkl',Y29)
    
    del Y29
    
    A30 = np.zeros([nvir, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 2672; Tree Level  2; Scaling  4/ 6 Result_size  4/ 4
    A30 += np.einsum('fcdjkl,abfi->cdabjkli',T3,A20, optimize='optimal')
    
    del A20
    
    #Contraction 2673; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('cdabjkli->abcdijkl',A30)
    
    #Contraction 2674; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bdacjkli->abcdijkl',A30)
    
    #Contraction 2675; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bcadjkli->abcdijkl',A30)
    
    #Contraction 2676; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('adbcjkli->abcdijkl',A30)
    
    #Contraction 2677; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('acbdjkli->abcdijkl',A30)
    
    #Contraction 2678; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('abcdjkli->abcdijkl',A30)
    
    #Contraction 2679; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('cdabiklj->abcdijkl',A30)
    
    #Contraction 2680; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bdaciklj->abcdijkl',A30)
    
    #Contraction 2681; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bcadiklj->abcdijkl',A30)
    
    #Contraction 2682; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('adbciklj->abcdijkl',A30)
    
    #Contraction 2683; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('acbdiklj->abcdijkl',A30)
    
    #Contraction 2684; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('abcdiklj->abcdijkl',A30)
    
    #Contraction 2685; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('cdabijlk->abcdijkl',A30)
    
    #Contraction 2686; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bdacijlk->abcdijkl',A30)
    
    #Contraction 2687; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bcadijlk->abcdijkl',A30)
    
    #Contraction 2688; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('adbcijlk->abcdijkl',A30)
    
    #Contraction 2689; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('acbdijlk->abcdijkl',A30)
    
    #Contraction 2690; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('abcdijlk->abcdijkl',A30)
    
    #Contraction 2691; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('cdabijkl->abcdijkl',A30)
    
    #Contraction 2692; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bdacijkl->abcdijkl',A30)
    
    #Contraction 2693; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bcadijkl->abcdijkl',A30)
    
    #Contraction 2694; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('adbcijkl->abcdijkl',A30)
    
    #Contraction 2695; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('acbdijkl->abcdijkl',A30)
    
    #Contraction 2696; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('abcdijkl->abcdijkl',A30)
    
    del A30
    
    D30 = np.zeros([nvir, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 2697; Tree Level  2; Scaling  4/ 6 Result_size  4/ 4
    D30 += np.einsum('aeij,bcdekl->abcdijkl',T2,Y28, optimize='optimal')
    
    del Y28
    
    #Contraction 2698; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('abcdijkl->abcdijkl',D30)
    
    #Contraction 2699; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('adcbijkl->abcdijkl',D30)
    
    #Contraction 2700; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('adbcijkl->abcdijkl',D30)
    
    #Contraction 2701; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('dbcaijkl->abcdijkl',D30)
    
    #Contraction 2702; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('dacbijkl->abcdijkl',D30)
    
    #Contraction 2703; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('dabcijkl->abcdijkl',D30)
    
    #Contraction 2704; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('bdcaijkl->abcdijkl',D30)
    
    #Contraction 2705; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('bacdijkl->abcdijkl',D30)
    
    #Contraction 2706; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('badcijkl->abcdijkl',D30)
    
    #Contraction 2707; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('cdbaijkl->abcdijkl',D30)
    
    #Contraction 2708; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('cabdijkl->abcdijkl',D30)
    
    #Contraction 2709; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('cadbijkl->abcdijkl',D30)
    
    #Contraction 2710; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('abcdikjl->abcdijkl',D30)
    
    #Contraction 2711; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('adcbikjl->abcdijkl',D30)
    
    #Contraction 2712; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('adbcikjl->abcdijkl',D30)
    
    #Contraction 2713; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('dbcaikjl->abcdijkl',D30)
    
    #Contraction 2714; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('dacbikjl->abcdijkl',D30)
    
    #Contraction 2715; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('dabcikjl->abcdijkl',D30)
    
    #Contraction 2716; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('bdcaikjl->abcdijkl',D30)
    
    #Contraction 2717; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('bacdikjl->abcdijkl',D30)
    
    #Contraction 2718; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('badcikjl->abcdijkl',D30)
    
    #Contraction 2719; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('cdbaikjl->abcdijkl',D30)
    
    #Contraction 2720; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('cabdikjl->abcdijkl',D30)
    
    #Contraction 2721; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('cadbikjl->abcdijkl',D30)
    
    #Contraction 2722; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('abcdiljk->abcdijkl',D30)
    
    #Contraction 2723; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('adcbiljk->abcdijkl',D30)
    
    #Contraction 2724; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('adbciljk->abcdijkl',D30)
    
    #Contraction 2725; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('dbcailjk->abcdijkl',D30)
    
    #Contraction 2726; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('dacbiljk->abcdijkl',D30)
    
    #Contraction 2727; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('dabciljk->abcdijkl',D30)
    
    #Contraction 2728; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('bdcailjk->abcdijkl',D30)
    
    #Contraction 2729; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('bacdiljk->abcdijkl',D30)
    
    #Contraction 2730; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('badciljk->abcdijkl',D30)
    
    #Contraction 2731; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('cdbailjk->abcdijkl',D30)
    
    #Contraction 2732; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('cabdiljk->abcdijkl',D30)
    
    #Contraction 2733; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('cadbiljk->abcdijkl',D30)
    
    #Contraction 2734; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('abcdjkil->abcdijkl',D30)
    
    #Contraction 2735; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('adcbjkil->abcdijkl',D30)
    
    #Contraction 2736; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('adbcjkil->abcdijkl',D30)
    
    #Contraction 2737; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('dbcajkil->abcdijkl',D30)
    
    #Contraction 2738; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('dacbjkil->abcdijkl',D30)
    
    #Contraction 2739; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('dabcjkil->abcdijkl',D30)
    
    #Contraction 2740; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('bdcajkil->abcdijkl',D30)
    
    #Contraction 2741; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('bacdjkil->abcdijkl',D30)
    
    #Contraction 2742; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('badcjkil->abcdijkl',D30)
    
    #Contraction 2743; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('cdbajkil->abcdijkl',D30)
    
    #Contraction 2744; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('cabdjkil->abcdijkl',D30)
    
    #Contraction 2745; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('cadbjkil->abcdijkl',D30)
    
    #Contraction 2746; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('abcdjlik->abcdijkl',D30)
    
    #Contraction 2747; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('adcbjlik->abcdijkl',D30)
    
    #Contraction 2748; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('adbcjlik->abcdijkl',D30)
    
    #Contraction 2749; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('dbcajlik->abcdijkl',D30)
    
    #Contraction 2750; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('dacbjlik->abcdijkl',D30)
    
    #Contraction 2751; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('dabcjlik->abcdijkl',D30)
    
    #Contraction 2752; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('bdcajlik->abcdijkl',D30)
    
    #Contraction 2753; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('bacdjlik->abcdijkl',D30)
    
    #Contraction 2754; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('badcjlik->abcdijkl',D30)
    
    #Contraction 2755; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('cdbajlik->abcdijkl',D30)
    
    #Contraction 2756; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('cabdjlik->abcdijkl',D30)
    
    #Contraction 2757; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('cadbjlik->abcdijkl',D30)
    
    #Contraction 2758; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('abcdklij->abcdijkl',D30)
    
    #Contraction 2759; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('adcbklij->abcdijkl',D30)
    
    #Contraction 2760; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('adbcklij->abcdijkl',D30)
    
    #Contraction 2761; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('dbcaklij->abcdijkl',D30)
    
    #Contraction 2762; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('dacbklij->abcdijkl',D30)
    
    #Contraction 2763; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('dabcklij->abcdijkl',D30)
    
    #Contraction 2764; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('bdcaklij->abcdijkl',D30)
    
    #Contraction 2765; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('bacdklij->abcdijkl',D30)
    
    #Contraction 2766; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('badcklij->abcdijkl',D30)
    
    #Contraction 2767; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('cdbaklij->abcdijkl',D30)
    
    #Contraction 2768; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('cabdklij->abcdijkl',D30)
    
    #Contraction 2769; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('cadbklij->abcdijkl',D30)
    
    del D30
    
    E30 = np.zeros([nvir, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 2770; Tree Level  2; Scaling  6/ 6 Result_size  4/ 4
    E30 += np.einsum('abeijm,mcdekl->abcdijkl',T3,I21, optimize='optimal')
    
    del I21
    
    #Contraction 2771; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('abcdijkl->abcdijkl',E30)
    
    #Contraction 2772; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('acbdijkl->abcdijkl',E30)
    
    #Contraction 2773; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('adbcijkl->abcdijkl',E30)
    
    #Contraction 2774; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('bcadijkl->abcdijkl',E30)
    
    #Contraction 2775; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('bdacijkl->abcdijkl',E30)
    
    #Contraction 2776; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('cdabijkl->abcdijkl',E30)
    
    #Contraction 2777; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('abcdikjl->abcdijkl',E30)
    
    #Contraction 2778; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('acbdikjl->abcdijkl',E30)
    
    #Contraction 2779; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('adbcikjl->abcdijkl',E30)
    
    #Contraction 2780; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('bcadikjl->abcdijkl',E30)
    
    #Contraction 2781; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('bdacikjl->abcdijkl',E30)
    
    #Contraction 2782; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('cdabikjl->abcdijkl',E30)
    
    #Contraction 2783; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('abcdiljk->abcdijkl',E30)
    
    #Contraction 2784; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('acbdiljk->abcdijkl',E30)
    
    #Contraction 2785; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('adbciljk->abcdijkl',E30)
    
    #Contraction 2786; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('bcadiljk->abcdijkl',E30)
    
    #Contraction 2787; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('bdaciljk->abcdijkl',E30)
    
    #Contraction 2788; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('cdabiljk->abcdijkl',E30)
    
    #Contraction 2789; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('abcdjkil->abcdijkl',E30)
    
    #Contraction 2790; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('acbdjkil->abcdijkl',E30)
    
    #Contraction 2791; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('adbcjkil->abcdijkl',E30)
    
    #Contraction 2792; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('bcadjkil->abcdijkl',E30)
    
    #Contraction 2793; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('bdacjkil->abcdijkl',E30)
    
    #Contraction 2794; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('cdabjkil->abcdijkl',E30)
    
    #Contraction 2795; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('abcdjlik->abcdijkl',E30)
    
    #Contraction 2796; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('acbdjlik->abcdijkl',E30)
    
    #Contraction 2797; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('adbcjlik->abcdijkl',E30)
    
    #Contraction 2798; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('bcadjlik->abcdijkl',E30)
    
    #Contraction 2799; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('bdacjlik->abcdijkl',E30)
    
    #Contraction 2800; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('cdabjlik->abcdijkl',E30)
    
    #Contraction 2801; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('abcdklij->abcdijkl',E30)
    
    #Contraction 2802; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('acbdklij->abcdijkl',E30)
    
    #Contraction 2803; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('adbcklij->abcdijkl',E30)
    
    #Contraction 2804; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('bcadklij->abcdijkl',E30)
    
    #Contraction 2805; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('bdacklij->abcdijkl',E30)
    
    #Contraction 2806; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('cdabklij->abcdijkl',E30)
    
    del E30
    
    G30 = np.zeros([nvir, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 2807; Tree Level  2; Scaling  6/ 4 Result_size  4/ 4
    G30 += np.einsum('bcjm,amdikl->bcadjikl',T2,D14, optimize='optimal')
    
    del D14
    
    #Contraction 2808; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('bcadjikl->abcdijkl',G30)
    
    #Contraction 2809; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('bdacjikl->abcdijkl',G30)
    
    #Contraction 2810; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('cdabjikl->abcdijkl',G30)
    
    #Contraction 2811; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('acbdjikl->abcdijkl',G30)
    
    #Contraction 2812; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('adbcjikl->abcdijkl',G30)
    
    #Contraction 2813; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('cdbajikl->abcdijkl',G30)
    
    #Contraction 2814; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('abcdjikl->abcdijkl',G30)
    
    #Contraction 2815; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('adcbjikl->abcdijkl',G30)
    
    #Contraction 2816; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('bdcajikl->abcdijkl',G30)
    
    #Contraction 2817; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('abdcjikl->abcdijkl',G30)
    
    #Contraction 2818; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('acdbjikl->abcdijkl',G30)
    
    #Contraction 2819; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('bcdajikl->abcdijkl',G30)
    
    #Contraction 2820; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('bcadijkl->abcdijkl',G30)
    
    #Contraction 2821; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('bdacijkl->abcdijkl',G30)
    
    #Contraction 2822; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('cdabijkl->abcdijkl',G30)
    
    #Contraction 2823; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('acbdijkl->abcdijkl',G30)
    
    #Contraction 2824; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('adbcijkl->abcdijkl',G30)
    
    #Contraction 2825; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('cdbaijkl->abcdijkl',G30)
    
    #Contraction 2826; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('abcdijkl->abcdijkl',G30)
    
    #Contraction 2827; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('adcbijkl->abcdijkl',G30)
    
    #Contraction 2828; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('bdcaijkl->abcdijkl',G30)
    
    #Contraction 2829; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('abdcijkl->abcdijkl',G30)
    
    #Contraction 2830; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('acdbijkl->abcdijkl',G30)
    
    #Contraction 2831; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('bcdaijkl->abcdijkl',G30)
    
    #Contraction 2832; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('bcadkjil->abcdijkl',G30)
    
    #Contraction 2833; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('bdackjil->abcdijkl',G30)
    
    #Contraction 2834; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('cdabkjil->abcdijkl',G30)
    
    #Contraction 2835; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('acbdkjil->abcdijkl',G30)
    
    #Contraction 2836; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('adbckjil->abcdijkl',G30)
    
    #Contraction 2837; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('cdbakjil->abcdijkl',G30)
    
    #Contraction 2838; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('abcdkjil->abcdijkl',G30)
    
    #Contraction 2839; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('adcbkjil->abcdijkl',G30)
    
    #Contraction 2840; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('bdcakjil->abcdijkl',G30)
    
    #Contraction 2841; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('abdckjil->abcdijkl',G30)
    
    #Contraction 2842; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('acdbkjil->abcdijkl',G30)
    
    #Contraction 2843; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('bcdakjil->abcdijkl',G30)
    
    #Contraction 2844; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('bcadljik->abcdijkl',G30)
    
    #Contraction 2845; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('bdacljik->abcdijkl',G30)
    
    #Contraction 2846; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('cdabljik->abcdijkl',G30)
    
    #Contraction 2847; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('acbdljik->abcdijkl',G30)
    
    #Contraction 2848; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('adbcljik->abcdijkl',G30)
    
    #Contraction 2849; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('cdbaljik->abcdijkl',G30)
    
    #Contraction 2850; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('abcdljik->abcdijkl',G30)
    
    #Contraction 2851; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('adcbljik->abcdijkl',G30)
    
    #Contraction 2852; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('bdcaljik->abcdijkl',G30)
    
    #Contraction 2853; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('abdcljik->abcdijkl',G30)
    
    #Contraction 2854; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('acdbljik->abcdijkl',G30)
    
    #Contraction 2855; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('bcdaljik->abcdijkl',G30)
    
    del G30
    
    I30 = np.zeros([nvir, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 2856; Tree Level  2; Scaling  6/ 4 Result_size  4/ 4
    I30 += np.einsum('bm,cdamjkli->bcdajkli',T1,G21, optimize='optimal')
    
    del G21
    
    #Contraction 2857; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bcdajkli->abcdijkl',I30)
    
    #Contraction 2858; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('cbdajkli->abcdijkl',I30)
    
    #Contraction 2859; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('dbcajkli->abcdijkl',I30)
    
    #Contraction 2860; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('acdbjkli->abcdijkl',I30)
    
    #Contraction 2861; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('cadbjkli->abcdijkl',I30)
    
    #Contraction 2862; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('dacbjkli->abcdijkl',I30)
    
    #Contraction 2863; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('abdcjkli->abcdijkl',I30)
    
    #Contraction 2864; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('badcjkli->abcdijkl',I30)
    
    #Contraction 2865; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('dabcjkli->abcdijkl',I30)
    
    #Contraction 2866; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('abcdjkli->abcdijkl',I30)
    
    #Contraction 2867; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bacdjkli->abcdijkl',I30)
    
    #Contraction 2868; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('cabdjkli->abcdijkl',I30)
    
    #Contraction 2869; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bcdaiklj->abcdijkl',I30)
    
    #Contraction 2870; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('cbdaiklj->abcdijkl',I30)
    
    #Contraction 2871; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('dbcaiklj->abcdijkl',I30)
    
    #Contraction 2872; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('acdbiklj->abcdijkl',I30)
    
    #Contraction 2873; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('cadbiklj->abcdijkl',I30)
    
    #Contraction 2874; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('dacbiklj->abcdijkl',I30)
    
    #Contraction 2875; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('abdciklj->abcdijkl',I30)
    
    #Contraction 2876; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('badciklj->abcdijkl',I30)
    
    #Contraction 2877; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('dabciklj->abcdijkl',I30)
    
    #Contraction 2878; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('abcdiklj->abcdijkl',I30)
    
    #Contraction 2879; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bacdiklj->abcdijkl',I30)
    
    #Contraction 2880; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('cabdiklj->abcdijkl',I30)
    
    #Contraction 2881; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bcdaijlk->abcdijkl',I30)
    
    #Contraction 2882; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('cbdaijlk->abcdijkl',I30)
    
    #Contraction 2883; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('dbcaijlk->abcdijkl',I30)
    
    #Contraction 2884; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('acdbijlk->abcdijkl',I30)
    
    #Contraction 2885; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('cadbijlk->abcdijkl',I30)
    
    #Contraction 2886; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('dacbijlk->abcdijkl',I30)
    
    #Contraction 2887; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('abdcijlk->abcdijkl',I30)
    
    #Contraction 2888; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('badcijlk->abcdijkl',I30)
    
    #Contraction 2889; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('dabcijlk->abcdijkl',I30)
    
    #Contraction 2890; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('abcdijlk->abcdijkl',I30)
    
    #Contraction 2891; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bacdijlk->abcdijkl',I30)
    
    #Contraction 2892; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('cabdijlk->abcdijkl',I30)
    
    #Contraction 2893; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bcdaijkl->abcdijkl',I30)
    
    #Contraction 2894; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('cbdaijkl->abcdijkl',I30)
    
    #Contraction 2895; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('dbcaijkl->abcdijkl',I30)
    
    #Contraction 2896; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('acdbijkl->abcdijkl',I30)
    
    #Contraction 2897; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('cadbijkl->abcdijkl',I30)
    
    #Contraction 2898; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('dacbijkl->abcdijkl',I30)
    
    #Contraction 2899; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('abdcijkl->abcdijkl',I30)
    
    #Contraction 2900; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('badcijkl->abcdijkl',I30)
    
    #Contraction 2901; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('dabcijkl->abcdijkl',I30)
    
    #Contraction 2902; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('abcdijkl->abcdijkl',I30)
    
    #Contraction 2903; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bacdijkl->abcdijkl',I30)
    
    #Contraction 2904; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('cabdijkl->abcdijkl',I30)
    
    del I30
    
    J30 = np.zeros([nvir, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 2905; Tree Level  2; Scaling  6/ 4 Result_size  4/ 4
    J30 += np.einsum('abmj,cdmkli->abcdjkli',T2,J12, optimize='optimal')
    
    del J12
    
    #Contraction 2906; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('abcdjkli->abcdijkl',J30)
    
    #Contraction 2907; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('acbdjkli->abcdijkl',J30)
    
    #Contraction 2908; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('adbcjkli->abcdijkl',J30)
    
    #Contraction 2909; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bcadjkli->abcdijkl',J30)
    
    #Contraction 2910; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bdacjkli->abcdijkl',J30)
    
    #Contraction 2911; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('cdabjkli->abcdijkl',J30)
    
    #Contraction 2912; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('abcdkjli->abcdijkl',J30)
    
    #Contraction 2913; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('acbdkjli->abcdijkl',J30)
    
    #Contraction 2914; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('adbckjli->abcdijkl',J30)
    
    #Contraction 2915; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bcadkjli->abcdijkl',J30)
    
    #Contraction 2916; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bdackjli->abcdijkl',J30)
    
    #Contraction 2917; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('cdabkjli->abcdijkl',J30)
    
    #Contraction 2918; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('abcdljki->abcdijkl',J30)
    
    #Contraction 2919; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('acbdljki->abcdijkl',J30)
    
    #Contraction 2920; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('adbcljki->abcdijkl',J30)
    
    #Contraction 2921; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bcadljki->abcdijkl',J30)
    
    #Contraction 2922; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bdacljki->abcdijkl',J30)
    
    #Contraction 2923; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('cdabljki->abcdijkl',J30)
    
    #Contraction 2924; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('abcdiklj->abcdijkl',J30)
    
    #Contraction 2925; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('acbdiklj->abcdijkl',J30)
    
    #Contraction 2926; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('adbciklj->abcdijkl',J30)
    
    #Contraction 2927; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bcadiklj->abcdijkl',J30)
    
    #Contraction 2928; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bdaciklj->abcdijkl',J30)
    
    #Contraction 2929; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('cdabiklj->abcdijkl',J30)
    
    #Contraction 2930; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('abcdkilj->abcdijkl',J30)
    
    #Contraction 2931; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('acbdkilj->abcdijkl',J30)
    
    #Contraction 2932; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('adbckilj->abcdijkl',J30)
    
    #Contraction 2933; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bcadkilj->abcdijkl',J30)
    
    #Contraction 2934; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bdackilj->abcdijkl',J30)
    
    #Contraction 2935; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('cdabkilj->abcdijkl',J30)
    
    #Contraction 2936; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('abcdlikj->abcdijkl',J30)
    
    #Contraction 2937; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('acbdlikj->abcdijkl',J30)
    
    #Contraction 2938; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('adbclikj->abcdijkl',J30)
    
    #Contraction 2939; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bcadlikj->abcdijkl',J30)
    
    #Contraction 2940; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bdaclikj->abcdijkl',J30)
    
    #Contraction 2941; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('cdablikj->abcdijkl',J30)
    
    #Contraction 2942; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('abcdijlk->abcdijkl',J30)
    
    #Contraction 2943; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('acbdijlk->abcdijkl',J30)
    
    #Contraction 2944; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('adbcijlk->abcdijkl',J30)
    
    #Contraction 2945; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bcadijlk->abcdijkl',J30)
    
    #Contraction 2946; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bdacijlk->abcdijkl',J30)
    
    #Contraction 2947; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('cdabijlk->abcdijkl',J30)
    
    #Contraction 2948; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('abcdjilk->abcdijkl',J30)
    
    #Contraction 2949; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('acbdjilk->abcdijkl',J30)
    
    #Contraction 2950; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('adbcjilk->abcdijkl',J30)
    
    #Contraction 2951; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bcadjilk->abcdijkl',J30)
    
    #Contraction 2952; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bdacjilk->abcdijkl',J30)
    
    #Contraction 2953; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('cdabjilk->abcdijkl',J30)
    
    #Contraction 2954; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('abcdlijk->abcdijkl',J30)
    
    #Contraction 2955; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('acbdlijk->abcdijkl',J30)
    
    #Contraction 2956; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('adbclijk->abcdijkl',J30)
    
    #Contraction 2957; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bcadlijk->abcdijkl',J30)
    
    #Contraction 2958; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bdaclijk->abcdijkl',J30)
    
    #Contraction 2959; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('cdablijk->abcdijkl',J30)
    
    #Contraction 2960; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('abcdijkl->abcdijkl',J30)
    
    #Contraction 2961; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('acbdijkl->abcdijkl',J30)
    
    #Contraction 2962; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('adbcijkl->abcdijkl',J30)
    
    #Contraction 2963; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bcadijkl->abcdijkl',J30)
    
    #Contraction 2964; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bdacijkl->abcdijkl',J30)
    
    #Contraction 2965; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('cdabijkl->abcdijkl',J30)
    
    #Contraction 2966; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('abcdjikl->abcdijkl',J30)
    
    #Contraction 2967; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('acbdjikl->abcdijkl',J30)
    
    #Contraction 2968; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('adbcjikl->abcdijkl',J30)
    
    #Contraction 2969; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bcadjikl->abcdijkl',J30)
    
    #Contraction 2970; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bdacjikl->abcdijkl',J30)
    
    #Contraction 2971; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('cdabjikl->abcdijkl',J30)
    
    #Contraction 2972; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('abcdkijl->abcdijkl',J30)
    
    #Contraction 2973; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('acbdkijl->abcdijkl',J30)
    
    #Contraction 2974; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('adbckijl->abcdijkl',J30)
    
    #Contraction 2975; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bcadkijl->abcdijkl',J30)
    
    #Contraction 2976; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bdackijl->abcdijkl',J30)
    
    #Contraction 2977; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('cdabkijl->abcdijkl',J30)
    
    del J30
    
    M30 = np.zeros([nvir, nvir, nocc, nvir, nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 2978; Tree Level  3; Scaling  5/ 5 Result_size  5/ 3
    M30 += np.einsum('fcdikl,mbfj->cdmbiklj',T3,X21, optimize='optimal')
    
    del X21
    
    X30 = np.zeros([nvir, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 2979; Tree Level  2; Scaling  6/ 4 Result_size  4/ 4
    X30 += np.einsum('am,cdmbiklj->acdbiklj',T1,M30, optimize='optimal')
    
    del M30
    
    #Contraction 2980; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('acdbiklj->abcdijkl',X30)
    
    #Contraction 2981; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('abdciklj->abcdijkl',X30)
    
    #Contraction 2982; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('abcdiklj->abcdijkl',X30)
    
    #Contraction 2983; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bcdaiklj->abcdijkl',X30)
    
    #Contraction 2984; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('badciklj->abcdijkl',X30)
    
    #Contraction 2985; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bacdiklj->abcdijkl',X30)
    
    #Contraction 2986; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('cbdaiklj->abcdijkl',X30)
    
    #Contraction 2987; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('cadbiklj->abcdijkl',X30)
    
    #Contraction 2988; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('cabdiklj->abcdijkl',X30)
    
    #Contraction 2989; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('dbcaiklj->abcdijkl',X30)
    
    #Contraction 2990; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('dacbiklj->abcdijkl',X30)
    
    #Contraction 2991; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('dabciklj->abcdijkl',X30)
    
    #Contraction 2992; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('acdbjkli->abcdijkl',X30)
    
    #Contraction 2993; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('abdcjkli->abcdijkl',X30)
    
    #Contraction 2994; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('abcdjkli->abcdijkl',X30)
    
    #Contraction 2995; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bcdajkli->abcdijkl',X30)
    
    #Contraction 2996; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('badcjkli->abcdijkl',X30)
    
    #Contraction 2997; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bacdjkli->abcdijkl',X30)
    
    #Contraction 2998; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('cbdajkli->abcdijkl',X30)
    
    #Contraction 2999; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('cadbjkli->abcdijkl',X30)
    
    #Contraction 3000; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('cabdjkli->abcdijkl',X30)
    
    #Contraction 3001; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('dbcajkli->abcdijkl',X30)
    
    #Contraction 3002; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('dacbjkli->abcdijkl',X30)
    
    #Contraction 3003; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('dabcjkli->abcdijkl',X30)
    
    #Contraction 3004; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('acdbjilk->abcdijkl',X30)
    
    #Contraction 3005; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('abdcjilk->abcdijkl',X30)
    
    #Contraction 3006; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('abcdjilk->abcdijkl',X30)
    
    #Contraction 3007; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bcdajilk->abcdijkl',X30)
    
    #Contraction 3008; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('badcjilk->abcdijkl',X30)
    
    #Contraction 3009; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bacdjilk->abcdijkl',X30)
    
    #Contraction 3010; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('cbdajilk->abcdijkl',X30)
    
    #Contraction 3011; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('cadbjilk->abcdijkl',X30)
    
    #Contraction 3012; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('cabdjilk->abcdijkl',X30)
    
    #Contraction 3013; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('dbcajilk->abcdijkl',X30)
    
    #Contraction 3014; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('dacbjilk->abcdijkl',X30)
    
    #Contraction 3015; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('dabcjilk->abcdijkl',X30)
    
    #Contraction 3016; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('acdbjikl->abcdijkl',X30)
    
    #Contraction 3017; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('abdcjikl->abcdijkl',X30)
    
    #Contraction 3018; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('abcdjikl->abcdijkl',X30)
    
    #Contraction 3019; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bcdajikl->abcdijkl',X30)
    
    #Contraction 3020; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('badcjikl->abcdijkl',X30)
    
    #Contraction 3021; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bacdjikl->abcdijkl',X30)
    
    #Contraction 3022; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('cbdajikl->abcdijkl',X30)
    
    #Contraction 3023; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('cadbjikl->abcdijkl',X30)
    
    #Contraction 3024; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('cabdjikl->abcdijkl',X30)
    
    #Contraction 3025; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('dbcajikl->abcdijkl',X30)
    
    #Contraction 3026; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('dacbjikl->abcdijkl',X30)
    
    #Contraction 3027; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('dabcjikl->abcdijkl',X30)
    
    del X30
    
    Y30 = np.zeros([nocc, nocc, nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 3028; Tree Level  3; Scaling  5/ 5 Result_size  5/ 1
    Y30 += np.einsum('mnef,aefijk->mnaijk',V7,T3, optimize='optimal')
    
    A31 = np.zeros([nvir, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 3029; Tree Level  2; Scaling  4/ 6 Result_size  4/ 4
    A31 += np.einsum('abeijk,cdel->abcdijkl',T3,I19, optimize='optimal')
    
    del I19
    
    #Contraction 3030; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('abcdijkl->abcdijkl',A31)
    
    #Contraction 3031; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('acbdijkl->abcdijkl',A31)
    
    #Contraction 3032; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('adbcijkl->abcdijkl',A31)
    
    #Contraction 3033; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('bcadijkl->abcdijkl',A31)
    
    #Contraction 3034; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('bdacijkl->abcdijkl',A31)
    
    #Contraction 3035; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('cdabijkl->abcdijkl',A31)
    
    #Contraction 3036; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('abcdijlk->abcdijkl',A31)
    
    #Contraction 3037; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('acbdijlk->abcdijkl',A31)
    
    #Contraction 3038; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('adbcijlk->abcdijkl',A31)
    
    #Contraction 3039; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('bcadijlk->abcdijkl',A31)
    
    #Contraction 3040; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('bdacijlk->abcdijkl',A31)
    
    #Contraction 3041; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('cdabijlk->abcdijkl',A31)
    
    #Contraction 3042; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('abcdiklj->abcdijkl',A31)
    
    #Contraction 3043; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('acbdiklj->abcdijkl',A31)
    
    #Contraction 3044; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('adbciklj->abcdijkl',A31)
    
    #Contraction 3045; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('bcadiklj->abcdijkl',A31)
    
    #Contraction 3046; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('bdaciklj->abcdijkl',A31)
    
    #Contraction 3047; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('cdabiklj->abcdijkl',A31)
    
    #Contraction 3048; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('abcdjkli->abcdijkl',A31)
    
    #Contraction 3049; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('acbdjkli->abcdijkl',A31)
    
    #Contraction 3050; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('adbcjkli->abcdijkl',A31)
    
    #Contraction 3051; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('bcadjkli->abcdijkl',A31)
    
    #Contraction 3052; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('bdacjkli->abcdijkl',A31)
    
    #Contraction 3053; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('cdabjkli->abcdijkl',A31)
    
    del A31
    
    D31 = np.zeros([nvir, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 3054; Tree Level  2; Scaling  6/ 4 Result_size  4/ 4
    D31 += np.einsum('amij,cdbklm->acdbijkl',V2,T3, optimize='optimal')
    
    #del V2
    
    #Contraction 3055; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('acdbijkl->abcdijkl',D31)
    
    #Contraction 3056; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('cadbijkl->abcdijkl',D31)
    
    #Contraction 3057; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('dacbijkl->abcdijkl',D31)
    
    #Contraction 3058; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bacdijkl->abcdijkl',D31)
    
    #Contraction 3059; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('acdbikjl->abcdijkl',D31)
    
    #Contraction 3060; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('cadbikjl->abcdijkl',D31)
    
    #Contraction 3061; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('dacbikjl->abcdijkl',D31)
    
    #Contraction 3062; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bacdikjl->abcdijkl',D31)
    
    #Contraction 3063; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('acdbiljk->abcdijkl',D31)
    
    #Contraction 3064; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('cadbiljk->abcdijkl',D31)
    
    #Contraction 3065; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('dacbiljk->abcdijkl',D31)
    
    #Contraction 3066; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bacdiljk->abcdijkl',D31)
    
    #Contraction 3067; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('acdbjkil->abcdijkl',D31)
    
    #Contraction 3068; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('cadbjkil->abcdijkl',D31)
    
    #Contraction 3069; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('dacbjkil->abcdijkl',D31)
    
    #Contraction 3070; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bacdjkil->abcdijkl',D31)
    
    #Contraction 3071; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('acdbjlik->abcdijkl',D31)
    
    #Contraction 3072; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('cadbjlik->abcdijkl',D31)
    
    #Contraction 3073; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('dacbjlik->abcdijkl',D31)
    
    #Contraction 3074; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bacdjlik->abcdijkl',D31)
    
    #Contraction 3075; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('acdbklij->abcdijkl',D31)
    
    #Contraction 3076; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('cadbklij->abcdijkl',D31)
    
    #Contraction 3077; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('dacbklij->abcdijkl',D31)
    
    #Contraction 3078; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bacdklij->abcdijkl',D31)
    
    del D31
    
    E31 = np.zeros([nvir, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 3079; Tree Level  2; Scaling  4/ 8 Result_size  4/ 4
    E31 += np.einsum('abef,cdefklij->abcdklij',V9,T4, optimize='optimal')
    
    #Contraction 3080; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('abcdklij->abcdijkl',E31)
    
    #Contraction 3081; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('acbdklij->abcdijkl',E31)
    
    #Contraction 3082; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('adbcklij->abcdijkl',E31)
    
    #Contraction 3083; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('bcadklij->abcdijkl',E31)
    
    #Contraction 3084; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('bdacklij->abcdijkl',E31)
    
    #Contraction 3085; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('cdabklij->abcdijkl',E31)
    
    del E31
    
    G31 = np.zeros([nvir, nvir, nvir, nocc], dtype=type_)
    
    #Contraction 3086; Tree Level  3; Scaling  5/ 3 Result_size  1/ 3
    G31 += np.einsum('abmn,mnfi->abfi',T2,I1, optimize='optimal')
    
    del I1
    
    I31 = np.zeros([nvir, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 3087; Tree Level  2; Scaling  4/ 6 Result_size  4/ 4
    I31 += np.einsum('fcdjkl,abfi->cdabjkli',T3,G31, optimize='optimal')
    
    del G31
    
    #Contraction 3088; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('cdabjkli->abcdijkl',I31)
    
    #Contraction 3089; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('bdacjkli->abcdijkl',I31)
    
    #Contraction 3090; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('bcadjkli->abcdijkl',I31)
    
    #Contraction 3091; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('adbcjkli->abcdijkl',I31)
    
    #Contraction 3092; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('acbdjkli->abcdijkl',I31)
    
    #Contraction 3093; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('abcdjkli->abcdijkl',I31)
    
    #Contraction 3094; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('cdabiklj->abcdijkl',I31)
    
    #Contraction 3095; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('bdaciklj->abcdijkl',I31)
    
    #Contraction 3096; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('bcadiklj->abcdijkl',I31)
    
    #Contraction 3097; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('adbciklj->abcdijkl',I31)
    
    #Contraction 3098; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('acbdiklj->abcdijkl',I31)
    
    #Contraction 3099; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('abcdiklj->abcdijkl',I31)
    
    #Contraction 3100; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('cdabijlk->abcdijkl',I31)
    
    #Contraction 3101; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('bdacijlk->abcdijkl',I31)
    
    #Contraction 3102; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('bcadijlk->abcdijkl',I31)
    
    #Contraction 3103; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('adbcijlk->abcdijkl',I31)
    
    #Contraction 3104; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('acbdijlk->abcdijkl',I31)
    
    #Contraction 3105; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('abcdijlk->abcdijkl',I31)
    
    #Contraction 3106; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('cdabijkl->abcdijkl',I31)
    
    #Contraction 3107; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('bdacijkl->abcdijkl',I31)
    
    #Contraction 3108; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('bcadijkl->abcdijkl',I31)
    
    #Contraction 3109; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('adbcijkl->abcdijkl',I31)
    
    #Contraction 3110; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('acbdijkl->abcdijkl',I31)
    
    #Contraction 3111; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('abcdijkl->abcdijkl',I31)
    
    del I31
    
    J31 = np.zeros([nvir, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 3112; Tree Level  2; Scaling  6/ 4 Result_size  4/ 4
    J31 += np.einsum('bcdnkl,anji->bcdaklji',T3,I7, optimize='optimal')
    
    del I7
    
    #Contraction 3113; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('bcdaklji->abcdijkl',J31)
    
    #Contraction 3114; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('acdbklji->abcdijkl',J31)
    
    #Contraction 3115; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('abdcklji->abcdijkl',J31)
    
    #Contraction 3116; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('abcdklji->abcdijkl',J31)
    
    #Contraction 3117; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('bcdajlki->abcdijkl',J31)
    
    #Contraction 3118; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('acdbjlki->abcdijkl',J31)
    
    #Contraction 3119; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('abdcjlki->abcdijkl',J31)
    
    #Contraction 3120; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('abcdjlki->abcdijkl',J31)
    
    #Contraction 3121; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('bcdajkli->abcdijkl',J31)
    
    #Contraction 3122; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('acdbjkli->abcdijkl',J31)
    
    #Contraction 3123; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('abdcjkli->abcdijkl',J31)
    
    #Contraction 3124; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('abcdjkli->abcdijkl',J31)
    
    #Contraction 3125; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('bcdaklij->abcdijkl',J31)
    
    #Contraction 3126; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('acdbklij->abcdijkl',J31)
    
    #Contraction 3127; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('abdcklij->abcdijkl',J31)
    
    #Contraction 3128; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('abcdklij->abcdijkl',J31)
    
    #Contraction 3129; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('bcdailkj->abcdijkl',J31)
    
    #Contraction 3130; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('acdbilkj->abcdijkl',J31)
    
    #Contraction 3131; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('abdcilkj->abcdijkl',J31)
    
    #Contraction 3132; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('abcdilkj->abcdijkl',J31)
    
    #Contraction 3133; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('bcdaiklj->abcdijkl',J31)
    
    #Contraction 3134; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('acdbiklj->abcdijkl',J31)
    
    #Contraction 3135; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('abdciklj->abcdijkl',J31)
    
    #Contraction 3136; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('abcdiklj->abcdijkl',J31)
    
    #Contraction 3137; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('bcdajlik->abcdijkl',J31)
    
    #Contraction 3138; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('acdbjlik->abcdijkl',J31)
    
    #Contraction 3139; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('abdcjlik->abcdijkl',J31)
    
    #Contraction 3140; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('abcdjlik->abcdijkl',J31)
    
    #Contraction 3141; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('bcdailjk->abcdijkl',J31)
    
    #Contraction 3142; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('acdbiljk->abcdijkl',J31)
    
    #Contraction 3143; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('abdciljk->abcdijkl',J31)
    
    #Contraction 3144; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('abcdiljk->abcdijkl',J31)
    
    #Contraction 3145; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('bcdaijlk->abcdijkl',J31)
    
    #Contraction 3146; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('acdbijlk->abcdijkl',J31)
    
    #Contraction 3147; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('abdcijlk->abcdijkl',J31)
    
    #Contraction 3148; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('abcdijlk->abcdijkl',J31)
    
    #Contraction 3149; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('bcdajkil->abcdijkl',J31)
    
    #Contraction 3150; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('acdbjkil->abcdijkl',J31)
    
    #Contraction 3151; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('abdcjkil->abcdijkl',J31)
    
    #Contraction 3152; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('abcdjkil->abcdijkl',J31)
    
    #Contraction 3153; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('bcdaikjl->abcdijkl',J31)
    
    #Contraction 3154; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('acdbikjl->abcdijkl',J31)
    
    #Contraction 3155; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('abdcikjl->abcdijkl',J31)
    
    #Contraction 3156; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('abcdikjl->abcdijkl',J31)
    
    #Contraction 3157; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('bcdaijkl->abcdijkl',J31)
    
    #Contraction 3158; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('acdbijkl->abcdijkl',J31)
    
    #Contraction 3159; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('abdcijkl->abcdijkl',J31)
    
    #Contraction 3160; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('abcdijkl->abcdijkl',J31)
    
    del J31
    
    M31 = np.zeros([nvir, nvir, nvir, nvir, nocc, nocc], dtype=type_)
    
    #Contraction 3161; Tree Level  3; Scaling  2/ 6 Result_size  2/ 4
    M31 += np.einsum('abef,fdjl->abdejl',V9,T2, optimize='optimal')
    
    #del V9
    
    X31 = np.zeros([nvir, nvir, nvir, nocc], dtype=type_)
    
    #Contraction 3162; Tree Level  3; Scaling  3/ 5 Result_size  1/ 3
    X31 += np.einsum('amef,bejm->abfj',V8,T2, optimize='optimal')
    
    #del V8
    
    Y31 = np.zeros([nvir, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 3163; Tree Level  2; Scaling  4/ 6 Result_size  4/ 4
    Y31 += np.einsum('fcdikl,abfj->cdabiklj',T3,X31, optimize='optimal')
    
    del X31
    
    #Contraction 3164; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('cdabiklj->abcdijkl',Y31)
    
    #Contraction 3165; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bdaciklj->abcdijkl',Y31)
    
    #Contraction 3166; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bcadiklj->abcdijkl',Y31)
    
    #Contraction 3167; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('cdbaiklj->abcdijkl',Y31)
    
    #Contraction 3168; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('adbciklj->abcdijkl',Y31)
    
    #Contraction 3169; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('acbdiklj->abcdijkl',Y31)
    
    #Contraction 3170; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bdcaiklj->abcdijkl',Y31)
    
    #Contraction 3171; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('adcbiklj->abcdijkl',Y31)
    
    #Contraction 3172; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('abcdiklj->abcdijkl',Y31)
    
    #Contraction 3173; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bcdaiklj->abcdijkl',Y31)
    
    #Contraction 3174; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('acdbiklj->abcdijkl',Y31)
    
    #Contraction 3175; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('abdciklj->abcdijkl',Y31)
    
    #Contraction 3176; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('cdabjkli->abcdijkl',Y31)
    
    #Contraction 3177; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bdacjkli->abcdijkl',Y31)
    
    #Contraction 3178; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bcadjkli->abcdijkl',Y31)
    
    #Contraction 3179; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('cdbajkli->abcdijkl',Y31)
    
    #Contraction 3180; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('adbcjkli->abcdijkl',Y31)
    
    #Contraction 3181; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('acbdjkli->abcdijkl',Y31)
    
    #Contraction 3182; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bdcajkli->abcdijkl',Y31)
    
    #Contraction 3183; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('adcbjkli->abcdijkl',Y31)
    
    #Contraction 3184; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('abcdjkli->abcdijkl',Y31)
    
    #Contraction 3185; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bcdajkli->abcdijkl',Y31)
    
    #Contraction 3186; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('acdbjkli->abcdijkl',Y31)
    
    #Contraction 3187; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('abdcjkli->abcdijkl',Y31)
    
    #Contraction 3188; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('cdabjilk->abcdijkl',Y31)
    
    #Contraction 3189; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bdacjilk->abcdijkl',Y31)
    
    #Contraction 3190; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bcadjilk->abcdijkl',Y31)
    
    #Contraction 3191; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('cdbajilk->abcdijkl',Y31)
    
    #Contraction 3192; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('adbcjilk->abcdijkl',Y31)
    
    #Contraction 3193; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('acbdjilk->abcdijkl',Y31)
    
    #Contraction 3194; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bdcajilk->abcdijkl',Y31)
    
    #Contraction 3195; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('adcbjilk->abcdijkl',Y31)
    
    #Contraction 3196; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('abcdjilk->abcdijkl',Y31)
    
    #Contraction 3197; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bcdajilk->abcdijkl',Y31)
    
    #Contraction 3198; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('acdbjilk->abcdijkl',Y31)
    
    #Contraction 3199; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('abdcjilk->abcdijkl',Y31)
    
    #Contraction 3200; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('cdabjikl->abcdijkl',Y31)
    
    #Contraction 3201; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bdacjikl->abcdijkl',Y31)
    
    #Contraction 3202; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bcadjikl->abcdijkl',Y31)
    
    #Contraction 3203; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('cdbajikl->abcdijkl',Y31)
    
    #Contraction 3204; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('adbcjikl->abcdijkl',Y31)
    
    #Contraction 3205; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('acbdjikl->abcdijkl',Y31)
    
    #Contraction 3206; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bdcajikl->abcdijkl',Y31)
    
    #Contraction 3207; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('adcbjikl->abcdijkl',Y31)
    
    #Contraction 3208; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('abcdjikl->abcdijkl',Y31)
    
    #Contraction 3209; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bcdajikl->abcdijkl',Y31)
    
    #Contraction 3210; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('acdbjikl->abcdijkl',Y31)
    
    #Contraction 3211; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('abdcjikl->abcdijkl',Y31)
    
    del Y31
    
    A32 = np.zeros([nvir, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 3212; Tree Level  2; Scaling  8/ 4 Result_size  4/ 4
    A32 += np.einsum('mnij,cdabklmn->cdabijkl',V1,T4, optimize='optimal')
    
    #del V1
    
    #Contraction 3213; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('cdabijkl->abcdijkl',A32)
    
    #Contraction 3214; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('cdabikjl->abcdijkl',A32)
    
    #Contraction 3215; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('cdabiljk->abcdijkl',A32)
    
    #Contraction 3216; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('cdabjkil->abcdijkl',A32)
    
    #Contraction 3217; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('cdabjlik->abcdijkl',A32)
    
    #Contraction 3218; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('cdabklij->abcdijkl',A32)
    
    del A32
    
    D32 = np.zeros([nvir, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 3219; Tree Level  2; Scaling  4/ 6 Result_size  4/ 4
    D32 += np.einsum('beji,acdekl->bacdjikl',T2,J23, optimize='optimal')
    
    del J23
    
    #Contraction 3220; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bacdjikl->abcdijkl',D32)
    
    #Contraction 3221; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('cabdjikl->abcdijkl',D32)
    
    #Contraction 3222; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('dabcjikl->abcdijkl',D32)
    
    #Contraction 3223; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('abcdjikl->abcdijkl',D32)
    
    #Contraction 3224; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('cbadjikl->abcdijkl',D32)
    
    #Contraction 3225; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('dbacjikl->abcdijkl',D32)
    
    #Contraction 3226; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('acbdjikl->abcdijkl',D32)
    
    #Contraction 3227; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bcadjikl->abcdijkl',D32)
    
    #Contraction 3228; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('dcabjikl->abcdijkl',D32)
    
    #Contraction 3229; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('adbcjikl->abcdijkl',D32)
    
    #Contraction 3230; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bdacjikl->abcdijkl',D32)
    
    #Contraction 3231; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('cdabjikl->abcdijkl',D32)
    
    #Contraction 3232; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bacdjkil->abcdijkl',D32)
    
    #Contraction 3233; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('cabdjkil->abcdijkl',D32)
    
    #Contraction 3234; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('dabcjkil->abcdijkl',D32)
    
    #Contraction 3235; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('abcdjkil->abcdijkl',D32)
    
    #Contraction 3236; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('cbadjkil->abcdijkl',D32)
    
    #Contraction 3237; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('dbacjkil->abcdijkl',D32)
    
    #Contraction 3238; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('acbdjkil->abcdijkl',D32)
    
    #Contraction 3239; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bcadjkil->abcdijkl',D32)
    
    #Contraction 3240; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('dcabjkil->abcdijkl',D32)
    
    #Contraction 3241; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('adbcjkil->abcdijkl',D32)
    
    #Contraction 3242; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bdacjkil->abcdijkl',D32)
    
    #Contraction 3243; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('cdabjkil->abcdijkl',D32)
    
    #Contraction 3244; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bacdjlik->abcdijkl',D32)
    
    #Contraction 3245; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('cabdjlik->abcdijkl',D32)
    
    #Contraction 3246; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('dabcjlik->abcdijkl',D32)
    
    #Contraction 3247; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('abcdjlik->abcdijkl',D32)
    
    #Contraction 3248; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('cbadjlik->abcdijkl',D32)
    
    #Contraction 3249; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('dbacjlik->abcdijkl',D32)
    
    #Contraction 3250; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('acbdjlik->abcdijkl',D32)
    
    #Contraction 3251; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bcadjlik->abcdijkl',D32)
    
    #Contraction 3252; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('dcabjlik->abcdijkl',D32)
    
    #Contraction 3253; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('adbcjlik->abcdijkl',D32)
    
    #Contraction 3254; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bdacjlik->abcdijkl',D32)
    
    #Contraction 3255; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('cdabjlik->abcdijkl',D32)
    
    #Contraction 3256; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bacdikjl->abcdijkl',D32)
    
    #Contraction 3257; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('cabdikjl->abcdijkl',D32)
    
    #Contraction 3258; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('dabcikjl->abcdijkl',D32)
    
    #Contraction 3259; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('abcdikjl->abcdijkl',D32)
    
    #Contraction 3260; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('cbadikjl->abcdijkl',D32)
    
    #Contraction 3261; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('dbacikjl->abcdijkl',D32)
    
    #Contraction 3262; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('acbdikjl->abcdijkl',D32)
    
    #Contraction 3263; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bcadikjl->abcdijkl',D32)
    
    #Contraction 3264; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('dcabikjl->abcdijkl',D32)
    
    #Contraction 3265; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('adbcikjl->abcdijkl',D32)
    
    #Contraction 3266; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bdacikjl->abcdijkl',D32)
    
    #Contraction 3267; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('cdabikjl->abcdijkl',D32)
    
    #Contraction 3268; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bacdiljk->abcdijkl',D32)
    
    #Contraction 3269; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('cabdiljk->abcdijkl',D32)
    
    #Contraction 3270; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('dabciljk->abcdijkl',D32)
    
    #Contraction 3271; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('abcdiljk->abcdijkl',D32)
    
    #Contraction 3272; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('cbadiljk->abcdijkl',D32)
    
    #Contraction 3273; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('dbaciljk->abcdijkl',D32)
    
    #Contraction 3274; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('acbdiljk->abcdijkl',D32)
    
    #Contraction 3275; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bcadiljk->abcdijkl',D32)
    
    #Contraction 3276; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('dcabiljk->abcdijkl',D32)
    
    #Contraction 3277; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('adbciljk->abcdijkl',D32)
    
    #Contraction 3278; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bdaciljk->abcdijkl',D32)
    
    #Contraction 3279; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('cdabiljk->abcdijkl',D32)
    
    #Contraction 3280; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bacdklji->abcdijkl',D32)
    
    #Contraction 3281; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('cabdklji->abcdijkl',D32)
    
    #Contraction 3282; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('dabcklji->abcdijkl',D32)
    
    #Contraction 3283; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('abcdklji->abcdijkl',D32)
    
    #Contraction 3284; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('cbadklji->abcdijkl',D32)
    
    #Contraction 3285; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('dbacklji->abcdijkl',D32)
    
    #Contraction 3286; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('acbdklji->abcdijkl',D32)
    
    #Contraction 3287; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bcadklji->abcdijkl',D32)
    
    #Contraction 3288; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('dcabklji->abcdijkl',D32)
    
    #Contraction 3289; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('adbcklji->abcdijkl',D32)
    
    #Contraction 3290; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bdacklji->abcdijkl',D32)
    
    #Contraction 3291; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('cdabklji->abcdijkl',D32)
    
    del D32
    
    E32 = np.zeros([nvir, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 3292; Tree Level  2; Scaling  6/ 4 Result_size  4/ 4
    E32 += np.einsum('abcdmjkl,mi->abcdjkli',T4,Y1, optimize='optimal')
    
    del Y1
    
    #Contraction 3293; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('abcdjkli->abcdijkl',E32)
    
    #Contraction 3294; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('abcdiklj->abcdijkl',E32)
    
    #Contraction 3295; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('abcdijlk->abcdijkl',E32)
    
    #Contraction 3296; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('abcdijkl->abcdijkl',E32)
    
    del E32
    
    G32 = np.zeros([nvir, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 3297; Tree Level  2; Scaling  4/ 6 Result_size  4/ 4
    G32 += np.einsum('fbcdijkl,af->bcdaijkl',T4,A16, optimize='optimal')
    
    del A16
    
    #Contraction 3298; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('bcdaijkl->abcdijkl',G32)
    
    #Contraction 3299; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('acdbijkl->abcdijkl',G32)
    
    #Contraction 3300; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('abdcijkl->abcdijkl',G32)
    
    #Contraction 3301; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('abcdijkl->abcdijkl',G32)
    
    del G32
    
    I32 = np.zeros([nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 3302; Tree Level  3; Scaling  6/ 2 Result_size  4/ 2
    I32 += np.einsum('bdnl,mnji->bdmlji',T2,E7, optimize='optimal')
    
    del E7
    
    J32 = np.zeros([nvir, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 3303; Tree Level  2; Scaling  6/ 4 Result_size  4/ 4
    J32 += np.einsum('acmk,bdmlji->acbdklji',T2,I32, optimize='optimal')
    
    del I32
    
    #Contraction 3304; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('acbdklji->abcdijkl',J32)
    
    #Contraction 3305; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('abcdklji->abcdijkl',J32)
    
    #Contraction 3306; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('adcbklji->abcdijkl',J32)
    
    #Contraction 3307; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('cbadklji->abcdijkl',J32)
    
    #Contraction 3308; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('cdabklji->abcdijkl',J32)
    
    #Contraction 3309; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('bdacklji->abcdijkl',J32)
    
    #Contraction 3310; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('acbdlkji->abcdijkl',J32)
    
    #Contraction 3311; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('abcdlkji->abcdijkl',J32)
    
    #Contraction 3312; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('adcblkji->abcdijkl',J32)
    
    #Contraction 3313; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('cbadlkji->abcdijkl',J32)
    
    #Contraction 3314; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('cdablkji->abcdijkl',J32)
    
    #Contraction 3315; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('bdaclkji->abcdijkl',J32)
    
    #Contraction 3316; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('acbdjlki->abcdijkl',J32)
    
    #Contraction 3317; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('abcdjlki->abcdijkl',J32)
    
    #Contraction 3318; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('adcbjlki->abcdijkl',J32)
    
    #Contraction 3319; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('cbadjlki->abcdijkl',J32)
    
    #Contraction 3320; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('cdabjlki->abcdijkl',J32)
    
    #Contraction 3321; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('bdacjlki->abcdijkl',J32)
    
    #Contraction 3322; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('acbdljki->abcdijkl',J32)
    
    #Contraction 3323; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('abcdljki->abcdijkl',J32)
    
    #Contraction 3324; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('adcbljki->abcdijkl',J32)
    
    #Contraction 3325; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('cbadljki->abcdijkl',J32)
    
    #Contraction 3326; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('cdabljki->abcdijkl',J32)
    
    #Contraction 3327; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('bdacljki->abcdijkl',J32)
    
    #Contraction 3328; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('acbdjkli->abcdijkl',J32)
    
    #Contraction 3329; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('abcdjkli->abcdijkl',J32)
    
    #Contraction 3330; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('adcbjkli->abcdijkl',J32)
    
    #Contraction 3331; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('cbadjkli->abcdijkl',J32)
    
    #Contraction 3332; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('cdabjkli->abcdijkl',J32)
    
    #Contraction 3333; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('bdacjkli->abcdijkl',J32)
    
    #Contraction 3334; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('acbdkjli->abcdijkl',J32)
    
    #Contraction 3335; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('abcdkjli->abcdijkl',J32)
    
    #Contraction 3336; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('adcbkjli->abcdijkl',J32)
    
    #Contraction 3337; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('cbadkjli->abcdijkl',J32)
    
    #Contraction 3338; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('cdabkjli->abcdijkl',J32)
    
    #Contraction 3339; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('bdackjli->abcdijkl',J32)
    
    #Contraction 3340; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('acbdklij->abcdijkl',J32)
    
    #Contraction 3341; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('abcdklij->abcdijkl',J32)
    
    #Contraction 3342; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('adcbklij->abcdijkl',J32)
    
    #Contraction 3343; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('cbadklij->abcdijkl',J32)
    
    #Contraction 3344; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('cdabklij->abcdijkl',J32)
    
    #Contraction 3345; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('bdacklij->abcdijkl',J32)
    
    #Contraction 3346; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('acbdlkij->abcdijkl',J32)
    
    #Contraction 3347; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('abcdlkij->abcdijkl',J32)
    
    #Contraction 3348; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('adcblkij->abcdijkl',J32)
    
    #Contraction 3349; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('cbadlkij->abcdijkl',J32)
    
    #Contraction 3350; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('cdablkij->abcdijkl',J32)
    
    #Contraction 3351; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('bdaclkij->abcdijkl',J32)
    
    #Contraction 3352; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('acbdilkj->abcdijkl',J32)
    
    #Contraction 3353; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('abcdilkj->abcdijkl',J32)
    
    #Contraction 3354; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('adcbilkj->abcdijkl',J32)
    
    #Contraction 3355; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('cbadilkj->abcdijkl',J32)
    
    #Contraction 3356; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('cdabilkj->abcdijkl',J32)
    
    #Contraction 3357; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('bdacilkj->abcdijkl',J32)
    
    #Contraction 3358; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('acbdlikj->abcdijkl',J32)
    
    #Contraction 3359; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('abcdlikj->abcdijkl',J32)
    
    #Contraction 3360; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('adcblikj->abcdijkl',J32)
    
    #Contraction 3361; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('cbadlikj->abcdijkl',J32)
    
    #Contraction 3362; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('cdablikj->abcdijkl',J32)
    
    #Contraction 3363; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('bdaclikj->abcdijkl',J32)
    
    #Contraction 3364; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('acbdiklj->abcdijkl',J32)
    
    #Contraction 3365; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('abcdiklj->abcdijkl',J32)
    
    #Contraction 3366; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('adcbiklj->abcdijkl',J32)
    
    #Contraction 3367; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('cbadiklj->abcdijkl',J32)
    
    #Contraction 3368; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('cdabiklj->abcdijkl',J32)
    
    #Contraction 3369; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('bdaciklj->abcdijkl',J32)
    
    #Contraction 3370; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('acbdkilj->abcdijkl',J32)
    
    #Contraction 3371; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('abcdkilj->abcdijkl',J32)
    
    #Contraction 3372; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('adcbkilj->abcdijkl',J32)
    
    #Contraction 3373; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('cbadkilj->abcdijkl',J32)
    
    #Contraction 3374; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('cdabkilj->abcdijkl',J32)
    
    #Contraction 3375; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('bdackilj->abcdijkl',J32)
    
    #Contraction 3376; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('acbdjlik->abcdijkl',J32)
    
    #Contraction 3377; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('abcdjlik->abcdijkl',J32)
    
    #Contraction 3378; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('adcbjlik->abcdijkl',J32)
    
    #Contraction 3379; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('cbadjlik->abcdijkl',J32)
    
    #Contraction 3380; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('cdabjlik->abcdijkl',J32)
    
    #Contraction 3381; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('bdacjlik->abcdijkl',J32)
    
    #Contraction 3382; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('acbdljik->abcdijkl',J32)
    
    #Contraction 3383; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('abcdljik->abcdijkl',J32)
    
    #Contraction 3384; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('adcbljik->abcdijkl',J32)
    
    #Contraction 3385; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('cbadljik->abcdijkl',J32)
    
    #Contraction 3386; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('cdabljik->abcdijkl',J32)
    
    #Contraction 3387; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('bdacljik->abcdijkl',J32)
    
    #Contraction 3388; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('acbdiljk->abcdijkl',J32)
    
    #Contraction 3389; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('abcdiljk->abcdijkl',J32)
    
    #Contraction 3390; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('adcbiljk->abcdijkl',J32)
    
    #Contraction 3391; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('cbadiljk->abcdijkl',J32)
    
    #Contraction 3392; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('cdabiljk->abcdijkl',J32)
    
    #Contraction 3393; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('bdaciljk->abcdijkl',J32)
    
    #Contraction 3394; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('acbdlijk->abcdijkl',J32)
    
    #Contraction 3395; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('abcdlijk->abcdijkl',J32)
    
    #Contraction 3396; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('adcblijk->abcdijkl',J32)
    
    #Contraction 3397; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('cbadlijk->abcdijkl',J32)
    
    #Contraction 3398; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('cdablijk->abcdijkl',J32)
    
    #Contraction 3399; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('bdaclijk->abcdijkl',J32)
    
    #Contraction 3400; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('acbdijlk->abcdijkl',J32)
    
    #Contraction 3401; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('abcdijlk->abcdijkl',J32)
    
    #Contraction 3402; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('adcbijlk->abcdijkl',J32)
    
    #Contraction 3403; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('cbadijlk->abcdijkl',J32)
    
    #Contraction 3404; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('cdabijlk->abcdijkl',J32)
    
    #Contraction 3405; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('bdacijlk->abcdijkl',J32)
    
    #Contraction 3406; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('acbdjilk->abcdijkl',J32)
    
    #Contraction 3407; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('abcdjilk->abcdijkl',J32)
    
    #Contraction 3408; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('adcbjilk->abcdijkl',J32)
    
    #Contraction 3409; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('cbadjilk->abcdijkl',J32)
    
    #Contraction 3410; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('cdabjilk->abcdijkl',J32)
    
    #Contraction 3411; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('bdacjilk->abcdijkl',J32)
    
    #Contraction 3412; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('acbdjkil->abcdijkl',J32)
    
    #Contraction 3413; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('abcdjkil->abcdijkl',J32)
    
    #Contraction 3414; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('adcbjkil->abcdijkl',J32)
    
    #Contraction 3415; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('cbadjkil->abcdijkl',J32)
    
    #Contraction 3416; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('cdabjkil->abcdijkl',J32)
    
    #Contraction 3417; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('bdacjkil->abcdijkl',J32)
    
    #Contraction 3418; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('acbdkjil->abcdijkl',J32)
    
    #Contraction 3419; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('abcdkjil->abcdijkl',J32)
    
    #Contraction 3420; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('adcbkjil->abcdijkl',J32)
    
    #Contraction 3421; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('cbadkjil->abcdijkl',J32)
    
    #Contraction 3422; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('cdabkjil->abcdijkl',J32)
    
    #Contraction 3423; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('bdackjil->abcdijkl',J32)
    
    #Contraction 3424; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('acbdikjl->abcdijkl',J32)
    
    #Contraction 3425; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('abcdikjl->abcdijkl',J32)
    
    #Contraction 3426; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('adcbikjl->abcdijkl',J32)
    
    #Contraction 3427; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('cbadikjl->abcdijkl',J32)
    
    #Contraction 3428; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('cdabikjl->abcdijkl',J32)
    
    #Contraction 3429; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('bdacikjl->abcdijkl',J32)
    
    #Contraction 3430; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('acbdkijl->abcdijkl',J32)
    
    #Contraction 3431; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('abcdkijl->abcdijkl',J32)
    
    #Contraction 3432; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('adcbkijl->abcdijkl',J32)
    
    #Contraction 3433; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('cbadkijl->abcdijkl',J32)
    
    #Contraction 3434; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('cdabkijl->abcdijkl',J32)
    
    #Contraction 3435; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('bdackijl->abcdijkl',J32)
    
    #Contraction 3436; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('acbdijkl->abcdijkl',J32)
    
    #Contraction 3437; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('abcdijkl->abcdijkl',J32)
    
    #Contraction 3438; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('adcbijkl->abcdijkl',J32)
    
    #Contraction 3439; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('cbadijkl->abcdijkl',J32)
    
    #Contraction 3440; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('cdabijkl->abcdijkl',J32)
    
    #Contraction 3441; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('bdacijkl->abcdijkl',J32)
    
    #Contraction 3442; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('acbdjikl->abcdijkl',J32)
    
    #Contraction 3443; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('abcdjikl->abcdijkl',J32)
    
    #Contraction 3444; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('adcbjikl->abcdijkl',J32)
    
    #Contraction 3445; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('cbadjikl->abcdijkl',J32)
    
    #Contraction 3446; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('cdabjikl->abcdijkl',J32)
    
    #Contraction 3447; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('bdacjikl->abcdijkl',J32)
    
    del J32
    
    M32 = np.zeros([nvir, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 3448; Tree Level  2; Scaling  6/ 4 Result_size  4/ 4
    M32 += np.einsum('bdnl,ancijk->bdaclijk',T2,G22, optimize='optimal')
    
    del G22
    
    #Contraction 3449; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bdaclijk->abcdijkl',M32)
    
    #Contraction 3450; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('cdablijk->abcdijkl',M32)
    
    #Contraction 3451; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('cbadlijk->abcdijkl',M32)
    
    #Contraction 3452; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bdcalijk->abcdijkl',M32)
    
    #Contraction 3453; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('adcblijk->abcdijkl',M32)
    
    #Contraction 3454; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('abcdlijk->abcdijkl',M32)
    
    #Contraction 3455; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('cdbalijk->abcdijkl',M32)
    
    #Contraction 3456; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('adbclijk->abcdijkl',M32)
    
    #Contraction 3457; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('acbdlijk->abcdijkl',M32)
    
    #Contraction 3458; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('cbdalijk->abcdijkl',M32)
    
    #Contraction 3459; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('abdclijk->abcdijkl',M32)
    
    #Contraction 3460; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('acdblijk->abcdijkl',M32)
    
    #Contraction 3461; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bdackijl->abcdijkl',M32)
    
    #Contraction 3462; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('cdabkijl->abcdijkl',M32)
    
    #Contraction 3463; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('cbadkijl->abcdijkl',M32)
    
    #Contraction 3464; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bdcakijl->abcdijkl',M32)
    
    #Contraction 3465; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('adcbkijl->abcdijkl',M32)
    
    #Contraction 3466; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('abcdkijl->abcdijkl',M32)
    
    #Contraction 3467; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('cdbakijl->abcdijkl',M32)
    
    #Contraction 3468; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('adbckijl->abcdijkl',M32)
    
    #Contraction 3469; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('acbdkijl->abcdijkl',M32)
    
    #Contraction 3470; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('cbdakijl->abcdijkl',M32)
    
    #Contraction 3471; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('abdckijl->abcdijkl',M32)
    
    #Contraction 3472; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('acdbkijl->abcdijkl',M32)
    
    #Contraction 3473; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bdacjikl->abcdijkl',M32)
    
    #Contraction 3474; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('cdabjikl->abcdijkl',M32)
    
    #Contraction 3475; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('cbadjikl->abcdijkl',M32)
    
    #Contraction 3476; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bdcajikl->abcdijkl',M32)
    
    #Contraction 3477; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('adcbjikl->abcdijkl',M32)
    
    #Contraction 3478; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('abcdjikl->abcdijkl',M32)
    
    #Contraction 3479; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('cdbajikl->abcdijkl',M32)
    
    #Contraction 3480; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('adbcjikl->abcdijkl',M32)
    
    #Contraction 3481; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('acbdjikl->abcdijkl',M32)
    
    #Contraction 3482; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('cbdajikl->abcdijkl',M32)
    
    #Contraction 3483; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('abdcjikl->abcdijkl',M32)
    
    #Contraction 3484; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('acdbjikl->abcdijkl',M32)
    
    #Contraction 3485; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bdacljik->abcdijkl',M32)
    
    #Contraction 3486; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('cdabljik->abcdijkl',M32)
    
    #Contraction 3487; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('cbadljik->abcdijkl',M32)
    
    #Contraction 3488; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bdcaljik->abcdijkl',M32)
    
    #Contraction 3489; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('adcbljik->abcdijkl',M32)
    
    #Contraction 3490; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('abcdljik->abcdijkl',M32)
    
    #Contraction 3491; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('cdbaljik->abcdijkl',M32)
    
    #Contraction 3492; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('adbcljik->abcdijkl',M32)
    
    #Contraction 3493; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('acbdljik->abcdijkl',M32)
    
    #Contraction 3494; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('cbdaljik->abcdijkl',M32)
    
    #Contraction 3495; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('abdcljik->abcdijkl',M32)
    
    #Contraction 3496; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('acdbljik->abcdijkl',M32)
    
    #Contraction 3497; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bdackjil->abcdijkl',M32)
    
    #Contraction 3498; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('cdabkjil->abcdijkl',M32)
    
    #Contraction 3499; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('cbadkjil->abcdijkl',M32)
    
    #Contraction 3500; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bdcakjil->abcdijkl',M32)
    
    #Contraction 3501; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('adcbkjil->abcdijkl',M32)
    
    #Contraction 3502; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('abcdkjil->abcdijkl',M32)
    
    #Contraction 3503; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('cdbakjil->abcdijkl',M32)
    
    #Contraction 3504; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('adbckjil->abcdijkl',M32)
    
    #Contraction 3505; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('acbdkjil->abcdijkl',M32)
    
    #Contraction 3506; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('cbdakjil->abcdijkl',M32)
    
    #Contraction 3507; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('abdckjil->abcdijkl',M32)
    
    #Contraction 3508; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('acdbkjil->abcdijkl',M32)
    
    #Contraction 3509; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bdacijkl->abcdijkl',M32)
    
    #Contraction 3510; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('cdabijkl->abcdijkl',M32)
    
    #Contraction 3511; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('cbadijkl->abcdijkl',M32)
    
    #Contraction 3512; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bdcaijkl->abcdijkl',M32)
    
    #Contraction 3513; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('adcbijkl->abcdijkl',M32)
    
    #Contraction 3514; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('abcdijkl->abcdijkl',M32)
    
    #Contraction 3515; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('cdbaijkl->abcdijkl',M32)
    
    #Contraction 3516; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('adbcijkl->abcdijkl',M32)
    
    #Contraction 3517; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('acbdijkl->abcdijkl',M32)
    
    #Contraction 3518; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('cbdaijkl->abcdijkl',M32)
    
    #Contraction 3519; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('abdcijkl->abcdijkl',M32)
    
    #Contraction 3520; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('acdbijkl->abcdijkl',M32)
    
    #Contraction 3521; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bdaclkij->abcdijkl',M32)
    
    #Contraction 3522; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('cdablkij->abcdijkl',M32)
    
    #Contraction 3523; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('cbadlkij->abcdijkl',M32)
    
    #Contraction 3524; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bdcalkij->abcdijkl',M32)
    
    #Contraction 3525; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('adcblkij->abcdijkl',M32)
    
    #Contraction 3526; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('abcdlkij->abcdijkl',M32)
    
    #Contraction 3527; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('cdbalkij->abcdijkl',M32)
    
    #Contraction 3528; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('adbclkij->abcdijkl',M32)
    
    #Contraction 3529; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('acbdlkij->abcdijkl',M32)
    
    #Contraction 3530; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('cbdalkij->abcdijkl',M32)
    
    #Contraction 3531; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('abdclkij->abcdijkl',M32)
    
    #Contraction 3532; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('acdblkij->abcdijkl',M32)
    
    #Contraction 3533; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bdacjkil->abcdijkl',M32)
    
    #Contraction 3534; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('cdabjkil->abcdijkl',M32)
    
    #Contraction 3535; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('cbadjkil->abcdijkl',M32)
    
    #Contraction 3536; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bdcajkil->abcdijkl',M32)
    
    #Contraction 3537; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('adcbjkil->abcdijkl',M32)
    
    #Contraction 3538; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('abcdjkil->abcdijkl',M32)
    
    #Contraction 3539; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('cdbajkil->abcdijkl',M32)
    
    #Contraction 3540; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('adbcjkil->abcdijkl',M32)
    
    #Contraction 3541; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('acbdjkil->abcdijkl',M32)
    
    #Contraction 3542; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('cbdajkil->abcdijkl',M32)
    
    #Contraction 3543; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('abdcjkil->abcdijkl',M32)
    
    #Contraction 3544; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('acdbjkil->abcdijkl',M32)
    
    #Contraction 3545; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bdacikjl->abcdijkl',M32)
    
    #Contraction 3546; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('cdabikjl->abcdijkl',M32)
    
    #Contraction 3547; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('cbadikjl->abcdijkl',M32)
    
    #Contraction 3548; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bdcaikjl->abcdijkl',M32)
    
    #Contraction 3549; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('adcbikjl->abcdijkl',M32)
    
    #Contraction 3550; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('abcdikjl->abcdijkl',M32)
    
    #Contraction 3551; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('cdbaikjl->abcdijkl',M32)
    
    #Contraction 3552; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('adbcikjl->abcdijkl',M32)
    
    #Contraction 3553; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('acbdikjl->abcdijkl',M32)
    
    #Contraction 3554; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('cbdaikjl->abcdijkl',M32)
    
    #Contraction 3555; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('abdcikjl->abcdijkl',M32)
    
    #Contraction 3556; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('acdbikjl->abcdijkl',M32)
    
    #Contraction 3557; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bdacklij->abcdijkl',M32)
    
    #Contraction 3558; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('cdabklij->abcdijkl',M32)
    
    #Contraction 3559; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('cbadklij->abcdijkl',M32)
    
    #Contraction 3560; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bdcaklij->abcdijkl',M32)
    
    #Contraction 3561; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('adcbklij->abcdijkl',M32)
    
    #Contraction 3562; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('abcdklij->abcdijkl',M32)
    
    #Contraction 3563; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('cdbaklij->abcdijkl',M32)
    
    #Contraction 3564; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('adbcklij->abcdijkl',M32)
    
    #Contraction 3565; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('acbdklij->abcdijkl',M32)
    
    #Contraction 3566; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('cbdaklij->abcdijkl',M32)
    
    #Contraction 3567; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('abdcklij->abcdijkl',M32)
    
    #Contraction 3568; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('acdbklij->abcdijkl',M32)
    
    #Contraction 3569; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bdacjlik->abcdijkl',M32)
    
    #Contraction 3570; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('cdabjlik->abcdijkl',M32)
    
    #Contraction 3571; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('cbadjlik->abcdijkl',M32)
    
    #Contraction 3572; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bdcajlik->abcdijkl',M32)
    
    #Contraction 3573; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('adcbjlik->abcdijkl',M32)
    
    #Contraction 3574; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('abcdjlik->abcdijkl',M32)
    
    #Contraction 3575; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('cdbajlik->abcdijkl',M32)
    
    #Contraction 3576; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('adbcjlik->abcdijkl',M32)
    
    #Contraction 3577; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('acbdjlik->abcdijkl',M32)
    
    #Contraction 3578; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('cbdajlik->abcdijkl',M32)
    
    #Contraction 3579; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('abdcjlik->abcdijkl',M32)
    
    #Contraction 3580; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('acdbjlik->abcdijkl',M32)
    
    #Contraction 3581; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bdaciljk->abcdijkl',M32)
    
    #Contraction 3582; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('cdabiljk->abcdijkl',M32)
    
    #Contraction 3583; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('cbadiljk->abcdijkl',M32)
    
    #Contraction 3584; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bdcailjk->abcdijkl',M32)
    
    #Contraction 3585; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('adcbiljk->abcdijkl',M32)
    
    #Contraction 3586; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('abcdiljk->abcdijkl',M32)
    
    #Contraction 3587; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('cdbailjk->abcdijkl',M32)
    
    #Contraction 3588; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('adbciljk->abcdijkl',M32)
    
    #Contraction 3589; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('acbdiljk->abcdijkl',M32)
    
    #Contraction 3590; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('cbdailjk->abcdijkl',M32)
    
    #Contraction 3591; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('abdciljk->abcdijkl',M32)
    
    #Contraction 3592; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('acdbiljk->abcdijkl',M32)
    
    del M32
    
    X32 = np.zeros([nvir, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 3593; Tree Level  2; Scaling  6/ 4 Result_size  4/ 4
    X32 += np.einsum('abcdnjkl,ni->abcdjkli',T4,Y16, optimize='optimal')
    
    del Y16
    
    #Contraction 3594; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('abcdjkli->abcdijkl',X32)
    
    #Contraction 3595; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('abcdiklj->abcdijkl',X32)
    
    #Contraction 3596; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('abcdijlk->abcdijkl',X32)
    
    #Contraction 3597; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('abcdijkl->abcdijkl',X32)
    
    del X32
    
    Y32 = np.zeros([nvir, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 3598; Tree Level  2; Scaling  4/ 6 Result_size  4/ 4
    Y32 += np.einsum('ceki,abdejl->cabdkijl',T2,M31, optimize='optimal')
    
    del M31
    
    #Contraction 3599; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('cabdkijl->abcdijkl',Y32)
    
    #Contraction 3600; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('dabckijl->abcdijkl',Y32)
    
    #Contraction 3601; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('bacdkijl->abcdijkl',Y32)
    
    #Contraction 3602; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('dacbkijl->abcdijkl',Y32)
    
    #Contraction 3603; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('badckijl->abcdijkl',Y32)
    
    #Contraction 3604; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('cadbkijl->abcdijkl',Y32)
    
    #Contraction 3605; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('abcdkijl->abcdijkl',Y32)
    
    #Contraction 3606; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('dbcakijl->abcdijkl',Y32)
    
    #Contraction 3607; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('abdckijl->abcdijkl',Y32)
    
    #Contraction 3608; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('cbdakijl->abcdijkl',Y32)
    
    #Contraction 3609; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('acdbkijl->abcdijkl',Y32)
    
    #Contraction 3610; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('bcdakijl->abcdijkl',Y32)
    
    #Contraction 3611; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('cabdkjil->abcdijkl',Y32)
    
    #Contraction 3612; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('dabckjil->abcdijkl',Y32)
    
    #Contraction 3613; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('bacdkjil->abcdijkl',Y32)
    
    #Contraction 3614; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('dacbkjil->abcdijkl',Y32)
    
    #Contraction 3615; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('badckjil->abcdijkl',Y32)
    
    #Contraction 3616; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('cadbkjil->abcdijkl',Y32)
    
    #Contraction 3617; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('abcdkjil->abcdijkl',Y32)
    
    #Contraction 3618; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('dbcakjil->abcdijkl',Y32)
    
    #Contraction 3619; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('abdckjil->abcdijkl',Y32)
    
    #Contraction 3620; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('cbdakjil->abcdijkl',Y32)
    
    #Contraction 3621; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('acdbkjil->abcdijkl',Y32)
    
    #Contraction 3622; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('bcdakjil->abcdijkl',Y32)
    
    #Contraction 3623; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('cabdklij->abcdijkl',Y32)
    
    #Contraction 3624; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('dabcklij->abcdijkl',Y32)
    
    #Contraction 3625; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('bacdklij->abcdijkl',Y32)
    
    #Contraction 3626; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('dacbklij->abcdijkl',Y32)
    
    #Contraction 3627; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('badcklij->abcdijkl',Y32)
    
    #Contraction 3628; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('cadbklij->abcdijkl',Y32)
    
    #Contraction 3629; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('abcdklij->abcdijkl',Y32)
    
    #Contraction 3630; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('dbcaklij->abcdijkl',Y32)
    
    #Contraction 3631; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('abdcklij->abcdijkl',Y32)
    
    #Contraction 3632; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('cbdaklij->abcdijkl',Y32)
    
    #Contraction 3633; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('acdbklij->abcdijkl',Y32)
    
    #Contraction 3634; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('bcdaklij->abcdijkl',Y32)
    
    #Contraction 3635; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('cabdijkl->abcdijkl',Y32)
    
    #Contraction 3636; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('dabcijkl->abcdijkl',Y32)
    
    #Contraction 3637; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('bacdijkl->abcdijkl',Y32)
    
    #Contraction 3638; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('dacbijkl->abcdijkl',Y32)
    
    #Contraction 3639; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('badcijkl->abcdijkl',Y32)
    
    #Contraction 3640; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('cadbijkl->abcdijkl',Y32)
    
    #Contraction 3641; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('abcdijkl->abcdijkl',Y32)
    
    #Contraction 3642; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('dbcaijkl->abcdijkl',Y32)
    
    #Contraction 3643; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('abdcijkl->abcdijkl',Y32)
    
    #Contraction 3644; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('cbdaijkl->abcdijkl',Y32)
    
    #Contraction 3645; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('acdbijkl->abcdijkl',Y32)
    
    #Contraction 3646; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('bcdaijkl->abcdijkl',Y32)
    
    #Contraction 3647; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('cabdilkj->abcdijkl',Y32)
    
    #Contraction 3648; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('dabcilkj->abcdijkl',Y32)
    
    #Contraction 3649; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('bacdilkj->abcdijkl',Y32)
    
    #Contraction 3650; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('dacbilkj->abcdijkl',Y32)
    
    #Contraction 3651; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('badcilkj->abcdijkl',Y32)
    
    #Contraction 3652; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('cadbilkj->abcdijkl',Y32)
    
    #Contraction 3653; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('abcdilkj->abcdijkl',Y32)
    
    #Contraction 3654; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('dbcailkj->abcdijkl',Y32)
    
    #Contraction 3655; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('abdcilkj->abcdijkl',Y32)
    
    #Contraction 3656; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('cbdailkj->abcdijkl',Y32)
    
    #Contraction 3657; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('acdbilkj->abcdijkl',Y32)
    
    #Contraction 3658; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('bcdailkj->abcdijkl',Y32)
    
    #Contraction 3659; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('cabdjlki->abcdijkl',Y32)
    
    #Contraction 3660; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('dabcjlki->abcdijkl',Y32)
    
    #Contraction 3661; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('bacdjlki->abcdijkl',Y32)
    
    #Contraction 3662; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('dacbjlki->abcdijkl',Y32)
    
    #Contraction 3663; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('badcjlki->abcdijkl',Y32)
    
    #Contraction 3664; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('cadbjlki->abcdijkl',Y32)
    
    #Contraction 3665; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('abcdjlki->abcdijkl',Y32)
    
    #Contraction 3666; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('dbcajlki->abcdijkl',Y32)
    
    #Contraction 3667; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('abdcjlki->abcdijkl',Y32)
    
    #Contraction 3668; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('cbdajlki->abcdijkl',Y32)
    
    #Contraction 3669; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('acdbjlki->abcdijkl',Y32)
    
    #Contraction 3670; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('bcdajlki->abcdijkl',Y32)
    
    del Y32
    
    A33 = np.zeros([nvir, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 3671; Tree Level  2; Scaling  8/ 4 Result_size  4/ 4
    A33 += np.einsum('bcdmnl,mnaijk->bcdalijk',T3,Y30, optimize='optimal')
    
    del Y30
    
    #Contraction 3672; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('bcdalijk->abcdijkl',A33)
    
    #Contraction 3673; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('acdblijk->abcdijkl',A33)
    
    #Contraction 3674; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('abdclijk->abcdijkl',A33)
    
    #Contraction 3675; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('abcdlijk->abcdijkl',A33)
    
    #Contraction 3676; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('bcdakijl->abcdijkl',A33)
    
    #Contraction 3677; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('acdbkijl->abcdijkl',A33)
    
    #Contraction 3678; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('abdckijl->abcdijkl',A33)
    
    #Contraction 3679; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('abcdkijl->abcdijkl',A33)
    
    #Contraction 3680; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('bcdajikl->abcdijkl',A33)
    
    #Contraction 3681; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('acdbjikl->abcdijkl',A33)
    
    #Contraction 3682; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('abdcjikl->abcdijkl',A33)
    
    #Contraction 3683; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('abcdjikl->abcdijkl',A33)
    
    #Contraction 3684; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('bcdaijkl->abcdijkl',A33)
    
    #Contraction 3685; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('acdbijkl->abcdijkl',A33)
    
    #Contraction 3686; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('abdcijkl->abcdijkl',A33)
    
    #Contraction 3687; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('abcdijkl->abcdijkl',A33)
    
    del A33
    
    D33 = np.zeros([nvir, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 3688; Tree Level  2; Scaling  6/ 4 Result_size  4/ 4
    D33 += np.einsum('bcdmkl,amij->bcdaklij',T3,J4, optimize='optimal')
    
    del J4
    
    #del T3
    
    #Contraction 3689; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bcdaklij->abcdijkl',D33)
    
    #Contraction 3690; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('acdbklij->abcdijkl',D33)
    
    #Contraction 3691; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('abdcklij->abcdijkl',D33)
    
    #Contraction 3692; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('abcdklij->abcdijkl',D33)
    
    #Contraction 3693; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bcdajlik->abcdijkl',D33)
    
    #Contraction 3694; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('acdbjlik->abcdijkl',D33)
    
    #Contraction 3695; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('abdcjlik->abcdijkl',D33)
    
    #Contraction 3696; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('abcdjlik->abcdijkl',D33)
    
    #Contraction 3697; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bcdajkil->abcdijkl',D33)
    
    #Contraction 3698; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('acdbjkil->abcdijkl',D33)
    
    #Contraction 3699; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('abdcjkil->abcdijkl',D33)
    
    #Contraction 3700; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('abcdjkil->abcdijkl',D33)
    
    #Contraction 3701; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bcdaklji->abcdijkl',D33)
    
    #Contraction 3702; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('acdbklji->abcdijkl',D33)
    
    #Contraction 3703; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('abdcklji->abcdijkl',D33)
    
    #Contraction 3704; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('abcdklji->abcdijkl',D33)
    
    #Contraction 3705; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bcdailjk->abcdijkl',D33)
    
    #Contraction 3706; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('acdbiljk->abcdijkl',D33)
    
    #Contraction 3707; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('abdciljk->abcdijkl',D33)
    
    #Contraction 3708; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('abcdiljk->abcdijkl',D33)
    
    #Contraction 3709; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bcdaikjl->abcdijkl',D33)
    
    #Contraction 3710; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('acdbikjl->abcdijkl',D33)
    
    #Contraction 3711; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('abdcikjl->abcdijkl',D33)
    
    #Contraction 3712; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('abcdikjl->abcdijkl',D33)
    
    #Contraction 3713; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bcdajlki->abcdijkl',D33)
    
    #Contraction 3714; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('acdbjlki->abcdijkl',D33)
    
    #Contraction 3715; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('abdcjlki->abcdijkl',D33)
    
    #Contraction 3716; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('abcdjlki->abcdijkl',D33)
    
    #Contraction 3717; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bcdailkj->abcdijkl',D33)
    
    #Contraction 3718; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('acdbilkj->abcdijkl',D33)
    
    #Contraction 3719; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('abdcilkj->abcdijkl',D33)
    
    #Contraction 3720; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('abcdilkj->abcdijkl',D33)
    
    #Contraction 3721; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bcdaijkl->abcdijkl',D33)
    
    #Contraction 3722; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('acdbijkl->abcdijkl',D33)
    
    #Contraction 3723; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('abdcijkl->abcdijkl',D33)
    
    #Contraction 3724; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('abcdijkl->abcdijkl',D33)
    
    #Contraction 3725; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bcdajkli->abcdijkl',D33)
    
    #Contraction 3726; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('acdbjkli->abcdijkl',D33)
    
    #Contraction 3727; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('abdcjkli->abcdijkl',D33)
    
    #Contraction 3728; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('abcdjkli->abcdijkl',D33)
    
    #Contraction 3729; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bcdaiklj->abcdijkl',D33)
    
    #Contraction 3730; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('acdbiklj->abcdijkl',D33)
    
    #Contraction 3731; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('abdciklj->abcdijkl',D33)
    
    #Contraction 3732; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('abcdiklj->abcdijkl',D33)
    
    #Contraction 3733; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bcdaijlk->abcdijkl',D33)
    
    #Contraction 3734; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('acdbijlk->abcdijkl',D33)
    
    #Contraction 3735; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('abdcijlk->abcdijkl',D33)
    
    #Contraction 3736; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('abcdijlk->abcdijkl',D33)
    
    del D33
    
    E33 = np.zeros([nvir, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 3737; Tree Level  2; Scaling  6/ 4 Result_size  4/ 4
    E33 += np.einsum('mi,bcdajklm->bcdaijkl',F1,T4, optimize='optimal')
    
    #del F1
    
    #Contraction 3738; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bcdaijkl->abcdijkl',E33)
    
    #Contraction 3739; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bcdajikl->abcdijkl',E33)
    
    #Contraction 3740; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bcdakijl->abcdijkl',E33)
    
    #Contraction 3741; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bcdalijk->abcdijkl',E33)
    
    del E33
    
    G33 = np.zeros([nvir, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 3742; Tree Level  2; Scaling  6/ 4 Result_size  4/ 4
    G33 += np.einsum('abcdnjkl,ni->abcdjkli',T4,J8, optimize='optimal')
    
    del J8
    
    #Contraction 3743; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('abcdjkli->abcdijkl',G33)
    
    #Contraction 3744; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('abcdiklj->abcdijkl',G33)
    
    #Contraction 3745; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('abcdijlk->abcdijkl',G33)
    
    #Contraction 3746; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('abcdijkl->abcdijkl',G33)
    
    del G33
    
    I33 = np.zeros([nocc, nocc, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 3747; Tree Level  3; Scaling  6/ 6 Result_size  6/ 2
    I33 += np.einsum('mnef,efcdijkl->mncdijkl',V7,T4, optimize='optimal')
    
    #del T4
    
    #del V7
    
    J33 = np.zeros([nvir, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 3748; Tree Level  2; Scaling  8/ 4 Result_size  4/ 4
    J33 += np.einsum('abmn,mncdijkl->abcdijkl',T2,I33, optimize='optimal')
    
    #Contraction 3749; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('abcdijkl->abcdijkl',J33)
    
    #Contraction 3750; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('acbdijkl->abcdijkl',J33)
    
    #Contraction 3751; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('adbcijkl->abcdijkl',J33)
    
    #Contraction 3752; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('bcadijkl->abcdijkl',J33)
    
    #Contraction 3753; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('bdacijkl->abcdijkl',J33)
    
    #Contraction 3754; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('cdabijkl->abcdijkl',J33)
    
    del J33
    
    M33 = np.zeros([nvir, nocc, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 3755; Tree Level  3; Scaling  7/ 3 Result_size  5/ 3
    M33 += np.einsum('bn,mncdijkl->bmcdijkl',T1,I33, optimize='optimal')
    
    del I33
    
    X33 = np.zeros([nvir, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 3756; Tree Level  2; Scaling  6/ 4 Result_size  4/ 4
    X33 += np.einsum('am,bmcdijkl->abcdijkl',T1,M33, optimize='optimal')
    
    del M33
    
    #Contraction 3757; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('abcdijkl->abcdijkl',X33)
    
    #Contraction 3758; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('acbdijkl->abcdijkl',X33)
    
    #Contraction 3759; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('adbcijkl->abcdijkl',X33)
    
    #Contraction 3760; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('bacdijkl->abcdijkl',X33)
    
    #Contraction 3761; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('bcadijkl->abcdijkl',X33)
    
    #Contraction 3762; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('bdacijkl->abcdijkl',X33)
    
    #Contraction 3763; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('cabdijkl->abcdijkl',X33)
    
    #Contraction 3764; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('cbadijkl->abcdijkl',X33)
    
    #Contraction 3765; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('cdabijkl->abcdijkl',X33)
    
    #Contraction 3766; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('dabcijkl->abcdijkl',X33)
    
    #Contraction 3767; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * np.einsum('dbacijkl->abcdijkl',X33)
    
    #Contraction 3768; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * np.einsum('dcabijkl->abcdijkl',X33)
    
    del X33
    
    Y33 = np.zeros([nvir, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 3769; Tree Level  2; Scaling  6/ 4 Result_size  4/ 4
    Y33 += np.einsum('cdml,ambijk->cdablijk',T2,E24, optimize='optimal')
    
    del E24
    
    #Contraction 3770; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('cdablijk->abcdijkl',Y33)
    
    #Contraction 3771; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bdaclijk->abcdijkl',Y33)
    
    #Contraction 3772; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bcadlijk->abcdijkl',Y33)
    
    #Contraction 3773; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('cdbalijk->abcdijkl',Y33)
    
    #Contraction 3774; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('adbclijk->abcdijkl',Y33)
    
    #Contraction 3775; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('acbdlijk->abcdijkl',Y33)
    
    #Contraction 3776; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bdcalijk->abcdijkl',Y33)
    
    #Contraction 3777; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('adcblijk->abcdijkl',Y33)
    
    #Contraction 3778; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('abcdlijk->abcdijkl',Y33)
    
    #Contraction 3779; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bcdalijk->abcdijkl',Y33)
    
    #Contraction 3780; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('acdblijk->abcdijkl',Y33)
    
    #Contraction 3781; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('abdclijk->abcdijkl',Y33)
    
    #Contraction 3782; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('cdabkijl->abcdijkl',Y33)
    
    #Contraction 3783; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bdackijl->abcdijkl',Y33)
    
    #Contraction 3784; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bcadkijl->abcdijkl',Y33)
    
    #Contraction 3785; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('cdbakijl->abcdijkl',Y33)
    
    #Contraction 3786; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('adbckijl->abcdijkl',Y33)
    
    #Contraction 3787; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('acbdkijl->abcdijkl',Y33)
    
    #Contraction 3788; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bdcakijl->abcdijkl',Y33)
    
    #Contraction 3789; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('adcbkijl->abcdijkl',Y33)
    
    #Contraction 3790; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('abcdkijl->abcdijkl',Y33)
    
    #Contraction 3791; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bcdakijl->abcdijkl',Y33)
    
    #Contraction 3792; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('acdbkijl->abcdijkl',Y33)
    
    #Contraction 3793; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('abdckijl->abcdijkl',Y33)
    
    #Contraction 3794; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('cdabjikl->abcdijkl',Y33)
    
    #Contraction 3795; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bdacjikl->abcdijkl',Y33)
    
    #Contraction 3796; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bcadjikl->abcdijkl',Y33)
    
    #Contraction 3797; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('cdbajikl->abcdijkl',Y33)
    
    #Contraction 3798; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('adbcjikl->abcdijkl',Y33)
    
    #Contraction 3799; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('acbdjikl->abcdijkl',Y33)
    
    #Contraction 3800; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bdcajikl->abcdijkl',Y33)
    
    #Contraction 3801; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('adcbjikl->abcdijkl',Y33)
    
    #Contraction 3802; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('abcdjikl->abcdijkl',Y33)
    
    #Contraction 3803; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bcdajikl->abcdijkl',Y33)
    
    #Contraction 3804; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('acdbjikl->abcdijkl',Y33)
    
    #Contraction 3805; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('abdcjikl->abcdijkl',Y33)
    
    #Contraction 3806; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('cdabljik->abcdijkl',Y33)
    
    #Contraction 3807; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bdacljik->abcdijkl',Y33)
    
    #Contraction 3808; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bcadljik->abcdijkl',Y33)
    
    #Contraction 3809; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('cdbaljik->abcdijkl',Y33)
    
    #Contraction 3810; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('adbcljik->abcdijkl',Y33)
    
    #Contraction 3811; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('acbdljik->abcdijkl',Y33)
    
    #Contraction 3812; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bdcaljik->abcdijkl',Y33)
    
    #Contraction 3813; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('adcbljik->abcdijkl',Y33)
    
    #Contraction 3814; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('abcdljik->abcdijkl',Y33)
    
    #Contraction 3815; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bcdaljik->abcdijkl',Y33)
    
    #Contraction 3816; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('acdbljik->abcdijkl',Y33)
    
    #Contraction 3817; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('abdcljik->abcdijkl',Y33)
    
    #Contraction 3818; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('cdabkjil->abcdijkl',Y33)
    
    #Contraction 3819; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bdackjil->abcdijkl',Y33)
    
    #Contraction 3820; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bcadkjil->abcdijkl',Y33)
    
    #Contraction 3821; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('cdbakjil->abcdijkl',Y33)
    
    #Contraction 3822; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('adbckjil->abcdijkl',Y33)
    
    #Contraction 3823; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('acbdkjil->abcdijkl',Y33)
    
    #Contraction 3824; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bdcakjil->abcdijkl',Y33)
    
    #Contraction 3825; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('adcbkjil->abcdijkl',Y33)
    
    #Contraction 3826; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('abcdkjil->abcdijkl',Y33)
    
    #Contraction 3827; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bcdakjil->abcdijkl',Y33)
    
    #Contraction 3828; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('acdbkjil->abcdijkl',Y33)
    
    #Contraction 3829; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('abdckjil->abcdijkl',Y33)
    
    #Contraction 3830; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('cdabijkl->abcdijkl',Y33)
    
    #Contraction 3831; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bdacijkl->abcdijkl',Y33)
    
    #Contraction 3832; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bcadijkl->abcdijkl',Y33)
    
    #Contraction 3833; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('cdbaijkl->abcdijkl',Y33)
    
    #Contraction 3834; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('adbcijkl->abcdijkl',Y33)
    
    #Contraction 3835; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('acbdijkl->abcdijkl',Y33)
    
    #Contraction 3836; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bdcaijkl->abcdijkl',Y33)
    
    #Contraction 3837; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('adcbijkl->abcdijkl',Y33)
    
    #Contraction 3838; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('abcdijkl->abcdijkl',Y33)
    
    #Contraction 3839; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bcdaijkl->abcdijkl',Y33)
    
    #Contraction 3840; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('acdbijkl->abcdijkl',Y33)
    
    #Contraction 3841; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('abdcijkl->abcdijkl',Y33)
    
    #Contraction 3842; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('cdablkij->abcdijkl',Y33)
    
    #Contraction 3843; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bdaclkij->abcdijkl',Y33)
    
    #Contraction 3844; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bcadlkij->abcdijkl',Y33)
    
    #Contraction 3845; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('cdbalkij->abcdijkl',Y33)
    
    #Contraction 3846; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('adbclkij->abcdijkl',Y33)
    
    #Contraction 3847; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('acbdlkij->abcdijkl',Y33)
    
    #Contraction 3848; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bdcalkij->abcdijkl',Y33)
    
    #Contraction 3849; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('adcblkij->abcdijkl',Y33)
    
    #Contraction 3850; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('abcdlkij->abcdijkl',Y33)
    
    #Contraction 3851; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bcdalkij->abcdijkl',Y33)
    
    #Contraction 3852; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('acdblkij->abcdijkl',Y33)
    
    #Contraction 3853; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('abdclkij->abcdijkl',Y33)
    
    #Contraction 3854; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('cdabjkil->abcdijkl',Y33)
    
    #Contraction 3855; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bdacjkil->abcdijkl',Y33)
    
    #Contraction 3856; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bcadjkil->abcdijkl',Y33)
    
    #Contraction 3857; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('cdbajkil->abcdijkl',Y33)
    
    #Contraction 3858; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('adbcjkil->abcdijkl',Y33)
    
    #Contraction 3859; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('acbdjkil->abcdijkl',Y33)
    
    #Contraction 3860; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bdcajkil->abcdijkl',Y33)
    
    #Contraction 3861; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('adcbjkil->abcdijkl',Y33)
    
    #Contraction 3862; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('abcdjkil->abcdijkl',Y33)
    
    #Contraction 3863; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bcdajkil->abcdijkl',Y33)
    
    #Contraction 3864; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('acdbjkil->abcdijkl',Y33)
    
    #Contraction 3865; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('abdcjkil->abcdijkl',Y33)
    
    #Contraction 3866; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('cdabikjl->abcdijkl',Y33)
    
    #Contraction 3867; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bdacikjl->abcdijkl',Y33)
    
    #Contraction 3868; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bcadikjl->abcdijkl',Y33)
    
    #Contraction 3869; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('cdbaikjl->abcdijkl',Y33)
    
    #Contraction 3870; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('adbcikjl->abcdijkl',Y33)
    
    #Contraction 3871; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('acbdikjl->abcdijkl',Y33)
    
    #Contraction 3872; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bdcaikjl->abcdijkl',Y33)
    
    #Contraction 3873; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('adcbikjl->abcdijkl',Y33)
    
    #Contraction 3874; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('abcdikjl->abcdijkl',Y33)
    
    #Contraction 3875; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bcdaikjl->abcdijkl',Y33)
    
    #Contraction 3876; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('acdbikjl->abcdijkl',Y33)
    
    #Contraction 3877; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('abdcikjl->abcdijkl',Y33)
    
    #Contraction 3878; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('cdabklij->abcdijkl',Y33)
    
    #Contraction 3879; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bdacklij->abcdijkl',Y33)
    
    #Contraction 3880; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bcadklij->abcdijkl',Y33)
    
    #Contraction 3881; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('cdbaklij->abcdijkl',Y33)
    
    #Contraction 3882; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('adbcklij->abcdijkl',Y33)
    
    #Contraction 3883; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('acbdklij->abcdijkl',Y33)
    
    #Contraction 3884; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bdcaklij->abcdijkl',Y33)
    
    #Contraction 3885; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('adcbklij->abcdijkl',Y33)
    
    #Contraction 3886; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('abcdklij->abcdijkl',Y33)
    
    #Contraction 3887; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bcdaklij->abcdijkl',Y33)
    
    #Contraction 3888; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('acdbklij->abcdijkl',Y33)
    
    #Contraction 3889; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('abdcklij->abcdijkl',Y33)
    
    #Contraction 3890; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('cdabjlik->abcdijkl',Y33)
    
    #Contraction 3891; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bdacjlik->abcdijkl',Y33)
    
    #Contraction 3892; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bcadjlik->abcdijkl',Y33)
    
    #Contraction 3893; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('cdbajlik->abcdijkl',Y33)
    
    #Contraction 3894; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('adbcjlik->abcdijkl',Y33)
    
    #Contraction 3895; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('acbdjlik->abcdijkl',Y33)
    
    #Contraction 3896; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bdcajlik->abcdijkl',Y33)
    
    #Contraction 3897; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('adcbjlik->abcdijkl',Y33)
    
    #Contraction 3898; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('abcdjlik->abcdijkl',Y33)
    
    #Contraction 3899; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bcdajlik->abcdijkl',Y33)
    
    #Contraction 3900; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('acdbjlik->abcdijkl',Y33)
    
    #Contraction 3901; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('abdcjlik->abcdijkl',Y33)
    
    #Contraction 3902; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('cdabiljk->abcdijkl',Y33)
    
    #Contraction 3903; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bdaciljk->abcdijkl',Y33)
    
    #Contraction 3904; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bcadiljk->abcdijkl',Y33)
    
    #Contraction 3905; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('cdbailjk->abcdijkl',Y33)
    
    #Contraction 3906; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('adbciljk->abcdijkl',Y33)
    
    #Contraction 3907; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('acbdiljk->abcdijkl',Y33)
    
    #Contraction 3908; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bdcailjk->abcdijkl',Y33)
    
    #Contraction 3909; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('adcbiljk->abcdijkl',Y33)
    
    #Contraction 3910; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('abcdiljk->abcdijkl',Y33)
    
    #Contraction 3911; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bcdailjk->abcdijkl',Y33)
    
    #Contraction 3912; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('acdbiljk->abcdijkl',Y33)
    
    #Contraction 3913; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('abdciljk->abcdijkl',Y33)
    
    del Y33
    
    A34 = np.zeros([nvir, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 3914; Tree Level  2; Scaling  6/ 4 Result_size  4/ 4
    A34 += np.einsum('bm,amcdijkl->bacdijkl',T1,X27, optimize='optimal')
    
    del X27
    
    #del T1
    
    #Contraction 3915; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('bacdijkl->abcdijkl',A34)
    
    #Contraction 3916; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('cabdijkl->abcdijkl',A34)
    
    #Contraction 3917; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('dabcijkl->abcdijkl',A34)
    
    #Contraction 3918; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('abcdijkl->abcdijkl',A34)
    
    #Contraction 3919; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('cbadijkl->abcdijkl',A34)
    
    #Contraction 3920; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('dbacijkl->abcdijkl',A34)
    
    #Contraction 3921; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('acbdijkl->abcdijkl',A34)
    
    #Contraction 3922; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('bcadijkl->abcdijkl',A34)
    
    #Contraction 3923; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('dcabijkl->abcdijkl',A34)
    
    #Contraction 3924; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('adbcijkl->abcdijkl',A34)
    
    #Contraction 3925; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * np.einsum('bdacijkl->abcdijkl',A34)
    
    #Contraction 3926; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * np.einsum('cdabijkl->abcdijkl',A34)
    
    del A34
    
    D34 = np.zeros([nvir, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 3927; Tree Level  2; Scaling  6/ 4 Result_size  4/ 4
    D34 += np.einsum('bdml,camjki->bdcaljki',T2,J20, optimize='optimal')
    
    del J20
    
    #del T2
    
    #Contraction 3928; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bdcaljki->abcdijkl',D34)
    
    #Contraction 3929; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('cdbaljki->abcdijkl',D34)
    
    #Contraction 3930; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('cbdaljki->abcdijkl',D34)
    
    #Contraction 3931; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bdacljki->abcdijkl',D34)
    
    #Contraction 3932; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('adbcljki->abcdijkl',D34)
    
    #Contraction 3933; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('abdcljki->abcdijkl',D34)
    
    #Contraction 3934; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('cdabljki->abcdijkl',D34)
    
    #Contraction 3935; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('adcbljki->abcdijkl',D34)
    
    #Contraction 3936; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('acdbljki->abcdijkl',D34)
    
    #Contraction 3937; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('cbadljki->abcdijkl',D34)
    
    #Contraction 3938; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('abcdljki->abcdijkl',D34)
    
    #Contraction 3939; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('acbdljki->abcdijkl',D34)
    
    #Contraction 3940; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bdcakjli->abcdijkl',D34)
    
    #Contraction 3941; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('cdbakjli->abcdijkl',D34)
    
    #Contraction 3942; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('cbdakjli->abcdijkl',D34)
    
    #Contraction 3943; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bdackjli->abcdijkl',D34)
    
    #Contraction 3944; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('adbckjli->abcdijkl',D34)
    
    #Contraction 3945; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('abdckjli->abcdijkl',D34)
    
    #Contraction 3946; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('cdabkjli->abcdijkl',D34)
    
    #Contraction 3947; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('adcbkjli->abcdijkl',D34)
    
    #Contraction 3948; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('acdbkjli->abcdijkl',D34)
    
    #Contraction 3949; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('cbadkjli->abcdijkl',D34)
    
    #Contraction 3950; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('abcdkjli->abcdijkl',D34)
    
    #Contraction 3951; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('acbdkjli->abcdijkl',D34)
    
    #Contraction 3952; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bdcajkli->abcdijkl',D34)
    
    #Contraction 3953; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('cdbajkli->abcdijkl',D34)
    
    #Contraction 3954; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('cbdajkli->abcdijkl',D34)
    
    #Contraction 3955; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bdacjkli->abcdijkl',D34)
    
    #Contraction 3956; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('adbcjkli->abcdijkl',D34)
    
    #Contraction 3957; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('abdcjkli->abcdijkl',D34)
    
    #Contraction 3958; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('cdabjkli->abcdijkl',D34)
    
    #Contraction 3959; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('adcbjkli->abcdijkl',D34)
    
    #Contraction 3960; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('acdbjkli->abcdijkl',D34)
    
    #Contraction 3961; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('cbadjkli->abcdijkl',D34)
    
    #Contraction 3962; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('abcdjkli->abcdijkl',D34)
    
    #Contraction 3963; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('acbdjkli->abcdijkl',D34)
    
    #Contraction 3964; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bdcalikj->abcdijkl',D34)
    
    #Contraction 3965; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('cdbalikj->abcdijkl',D34)
    
    #Contraction 3966; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('cbdalikj->abcdijkl',D34)
    
    #Contraction 3967; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bdaclikj->abcdijkl',D34)
    
    #Contraction 3968; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('adbclikj->abcdijkl',D34)
    
    #Contraction 3969; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('abdclikj->abcdijkl',D34)
    
    #Contraction 3970; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('cdablikj->abcdijkl',D34)
    
    #Contraction 3971; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('adcblikj->abcdijkl',D34)
    
    #Contraction 3972; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('acdblikj->abcdijkl',D34)
    
    #Contraction 3973; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('cbadlikj->abcdijkl',D34)
    
    #Contraction 3974; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('abcdlikj->abcdijkl',D34)
    
    #Contraction 3975; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('acbdlikj->abcdijkl',D34)
    
    #Contraction 3976; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bdcakilj->abcdijkl',D34)
    
    #Contraction 3977; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('cdbakilj->abcdijkl',D34)
    
    #Contraction 3978; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('cbdakilj->abcdijkl',D34)
    
    #Contraction 3979; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bdackilj->abcdijkl',D34)
    
    #Contraction 3980; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('adbckilj->abcdijkl',D34)
    
    #Contraction 3981; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('abdckilj->abcdijkl',D34)
    
    #Contraction 3982; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('cdabkilj->abcdijkl',D34)
    
    #Contraction 3983; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('adcbkilj->abcdijkl',D34)
    
    #Contraction 3984; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('acdbkilj->abcdijkl',D34)
    
    #Contraction 3985; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('cbadkilj->abcdijkl',D34)
    
    #Contraction 3986; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('abcdkilj->abcdijkl',D34)
    
    #Contraction 3987; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('acbdkilj->abcdijkl',D34)
    
    #Contraction 3988; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bdcaiklj->abcdijkl',D34)
    
    #Contraction 3989; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('cdbaiklj->abcdijkl',D34)
    
    #Contraction 3990; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('cbdaiklj->abcdijkl',D34)
    
    #Contraction 3991; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bdaciklj->abcdijkl',D34)
    
    #Contraction 3992; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('adbciklj->abcdijkl',D34)
    
    #Contraction 3993; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('abdciklj->abcdijkl',D34)
    
    #Contraction 3994; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('cdabiklj->abcdijkl',D34)
    
    #Contraction 3995; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('adcbiklj->abcdijkl',D34)
    
    #Contraction 3996; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('acdbiklj->abcdijkl',D34)
    
    #Contraction 3997; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('cbadiklj->abcdijkl',D34)
    
    #Contraction 3998; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('abcdiklj->abcdijkl',D34)
    
    #Contraction 3999; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('acbdiklj->abcdijkl',D34)
    
    #Contraction 4000; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bdcalijk->abcdijkl',D34)
    
    #Contraction 4001; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('cdbalijk->abcdijkl',D34)
    
    #Contraction 4002; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('cbdalijk->abcdijkl',D34)
    
    #Contraction 4003; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bdaclijk->abcdijkl',D34)
    
    #Contraction 4004; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('adbclijk->abcdijkl',D34)
    
    #Contraction 4005; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('abdclijk->abcdijkl',D34)
    
    #Contraction 4006; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('cdablijk->abcdijkl',D34)
    
    #Contraction 4007; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('adcblijk->abcdijkl',D34)
    
    #Contraction 4008; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('acdblijk->abcdijkl',D34)
    
    #Contraction 4009; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('cbadlijk->abcdijkl',D34)
    
    #Contraction 4010; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('abcdlijk->abcdijkl',D34)
    
    #Contraction 4011; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('acbdlijk->abcdijkl',D34)
    
    #Contraction 4012; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bdcajilk->abcdijkl',D34)
    
    #Contraction 4013; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('cdbajilk->abcdijkl',D34)
    
    #Contraction 4014; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('cbdajilk->abcdijkl',D34)
    
    #Contraction 4015; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bdacjilk->abcdijkl',D34)
    
    #Contraction 4016; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('adbcjilk->abcdijkl',D34)
    
    #Contraction 4017; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('abdcjilk->abcdijkl',D34)
    
    #Contraction 4018; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('cdabjilk->abcdijkl',D34)
    
    #Contraction 4019; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('adcbjilk->abcdijkl',D34)
    
    #Contraction 4020; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('acdbjilk->abcdijkl',D34)
    
    #Contraction 4021; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('cbadjilk->abcdijkl',D34)
    
    #Contraction 4022; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('abcdjilk->abcdijkl',D34)
    
    #Contraction 4023; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('acbdjilk->abcdijkl',D34)
    
    #Contraction 4024; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bdcaijlk->abcdijkl',D34)
    
    #Contraction 4025; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('cdbaijlk->abcdijkl',D34)
    
    #Contraction 4026; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('cbdaijlk->abcdijkl',D34)
    
    #Contraction 4027; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bdacijlk->abcdijkl',D34)
    
    #Contraction 4028; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('adbcijlk->abcdijkl',D34)
    
    #Contraction 4029; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('abdcijlk->abcdijkl',D34)
    
    #Contraction 4030; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('cdabijlk->abcdijkl',D34)
    
    #Contraction 4031; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('adcbijlk->abcdijkl',D34)
    
    #Contraction 4032; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('acdbijlk->abcdijkl',D34)
    
    #Contraction 4033; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('cbadijlk->abcdijkl',D34)
    
    #Contraction 4034; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('abcdijlk->abcdijkl',D34)
    
    #Contraction 4035; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('acbdijlk->abcdijkl',D34)
    
    #Contraction 4036; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bdcakijl->abcdijkl',D34)
    
    #Contraction 4037; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('cdbakijl->abcdijkl',D34)
    
    #Contraction 4038; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('cbdakijl->abcdijkl',D34)
    
    #Contraction 4039; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bdackijl->abcdijkl',D34)
    
    #Contraction 4040; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('adbckijl->abcdijkl',D34)
    
    #Contraction 4041; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('abdckijl->abcdijkl',D34)
    
    #Contraction 4042; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('cdabkijl->abcdijkl',D34)
    
    #Contraction 4043; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('adcbkijl->abcdijkl',D34)
    
    #Contraction 4044; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('acdbkijl->abcdijkl',D34)
    
    #Contraction 4045; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('cbadkijl->abcdijkl',D34)
    
    #Contraction 4046; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('abcdkijl->abcdijkl',D34)
    
    #Contraction 4047; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('acbdkijl->abcdijkl',D34)
    
    #Contraction 4048; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bdcajikl->abcdijkl',D34)
    
    #Contraction 4049; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('cdbajikl->abcdijkl',D34)
    
    #Contraction 4050; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('cbdajikl->abcdijkl',D34)
    
    #Contraction 4051; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bdacjikl->abcdijkl',D34)
    
    #Contraction 4052; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('adbcjikl->abcdijkl',D34)
    
    #Contraction 4053; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('abdcjikl->abcdijkl',D34)
    
    #Contraction 4054; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('cdabjikl->abcdijkl',D34)
    
    #Contraction 4055; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('adcbjikl->abcdijkl',D34)
    
    #Contraction 4056; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('acdbjikl->abcdijkl',D34)
    
    #Contraction 4057; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('cbadjikl->abcdijkl',D34)
    
    #Contraction 4058; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('abcdjikl->abcdijkl',D34)
    
    #Contraction 4059; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('acbdjikl->abcdijkl',D34)
    
    #Contraction 4060; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('bdcaijkl->abcdijkl',D34)
    
    #Contraction 4061; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('cdbaijkl->abcdijkl',D34)
    
    #Contraction 4062; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('cbdaijkl->abcdijkl',D34)
    
    #Contraction 4063; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('bdacijkl->abcdijkl',D34)
    
    #Contraction 4064; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('adbcijkl->abcdijkl',D34)
    
    #Contraction 4065; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('abdcijkl->abcdijkl',D34)
    
    #Contraction 4066; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('cdabijkl->abcdijkl',D34)
    
    #Contraction 4067; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('adcbijkl->abcdijkl',D34)
    
    #Contraction 4068; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('acdbijkl->abcdijkl',D34)
    
    #Contraction 4069; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('cbadijkl->abcdijkl',D34)
    
    #Contraction 4070; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += np.einsum('abcdijkl->abcdijkl',D34)
    
    #Contraction 4071; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * np.einsum('acbdijkl->abcdijkl',D34)
    
    del D34
    
    return([Z0[0], Z1, Z2, Z3, Z4])
    
# end of numpy_tenpi_ccsdtq
