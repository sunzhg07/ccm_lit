#!/usr/bin/python
# -*- coding: utf-8 -*-

# CCSDTQ_T6 for simple use with numpy

import numpy as np #numpy arrays
import opt_einsum as oe
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
    A1 += oe.contract('ijab,ai->jb',V7,T1, optimize='optimal')
    
    #Contraction 2; Tree Level  0; Scaling  2/ 2 Result_size  0/ 0
    Z0 += oe.contract('ia,ai->',F3,T1, optimize='optimal')
    
    #Contraction 3; Tree Level  1; Scaling  2/ 2 Result_size  0/ 0
    Z0 += 0.5 * oe.contract('bj,jb->',T1,A1, optimize='optimal')
    
    #Contraction 4; Tree Level  1; Scaling  4/ 4 Result_size  0/ 0
    Z0 += 0.25 * oe.contract('ijab,abij->',V7,T2, optimize='optimal')
    
    I1 = np.zeros([nocc, nocc, nvir, nocc], dtype=type_)
    
    #Contraction 5; Tree Level  3; Scaling  3/ 3 Result_size  3/ 1
    I1 += oe.contract('jkbc,bi->jkci',V7,T1, optimize='optimal')
    
    I3 = np.zeros([nocc, nocc], dtype=type_)
    
    #Contraction 6; Tree Level  2; Scaling  4/ 2 Result_size  2/ 0
    I3 += -2.0 * oe.contract('jkib,bk->ji',V4,T1, optimize='optimal')
    
    M1 = np.zeros([nocc, nocc], dtype=type_)
    
    #Contraction 7; Tree Level  3; Scaling  4/ 4 Result_size  2/ 0
    M1 += oe.contract('jkbc,bcki->ji',V7,T2, optimize='optimal')
    
    X1 = np.zeros([nvir, nvir], dtype=type_)
    
    #Contraction 8; Tree Level  3; Scaling  4/ 4 Result_size  0/ 2
    X1 += oe.contract('klcd,dbkl->bc',V7,T2, optimize='optimal')
    
    Y1 = np.zeros([nocc, nocc], dtype=type_)
    
    #Contraction 9; Tree Level  3; Scaling  2/ 2 Result_size  2/ 0
    Y1 += oe.contract('jb,bi->ji',F3,T1, optimize='optimal')
    
    #Contraction 10; Tree Level  2; Scaling  2/ 0 Result_size  2/ 0
    I3 += -2.0 * oe.contract('ji->ji',Y1)
    
    #Contraction 11; Tree Level  0; Scaling  5/ 3 Result_size  1/ 1
    Z1 += -0.5 * oe.contract('jkib,abjk->ai',V4,T2, optimize='optimal')
    
    #Contraction 12; Tree Level  1; Scaling  5/ 5 Result_size  1/ 1
    Z1 += 0.25 * oe.contract('jkbc,abcijk->ai',V7,T3, optimize='optimal')
    
    G2 = np.zeros([nocc, nvir, nvir, nocc], dtype=type_)
    
    #Contraction 13; Tree Level  3; Scaling  4/ 4 Result_size  2/ 2
    G2 += oe.contract('klcd,dblj->kbcj',V7,T2, optimize='optimal')
    
    I2 = np.zeros([nocc, nvir, nvir, nocc], dtype=type_)
    
    #Contraction 14; Tree Level  2; Scaling  2/ 2 Result_size  2/ 2
    I2 += oe.contract('jabi->jabi',G2)
    
    #Contraction 15; Tree Level  2; Scaling  2/ 2 Result_size  2/ 2
    I2 += oe.contract('ajib->jabi',V5)
    
    X2 = np.zeros([nvir, nvir], dtype=type_)
    
    #Contraction 16; Tree Level  2; Scaling  2/ 4 Result_size  0/ 2
    X2 += 2.0 * oe.contract('ajbc,cj->ab',V8,T1, optimize='optimal')
    
    #Contraction 17; Tree Level  2; Scaling  0/ 2 Result_size  0/ 2
    X2 += oe.contract('ab->ab',X1)
    
    #Contraction 18; Tree Level  2; Scaling  0/ 2 Result_size  0/ 2
    X2 += 2.0 * oe.contract('ab->ab',F4)
    
    #Contraction 19; Tree Level  1; Scaling  1/ 3 Result_size  1/ 1
    Z1 += 0.5 * oe.contract('bi,ab->ai',T1,X2, optimize='optimal')
    
    del X2
    
    #Contraction 20; Tree Level  2; Scaling  2/ 0 Result_size  2/ 0
    I3 += -2.0 * oe.contract('ji->ji',F1)
    
    #Contraction 21; Tree Level  1; Scaling  3/ 3 Result_size  1/ 1
    Z1 += oe.contract('jb,abij->ai',F3,T2, optimize='optimal')
    
    #Contraction 22; Tree Level  2; Scaling  4/ 2 Result_size  2/ 0
    I3 += -2.0 * oe.contract('ck,jkci->ji',T1,I1, optimize='optimal')
    
    #Contraction 23; Tree Level  2; Scaling  2/ 0 Result_size  2/ 0
    I3 += oe.contract('ji->ji',M1)
    
    #Contraction 24; Tree Level  1; Scaling  3/ 1 Result_size  1/ 1
    Z1 += 0.5 * oe.contract('aj,ji->ai',T1,I3, optimize='optimal')
    
    del I3
    
    #Contraction 25; Tree Level  1; Scaling  3/ 5 Result_size  1/ 1
    Z1 += 0.5 * oe.contract('ajbc,bcij->ai',V8,T2, optimize='optimal')
    
    #Contraction 26; Tree Level  1; Scaling  3/ 3 Result_size  1/ 1
    Z1 += oe.contract('bj,jabi->ai',T1,I2, optimize='optimal')
    
    del I2
    
    #Contraction 27; Tree Level  1; Scaling  1/ 1 Result_size  1/ 1
    Z1 += oe.contract('ai->ai',F2)
    
    #del F2
    
    Y3 = np.zeros([nocc, nvir, nocc, nocc], dtype=type_)
    
    #Contraction 28; Tree Level  3; Scaling  5/ 3 Result_size  3/ 1
    Y3 += oe.contract('klic,cblj->kbij',V4,T2, optimize='optimal')
    
    A4 = np.zeros([nvir, nvir, nocc, nocc], dtype=type_)
    
    #Contraction 29; Tree Level  2; Scaling  4/ 6 Result_size  2/ 2
    A4 += oe.contract('akcd,bcdjik->abji',V8,T3, optimize='optimal')
    
    #Contraction 30; Tree Level  0; Scaling  2/ 2 Result_size  2/ 2
    Z2 += 0.5 * oe.contract('abji->abij',A4)
    
    #Contraction 31; Tree Level  1; Scaling  2/ 2 Result_size  2/ 2
    Z2 += -0.5 * oe.contract('baji->abij',A4)
    
    del A4
    
    #Contraction 32; Tree Level  1; Scaling  4/ 4 Result_size  2/ 2
    Z2 += oe.contract('dablij,ld->abij',T3,A1, optimize='optimal')
    
    E4 = np.zeros([nvir, nvir, nocc, nocc], dtype=type_)
    
    #Contraction 33; Tree Level  2; Scaling  4/ 2 Result_size  2/ 2
    E4 += oe.contract('abkj,ki->abji',T2,Y1, optimize='optimal')
    
    #Contraction 34; Tree Level  1; Scaling  2/ 2 Result_size  2/ 2
    Z2 += -1.0 * oe.contract('abji->abij',E4)
    
    #Contraction 35; Tree Level  1; Scaling  2/ 2 Result_size  2/ 2
    Z2 += oe.contract('abij->abij',E4)
    
    del E4
    
    G4 = np.zeros([nvir, nvir, nocc, nocc], dtype=type_)
    
    #Contraction 36; Tree Level  2; Scaling  4/ 2 Result_size  2/ 2
    G4 += oe.contract('ki,bajk->baij',F1,T2, optimize='optimal')
    
    #Contraction 37; Tree Level  1; Scaling  2/ 2 Result_size  2/ 2
    Z2 += -1.0 * oe.contract('baij->abij',G4)
    
    #Contraction 38; Tree Level  1; Scaling  2/ 2 Result_size  2/ 2
    Z2 += oe.contract('baji->abij',G4)
    
    del G4
    
    #Contraction 39; Tree Level  1; Scaling  6/ 6 Result_size  2/ 2
    Z2 += 0.25 * oe.contract('klcd,abcdijkl->abij',V7,T4, optimize='optimal')
    
    J4 = np.zeros([nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 40; Tree Level  3; Scaling  3/ 3 Result_size  3/ 1
    J4 += oe.contract('akic,cj->akij',V5,T1, optimize='optimal')
    
    M4 = np.zeros([nvir, nocc, nvir, nocc], dtype=type_)
    
    #Contraction 41; Tree Level  4; Scaling  2/ 4 Result_size  2/ 2
    M4 += oe.contract('akcd,dj->akcj',V8,T1, optimize='optimal')
    
    X4 = np.zeros([nvir, nvir, nocc, nocc], dtype=type_)
    
    #Contraction 42; Tree Level  2; Scaling  4/ 4 Result_size  2/ 2
    X4 += oe.contract('akic,bcjk->abij',V5,T2, optimize='optimal')
    
    #Contraction 43; Tree Level  1; Scaling  2/ 2 Result_size  2/ 2
    Z2 += oe.contract('abij->abij',X4)
    
    #Contraction 44; Tree Level  1; Scaling  2/ 2 Result_size  2/ 2
    Z2 += -1.0 * oe.contract('baij->abij',X4)
    
    #Contraction 45; Tree Level  1; Scaling  2/ 2 Result_size  2/ 2
    Z2 += -1.0 * oe.contract('abji->abij',X4)
    
    #Contraction 46; Tree Level  1; Scaling  2/ 2 Result_size  2/ 2
    Z2 += oe.contract('baji->abij',X4)
    
    del X4
    
    Y4 = np.zeros([nvir, nvir, nocc, nocc], dtype=type_)
    
    #Contraction 47; Tree Level  2; Scaling  4/ 2 Result_size  2/ 2
    Y4 += oe.contract('akij,bk->abij',V2,T1, optimize='optimal')
    
    #Contraction 48; Tree Level  1; Scaling  2/ 2 Result_size  2/ 2
    Z2 += -1.0 * oe.contract('abij->abij',Y4)
    
    #Contraction 49; Tree Level  1; Scaling  2/ 2 Result_size  2/ 2
    Z2 += oe.contract('baij->abij',Y4)
    
    del Y4
    
    A5 = np.zeros([nocc, nocc], dtype=type_)
    
    #Contraction 50; Tree Level  3; Scaling  4/ 2 Result_size  2/ 0
    A5 += oe.contract('dk,kldi->li',T1,I1, optimize='optimal')
    
    D5 = np.zeros([nvir, nvir, nocc, nocc], dtype=type_)
    
    #Contraction 51; Tree Level  2; Scaling  2/ 4 Result_size  2/ 2
    D5 += oe.contract('abic,cj->abij',V6,T1, optimize='optimal')
    
    #Contraction 52; Tree Level  1; Scaling  2/ 2 Result_size  2/ 2
    Z2 += oe.contract('abij->abij',D5)
    
    #Contraction 53; Tree Level  1; Scaling  2/ 2 Result_size  2/ 2
    Z2 += -1.0 * oe.contract('abji->abij',D5)
    
    del D5
    
    E5 = np.zeros([nocc, nvir, nocc, nocc], dtype=type_)
    
    #Contraction 54; Tree Level  3; Scaling  3/ 3 Result_size  3/ 1
    E5 += oe.contract('kc,cbij->kbij',F3,T2, optimize='optimal')
    
    G5 = np.zeros([nvir, nocc, nvir, nocc], dtype=type_)
    
    #Contraction 55; Tree Level  3; Scaling  2/ 4 Result_size  2/ 2
    G5 += oe.contract('akcd,ci->akdi',V8,T1, optimize='optimal')
    
    I5 = np.zeros([nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 56; Tree Level  2; Scaling  4/ 4 Result_size  4/ 0
    I5 += oe.contract('klcd,cdij->klij',V7,T2, optimize='optimal')
    
    J5 = np.zeros([nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 57; Tree Level  2; Scaling  5/ 1 Result_size  3/ 1
    J5 += oe.contract('bl,klij->bkij',T1,I5, optimize='optimal')
    
    M5 = np.zeros([nvir, nvir, nvir, nocc], dtype=type_)
    
    #Contraction 58; Tree Level  3; Scaling  1/ 5 Result_size  1/ 3
    M5 += oe.contract('abcd,dj->abcj',V9,T1, optimize='optimal')
    
    X5 = np.zeros([nvir, nvir, nocc, nocc], dtype=type_)
    
    #Contraction 59; Tree Level  2; Scaling  2/ 4 Result_size  2/ 2
    X5 += oe.contract('acij,bc->abij',T2,X1, optimize='optimal')
    
    del X1
    
    #Contraction 60; Tree Level  1; Scaling  2/ 2 Result_size  2/ 2
    Z2 += 0.5 * oe.contract('abij->abij',X5)
    
    #Contraction 61; Tree Level  1; Scaling  2/ 2 Result_size  2/ 2
    Z2 += -0.5 * oe.contract('baij->abij',X5)
    
    del X5
    
    Y5 = np.zeros([nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 62; Tree Level  3; Scaling  5/ 3 Result_size  3/ 1
    Y5 += oe.contract('dblj,kldi->bkji',T2,I1, optimize='optimal')
    
    A6 = np.zeros([nvir, nvir, nocc, nocc], dtype=type_)
    
    #Contraction 63; Tree Level  2; Scaling  2/ 4 Result_size  2/ 2
    A6 += oe.contract('ac,bcji->abji',F4,T2, optimize='optimal')
    
    #Contraction 64; Tree Level  1; Scaling  2/ 2 Result_size  2/ 2
    Z2 += oe.contract('abji->abij',A6)
    
    #Contraction 65; Tree Level  1; Scaling  2/ 2 Result_size  2/ 2
    Z2 += -1.0 * oe.contract('baji->abij',A6)
    
    del A6
    
    D6 = np.zeros([nocc, nvir], dtype=type_)
    
    #Contraction 66; Tree Level  4; Scaling  3/ 3 Result_size  1/ 1
    D6 += oe.contract('klcd,cl->kd',V7,T1, optimize='optimal')
    
    E6 = np.zeros([nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 67; Tree Level  3; Scaling  3/ 3 Result_size  3/ 1
    E6 += oe.contract('dbij,kd->bkij',T2,D6, optimize='optimal')
    
    G6 = np.zeros([nvir, nvir, nocc, nocc], dtype=type_)
    
    #Contraction 68; Tree Level  2; Scaling  2/ 4 Result_size  2/ 2
    G6 += oe.contract('ci,abcj->abij',T1,M5, optimize='optimal')
    
    del M5
    
    #Contraction 69; Tree Level  1; Scaling  2/ 2 Result_size  2/ 2
    Z2 += 0.5 * oe.contract('abij->abij',G6)
    
    #Contraction 70; Tree Level  1; Scaling  2/ 2 Result_size  2/ 2
    Z2 += -0.5 * oe.contract('abji->abij',G6)
    
    del G6
    
    #Contraction 71; Tree Level  1; Scaling  6/ 2 Result_size  2/ 2
    Z2 += 0.25 * oe.contract('abkl,klij->abij',T2,I5, optimize='optimal')
    
    J6 = np.zeros([nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 72; Tree Level  3; Scaling  3/ 5 Result_size  3/ 1
    J6 += oe.contract('akcd,cdij->akij',V8,T2, optimize='optimal')
    
    M6 = np.zeros([nvir, nvir, nocc, nocc], dtype=type_)
    
    #Contraction 73; Tree Level  2; Scaling  4/ 2 Result_size  2/ 2
    M6 += oe.contract('bk,akij->baij',T1,J6, optimize='optimal')
    
    #Contraction 74; Tree Level  1; Scaling  2/ 2 Result_size  2/ 2
    Z2 += -0.5 * oe.contract('baij->abij',M6)
    
    #Contraction 75; Tree Level  1; Scaling  2/ 2 Result_size  2/ 2
    Z2 += 0.5 * oe.contract('abij->abij',M6)
    
    del M6
    
    G12 = np.zeros([nocc, nvir, nocc, nocc], dtype=type_)
    
    #Contraction 76; Tree Level  2; Scaling  3/ 1 Result_size  3/ 1
    G12 += -1.0 * oe.contract('kaij->kaji',E5)
    
    #Contraction 77; Tree Level  2; Scaling  3/ 1 Result_size  3/ 1
    G12 += oe.contract('akij->kaji',E6)
    
    Y6 = np.zeros([nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 78; Tree Level  3; Scaling  4/ 2 Result_size  4/ 0
    Y6 += oe.contract('klic,cj->klij',V4,T1, optimize='optimal')
    
    G8 = np.zeros([nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 79; Tree Level  2; Scaling  4/ 0 Result_size  4/ 0
    G8 += -2.0 * oe.contract('klji->klij',Y6)
    
    #Contraction 80; Tree Level  2; Scaling  4/ 0 Result_size  4/ 0
    G8 += 2.0 * oe.contract('klij->klij',V1)
    
    A11 = np.zeros([nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 81; Tree Level  3; Scaling  4/ 0 Result_size  4/ 0
    A11 += oe.contract('klji->lkij',Y6)
    
    #Contraction 82; Tree Level  3; Scaling  4/ 0 Result_size  4/ 0
    A11 += -1.0 * oe.contract('lkji->lkij',Y6)
    
    E7 = np.zeros([nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 83; Tree Level  3; Scaling  4/ 2 Result_size  4/ 0
    E7 += oe.contract('dj,kldi->klji',T1,I1, optimize='optimal')
    
    #Contraction 84; Tree Level  2; Scaling  4/ 0 Result_size  4/ 0
    G8 += oe.contract('klji->klij',E7)
    
    #Contraction 85; Tree Level  2; Scaling  4/ 0 Result_size  4/ 0
    G8 += -1.0 * oe.contract('klij->klij',E7)
    
    I7 = np.zeros([nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 86; Tree Level  3; Scaling  5/ 1 Result_size  3/ 1
    I7 += oe.contract('al,lmji->amji',T1,E7, optimize='optimal')
    
    #Contraction 87; Tree Level  2; Scaling  4/ 0 Result_size  4/ 0
    G8 += 2.0 * oe.contract('klij->klij',Y6)
    
    #Contraction 88; Tree Level  2; Scaling  3/ 1 Result_size  3/ 1
    G12 += oe.contract('akij->kaji',J4)
    
    #Contraction 89; Tree Level  2; Scaling  3/ 1 Result_size  3/ 1
    G12 += -1.0 * oe.contract('akji->kaji',J4)
    
    A12 = np.zeros([nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 90; Tree Level  2; Scaling  3/ 1 Result_size  3/ 1
    A12 += 4.0 * oe.contract('kbij->bkji',Y3)
    
    #Contraction 91; Tree Level  2; Scaling  3/ 1 Result_size  3/ 1
    A12 += -4.0 * oe.contract('kbji->bkji',Y3)
    
    A8 = np.zeros([nvir, nvir, nocc, nocc], dtype=type_)
    
    #Contraction 92; Tree Level  2; Scaling  4/ 4 Result_size  2/ 2
    A8 += oe.contract('dbkj,akdi->baji',T2,G5, optimize='optimal')
    
    #Contraction 93; Tree Level  1; Scaling  2/ 2 Result_size  2/ 2
    Z2 += oe.contract('baji->abij',A8)
    
    #Contraction 94; Tree Level  1; Scaling  2/ 2 Result_size  2/ 2
    Z2 += -1.0 * oe.contract('abji->abij',A8)
    
    #Contraction 95; Tree Level  1; Scaling  2/ 2 Result_size  2/ 2
    Z2 += -1.0 * oe.contract('baij->abij',A8)
    
    #Contraction 96; Tree Level  1; Scaling  2/ 2 Result_size  2/ 2
    Z2 += oe.contract('abij->abij',A8)
    
    del A8
    
    D8 = np.zeros([nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 97; Tree Level  3; Scaling  5/ 1 Result_size  3/ 1
    D8 += oe.contract('bl,klji->bkji',T1,E7, optimize='optimal')
    
    #Contraction 98; Tree Level  2; Scaling  3/ 1 Result_size  3/ 1
    A12 += oe.contract('bkij->bkji',D8)
    
    #Contraction 99; Tree Level  2; Scaling  3/ 1 Result_size  3/ 1
    A12 += -1.0 * oe.contract('bkij->bkji',I7)
    
    #Contraction 100; Tree Level  1; Scaling  4/ 4 Result_size  2/ 2
    Z2 += oe.contract('kc,abcijk->abij',F3,T3, optimize='optimal')
    
    J8 = np.zeros([nocc, nocc], dtype=type_)
    
    #Contraction 101; Tree Level  3; Scaling  4/ 2 Result_size  2/ 0
    J8 += oe.contract('klic,ck->li',V4,T1, optimize='optimal')
    
    M8 = np.zeros([nvir, nvir, nocc, nocc], dtype=type_)
    
    #Contraction 102; Tree Level  2; Scaling  4/ 4 Result_size  2/ 2
    M8 += oe.contract('acik,kbcj->abij',T2,G2, optimize='optimal')
    
    #Contraction 103; Tree Level  1; Scaling  2/ 2 Result_size  2/ 2
    Z2 += 0.5 * oe.contract('abij->abij',M8)
    
    #Contraction 104; Tree Level  1; Scaling  2/ 2 Result_size  2/ 2
    Z2 += -0.5 * oe.contract('baij->abij',M8)
    
    #Contraction 105; Tree Level  1; Scaling  2/ 2 Result_size  2/ 2
    Z2 += -0.5 * oe.contract('abji->abij',M8)
    
    #Contraction 106; Tree Level  1; Scaling  2/ 2 Result_size  2/ 2
    Z2 += 0.5 * oe.contract('baji->abij',M8)
    
    del M8
    
    X8 = np.zeros([nvir, nvir, nocc, nocc], dtype=type_)
    
    #Contraction 107; Tree Level  2; Scaling  4/ 2 Result_size  2/ 2
    X8 += oe.contract('ablj,li->abji',T2,A5, optimize='optimal')
    
    #Contraction 108; Tree Level  1; Scaling  2/ 2 Result_size  2/ 2
    Z2 += oe.contract('abji->abij',X8)
    
    #Contraction 109; Tree Level  1; Scaling  2/ 2 Result_size  2/ 2
    Z2 += -1.0 * oe.contract('abij->abij',X8)
    
    del X8
    
    Y8 = np.zeros([nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 110; Tree Level  3; Scaling  5/ 1 Result_size  3/ 1
    Y8 += oe.contract('al,lmij->amij',T1,I5, optimize='optimal')
    
    A9 = np.zeros([nvir, nvir, nocc, nocc], dtype=type_)
    
    #Contraction 111; Tree Level  2; Scaling  6/ 4 Result_size  2/ 2
    A9 += oe.contract('klic,bacjkl->baij',V4,T3, optimize='optimal')
    
    #Contraction 112; Tree Level  1; Scaling  2/ 2 Result_size  2/ 2
    Z2 += -0.5 * oe.contract('baij->abij',A9)
    
    #Contraction 113; Tree Level  1; Scaling  2/ 2 Result_size  2/ 2
    Z2 += 0.5 * oe.contract('baji->abij',A9)
    
    del A9
    
    D9 = np.zeros([nvir, nvir, nocc, nocc], dtype=type_)
    
    #Contraction 114; Tree Level  2; Scaling  4/ 2 Result_size  2/ 2
    D9 += oe.contract('ak,bkji->abji',T1,Y5, optimize='optimal')
    
    #Contraction 115; Tree Level  1; Scaling  2/ 2 Result_size  2/ 2
    Z2 += -1.0 * oe.contract('abji->abij',D9)
    
    #Contraction 116; Tree Level  1; Scaling  2/ 2 Result_size  2/ 2
    Z2 += oe.contract('baji->abij',D9)
    
    #Contraction 117; Tree Level  1; Scaling  2/ 2 Result_size  2/ 2
    Z2 += oe.contract('abij->abij',D9)
    
    #Contraction 118; Tree Level  1; Scaling  2/ 2 Result_size  2/ 2
    Z2 += -1.0 * oe.contract('baij->abij',D9)
    
    del D9
    
    E9 = np.zeros([nvir, nvir, nocc, nocc], dtype=type_)
    
    #Contraction 119; Tree Level  2; Scaling  6/ 4 Result_size  2/ 2
    E9 += oe.contract('dabklj,kldi->abji',T3,I1, optimize='optimal')
    
    #Contraction 120; Tree Level  1; Scaling  2/ 2 Result_size  2/ 2
    Z2 += 0.5 * oe.contract('abji->abij',E9)
    
    #Contraction 121; Tree Level  1; Scaling  2/ 2 Result_size  2/ 2
    Z2 += -0.5 * oe.contract('abij->abij',E9)
    
    del E9
    
    G9 = np.zeros([nvir, nvir, nocc, nocc], dtype=type_)
    
    #Contraction 122; Tree Level  2; Scaling  4/ 2 Result_size  2/ 2
    G9 += oe.contract('abik,kj->abij',T2,M1, optimize='optimal')
    
    del M1
    
    #Contraction 123; Tree Level  1; Scaling  2/ 2 Result_size  2/ 2
    Z2 += 0.5 * oe.contract('abij->abij',G9)
    
    #Contraction 124; Tree Level  1; Scaling  2/ 2 Result_size  2/ 2
    Z2 += -0.5 * oe.contract('abji->abij',G9)
    
    del G9
    
    I9 = np.zeros([nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 125; Tree Level  3; Scaling  3/ 3 Result_size  3/ 1
    I9 += oe.contract('ci,akcj->akij',T1,M4, optimize='optimal')
    
    del M4
    
    J9 = np.zeros([nvir, nvir, nocc, nocc], dtype=type_)
    
    #Contraction 126; Tree Level  2; Scaling  4/ 2 Result_size  2/ 2
    J9 += oe.contract('bk,akij->baij',T1,I9, optimize='optimal')
    
    #Contraction 127; Tree Level  1; Scaling  2/ 2 Result_size  2/ 2
    Z2 += -0.5 * oe.contract('baij->abij',J9)
    
    #Contraction 128; Tree Level  1; Scaling  2/ 2 Result_size  2/ 2
    Z2 += 0.5 * oe.contract('abij->abij',J9)
    
    #Contraction 129; Tree Level  1; Scaling  2/ 2 Result_size  2/ 2
    Z2 += 0.5 * oe.contract('baji->abij',J9)
    
    #Contraction 130; Tree Level  1; Scaling  2/ 2 Result_size  2/ 2
    Z2 += -0.5 * oe.contract('abji->abij',J9)
    
    del J9
    
    M9 = np.zeros([nocc, nvir, nocc, nocc], dtype=type_)
    
    #Contraction 131; Tree Level  3; Scaling  5/ 5 Result_size  3/ 1
    M9 += oe.contract('klcd,cdblij->kbij',V7,T3, optimize='optimal')
    
    X9 = np.zeros([nvir, nvir, nocc, nocc], dtype=type_)
    
    #Contraction 132; Tree Level  2; Scaling  4/ 2 Result_size  2/ 2
    X9 += oe.contract('ak,kbij->abij',T1,M9, optimize='optimal')
    
    #Contraction 133; Tree Level  1; Scaling  2/ 2 Result_size  2/ 2
    Z2 += 0.5 * oe.contract('abij->abij',X9)
    
    #Contraction 134; Tree Level  1; Scaling  2/ 2 Result_size  2/ 2
    Z2 += -0.5 * oe.contract('baij->abij',X9)
    
    del X9
    
    #Contraction 135; Tree Level  1; Scaling  6/ 2 Result_size  2/ 2
    Z2 += 0.25 * oe.contract('abkl,klij->abij',T2,G8, optimize='optimal')
    
    del G8
    
    #Contraction 136; Tree Level  1; Scaling  2/ 6 Result_size  2/ 2
    Z2 += 0.5 * oe.contract('abcd,cdij->abij',V9,T2, optimize='optimal')
    
    #Contraction 137; Tree Level  1; Scaling  4/ 2 Result_size  2/ 2
    Z2 += 0.25 * oe.contract('ak,bkij->abij',T1,J5, optimize='optimal')
    
    del J5
    
    E10 = np.zeros([nvir, nvir, nocc, nocc], dtype=type_)
    
    #Contraction 138; Tree Level  2; Scaling  4/ 2 Result_size  2/ 2
    E10 += oe.contract('ablj,li->abji',T2,J8, optimize='optimal')
    
    #Contraction 139; Tree Level  1; Scaling  2/ 2 Result_size  2/ 2
    Z2 += oe.contract('abji->abij',E10)
    
    #Contraction 140; Tree Level  1; Scaling  2/ 2 Result_size  2/ 2
    Z2 += -1.0 * oe.contract('abij->abij',E10)
    
    del E10
    
    #Contraction 141; Tree Level  3; Scaling  4/ 0 Result_size  4/ 0
    A11 += -1.0 * oe.contract('klij->lkij',V1)
    
    #Contraction 142; Tree Level  3; Scaling  4/ 0 Result_size  4/ 0
    A11 += oe.contract('lkij->lkij',V1)
    
    I10 = np.zeros([nvir, nvir], dtype=type_)
    
    #Contraction 143; Tree Level  3; Scaling  2/ 4 Result_size  0/ 2
    I10 += oe.contract('akcd,ck->ad',V8,T1, optimize='optimal')
    
    J10 = np.zeros([nvir, nvir, nocc, nocc], dtype=type_)
    
    #Contraction 144; Tree Level  2; Scaling  2/ 4 Result_size  2/ 2
    J10 += oe.contract('dbij,ad->baij',T2,I10, optimize='optimal')
    
    #Contraction 145; Tree Level  1; Scaling  2/ 2 Result_size  2/ 2
    Z2 += -1.0 * oe.contract('baij->abij',J10)
    
    #Contraction 146; Tree Level  1; Scaling  2/ 2 Result_size  2/ 2
    Z2 += oe.contract('abij->abij',J10)
    
    del J10
    
    #Contraction 147; Tree Level  3; Scaling  4/ 0 Result_size  4/ 0
    A11 += -1.0 * oe.contract('klij->lkij',Y6)
    
    #Contraction 148; Tree Level  2; Scaling  3/ 1 Result_size  3/ 1
    A12 += -4.0 * oe.contract('bkij->bkji',J4)
    
    #Contraction 149; Tree Level  2; Scaling  3/ 1 Result_size  3/ 1
    A12 += 4.0 * oe.contract('bkji->bkji',J4)
    
    #Contraction 150; Tree Level  3; Scaling  4/ 0 Result_size  4/ 0
    A11 += oe.contract('lkij->lkij',Y6)
    
    #Contraction 151; Tree Level  2; Scaling  5/ 1 Result_size  3/ 1
    A12 += 2.0 * oe.contract('bl,lkij->bkji',T1,A11, optimize='optimal')
    
    del A11
    
    #Contraction 152; Tree Level  2; Scaling  3/ 1 Result_size  3/ 1
    A12 += -4.0 * oe.contract('bkij->bkji',E6)
    
    del E6
    
    #Contraction 153; Tree Level  2; Scaling  3/ 1 Result_size  3/ 1
    A12 += 4.0 * oe.contract('kbij->bkji',E5)
    
    del E5
    
    #Contraction 154; Tree Level  2; Scaling  3/ 1 Result_size  3/ 1
    A12 += oe.contract('bkij->bkji',Y8)
    
    #Contraction 155; Tree Level  2; Scaling  3/ 1 Result_size  3/ 1
    A12 += -1.0 * oe.contract('bkji->bkji',D8)
    
    del D8
    
    #Contraction 156; Tree Level  2; Scaling  3/ 1 Result_size  3/ 1
    A12 += oe.contract('bkji->bkji',I7)
    
    #Contraction 157; Tree Level  1; Scaling  4/ 2 Result_size  2/ 2
    Z2 += -0.25 * oe.contract('al,blji->abij',T1,A12, optimize='optimal')
    
    del A12
    
    #Contraction 158; Tree Level  2; Scaling  3/ 1 Result_size  3/ 1
    G12 += -1.0 * oe.contract('kaij->kaji',Y3)
    
    #Contraction 159; Tree Level  2; Scaling  3/ 1 Result_size  3/ 1
    G12 += oe.contract('kaji->kaji',Y3)
    
    #Contraction 160; Tree Level  1; Scaling  4/ 2 Result_size  2/ 2
    Z2 += -1.0 * oe.contract('bk,kaji->abij',T1,G12, optimize='optimal')
    
    del G12
    
    #Contraction 161; Tree Level  1; Scaling  2/ 2 Result_size  2/ 2
    Z2 += oe.contract('abij->abij',V3)
    
    #del V3
    
    J12 = np.zeros([nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 162; Tree Level  3; Scaling  6/ 4 Result_size  4/ 2
    J12 += oe.contract('ebcmjk,lmei->bcljki',T3,I1, optimize='optimal')
    
    M12 = np.zeros([nocc, nocc, nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 163; Tree Level  4; Scaling  5/ 3 Result_size  5/ 1
    M12 += oe.contract('lmid,dcjk->lmcijk',V4,T2, optimize='optimal')
    
    X12 = np.zeros([nocc, nvir, nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 164; Tree Level  3; Scaling  6/ 4 Result_size  4/ 2
    X12 += oe.contract('lmid,dbcmjk->lbcijk',V4,T3, optimize='optimal')
    
    Y12 = np.zeros([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 165; Tree Level  2; Scaling  5/ 3 Result_size  3/ 3
    Y12 += oe.contract('abil,lcjk->abcijk',T2,M9, optimize='optimal')
    
    #Contraction 166; Tree Level  0; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * oe.contract('abcijk->abcijk',Y12)
    
    #Contraction 167; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.5 * oe.contract('acbijk->abcijk',Y12)
    
    #Contraction 168; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * oe.contract('bcaijk->abcijk',Y12)
    
    #Contraction 169; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.5 * oe.contract('abcjik->abcijk',Y12)
    
    #Contraction 170; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * oe.contract('acbjik->abcijk',Y12)
    
    #Contraction 171; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.5 * oe.contract('bcajik->abcijk',Y12)
    
    #Contraction 172; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * oe.contract('abckij->abcijk',Y12)
    
    #Contraction 173; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.5 * oe.contract('acbkij->abcijk',Y12)
    
    #Contraction 174; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * oe.contract('bcakij->abcijk',Y12)
    
    del Y12
    
    A13 = np.zeros([nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 175; Tree Level  3; Scaling  5/ 1 Result_size  3/ 1
    A13 += oe.contract('al,lmij->amij',T1,Y6, optimize='optimal')
    
    D13 = np.zeros([nvir, nocc, nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 176; Tree Level  3; Scaling  6/ 2 Result_size  4/ 2
    D13 += oe.contract('bm,lmcijk->blcijk',T1,M12, optimize='optimal')
    
    E13 = np.zeros([nocc, nvir, nocc, nocc], dtype=type_)
    
    #Contraction 177; Tree Level  3; Scaling  5/ 1 Result_size  3/ 1
    E13 += oe.contract('lmij,al->maij',V1,T1, optimize='optimal')
    
    G13 = np.zeros([nocc, nocc, nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 178; Tree Level  4; Scaling  5/ 5 Result_size  5/ 1
    G13 += oe.contract('lmde,decijk->lmcijk',V7,T3, optimize='optimal')
    
    I13 = np.zeros([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 179; Tree Level  2; Scaling  5/ 3 Result_size  3/ 3
    I13 += oe.contract('abcmjk,mi->abcjki',T3,J8, optimize='optimal')
    
    #Contraction 180; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('abcjki->abcijk',I13)
    
    #Contraction 181; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('abcikj->abcijk',I13)
    
    #Contraction 182; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('abcijk->abcijk',I13)
    
    del I13
    
    J13 = np.zeros([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 183; Tree Level  2; Scaling  5/ 3 Result_size  3/ 3
    J13 += oe.contract('bcmk,amij->bcakij',T2,A13, optimize='optimal')
    
    #Contraction 184; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('bcakij->abcijk',J13)
    
    #Contraction 185; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('acbkij->abcijk',J13)
    
    #Contraction 186; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('abckij->abcijk',J13)
    
    #Contraction 187; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('bcajik->abcijk',J13)
    
    #Contraction 188; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('acbjik->abcijk',J13)
    
    #Contraction 189; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('abcjik->abcijk',J13)
    
    #Contraction 190; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('bcakji->abcijk',J13)
    
    #Contraction 191; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('acbkji->abcijk',J13)
    
    #Contraction 192; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('abckji->abcijk',J13)
    
    #Contraction 193; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('bcaijk->abcijk',J13)
    
    #Contraction 194; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('acbijk->abcijk',J13)
    
    #Contraction 195; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('abcijk->abcijk',J13)
    
    #Contraction 196; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('bcajki->abcijk',J13)
    
    #Contraction 197; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('acbjki->abcijk',J13)
    
    #Contraction 198; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('abcjki->abcijk',J13)
    
    #Contraction 199; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('bcaikj->abcijk',J13)
    
    #Contraction 200; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('acbikj->abcijk',J13)
    
    #Contraction 201; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('abcikj->abcijk',J13)
    
    del J13
    
    M13 = np.zeros([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 202; Tree Level  2; Scaling  7/ 5 Result_size  3/ 3
    M13 += oe.contract('eabclmjk,lmei->abcjki',T4,I1, optimize='optimal')
    
    #Contraction 203; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * oe.contract('abcjki->abcijk',M13)
    
    #Contraction 204; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.5 * oe.contract('abcikj->abcijk',M13)
    
    #Contraction 205; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * oe.contract('abcijk->abcijk',M13)
    
    del M13
    
    X13 = np.zeros([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 206; Tree Level  2; Scaling  3/ 5 Result_size  3/ 3
    X13 += oe.contract('abid,cdkj->abcikj',V6,T2, optimize='optimal')
    
    #Contraction 207; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('abcikj->abcijk',X13)
    
    #Contraction 208; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('acbikj->abcijk',X13)
    
    #Contraction 209; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('bcaikj->abcijk',X13)
    
    #Contraction 210; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('abckij->abcijk',X13)
    
    #Contraction 211; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('acbkij->abcijk',X13)
    
    #Contraction 212; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('bcakij->abcijk',X13)
    
    #Contraction 213; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('abcjik->abcijk',X13)
    
    #Contraction 214; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('acbjik->abcijk',X13)
    
    #Contraction 215; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('bcajik->abcijk',X13)
    
    del X13
    
    Y13 = np.zeros([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 216; Tree Level  2; Scaling  5/ 7 Result_size  3/ 3
    Y13 += oe.contract('alde,bcdejkil->abcjki',V8,T4, optimize='optimal')
    
    #Contraction 217; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * oe.contract('abcjki->abcijk',Y13)
    
    #Contraction 218; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.5 * oe.contract('bacjki->abcijk',Y13)
    
    #Contraction 219; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * oe.contract('cabjki->abcijk',Y13)
    
    del Y13
    
    A14 = np.zeros([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 220; Tree Level  2; Scaling  5/ 3 Result_size  3/ 3
    A14 += oe.contract('bclk,alij->bcakij',T2,J4, optimize='optimal')
    
    #Contraction 221; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('bcakij->abcijk',A14)
    
    #Contraction 222; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('acbkij->abcijk',A14)
    
    #Contraction 223; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('abckij->abcijk',A14)
    
    #Contraction 224; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('bcajik->abcijk',A14)
    
    #Contraction 225; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('acbjik->abcijk',A14)
    
    #Contraction 226; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('abcjik->abcijk',A14)
    
    #Contraction 227; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('bcakji->abcijk',A14)
    
    #Contraction 228; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('acbkji->abcijk',A14)
    
    #Contraction 229; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('abckji->abcijk',A14)
    
    #Contraction 230; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('bcaijk->abcijk',A14)
    
    #Contraction 231; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('acbijk->abcijk',A14)
    
    #Contraction 232; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('abcijk->abcijk',A14)
    
    #Contraction 233; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('bcajki->abcijk',A14)
    
    #Contraction 234; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('acbjki->abcijk',A14)
    
    #Contraction 235; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('abcjki->abcijk',A14)
    
    #Contraction 236; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('bcaikj->abcijk',A14)
    
    #Contraction 237; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('acbikj->abcijk',A14)
    
    #Contraction 238; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('abcikj->abcijk',A14)
    
    del A14
    
    D14 = np.zeros([nvir, nocc, nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 239; Tree Level  3; Scaling  4/ 6 Result_size  4/ 2
    D14 += oe.contract('alde,decijk->alcijk',V8,T3, optimize='optimal')
    
    E14 = np.zeros([nocc, nvir, nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 240; Tree Level  3; Scaling  6/ 6 Result_size  4/ 2
    E14 += oe.contract('lmde,debcmijk->lbcijk',V7,T4, optimize='optimal')
    
    G14 = np.zeros([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 241; Tree Level  2; Scaling  5/ 3 Result_size  3/ 3
    G14 += oe.contract('al,lbcijk->abcijk',T1,E14, optimize='optimal')
    
    #Contraction 242; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * oe.contract('abcijk->abcijk',G14)
    
    #Contraction 243; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.5 * oe.contract('bacijk->abcijk',G14)
    
    #Contraction 244; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * oe.contract('cabijk->abcijk',G14)
    
    del G14
    
    I14 = np.zeros([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 245; Tree Level  2; Scaling  5/ 3 Result_size  3/ 3
    I14 += oe.contract('bclk,alij->bcakij',T2,J6, optimize='optimal')
    
    #Contraction 246; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.5 * oe.contract('bcakij->abcijk',I14)
    
    #Contraction 247; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * oe.contract('acbkij->abcijk',I14)
    
    #Contraction 248; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.5 * oe.contract('abckij->abcijk',I14)
    
    #Contraction 249; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * oe.contract('bcajik->abcijk',I14)
    
    #Contraction 250; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.5 * oe.contract('acbjik->abcijk',I14)
    
    #Contraction 251; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * oe.contract('abcjik->abcijk',I14)
    
    #Contraction 252; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.5 * oe.contract('bcaijk->abcijk',I14)
    
    #Contraction 253; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * oe.contract('acbijk->abcijk',I14)
    
    #Contraction 254; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.5 * oe.contract('abcijk->abcijk',I14)
    
    del I14
    
    J14 = np.zeros([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 255; Tree Level  2; Scaling  5/ 3 Result_size  3/ 3
    J14 += oe.contract('al,bcljki->abcjki',T1,J12, optimize='optimal')
    
    #Contraction 256; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('abcjki->abcijk',J14)
    
    #Contraction 257; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('bacjki->abcijk',J14)
    
    #Contraction 258; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('cabjki->abcijk',J14)
    
    #Contraction 259; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('abcikj->abcijk',J14)
    
    #Contraction 260; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('bacikj->abcijk',J14)
    
    #Contraction 261; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('cabikj->abcijk',J14)
    
    #Contraction 262; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('abcijk->abcijk',J14)
    
    #Contraction 263; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('bacijk->abcijk',J14)
    
    #Contraction 264; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('cabijk->abcijk',J14)
    
    del J14
    
    M14 = np.zeros([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 265; Tree Level  2; Scaling  7/ 3 Result_size  3/ 3
    M14 += oe.contract('lmij,cabklm->cabijk',V1,T3, optimize='optimal')
    
    #Contraction 266; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * oe.contract('cabijk->abcijk',M14)
    
    #Contraction 267; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.5 * oe.contract('cabikj->abcijk',M14)
    
    #Contraction 268; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * oe.contract('cabjki->abcijk',M14)
    
    del M14
    
    X14 = np.zeros([nvir, nocc, nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 269; Tree Level  3; Scaling  6/ 2 Result_size  4/ 2
    X14 += oe.contract('bm,lmcijk->blcijk',T1,G13, optimize='optimal')
    
    Y14 = np.zeros([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 270; Tree Level  2; Scaling  5/ 3 Result_size  3/ 3
    Y14 += oe.contract('al,blcijk->abcijk',T1,X14, optimize='optimal')
    
    del X14
    
    #Contraction 271; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.25 * oe.contract('abcijk->abcijk',Y14)
    
    #Contraction 272; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.25 * oe.contract('acbijk->abcijk',Y14)
    
    #Contraction 273; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.25 * oe.contract('bacijk->abcijk',Y14)
    
    #Contraction 274; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.25 * oe.contract('bcaijk->abcijk',Y14)
    
    #Contraction 275; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.25 * oe.contract('cabijk->abcijk',Y14)
    
    #Contraction 276; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.25 * oe.contract('cbaijk->abcijk',Y14)
    
    del Y14
    
    A15 = np.zeros([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 277; Tree Level  2; Scaling  5/ 3 Result_size  3/ 3
    A15 += oe.contract('li,bcajkl->bcaijk',F1,T3, optimize='optimal')
    
    #Contraction 278; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('bcaijk->abcijk',A15)
    
    #Contraction 279; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('bcajik->abcijk',A15)
    
    #Contraction 280; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('bcakij->abcijk',A15)
    
    del A15
    
    D15 = np.zeros([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 281; Tree Level  2; Scaling  5/ 5 Result_size  3/ 3
    D15 += oe.contract('ebcljk,alei->bcajki',T3,G5, optimize='optimal')
    
    #Contraction 282; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('bcajki->abcijk',D15)
    
    #Contraction 283; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('acbjki->abcijk',D15)
    
    #Contraction 284; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('abcjki->abcijk',D15)
    
    #Contraction 285; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('bcaikj->abcijk',D15)
    
    #Contraction 286; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('acbikj->abcijk',D15)
    
    #Contraction 287; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('abcikj->abcijk',D15)
    
    #Contraction 288; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('bcaijk->abcijk',D15)
    
    #Contraction 289; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('acbijk->abcijk',D15)
    
    #Contraction 290; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('abcijk->abcijk',D15)
    
    del D15
    
    E15 = np.zeros([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 291; Tree Level  2; Scaling  5/ 3 Result_size  3/ 3
    E15 += oe.contract('ablj,clki->abcjki',T2,Y5, optimize='optimal')
    
    del Y5
    
    #Contraction 292; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('abcjki->abcijk',E15)
    
    #Contraction 293; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('acbjki->abcijk',E15)
    
    #Contraction 294; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('bcajki->abcijk',E15)
    
    #Contraction 295; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('abckji->abcijk',E15)
    
    #Contraction 296; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('acbkji->abcijk',E15)
    
    #Contraction 297; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('bcakji->abcijk',E15)
    
    #Contraction 298; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('abcikj->abcijk',E15)
    
    #Contraction 299; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('acbikj->abcijk',E15)
    
    #Contraction 300; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('bcaikj->abcijk',E15)
    
    #Contraction 301; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('abckij->abcijk',E15)
    
    #Contraction 302; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('acbkij->abcijk',E15)
    
    #Contraction 303; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('bcakij->abcijk',E15)
    
    #Contraction 304; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('abcijk->abcijk',E15)
    
    #Contraction 305; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('acbijk->abcijk',E15)
    
    #Contraction 306; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('bcaijk->abcijk',E15)
    
    #Contraction 307; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('abcjik->abcijk',E15)
    
    #Contraction 308; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('acbjik->abcijk',E15)
    
    #Contraction 309; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('bcajik->abcijk',E15)
    
    del E15
    
    G15 = np.zeros([nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 310; Tree Level  3; Scaling  4/ 4 Result_size  4/ 2
    G15 += oe.contract('ebcijk,le->bclijk',T3,D6, optimize='optimal')
    
    I15 = np.zeros([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 311; Tree Level  2; Scaling  3/ 5 Result_size  3/ 3
    I15 += oe.contract('ebcijk,ae->bcaijk',T3,I10, optimize='optimal')
    
    #Contraction 312; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('bcaijk->abcijk',I15)
    
    #Contraction 313; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('acbijk->abcijk',I15)
    
    #Contraction 314; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('abcijk->abcijk',I15)
    
    del I15
    
    J15 = np.zeros([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 315; Tree Level  2; Scaling  5/ 3 Result_size  3/ 3
    J15 += oe.contract('bclk,alij->bcakij',T2,I9, optimize='optimal')
    
    #Contraction 316; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.5 * oe.contract('bcakij->abcijk',J15)
    
    #Contraction 317; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * oe.contract('acbkij->abcijk',J15)
    
    #Contraction 318; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.5 * oe.contract('abckij->abcijk',J15)
    
    #Contraction 319; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * oe.contract('bcajik->abcijk',J15)
    
    #Contraction 320; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.5 * oe.contract('acbjik->abcijk',J15)
    
    #Contraction 321; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * oe.contract('abcjik->abcijk',J15)
    
    #Contraction 322; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * oe.contract('bcakji->abcijk',J15)
    
    #Contraction 323; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.5 * oe.contract('acbkji->abcijk',J15)
    
    #Contraction 324; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * oe.contract('abckji->abcijk',J15)
    
    #Contraction 325; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.5 * oe.contract('bcaijk->abcijk',J15)
    
    #Contraction 326; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * oe.contract('acbijk->abcijk',J15)
    
    #Contraction 327; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.5 * oe.contract('abcijk->abcijk',J15)
    
    #Contraction 328; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.5 * oe.contract('bcajki->abcijk',J15)
    
    #Contraction 329; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * oe.contract('acbjki->abcijk',J15)
    
    #Contraction 330; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.5 * oe.contract('abcjki->abcijk',J15)
    
    #Contraction 331; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * oe.contract('bcaikj->abcijk',J15)
    
    #Contraction 332; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.5 * oe.contract('acbikj->abcijk',J15)
    
    #Contraction 333; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * oe.contract('abcikj->abcijk',J15)
    
    del J15
    
    M15 = np.zeros([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 334; Tree Level  2; Scaling  5/ 3 Result_size  3/ 3
    M15 += oe.contract('alij,cbkl->acbijk',V2,T2, optimize='optimal')
    
    #Contraction 335; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('acbijk->abcijk',M15)
    
    #Contraction 336; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('cabijk->abcijk',M15)
    
    #Contraction 337; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('bacijk->abcijk',M15)
    
    #Contraction 338; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('acbikj->abcijk',M15)
    
    #Contraction 339; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('cabikj->abcijk',M15)
    
    #Contraction 340; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('bacikj->abcijk',M15)
    
    #Contraction 341; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('acbjki->abcijk',M15)
    
    #Contraction 342; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('cabjki->abcijk',M15)
    
    #Contraction 343; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('bacjki->abcijk',M15)
    
    del M15
    
    X15 = np.zeros([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 344; Tree Level  2; Scaling  5/ 5 Result_size  3/ 3
    X15 += oe.contract('alid,bcdjkl->abcijk',V5,T3, optimize='optimal')
    
    #Contraction 345; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('abcijk->abcijk',X15)
    
    #Contraction 346; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('bacijk->abcijk',X15)
    
    #Contraction 347; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('cabijk->abcijk',X15)
    
    #Contraction 348; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('abcjik->abcijk',X15)
    
    #Contraction 349; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('bacjik->abcijk',X15)
    
    #Contraction 350; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('cabjik->abcijk',X15)
    
    #Contraction 351; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('abckij->abcijk',X15)
    
    #Contraction 352; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('backij->abcijk',X15)
    
    #Contraction 353; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('cabkij->abcijk',X15)
    
    del X15
    
    Y15 = np.zeros([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 354; Tree Level  2; Scaling  5/ 3 Result_size  3/ 3
    Y15 += oe.contract('bl,alcijk->bacijk',T1,D14, optimize='optimal')
    
    #Contraction 355; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.5 * oe.contract('bacijk->abcijk',Y15)
    
    #Contraction 356; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * oe.contract('cabijk->abcijk',Y15)
    
    #Contraction 357; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * oe.contract('abcijk->abcijk',Y15)
    
    #Contraction 358; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.5 * oe.contract('cbaijk->abcijk',Y15)
    
    #Contraction 359; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.5 * oe.contract('acbijk->abcijk',Y15)
    
    #Contraction 360; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * oe.contract('bcaijk->abcijk',Y15)
    
    del Y15
    
    A16 = np.zeros([nvir, nvir], dtype=type_)
    
    #Contraction 361; Tree Level  3; Scaling  4/ 4 Result_size  0/ 2
    A16 += oe.contract('lmde,adlm->ae',V7,T2, optimize='optimal')
    
    D16 = np.zeros([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 362; Tree Level  2; Scaling  5/ 3 Result_size  3/ 3
    D16 += oe.contract('abcljk,li->abcjki',T3,Y1, optimize='optimal')
    
    #Contraction 363; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('abcjki->abcijk',D16)
    
    #Contraction 364; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('abcikj->abcijk',D16)
    
    #Contraction 365; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('abcijk->abcijk',D16)
    
    del D16
    
    E16 = np.zeros([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 366; Tree Level  2; Scaling  5/ 3 Result_size  3/ 3
    E16 += oe.contract('al,lbcijk->abcijk',T1,X12, optimize='optimal')
    
    #Contraction 367; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('abcijk->abcijk',E16)
    
    #Contraction 368; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('bacijk->abcijk',E16)
    
    #Contraction 369; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('cabijk->abcijk',E16)
    
    #Contraction 370; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('abcjik->abcijk',E16)
    
    #Contraction 371; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('bacjik->abcijk',E16)
    
    #Contraction 372; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('cabjik->abcijk',E16)
    
    #Contraction 373; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('abckij->abcijk',E16)
    
    #Contraction 374; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('backij->abcijk',E16)
    
    #Contraction 375; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('cabkij->abcijk',E16)
    
    del E16
    
    G16 = np.zeros([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 376; Tree Level  2; Scaling  5/ 3 Result_size  3/ 3
    G16 += oe.contract('abcmjk,mi->abcjki',T3,A5, optimize='optimal')
    
    #Contraction 377; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('abcjki->abcijk',G16)
    
    #Contraction 378; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('abcikj->abcijk',G16)
    
    #Contraction 379; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('abcijk->abcijk',G16)
    
    del G16
    
    I16 = np.zeros([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 380; Tree Level  2; Scaling  7/ 3 Result_size  3/ 3
    I16 += oe.contract('abclmk,lmij->abckij',T3,Y6, optimize='optimal')
    
    #Contraction 381; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * oe.contract('abckij->abcijk',I16)
    
    #Contraction 382; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.5 * oe.contract('abcjik->abcijk',I16)
    
    #Contraction 383; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.5 * oe.contract('abckji->abcijk',I16)
    
    #Contraction 384; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * oe.contract('abcijk->abcijk',I16)
    
    #Contraction 385; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * oe.contract('abcjki->abcijk',I16)
    
    #Contraction 386; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.5 * oe.contract('abcikj->abcijk',I16)
    
    del I16
    
    J16 = np.zeros([nvir, nvir, nvir, nocc], dtype=type_)
    
    #Contraction 387; Tree Level  3; Scaling  3/ 5 Result_size  1/ 3
    J16 += oe.contract('alde,eclk->acdk',V8,T2, optimize='optimal')
    
    M16 = np.zeros([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 388; Tree Level  2; Scaling  5/ 3 Result_size  3/ 3
    M16 += oe.contract('al,bclijk->abcijk',T1,G15, optimize='optimal')
    
    del G15
    
    #Contraction 389; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('abcijk->abcijk',M16)
    
    #Contraction 390; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('bacijk->abcijk',M16)
    
    #Contraction 391; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('cabijk->abcijk',M16)
    
    del M16
    
    X16 = np.zeros([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 392; Tree Level  2; Scaling  3/ 5 Result_size  3/ 3
    X16 += oe.contract('ad,bcdjki->abcjki',F4,T3, optimize='optimal')
    
    #Contraction 393; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('abcjki->abcijk',X16)
    
    #Contraction 394; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('bacjki->abcijk',X16)
    
    #Contraction 395; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('cabjki->abcijk',X16)
    
    del X16
    
    Y16 = np.zeros([nocc, nocc], dtype=type_)
    
    #Contraction 396; Tree Level  3; Scaling  4/ 4 Result_size  2/ 0
    Y16 += oe.contract('lmde,deil->mi',V7,T2, optimize='optimal')
    
    A17 = np.zeros([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 397; Tree Level  2; Scaling  5/ 3 Result_size  3/ 3
    A17 += oe.contract('abcmjk,mi->abcjki',T3,Y16, optimize='optimal')
    
    #Contraction 398; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * oe.contract('abcjki->abcijk',A17)
    
    #Contraction 399; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.5 * oe.contract('abcikj->abcijk',A17)
    
    #Contraction 400; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * oe.contract('abcijk->abcijk',A17)
    
    del A17
    
    D17 = np.zeros([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 401; Tree Level  2; Scaling  5/ 3 Result_size  3/ 3
    D17 += oe.contract('bcmk,amij->bcakij',T2,Y8, optimize='optimal')
    
    #Contraction 402; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * oe.contract('bcakij->abcijk',D17)
    
    #Contraction 403; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.5 * oe.contract('acbkij->abcijk',D17)
    
    #Contraction 404; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * oe.contract('abckij->abcijk',D17)
    
    #Contraction 405; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.5 * oe.contract('bcajik->abcijk',D17)
    
    #Contraction 406; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * oe.contract('acbjik->abcijk',D17)
    
    #Contraction 407; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.5 * oe.contract('abcjik->abcijk',D17)
    
    #Contraction 408; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * oe.contract('bcaijk->abcijk',D17)
    
    #Contraction 409; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.5 * oe.contract('acbijk->abcijk',D17)
    
    #Contraction 410; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * oe.contract('abcijk->abcijk',D17)
    
    del D17
    
    E17 = np.zeros([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 411; Tree Level  2; Scaling  5/ 3 Result_size  3/ 3
    E17 += oe.contract('al,blcijk->abcijk',T1,D13, optimize='optimal')
    
    del D13
    
    #Contraction 412; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * oe.contract('abcijk->abcijk',E17)
    
    #Contraction 413; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.5 * oe.contract('acbijk->abcijk',E17)
    
    #Contraction 414; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.5 * oe.contract('bacijk->abcijk',E17)
    
    #Contraction 415; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * oe.contract('bcaijk->abcijk',E17)
    
    #Contraction 416; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * oe.contract('cabijk->abcijk',E17)
    
    #Contraction 417; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.5 * oe.contract('cbaijk->abcijk',E17)
    
    #Contraction 418; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.5 * oe.contract('abcjik->abcijk',E17)
    
    #Contraction 419; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * oe.contract('acbjik->abcijk',E17)
    
    #Contraction 420; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * oe.contract('bacjik->abcijk',E17)
    
    #Contraction 421; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.5 * oe.contract('bcajik->abcijk',E17)
    
    #Contraction 422; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.5 * oe.contract('cabjik->abcijk',E17)
    
    #Contraction 423; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * oe.contract('cbajik->abcijk',E17)
    
    #Contraction 424; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * oe.contract('abckij->abcijk',E17)
    
    #Contraction 425; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.5 * oe.contract('acbkij->abcijk',E17)
    
    #Contraction 426; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.5 * oe.contract('backij->abcijk',E17)
    
    #Contraction 427; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * oe.contract('bcakij->abcijk',E17)
    
    #Contraction 428; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * oe.contract('cabkij->abcijk',E17)
    
    #Contraction 429; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.5 * oe.contract('cbakij->abcijk',E17)
    
    del E17
    
    G17 = np.zeros([nocc, nvir, nvir, nocc], dtype=type_)
    
    #Contraction 430; Tree Level  3; Scaling  4/ 4 Result_size  2/ 2
    G17 += oe.contract('lmde,adil->maei',V7,T2, optimize='optimal')
    
    I17 = np.zeros([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 431; Tree Level  2; Scaling  5/ 5 Result_size  3/ 3
    I17 += oe.contract('ebcmjk,maei->bcajki',T3,G17, optimize='optimal')
    
    #Contraction 432; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('bcajki->abcijk',I17)
    
    #Contraction 433; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('acbjki->abcijk',I17)
    
    #Contraction 434; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('abcjki->abcijk',I17)
    
    #Contraction 435; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('bcaikj->abcijk',I17)
    
    #Contraction 436; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('acbikj->abcijk',I17)
    
    #Contraction 437; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('abcikj->abcijk',I17)
    
    #Contraction 438; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('bcaijk->abcijk',I17)
    
    #Contraction 439; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('acbijk->abcijk',I17)
    
    #Contraction 440; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('abcijk->abcijk',I17)
    
    del I17
    
    J17 = np.zeros([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 441; Tree Level  2; Scaling  7/ 5 Result_size  3/ 3
    J17 += oe.contract('lmid,bcadjklm->bcaijk',V4,T4, optimize='optimal')
    
    #Contraction 442; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.5 * oe.contract('bcaijk->abcijk',J17)
    
    #Contraction 443; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * oe.contract('bcajik->abcijk',J17)
    
    #Contraction 444; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.5 * oe.contract('bcakij->abcijk',J17)
    
    del J17
    
    M17 = np.zeros([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 445; Tree Level  2; Scaling  5/ 3 Result_size  3/ 3
    M17 += oe.contract('bcmk,maij->bcakij',T2,E13, optimize='optimal')
    
    #Contraction 446; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('bcakij->abcijk',M17)
    
    #Contraction 447; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('acbkij->abcijk',M17)
    
    #Contraction 448; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('abckij->abcijk',M17)
    
    #Contraction 449; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('bcajik->abcijk',M17)
    
    #Contraction 450; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('acbjik->abcijk',M17)
    
    #Contraction 451; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('abcjik->abcijk',M17)
    
    #Contraction 452; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('bcaijk->abcijk',M17)
    
    #Contraction 453; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('acbijk->abcijk',M17)
    
    #Contraction 454; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('abcijk->abcijk',M17)
    
    del M17
    
    X17 = np.zeros([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 455; Tree Level  2; Scaling  3/ 5 Result_size  3/ 3
    X17 += oe.contract('bdji,acdk->bacjik',T2,J16, optimize='optimal')
    
    del J16
    
    #Contraction 456; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('bacjik->abcijk',X17)
    
    #Contraction 457; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('cabjik->abcijk',X17)
    
    #Contraction 458; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('abcjik->abcijk',X17)
    
    #Contraction 459; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('cbajik->abcijk',X17)
    
    #Contraction 460; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('acbjik->abcijk',X17)
    
    #Contraction 461; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('bcajik->abcijk',X17)
    
    #Contraction 462; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('bacjki->abcijk',X17)
    
    #Contraction 463; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('cabjki->abcijk',X17)
    
    #Contraction 464; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('abcjki->abcijk',X17)
    
    #Contraction 465; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('cbajki->abcijk',X17)
    
    #Contraction 466; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('acbjki->abcijk',X17)
    
    #Contraction 467; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('bcajki->abcijk',X17)
    
    #Contraction 468; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('bacikj->abcijk',X17)
    
    #Contraction 469; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('cabikj->abcijk',X17)
    
    #Contraction 470; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('abcikj->abcijk',X17)
    
    #Contraction 471; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('cbaikj->abcijk',X17)
    
    #Contraction 472; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('acbikj->abcijk',X17)
    
    #Contraction 473; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('bcaikj->abcijk',X17)
    
    del X17
    
    Y17 = np.zeros([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 474; Tree Level  2; Scaling  7/ 3 Result_size  3/ 3
    Y17 += oe.contract('abclmk,lmji->abckji',T3,E7, optimize='optimal')
    
    #Contraction 475; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.25 * oe.contract('abckji->abcijk',Y17)
    
    #Contraction 476; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.25 * oe.contract('abcjki->abcijk',Y17)
    
    #Contraction 477; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.25 * oe.contract('abckij->abcijk',Y17)
    
    #Contraction 478; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.25 * oe.contract('abcikj->abcijk',Y17)
    
    #Contraction 479; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.25 * oe.contract('abcjik->abcijk',Y17)
    
    #Contraction 480; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.25 * oe.contract('abcijk->abcijk',Y17)
    
    del Y17
    
    A18 = np.zeros([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 481; Tree Level  2; Scaling  5/ 3 Result_size  3/ 3
    A18 += oe.contract('bajl,lcik->bacjik',T2,Y3, optimize='optimal')
    
    del Y3
    
    #Contraction 482; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('bacjik->abcijk',A18)
    
    #Contraction 483; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('bcajik->abcijk',A18)
    
    #Contraction 484; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('acbjik->abcijk',A18)
    
    #Contraction 485; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('backij->abcijk',A18)
    
    #Contraction 486; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('bcakij->abcijk',A18)
    
    #Contraction 487; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('acbkij->abcijk',A18)
    
    #Contraction 488; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('bacijk->abcijk',A18)
    
    #Contraction 489; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('bcaijk->abcijk',A18)
    
    #Contraction 490; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('acbijk->abcijk',A18)
    
    #Contraction 491; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('backji->abcijk',A18)
    
    #Contraction 492; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('bcakji->abcijk',A18)
    
    #Contraction 493; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('acbkji->abcijk',A18)
    
    #Contraction 494; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('bacikj->abcijk',A18)
    
    #Contraction 495; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('bcaikj->abcijk',A18)
    
    #Contraction 496; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('acbikj->abcijk',A18)
    
    #Contraction 497; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('bacjki->abcijk',A18)
    
    #Contraction 498; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('bcajki->abcijk',A18)
    
    #Contraction 499; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('acbjki->abcijk',A18)
    
    del A18
    
    D18 = np.zeros([nvir, nocc, nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 500; Tree Level  3; Scaling  4/ 4 Result_size  4/ 2
    D18 += oe.contract('alid,dcjk->alcijk',V5,T2, optimize='optimal')
    
    E18 = np.zeros([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 501; Tree Level  2; Scaling  5/ 3 Result_size  3/ 3
    E18 += oe.contract('bl,alcijk->bacijk',T1,D18, optimize='optimal')
    
    del D18
    
    #Contraction 502; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('bacijk->abcijk',E18)
    
    #Contraction 503; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('cabijk->abcijk',E18)
    
    #Contraction 504; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('abcijk->abcijk',E18)
    
    #Contraction 505; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('cbaijk->abcijk',E18)
    
    #Contraction 506; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('acbijk->abcijk',E18)
    
    #Contraction 507; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('bcaijk->abcijk',E18)
    
    #Contraction 508; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('bacjik->abcijk',E18)
    
    #Contraction 509; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('cabjik->abcijk',E18)
    
    #Contraction 510; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('abcjik->abcijk',E18)
    
    #Contraction 511; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('cbajik->abcijk',E18)
    
    #Contraction 512; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('acbjik->abcijk',E18)
    
    #Contraction 513; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('bcajik->abcijk',E18)
    
    #Contraction 514; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('backij->abcijk',E18)
    
    #Contraction 515; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('cabkij->abcijk',E18)
    
    #Contraction 516; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('abckij->abcijk',E18)
    
    #Contraction 517; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('cbakij->abcijk',E18)
    
    #Contraction 518; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('acbkij->abcijk',E18)
    
    #Contraction 519; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('bcakij->abcijk',E18)
    
    del E18
    
    G18 = np.zeros([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 520; Tree Level  2; Scaling  5/ 3 Result_size  3/ 3
    G18 += oe.contract('bcmk,amji->bcakji',T2,I7, optimize='optimal')
    
    #Contraction 521; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * oe.contract('bcakji->abcijk',G18)
    
    #Contraction 522; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.5 * oe.contract('acbkji->abcijk',G18)
    
    #Contraction 523; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * oe.contract('abckji->abcijk',G18)
    
    #Contraction 524; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.5 * oe.contract('bcajki->abcijk',G18)
    
    #Contraction 525; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * oe.contract('acbjki->abcijk',G18)
    
    #Contraction 526; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.5 * oe.contract('abcjki->abcijk',G18)
    
    #Contraction 527; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.5 * oe.contract('bcakij->abcijk',G18)
    
    #Contraction 528; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * oe.contract('acbkij->abcijk',G18)
    
    #Contraction 529; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.5 * oe.contract('abckij->abcijk',G18)
    
    #Contraction 530; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * oe.contract('bcaikj->abcijk',G18)
    
    #Contraction 531; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.5 * oe.contract('acbikj->abcijk',G18)
    
    #Contraction 532; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * oe.contract('abcikj->abcijk',G18)
    
    #Contraction 533; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * oe.contract('bcajik->abcijk',G18)
    
    #Contraction 534; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.5 * oe.contract('acbjik->abcijk',G18)
    
    #Contraction 535; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * oe.contract('abcjik->abcijk',G18)
    
    #Contraction 536; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.5 * oe.contract('bcaijk->abcijk',G18)
    
    #Contraction 537; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * oe.contract('acbijk->abcijk',G18)
    
    #Contraction 538; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.5 * oe.contract('abcijk->abcijk',G18)
    
    del G18
    
    I18 = np.zeros([nocc, nvir, nocc, nocc], dtype=type_)
    
    #Contraction 539; Tree Level  3; Scaling  3/ 3 Result_size  3/ 1
    I18 += oe.contract('ld,adij->laij',F3,T2, optimize='optimal')
    
    J18 = np.zeros([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 540; Tree Level  2; Scaling  5/ 3 Result_size  3/ 3
    J18 += oe.contract('bclk,laij->bcakij',T2,I18, optimize='optimal')
    
    #Contraction 541; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('bcakij->abcijk',J18)
    
    #Contraction 542; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('acbkij->abcijk',J18)
    
    #Contraction 543; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('abckij->abcijk',J18)
    
    #Contraction 544; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('bcajik->abcijk',J18)
    
    #Contraction 545; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('acbjik->abcijk',J18)
    
    #Contraction 546; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('abcjik->abcijk',J18)
    
    #Contraction 547; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('bcaijk->abcijk',J18)
    
    #Contraction 548; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('acbijk->abcijk',J18)
    
    #Contraction 549; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('abcijk->abcijk',J18)
    
    del J18
    
    M18 = np.zeros([nvir, nvir, nocc, nvir], dtype=type_)
    
    #Contraction 550; Tree Level  3; Scaling  5/ 3 Result_size  1/ 3
    M18 += oe.contract('lmid,aclm->acid',V4,T2, optimize='optimal')
    
    X18 = np.zeros([nvir, nocc, nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 551; Tree Level  3; Scaling  5/ 3 Result_size  5/ 1
    X18 += oe.contract('ecjk,lmei->clmjki',T2,I1, optimize='optimal')
    
    Y18 = np.zeros([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 552; Tree Level  2; Scaling  7/ 3 Result_size  3/ 3
    Y18 += oe.contract('ablm,clmjki->abcjki',T2,X18, optimize='optimal')
    
    #Contraction 553; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * oe.contract('abcjki->abcijk',Y18)
    
    #Contraction 554; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.5 * oe.contract('cbajki->abcijk',Y18)
    
    #Contraction 555; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * oe.contract('cabjki->abcijk',Y18)
    
    #Contraction 556; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.5 * oe.contract('abcikj->abcijk',Y18)
    
    #Contraction 557; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * oe.contract('cbaikj->abcijk',Y18)
    
    #Contraction 558; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.5 * oe.contract('cabikj->abcijk',Y18)
    
    #Contraction 559; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * oe.contract('abcijk->abcijk',Y18)
    
    #Contraction 560; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.5 * oe.contract('cbaijk->abcijk',Y18)
    
    #Contraction 561; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * oe.contract('cabijk->abcijk',Y18)
    
    del Y18
    
    A19 = np.zeros([nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 562; Tree Level  3; Scaling  6/ 2 Result_size  4/ 2
    A19 += oe.contract('bm,clmjki->bcljki',T1,X18, optimize='optimal')
    
    D19 = np.zeros([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 563; Tree Level  2; Scaling  5/ 3 Result_size  3/ 3
    D19 += oe.contract('al,bcljki->abcjki',T1,A19, optimize='optimal')
    
    del A19
    
    #Contraction 564; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * oe.contract('abcjki->abcijk',D19)
    
    #Contraction 565; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.5 * oe.contract('acbjki->abcijk',D19)
    
    #Contraction 566; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.5 * oe.contract('bacjki->abcijk',D19)
    
    #Contraction 567; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * oe.contract('bcajki->abcijk',D19)
    
    #Contraction 568; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * oe.contract('cabjki->abcijk',D19)
    
    #Contraction 569; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.5 * oe.contract('cbajki->abcijk',D19)
    
    #Contraction 570; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.5 * oe.contract('abcikj->abcijk',D19)
    
    #Contraction 571; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * oe.contract('acbikj->abcijk',D19)
    
    #Contraction 572; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * oe.contract('bacikj->abcijk',D19)
    
    #Contraction 573; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.5 * oe.contract('bcaikj->abcijk',D19)
    
    #Contraction 574; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.5 * oe.contract('cabikj->abcijk',D19)
    
    #Contraction 575; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * oe.contract('cbaikj->abcijk',D19)
    
    #Contraction 576; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * oe.contract('abcijk->abcijk',D19)
    
    #Contraction 577; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.5 * oe.contract('acbijk->abcijk',D19)
    
    #Contraction 578; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.5 * oe.contract('bacijk->abcijk',D19)
    
    #Contraction 579; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * oe.contract('bcaijk->abcijk',D19)
    
    #Contraction 580; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * oe.contract('cabijk->abcijk',D19)
    
    #Contraction 581; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.5 * oe.contract('cbaijk->abcijk',D19)
    
    del D19
    
    E19 = np.zeros([nvir, nocc, nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 582; Tree Level  3; Scaling  4/ 4 Result_size  4/ 2
    E19 += oe.contract('dbij,lcdk->blcijk',T2,G2, optimize='optimal')
    
    G19 = np.zeros([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 583; Tree Level  2; Scaling  5/ 3 Result_size  3/ 3
    G19 += oe.contract('al,blcijk->abcijk',T1,E19, optimize='optimal')
    
    del E19
    
    #Contraction 584; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('abcijk->abcijk',G19)
    
    #Contraction 585; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('acbijk->abcijk',G19)
    
    #Contraction 586; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('bacijk->abcijk',G19)
    
    #Contraction 587; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('bcaijk->abcijk',G19)
    
    #Contraction 588; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('cabijk->abcijk',G19)
    
    #Contraction 589; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('cbaijk->abcijk',G19)
    
    #Contraction 590; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('abcikj->abcijk',G19)
    
    #Contraction 591; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('acbikj->abcijk',G19)
    
    #Contraction 592; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('bacikj->abcijk',G19)
    
    #Contraction 593; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('bcaikj->abcijk',G19)
    
    #Contraction 594; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('cabikj->abcijk',G19)
    
    #Contraction 595; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('cbaikj->abcijk',G19)
    
    #Contraction 596; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('abcjki->abcijk',G19)
    
    #Contraction 597; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('acbjki->abcijk',G19)
    
    #Contraction 598; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('bacjki->abcijk',G19)
    
    #Contraction 599; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('bcajki->abcijk',G19)
    
    #Contraction 600; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('cabjki->abcijk',G19)
    
    #Contraction 601; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('cbajki->abcijk',G19)
    
    del G19
    
    I19 = np.zeros([nvir, nvir, nvir, nocc], dtype=type_)
    
    #Contraction 602; Tree Level  3; Scaling  5/ 5 Result_size  1/ 3
    I19 += oe.contract('lmde,ebclmk->bcdk',V7,T3, optimize='optimal')
    
    J19 = np.zeros([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 603; Tree Level  2; Scaling  3/ 5 Result_size  3/ 3
    J19 += oe.contract('adij,bcdk->abcijk',T2,I19, optimize='optimal')
    
    #Contraction 604; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * oe.contract('abcijk->abcijk',J19)
    
    #Contraction 605; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.5 * oe.contract('bacijk->abcijk',J19)
    
    #Contraction 606; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * oe.contract('cabijk->abcijk',J19)
    
    #Contraction 607; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.5 * oe.contract('abcikj->abcijk',J19)
    
    #Contraction 608; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * oe.contract('bacikj->abcijk',J19)
    
    #Contraction 609; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.5 * oe.contract('cabikj->abcijk',J19)
    
    #Contraction 610; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * oe.contract('abcjki->abcijk',J19)
    
    #Contraction 611; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.5 * oe.contract('bacjki->abcijk',J19)
    
    #Contraction 612; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * oe.contract('cabjki->abcijk',J19)
    
    del J19
    
    M19 = np.zeros([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 613; Tree Level  2; Scaling  3/ 5 Result_size  3/ 3
    M19 += oe.contract('ebcijk,ae->bcaijk',T3,A16, optimize='optimal')
    
    #Contraction 614; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * oe.contract('bcaijk->abcijk',M19)
    
    #Contraction 615; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.5 * oe.contract('acbijk->abcijk',M19)
    
    #Contraction 616; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * oe.contract('abcijk->abcijk',M19)
    
    del M19
    
    X19 = np.zeros([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 617; Tree Level  2; Scaling  3/ 5 Result_size  3/ 3
    X19 += oe.contract('bdjk,acid->bacjki',T2,M18, optimize='optimal')
    
    #Contraction 618; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * oe.contract('bacjki->abcijk',X19)
    
    #Contraction 619; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.5 * oe.contract('abcjki->abcijk',X19)
    
    #Contraction 620; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * oe.contract('cbajki->abcijk',X19)
    
    #Contraction 621; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.5 * oe.contract('bacikj->abcijk',X19)
    
    #Contraction 622; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * oe.contract('abcikj->abcijk',X19)
    
    #Contraction 623; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.5 * oe.contract('cbaikj->abcijk',X19)
    
    #Contraction 624; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * oe.contract('bacijk->abcijk',X19)
    
    #Contraction 625; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.5 * oe.contract('abcijk->abcijk',X19)
    
    #Contraction 626; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * oe.contract('cbaijk->abcijk',X19)
    
    del X19
    
    Y19 = np.zeros([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 627; Tree Level  2; Scaling  7/ 3 Result_size  3/ 3
    Y19 += oe.contract('abclmk,lmij->abckij',T3,I5, optimize='optimal')
    
    #Contraction 628; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.25 * oe.contract('abckij->abcijk',Y19)
    
    #Contraction 629; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.25 * oe.contract('abcjik->abcijk',Y19)
    
    #Contraction 630; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.25 * oe.contract('abcijk->abcijk',Y19)
    
    del Y19
    
    A20 = np.zeros([nvir, nvir, nvir, nocc], dtype=type_)
    
    #Contraction 631; Tree Level  3; Scaling  1/ 5 Result_size  1/ 3
    A20 += oe.contract('abde,di->abei',V9,T1, optimize='optimal')
    
    D20 = np.zeros([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 632; Tree Level  2; Scaling  3/ 5 Result_size  3/ 3
    D20 += oe.contract('ecjk,abei->cabjki',T2,A20, optimize='optimal')
    
    #Contraction 633; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('cabjki->abcijk',D20)
    
    #Contraction 634; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('bacjki->abcijk',D20)
    
    #Contraction 635; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('abcjki->abcijk',D20)
    
    #Contraction 636; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('cabikj->abcijk',D20)
    
    #Contraction 637; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('bacikj->abcijk',D20)
    
    #Contraction 638; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('abcikj->abcijk',D20)
    
    #Contraction 639; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('cabijk->abcijk',D20)
    
    #Contraction 640; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('bacijk->abcijk',D20)
    
    #Contraction 641; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('abcijk->abcijk',D20)
    
    del D20
    
    #Contraction 642; Tree Level  1; Scaling  5/ 5 Result_size  3/ 3
    Z3 += oe.contract('ld,abcdijkl->abcijk',F3,T4, optimize='optimal')
    
    #Contraction 643; Tree Level  1; Scaling  5/ 5 Result_size  3/ 3
    Z3 += oe.contract('eabcmijk,me->abcijk',T4,A1, optimize='optimal')
    
    I20 = np.zeros([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 644; Tree Level  2; Scaling  7/ 3 Result_size  3/ 3
    I20 += oe.contract('ablm,lmcijk->abcijk',T2,G13, optimize='optimal')
    
    #Contraction 645; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.25 * oe.contract('abcijk->abcijk',I20)
    
    #Contraction 646; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.25 * oe.contract('acbijk->abcijk',I20)
    
    #Contraction 647; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.25 * oe.contract('bcaijk->abcijk',I20)
    
    del I20
    
    J20 = np.zeros([nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 648; Tree Level  3; Scaling  4/ 4 Result_size  4/ 2
    J20 += oe.contract('ecjk,alei->caljki',T2,G5, optimize='optimal')
    
    M20 = np.zeros([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 649; Tree Level  2; Scaling  5/ 3 Result_size  3/ 3
    M20 += oe.contract('bl,caljki->bcajki',T1,J20, optimize='optimal')
    
    #Contraction 650; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('bcajki->abcijk',M20)
    
    #Contraction 651; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('cbajki->abcijk',M20)
    
    #Contraction 652; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('acbjki->abcijk',M20)
    
    #Contraction 653; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('cabjki->abcijk',M20)
    
    #Contraction 654; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('abcjki->abcijk',M20)
    
    #Contraction 655; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('bacjki->abcijk',M20)
    
    #Contraction 656; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('bcaikj->abcijk',M20)
    
    #Contraction 657; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('cbaikj->abcijk',M20)
    
    #Contraction 658; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('acbikj->abcijk',M20)
    
    #Contraction 659; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('cabikj->abcijk',M20)
    
    #Contraction 660; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('abcikj->abcijk',M20)
    
    #Contraction 661; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('bacikj->abcijk',M20)
    
    #Contraction 662; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('bcaijk->abcijk',M20)
    
    #Contraction 663; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('cbaijk->abcijk',M20)
    
    #Contraction 664; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('acbijk->abcijk',M20)
    
    #Contraction 665; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('cabijk->abcijk',M20)
    
    #Contraction 666; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('abcijk->abcijk',M20)
    
    #Contraction 667; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('bacijk->abcijk',M20)
    
    del M20
    
    X20 = np.zeros([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 668; Tree Level  2; Scaling  3/ 7 Result_size  3/ 3
    X20 += oe.contract('abde,cdekij->abckij',V9,T3, optimize='optimal')
    
    #Contraction 669; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * oe.contract('abckij->abcijk',X20)
    
    #Contraction 670; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -0.5 * oe.contract('acbkij->abcijk',X20)
    
    #Contraction 671; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += 0.5 * oe.contract('bcakij->abcijk',X20)
    
    del X20
    
    Y20 = np.zeros([nocc, nvir, nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 672; Tree Level  3; Scaling  4/ 4 Result_size  4/ 2
    Y20 += oe.contract('ld,dbcijk->lbcijk',F3,T3, optimize='optimal')
    
    A21 = np.zeros([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 673; Tree Level  2; Scaling  5/ 3 Result_size  3/ 3
    A21 += oe.contract('al,lbcijk->abcijk',T1,Y20, optimize='optimal')
    
    #Contraction 674; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('abcijk->abcijk',A21)
    
    #Contraction 675; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('bacijk->abcijk',A21)
    
    #Contraction 676; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('cabijk->abcijk',A21)
    
    del A21
    
    D21 = np.zeros([nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 677; Tree Level  3; Scaling  3/ 3 Result_size  3/ 1
    D21 += oe.contract('ebij,me->bmij',T2,A1, optimize='optimal')
    
    E21 = np.zeros([nvir, nvir, nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 678; Tree Level  2; Scaling  5/ 3 Result_size  3/ 3
    E21 += oe.contract('acmk,bmij->acbkij',T2,D21, optimize='optimal')
    
    #Contraction 679; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('acbkij->abcijk',E21)
    
    #Contraction 680; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('bcakij->abcijk',E21)
    
    #Contraction 681; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('backij->abcijk',E21)
    
    #Contraction 682; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('acbjik->abcijk',E21)
    
    #Contraction 683; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('bcajik->abcijk',E21)
    
    #Contraction 684; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('bacjik->abcijk',E21)
    
    #Contraction 685; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('acbijk->abcijk',E21)
    
    #Contraction 686; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += oe.contract('bcaijk->abcijk',E21)
    
    #Contraction 687; Tree Level  1; Scaling  3/ 3 Result_size  3/ 3
    Z3 += -1.0 * oe.contract('bacijk->abcijk',E21)
    
    del E21
    
    G21 = np.zeros([nvir, nvir, nvir, nocc, nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 688; Tree Level  3; Scaling  5/ 5 Result_size  5/ 3
    G21 += oe.contract('fcdjkl,amfi->cdamjkli',T3,G5, optimize='optimal')
    
    I21 = np.zeros([nocc, nvir, nvir, nvir, nocc, nocc], dtype=type_)
    
    #Contraction 689; Tree Level  4; Scaling  5/ 5 Result_size  3/ 3
    I21 += oe.contract('mnef,fcdnkl->mcdekl',V7,T3, optimize='optimal')
    
    J21 = np.zeros([nvir, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 690; Tree Level  2; Scaling  4/ 6 Result_size  4/ 4
    J21 += oe.contract('abie,cdeklj->abcdiklj',V6,T3, optimize='optimal')
    
    #del V6
    
    #Contraction 691; Tree Level  0; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('abcdiklj->abcdijkl',J21)
    
    #Contraction 692; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('acbdiklj->abcdijkl',J21)
    
    #Contraction 693; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('adbciklj->abcdijkl',J21)
    
    #Contraction 694; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bcadiklj->abcdijkl',J21)
    
    #Contraction 695; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bdaciklj->abcdijkl',J21)
    
    #Contraction 696; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('cdabiklj->abcdijkl',J21)
    
    #Contraction 697; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('abcdkilj->abcdijkl',J21)
    
    #Contraction 698; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('acbdkilj->abcdijkl',J21)
    
    #Contraction 699; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('adbckilj->abcdijkl',J21)
    
    #Contraction 700; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bcadkilj->abcdijkl',J21)
    
    #Contraction 701; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bdackilj->abcdijkl',J21)
    
    #Contraction 702; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('cdabkilj->abcdijkl',J21)
    
    #Contraction 703; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('abcdlikj->abcdijkl',J21)
    
    #Contraction 704; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('acbdlikj->abcdijkl',J21)
    
    #Contraction 705; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('adbclikj->abcdijkl',J21)
    
    #Contraction 706; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bcadlikj->abcdijkl',J21)
    
    #Contraction 707; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bdaclikj->abcdijkl',J21)
    
    #Contraction 708; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('cdablikj->abcdijkl',J21)
    
    #Contraction 709; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('abcdjikl->abcdijkl',J21)
    
    #Contraction 710; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('acbdjikl->abcdijkl',J21)
    
    #Contraction 711; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('adbcjikl->abcdijkl',J21)
    
    #Contraction 712; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bcadjikl->abcdijkl',J21)
    
    #Contraction 713; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bdacjikl->abcdijkl',J21)
    
    #Contraction 714; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('cdabjikl->abcdijkl',J21)
    
    del J21
    
    M21 = np.zeros([nvir, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 715; Tree Level  2; Scaling  6/ 4 Result_size  4/ 4
    M21 += oe.contract('bcdmkl,maij->bcdaklij',T3,I18, optimize='optimal')
    
    del I18
    
    #Contraction 716; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bcdaklij->abcdijkl',M21)
    
    #Contraction 717; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('acdbklij->abcdijkl',M21)
    
    #Contraction 718; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('abdcklij->abcdijkl',M21)
    
    #Contraction 719; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('abcdklij->abcdijkl',M21)
    
    #Contraction 720; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bcdajlik->abcdijkl',M21)
    
    #Contraction 721; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('acdbjlik->abcdijkl',M21)
    
    #Contraction 722; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('abdcjlik->abcdijkl',M21)
    
    #Contraction 723; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('abcdjlik->abcdijkl',M21)
    
    #Contraction 724; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bcdajkil->abcdijkl',M21)
    
    #Contraction 725; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('acdbjkil->abcdijkl',M21)
    
    #Contraction 726; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('abdcjkil->abcdijkl',M21)
    
    #Contraction 727; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('abcdjkil->abcdijkl',M21)
    
    #Contraction 728; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bcdailjk->abcdijkl',M21)
    
    #Contraction 729; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('acdbiljk->abcdijkl',M21)
    
    #Contraction 730; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('abdciljk->abcdijkl',M21)
    
    #Contraction 731; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('abcdiljk->abcdijkl',M21)
    
    #Contraction 732; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bcdaikjl->abcdijkl',M21)
    
    #Contraction 733; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('acdbikjl->abcdijkl',M21)
    
    #Contraction 734; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('abdcikjl->abcdijkl',M21)
    
    #Contraction 735; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('abcdikjl->abcdijkl',M21)
    
    #Contraction 736; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bcdaijkl->abcdijkl',M21)
    
    #Contraction 737; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('acdbijkl->abcdijkl',M21)
    
    #Contraction 738; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('abdcijkl->abcdijkl',M21)
    
    #Contraction 739; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('abcdijkl->abcdijkl',M21)
    
    del M21
    
    X21 = np.zeros([nocc, nvir, nvir, nocc], dtype=type_)
    
    #Contraction 740; Tree Level  4; Scaling  4/ 4 Result_size  2/ 2
    X21 += oe.contract('mnef,ebnj->mbfj',V7,T2, optimize='optimal')
    
    Y21 = np.zeros([nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 741; Tree Level  3; Scaling  5/ 3 Result_size  3/ 1
    Y21 += oe.contract('fbmj,mnfi->bnji',T2,I1, optimize='optimal')
    
    A22 = np.zeros([nvir, nocc, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 742; Tree Level  3; Scaling  5/ 5 Result_size  5/ 3
    A22 += oe.contract('ebij,mcdekl->bmcdijkl',T2,I21, optimize='optimal')
    
    D22 = np.zeros([nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 743; Tree Level  3; Scaling  4/ 4 Result_size  4/ 2
    D22 += oe.contract('fcdjkl,nf->cdnjkl',T3,A1, optimize='optimal')
    
    del A1
    
    E22 = np.zeros([nvir, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 744; Tree Level  2; Scaling  8/ 4 Result_size  4/ 4
    E22 += oe.contract('abdmnl,cmnjki->abdcljki',T3,X18, optimize='optimal')
    
    #Contraction 745; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('abdcljki->abcdijkl',E22)
    
    #Contraction 746; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('cbdaljki->abcdijkl',E22)
    
    #Contraction 747; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('cadbljki->abcdijkl',E22)
    
    #Contraction 748; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('cabdljki->abcdijkl',E22)
    
    #Contraction 749; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('abdckjli->abcdijkl',E22)
    
    #Contraction 750; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('cbdakjli->abcdijkl',E22)
    
    #Contraction 751; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('cadbkjli->abcdijkl',E22)
    
    #Contraction 752; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('cabdkjli->abcdijkl',E22)
    
    #Contraction 753; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('abdcjkli->abcdijkl',E22)
    
    #Contraction 754; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('cbdajkli->abcdijkl',E22)
    
    #Contraction 755; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('cadbjkli->abcdijkl',E22)
    
    #Contraction 756; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('cabdjkli->abcdijkl',E22)
    
    #Contraction 757; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('abdclikj->abcdijkl',E22)
    
    #Contraction 758; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('cbdalikj->abcdijkl',E22)
    
    #Contraction 759; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('cadblikj->abcdijkl',E22)
    
    #Contraction 760; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('cabdlikj->abcdijkl',E22)
    
    #Contraction 761; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('abdckilj->abcdijkl',E22)
    
    #Contraction 762; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('cbdakilj->abcdijkl',E22)
    
    #Contraction 763; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('cadbkilj->abcdijkl',E22)
    
    #Contraction 764; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('cabdkilj->abcdijkl',E22)
    
    #Contraction 765; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('abdciklj->abcdijkl',E22)
    
    #Contraction 766; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('cbdaiklj->abcdijkl',E22)
    
    #Contraction 767; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('cadbiklj->abcdijkl',E22)
    
    #Contraction 768; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('cabdiklj->abcdijkl',E22)
    
    #Contraction 769; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('abdclijk->abcdijkl',E22)
    
    #Contraction 770; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('cbdalijk->abcdijkl',E22)
    
    #Contraction 771; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('cadblijk->abcdijkl',E22)
    
    #Contraction 772; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('cabdlijk->abcdijkl',E22)
    
    #Contraction 773; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('abdcjilk->abcdijkl',E22)
    
    #Contraction 774; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('cbdajilk->abcdijkl',E22)
    
    #Contraction 775; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('cadbjilk->abcdijkl',E22)
    
    #Contraction 776; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('cabdjilk->abcdijkl',E22)
    
    #Contraction 777; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('abdcijlk->abcdijkl',E22)
    
    #Contraction 778; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('cbdaijlk->abcdijkl',E22)
    
    #Contraction 779; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('cadbijlk->abcdijkl',E22)
    
    #Contraction 780; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('cabdijlk->abcdijkl',E22)
    
    #Contraction 781; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('abdckijl->abcdijkl',E22)
    
    #Contraction 782; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('cbdakijl->abcdijkl',E22)
    
    #Contraction 783; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('cadbkijl->abcdijkl',E22)
    
    #Contraction 784; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('cabdkijl->abcdijkl',E22)
    
    #Contraction 785; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('abdcjikl->abcdijkl',E22)
    
    #Contraction 786; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('cbdajikl->abcdijkl',E22)
    
    #Contraction 787; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('cadbjikl->abcdijkl',E22)
    
    #Contraction 788; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('cabdjikl->abcdijkl',E22)
    
    #Contraction 789; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('abdcijkl->abcdijkl',E22)
    
    #Contraction 790; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('cbdaijkl->abcdijkl',E22)
    
    #Contraction 791; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('cadbijkl->abcdijkl',E22)
    
    #Contraction 792; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('cabdijkl->abcdijkl',E22)
    
    del E22
    
    G22 = np.zeros([nvir, nocc, nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 793; Tree Level  3; Scaling  6/ 2 Result_size  4/ 2
    G22 += oe.contract('am,mncijk->ancijk',T1,M12, optimize='optimal')
    
    del M12
    
    I22 = np.zeros([nvir, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 794; Tree Level  2; Scaling  6/ 4 Result_size  4/ 4
    I22 += oe.contract('bajm,mcdikl->bacdjikl',T2,X12, optimize='optimal')
    
    del X12
    
    #Contraction 795; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bacdjikl->abcdijkl',I22)
    
    #Contraction 796; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bcadjikl->abcdijkl',I22)
    
    #Contraction 797; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bdacjikl->abcdijkl',I22)
    
    #Contraction 798; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('acbdjikl->abcdijkl',I22)
    
    #Contraction 799; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('adbcjikl->abcdijkl',I22)
    
    #Contraction 800; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('cdbajikl->abcdijkl',I22)
    
    #Contraction 801; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bacdkijl->abcdijkl',I22)
    
    #Contraction 802; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bcadkijl->abcdijkl',I22)
    
    #Contraction 803; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bdackijl->abcdijkl',I22)
    
    #Contraction 804; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('acbdkijl->abcdijkl',I22)
    
    #Contraction 805; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('adbckijl->abcdijkl',I22)
    
    #Contraction 806; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('cdbakijl->abcdijkl',I22)
    
    #Contraction 807; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bacdlijk->abcdijkl',I22)
    
    #Contraction 808; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bcadlijk->abcdijkl',I22)
    
    #Contraction 809; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bdaclijk->abcdijkl',I22)
    
    #Contraction 810; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('acbdlijk->abcdijkl',I22)
    
    #Contraction 811; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('adbclijk->abcdijkl',I22)
    
    #Contraction 812; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('cdbalijk->abcdijkl',I22)
    
    #Contraction 813; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bacdijkl->abcdijkl',I22)
    
    #Contraction 814; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bcadijkl->abcdijkl',I22)
    
    #Contraction 815; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bdacijkl->abcdijkl',I22)
    
    #Contraction 816; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('acbdijkl->abcdijkl',I22)
    
    #Contraction 817; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('adbcijkl->abcdijkl',I22)
    
    #Contraction 818; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('cdbaijkl->abcdijkl',I22)
    
    #Contraction 819; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bacdkjil->abcdijkl',I22)
    
    #Contraction 820; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bcadkjil->abcdijkl',I22)
    
    #Contraction 821; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bdackjil->abcdijkl',I22)
    
    #Contraction 822; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('acbdkjil->abcdijkl',I22)
    
    #Contraction 823; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('adbckjil->abcdijkl',I22)
    
    #Contraction 824; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('cdbakjil->abcdijkl',I22)
    
    #Contraction 825; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bacdljik->abcdijkl',I22)
    
    #Contraction 826; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bcadljik->abcdijkl',I22)
    
    #Contraction 827; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bdacljik->abcdijkl',I22)
    
    #Contraction 828; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('acbdljik->abcdijkl',I22)
    
    #Contraction 829; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('adbcljik->abcdijkl',I22)
    
    #Contraction 830; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('cdbaljik->abcdijkl',I22)
    
    #Contraction 831; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bacdikjl->abcdijkl',I22)
    
    #Contraction 832; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bcadikjl->abcdijkl',I22)
    
    #Contraction 833; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bdacikjl->abcdijkl',I22)
    
    #Contraction 834; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('acbdikjl->abcdijkl',I22)
    
    #Contraction 835; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('adbcikjl->abcdijkl',I22)
    
    #Contraction 836; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('cdbaikjl->abcdijkl',I22)
    
    #Contraction 837; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bacdjkil->abcdijkl',I22)
    
    #Contraction 838; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bcadjkil->abcdijkl',I22)
    
    #Contraction 839; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bdacjkil->abcdijkl',I22)
    
    #Contraction 840; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('acbdjkil->abcdijkl',I22)
    
    #Contraction 841; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('adbcjkil->abcdijkl',I22)
    
    #Contraction 842; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('cdbajkil->abcdijkl',I22)
    
    #Contraction 843; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bacdlkij->abcdijkl',I22)
    
    #Contraction 844; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bcadlkij->abcdijkl',I22)
    
    #Contraction 845; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bdaclkij->abcdijkl',I22)
    
    #Contraction 846; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('acbdlkij->abcdijkl',I22)
    
    #Contraction 847; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('adbclkij->abcdijkl',I22)
    
    #Contraction 848; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('cdbalkij->abcdijkl',I22)
    
    #Contraction 849; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bacdiljk->abcdijkl',I22)
    
    #Contraction 850; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bcadiljk->abcdijkl',I22)
    
    #Contraction 851; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bdaciljk->abcdijkl',I22)
    
    #Contraction 852; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('acbdiljk->abcdijkl',I22)
    
    #Contraction 853; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('adbciljk->abcdijkl',I22)
    
    #Contraction 854; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('cdbailjk->abcdijkl',I22)
    
    #Contraction 855; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bacdjlik->abcdijkl',I22)
    
    #Contraction 856; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bcadjlik->abcdijkl',I22)
    
    #Contraction 857; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bdacjlik->abcdijkl',I22)
    
    #Contraction 858; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('acbdjlik->abcdijkl',I22)
    
    #Contraction 859; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('adbcjlik->abcdijkl',I22)
    
    #Contraction 860; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('cdbajlik->abcdijkl',I22)
    
    #Contraction 861; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bacdklij->abcdijkl',I22)
    
    #Contraction 862; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bcadklij->abcdijkl',I22)
    
    #Contraction 863; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bdacklij->abcdijkl',I22)
    
    #Contraction 864; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('acbdklij->abcdijkl',I22)
    
    #Contraction 865; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('adbcklij->abcdijkl',I22)
    
    #Contraction 866; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('cdbaklij->abcdijkl',I22)
    
    del I22
    
    J22 = np.zeros([nocc, nocc, nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 867; Tree Level  3; Scaling  5/ 3 Result_size  5/ 1
    J22 += oe.contract('mnie,bejk->mnbijk',V4,T2, optimize='optimal')
    
    M22 = np.zeros([nvir, nocc, nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 868; Tree Level  3; Scaling  4/ 4 Result_size  4/ 2
    M22 += oe.contract('aeij,mdel->amdijl',T2,G2, optimize='optimal')
    
    del G2
    
    X22 = np.zeros([nvir, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 869; Tree Level  2; Scaling  6/ 4 Result_size  4/ 4
    X22 += oe.contract('bcdnkl,anij->bcdaklij',T3,A13, optimize='optimal')
    
    del A13
    
    #Contraction 870; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bcdaklij->abcdijkl',X22)
    
    #Contraction 871; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('acdbklij->abcdijkl',X22)
    
    #Contraction 872; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('abdcklij->abcdijkl',X22)
    
    #Contraction 873; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('abcdklij->abcdijkl',X22)
    
    #Contraction 874; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bcdajlik->abcdijkl',X22)
    
    #Contraction 875; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('acdbjlik->abcdijkl',X22)
    
    #Contraction 876; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('abdcjlik->abcdijkl',X22)
    
    #Contraction 877; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('abcdjlik->abcdijkl',X22)
    
    #Contraction 878; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bcdajkil->abcdijkl',X22)
    
    #Contraction 879; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('acdbjkil->abcdijkl',X22)
    
    #Contraction 880; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('abdcjkil->abcdijkl',X22)
    
    #Contraction 881; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('abcdjkil->abcdijkl',X22)
    
    #Contraction 882; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bcdaklji->abcdijkl',X22)
    
    #Contraction 883; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('acdbklji->abcdijkl',X22)
    
    #Contraction 884; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('abdcklji->abcdijkl',X22)
    
    #Contraction 885; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('abcdklji->abcdijkl',X22)
    
    #Contraction 886; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bcdailjk->abcdijkl',X22)
    
    #Contraction 887; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('acdbiljk->abcdijkl',X22)
    
    #Contraction 888; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('abdciljk->abcdijkl',X22)
    
    #Contraction 889; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('abcdiljk->abcdijkl',X22)
    
    #Contraction 890; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bcdaikjl->abcdijkl',X22)
    
    #Contraction 891; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('acdbikjl->abcdijkl',X22)
    
    #Contraction 892; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('abdcikjl->abcdijkl',X22)
    
    #Contraction 893; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('abcdikjl->abcdijkl',X22)
    
    #Contraction 894; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bcdajlki->abcdijkl',X22)
    
    #Contraction 895; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('acdbjlki->abcdijkl',X22)
    
    #Contraction 896; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('abdcjlki->abcdijkl',X22)
    
    #Contraction 897; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('abcdjlki->abcdijkl',X22)
    
    #Contraction 898; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bcdailkj->abcdijkl',X22)
    
    #Contraction 899; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('acdbilkj->abcdijkl',X22)
    
    #Contraction 900; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('abdcilkj->abcdijkl',X22)
    
    #Contraction 901; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('abcdilkj->abcdijkl',X22)
    
    #Contraction 902; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bcdaijkl->abcdijkl',X22)
    
    #Contraction 903; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('acdbijkl->abcdijkl',X22)
    
    #Contraction 904; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('abdcijkl->abcdijkl',X22)
    
    #Contraction 905; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('abcdijkl->abcdijkl',X22)
    
    #Contraction 906; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bcdajkli->abcdijkl',X22)
    
    #Contraction 907; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('acdbjkli->abcdijkl',X22)
    
    #Contraction 908; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('abdcjkli->abcdijkl',X22)
    
    #Contraction 909; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('abcdjkli->abcdijkl',X22)
    
    #Contraction 910; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bcdaiklj->abcdijkl',X22)
    
    #Contraction 911; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('acdbiklj->abcdijkl',X22)
    
    #Contraction 912; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('abdciklj->abcdijkl',X22)
    
    #Contraction 913; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('abcdiklj->abcdijkl',X22)
    
    #Contraction 914; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bcdaijlk->abcdijkl',X22)
    
    #Contraction 915; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('acdbijlk->abcdijkl',X22)
    
    #Contraction 916; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('abdcijlk->abcdijkl',X22)
    
    #Contraction 917; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('abcdijlk->abcdijkl',X22)
    
    del X22
    
    Y22 = np.zeros([nvir, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 918; Tree Level  2; Scaling  6/ 4 Result_size  4/ 4
    Y22 += oe.contract('bcdnkl,anij->bcdaklij',T3,Y8, optimize='optimal')
    
    del Y8
    
    #Contraction 919; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('bcdaklij->abcdijkl',Y22)
    
    #Contraction 920; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('acdbklij->abcdijkl',Y22)
    
    #Contraction 921; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('abdcklij->abcdijkl',Y22)
    
    #Contraction 922; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('abcdklij->abcdijkl',Y22)
    
    #Contraction 923; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('bcdajlik->abcdijkl',Y22)
    
    #Contraction 924; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('acdbjlik->abcdijkl',Y22)
    
    #Contraction 925; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('abdcjlik->abcdijkl',Y22)
    
    #Contraction 926; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('abcdjlik->abcdijkl',Y22)
    
    #Contraction 927; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('bcdajkil->abcdijkl',Y22)
    
    #Contraction 928; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('acdbjkil->abcdijkl',Y22)
    
    #Contraction 929; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('abdcjkil->abcdijkl',Y22)
    
    #Contraction 930; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('abcdjkil->abcdijkl',Y22)
    
    #Contraction 931; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('bcdailjk->abcdijkl',Y22)
    
    #Contraction 932; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('acdbiljk->abcdijkl',Y22)
    
    #Contraction 933; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('abdciljk->abcdijkl',Y22)
    
    #Contraction 934; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('abcdiljk->abcdijkl',Y22)
    
    #Contraction 935; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('bcdaikjl->abcdijkl',Y22)
    
    #Contraction 936; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('acdbikjl->abcdijkl',Y22)
    
    #Contraction 937; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('abdcikjl->abcdijkl',Y22)
    
    #Contraction 938; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('abcdikjl->abcdijkl',Y22)
    
    #Contraction 939; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('bcdaijkl->abcdijkl',Y22)
    
    #Contraction 940; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('acdbijkl->abcdijkl',Y22)
    
    #Contraction 941; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('abdcijkl->abcdijkl',Y22)
    
    #Contraction 942; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('abcdijkl->abcdijkl',Y22)
    
    del Y22
    
    A23 = np.zeros([nvir, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 943; Tree Level  2; Scaling  4/ 6 Result_size  4/ 4
    A23 += oe.contract('ae,bcdejkli->abcdjkli',F4,T4, optimize='optimal')
    
    #del F4
    
    #Contraction 944; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('abcdjkli->abcdijkl',A23)
    
    #Contraction 945; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bacdjkli->abcdijkl',A23)
    
    #Contraction 946; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('cabdjkli->abcdijkl',A23)
    
    #Contraction 947; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('dabcjkli->abcdijkl',A23)
    
    del A23
    
    D23 = np.zeros([nvir, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 948; Tree Level  2; Scaling  8/ 4 Result_size  4/ 4
    D23 += oe.contract('abcdmnkl,mnij->abcdklij',T4,I5, optimize='optimal')
    
    #Contraction 949; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('abcdklij->abcdijkl',D23)
    
    #Contraction 950; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('abcdjlik->abcdijkl',D23)
    
    #Contraction 951; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('abcdjkil->abcdijkl',D23)
    
    #Contraction 952; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('abcdiljk->abcdijkl',D23)
    
    #Contraction 953; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('abcdikjl->abcdijkl',D23)
    
    #Contraction 954; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('abcdijkl->abcdijkl',D23)
    
    del D23
    
    E23 = np.zeros([nvir, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 955; Tree Level  2; Scaling  6/ 4 Result_size  4/ 4
    E23 += oe.contract('acdnkl,bnij->acdbklij',T3,D21, optimize='optimal')
    
    del D21
    
    #Contraction 956; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('acdbklij->abcdijkl',E23)
    
    #Contraction 957; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bcdaklij->abcdijkl',E23)
    
    #Contraction 958; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('badcklij->abcdijkl',E23)
    
    #Contraction 959; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bacdklij->abcdijkl',E23)
    
    #Contraction 960; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('acdbjlik->abcdijkl',E23)
    
    #Contraction 961; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bcdajlik->abcdijkl',E23)
    
    #Contraction 962; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('badcjlik->abcdijkl',E23)
    
    #Contraction 963; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bacdjlik->abcdijkl',E23)
    
    #Contraction 964; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('acdbjkil->abcdijkl',E23)
    
    #Contraction 965; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bcdajkil->abcdijkl',E23)
    
    #Contraction 966; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('badcjkil->abcdijkl',E23)
    
    #Contraction 967; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bacdjkil->abcdijkl',E23)
    
    #Contraction 968; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('acdbiljk->abcdijkl',E23)
    
    #Contraction 969; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bcdailjk->abcdijkl',E23)
    
    #Contraction 970; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('badciljk->abcdijkl',E23)
    
    #Contraction 971; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bacdiljk->abcdijkl',E23)
    
    #Contraction 972; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('acdbikjl->abcdijkl',E23)
    
    #Contraction 973; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bcdaikjl->abcdijkl',E23)
    
    #Contraction 974; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('badcikjl->abcdijkl',E23)
    
    #Contraction 975; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bacdikjl->abcdijkl',E23)
    
    #Contraction 976; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('acdbijkl->abcdijkl',E23)
    
    #Contraction 977; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bcdaijkl->abcdijkl',E23)
    
    #Contraction 978; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('badcijkl->abcdijkl',E23)
    
    #Contraction 979; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bacdijkl->abcdijkl',E23)
    
    del E23
    
    G23 = np.zeros([nvir, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 980; Tree Level  2; Scaling  6/ 4 Result_size  4/ 4
    G23 += oe.contract('abim,mcdjkl->abcdijkl',T2,Y20, optimize='optimal')
    
    del Y20
    
    #Contraction 981; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('abcdijkl->abcdijkl',G23)
    
    #Contraction 982; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('acbdijkl->abcdijkl',G23)
    
    #Contraction 983; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('adbcijkl->abcdijkl',G23)
    
    #Contraction 984; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bcadijkl->abcdijkl',G23)
    
    #Contraction 985; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bdacijkl->abcdijkl',G23)
    
    #Contraction 986; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('cdabijkl->abcdijkl',G23)
    
    #Contraction 987; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('abcdjikl->abcdijkl',G23)
    
    #Contraction 988; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('acbdjikl->abcdijkl',G23)
    
    #Contraction 989; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('adbcjikl->abcdijkl',G23)
    
    #Contraction 990; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bcadjikl->abcdijkl',G23)
    
    #Contraction 991; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bdacjikl->abcdijkl',G23)
    
    #Contraction 992; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('cdabjikl->abcdijkl',G23)
    
    #Contraction 993; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('abcdkijl->abcdijkl',G23)
    
    #Contraction 994; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('acbdkijl->abcdijkl',G23)
    
    #Contraction 995; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('adbckijl->abcdijkl',G23)
    
    #Contraction 996; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bcadkijl->abcdijkl',G23)
    
    #Contraction 997; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bdackijl->abcdijkl',G23)
    
    #Contraction 998; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('cdabkijl->abcdijkl',G23)
    
    #Contraction 999; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('abcdlijk->abcdijkl',G23)
    
    #Contraction 1000; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('acbdlijk->abcdijkl',G23)
    
    #Contraction 1001; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('adbclijk->abcdijkl',G23)
    
    #Contraction 1002; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bcadlijk->abcdijkl',G23)
    
    #Contraction 1003; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bdaclijk->abcdijkl',G23)
    
    #Contraction 1004; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('cdablijk->abcdijkl',G23)
    
    del G23
    
    I23 = np.zeros([nvir, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 1005; Tree Level  2; Scaling  8/ 4 Result_size  4/ 4
    I23 += oe.contract('abcdmnkl,mnij->abcdklij',T4,Y6, optimize='optimal')
    
    #Contraction 1006; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('abcdklij->abcdijkl',I23)
    
    #Contraction 1007; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('abcdjlik->abcdijkl',I23)
    
    #Contraction 1008; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('abcdjkil->abcdijkl',I23)
    
    #Contraction 1009; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('abcdklji->abcdijkl',I23)
    
    #Contraction 1010; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('abcdiljk->abcdijkl',I23)
    
    #Contraction 1011; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('abcdikjl->abcdijkl',I23)
    
    #Contraction 1012; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('abcdjlki->abcdijkl',I23)
    
    #Contraction 1013; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('abcdilkj->abcdijkl',I23)
    
    #Contraction 1014; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('abcdijkl->abcdijkl',I23)
    
    #Contraction 1015; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('abcdjkli->abcdijkl',I23)
    
    #Contraction 1016; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('abcdiklj->abcdijkl',I23)
    
    #Contraction 1017; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('abcdijlk->abcdijkl',I23)
    
    del I23
    
    J23 = np.zeros([nvir, nvir, nvir, nvir, nocc, nocc], dtype=type_)
    
    #Contraction 1018; Tree Level  3; Scaling  4/ 6 Result_size  2/ 4
    J23 += oe.contract('amef,fcdmkl->acdekl',V8,T3, optimize='optimal')
    
    M23 = np.zeros([nvir, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 1019; Tree Level  2; Scaling  4/ 6 Result_size  4/ 4
    M23 += oe.contract('fbcdijkl,af->bcdaijkl',T4,I10, optimize='optimal')
    
    del I10
    
    #Contraction 1020; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bcdaijkl->abcdijkl',M23)
    
    #Contraction 1021; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('acdbijkl->abcdijkl',M23)
    
    #Contraction 1022; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('abdcijkl->abcdijkl',M23)
    
    #Contraction 1023; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('abcdijkl->abcdijkl',M23)
    
    del M23
    
    X23 = np.zeros([nvir, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 1024; Tree Level  2; Scaling  6/ 4 Result_size  4/ 4
    X23 += oe.contract('bani,cdnjkl->bacdijkl',T2,D22, optimize='optimal')
    
    del D22
    
    #Contraction 1025; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bacdijkl->abcdijkl',X23)
    
    #Contraction 1026; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bcadijkl->abcdijkl',X23)
    
    #Contraction 1027; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bdacijkl->abcdijkl',X23)
    
    #Contraction 1028; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('acbdijkl->abcdijkl',X23)
    
    #Contraction 1029; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('adbcijkl->abcdijkl',X23)
    
    #Contraction 1030; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('cdbaijkl->abcdijkl',X23)
    
    #Contraction 1031; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bacdjikl->abcdijkl',X23)
    
    #Contraction 1032; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bcadjikl->abcdijkl',X23)
    
    #Contraction 1033; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bdacjikl->abcdijkl',X23)
    
    #Contraction 1034; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('acbdjikl->abcdijkl',X23)
    
    #Contraction 1035; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('adbcjikl->abcdijkl',X23)
    
    #Contraction 1036; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('cdbajikl->abcdijkl',X23)
    
    #Contraction 1037; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bacdkijl->abcdijkl',X23)
    
    #Contraction 1038; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bcadkijl->abcdijkl',X23)
    
    #Contraction 1039; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bdackijl->abcdijkl',X23)
    
    #Contraction 1040; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('acbdkijl->abcdijkl',X23)
    
    #Contraction 1041; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('adbckijl->abcdijkl',X23)
    
    #Contraction 1042; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('cdbakijl->abcdijkl',X23)
    
    #Contraction 1043; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bacdlijk->abcdijkl',X23)
    
    #Contraction 1044; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bcadlijk->abcdijkl',X23)
    
    #Contraction 1045; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bdaclijk->abcdijkl',X23)
    
    #Contraction 1046; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('acbdlijk->abcdijkl',X23)
    
    #Contraction 1047; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('adbclijk->abcdijkl',X23)
    
    #Contraction 1048; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('cdbalijk->abcdijkl',X23)
    
    del X23
    
    Y23 = np.zeros([nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 1049; Tree Level  3; Scaling  6/ 2 Result_size  4/ 2
    Y23 += oe.contract('bdnl,mnij->bdmlij',T2,Y6, optimize='optimal')
    
    del Y6
    
    A24 = np.zeros([nvir, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 1050; Tree Level  2; Scaling  6/ 4 Result_size  4/ 4
    A24 += oe.contract('bcdnkl,naij->bcdaklij',T3,E13, optimize='optimal')
    
    del E13
    
    #Contraction 1051; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bcdaklij->abcdijkl',A24)
    
    #Contraction 1052; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('acdbklij->abcdijkl',A24)
    
    #Contraction 1053; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('abdcklij->abcdijkl',A24)
    
    #Contraction 1054; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('abcdklij->abcdijkl',A24)
    
    #Contraction 1055; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bcdajlik->abcdijkl',A24)
    
    #Contraction 1056; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('acdbjlik->abcdijkl',A24)
    
    #Contraction 1057; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('abdcjlik->abcdijkl',A24)
    
    #Contraction 1058; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('abcdjlik->abcdijkl',A24)
    
    #Contraction 1059; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bcdajkil->abcdijkl',A24)
    
    #Contraction 1060; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('acdbjkil->abcdijkl',A24)
    
    #Contraction 1061; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('abdcjkil->abcdijkl',A24)
    
    #Contraction 1062; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('abcdjkil->abcdijkl',A24)
    
    #Contraction 1063; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bcdailjk->abcdijkl',A24)
    
    #Contraction 1064; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('acdbiljk->abcdijkl',A24)
    
    #Contraction 1065; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('abdciljk->abcdijkl',A24)
    
    #Contraction 1066; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('abcdiljk->abcdijkl',A24)
    
    #Contraction 1067; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bcdaikjl->abcdijkl',A24)
    
    #Contraction 1068; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('acdbikjl->abcdijkl',A24)
    
    #Contraction 1069; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('abdcikjl->abcdijkl',A24)
    
    #Contraction 1070; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('abcdikjl->abcdijkl',A24)
    
    #Contraction 1071; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bcdaijkl->abcdijkl',A24)
    
    #Contraction 1072; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('acdbijkl->abcdijkl',A24)
    
    #Contraction 1073; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('abdcijkl->abcdijkl',A24)
    
    #Contraction 1074; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('abcdijkl->abcdijkl',A24)
    
    del A24
    
    D24 = np.zeros([nvir, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 1075; Tree Level  2; Scaling  6/ 4 Result_size  4/ 4
    D24 += oe.contract('bcdmkl,amij->bcdaklij',T3,I9, optimize='optimal')
    
    del I9
    
    #Contraction 1076; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('bcdaklij->abcdijkl',D24)
    
    #Contraction 1077; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('acdbklij->abcdijkl',D24)
    
    #Contraction 1078; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('abdcklij->abcdijkl',D24)
    
    #Contraction 1079; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('abcdklij->abcdijkl',D24)
    
    #Contraction 1080; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('bcdajlik->abcdijkl',D24)
    
    #Contraction 1081; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('acdbjlik->abcdijkl',D24)
    
    #Contraction 1082; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('abdcjlik->abcdijkl',D24)
    
    #Contraction 1083; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('abcdjlik->abcdijkl',D24)
    
    #Contraction 1084; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('bcdajkil->abcdijkl',D24)
    
    #Contraction 1085; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('acdbjkil->abcdijkl',D24)
    
    #Contraction 1086; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('abdcjkil->abcdijkl',D24)
    
    #Contraction 1087; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('abcdjkil->abcdijkl',D24)
    
    #Contraction 1088; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('bcdaklji->abcdijkl',D24)
    
    #Contraction 1089; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('acdbklji->abcdijkl',D24)
    
    #Contraction 1090; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('abdcklji->abcdijkl',D24)
    
    #Contraction 1091; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('abcdklji->abcdijkl',D24)
    
    #Contraction 1092; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('bcdailjk->abcdijkl',D24)
    
    #Contraction 1093; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('acdbiljk->abcdijkl',D24)
    
    #Contraction 1094; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('abdciljk->abcdijkl',D24)
    
    #Contraction 1095; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('abcdiljk->abcdijkl',D24)
    
    #Contraction 1096; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('bcdaikjl->abcdijkl',D24)
    
    #Contraction 1097; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('acdbikjl->abcdijkl',D24)
    
    #Contraction 1098; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('abdcikjl->abcdijkl',D24)
    
    #Contraction 1099; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('abcdikjl->abcdijkl',D24)
    
    #Contraction 1100; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('bcdajlki->abcdijkl',D24)
    
    #Contraction 1101; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('acdbjlki->abcdijkl',D24)
    
    #Contraction 1102; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('abdcjlki->abcdijkl',D24)
    
    #Contraction 1103; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('abcdjlki->abcdijkl',D24)
    
    #Contraction 1104; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('bcdailkj->abcdijkl',D24)
    
    #Contraction 1105; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('acdbilkj->abcdijkl',D24)
    
    #Contraction 1106; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('abdcilkj->abcdijkl',D24)
    
    #Contraction 1107; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('abcdilkj->abcdijkl',D24)
    
    #Contraction 1108; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('bcdaijkl->abcdijkl',D24)
    
    #Contraction 1109; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('acdbijkl->abcdijkl',D24)
    
    #Contraction 1110; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('abdcijkl->abcdijkl',D24)
    
    #Contraction 1111; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('abcdijkl->abcdijkl',D24)
    
    #Contraction 1112; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('bcdajkli->abcdijkl',D24)
    
    #Contraction 1113; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('acdbjkli->abcdijkl',D24)
    
    #Contraction 1114; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('abdcjkli->abcdijkl',D24)
    
    #Contraction 1115; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('abcdjkli->abcdijkl',D24)
    
    #Contraction 1116; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('bcdaiklj->abcdijkl',D24)
    
    #Contraction 1117; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('acdbiklj->abcdijkl',D24)
    
    #Contraction 1118; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('abdciklj->abcdijkl',D24)
    
    #Contraction 1119; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('abcdiklj->abcdijkl',D24)
    
    #Contraction 1120; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('bcdaijlk->abcdijkl',D24)
    
    #Contraction 1121; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('acdbijlk->abcdijkl',D24)
    
    #Contraction 1122; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('abdcijlk->abcdijkl',D24)
    
    #Contraction 1123; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('abcdijlk->abcdijkl',D24)
    
    del D24
    
    E24 = np.zeros([nvir, nocc, nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 1124; Tree Level  3; Scaling  4/ 4 Result_size  4/ 2
    E24 += oe.contract('amie,bejk->ambijk',V5,T2, optimize='optimal')
    
    G24 = np.zeros([nvir, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 1125; Tree Level  2; Scaling  6/ 6 Result_size  4/ 4
    G24 += oe.contract('amie,bcdejklm->abcdijkl',V5,T4, optimize='optimal')
    
    #Contraction 1126; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('abcdijkl->abcdijkl',G24)
    
    #Contraction 1127; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bacdijkl->abcdijkl',G24)
    
    #Contraction 1128; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('cabdijkl->abcdijkl',G24)
    
    #Contraction 1129; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('dabcijkl->abcdijkl',G24)
    
    #Contraction 1130; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('abcdjikl->abcdijkl',G24)
    
    #Contraction 1131; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bacdjikl->abcdijkl',G24)
    
    #Contraction 1132; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('cabdjikl->abcdijkl',G24)
    
    #Contraction 1133; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('dabcjikl->abcdijkl',G24)
    
    #Contraction 1134; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('abcdkijl->abcdijkl',G24)
    
    #Contraction 1135; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bacdkijl->abcdijkl',G24)
    
    #Contraction 1136; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('cabdkijl->abcdijkl',G24)
    
    #Contraction 1137; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('dabckijl->abcdijkl',G24)
    
    #Contraction 1138; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('abcdlijk->abcdijkl',G24)
    
    #Contraction 1139; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bacdlijk->abcdijkl',G24)
    
    #Contraction 1140; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('cabdlijk->abcdijkl',G24)
    
    #Contraction 1141; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('dabclijk->abcdijkl',G24)
    
    del G24
    
    I24 = np.zeros([nvir, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 1142; Tree Level  2; Scaling  6/ 6 Result_size  4/ 4
    I24 += oe.contract('fbcdmjkl,amfi->bcdajkli',T4,G5, optimize='optimal')
    
    del G5
    
    #Contraction 1143; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bcdajkli->abcdijkl',I24)
    
    #Contraction 1144; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('acdbjkli->abcdijkl',I24)
    
    #Contraction 1145; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('abdcjkli->abcdijkl',I24)
    
    #Contraction 1146; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('abcdjkli->abcdijkl',I24)
    
    #Contraction 1147; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bcdaiklj->abcdijkl',I24)
    
    #Contraction 1148; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('acdbiklj->abcdijkl',I24)
    
    #Contraction 1149; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('abdciklj->abcdijkl',I24)
    
    #Contraction 1150; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('abcdiklj->abcdijkl',I24)
    
    #Contraction 1151; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bcdaijlk->abcdijkl',I24)
    
    #Contraction 1152; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('acdbijlk->abcdijkl',I24)
    
    #Contraction 1153; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('abdcijlk->abcdijkl',I24)
    
    #Contraction 1154; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('abcdijlk->abcdijkl',I24)
    
    #Contraction 1155; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bcdaijkl->abcdijkl',I24)
    
    #Contraction 1156; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('acdbijkl->abcdijkl',I24)
    
    #Contraction 1157; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('abdcijkl->abcdijkl',I24)
    
    #Contraction 1158; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('abcdijkl->abcdijkl',I24)
    
    del I24
    
    J24 = np.zeros([nocc, nocc, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 1159; Tree Level  4; Scaling  6/ 4 Result_size  6/ 2
    J24 += oe.contract('mnie,ecdjkl->mncdijkl',V4,T3, optimize='optimal')
    
    M24 = np.zeros([nvir, nocc, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 1160; Tree Level  3; Scaling  7/ 3 Result_size  5/ 3
    M24 += oe.contract('bn,mncdijkl->bmcdijkl',T1,J24, optimize='optimal')
    
    del J24
    
    X24 = np.zeros([nvir, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 1161; Tree Level  2; Scaling  6/ 4 Result_size  4/ 4
    X24 += oe.contract('am,bmcdijkl->abcdijkl',T1,M24, optimize='optimal')
    
    del M24
    
    #Contraction 1162; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('abcdijkl->abcdijkl',X24)
    
    #Contraction 1163; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('acbdijkl->abcdijkl',X24)
    
    #Contraction 1164; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('adbcijkl->abcdijkl',X24)
    
    #Contraction 1165; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('bacdijkl->abcdijkl',X24)
    
    #Contraction 1166; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('bcadijkl->abcdijkl',X24)
    
    #Contraction 1167; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('bdacijkl->abcdijkl',X24)
    
    #Contraction 1168; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('cabdijkl->abcdijkl',X24)
    
    #Contraction 1169; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('cbadijkl->abcdijkl',X24)
    
    #Contraction 1170; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('cdabijkl->abcdijkl',X24)
    
    #Contraction 1171; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('dabcijkl->abcdijkl',X24)
    
    #Contraction 1172; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('dbacijkl->abcdijkl',X24)
    
    #Contraction 1173; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('dcabijkl->abcdijkl',X24)
    
    #Contraction 1174; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('abcdjikl->abcdijkl',X24)
    
    #Contraction 1175; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('acbdjikl->abcdijkl',X24)
    
    #Contraction 1176; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('adbcjikl->abcdijkl',X24)
    
    #Contraction 1177; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('bacdjikl->abcdijkl',X24)
    
    #Contraction 1178; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('bcadjikl->abcdijkl',X24)
    
    #Contraction 1179; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('bdacjikl->abcdijkl',X24)
    
    #Contraction 1180; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('cabdjikl->abcdijkl',X24)
    
    #Contraction 1181; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('cbadjikl->abcdijkl',X24)
    
    #Contraction 1182; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('cdabjikl->abcdijkl',X24)
    
    #Contraction 1183; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('dabcjikl->abcdijkl',X24)
    
    #Contraction 1184; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('dbacjikl->abcdijkl',X24)
    
    #Contraction 1185; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('dcabjikl->abcdijkl',X24)
    
    #Contraction 1186; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('abcdkijl->abcdijkl',X24)
    
    #Contraction 1187; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('acbdkijl->abcdijkl',X24)
    
    #Contraction 1188; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('adbckijl->abcdijkl',X24)
    
    #Contraction 1189; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('bacdkijl->abcdijkl',X24)
    
    #Contraction 1190; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('bcadkijl->abcdijkl',X24)
    
    #Contraction 1191; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('bdackijl->abcdijkl',X24)
    
    #Contraction 1192; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('cabdkijl->abcdijkl',X24)
    
    #Contraction 1193; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('cbadkijl->abcdijkl',X24)
    
    #Contraction 1194; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('cdabkijl->abcdijkl',X24)
    
    #Contraction 1195; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('dabckijl->abcdijkl',X24)
    
    #Contraction 1196; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('dbackijl->abcdijkl',X24)
    
    #Contraction 1197; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('dcabkijl->abcdijkl',X24)
    
    #Contraction 1198; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('abcdlijk->abcdijkl',X24)
    
    #Contraction 1199; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('acbdlijk->abcdijkl',X24)
    
    #Contraction 1200; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('adbclijk->abcdijkl',X24)
    
    #Contraction 1201; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('bacdlijk->abcdijkl',X24)
    
    #Contraction 1202; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('bcadlijk->abcdijkl',X24)
    
    #Contraction 1203; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('bdaclijk->abcdijkl',X24)
    
    #Contraction 1204; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('cabdlijk->abcdijkl',X24)
    
    #Contraction 1205; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('cbadlijk->abcdijkl',X24)
    
    #Contraction 1206; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('cdablijk->abcdijkl',X24)
    
    #Contraction 1207; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('dabclijk->abcdijkl',X24)
    
    #Contraction 1208; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('dbaclijk->abcdijkl',X24)
    
    #Contraction 1209; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('dcablijk->abcdijkl',X24)
    
    del X24
    
    Y24 = np.zeros([nvir, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 1210; Tree Level  2; Scaling  4/ 6 Result_size  4/ 4
    Y24 += oe.contract('ecdjkl,abie->cdabjkli',T3,M18, optimize='optimal')
    
    del M18
    
    #Contraction 1211; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('cdabjkli->abcdijkl',Y24)
    
    #Contraction 1212; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('bdacjkli->abcdijkl',Y24)
    
    #Contraction 1213; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('bcadjkli->abcdijkl',Y24)
    
    #Contraction 1214; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('adbcjkli->abcdijkl',Y24)
    
    #Contraction 1215; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('acbdjkli->abcdijkl',Y24)
    
    #Contraction 1216; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('abcdjkli->abcdijkl',Y24)
    
    #Contraction 1217; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('cdabiklj->abcdijkl',Y24)
    
    #Contraction 1218; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('bdaciklj->abcdijkl',Y24)
    
    #Contraction 1219; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('bcadiklj->abcdijkl',Y24)
    
    #Contraction 1220; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('adbciklj->abcdijkl',Y24)
    
    #Contraction 1221; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('acbdiklj->abcdijkl',Y24)
    
    #Contraction 1222; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('abcdiklj->abcdijkl',Y24)
    
    #Contraction 1223; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('cdabijlk->abcdijkl',Y24)
    
    #Contraction 1224; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('bdacijlk->abcdijkl',Y24)
    
    #Contraction 1225; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('bcadijlk->abcdijkl',Y24)
    
    #Contraction 1226; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('adbcijlk->abcdijkl',Y24)
    
    #Contraction 1227; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('acbdijlk->abcdijkl',Y24)
    
    #Contraction 1228; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('abcdijlk->abcdijkl',Y24)
    
    #Contraction 1229; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('cdabijkl->abcdijkl',Y24)
    
    #Contraction 1230; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('bdacijkl->abcdijkl',Y24)
    
    #Contraction 1231; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('bcadijkl->abcdijkl',Y24)
    
    #Contraction 1232; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('adbcijkl->abcdijkl',Y24)
    
    #Contraction 1233; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('acbdijkl->abcdijkl',Y24)
    
    #Contraction 1234; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('abcdijkl->abcdijkl',Y24)
    
    del Y24
    
    A25 = np.zeros([nvir, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 1235; Tree Level  2; Scaling  8/ 4 Result_size  4/ 4
    A25 += oe.contract('abcdmnkl,mnji->abcdklji',T4,E7, optimize='optimal')
    
    #Contraction 1236; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('abcdklji->abcdijkl',A25)
    
    #Contraction 1237; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('abcdjlki->abcdijkl',A25)
    
    #Contraction 1238; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('abcdjkli->abcdijkl',A25)
    
    #Contraction 1239; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('abcdklij->abcdijkl',A25)
    
    #Contraction 1240; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('abcdilkj->abcdijkl',A25)
    
    #Contraction 1241; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('abcdiklj->abcdijkl',A25)
    
    #Contraction 1242; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('abcdjlik->abcdijkl',A25)
    
    #Contraction 1243; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('abcdiljk->abcdijkl',A25)
    
    #Contraction 1244; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('abcdijlk->abcdijkl',A25)
    
    #Contraction 1245; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('abcdjkil->abcdijkl',A25)
    
    #Contraction 1246; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('abcdikjl->abcdijkl',A25)
    
    #Contraction 1247; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('abcdijkl->abcdijkl',A25)
    
    del A25
    
    D25 = np.zeros([nvir, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 1248; Tree Level  2; Scaling  6/ 4 Result_size  4/ 4
    D25 += oe.contract('am,bmcdijkl->abcdijkl',T1,A22, optimize='optimal')
    
    del A22
    
    #Contraction 1249; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('abcdijkl->abcdijkl',D25)
    
    #Contraction 1250; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('acbdijkl->abcdijkl',D25)
    
    #Contraction 1251; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('adbcijkl->abcdijkl',D25)
    
    #Contraction 1252; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bacdijkl->abcdijkl',D25)
    
    #Contraction 1253; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bcadijkl->abcdijkl',D25)
    
    #Contraction 1254; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bdacijkl->abcdijkl',D25)
    
    #Contraction 1255; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('cabdijkl->abcdijkl',D25)
    
    #Contraction 1256; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('cbadijkl->abcdijkl',D25)
    
    #Contraction 1257; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('cdabijkl->abcdijkl',D25)
    
    #Contraction 1258; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('dabcijkl->abcdijkl',D25)
    
    #Contraction 1259; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('dbacijkl->abcdijkl',D25)
    
    #Contraction 1260; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('dcabijkl->abcdijkl',D25)
    
    #Contraction 1261; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('abcdikjl->abcdijkl',D25)
    
    #Contraction 1262; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('acbdikjl->abcdijkl',D25)
    
    #Contraction 1263; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('adbcikjl->abcdijkl',D25)
    
    #Contraction 1264; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bacdikjl->abcdijkl',D25)
    
    #Contraction 1265; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bcadikjl->abcdijkl',D25)
    
    #Contraction 1266; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bdacikjl->abcdijkl',D25)
    
    #Contraction 1267; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('cabdikjl->abcdijkl',D25)
    
    #Contraction 1268; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('cbadikjl->abcdijkl',D25)
    
    #Contraction 1269; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('cdabikjl->abcdijkl',D25)
    
    #Contraction 1270; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('dabcikjl->abcdijkl',D25)
    
    #Contraction 1271; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('dbacikjl->abcdijkl',D25)
    
    #Contraction 1272; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('dcabikjl->abcdijkl',D25)
    
    #Contraction 1273; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('abcdiljk->abcdijkl',D25)
    
    #Contraction 1274; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('acbdiljk->abcdijkl',D25)
    
    #Contraction 1275; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('adbciljk->abcdijkl',D25)
    
    #Contraction 1276; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bacdiljk->abcdijkl',D25)
    
    #Contraction 1277; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bcadiljk->abcdijkl',D25)
    
    #Contraction 1278; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bdaciljk->abcdijkl',D25)
    
    #Contraction 1279; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('cabdiljk->abcdijkl',D25)
    
    #Contraction 1280; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('cbadiljk->abcdijkl',D25)
    
    #Contraction 1281; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('cdabiljk->abcdijkl',D25)
    
    #Contraction 1282; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('dabciljk->abcdijkl',D25)
    
    #Contraction 1283; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('dbaciljk->abcdijkl',D25)
    
    #Contraction 1284; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('dcabiljk->abcdijkl',D25)
    
    #Contraction 1285; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('abcdjkil->abcdijkl',D25)
    
    #Contraction 1286; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('acbdjkil->abcdijkl',D25)
    
    #Contraction 1287; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('adbcjkil->abcdijkl',D25)
    
    #Contraction 1288; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bacdjkil->abcdijkl',D25)
    
    #Contraction 1289; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bcadjkil->abcdijkl',D25)
    
    #Contraction 1290; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bdacjkil->abcdijkl',D25)
    
    #Contraction 1291; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('cabdjkil->abcdijkl',D25)
    
    #Contraction 1292; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('cbadjkil->abcdijkl',D25)
    
    #Contraction 1293; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('cdabjkil->abcdijkl',D25)
    
    #Contraction 1294; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('dabcjkil->abcdijkl',D25)
    
    #Contraction 1295; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('dbacjkil->abcdijkl',D25)
    
    #Contraction 1296; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('dcabjkil->abcdijkl',D25)
    
    #Contraction 1297; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('abcdjlik->abcdijkl',D25)
    
    #Contraction 1298; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('acbdjlik->abcdijkl',D25)
    
    #Contraction 1299; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('adbcjlik->abcdijkl',D25)
    
    #Contraction 1300; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bacdjlik->abcdijkl',D25)
    
    #Contraction 1301; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bcadjlik->abcdijkl',D25)
    
    #Contraction 1302; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bdacjlik->abcdijkl',D25)
    
    #Contraction 1303; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('cabdjlik->abcdijkl',D25)
    
    #Contraction 1304; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('cbadjlik->abcdijkl',D25)
    
    #Contraction 1305; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('cdabjlik->abcdijkl',D25)
    
    #Contraction 1306; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('dabcjlik->abcdijkl',D25)
    
    #Contraction 1307; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('dbacjlik->abcdijkl',D25)
    
    #Contraction 1308; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('dcabjlik->abcdijkl',D25)
    
    #Contraction 1309; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('abcdklij->abcdijkl',D25)
    
    #Contraction 1310; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('acbdklij->abcdijkl',D25)
    
    #Contraction 1311; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('adbcklij->abcdijkl',D25)
    
    #Contraction 1312; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bacdklij->abcdijkl',D25)
    
    #Contraction 1313; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bcadklij->abcdijkl',D25)
    
    #Contraction 1314; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bdacklij->abcdijkl',D25)
    
    #Contraction 1315; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('cabdklij->abcdijkl',D25)
    
    #Contraction 1316; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('cbadklij->abcdijkl',D25)
    
    #Contraction 1317; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('cdabklij->abcdijkl',D25)
    
    #Contraction 1318; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('dabcklij->abcdijkl',D25)
    
    #Contraction 1319; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('dbacklij->abcdijkl',D25)
    
    #Contraction 1320; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('dcabklij->abcdijkl',D25)
    
    del D25
    
    E25 = np.zeros([nvir, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 1321; Tree Level  2; Scaling  6/ 4 Result_size  4/ 4
    E25 += oe.contract('abim,mcdjkl->abcdijkl',T2,E14, optimize='optimal')
    
    del E14
    
    #Contraction 1322; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('abcdijkl->abcdijkl',E25)
    
    #Contraction 1323; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('acbdijkl->abcdijkl',E25)
    
    #Contraction 1324; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('adbcijkl->abcdijkl',E25)
    
    #Contraction 1325; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('bcadijkl->abcdijkl',E25)
    
    #Contraction 1326; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('bdacijkl->abcdijkl',E25)
    
    #Contraction 1327; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('cdabijkl->abcdijkl',E25)
    
    #Contraction 1328; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('abcdjikl->abcdijkl',E25)
    
    #Contraction 1329; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('acbdjikl->abcdijkl',E25)
    
    #Contraction 1330; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('adbcjikl->abcdijkl',E25)
    
    #Contraction 1331; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('bcadjikl->abcdijkl',E25)
    
    #Contraction 1332; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('bdacjikl->abcdijkl',E25)
    
    #Contraction 1333; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('cdabjikl->abcdijkl',E25)
    
    #Contraction 1334; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('abcdkijl->abcdijkl',E25)
    
    #Contraction 1335; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('acbdkijl->abcdijkl',E25)
    
    #Contraction 1336; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('adbckijl->abcdijkl',E25)
    
    #Contraction 1337; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('bcadkijl->abcdijkl',E25)
    
    #Contraction 1338; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('bdackijl->abcdijkl',E25)
    
    #Contraction 1339; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('cdabkijl->abcdijkl',E25)
    
    #Contraction 1340; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('abcdlijk->abcdijkl',E25)
    
    #Contraction 1341; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('acbdlijk->abcdijkl',E25)
    
    #Contraction 1342; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('adbclijk->abcdijkl',E25)
    
    #Contraction 1343; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('bcadlijk->abcdijkl',E25)
    
    #Contraction 1344; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('bdaclijk->abcdijkl',E25)
    
    #Contraction 1345; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('cdablijk->abcdijkl',E25)
    
    del E25
    
    G25 = np.zeros([nvir, nocc, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 1346; Tree Level  3; Scaling  5/ 5 Result_size  5/ 3
    G25 += oe.contract('amie,ecdjkl->amcdijkl',V5,T3, optimize='optimal')
    
    #del V5
    
    I25 = np.zeros([nocc, nvir, nocc, nocc], dtype=type_)
    
    #Contraction 1347; Tree Level  3; Scaling  5/ 3 Result_size  3/ 1
    I25 += oe.contract('mnie,bejm->nbij',V4,T2, optimize='optimal')
    
    J25 = np.zeros([nvir, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 1348; Tree Level  2; Scaling  6/ 4 Result_size  4/ 4
    J25 += oe.contract('acdnkl,nbij->acdbklij',T3,I25, optimize='optimal')
    
    del I25
    
    #Contraction 1349; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('acdbklij->abcdijkl',J25)
    
    #Contraction 1350; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bcdaklij->abcdijkl',J25)
    
    #Contraction 1351; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('badcklij->abcdijkl',J25)
    
    #Contraction 1352; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bacdklij->abcdijkl',J25)
    
    #Contraction 1353; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('acdbjlik->abcdijkl',J25)
    
    #Contraction 1354; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bcdajlik->abcdijkl',J25)
    
    #Contraction 1355; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('badcjlik->abcdijkl',J25)
    
    #Contraction 1356; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bacdjlik->abcdijkl',J25)
    
    #Contraction 1357; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('acdbjkil->abcdijkl',J25)
    
    #Contraction 1358; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bcdajkil->abcdijkl',J25)
    
    #Contraction 1359; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('badcjkil->abcdijkl',J25)
    
    #Contraction 1360; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bacdjkil->abcdijkl',J25)
    
    #Contraction 1361; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('acdbklji->abcdijkl',J25)
    
    #Contraction 1362; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bcdaklji->abcdijkl',J25)
    
    #Contraction 1363; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('badcklji->abcdijkl',J25)
    
    #Contraction 1364; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bacdklji->abcdijkl',J25)
    
    #Contraction 1365; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('acdbiljk->abcdijkl',J25)
    
    #Contraction 1366; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bcdailjk->abcdijkl',J25)
    
    #Contraction 1367; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('badciljk->abcdijkl',J25)
    
    #Contraction 1368; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bacdiljk->abcdijkl',J25)
    
    #Contraction 1369; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('acdbikjl->abcdijkl',J25)
    
    #Contraction 1370; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bcdaikjl->abcdijkl',J25)
    
    #Contraction 1371; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('badcikjl->abcdijkl',J25)
    
    #Contraction 1372; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bacdikjl->abcdijkl',J25)
    
    #Contraction 1373; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('acdbjlki->abcdijkl',J25)
    
    #Contraction 1374; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bcdajlki->abcdijkl',J25)
    
    #Contraction 1375; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('badcjlki->abcdijkl',J25)
    
    #Contraction 1376; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bacdjlki->abcdijkl',J25)
    
    #Contraction 1377; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('acdbilkj->abcdijkl',J25)
    
    #Contraction 1378; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bcdailkj->abcdijkl',J25)
    
    #Contraction 1379; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('badcilkj->abcdijkl',J25)
    
    #Contraction 1380; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bacdilkj->abcdijkl',J25)
    
    #Contraction 1381; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('acdbijkl->abcdijkl',J25)
    
    #Contraction 1382; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bcdaijkl->abcdijkl',J25)
    
    #Contraction 1383; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('badcijkl->abcdijkl',J25)
    
    #Contraction 1384; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bacdijkl->abcdijkl',J25)
    
    #Contraction 1385; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('acdbjkli->abcdijkl',J25)
    
    #Contraction 1386; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bcdajkli->abcdijkl',J25)
    
    #Contraction 1387; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('badcjkli->abcdijkl',J25)
    
    #Contraction 1388; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bacdjkli->abcdijkl',J25)
    
    #Contraction 1389; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('acdbiklj->abcdijkl',J25)
    
    #Contraction 1390; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bcdaiklj->abcdijkl',J25)
    
    #Contraction 1391; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('badciklj->abcdijkl',J25)
    
    #Contraction 1392; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bacdiklj->abcdijkl',J25)
    
    #Contraction 1393; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('acdbijlk->abcdijkl',J25)
    
    #Contraction 1394; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bcdaijlk->abcdijkl',J25)
    
    #Contraction 1395; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('badcijlk->abcdijkl',J25)
    
    #Contraction 1396; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bacdijlk->abcdijkl',J25)
    
    del J25
    
    M25 = np.zeros([nvir, nocc, nvir, nvir, nocc, nocc], dtype=type_)
    
    #Contraction 1397; Tree Level  4; Scaling  3/ 5 Result_size  3/ 3
    M25 += oe.contract('amef,fdjl->amdejl',V8,T2, optimize='optimal')
    
    X25 = np.zeros([nvir, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 1398; Tree Level  2; Scaling  8/ 4 Result_size  4/ 4
    X25 += oe.contract('acdmnl,mnbijk->acdblijk',T3,J22, optimize='optimal')
    
    del J22
    
    #Contraction 1399; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('acdblijk->abcdijkl',X25)
    
    #Contraction 1400; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('bcdalijk->abcdijkl',X25)
    
    #Contraction 1401; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('badclijk->abcdijkl',X25)
    
    #Contraction 1402; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('bacdlijk->abcdijkl',X25)
    
    #Contraction 1403; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('acdbkijl->abcdijkl',X25)
    
    #Contraction 1404; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('bcdakijl->abcdijkl',X25)
    
    #Contraction 1405; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('badckijl->abcdijkl',X25)
    
    #Contraction 1406; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('bacdkijl->abcdijkl',X25)
    
    #Contraction 1407; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('acdbjikl->abcdijkl',X25)
    
    #Contraction 1408; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('bcdajikl->abcdijkl',X25)
    
    #Contraction 1409; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('badcjikl->abcdijkl',X25)
    
    #Contraction 1410; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('bacdjikl->abcdijkl',X25)
    
    #Contraction 1411; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('acdbljik->abcdijkl',X25)
    
    #Contraction 1412; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('bcdaljik->abcdijkl',X25)
    
    #Contraction 1413; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('badcljik->abcdijkl',X25)
    
    #Contraction 1414; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('bacdljik->abcdijkl',X25)
    
    #Contraction 1415; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('acdbkjil->abcdijkl',X25)
    
    #Contraction 1416; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('bcdakjil->abcdijkl',X25)
    
    #Contraction 1417; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('badckjil->abcdijkl',X25)
    
    #Contraction 1418; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('bacdkjil->abcdijkl',X25)
    
    #Contraction 1419; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('acdbijkl->abcdijkl',X25)
    
    #Contraction 1420; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('bcdaijkl->abcdijkl',X25)
    
    #Contraction 1421; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('badcijkl->abcdijkl',X25)
    
    #Contraction 1422; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('bacdijkl->abcdijkl',X25)
    
    #Contraction 1423; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('acdblkij->abcdijkl',X25)
    
    #Contraction 1424; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('bcdalkij->abcdijkl',X25)
    
    #Contraction 1425; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('badclkij->abcdijkl',X25)
    
    #Contraction 1426; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('bacdlkij->abcdijkl',X25)
    
    #Contraction 1427; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('acdbjkil->abcdijkl',X25)
    
    #Contraction 1428; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('bcdajkil->abcdijkl',X25)
    
    #Contraction 1429; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('badcjkil->abcdijkl',X25)
    
    #Contraction 1430; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('bacdjkil->abcdijkl',X25)
    
    #Contraction 1431; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('acdbikjl->abcdijkl',X25)
    
    #Contraction 1432; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('bcdaikjl->abcdijkl',X25)
    
    #Contraction 1433; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('badcikjl->abcdijkl',X25)
    
    #Contraction 1434; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('bacdikjl->abcdijkl',X25)
    
    #Contraction 1435; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('acdbklij->abcdijkl',X25)
    
    #Contraction 1436; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('bcdaklij->abcdijkl',X25)
    
    #Contraction 1437; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('badcklij->abcdijkl',X25)
    
    #Contraction 1438; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('bacdklij->abcdijkl',X25)
    
    #Contraction 1439; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('acdbjlik->abcdijkl',X25)
    
    #Contraction 1440; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('bcdajlik->abcdijkl',X25)
    
    #Contraction 1441; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('badcjlik->abcdijkl',X25)
    
    #Contraction 1442; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('bacdjlik->abcdijkl',X25)
    
    #Contraction 1443; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('acdbiljk->abcdijkl',X25)
    
    #Contraction 1444; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('bcdailjk->abcdijkl',X25)
    
    #Contraction 1445; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('badciljk->abcdijkl',X25)
    
    #Contraction 1446; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('bacdiljk->abcdijkl',X25)
    
    del X25
    
    Y25 = np.zeros([nvir, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 1447; Tree Level  2; Scaling  6/ 4 Result_size  4/ 4
    Y25 += oe.contract('abcijm,mdkl->abcdijkl',T3,M9, optimize='optimal')
    
    del M9
    
    #Contraction 1448; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('abcdijkl->abcdijkl',Y25)
    
    #Contraction 1449; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('abdcijkl->abcdijkl',Y25)
    
    #Contraction 1450; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('acdbijkl->abcdijkl',Y25)
    
    #Contraction 1451; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('bcdaijkl->abcdijkl',Y25)
    
    #Contraction 1452; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('abcdikjl->abcdijkl',Y25)
    
    #Contraction 1453; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('abdcikjl->abcdijkl',Y25)
    
    #Contraction 1454; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('acdbikjl->abcdijkl',Y25)
    
    #Contraction 1455; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('bcdaikjl->abcdijkl',Y25)
    
    #Contraction 1456; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('abcdiljk->abcdijkl',Y25)
    
    #Contraction 1457; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('abdciljk->abcdijkl',Y25)
    
    #Contraction 1458; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('acdbiljk->abcdijkl',Y25)
    
    #Contraction 1459; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('bcdailjk->abcdijkl',Y25)
    
    #Contraction 1460; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('abcdjkil->abcdijkl',Y25)
    
    #Contraction 1461; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('abdcjkil->abcdijkl',Y25)
    
    #Contraction 1462; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('acdbjkil->abcdijkl',Y25)
    
    #Contraction 1463; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('bcdajkil->abcdijkl',Y25)
    
    #Contraction 1464; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('abcdjlik->abcdijkl',Y25)
    
    #Contraction 1465; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('abdcjlik->abcdijkl',Y25)
    
    #Contraction 1466; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('acdbjlik->abcdijkl',Y25)
    
    #Contraction 1467; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('bcdajlik->abcdijkl',Y25)
    
    #Contraction 1468; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('abcdklij->abcdijkl',Y25)
    
    #Contraction 1469; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('abdcklij->abcdijkl',Y25)
    
    #Contraction 1470; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('acdbklij->abcdijkl',Y25)
    
    #Contraction 1471; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('bcdaklij->abcdijkl',Y25)
    
    del Y25
    
    A26 = np.zeros([nvir, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 1472; Tree Level  2; Scaling  6/ 4 Result_size  4/ 4
    A26 += oe.contract('bcdmkl,amij->bcdaklij',T3,J6, optimize='optimal')
    
    del J6
    
    #Contraction 1473; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('bcdaklij->abcdijkl',A26)
    
    #Contraction 1474; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('acdbklij->abcdijkl',A26)
    
    #Contraction 1475; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('abdcklij->abcdijkl',A26)
    
    #Contraction 1476; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('abcdklij->abcdijkl',A26)
    
    #Contraction 1477; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('bcdajlik->abcdijkl',A26)
    
    #Contraction 1478; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('acdbjlik->abcdijkl',A26)
    
    #Contraction 1479; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('abdcjlik->abcdijkl',A26)
    
    #Contraction 1480; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('abcdjlik->abcdijkl',A26)
    
    #Contraction 1481; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('bcdajkil->abcdijkl',A26)
    
    #Contraction 1482; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('acdbjkil->abcdijkl',A26)
    
    #Contraction 1483; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('abdcjkil->abcdijkl',A26)
    
    #Contraction 1484; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('abcdjkil->abcdijkl',A26)
    
    #Contraction 1485; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('bcdailjk->abcdijkl',A26)
    
    #Contraction 1486; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('acdbiljk->abcdijkl',A26)
    
    #Contraction 1487; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('abdciljk->abcdijkl',A26)
    
    #Contraction 1488; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('abcdiljk->abcdijkl',A26)
    
    #Contraction 1489; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('bcdaikjl->abcdijkl',A26)
    
    #Contraction 1490; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('acdbikjl->abcdijkl',A26)
    
    #Contraction 1491; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('abdcikjl->abcdijkl',A26)
    
    #Contraction 1492; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('abcdikjl->abcdijkl',A26)
    
    #Contraction 1493; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('bcdaijkl->abcdijkl',A26)
    
    #Contraction 1494; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('acdbijkl->abcdijkl',A26)
    
    #Contraction 1495; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('abdcijkl->abcdijkl',A26)
    
    #Contraction 1496; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('abcdijkl->abcdijkl',A26)
    
    del A26
    
    D26 = np.zeros([nvir, nvir, nocc, nvir, nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 1497; Tree Level  3; Scaling  5/ 5 Result_size  5/ 3
    D26 += oe.contract('ecik,amdejl->camdikjl',T2,M25, optimize='optimal')
    
    del M25
    
    E26 = np.zeros([nvir, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 1498; Tree Level  2; Scaling  6/ 4 Result_size  4/ 4
    E26 += oe.contract('bm,camdikjl->bcadikjl',T1,D26, optimize='optimal')
    
    del D26
    
    #Contraction 1499; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('bcadikjl->abcdijkl',E26)
    
    #Contraction 1500; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('bdacikjl->abcdijkl',E26)
    
    #Contraction 1501; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('cbadikjl->abcdijkl',E26)
    
    #Contraction 1502; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('cdabikjl->abcdijkl',E26)
    
    #Contraction 1503; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('dbacikjl->abcdijkl',E26)
    
    #Contraction 1504; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('dcabikjl->abcdijkl',E26)
    
    #Contraction 1505; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('acbdikjl->abcdijkl',E26)
    
    #Contraction 1506; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('adbcikjl->abcdijkl',E26)
    
    #Contraction 1507; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('cabdikjl->abcdijkl',E26)
    
    #Contraction 1508; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('cdbaikjl->abcdijkl',E26)
    
    #Contraction 1509; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('dabcikjl->abcdijkl',E26)
    
    #Contraction 1510; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('dcbaikjl->abcdijkl',E26)
    
    #Contraction 1511; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('abcdikjl->abcdijkl',E26)
    
    #Contraction 1512; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('adcbikjl->abcdijkl',E26)
    
    #Contraction 1513; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('bacdikjl->abcdijkl',E26)
    
    #Contraction 1514; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('bdcaikjl->abcdijkl',E26)
    
    #Contraction 1515; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('dacbikjl->abcdijkl',E26)
    
    #Contraction 1516; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('dbcaikjl->abcdijkl',E26)
    
    #Contraction 1517; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('abdcikjl->abcdijkl',E26)
    
    #Contraction 1518; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('acdbikjl->abcdijkl',E26)
    
    #Contraction 1519; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('badcikjl->abcdijkl',E26)
    
    #Contraction 1520; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('bcdaikjl->abcdijkl',E26)
    
    #Contraction 1521; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('cadbikjl->abcdijkl',E26)
    
    #Contraction 1522; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('cbdaikjl->abcdijkl',E26)
    
    #Contraction 1523; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('bcadijkl->abcdijkl',E26)
    
    #Contraction 1524; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('bdacijkl->abcdijkl',E26)
    
    #Contraction 1525; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('cbadijkl->abcdijkl',E26)
    
    #Contraction 1526; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('cdabijkl->abcdijkl',E26)
    
    #Contraction 1527; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('dbacijkl->abcdijkl',E26)
    
    #Contraction 1528; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('dcabijkl->abcdijkl',E26)
    
    #Contraction 1529; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('acbdijkl->abcdijkl',E26)
    
    #Contraction 1530; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('adbcijkl->abcdijkl',E26)
    
    #Contraction 1531; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('cabdijkl->abcdijkl',E26)
    
    #Contraction 1532; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('cdbaijkl->abcdijkl',E26)
    
    #Contraction 1533; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('dabcijkl->abcdijkl',E26)
    
    #Contraction 1534; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('dcbaijkl->abcdijkl',E26)
    
    #Contraction 1535; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('abcdijkl->abcdijkl',E26)
    
    #Contraction 1536; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('adcbijkl->abcdijkl',E26)
    
    #Contraction 1537; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('bacdijkl->abcdijkl',E26)
    
    #Contraction 1538; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('bdcaijkl->abcdijkl',E26)
    
    #Contraction 1539; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('dacbijkl->abcdijkl',E26)
    
    #Contraction 1540; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('dbcaijkl->abcdijkl',E26)
    
    #Contraction 1541; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('abdcijkl->abcdijkl',E26)
    
    #Contraction 1542; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('acdbijkl->abcdijkl',E26)
    
    #Contraction 1543; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('badcijkl->abcdijkl',E26)
    
    #Contraction 1544; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('bcdaijkl->abcdijkl',E26)
    
    #Contraction 1545; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('cadbijkl->abcdijkl',E26)
    
    #Contraction 1546; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('cbdaijkl->abcdijkl',E26)
    
    #Contraction 1547; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('bcadilkj->abcdijkl',E26)
    
    #Contraction 1548; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('bdacilkj->abcdijkl',E26)
    
    #Contraction 1549; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('cbadilkj->abcdijkl',E26)
    
    #Contraction 1550; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('cdabilkj->abcdijkl',E26)
    
    #Contraction 1551; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('dbacilkj->abcdijkl',E26)
    
    #Contraction 1552; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('dcabilkj->abcdijkl',E26)
    
    #Contraction 1553; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('acbdilkj->abcdijkl',E26)
    
    #Contraction 1554; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('adbcilkj->abcdijkl',E26)
    
    #Contraction 1555; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('cabdilkj->abcdijkl',E26)
    
    #Contraction 1556; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('cdbailkj->abcdijkl',E26)
    
    #Contraction 1557; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('dabcilkj->abcdijkl',E26)
    
    #Contraction 1558; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('dcbailkj->abcdijkl',E26)
    
    #Contraction 1559; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('abcdilkj->abcdijkl',E26)
    
    #Contraction 1560; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('adcbilkj->abcdijkl',E26)
    
    #Contraction 1561; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('bacdilkj->abcdijkl',E26)
    
    #Contraction 1562; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('bdcailkj->abcdijkl',E26)
    
    #Contraction 1563; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('dacbilkj->abcdijkl',E26)
    
    #Contraction 1564; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('dbcailkj->abcdijkl',E26)
    
    #Contraction 1565; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('abdcilkj->abcdijkl',E26)
    
    #Contraction 1566; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('acdbilkj->abcdijkl',E26)
    
    #Contraction 1567; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('badcilkj->abcdijkl',E26)
    
    #Contraction 1568; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('bcdailkj->abcdijkl',E26)
    
    #Contraction 1569; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('cadbilkj->abcdijkl',E26)
    
    #Contraction 1570; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('cbdailkj->abcdijkl',E26)
    
    #Contraction 1571; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('bcadkjil->abcdijkl',E26)
    
    #Contraction 1572; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('bdackjil->abcdijkl',E26)
    
    #Contraction 1573; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('cbadkjil->abcdijkl',E26)
    
    #Contraction 1574; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('cdabkjil->abcdijkl',E26)
    
    #Contraction 1575; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('dbackjil->abcdijkl',E26)
    
    #Contraction 1576; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('dcabkjil->abcdijkl',E26)
    
    #Contraction 1577; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('acbdkjil->abcdijkl',E26)
    
    #Contraction 1578; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('adbckjil->abcdijkl',E26)
    
    #Contraction 1579; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('cabdkjil->abcdijkl',E26)
    
    #Contraction 1580; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('cdbakjil->abcdijkl',E26)
    
    #Contraction 1581; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('dabckjil->abcdijkl',E26)
    
    #Contraction 1582; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('dcbakjil->abcdijkl',E26)
    
    #Contraction 1583; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('abcdkjil->abcdijkl',E26)
    
    #Contraction 1584; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('adcbkjil->abcdijkl',E26)
    
    #Contraction 1585; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('bacdkjil->abcdijkl',E26)
    
    #Contraction 1586; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('bdcakjil->abcdijkl',E26)
    
    #Contraction 1587; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('dacbkjil->abcdijkl',E26)
    
    #Contraction 1588; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('dbcakjil->abcdijkl',E26)
    
    #Contraction 1589; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('abdckjil->abcdijkl',E26)
    
    #Contraction 1590; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('acdbkjil->abcdijkl',E26)
    
    #Contraction 1591; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('badckjil->abcdijkl',E26)
    
    #Contraction 1592; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('bcdakjil->abcdijkl',E26)
    
    #Contraction 1593; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('cadbkjil->abcdijkl',E26)
    
    #Contraction 1594; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('cbdakjil->abcdijkl',E26)
    
    #Contraction 1595; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('bcadklij->abcdijkl',E26)
    
    #Contraction 1596; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('bdacklij->abcdijkl',E26)
    
    #Contraction 1597; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('cbadklij->abcdijkl',E26)
    
    #Contraction 1598; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('cdabklij->abcdijkl',E26)
    
    #Contraction 1599; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('dbacklij->abcdijkl',E26)
    
    #Contraction 1600; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('dcabklij->abcdijkl',E26)
    
    #Contraction 1601; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('acbdklij->abcdijkl',E26)
    
    #Contraction 1602; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('adbcklij->abcdijkl',E26)
    
    #Contraction 1603; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('cabdklij->abcdijkl',E26)
    
    #Contraction 1604; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('cdbaklij->abcdijkl',E26)
    
    #Contraction 1605; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('dabcklij->abcdijkl',E26)
    
    #Contraction 1606; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('dcbaklij->abcdijkl',E26)
    
    #Contraction 1607; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('abcdklij->abcdijkl',E26)
    
    #Contraction 1608; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('adcbklij->abcdijkl',E26)
    
    #Contraction 1609; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('bacdklij->abcdijkl',E26)
    
    #Contraction 1610; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('bdcaklij->abcdijkl',E26)
    
    #Contraction 1611; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('dacbklij->abcdijkl',E26)
    
    #Contraction 1612; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('dbcaklij->abcdijkl',E26)
    
    #Contraction 1613; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('abdcklij->abcdijkl',E26)
    
    #Contraction 1614; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('acdbklij->abcdijkl',E26)
    
    #Contraction 1615; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('badcklij->abcdijkl',E26)
    
    #Contraction 1616; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('bcdaklij->abcdijkl',E26)
    
    #Contraction 1617; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('cadbklij->abcdijkl',E26)
    
    #Contraction 1618; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('cbdaklij->abcdijkl',E26)
    
    #Contraction 1619; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('bcadjlik->abcdijkl',E26)
    
    #Contraction 1620; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('bdacjlik->abcdijkl',E26)
    
    #Contraction 1621; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('cbadjlik->abcdijkl',E26)
    
    #Contraction 1622; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('cdabjlik->abcdijkl',E26)
    
    #Contraction 1623; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('dbacjlik->abcdijkl',E26)
    
    #Contraction 1624; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('dcabjlik->abcdijkl',E26)
    
    #Contraction 1625; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('acbdjlik->abcdijkl',E26)
    
    #Contraction 1626; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('adbcjlik->abcdijkl',E26)
    
    #Contraction 1627; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('cabdjlik->abcdijkl',E26)
    
    #Contraction 1628; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('cdbajlik->abcdijkl',E26)
    
    #Contraction 1629; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('dabcjlik->abcdijkl',E26)
    
    #Contraction 1630; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('dcbajlik->abcdijkl',E26)
    
    #Contraction 1631; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('abcdjlik->abcdijkl',E26)
    
    #Contraction 1632; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('adcbjlik->abcdijkl',E26)
    
    #Contraction 1633; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('bacdjlik->abcdijkl',E26)
    
    #Contraction 1634; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('bdcajlik->abcdijkl',E26)
    
    #Contraction 1635; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('dacbjlik->abcdijkl',E26)
    
    #Contraction 1636; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('dbcajlik->abcdijkl',E26)
    
    #Contraction 1637; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('abdcjlik->abcdijkl',E26)
    
    #Contraction 1638; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('acdbjlik->abcdijkl',E26)
    
    #Contraction 1639; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('badcjlik->abcdijkl',E26)
    
    #Contraction 1640; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('bcdajlik->abcdijkl',E26)
    
    #Contraction 1641; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('cadbjlik->abcdijkl',E26)
    
    #Contraction 1642; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('cbdajlik->abcdijkl',E26)
    
    del E26
    
    G26 = np.zeros([nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 1643; Tree Level  3; Scaling  6/ 2 Result_size  4/ 2
    G26 += oe.contract('bdnl,mnij->bdmlij',T2,I5, optimize='optimal')
    
    del I5
    
    I26 = np.zeros([nvir, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 1644; Tree Level  2; Scaling  6/ 4 Result_size  4/ 4
    I26 += oe.contract('acmk,bdmlij->acbdklij',T2,G26, optimize='optimal')
    
    del G26
    
    #Contraction 1645; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('acbdklij->abcdijkl',I26)
    
    #Contraction 1646; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('abcdklij->abcdijkl',I26)
    
    #Contraction 1647; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('adcbklij->abcdijkl',I26)
    
    #Contraction 1648; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('cbadklij->abcdijkl',I26)
    
    #Contraction 1649; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('cdabklij->abcdijkl',I26)
    
    #Contraction 1650; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('bdacklij->abcdijkl',I26)
    
    #Contraction 1651; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('acbdlkij->abcdijkl',I26)
    
    #Contraction 1652; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('abcdlkij->abcdijkl',I26)
    
    #Contraction 1653; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('adcblkij->abcdijkl',I26)
    
    #Contraction 1654; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('cbadlkij->abcdijkl',I26)
    
    #Contraction 1655; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('cdablkij->abcdijkl',I26)
    
    #Contraction 1656; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('bdaclkij->abcdijkl',I26)
    
    #Contraction 1657; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('acbdjlik->abcdijkl',I26)
    
    #Contraction 1658; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('abcdjlik->abcdijkl',I26)
    
    #Contraction 1659; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('adcbjlik->abcdijkl',I26)
    
    #Contraction 1660; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('cbadjlik->abcdijkl',I26)
    
    #Contraction 1661; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('cdabjlik->abcdijkl',I26)
    
    #Contraction 1662; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('bdacjlik->abcdijkl',I26)
    
    #Contraction 1663; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('acbdljik->abcdijkl',I26)
    
    #Contraction 1664; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('abcdljik->abcdijkl',I26)
    
    #Contraction 1665; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('adcbljik->abcdijkl',I26)
    
    #Contraction 1666; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('cbadljik->abcdijkl',I26)
    
    #Contraction 1667; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('cdabljik->abcdijkl',I26)
    
    #Contraction 1668; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('bdacljik->abcdijkl',I26)
    
    #Contraction 1669; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('acbdjkil->abcdijkl',I26)
    
    #Contraction 1670; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('abcdjkil->abcdijkl',I26)
    
    #Contraction 1671; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('adcbjkil->abcdijkl',I26)
    
    #Contraction 1672; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('cbadjkil->abcdijkl',I26)
    
    #Contraction 1673; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('cdabjkil->abcdijkl',I26)
    
    #Contraction 1674; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('bdacjkil->abcdijkl',I26)
    
    #Contraction 1675; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('acbdkjil->abcdijkl',I26)
    
    #Contraction 1676; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('abcdkjil->abcdijkl',I26)
    
    #Contraction 1677; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('adcbkjil->abcdijkl',I26)
    
    #Contraction 1678; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('cbadkjil->abcdijkl',I26)
    
    #Contraction 1679; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('cdabkjil->abcdijkl',I26)
    
    #Contraction 1680; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('bdackjil->abcdijkl',I26)
    
    #Contraction 1681; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('acbdiljk->abcdijkl',I26)
    
    #Contraction 1682; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('abcdiljk->abcdijkl',I26)
    
    #Contraction 1683; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('adcbiljk->abcdijkl',I26)
    
    #Contraction 1684; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('cbadiljk->abcdijkl',I26)
    
    #Contraction 1685; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('cdabiljk->abcdijkl',I26)
    
    #Contraction 1686; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('bdaciljk->abcdijkl',I26)
    
    #Contraction 1687; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('acbdlijk->abcdijkl',I26)
    
    #Contraction 1688; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('abcdlijk->abcdijkl',I26)
    
    #Contraction 1689; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('adcblijk->abcdijkl',I26)
    
    #Contraction 1690; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('cbadlijk->abcdijkl',I26)
    
    #Contraction 1691; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('cdablijk->abcdijkl',I26)
    
    #Contraction 1692; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('bdaclijk->abcdijkl',I26)
    
    #Contraction 1693; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('acbdikjl->abcdijkl',I26)
    
    #Contraction 1694; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('abcdikjl->abcdijkl',I26)
    
    #Contraction 1695; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('adcbikjl->abcdijkl',I26)
    
    #Contraction 1696; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('cbadikjl->abcdijkl',I26)
    
    #Contraction 1697; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('cdabikjl->abcdijkl',I26)
    
    #Contraction 1698; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('bdacikjl->abcdijkl',I26)
    
    #Contraction 1699; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('acbdkijl->abcdijkl',I26)
    
    #Contraction 1700; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('abcdkijl->abcdijkl',I26)
    
    #Contraction 1701; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('adcbkijl->abcdijkl',I26)
    
    #Contraction 1702; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('cbadkijl->abcdijkl',I26)
    
    #Contraction 1703; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('cdabkijl->abcdijkl',I26)
    
    #Contraction 1704; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('bdackijl->abcdijkl',I26)
    
    #Contraction 1705; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('acbdijkl->abcdijkl',I26)
    
    #Contraction 1706; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('abcdijkl->abcdijkl',I26)
    
    #Contraction 1707; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('adcbijkl->abcdijkl',I26)
    
    #Contraction 1708; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('cbadijkl->abcdijkl',I26)
    
    #Contraction 1709; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('cdabijkl->abcdijkl',I26)
    
    #Contraction 1710; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('bdacijkl->abcdijkl',I26)
    
    #Contraction 1711; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('acbdjikl->abcdijkl',I26)
    
    #Contraction 1712; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('abcdjikl->abcdijkl',I26)
    
    #Contraction 1713; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('adcbjikl->abcdijkl',I26)
    
    #Contraction 1714; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('cbadjikl->abcdijkl',I26)
    
    #Contraction 1715; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('cdabjikl->abcdijkl',I26)
    
    #Contraction 1716; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('bdacjikl->abcdijkl',I26)
    
    del I26
    
    J26 = np.zeros([nocc, nvir, nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 1717; Tree Level  3; Scaling  6/ 2 Result_size  4/ 2
    J26 += oe.contract('mnij,bdnl->mbdijl',V1,T2, optimize='optimal')
    
    M26 = np.zeros([nvir, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 1718; Tree Level  2; Scaling  6/ 4 Result_size  4/ 4
    M26 += oe.contract('cakm,mbdijl->cabdkijl',T2,J26, optimize='optimal')
    
    del J26
    
    #Contraction 1719; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('cabdkijl->abcdijkl',M26)
    
    #Contraction 1720; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('cbadkijl->abcdijkl',M26)
    
    #Contraction 1721; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('cdabkijl->abcdijkl',M26)
    
    #Contraction 1722; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('abcdkijl->abcdijkl',M26)
    
    #Contraction 1723; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('adcbkijl->abcdijkl',M26)
    
    #Contraction 1724; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('bdcakijl->abcdijkl',M26)
    
    #Contraction 1725; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('cabdlijk->abcdijkl',M26)
    
    #Contraction 1726; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('cbadlijk->abcdijkl',M26)
    
    #Contraction 1727; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('cdablijk->abcdijkl',M26)
    
    #Contraction 1728; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('abcdlijk->abcdijkl',M26)
    
    #Contraction 1729; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('adcblijk->abcdijkl',M26)
    
    #Contraction 1730; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('bdcalijk->abcdijkl',M26)
    
    #Contraction 1731; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('cabdjikl->abcdijkl',M26)
    
    #Contraction 1732; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('cbadjikl->abcdijkl',M26)
    
    #Contraction 1733; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('cdabjikl->abcdijkl',M26)
    
    #Contraction 1734; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('abcdjikl->abcdijkl',M26)
    
    #Contraction 1735; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('adcbjikl->abcdijkl',M26)
    
    #Contraction 1736; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('bdcajikl->abcdijkl',M26)
    
    #Contraction 1737; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('cabdlikj->abcdijkl',M26)
    
    #Contraction 1738; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('cbadlikj->abcdijkl',M26)
    
    #Contraction 1739; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('cdablikj->abcdijkl',M26)
    
    #Contraction 1740; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('abcdlikj->abcdijkl',M26)
    
    #Contraction 1741; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('adcblikj->abcdijkl',M26)
    
    #Contraction 1742; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('bdcalikj->abcdijkl',M26)
    
    #Contraction 1743; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('cabdjilk->abcdijkl',M26)
    
    #Contraction 1744; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('cbadjilk->abcdijkl',M26)
    
    #Contraction 1745; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('cdabjilk->abcdijkl',M26)
    
    #Contraction 1746; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('abcdjilk->abcdijkl',M26)
    
    #Contraction 1747; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('adcbjilk->abcdijkl',M26)
    
    #Contraction 1748; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('bdcajilk->abcdijkl',M26)
    
    #Contraction 1749; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('cabdkilj->abcdijkl',M26)
    
    #Contraction 1750; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('cbadkilj->abcdijkl',M26)
    
    #Contraction 1751; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('cdabkilj->abcdijkl',M26)
    
    #Contraction 1752; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('abcdkilj->abcdijkl',M26)
    
    #Contraction 1753; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('adcbkilj->abcdijkl',M26)
    
    #Contraction 1754; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('bdcakilj->abcdijkl',M26)
    
    #Contraction 1755; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('cabdijkl->abcdijkl',M26)
    
    #Contraction 1756; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('cbadijkl->abcdijkl',M26)
    
    #Contraction 1757; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('cdabijkl->abcdijkl',M26)
    
    #Contraction 1758; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('abcdijkl->abcdijkl',M26)
    
    #Contraction 1759; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('adcbijkl->abcdijkl',M26)
    
    #Contraction 1760; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('bdcaijkl->abcdijkl',M26)
    
    #Contraction 1761; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('cabdljki->abcdijkl',M26)
    
    #Contraction 1762; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('cbadljki->abcdijkl',M26)
    
    #Contraction 1763; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('cdabljki->abcdijkl',M26)
    
    #Contraction 1764; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('abcdljki->abcdijkl',M26)
    
    #Contraction 1765; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('adcbljki->abcdijkl',M26)
    
    #Contraction 1766; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('bdcaljki->abcdijkl',M26)
    
    #Contraction 1767; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('cabdijlk->abcdijkl',M26)
    
    #Contraction 1768; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('cbadijlk->abcdijkl',M26)
    
    #Contraction 1769; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('cdabijlk->abcdijkl',M26)
    
    #Contraction 1770; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('abcdijlk->abcdijkl',M26)
    
    #Contraction 1771; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('adcbijlk->abcdijkl',M26)
    
    #Contraction 1772; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('bdcaijlk->abcdijkl',M26)
    
    #Contraction 1773; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('cabdkjli->abcdijkl',M26)
    
    #Contraction 1774; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('cbadkjli->abcdijkl',M26)
    
    #Contraction 1775; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('cdabkjli->abcdijkl',M26)
    
    #Contraction 1776; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('abcdkjli->abcdijkl',M26)
    
    #Contraction 1777; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('adcbkjli->abcdijkl',M26)
    
    #Contraction 1778; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('bdcakjli->abcdijkl',M26)
    
    #Contraction 1779; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('cabdiklj->abcdijkl',M26)
    
    #Contraction 1780; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('cbadiklj->abcdijkl',M26)
    
    #Contraction 1781; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('cdabiklj->abcdijkl',M26)
    
    #Contraction 1782; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('abcdiklj->abcdijkl',M26)
    
    #Contraction 1783; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('adcbiklj->abcdijkl',M26)
    
    #Contraction 1784; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('bdcaiklj->abcdijkl',M26)
    
    #Contraction 1785; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('cabdjkli->abcdijkl',M26)
    
    #Contraction 1786; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('cbadjkli->abcdijkl',M26)
    
    #Contraction 1787; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('cdabjkli->abcdijkl',M26)
    
    #Contraction 1788; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('abcdjkli->abcdijkl',M26)
    
    #Contraction 1789; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('adcbjkli->abcdijkl',M26)
    
    #Contraction 1790; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('bdcajkli->abcdijkl',M26)
    
    del M26
    
    X26 = np.zeros([nvir, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 1791; Tree Level  2; Scaling  6/ 4 Result_size  4/ 4
    X26 += oe.contract('bm,amcdijkl->bacdijkl',T1,G25, optimize='optimal')
    
    del G25
    
    #Contraction 1792; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bacdijkl->abcdijkl',X26)
    
    #Contraction 1793; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('cabdijkl->abcdijkl',X26)
    
    #Contraction 1794; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('dabcijkl->abcdijkl',X26)
    
    #Contraction 1795; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('abcdijkl->abcdijkl',X26)
    
    #Contraction 1796; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('cbadijkl->abcdijkl',X26)
    
    #Contraction 1797; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('dbacijkl->abcdijkl',X26)
    
    #Contraction 1798; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('acbdijkl->abcdijkl',X26)
    
    #Contraction 1799; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bcadijkl->abcdijkl',X26)
    
    #Contraction 1800; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('dcabijkl->abcdijkl',X26)
    
    #Contraction 1801; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('adbcijkl->abcdijkl',X26)
    
    #Contraction 1802; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bdacijkl->abcdijkl',X26)
    
    #Contraction 1803; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('cdabijkl->abcdijkl',X26)
    
    #Contraction 1804; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bacdjikl->abcdijkl',X26)
    
    #Contraction 1805; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('cabdjikl->abcdijkl',X26)
    
    #Contraction 1806; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('dabcjikl->abcdijkl',X26)
    
    #Contraction 1807; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('abcdjikl->abcdijkl',X26)
    
    #Contraction 1808; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('cbadjikl->abcdijkl',X26)
    
    #Contraction 1809; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('dbacjikl->abcdijkl',X26)
    
    #Contraction 1810; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('acbdjikl->abcdijkl',X26)
    
    #Contraction 1811; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bcadjikl->abcdijkl',X26)
    
    #Contraction 1812; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('dcabjikl->abcdijkl',X26)
    
    #Contraction 1813; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('adbcjikl->abcdijkl',X26)
    
    #Contraction 1814; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bdacjikl->abcdijkl',X26)
    
    #Contraction 1815; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('cdabjikl->abcdijkl',X26)
    
    #Contraction 1816; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bacdkijl->abcdijkl',X26)
    
    #Contraction 1817; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('cabdkijl->abcdijkl',X26)
    
    #Contraction 1818; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('dabckijl->abcdijkl',X26)
    
    #Contraction 1819; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('abcdkijl->abcdijkl',X26)
    
    #Contraction 1820; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('cbadkijl->abcdijkl',X26)
    
    #Contraction 1821; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('dbackijl->abcdijkl',X26)
    
    #Contraction 1822; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('acbdkijl->abcdijkl',X26)
    
    #Contraction 1823; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bcadkijl->abcdijkl',X26)
    
    #Contraction 1824; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('dcabkijl->abcdijkl',X26)
    
    #Contraction 1825; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('adbckijl->abcdijkl',X26)
    
    #Contraction 1826; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bdackijl->abcdijkl',X26)
    
    #Contraction 1827; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('cdabkijl->abcdijkl',X26)
    
    #Contraction 1828; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bacdlijk->abcdijkl',X26)
    
    #Contraction 1829; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('cabdlijk->abcdijkl',X26)
    
    #Contraction 1830; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('dabclijk->abcdijkl',X26)
    
    #Contraction 1831; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('abcdlijk->abcdijkl',X26)
    
    #Contraction 1832; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('cbadlijk->abcdijkl',X26)
    
    #Contraction 1833; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('dbaclijk->abcdijkl',X26)
    
    #Contraction 1834; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('acbdlijk->abcdijkl',X26)
    
    #Contraction 1835; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bcadlijk->abcdijkl',X26)
    
    #Contraction 1836; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('dcablijk->abcdijkl',X26)
    
    #Contraction 1837; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('adbclijk->abcdijkl',X26)
    
    #Contraction 1838; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bdaclijk->abcdijkl',X26)
    
    #Contraction 1839; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('cdablijk->abcdijkl',X26)
    
    del X26
    
    Y26 = np.zeros([nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 1840; Tree Level  3; Scaling  6/ 2 Result_size  4/ 2
    Y26 += oe.contract('am,cmnjki->acnjki',T1,X18, optimize='optimal')
    
    del X18
    
    A27 = np.zeros([nvir, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 1841; Tree Level  2; Scaling  6/ 4 Result_size  4/ 4
    A27 += oe.contract('bdnl,acnjki->bdacljki',T2,Y26, optimize='optimal')
    
    del Y26
    
    #Contraction 1842; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bdacljki->abcdijkl',A27)
    
    #Contraction 1843; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('cdabljki->abcdijkl',A27)
    
    #Contraction 1844; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('cbadljki->abcdijkl',A27)
    
    #Contraction 1845; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bdcaljki->abcdijkl',A27)
    
    #Contraction 1846; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('adcbljki->abcdijkl',A27)
    
    #Contraction 1847; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('abcdljki->abcdijkl',A27)
    
    #Contraction 1848; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('cdbaljki->abcdijkl',A27)
    
    #Contraction 1849; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('adbcljki->abcdijkl',A27)
    
    #Contraction 1850; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('acbdljki->abcdijkl',A27)
    
    #Contraction 1851; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('cbdaljki->abcdijkl',A27)
    
    #Contraction 1852; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('abdcljki->abcdijkl',A27)
    
    #Contraction 1853; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('acdbljki->abcdijkl',A27)
    
    #Contraction 1854; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bdackjli->abcdijkl',A27)
    
    #Contraction 1855; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('cdabkjli->abcdijkl',A27)
    
    #Contraction 1856; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('cbadkjli->abcdijkl',A27)
    
    #Contraction 1857; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bdcakjli->abcdijkl',A27)
    
    #Contraction 1858; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('adcbkjli->abcdijkl',A27)
    
    #Contraction 1859; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('abcdkjli->abcdijkl',A27)
    
    #Contraction 1860; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('cdbakjli->abcdijkl',A27)
    
    #Contraction 1861; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('adbckjli->abcdijkl',A27)
    
    #Contraction 1862; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('acbdkjli->abcdijkl',A27)
    
    #Contraction 1863; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('cbdakjli->abcdijkl',A27)
    
    #Contraction 1864; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('abdckjli->abcdijkl',A27)
    
    #Contraction 1865; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('acdbkjli->abcdijkl',A27)
    
    #Contraction 1866; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bdacjkli->abcdijkl',A27)
    
    #Contraction 1867; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('cdabjkli->abcdijkl',A27)
    
    #Contraction 1868; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('cbadjkli->abcdijkl',A27)
    
    #Contraction 1869; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bdcajkli->abcdijkl',A27)
    
    #Contraction 1870; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('adcbjkli->abcdijkl',A27)
    
    #Contraction 1871; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('abcdjkli->abcdijkl',A27)
    
    #Contraction 1872; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('cdbajkli->abcdijkl',A27)
    
    #Contraction 1873; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('adbcjkli->abcdijkl',A27)
    
    #Contraction 1874; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('acbdjkli->abcdijkl',A27)
    
    #Contraction 1875; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('cbdajkli->abcdijkl',A27)
    
    #Contraction 1876; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('abdcjkli->abcdijkl',A27)
    
    #Contraction 1877; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('acdbjkli->abcdijkl',A27)
    
    #Contraction 1878; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bdaclikj->abcdijkl',A27)
    
    #Contraction 1879; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('cdablikj->abcdijkl',A27)
    
    #Contraction 1880; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('cbadlikj->abcdijkl',A27)
    
    #Contraction 1881; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bdcalikj->abcdijkl',A27)
    
    #Contraction 1882; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('adcblikj->abcdijkl',A27)
    
    #Contraction 1883; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('abcdlikj->abcdijkl',A27)
    
    #Contraction 1884; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('cdbalikj->abcdijkl',A27)
    
    #Contraction 1885; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('adbclikj->abcdijkl',A27)
    
    #Contraction 1886; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('acbdlikj->abcdijkl',A27)
    
    #Contraction 1887; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('cbdalikj->abcdijkl',A27)
    
    #Contraction 1888; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('abdclikj->abcdijkl',A27)
    
    #Contraction 1889; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('acdblikj->abcdijkl',A27)
    
    #Contraction 1890; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bdackilj->abcdijkl',A27)
    
    #Contraction 1891; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('cdabkilj->abcdijkl',A27)
    
    #Contraction 1892; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('cbadkilj->abcdijkl',A27)
    
    #Contraction 1893; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bdcakilj->abcdijkl',A27)
    
    #Contraction 1894; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('adcbkilj->abcdijkl',A27)
    
    #Contraction 1895; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('abcdkilj->abcdijkl',A27)
    
    #Contraction 1896; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('cdbakilj->abcdijkl',A27)
    
    #Contraction 1897; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('adbckilj->abcdijkl',A27)
    
    #Contraction 1898; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('acbdkilj->abcdijkl',A27)
    
    #Contraction 1899; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('cbdakilj->abcdijkl',A27)
    
    #Contraction 1900; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('abdckilj->abcdijkl',A27)
    
    #Contraction 1901; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('acdbkilj->abcdijkl',A27)
    
    #Contraction 1902; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bdaciklj->abcdijkl',A27)
    
    #Contraction 1903; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('cdabiklj->abcdijkl',A27)
    
    #Contraction 1904; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('cbadiklj->abcdijkl',A27)
    
    #Contraction 1905; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bdcaiklj->abcdijkl',A27)
    
    #Contraction 1906; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('adcbiklj->abcdijkl',A27)
    
    #Contraction 1907; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('abcdiklj->abcdijkl',A27)
    
    #Contraction 1908; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('cdbaiklj->abcdijkl',A27)
    
    #Contraction 1909; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('adbciklj->abcdijkl',A27)
    
    #Contraction 1910; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('acbdiklj->abcdijkl',A27)
    
    #Contraction 1911; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('cbdaiklj->abcdijkl',A27)
    
    #Contraction 1912; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('abdciklj->abcdijkl',A27)
    
    #Contraction 1913; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('acdbiklj->abcdijkl',A27)
    
    #Contraction 1914; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bdaclijk->abcdijkl',A27)
    
    #Contraction 1915; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('cdablijk->abcdijkl',A27)
    
    #Contraction 1916; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('cbadlijk->abcdijkl',A27)
    
    #Contraction 1917; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bdcalijk->abcdijkl',A27)
    
    #Contraction 1918; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('adcblijk->abcdijkl',A27)
    
    #Contraction 1919; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('abcdlijk->abcdijkl',A27)
    
    #Contraction 1920; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('cdbalijk->abcdijkl',A27)
    
    #Contraction 1921; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('adbclijk->abcdijkl',A27)
    
    #Contraction 1922; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('acbdlijk->abcdijkl',A27)
    
    #Contraction 1923; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('cbdalijk->abcdijkl',A27)
    
    #Contraction 1924; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('abdclijk->abcdijkl',A27)
    
    #Contraction 1925; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('acdblijk->abcdijkl',A27)
    
    #Contraction 1926; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bdacjilk->abcdijkl',A27)
    
    #Contraction 1927; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('cdabjilk->abcdijkl',A27)
    
    #Contraction 1928; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('cbadjilk->abcdijkl',A27)
    
    #Contraction 1929; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bdcajilk->abcdijkl',A27)
    
    #Contraction 1930; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('adcbjilk->abcdijkl',A27)
    
    #Contraction 1931; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('abcdjilk->abcdijkl',A27)
    
    #Contraction 1932; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('cdbajilk->abcdijkl',A27)
    
    #Contraction 1933; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('adbcjilk->abcdijkl',A27)
    
    #Contraction 1934; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('acbdjilk->abcdijkl',A27)
    
    #Contraction 1935; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('cbdajilk->abcdijkl',A27)
    
    #Contraction 1936; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('abdcjilk->abcdijkl',A27)
    
    #Contraction 1937; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('acdbjilk->abcdijkl',A27)
    
    #Contraction 1938; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bdacijlk->abcdijkl',A27)
    
    #Contraction 1939; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('cdabijlk->abcdijkl',A27)
    
    #Contraction 1940; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('cbadijlk->abcdijkl',A27)
    
    #Contraction 1941; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bdcaijlk->abcdijkl',A27)
    
    #Contraction 1942; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('adcbijlk->abcdijkl',A27)
    
    #Contraction 1943; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('abcdijlk->abcdijkl',A27)
    
    #Contraction 1944; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('cdbaijlk->abcdijkl',A27)
    
    #Contraction 1945; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('adbcijlk->abcdijkl',A27)
    
    #Contraction 1946; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('acbdijlk->abcdijkl',A27)
    
    #Contraction 1947; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('cbdaijlk->abcdijkl',A27)
    
    #Contraction 1948; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('abdcijlk->abcdijkl',A27)
    
    #Contraction 1949; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('acdbijlk->abcdijkl',A27)
    
    #Contraction 1950; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bdackijl->abcdijkl',A27)
    
    #Contraction 1951; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('cdabkijl->abcdijkl',A27)
    
    #Contraction 1952; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('cbadkijl->abcdijkl',A27)
    
    #Contraction 1953; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bdcakijl->abcdijkl',A27)
    
    #Contraction 1954; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('adcbkijl->abcdijkl',A27)
    
    #Contraction 1955; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('abcdkijl->abcdijkl',A27)
    
    #Contraction 1956; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('cdbakijl->abcdijkl',A27)
    
    #Contraction 1957; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('adbckijl->abcdijkl',A27)
    
    #Contraction 1958; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('acbdkijl->abcdijkl',A27)
    
    #Contraction 1959; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('cbdakijl->abcdijkl',A27)
    
    #Contraction 1960; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('abdckijl->abcdijkl',A27)
    
    #Contraction 1961; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('acdbkijl->abcdijkl',A27)
    
    #Contraction 1962; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bdacjikl->abcdijkl',A27)
    
    #Contraction 1963; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('cdabjikl->abcdijkl',A27)
    
    #Contraction 1964; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('cbadjikl->abcdijkl',A27)
    
    #Contraction 1965; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bdcajikl->abcdijkl',A27)
    
    #Contraction 1966; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('adcbjikl->abcdijkl',A27)
    
    #Contraction 1967; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('abcdjikl->abcdijkl',A27)
    
    #Contraction 1968; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('cdbajikl->abcdijkl',A27)
    
    #Contraction 1969; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('adbcjikl->abcdijkl',A27)
    
    #Contraction 1970; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('acbdjikl->abcdijkl',A27)
    
    #Contraction 1971; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('cbdajikl->abcdijkl',A27)
    
    #Contraction 1972; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('abdcjikl->abcdijkl',A27)
    
    #Contraction 1973; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('acdbjikl->abcdijkl',A27)
    
    #Contraction 1974; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bdacijkl->abcdijkl',A27)
    
    #Contraction 1975; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('cdabijkl->abcdijkl',A27)
    
    #Contraction 1976; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('cbadijkl->abcdijkl',A27)
    
    #Contraction 1977; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bdcaijkl->abcdijkl',A27)
    
    #Contraction 1978; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('adcbijkl->abcdijkl',A27)
    
    #Contraction 1979; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('abcdijkl->abcdijkl',A27)
    
    #Contraction 1980; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('cdbaijkl->abcdijkl',A27)
    
    #Contraction 1981; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('adbcijkl->abcdijkl',A27)
    
    #Contraction 1982; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('acbdijkl->abcdijkl',A27)
    
    #Contraction 1983; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('cbdaijkl->abcdijkl',A27)
    
    #Contraction 1984; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('abdcijkl->abcdijkl',A27)
    
    #Contraction 1985; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('acdbijkl->abcdijkl',A27)
    
    del A27
    
    D27 = np.zeros([nvir, nvir, nvir, nocc, nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 1986; Tree Level  3; Scaling  7/ 5 Result_size  5/ 3
    D27 += oe.contract('fbcdnjkl,mnfi->bcdmjkli',T4,I1, optimize='optimal')
    
    E27 = np.zeros([nvir, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 1987; Tree Level  2; Scaling  6/ 4 Result_size  4/ 4
    E27 += oe.contract('am,bcdmjkli->abcdjkli',T1,D27, optimize='optimal')
    
    del D27
    
    #Contraction 1988; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('abcdjkli->abcdijkl',E27)
    
    #Contraction 1989; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bacdjkli->abcdijkl',E27)
    
    #Contraction 1990; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('cabdjkli->abcdijkl',E27)
    
    #Contraction 1991; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('dabcjkli->abcdijkl',E27)
    
    #Contraction 1992; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('abcdiklj->abcdijkl',E27)
    
    #Contraction 1993; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bacdiklj->abcdijkl',E27)
    
    #Contraction 1994; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('cabdiklj->abcdijkl',E27)
    
    #Contraction 1995; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('dabciklj->abcdijkl',E27)
    
    #Contraction 1996; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('abcdijlk->abcdijkl',E27)
    
    #Contraction 1997; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bacdijlk->abcdijkl',E27)
    
    #Contraction 1998; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('cabdijlk->abcdijkl',E27)
    
    #Contraction 1999; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('dabcijlk->abcdijkl',E27)
    
    #Contraction 2000; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('abcdijkl->abcdijkl',E27)
    
    #Contraction 2001; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bacdijkl->abcdijkl',E27)
    
    #Contraction 2002; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('cabdijkl->abcdijkl',E27)
    
    #Contraction 2003; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('dabcijkl->abcdijkl',E27)
    
    del E27
    
    G27 = np.zeros([nvir, nvir, nocc, nocc, nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 2004; Tree Level  4; Scaling  6/ 4 Result_size  6/ 2
    G27 += oe.contract('fcdjkl,mnfi->cdmnjkli',T3,I1, optimize='optimal')
    
    I27 = np.zeros([nvir, nvir, nvir, nocc, nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 2005; Tree Level  3; Scaling  7/ 3 Result_size  5/ 3
    I27 += oe.contract('bn,cdmnjkli->bcdmjkli',T1,G27, optimize='optimal')
    
    del G27
    
    J27 = np.zeros([nvir, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 2006; Tree Level  2; Scaling  6/ 4 Result_size  4/ 4
    J27 += oe.contract('am,bcdmjkli->abcdjkli',T1,I27, optimize='optimal')
    
    del I27
    
    #Contraction 2007; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('abcdjkli->abcdijkl',J27)
    
    #Contraction 2008; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('acbdjkli->abcdijkl',J27)
    
    #Contraction 2009; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('adbcjkli->abcdijkl',J27)
    
    #Contraction 2010; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('bacdjkli->abcdijkl',J27)
    
    #Contraction 2011; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('bcadjkli->abcdijkl',J27)
    
    #Contraction 2012; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('bdacjkli->abcdijkl',J27)
    
    #Contraction 2013; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('cabdjkli->abcdijkl',J27)
    
    #Contraction 2014; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('cbadjkli->abcdijkl',J27)
    
    #Contraction 2015; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('cdabjkli->abcdijkl',J27)
    
    #Contraction 2016; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('dabcjkli->abcdijkl',J27)
    
    #Contraction 2017; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('dbacjkli->abcdijkl',J27)
    
    #Contraction 2018; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('dcabjkli->abcdijkl',J27)
    
    #Contraction 2019; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('abcdiklj->abcdijkl',J27)
    
    #Contraction 2020; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('acbdiklj->abcdijkl',J27)
    
    #Contraction 2021; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('adbciklj->abcdijkl',J27)
    
    #Contraction 2022; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('bacdiklj->abcdijkl',J27)
    
    #Contraction 2023; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('bcadiklj->abcdijkl',J27)
    
    #Contraction 2024; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('bdaciklj->abcdijkl',J27)
    
    #Contraction 2025; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('cabdiklj->abcdijkl',J27)
    
    #Contraction 2026; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('cbadiklj->abcdijkl',J27)
    
    #Contraction 2027; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('cdabiklj->abcdijkl',J27)
    
    #Contraction 2028; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('dabciklj->abcdijkl',J27)
    
    #Contraction 2029; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('dbaciklj->abcdijkl',J27)
    
    #Contraction 2030; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('dcabiklj->abcdijkl',J27)
    
    #Contraction 2031; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('abcdijlk->abcdijkl',J27)
    
    #Contraction 2032; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('acbdijlk->abcdijkl',J27)
    
    #Contraction 2033; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('adbcijlk->abcdijkl',J27)
    
    #Contraction 2034; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('bacdijlk->abcdijkl',J27)
    
    #Contraction 2035; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('bcadijlk->abcdijkl',J27)
    
    #Contraction 2036; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('bdacijlk->abcdijkl',J27)
    
    #Contraction 2037; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('cabdijlk->abcdijkl',J27)
    
    #Contraction 2038; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('cbadijlk->abcdijkl',J27)
    
    #Contraction 2039; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('cdabijlk->abcdijkl',J27)
    
    #Contraction 2040; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('dabcijlk->abcdijkl',J27)
    
    #Contraction 2041; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('dbacijlk->abcdijkl',J27)
    
    #Contraction 2042; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('dcabijlk->abcdijkl',J27)
    
    #Contraction 2043; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('abcdijkl->abcdijkl',J27)
    
    #Contraction 2044; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('acbdijkl->abcdijkl',J27)
    
    #Contraction 2045; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('adbcijkl->abcdijkl',J27)
    
    #Contraction 2046; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('bacdijkl->abcdijkl',J27)
    
    #Contraction 2047; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('bcadijkl->abcdijkl',J27)
    
    #Contraction 2048; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('bdacijkl->abcdijkl',J27)
    
    #Contraction 2049; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('cabdijkl->abcdijkl',J27)
    
    #Contraction 2050; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('cbadijkl->abcdijkl',J27)
    
    #Contraction 2051; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('cdabijkl->abcdijkl',J27)
    
    #Contraction 2052; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('dabcijkl->abcdijkl',J27)
    
    #Contraction 2053; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('dbacijkl->abcdijkl',J27)
    
    #Contraction 2054; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('dcabijkl->abcdijkl',J27)
    
    del J27
    
    M27 = np.zeros([nvir, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 2055; Tree Level  2; Scaling  6/ 4 Result_size  4/ 4
    M27 += oe.contract('acmk,bdmlij->acbdklij',T2,Y23, optimize='optimal')
    
    del Y23
    
    #Contraction 2056; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('acbdklij->abcdijkl',M27)
    
    #Contraction 2057; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('abcdklij->abcdijkl',M27)
    
    #Contraction 2058; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('adcbklij->abcdijkl',M27)
    
    #Contraction 2059; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('cbadklij->abcdijkl',M27)
    
    #Contraction 2060; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('cdabklij->abcdijkl',M27)
    
    #Contraction 2061; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('bdacklij->abcdijkl',M27)
    
    #Contraction 2062; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('acbdlkij->abcdijkl',M27)
    
    #Contraction 2063; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('abcdlkij->abcdijkl',M27)
    
    #Contraction 2064; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('adcblkij->abcdijkl',M27)
    
    #Contraction 2065; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('cbadlkij->abcdijkl',M27)
    
    #Contraction 2066; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('cdablkij->abcdijkl',M27)
    
    #Contraction 2067; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('bdaclkij->abcdijkl',M27)
    
    #Contraction 2068; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('acbdjlik->abcdijkl',M27)
    
    #Contraction 2069; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('abcdjlik->abcdijkl',M27)
    
    #Contraction 2070; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('adcbjlik->abcdijkl',M27)
    
    #Contraction 2071; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('cbadjlik->abcdijkl',M27)
    
    #Contraction 2072; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('cdabjlik->abcdijkl',M27)
    
    #Contraction 2073; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('bdacjlik->abcdijkl',M27)
    
    #Contraction 2074; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('acbdljik->abcdijkl',M27)
    
    #Contraction 2075; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('abcdljik->abcdijkl',M27)
    
    #Contraction 2076; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('adcbljik->abcdijkl',M27)
    
    #Contraction 2077; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('cbadljik->abcdijkl',M27)
    
    #Contraction 2078; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('cdabljik->abcdijkl',M27)
    
    #Contraction 2079; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('bdacljik->abcdijkl',M27)
    
    #Contraction 2080; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('acbdjkil->abcdijkl',M27)
    
    #Contraction 2081; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('abcdjkil->abcdijkl',M27)
    
    #Contraction 2082; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('adcbjkil->abcdijkl',M27)
    
    #Contraction 2083; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('cbadjkil->abcdijkl',M27)
    
    #Contraction 2084; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('cdabjkil->abcdijkl',M27)
    
    #Contraction 2085; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('bdacjkil->abcdijkl',M27)
    
    #Contraction 2086; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('acbdkjil->abcdijkl',M27)
    
    #Contraction 2087; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('abcdkjil->abcdijkl',M27)
    
    #Contraction 2088; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('adcbkjil->abcdijkl',M27)
    
    #Contraction 2089; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('cbadkjil->abcdijkl',M27)
    
    #Contraction 2090; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('cdabkjil->abcdijkl',M27)
    
    #Contraction 2091; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('bdackjil->abcdijkl',M27)
    
    #Contraction 2092; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('acbdklji->abcdijkl',M27)
    
    #Contraction 2093; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('abcdklji->abcdijkl',M27)
    
    #Contraction 2094; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('adcbklji->abcdijkl',M27)
    
    #Contraction 2095; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('cbadklji->abcdijkl',M27)
    
    #Contraction 2096; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('cdabklji->abcdijkl',M27)
    
    #Contraction 2097; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('bdacklji->abcdijkl',M27)
    
    #Contraction 2098; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('acbdlkji->abcdijkl',M27)
    
    #Contraction 2099; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('abcdlkji->abcdijkl',M27)
    
    #Contraction 2100; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('adcblkji->abcdijkl',M27)
    
    #Contraction 2101; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('cbadlkji->abcdijkl',M27)
    
    #Contraction 2102; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('cdablkji->abcdijkl',M27)
    
    #Contraction 2103; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('bdaclkji->abcdijkl',M27)
    
    #Contraction 2104; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('acbdiljk->abcdijkl',M27)
    
    #Contraction 2105; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('abcdiljk->abcdijkl',M27)
    
    #Contraction 2106; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('adcbiljk->abcdijkl',M27)
    
    #Contraction 2107; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('cbadiljk->abcdijkl',M27)
    
    #Contraction 2108; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('cdabiljk->abcdijkl',M27)
    
    #Contraction 2109; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('bdaciljk->abcdijkl',M27)
    
    #Contraction 2110; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('acbdlijk->abcdijkl',M27)
    
    #Contraction 2111; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('abcdlijk->abcdijkl',M27)
    
    #Contraction 2112; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('adcblijk->abcdijkl',M27)
    
    #Contraction 2113; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('cbadlijk->abcdijkl',M27)
    
    #Contraction 2114; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('cdablijk->abcdijkl',M27)
    
    #Contraction 2115; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('bdaclijk->abcdijkl',M27)
    
    #Contraction 2116; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('acbdikjl->abcdijkl',M27)
    
    #Contraction 2117; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('abcdikjl->abcdijkl',M27)
    
    #Contraction 2118; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('adcbikjl->abcdijkl',M27)
    
    #Contraction 2119; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('cbadikjl->abcdijkl',M27)
    
    #Contraction 2120; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('cdabikjl->abcdijkl',M27)
    
    #Contraction 2121; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('bdacikjl->abcdijkl',M27)
    
    #Contraction 2122; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('acbdkijl->abcdijkl',M27)
    
    #Contraction 2123; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('abcdkijl->abcdijkl',M27)
    
    #Contraction 2124; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('adcbkijl->abcdijkl',M27)
    
    #Contraction 2125; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('cbadkijl->abcdijkl',M27)
    
    #Contraction 2126; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('cdabkijl->abcdijkl',M27)
    
    #Contraction 2127; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('bdackijl->abcdijkl',M27)
    
    #Contraction 2128; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('acbdjlki->abcdijkl',M27)
    
    #Contraction 2129; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('abcdjlki->abcdijkl',M27)
    
    #Contraction 2130; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('adcbjlki->abcdijkl',M27)
    
    #Contraction 2131; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('cbadjlki->abcdijkl',M27)
    
    #Contraction 2132; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('cdabjlki->abcdijkl',M27)
    
    #Contraction 2133; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('bdacjlki->abcdijkl',M27)
    
    #Contraction 2134; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('acbdljki->abcdijkl',M27)
    
    #Contraction 2135; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('abcdljki->abcdijkl',M27)
    
    #Contraction 2136; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('adcbljki->abcdijkl',M27)
    
    #Contraction 2137; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('cbadljki->abcdijkl',M27)
    
    #Contraction 2138; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('cdabljki->abcdijkl',M27)
    
    #Contraction 2139; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('bdacljki->abcdijkl',M27)
    
    #Contraction 2140; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('acbdilkj->abcdijkl',M27)
    
    #Contraction 2141; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('abcdilkj->abcdijkl',M27)
    
    #Contraction 2142; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('adcbilkj->abcdijkl',M27)
    
    #Contraction 2143; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('cbadilkj->abcdijkl',M27)
    
    #Contraction 2144; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('cdabilkj->abcdijkl',M27)
    
    #Contraction 2145; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('bdacilkj->abcdijkl',M27)
    
    #Contraction 2146; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('acbdlikj->abcdijkl',M27)
    
    #Contraction 2147; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('abcdlikj->abcdijkl',M27)
    
    #Contraction 2148; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('adcblikj->abcdijkl',M27)
    
    #Contraction 2149; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('cbadlikj->abcdijkl',M27)
    
    #Contraction 2150; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('cdablikj->abcdijkl',M27)
    
    #Contraction 2151; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('bdaclikj->abcdijkl',M27)
    
    #Contraction 2152; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('acbdijkl->abcdijkl',M27)
    
    #Contraction 2153; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('abcdijkl->abcdijkl',M27)
    
    #Contraction 2154; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('adcbijkl->abcdijkl',M27)
    
    #Contraction 2155; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('cbadijkl->abcdijkl',M27)
    
    #Contraction 2156; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('cdabijkl->abcdijkl',M27)
    
    #Contraction 2157; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('bdacijkl->abcdijkl',M27)
    
    #Contraction 2158; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('acbdjikl->abcdijkl',M27)
    
    #Contraction 2159; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('abcdjikl->abcdijkl',M27)
    
    #Contraction 2160; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('adcbjikl->abcdijkl',M27)
    
    #Contraction 2161; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('cbadjikl->abcdijkl',M27)
    
    #Contraction 2162; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('cdabjikl->abcdijkl',M27)
    
    #Contraction 2163; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('bdacjikl->abcdijkl',M27)
    
    #Contraction 2164; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('acbdjkli->abcdijkl',M27)
    
    #Contraction 2165; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('abcdjkli->abcdijkl',M27)
    
    #Contraction 2166; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('adcbjkli->abcdijkl',M27)
    
    #Contraction 2167; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('cbadjkli->abcdijkl',M27)
    
    #Contraction 2168; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('cdabjkli->abcdijkl',M27)
    
    #Contraction 2169; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('bdacjkli->abcdijkl',M27)
    
    #Contraction 2170; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('acbdkjli->abcdijkl',M27)
    
    #Contraction 2171; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('abcdkjli->abcdijkl',M27)
    
    #Contraction 2172; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('adcbkjli->abcdijkl',M27)
    
    #Contraction 2173; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('cbadkjli->abcdijkl',M27)
    
    #Contraction 2174; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('cdabkjli->abcdijkl',M27)
    
    #Contraction 2175; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('bdackjli->abcdijkl',M27)
    
    #Contraction 2176; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('acbdiklj->abcdijkl',M27)
    
    #Contraction 2177; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('abcdiklj->abcdijkl',M27)
    
    #Contraction 2178; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('adcbiklj->abcdijkl',M27)
    
    #Contraction 2179; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('cbadiklj->abcdijkl',M27)
    
    #Contraction 2180; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('cdabiklj->abcdijkl',M27)
    
    #Contraction 2181; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('bdaciklj->abcdijkl',M27)
    
    #Contraction 2182; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('acbdkilj->abcdijkl',M27)
    
    #Contraction 2183; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('abcdkilj->abcdijkl',M27)
    
    #Contraction 2184; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('adcbkilj->abcdijkl',M27)
    
    #Contraction 2185; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('cbadkilj->abcdijkl',M27)
    
    #Contraction 2186; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('cdabkilj->abcdijkl',M27)
    
    #Contraction 2187; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('bdackilj->abcdijkl',M27)
    
    #Contraction 2188; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('acbdijlk->abcdijkl',M27)
    
    #Contraction 2189; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('abcdijlk->abcdijkl',M27)
    
    #Contraction 2190; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('adcbijlk->abcdijkl',M27)
    
    #Contraction 2191; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('cbadijlk->abcdijkl',M27)
    
    #Contraction 2192; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('cdabijlk->abcdijkl',M27)
    
    #Contraction 2193; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('bdacijlk->abcdijkl',M27)
    
    #Contraction 2194; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('acbdjilk->abcdijkl',M27)
    
    #Contraction 2195; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('abcdjilk->abcdijkl',M27)
    
    #Contraction 2196; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('adcbjilk->abcdijkl',M27)
    
    #Contraction 2197; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('cbadjilk->abcdijkl',M27)
    
    #Contraction 2198; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('cdabjilk->abcdijkl',M27)
    
    #Contraction 2199; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('bdacjilk->abcdijkl',M27)
    
    del M27
    
    X27 = np.zeros([nvir, nocc, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 2200; Tree Level  3; Scaling  5/ 7 Result_size  5/ 3
    X27 += oe.contract('amef,efcdijkl->amcdijkl',V8,T4, optimize='optimal')
    
    Y27 = np.zeros([nvir, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 2201; Tree Level  2; Scaling  6/ 4 Result_size  4/ 4
    Y27 += oe.contract('abcdnjkl,ni->abcdjkli',T4,A5, optimize='optimal')
    
    del A5
    
    #Contraction 2202; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('abcdjkli->abcdijkl',Y27)
    
    #Contraction 2203; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('abcdiklj->abcdijkl',Y27)
    
    #Contraction 2204; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('abcdijlk->abcdijkl',Y27)
    
    #Contraction 2205; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('abcdijkl->abcdijkl',Y27)
    
    del Y27
    
    A28 = np.zeros([nvir, nvir, nvir, nocc, nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 2206; Tree Level  3; Scaling  5/ 5 Result_size  5/ 3
    A28 += oe.contract('fbcdijkl,mf->bcdmijkl',T4,D6, optimize='optimal')
    
    del D6
    
    D28 = np.zeros([nvir, nocc, nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 2207; Tree Level  3; Scaling  6/ 2 Result_size  4/ 2
    D28 += oe.contract('am,mndikl->andikl',T1,G13, optimize='optimal')
    
    del G13
    
    E28 = np.zeros([nvir, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 2208; Tree Level  2; Scaling  6/ 4 Result_size  4/ 4
    E28 += oe.contract('cbnj,andikl->cbadjikl',T2,D28, optimize='optimal')
    
    del D28
    
    #Contraction 2209; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('cbadjikl->abcdijkl',E28)
    
    #Contraction 2210; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('cdabjikl->abcdijkl',E28)
    
    #Contraction 2211; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('bdacjikl->abcdijkl',E28)
    
    #Contraction 2212; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('abcdjikl->abcdijkl',E28)
    
    #Contraction 2213; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('adcbjikl->abcdijkl',E28)
    
    #Contraction 2214; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('bdcajikl->abcdijkl',E28)
    
    #Contraction 2215; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('acbdjikl->abcdijkl',E28)
    
    #Contraction 2216; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('adbcjikl->abcdijkl',E28)
    
    #Contraction 2217; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('cdbajikl->abcdijkl',E28)
    
    #Contraction 2218; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('acdbjikl->abcdijkl',E28)
    
    #Contraction 2219; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('abdcjikl->abcdijkl',E28)
    
    #Contraction 2220; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('cbdajikl->abcdijkl',E28)
    
    #Contraction 2221; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('cbadijkl->abcdijkl',E28)
    
    #Contraction 2222; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('cdabijkl->abcdijkl',E28)
    
    #Contraction 2223; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('bdacijkl->abcdijkl',E28)
    
    #Contraction 2224; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('abcdijkl->abcdijkl',E28)
    
    #Contraction 2225; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('adcbijkl->abcdijkl',E28)
    
    #Contraction 2226; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('bdcaijkl->abcdijkl',E28)
    
    #Contraction 2227; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('acbdijkl->abcdijkl',E28)
    
    #Contraction 2228; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('adbcijkl->abcdijkl',E28)
    
    #Contraction 2229; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('cdbaijkl->abcdijkl',E28)
    
    #Contraction 2230; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('acdbijkl->abcdijkl',E28)
    
    #Contraction 2231; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('abdcijkl->abcdijkl',E28)
    
    #Contraction 2232; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('cbdaijkl->abcdijkl',E28)
    
    #Contraction 2233; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('cbadkjil->abcdijkl',E28)
    
    #Contraction 2234; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('cdabkjil->abcdijkl',E28)
    
    #Contraction 2235; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('bdackjil->abcdijkl',E28)
    
    #Contraction 2236; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('abcdkjil->abcdijkl',E28)
    
    #Contraction 2237; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('adcbkjil->abcdijkl',E28)
    
    #Contraction 2238; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('bdcakjil->abcdijkl',E28)
    
    #Contraction 2239; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('acbdkjil->abcdijkl',E28)
    
    #Contraction 2240; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('adbckjil->abcdijkl',E28)
    
    #Contraction 2241; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('cdbakjil->abcdijkl',E28)
    
    #Contraction 2242; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('acdbkjil->abcdijkl',E28)
    
    #Contraction 2243; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('abdckjil->abcdijkl',E28)
    
    #Contraction 2244; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('cbdakjil->abcdijkl',E28)
    
    #Contraction 2245; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('cbadljik->abcdijkl',E28)
    
    #Contraction 2246; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('cdabljik->abcdijkl',E28)
    
    #Contraction 2247; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('bdacljik->abcdijkl',E28)
    
    #Contraction 2248; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('abcdljik->abcdijkl',E28)
    
    #Contraction 2249; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('adcbljik->abcdijkl',E28)
    
    #Contraction 2250; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('bdcaljik->abcdijkl',E28)
    
    #Contraction 2251; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('acbdljik->abcdijkl',E28)
    
    #Contraction 2252; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('adbcljik->abcdijkl',E28)
    
    #Contraction 2253; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('cdbaljik->abcdijkl',E28)
    
    #Contraction 2254; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('acdbljik->abcdijkl',E28)
    
    #Contraction 2255; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('abdcljik->abcdijkl',E28)
    
    #Contraction 2256; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('cbdaljik->abcdijkl',E28)
    
    del E28
    
    G28 = np.zeros([nvir, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 2257; Tree Level  2; Scaling  6/ 4 Result_size  4/ 4
    G28 += oe.contract('bcmk,amdijl->bcadkijl',T2,M22, optimize='optimal')
    
    del M22
    
    #Contraction 2258; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bcadkijl->abcdijkl',G28)
    
    #Contraction 2259; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bdackijl->abcdijkl',G28)
    
    #Contraction 2260; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('cdabkijl->abcdijkl',G28)
    
    #Contraction 2261; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('acbdkijl->abcdijkl',G28)
    
    #Contraction 2262; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('adbckijl->abcdijkl',G28)
    
    #Contraction 2263; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('cdbakijl->abcdijkl',G28)
    
    #Contraction 2264; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('abcdkijl->abcdijkl',G28)
    
    #Contraction 2265; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('adcbkijl->abcdijkl',G28)
    
    #Contraction 2266; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bdcakijl->abcdijkl',G28)
    
    #Contraction 2267; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('abdckijl->abcdijkl',G28)
    
    #Contraction 2268; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('acdbkijl->abcdijkl',G28)
    
    #Contraction 2269; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bcdakijl->abcdijkl',G28)
    
    #Contraction 2270; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bcadlijk->abcdijkl',G28)
    
    #Contraction 2271; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bdaclijk->abcdijkl',G28)
    
    #Contraction 2272; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('cdablijk->abcdijkl',G28)
    
    #Contraction 2273; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('acbdlijk->abcdijkl',G28)
    
    #Contraction 2274; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('adbclijk->abcdijkl',G28)
    
    #Contraction 2275; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('cdbalijk->abcdijkl',G28)
    
    #Contraction 2276; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('abcdlijk->abcdijkl',G28)
    
    #Contraction 2277; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('adcblijk->abcdijkl',G28)
    
    #Contraction 2278; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bdcalijk->abcdijkl',G28)
    
    #Contraction 2279; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('abdclijk->abcdijkl',G28)
    
    #Contraction 2280; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('acdblijk->abcdijkl',G28)
    
    #Contraction 2281; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bcdalijk->abcdijkl',G28)
    
    #Contraction 2282; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bcadjikl->abcdijkl',G28)
    
    #Contraction 2283; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bdacjikl->abcdijkl',G28)
    
    #Contraction 2284; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('cdabjikl->abcdijkl',G28)
    
    #Contraction 2285; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('acbdjikl->abcdijkl',G28)
    
    #Contraction 2286; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('adbcjikl->abcdijkl',G28)
    
    #Contraction 2287; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('cdbajikl->abcdijkl',G28)
    
    #Contraction 2288; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('abcdjikl->abcdijkl',G28)
    
    #Contraction 2289; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('adcbjikl->abcdijkl',G28)
    
    #Contraction 2290; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bdcajikl->abcdijkl',G28)
    
    #Contraction 2291; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('abdcjikl->abcdijkl',G28)
    
    #Contraction 2292; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('acdbjikl->abcdijkl',G28)
    
    #Contraction 2293; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bcdajikl->abcdijkl',G28)
    
    #Contraction 2294; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bcadlikj->abcdijkl',G28)
    
    #Contraction 2295; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bdaclikj->abcdijkl',G28)
    
    #Contraction 2296; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('cdablikj->abcdijkl',G28)
    
    #Contraction 2297; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('acbdlikj->abcdijkl',G28)
    
    #Contraction 2298; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('adbclikj->abcdijkl',G28)
    
    #Contraction 2299; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('cdbalikj->abcdijkl',G28)
    
    #Contraction 2300; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('abcdlikj->abcdijkl',G28)
    
    #Contraction 2301; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('adcblikj->abcdijkl',G28)
    
    #Contraction 2302; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bdcalikj->abcdijkl',G28)
    
    #Contraction 2303; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('abdclikj->abcdijkl',G28)
    
    #Contraction 2304; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('acdblikj->abcdijkl',G28)
    
    #Contraction 2305; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bcdalikj->abcdijkl',G28)
    
    #Contraction 2306; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bcadjilk->abcdijkl',G28)
    
    #Contraction 2307; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bdacjilk->abcdijkl',G28)
    
    #Contraction 2308; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('cdabjilk->abcdijkl',G28)
    
    #Contraction 2309; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('acbdjilk->abcdijkl',G28)
    
    #Contraction 2310; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('adbcjilk->abcdijkl',G28)
    
    #Contraction 2311; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('cdbajilk->abcdijkl',G28)
    
    #Contraction 2312; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('abcdjilk->abcdijkl',G28)
    
    #Contraction 2313; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('adcbjilk->abcdijkl',G28)
    
    #Contraction 2314; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bdcajilk->abcdijkl',G28)
    
    #Contraction 2315; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('abdcjilk->abcdijkl',G28)
    
    #Contraction 2316; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('acdbjilk->abcdijkl',G28)
    
    #Contraction 2317; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bcdajilk->abcdijkl',G28)
    
    #Contraction 2318; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bcadkilj->abcdijkl',G28)
    
    #Contraction 2319; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bdackilj->abcdijkl',G28)
    
    #Contraction 2320; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('cdabkilj->abcdijkl',G28)
    
    #Contraction 2321; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('acbdkilj->abcdijkl',G28)
    
    #Contraction 2322; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('adbckilj->abcdijkl',G28)
    
    #Contraction 2323; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('cdbakilj->abcdijkl',G28)
    
    #Contraction 2324; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('abcdkilj->abcdijkl',G28)
    
    #Contraction 2325; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('adcbkilj->abcdijkl',G28)
    
    #Contraction 2326; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bdcakilj->abcdijkl',G28)
    
    #Contraction 2327; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('abdckilj->abcdijkl',G28)
    
    #Contraction 2328; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('acdbkilj->abcdijkl',G28)
    
    #Contraction 2329; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bcdakilj->abcdijkl',G28)
    
    #Contraction 2330; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bcadijkl->abcdijkl',G28)
    
    #Contraction 2331; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bdacijkl->abcdijkl',G28)
    
    #Contraction 2332; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('cdabijkl->abcdijkl',G28)
    
    #Contraction 2333; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('acbdijkl->abcdijkl',G28)
    
    #Contraction 2334; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('adbcijkl->abcdijkl',G28)
    
    #Contraction 2335; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('cdbaijkl->abcdijkl',G28)
    
    #Contraction 2336; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('abcdijkl->abcdijkl',G28)
    
    #Contraction 2337; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('adcbijkl->abcdijkl',G28)
    
    #Contraction 2338; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bdcaijkl->abcdijkl',G28)
    
    #Contraction 2339; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('abdcijkl->abcdijkl',G28)
    
    #Contraction 2340; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('acdbijkl->abcdijkl',G28)
    
    #Contraction 2341; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bcdaijkl->abcdijkl',G28)
    
    #Contraction 2342; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bcadljki->abcdijkl',G28)
    
    #Contraction 2343; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bdacljki->abcdijkl',G28)
    
    #Contraction 2344; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('cdabljki->abcdijkl',G28)
    
    #Contraction 2345; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('acbdljki->abcdijkl',G28)
    
    #Contraction 2346; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('adbcljki->abcdijkl',G28)
    
    #Contraction 2347; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('cdbaljki->abcdijkl',G28)
    
    #Contraction 2348; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('abcdljki->abcdijkl',G28)
    
    #Contraction 2349; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('adcbljki->abcdijkl',G28)
    
    #Contraction 2350; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bdcaljki->abcdijkl',G28)
    
    #Contraction 2351; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('abdcljki->abcdijkl',G28)
    
    #Contraction 2352; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('acdbljki->abcdijkl',G28)
    
    #Contraction 2353; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bcdaljki->abcdijkl',G28)
    
    #Contraction 2354; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bcadijlk->abcdijkl',G28)
    
    #Contraction 2355; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bdacijlk->abcdijkl',G28)
    
    #Contraction 2356; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('cdabijlk->abcdijkl',G28)
    
    #Contraction 2357; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('acbdijlk->abcdijkl',G28)
    
    #Contraction 2358; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('adbcijlk->abcdijkl',G28)
    
    #Contraction 2359; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('cdbaijlk->abcdijkl',G28)
    
    #Contraction 2360; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('abcdijlk->abcdijkl',G28)
    
    #Contraction 2361; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('adcbijlk->abcdijkl',G28)
    
    #Contraction 2362; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bdcaijlk->abcdijkl',G28)
    
    #Contraction 2363; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('abdcijlk->abcdijkl',G28)
    
    #Contraction 2364; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('acdbijlk->abcdijkl',G28)
    
    #Contraction 2365; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bcdaijlk->abcdijkl',G28)
    
    #Contraction 2366; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bcadkjli->abcdijkl',G28)
    
    #Contraction 2367; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bdackjli->abcdijkl',G28)
    
    #Contraction 2368; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('cdabkjli->abcdijkl',G28)
    
    #Contraction 2369; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('acbdkjli->abcdijkl',G28)
    
    #Contraction 2370; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('adbckjli->abcdijkl',G28)
    
    #Contraction 2371; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('cdbakjli->abcdijkl',G28)
    
    #Contraction 2372; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('abcdkjli->abcdijkl',G28)
    
    #Contraction 2373; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('adcbkjli->abcdijkl',G28)
    
    #Contraction 2374; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bdcakjli->abcdijkl',G28)
    
    #Contraction 2375; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('abdckjli->abcdijkl',G28)
    
    #Contraction 2376; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('acdbkjli->abcdijkl',G28)
    
    #Contraction 2377; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bcdakjli->abcdijkl',G28)
    
    #Contraction 2378; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bcadiklj->abcdijkl',G28)
    
    #Contraction 2379; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bdaciklj->abcdijkl',G28)
    
    #Contraction 2380; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('cdabiklj->abcdijkl',G28)
    
    #Contraction 2381; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('acbdiklj->abcdijkl',G28)
    
    #Contraction 2382; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('adbciklj->abcdijkl',G28)
    
    #Contraction 2383; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('cdbaiklj->abcdijkl',G28)
    
    #Contraction 2384; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('abcdiklj->abcdijkl',G28)
    
    #Contraction 2385; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('adcbiklj->abcdijkl',G28)
    
    #Contraction 2386; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bdcaiklj->abcdijkl',G28)
    
    #Contraction 2387; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('abdciklj->abcdijkl',G28)
    
    #Contraction 2388; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('acdbiklj->abcdijkl',G28)
    
    #Contraction 2389; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bcdaiklj->abcdijkl',G28)
    
    #Contraction 2390; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bcadjkli->abcdijkl',G28)
    
    #Contraction 2391; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bdacjkli->abcdijkl',G28)
    
    #Contraction 2392; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('cdabjkli->abcdijkl',G28)
    
    #Contraction 2393; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('acbdjkli->abcdijkl',G28)
    
    #Contraction 2394; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('adbcjkli->abcdijkl',G28)
    
    #Contraction 2395; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('cdbajkli->abcdijkl',G28)
    
    #Contraction 2396; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('abcdjkli->abcdijkl',G28)
    
    #Contraction 2397; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('adcbjkli->abcdijkl',G28)
    
    #Contraction 2398; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bdcajkli->abcdijkl',G28)
    
    #Contraction 2399; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('abdcjkli->abcdijkl',G28)
    
    #Contraction 2400; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('acdbjkli->abcdijkl',G28)
    
    #Contraction 2401; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bcdajkli->abcdijkl',G28)
    
    del G28
    
    I28 = np.zeros([nocc, nocc, nvir, nvir, nocc, nocc], dtype=type_)
    
    #Contraction 2402; Tree Level  5; Scaling  4/ 4 Result_size  4/ 2
    I28 += oe.contract('mnef,fdjl->mndejl',V7,T2, optimize='optimal')
    
    J28 = np.zeros([nvir, nocc, nocc, nvir, nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 2403; Tree Level  4; Scaling  6/ 4 Result_size  6/ 2
    J28 += oe.contract('ecik,mndejl->cmndikjl',T2,I28, optimize='optimal')
    
    M28 = np.zeros([nvir, nvir, nocc, nvir, nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 2404; Tree Level  3; Scaling  7/ 3 Result_size  5/ 3
    M28 += oe.contract('bn,cmndikjl->bcmdikjl',T1,J28, optimize='optimal')
    
    del J28
    
    X28 = np.zeros([nvir, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 2405; Tree Level  2; Scaling  6/ 4 Result_size  4/ 4
    X28 += oe.contract('am,bcmdikjl->abcdikjl',T1,M28, optimize='optimal')
    
    del M28
    
    #Contraction 2406; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('abcdikjl->abcdijkl',X28)
    
    #Contraction 2407; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('abdcikjl->abcdijkl',X28)
    
    #Contraction 2408; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('acbdikjl->abcdijkl',X28)
    
    #Contraction 2409; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('acdbikjl->abcdijkl',X28)
    
    #Contraction 2410; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('adbcikjl->abcdijkl',X28)
    
    #Contraction 2411; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('adcbikjl->abcdijkl',X28)
    
    #Contraction 2412; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('bacdikjl->abcdijkl',X28)
    
    #Contraction 2413; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('badcikjl->abcdijkl',X28)
    
    #Contraction 2414; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('bcadikjl->abcdijkl',X28)
    
    #Contraction 2415; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('bcdaikjl->abcdijkl',X28)
    
    #Contraction 2416; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('bdacikjl->abcdijkl',X28)
    
    #Contraction 2417; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('bdcaikjl->abcdijkl',X28)
    
    #Contraction 2418; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('cabdikjl->abcdijkl',X28)
    
    #Contraction 2419; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('cadbikjl->abcdijkl',X28)
    
    #Contraction 2420; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('cbadikjl->abcdijkl',X28)
    
    #Contraction 2421; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('cbdaikjl->abcdijkl',X28)
    
    #Contraction 2422; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('cdabikjl->abcdijkl',X28)
    
    #Contraction 2423; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('cdbaikjl->abcdijkl',X28)
    
    #Contraction 2424; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('dabcikjl->abcdijkl',X28)
    
    #Contraction 2425; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('dacbikjl->abcdijkl',X28)
    
    #Contraction 2426; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('dbacikjl->abcdijkl',X28)
    
    #Contraction 2427; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('dbcaikjl->abcdijkl',X28)
    
    #Contraction 2428; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('dcabikjl->abcdijkl',X28)
    
    #Contraction 2429; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('dcbaikjl->abcdijkl',X28)
    
    #Contraction 2430; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('abcdijkl->abcdijkl',X28)
    
    #Contraction 2431; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('abdcijkl->abcdijkl',X28)
    
    #Contraction 2432; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('acbdijkl->abcdijkl',X28)
    
    #Contraction 2433; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('acdbijkl->abcdijkl',X28)
    
    #Contraction 2434; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('adbcijkl->abcdijkl',X28)
    
    #Contraction 2435; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('adcbijkl->abcdijkl',X28)
    
    #Contraction 2436; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('bacdijkl->abcdijkl',X28)
    
    #Contraction 2437; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('badcijkl->abcdijkl',X28)
    
    #Contraction 2438; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('bcadijkl->abcdijkl',X28)
    
    #Contraction 2439; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('bcdaijkl->abcdijkl',X28)
    
    #Contraction 2440; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('bdacijkl->abcdijkl',X28)
    
    #Contraction 2441; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('bdcaijkl->abcdijkl',X28)
    
    #Contraction 2442; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('cabdijkl->abcdijkl',X28)
    
    #Contraction 2443; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('cadbijkl->abcdijkl',X28)
    
    #Contraction 2444; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('cbadijkl->abcdijkl',X28)
    
    #Contraction 2445; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('cbdaijkl->abcdijkl',X28)
    
    #Contraction 2446; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('cdabijkl->abcdijkl',X28)
    
    #Contraction 2447; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('cdbaijkl->abcdijkl',X28)
    
    #Contraction 2448; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('dabcijkl->abcdijkl',X28)
    
    #Contraction 2449; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('dacbijkl->abcdijkl',X28)
    
    #Contraction 2450; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('dbacijkl->abcdijkl',X28)
    
    #Contraction 2451; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('dbcaijkl->abcdijkl',X28)
    
    #Contraction 2452; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('dcabijkl->abcdijkl',X28)
    
    #Contraction 2453; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('dcbaijkl->abcdijkl',X28)
    
    #Contraction 2454; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('abcdilkj->abcdijkl',X28)
    
    #Contraction 2455; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('abdcilkj->abcdijkl',X28)
    
    #Contraction 2456; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('acbdilkj->abcdijkl',X28)
    
    #Contraction 2457; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('acdbilkj->abcdijkl',X28)
    
    #Contraction 2458; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('adbcilkj->abcdijkl',X28)
    
    #Contraction 2459; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('adcbilkj->abcdijkl',X28)
    
    #Contraction 2460; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('bacdilkj->abcdijkl',X28)
    
    #Contraction 2461; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('badcilkj->abcdijkl',X28)
    
    #Contraction 2462; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('bcadilkj->abcdijkl',X28)
    
    #Contraction 2463; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('bcdailkj->abcdijkl',X28)
    
    #Contraction 2464; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('bdacilkj->abcdijkl',X28)
    
    #Contraction 2465; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('bdcailkj->abcdijkl',X28)
    
    #Contraction 2466; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('cabdilkj->abcdijkl',X28)
    
    #Contraction 2467; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('cadbilkj->abcdijkl',X28)
    
    #Contraction 2468; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('cbadilkj->abcdijkl',X28)
    
    #Contraction 2469; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('cbdailkj->abcdijkl',X28)
    
    #Contraction 2470; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('cdabilkj->abcdijkl',X28)
    
    #Contraction 2471; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('cdbailkj->abcdijkl',X28)
    
    #Contraction 2472; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('dabcilkj->abcdijkl',X28)
    
    #Contraction 2473; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('dacbilkj->abcdijkl',X28)
    
    #Contraction 2474; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('dbacilkj->abcdijkl',X28)
    
    #Contraction 2475; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('dbcailkj->abcdijkl',X28)
    
    #Contraction 2476; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('dcabilkj->abcdijkl',X28)
    
    #Contraction 2477; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('dcbailkj->abcdijkl',X28)
    
    #Contraction 2478; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('abcdkjil->abcdijkl',X28)
    
    #Contraction 2479; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('abdckjil->abcdijkl',X28)
    
    #Contraction 2480; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('acbdkjil->abcdijkl',X28)
    
    #Contraction 2481; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('acdbkjil->abcdijkl',X28)
    
    #Contraction 2482; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('adbckjil->abcdijkl',X28)
    
    #Contraction 2483; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('adcbkjil->abcdijkl',X28)
    
    #Contraction 2484; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('bacdkjil->abcdijkl',X28)
    
    #Contraction 2485; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('badckjil->abcdijkl',X28)
    
    #Contraction 2486; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('bcadkjil->abcdijkl',X28)
    
    #Contraction 2487; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('bcdakjil->abcdijkl',X28)
    
    #Contraction 2488; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('bdackjil->abcdijkl',X28)
    
    #Contraction 2489; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('bdcakjil->abcdijkl',X28)
    
    #Contraction 2490; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('cabdkjil->abcdijkl',X28)
    
    #Contraction 2491; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('cadbkjil->abcdijkl',X28)
    
    #Contraction 2492; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('cbadkjil->abcdijkl',X28)
    
    #Contraction 2493; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('cbdakjil->abcdijkl',X28)
    
    #Contraction 2494; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('cdabkjil->abcdijkl',X28)
    
    #Contraction 2495; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('cdbakjil->abcdijkl',X28)
    
    #Contraction 2496; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('dabckjil->abcdijkl',X28)
    
    #Contraction 2497; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('dacbkjil->abcdijkl',X28)
    
    #Contraction 2498; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('dbackjil->abcdijkl',X28)
    
    #Contraction 2499; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('dbcakjil->abcdijkl',X28)
    
    #Contraction 2500; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('dcabkjil->abcdijkl',X28)
    
    #Contraction 2501; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('dcbakjil->abcdijkl',X28)
    
    #Contraction 2502; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('abcdklij->abcdijkl',X28)
    
    #Contraction 2503; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('abdcklij->abcdijkl',X28)
    
    #Contraction 2504; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('acbdklij->abcdijkl',X28)
    
    #Contraction 2505; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('acdbklij->abcdijkl',X28)
    
    #Contraction 2506; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('adbcklij->abcdijkl',X28)
    
    #Contraction 2507; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('adcbklij->abcdijkl',X28)
    
    #Contraction 2508; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('bacdklij->abcdijkl',X28)
    
    #Contraction 2509; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('badcklij->abcdijkl',X28)
    
    #Contraction 2510; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('bcadklij->abcdijkl',X28)
    
    #Contraction 2511; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('bcdaklij->abcdijkl',X28)
    
    #Contraction 2512; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('bdacklij->abcdijkl',X28)
    
    #Contraction 2513; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('bdcaklij->abcdijkl',X28)
    
    #Contraction 2514; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('cabdklij->abcdijkl',X28)
    
    #Contraction 2515; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('cadbklij->abcdijkl',X28)
    
    #Contraction 2516; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('cbadklij->abcdijkl',X28)
    
    #Contraction 2517; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('cbdaklij->abcdijkl',X28)
    
    #Contraction 2518; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('cdabklij->abcdijkl',X28)
    
    #Contraction 2519; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('cdbaklij->abcdijkl',X28)
    
    #Contraction 2520; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('dabcklij->abcdijkl',X28)
    
    #Contraction 2521; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('dacbklij->abcdijkl',X28)
    
    #Contraction 2522; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('dbacklij->abcdijkl',X28)
    
    #Contraction 2523; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('dbcaklij->abcdijkl',X28)
    
    #Contraction 2524; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('dcabklij->abcdijkl',X28)
    
    #Contraction 2525; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('dcbaklij->abcdijkl',X28)
    
    #Contraction 2526; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('abcdjlik->abcdijkl',X28)
    
    #Contraction 2527; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('abdcjlik->abcdijkl',X28)
    
    #Contraction 2528; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('acbdjlik->abcdijkl',X28)
    
    #Contraction 2529; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('acdbjlik->abcdijkl',X28)
    
    #Contraction 2530; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('adbcjlik->abcdijkl',X28)
    
    #Contraction 2531; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('adcbjlik->abcdijkl',X28)
    
    #Contraction 2532; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('bacdjlik->abcdijkl',X28)
    
    #Contraction 2533; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('badcjlik->abcdijkl',X28)
    
    #Contraction 2534; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('bcadjlik->abcdijkl',X28)
    
    #Contraction 2535; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('bcdajlik->abcdijkl',X28)
    
    #Contraction 2536; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('bdacjlik->abcdijkl',X28)
    
    #Contraction 2537; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('bdcajlik->abcdijkl',X28)
    
    #Contraction 2538; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('cabdjlik->abcdijkl',X28)
    
    #Contraction 2539; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('cadbjlik->abcdijkl',X28)
    
    #Contraction 2540; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('cbadjlik->abcdijkl',X28)
    
    #Contraction 2541; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('cbdajlik->abcdijkl',X28)
    
    #Contraction 2542; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('cdabjlik->abcdijkl',X28)
    
    #Contraction 2543; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('cdbajlik->abcdijkl',X28)
    
    #Contraction 2544; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('dabcjlik->abcdijkl',X28)
    
    #Contraction 2545; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('dacbjlik->abcdijkl',X28)
    
    #Contraction 2546; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('dbacjlik->abcdijkl',X28)
    
    #Contraction 2547; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('dbcajlik->abcdijkl',X28)
    
    #Contraction 2548; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('dcabjlik->abcdijkl',X28)
    
    #Contraction 2549; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('dcbajlik->abcdijkl',X28)
    
    del X28
    
    Y28 = np.zeros([nvir, nvir, nvir, nvir, nocc, nocc], dtype=type_)
    
    #Contraction 2550; Tree Level  3; Scaling  6/ 4 Result_size  2/ 4
    Y28 += oe.contract('bcmn,mndekl->bcdekl',T2,I28, optimize='optimal')
    
    del I28
    
    A29 = np.zeros([nvir, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 2551; Tree Level  2; Scaling  6/ 4 Result_size  4/ 4
    A29 += oe.contract('am,bcdmijkl->abcdijkl',T1,A28, optimize='optimal')
    
    del A28
    
    #Contraction 2552; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('abcdijkl->abcdijkl',A29)
    
    #Contraction 2553; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bacdijkl->abcdijkl',A29)
    
    #Contraction 2554; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('cabdijkl->abcdijkl',A29)
    
    #Contraction 2555; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('dabcijkl->abcdijkl',A29)
    
    del A29
    
    D29 = np.zeros([nvir, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 2556; Tree Level  2; Scaling  6/ 4 Result_size  4/ 4
    D29 += oe.contract('acdnkl,bnji->acdbklji',T3,Y21, optimize='optimal')
    
    del Y21
    
    #Contraction 2557; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('acdbklji->abcdijkl',D29)
    
    #Contraction 2558; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bcdaklji->abcdijkl',D29)
    
    #Contraction 2559; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('badcklji->abcdijkl',D29)
    
    #Contraction 2560; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bacdklji->abcdijkl',D29)
    
    #Contraction 2561; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('acdbjlki->abcdijkl',D29)
    
    #Contraction 2562; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bcdajlki->abcdijkl',D29)
    
    #Contraction 2563; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('badcjlki->abcdijkl',D29)
    
    #Contraction 2564; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bacdjlki->abcdijkl',D29)
    
    #Contraction 2565; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('acdbjkli->abcdijkl',D29)
    
    #Contraction 2566; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bcdajkli->abcdijkl',D29)
    
    #Contraction 2567; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('badcjkli->abcdijkl',D29)
    
    #Contraction 2568; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bacdjkli->abcdijkl',D29)
    
    #Contraction 2569; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('acdbklij->abcdijkl',D29)
    
    #Contraction 2570; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bcdaklij->abcdijkl',D29)
    
    #Contraction 2571; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('badcklij->abcdijkl',D29)
    
    #Contraction 2572; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bacdklij->abcdijkl',D29)
    
    #Contraction 2573; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('acdbilkj->abcdijkl',D29)
    
    #Contraction 2574; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bcdailkj->abcdijkl',D29)
    
    #Contraction 2575; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('badcilkj->abcdijkl',D29)
    
    #Contraction 2576; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bacdilkj->abcdijkl',D29)
    
    #Contraction 2577; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('acdbiklj->abcdijkl',D29)
    
    #Contraction 2578; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bcdaiklj->abcdijkl',D29)
    
    #Contraction 2579; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('badciklj->abcdijkl',D29)
    
    #Contraction 2580; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bacdiklj->abcdijkl',D29)
    
    #Contraction 2581; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('acdbjlik->abcdijkl',D29)
    
    #Contraction 2582; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bcdajlik->abcdijkl',D29)
    
    #Contraction 2583; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('badcjlik->abcdijkl',D29)
    
    #Contraction 2584; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bacdjlik->abcdijkl',D29)
    
    #Contraction 2585; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('acdbiljk->abcdijkl',D29)
    
    #Contraction 2586; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bcdailjk->abcdijkl',D29)
    
    #Contraction 2587; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('badciljk->abcdijkl',D29)
    
    #Contraction 2588; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bacdiljk->abcdijkl',D29)
    
    #Contraction 2589; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('acdbijlk->abcdijkl',D29)
    
    #Contraction 2590; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bcdaijlk->abcdijkl',D29)
    
    #Contraction 2591; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('badcijlk->abcdijkl',D29)
    
    #Contraction 2592; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bacdijlk->abcdijkl',D29)
    
    #Contraction 2593; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('acdbjkil->abcdijkl',D29)
    
    #Contraction 2594; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bcdajkil->abcdijkl',D29)
    
    #Contraction 2595; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('badcjkil->abcdijkl',D29)
    
    #Contraction 2596; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bacdjkil->abcdijkl',D29)
    
    #Contraction 2597; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('acdbikjl->abcdijkl',D29)
    
    #Contraction 2598; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bcdaikjl->abcdijkl',D29)
    
    #Contraction 2599; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('badcikjl->abcdijkl',D29)
    
    #Contraction 2600; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bacdikjl->abcdijkl',D29)
    
    #Contraction 2601; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('acdbijkl->abcdijkl',D29)
    
    #Contraction 2602; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bcdaijkl->abcdijkl',D29)
    
    #Contraction 2603; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('badcijkl->abcdijkl',D29)
    
    #Contraction 2604; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bacdijkl->abcdijkl',D29)
    
    del D29
    
    E29 = np.zeros([nocc, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 2605; Tree Level  3; Scaling  5/ 5 Result_size  5/ 3
    E29 += oe.contract('me,ebcdijkl->mbcdijkl',F3,T4, optimize='optimal')
    
    #del F3
    
    G29 = np.zeros([nvir, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 2606; Tree Level  2; Scaling  6/ 4 Result_size  4/ 4
    G29 += oe.contract('am,mbcdijkl->abcdijkl',T1,E29, optimize='optimal')
    
    del E29
    
    #Contraction 2607; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('abcdijkl->abcdijkl',G29)
    
    #Contraction 2608; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bacdijkl->abcdijkl',G29)
    
    #Contraction 2609; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('cabdijkl->abcdijkl',G29)
    
    #Contraction 2610; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('dabcijkl->abcdijkl',G29)
    
    del G29
    
    I29 = np.zeros([nvir, nvir, nvir, nvir, nocc, nocc], dtype=type_)
    
    #Contraction 2611; Tree Level  3; Scaling  6/ 6 Result_size  2/ 4
    I29 += oe.contract('mnef,fbcdmnkl->bcdekl',V7,T4, optimize='optimal')
    
    J29 = np.zeros([nvir, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 2612; Tree Level  2; Scaling  4/ 6 Result_size  4/ 4
    J29 += oe.contract('aeij,bcdekl->abcdijkl',T2,I29, optimize='optimal')
    
    del I29
    
    #Contraction 2613; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('abcdijkl->abcdijkl',J29)
    
    #Contraction 2614; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('bacdijkl->abcdijkl',J29)
    
    #Contraction 2615; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('cabdijkl->abcdijkl',J29)
    
    #Contraction 2616; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('dabcijkl->abcdijkl',J29)
    
    #Contraction 2617; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('abcdikjl->abcdijkl',J29)
    
    #Contraction 2618; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('bacdikjl->abcdijkl',J29)
    
    #Contraction 2619; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('cabdikjl->abcdijkl',J29)
    
    #Contraction 2620; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('dabcikjl->abcdijkl',J29)
    
    #Contraction 2621; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('abcdiljk->abcdijkl',J29)
    
    #Contraction 2622; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('bacdiljk->abcdijkl',J29)
    
    #Contraction 2623; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('cabdiljk->abcdijkl',J29)
    
    #Contraction 2624; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('dabciljk->abcdijkl',J29)
    
    #Contraction 2625; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('abcdjkil->abcdijkl',J29)
    
    #Contraction 2626; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('bacdjkil->abcdijkl',J29)
    
    #Contraction 2627; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('cabdjkil->abcdijkl',J29)
    
    #Contraction 2628; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('dabcjkil->abcdijkl',J29)
    
    #Contraction 2629; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('abcdjlik->abcdijkl',J29)
    
    #Contraction 2630; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('bacdjlik->abcdijkl',J29)
    
    #Contraction 2631; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('cabdjlik->abcdijkl',J29)
    
    #Contraction 2632; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('dabcjlik->abcdijkl',J29)
    
    #Contraction 2633; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('abcdklij->abcdijkl',J29)
    
    #Contraction 2634; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('bacdklij->abcdijkl',J29)
    
    #Contraction 2635; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('cabdklij->abcdijkl',J29)
    
    #Contraction 2636; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('dabcklij->abcdijkl',J29)
    
    del J29
    
    M29 = np.zeros([nocc, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 2637; Tree Level  3; Scaling  7/ 5 Result_size  5/ 3
    M29 += oe.contract('mnie,ebcdnjkl->mbcdijkl',V4,T4, optimize='optimal')
    
    #del V4
    
    X29 = np.zeros([nvir, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 2638; Tree Level  2; Scaling  6/ 4 Result_size  4/ 4
    X29 += oe.contract('am,mbcdijkl->abcdijkl',T1,M29, optimize='optimal')
    
    del M29
    
    #Contraction 2639; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('abcdijkl->abcdijkl',X29)
    
    #Contraction 2640; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bacdijkl->abcdijkl',X29)
    
    #Contraction 2641; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('cabdijkl->abcdijkl',X29)
    
    #Contraction 2642; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('dabcijkl->abcdijkl',X29)
    
    #Contraction 2643; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('abcdjikl->abcdijkl',X29)
    
    #Contraction 2644; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bacdjikl->abcdijkl',X29)
    
    #Contraction 2645; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('cabdjikl->abcdijkl',X29)
    
    #Contraction 2646; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('dabcjikl->abcdijkl',X29)
    
    #Contraction 2647; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('abcdkijl->abcdijkl',X29)
    
    #Contraction 2648; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bacdkijl->abcdijkl',X29)
    
    #Contraction 2649; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('cabdkijl->abcdijkl',X29)
    
    #Contraction 2650; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('dabckijl->abcdijkl',X29)
    
    #Contraction 2651; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('abcdlijk->abcdijkl',X29)
    
    #Contraction 2652; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bacdlijk->abcdijkl',X29)
    
    #Contraction 2653; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('cabdlijk->abcdijkl',X29)
    
    #Contraction 2654; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('dabclijk->abcdijkl',X29)
    
    del X29
    
    Y29 = np.zeros([nvir, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 2655; Tree Level  2; Scaling  6/ 6 Result_size  4/ 4
    Y29 += oe.contract('fbcdnjkl,nafi->bcdajkli',T4,G17, optimize='optimal')
    
    del G17
    
    #Contraction 2656; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bcdajkli->abcdijkl',Y29)
    
    #Contraction 2657; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('acdbjkli->abcdijkl',Y29)
    
    #Contraction 2658; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('abdcjkli->abcdijkl',Y29)
    
    #Contraction 2659; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('abcdjkli->abcdijkl',Y29)
    
    #Contraction 2660; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bcdaiklj->abcdijkl',Y29)
    
    #Contraction 2661; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('acdbiklj->abcdijkl',Y29)
    
    #Contraction 2662; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('abdciklj->abcdijkl',Y29)
    
    #Contraction 2663; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('abcdiklj->abcdijkl',Y29)
    
    #Contraction 2664; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bcdaijlk->abcdijkl',Y29)
    
    #Contraction 2665; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('acdbijlk->abcdijkl',Y29)
    
    #Contraction 2666; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('abdcijlk->abcdijkl',Y29)
    
    #Contraction 2667; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('abcdijlk->abcdijkl',Y29)
    
    #Contraction 2668; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bcdaijkl->abcdijkl',Y29)
    
    #Contraction 2669; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('acdbijkl->abcdijkl',Y29)
    
    #Contraction 2670; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('abdcijkl->abcdijkl',Y29)
    
    #Contraction 2671; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('abcdijkl->abcdijkl',Y29)
    
    del Y29
    
    A30 = np.zeros([nvir, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 2672; Tree Level  2; Scaling  4/ 6 Result_size  4/ 4
    A30 += oe.contract('fcdjkl,abfi->cdabjkli',T3,A20, optimize='optimal')
    
    del A20
    
    #Contraction 2673; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('cdabjkli->abcdijkl',A30)
    
    #Contraction 2674; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bdacjkli->abcdijkl',A30)
    
    #Contraction 2675; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bcadjkli->abcdijkl',A30)
    
    #Contraction 2676; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('adbcjkli->abcdijkl',A30)
    
    #Contraction 2677; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('acbdjkli->abcdijkl',A30)
    
    #Contraction 2678; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('abcdjkli->abcdijkl',A30)
    
    #Contraction 2679; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('cdabiklj->abcdijkl',A30)
    
    #Contraction 2680; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bdaciklj->abcdijkl',A30)
    
    #Contraction 2681; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bcadiklj->abcdijkl',A30)
    
    #Contraction 2682; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('adbciklj->abcdijkl',A30)
    
    #Contraction 2683; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('acbdiklj->abcdijkl',A30)
    
    #Contraction 2684; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('abcdiklj->abcdijkl',A30)
    
    #Contraction 2685; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('cdabijlk->abcdijkl',A30)
    
    #Contraction 2686; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bdacijlk->abcdijkl',A30)
    
    #Contraction 2687; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bcadijlk->abcdijkl',A30)
    
    #Contraction 2688; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('adbcijlk->abcdijkl',A30)
    
    #Contraction 2689; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('acbdijlk->abcdijkl',A30)
    
    #Contraction 2690; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('abcdijlk->abcdijkl',A30)
    
    #Contraction 2691; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('cdabijkl->abcdijkl',A30)
    
    #Contraction 2692; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bdacijkl->abcdijkl',A30)
    
    #Contraction 2693; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bcadijkl->abcdijkl',A30)
    
    #Contraction 2694; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('adbcijkl->abcdijkl',A30)
    
    #Contraction 2695; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('acbdijkl->abcdijkl',A30)
    
    #Contraction 2696; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('abcdijkl->abcdijkl',A30)
    
    del A30
    
    D30 = np.zeros([nvir, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 2697; Tree Level  2; Scaling  4/ 6 Result_size  4/ 4
    D30 += oe.contract('aeij,bcdekl->abcdijkl',T2,Y28, optimize='optimal')
    
    del Y28
    
    #Contraction 2698; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('abcdijkl->abcdijkl',D30)
    
    #Contraction 2699; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('adcbijkl->abcdijkl',D30)
    
    #Contraction 2700; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('adbcijkl->abcdijkl',D30)
    
    #Contraction 2701; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('dbcaijkl->abcdijkl',D30)
    
    #Contraction 2702; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('dacbijkl->abcdijkl',D30)
    
    #Contraction 2703; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('dabcijkl->abcdijkl',D30)
    
    #Contraction 2704; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('bdcaijkl->abcdijkl',D30)
    
    #Contraction 2705; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('bacdijkl->abcdijkl',D30)
    
    #Contraction 2706; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('badcijkl->abcdijkl',D30)
    
    #Contraction 2707; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('cdbaijkl->abcdijkl',D30)
    
    #Contraction 2708; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('cabdijkl->abcdijkl',D30)
    
    #Contraction 2709; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('cadbijkl->abcdijkl',D30)
    
    #Contraction 2710; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('abcdikjl->abcdijkl',D30)
    
    #Contraction 2711; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('adcbikjl->abcdijkl',D30)
    
    #Contraction 2712; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('adbcikjl->abcdijkl',D30)
    
    #Contraction 2713; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('dbcaikjl->abcdijkl',D30)
    
    #Contraction 2714; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('dacbikjl->abcdijkl',D30)
    
    #Contraction 2715; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('dabcikjl->abcdijkl',D30)
    
    #Contraction 2716; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('bdcaikjl->abcdijkl',D30)
    
    #Contraction 2717; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('bacdikjl->abcdijkl',D30)
    
    #Contraction 2718; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('badcikjl->abcdijkl',D30)
    
    #Contraction 2719; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('cdbaikjl->abcdijkl',D30)
    
    #Contraction 2720; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('cabdikjl->abcdijkl',D30)
    
    #Contraction 2721; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('cadbikjl->abcdijkl',D30)
    
    #Contraction 2722; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('abcdiljk->abcdijkl',D30)
    
    #Contraction 2723; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('adcbiljk->abcdijkl',D30)
    
    #Contraction 2724; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('adbciljk->abcdijkl',D30)
    
    #Contraction 2725; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('dbcailjk->abcdijkl',D30)
    
    #Contraction 2726; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('dacbiljk->abcdijkl',D30)
    
    #Contraction 2727; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('dabciljk->abcdijkl',D30)
    
    #Contraction 2728; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('bdcailjk->abcdijkl',D30)
    
    #Contraction 2729; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('bacdiljk->abcdijkl',D30)
    
    #Contraction 2730; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('badciljk->abcdijkl',D30)
    
    #Contraction 2731; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('cdbailjk->abcdijkl',D30)
    
    #Contraction 2732; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('cabdiljk->abcdijkl',D30)
    
    #Contraction 2733; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('cadbiljk->abcdijkl',D30)
    
    #Contraction 2734; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('abcdjkil->abcdijkl',D30)
    
    #Contraction 2735; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('adcbjkil->abcdijkl',D30)
    
    #Contraction 2736; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('adbcjkil->abcdijkl',D30)
    
    #Contraction 2737; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('dbcajkil->abcdijkl',D30)
    
    #Contraction 2738; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('dacbjkil->abcdijkl',D30)
    
    #Contraction 2739; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('dabcjkil->abcdijkl',D30)
    
    #Contraction 2740; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('bdcajkil->abcdijkl',D30)
    
    #Contraction 2741; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('bacdjkil->abcdijkl',D30)
    
    #Contraction 2742; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('badcjkil->abcdijkl',D30)
    
    #Contraction 2743; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('cdbajkil->abcdijkl',D30)
    
    #Contraction 2744; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('cabdjkil->abcdijkl',D30)
    
    #Contraction 2745; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('cadbjkil->abcdijkl',D30)
    
    #Contraction 2746; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('abcdjlik->abcdijkl',D30)
    
    #Contraction 2747; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('adcbjlik->abcdijkl',D30)
    
    #Contraction 2748; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('adbcjlik->abcdijkl',D30)
    
    #Contraction 2749; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('dbcajlik->abcdijkl',D30)
    
    #Contraction 2750; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('dacbjlik->abcdijkl',D30)
    
    #Contraction 2751; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('dabcjlik->abcdijkl',D30)
    
    #Contraction 2752; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('bdcajlik->abcdijkl',D30)
    
    #Contraction 2753; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('bacdjlik->abcdijkl',D30)
    
    #Contraction 2754; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('badcjlik->abcdijkl',D30)
    
    #Contraction 2755; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('cdbajlik->abcdijkl',D30)
    
    #Contraction 2756; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('cabdjlik->abcdijkl',D30)
    
    #Contraction 2757; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('cadbjlik->abcdijkl',D30)
    
    #Contraction 2758; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('abcdklij->abcdijkl',D30)
    
    #Contraction 2759; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('adcbklij->abcdijkl',D30)
    
    #Contraction 2760; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('adbcklij->abcdijkl',D30)
    
    #Contraction 2761; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('dbcaklij->abcdijkl',D30)
    
    #Contraction 2762; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('dacbklij->abcdijkl',D30)
    
    #Contraction 2763; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('dabcklij->abcdijkl',D30)
    
    #Contraction 2764; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('bdcaklij->abcdijkl',D30)
    
    #Contraction 2765; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('bacdklij->abcdijkl',D30)
    
    #Contraction 2766; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('badcklij->abcdijkl',D30)
    
    #Contraction 2767; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('cdbaklij->abcdijkl',D30)
    
    #Contraction 2768; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('cabdklij->abcdijkl',D30)
    
    #Contraction 2769; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('cadbklij->abcdijkl',D30)
    
    del D30
    
    E30 = np.zeros([nvir, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 2770; Tree Level  2; Scaling  6/ 6 Result_size  4/ 4
    E30 += oe.contract('abeijm,mcdekl->abcdijkl',T3,I21, optimize='optimal')
    
    del I21
    
    #Contraction 2771; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('abcdijkl->abcdijkl',E30)
    
    #Contraction 2772; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('acbdijkl->abcdijkl',E30)
    
    #Contraction 2773; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('adbcijkl->abcdijkl',E30)
    
    #Contraction 2774; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('bcadijkl->abcdijkl',E30)
    
    #Contraction 2775; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('bdacijkl->abcdijkl',E30)
    
    #Contraction 2776; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('cdabijkl->abcdijkl',E30)
    
    #Contraction 2777; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('abcdikjl->abcdijkl',E30)
    
    #Contraction 2778; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('acbdikjl->abcdijkl',E30)
    
    #Contraction 2779; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('adbcikjl->abcdijkl',E30)
    
    #Contraction 2780; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('bcadikjl->abcdijkl',E30)
    
    #Contraction 2781; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('bdacikjl->abcdijkl',E30)
    
    #Contraction 2782; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('cdabikjl->abcdijkl',E30)
    
    #Contraction 2783; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('abcdiljk->abcdijkl',E30)
    
    #Contraction 2784; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('acbdiljk->abcdijkl',E30)
    
    #Contraction 2785; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('adbciljk->abcdijkl',E30)
    
    #Contraction 2786; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('bcadiljk->abcdijkl',E30)
    
    #Contraction 2787; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('bdaciljk->abcdijkl',E30)
    
    #Contraction 2788; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('cdabiljk->abcdijkl',E30)
    
    #Contraction 2789; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('abcdjkil->abcdijkl',E30)
    
    #Contraction 2790; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('acbdjkil->abcdijkl',E30)
    
    #Contraction 2791; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('adbcjkil->abcdijkl',E30)
    
    #Contraction 2792; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('bcadjkil->abcdijkl',E30)
    
    #Contraction 2793; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('bdacjkil->abcdijkl',E30)
    
    #Contraction 2794; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('cdabjkil->abcdijkl',E30)
    
    #Contraction 2795; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('abcdjlik->abcdijkl',E30)
    
    #Contraction 2796; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('acbdjlik->abcdijkl',E30)
    
    #Contraction 2797; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('adbcjlik->abcdijkl',E30)
    
    #Contraction 2798; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('bcadjlik->abcdijkl',E30)
    
    #Contraction 2799; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('bdacjlik->abcdijkl',E30)
    
    #Contraction 2800; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('cdabjlik->abcdijkl',E30)
    
    #Contraction 2801; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('abcdklij->abcdijkl',E30)
    
    #Contraction 2802; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('acbdklij->abcdijkl',E30)
    
    #Contraction 2803; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('adbcklij->abcdijkl',E30)
    
    #Contraction 2804; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('bcadklij->abcdijkl',E30)
    
    #Contraction 2805; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('bdacklij->abcdijkl',E30)
    
    #Contraction 2806; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('cdabklij->abcdijkl',E30)
    
    del E30
    
    G30 = np.zeros([nvir, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 2807; Tree Level  2; Scaling  6/ 4 Result_size  4/ 4
    G30 += oe.contract('bcjm,amdikl->bcadjikl',T2,D14, optimize='optimal')
    
    del D14
    
    #Contraction 2808; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('bcadjikl->abcdijkl',G30)
    
    #Contraction 2809; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('bdacjikl->abcdijkl',G30)
    
    #Contraction 2810; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('cdabjikl->abcdijkl',G30)
    
    #Contraction 2811; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('acbdjikl->abcdijkl',G30)
    
    #Contraction 2812; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('adbcjikl->abcdijkl',G30)
    
    #Contraction 2813; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('cdbajikl->abcdijkl',G30)
    
    #Contraction 2814; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('abcdjikl->abcdijkl',G30)
    
    #Contraction 2815; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('adcbjikl->abcdijkl',G30)
    
    #Contraction 2816; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('bdcajikl->abcdijkl',G30)
    
    #Contraction 2817; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('abdcjikl->abcdijkl',G30)
    
    #Contraction 2818; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('acdbjikl->abcdijkl',G30)
    
    #Contraction 2819; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('bcdajikl->abcdijkl',G30)
    
    #Contraction 2820; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('bcadijkl->abcdijkl',G30)
    
    #Contraction 2821; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('bdacijkl->abcdijkl',G30)
    
    #Contraction 2822; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('cdabijkl->abcdijkl',G30)
    
    #Contraction 2823; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('acbdijkl->abcdijkl',G30)
    
    #Contraction 2824; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('adbcijkl->abcdijkl',G30)
    
    #Contraction 2825; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('cdbaijkl->abcdijkl',G30)
    
    #Contraction 2826; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('abcdijkl->abcdijkl',G30)
    
    #Contraction 2827; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('adcbijkl->abcdijkl',G30)
    
    #Contraction 2828; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('bdcaijkl->abcdijkl',G30)
    
    #Contraction 2829; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('abdcijkl->abcdijkl',G30)
    
    #Contraction 2830; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('acdbijkl->abcdijkl',G30)
    
    #Contraction 2831; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('bcdaijkl->abcdijkl',G30)
    
    #Contraction 2832; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('bcadkjil->abcdijkl',G30)
    
    #Contraction 2833; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('bdackjil->abcdijkl',G30)
    
    #Contraction 2834; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('cdabkjil->abcdijkl',G30)
    
    #Contraction 2835; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('acbdkjil->abcdijkl',G30)
    
    #Contraction 2836; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('adbckjil->abcdijkl',G30)
    
    #Contraction 2837; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('cdbakjil->abcdijkl',G30)
    
    #Contraction 2838; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('abcdkjil->abcdijkl',G30)
    
    #Contraction 2839; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('adcbkjil->abcdijkl',G30)
    
    #Contraction 2840; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('bdcakjil->abcdijkl',G30)
    
    #Contraction 2841; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('abdckjil->abcdijkl',G30)
    
    #Contraction 2842; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('acdbkjil->abcdijkl',G30)
    
    #Contraction 2843; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('bcdakjil->abcdijkl',G30)
    
    #Contraction 2844; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('bcadljik->abcdijkl',G30)
    
    #Contraction 2845; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('bdacljik->abcdijkl',G30)
    
    #Contraction 2846; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('cdabljik->abcdijkl',G30)
    
    #Contraction 2847; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('acbdljik->abcdijkl',G30)
    
    #Contraction 2848; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('adbcljik->abcdijkl',G30)
    
    #Contraction 2849; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('cdbaljik->abcdijkl',G30)
    
    #Contraction 2850; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('abcdljik->abcdijkl',G30)
    
    #Contraction 2851; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('adcbljik->abcdijkl',G30)
    
    #Contraction 2852; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('bdcaljik->abcdijkl',G30)
    
    #Contraction 2853; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('abdcljik->abcdijkl',G30)
    
    #Contraction 2854; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('acdbljik->abcdijkl',G30)
    
    #Contraction 2855; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('bcdaljik->abcdijkl',G30)
    
    del G30
    
    I30 = np.zeros([nvir, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 2856; Tree Level  2; Scaling  6/ 4 Result_size  4/ 4
    I30 += oe.contract('bm,cdamjkli->bcdajkli',T1,G21, optimize='optimal')
    
    del G21
    
    #Contraction 2857; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bcdajkli->abcdijkl',I30)
    
    #Contraction 2858; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('cbdajkli->abcdijkl',I30)
    
    #Contraction 2859; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('dbcajkli->abcdijkl',I30)
    
    #Contraction 2860; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('acdbjkli->abcdijkl',I30)
    
    #Contraction 2861; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('cadbjkli->abcdijkl',I30)
    
    #Contraction 2862; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('dacbjkli->abcdijkl',I30)
    
    #Contraction 2863; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('abdcjkli->abcdijkl',I30)
    
    #Contraction 2864; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('badcjkli->abcdijkl',I30)
    
    #Contraction 2865; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('dabcjkli->abcdijkl',I30)
    
    #Contraction 2866; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('abcdjkli->abcdijkl',I30)
    
    #Contraction 2867; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bacdjkli->abcdijkl',I30)
    
    #Contraction 2868; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('cabdjkli->abcdijkl',I30)
    
    #Contraction 2869; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bcdaiklj->abcdijkl',I30)
    
    #Contraction 2870; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('cbdaiklj->abcdijkl',I30)
    
    #Contraction 2871; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('dbcaiklj->abcdijkl',I30)
    
    #Contraction 2872; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('acdbiklj->abcdijkl',I30)
    
    #Contraction 2873; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('cadbiklj->abcdijkl',I30)
    
    #Contraction 2874; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('dacbiklj->abcdijkl',I30)
    
    #Contraction 2875; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('abdciklj->abcdijkl',I30)
    
    #Contraction 2876; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('badciklj->abcdijkl',I30)
    
    #Contraction 2877; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('dabciklj->abcdijkl',I30)
    
    #Contraction 2878; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('abcdiklj->abcdijkl',I30)
    
    #Contraction 2879; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bacdiklj->abcdijkl',I30)
    
    #Contraction 2880; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('cabdiklj->abcdijkl',I30)
    
    #Contraction 2881; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bcdaijlk->abcdijkl',I30)
    
    #Contraction 2882; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('cbdaijlk->abcdijkl',I30)
    
    #Contraction 2883; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('dbcaijlk->abcdijkl',I30)
    
    #Contraction 2884; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('acdbijlk->abcdijkl',I30)
    
    #Contraction 2885; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('cadbijlk->abcdijkl',I30)
    
    #Contraction 2886; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('dacbijlk->abcdijkl',I30)
    
    #Contraction 2887; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('abdcijlk->abcdijkl',I30)
    
    #Contraction 2888; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('badcijlk->abcdijkl',I30)
    
    #Contraction 2889; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('dabcijlk->abcdijkl',I30)
    
    #Contraction 2890; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('abcdijlk->abcdijkl',I30)
    
    #Contraction 2891; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bacdijlk->abcdijkl',I30)
    
    #Contraction 2892; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('cabdijlk->abcdijkl',I30)
    
    #Contraction 2893; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bcdaijkl->abcdijkl',I30)
    
    #Contraction 2894; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('cbdaijkl->abcdijkl',I30)
    
    #Contraction 2895; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('dbcaijkl->abcdijkl',I30)
    
    #Contraction 2896; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('acdbijkl->abcdijkl',I30)
    
    #Contraction 2897; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('cadbijkl->abcdijkl',I30)
    
    #Contraction 2898; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('dacbijkl->abcdijkl',I30)
    
    #Contraction 2899; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('abdcijkl->abcdijkl',I30)
    
    #Contraction 2900; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('badcijkl->abcdijkl',I30)
    
    #Contraction 2901; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('dabcijkl->abcdijkl',I30)
    
    #Contraction 2902; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('abcdijkl->abcdijkl',I30)
    
    #Contraction 2903; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bacdijkl->abcdijkl',I30)
    
    #Contraction 2904; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('cabdijkl->abcdijkl',I30)
    
    del I30
    
    J30 = np.zeros([nvir, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 2905; Tree Level  2; Scaling  6/ 4 Result_size  4/ 4
    J30 += oe.contract('abmj,cdmkli->abcdjkli',T2,J12, optimize='optimal')
    
    del J12
    
    #Contraction 2906; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('abcdjkli->abcdijkl',J30)
    
    #Contraction 2907; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('acbdjkli->abcdijkl',J30)
    
    #Contraction 2908; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('adbcjkli->abcdijkl',J30)
    
    #Contraction 2909; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bcadjkli->abcdijkl',J30)
    
    #Contraction 2910; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bdacjkli->abcdijkl',J30)
    
    #Contraction 2911; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('cdabjkli->abcdijkl',J30)
    
    #Contraction 2912; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('abcdkjli->abcdijkl',J30)
    
    #Contraction 2913; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('acbdkjli->abcdijkl',J30)
    
    #Contraction 2914; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('adbckjli->abcdijkl',J30)
    
    #Contraction 2915; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bcadkjli->abcdijkl',J30)
    
    #Contraction 2916; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bdackjli->abcdijkl',J30)
    
    #Contraction 2917; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('cdabkjli->abcdijkl',J30)
    
    #Contraction 2918; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('abcdljki->abcdijkl',J30)
    
    #Contraction 2919; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('acbdljki->abcdijkl',J30)
    
    #Contraction 2920; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('adbcljki->abcdijkl',J30)
    
    #Contraction 2921; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bcadljki->abcdijkl',J30)
    
    #Contraction 2922; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bdacljki->abcdijkl',J30)
    
    #Contraction 2923; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('cdabljki->abcdijkl',J30)
    
    #Contraction 2924; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('abcdiklj->abcdijkl',J30)
    
    #Contraction 2925; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('acbdiklj->abcdijkl',J30)
    
    #Contraction 2926; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('adbciklj->abcdijkl',J30)
    
    #Contraction 2927; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bcadiklj->abcdijkl',J30)
    
    #Contraction 2928; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bdaciklj->abcdijkl',J30)
    
    #Contraction 2929; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('cdabiklj->abcdijkl',J30)
    
    #Contraction 2930; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('abcdkilj->abcdijkl',J30)
    
    #Contraction 2931; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('acbdkilj->abcdijkl',J30)
    
    #Contraction 2932; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('adbckilj->abcdijkl',J30)
    
    #Contraction 2933; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bcadkilj->abcdijkl',J30)
    
    #Contraction 2934; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bdackilj->abcdijkl',J30)
    
    #Contraction 2935; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('cdabkilj->abcdijkl',J30)
    
    #Contraction 2936; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('abcdlikj->abcdijkl',J30)
    
    #Contraction 2937; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('acbdlikj->abcdijkl',J30)
    
    #Contraction 2938; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('adbclikj->abcdijkl',J30)
    
    #Contraction 2939; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bcadlikj->abcdijkl',J30)
    
    #Contraction 2940; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bdaclikj->abcdijkl',J30)
    
    #Contraction 2941; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('cdablikj->abcdijkl',J30)
    
    #Contraction 2942; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('abcdijlk->abcdijkl',J30)
    
    #Contraction 2943; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('acbdijlk->abcdijkl',J30)
    
    #Contraction 2944; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('adbcijlk->abcdijkl',J30)
    
    #Contraction 2945; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bcadijlk->abcdijkl',J30)
    
    #Contraction 2946; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bdacijlk->abcdijkl',J30)
    
    #Contraction 2947; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('cdabijlk->abcdijkl',J30)
    
    #Contraction 2948; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('abcdjilk->abcdijkl',J30)
    
    #Contraction 2949; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('acbdjilk->abcdijkl',J30)
    
    #Contraction 2950; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('adbcjilk->abcdijkl',J30)
    
    #Contraction 2951; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bcadjilk->abcdijkl',J30)
    
    #Contraction 2952; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bdacjilk->abcdijkl',J30)
    
    #Contraction 2953; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('cdabjilk->abcdijkl',J30)
    
    #Contraction 2954; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('abcdlijk->abcdijkl',J30)
    
    #Contraction 2955; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('acbdlijk->abcdijkl',J30)
    
    #Contraction 2956; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('adbclijk->abcdijkl',J30)
    
    #Contraction 2957; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bcadlijk->abcdijkl',J30)
    
    #Contraction 2958; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bdaclijk->abcdijkl',J30)
    
    #Contraction 2959; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('cdablijk->abcdijkl',J30)
    
    #Contraction 2960; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('abcdijkl->abcdijkl',J30)
    
    #Contraction 2961; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('acbdijkl->abcdijkl',J30)
    
    #Contraction 2962; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('adbcijkl->abcdijkl',J30)
    
    #Contraction 2963; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bcadijkl->abcdijkl',J30)
    
    #Contraction 2964; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bdacijkl->abcdijkl',J30)
    
    #Contraction 2965; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('cdabijkl->abcdijkl',J30)
    
    #Contraction 2966; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('abcdjikl->abcdijkl',J30)
    
    #Contraction 2967; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('acbdjikl->abcdijkl',J30)
    
    #Contraction 2968; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('adbcjikl->abcdijkl',J30)
    
    #Contraction 2969; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bcadjikl->abcdijkl',J30)
    
    #Contraction 2970; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bdacjikl->abcdijkl',J30)
    
    #Contraction 2971; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('cdabjikl->abcdijkl',J30)
    
    #Contraction 2972; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('abcdkijl->abcdijkl',J30)
    
    #Contraction 2973; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('acbdkijl->abcdijkl',J30)
    
    #Contraction 2974; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('adbckijl->abcdijkl',J30)
    
    #Contraction 2975; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bcadkijl->abcdijkl',J30)
    
    #Contraction 2976; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bdackijl->abcdijkl',J30)
    
    #Contraction 2977; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('cdabkijl->abcdijkl',J30)
    
    del J30
    
    M30 = np.zeros([nvir, nvir, nocc, nvir, nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 2978; Tree Level  3; Scaling  5/ 5 Result_size  5/ 3
    M30 += oe.contract('fcdikl,mbfj->cdmbiklj',T3,X21, optimize='optimal')
    
    del X21
    
    X30 = np.zeros([nvir, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 2979; Tree Level  2; Scaling  6/ 4 Result_size  4/ 4
    X30 += oe.contract('am,cdmbiklj->acdbiklj',T1,M30, optimize='optimal')
    
    del M30
    
    #Contraction 2980; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('acdbiklj->abcdijkl',X30)
    
    #Contraction 2981; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('abdciklj->abcdijkl',X30)
    
    #Contraction 2982; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('abcdiklj->abcdijkl',X30)
    
    #Contraction 2983; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bcdaiklj->abcdijkl',X30)
    
    #Contraction 2984; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('badciklj->abcdijkl',X30)
    
    #Contraction 2985; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bacdiklj->abcdijkl',X30)
    
    #Contraction 2986; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('cbdaiklj->abcdijkl',X30)
    
    #Contraction 2987; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('cadbiklj->abcdijkl',X30)
    
    #Contraction 2988; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('cabdiklj->abcdijkl',X30)
    
    #Contraction 2989; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('dbcaiklj->abcdijkl',X30)
    
    #Contraction 2990; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('dacbiklj->abcdijkl',X30)
    
    #Contraction 2991; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('dabciklj->abcdijkl',X30)
    
    #Contraction 2992; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('acdbjkli->abcdijkl',X30)
    
    #Contraction 2993; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('abdcjkli->abcdijkl',X30)
    
    #Contraction 2994; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('abcdjkli->abcdijkl',X30)
    
    #Contraction 2995; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bcdajkli->abcdijkl',X30)
    
    #Contraction 2996; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('badcjkli->abcdijkl',X30)
    
    #Contraction 2997; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bacdjkli->abcdijkl',X30)
    
    #Contraction 2998; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('cbdajkli->abcdijkl',X30)
    
    #Contraction 2999; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('cadbjkli->abcdijkl',X30)
    
    #Contraction 3000; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('cabdjkli->abcdijkl',X30)
    
    #Contraction 3001; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('dbcajkli->abcdijkl',X30)
    
    #Contraction 3002; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('dacbjkli->abcdijkl',X30)
    
    #Contraction 3003; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('dabcjkli->abcdijkl',X30)
    
    #Contraction 3004; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('acdbjilk->abcdijkl',X30)
    
    #Contraction 3005; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('abdcjilk->abcdijkl',X30)
    
    #Contraction 3006; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('abcdjilk->abcdijkl',X30)
    
    #Contraction 3007; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bcdajilk->abcdijkl',X30)
    
    #Contraction 3008; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('badcjilk->abcdijkl',X30)
    
    #Contraction 3009; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bacdjilk->abcdijkl',X30)
    
    #Contraction 3010; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('cbdajilk->abcdijkl',X30)
    
    #Contraction 3011; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('cadbjilk->abcdijkl',X30)
    
    #Contraction 3012; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('cabdjilk->abcdijkl',X30)
    
    #Contraction 3013; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('dbcajilk->abcdijkl',X30)
    
    #Contraction 3014; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('dacbjilk->abcdijkl',X30)
    
    #Contraction 3015; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('dabcjilk->abcdijkl',X30)
    
    #Contraction 3016; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('acdbjikl->abcdijkl',X30)
    
    #Contraction 3017; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('abdcjikl->abcdijkl',X30)
    
    #Contraction 3018; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('abcdjikl->abcdijkl',X30)
    
    #Contraction 3019; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bcdajikl->abcdijkl',X30)
    
    #Contraction 3020; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('badcjikl->abcdijkl',X30)
    
    #Contraction 3021; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bacdjikl->abcdijkl',X30)
    
    #Contraction 3022; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('cbdajikl->abcdijkl',X30)
    
    #Contraction 3023; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('cadbjikl->abcdijkl',X30)
    
    #Contraction 3024; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('cabdjikl->abcdijkl',X30)
    
    #Contraction 3025; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('dbcajikl->abcdijkl',X30)
    
    #Contraction 3026; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('dacbjikl->abcdijkl',X30)
    
    #Contraction 3027; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('dabcjikl->abcdijkl',X30)
    
    del X30
    
    Y30 = np.zeros([nocc, nocc, nvir, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 3028; Tree Level  3; Scaling  5/ 5 Result_size  5/ 1
    Y30 += oe.contract('mnef,aefijk->mnaijk',V7,T3, optimize='optimal')
    
    A31 = np.zeros([nvir, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 3029; Tree Level  2; Scaling  4/ 6 Result_size  4/ 4
    A31 += oe.contract('abeijk,cdel->abcdijkl',T3,I19, optimize='optimal')
    
    del I19
    
    #Contraction 3030; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('abcdijkl->abcdijkl',A31)
    
    #Contraction 3031; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('acbdijkl->abcdijkl',A31)
    
    #Contraction 3032; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('adbcijkl->abcdijkl',A31)
    
    #Contraction 3033; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('bcadijkl->abcdijkl',A31)
    
    #Contraction 3034; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('bdacijkl->abcdijkl',A31)
    
    #Contraction 3035; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('cdabijkl->abcdijkl',A31)
    
    #Contraction 3036; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('abcdijlk->abcdijkl',A31)
    
    #Contraction 3037; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('acbdijlk->abcdijkl',A31)
    
    #Contraction 3038; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('adbcijlk->abcdijkl',A31)
    
    #Contraction 3039; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('bcadijlk->abcdijkl',A31)
    
    #Contraction 3040; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('bdacijlk->abcdijkl',A31)
    
    #Contraction 3041; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('cdabijlk->abcdijkl',A31)
    
    #Contraction 3042; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('abcdiklj->abcdijkl',A31)
    
    #Contraction 3043; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('acbdiklj->abcdijkl',A31)
    
    #Contraction 3044; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('adbciklj->abcdijkl',A31)
    
    #Contraction 3045; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('bcadiklj->abcdijkl',A31)
    
    #Contraction 3046; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('bdaciklj->abcdijkl',A31)
    
    #Contraction 3047; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('cdabiklj->abcdijkl',A31)
    
    #Contraction 3048; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('abcdjkli->abcdijkl',A31)
    
    #Contraction 3049; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('acbdjkli->abcdijkl',A31)
    
    #Contraction 3050; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('adbcjkli->abcdijkl',A31)
    
    #Contraction 3051; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('bcadjkli->abcdijkl',A31)
    
    #Contraction 3052; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('bdacjkli->abcdijkl',A31)
    
    #Contraction 3053; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('cdabjkli->abcdijkl',A31)
    
    del A31
    
    D31 = np.zeros([nvir, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 3054; Tree Level  2; Scaling  6/ 4 Result_size  4/ 4
    D31 += oe.contract('amij,cdbklm->acdbijkl',V2,T3, optimize='optimal')
    
    #del V2
    
    #Contraction 3055; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('acdbijkl->abcdijkl',D31)
    
    #Contraction 3056; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('cadbijkl->abcdijkl',D31)
    
    #Contraction 3057; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('dacbijkl->abcdijkl',D31)
    
    #Contraction 3058; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bacdijkl->abcdijkl',D31)
    
    #Contraction 3059; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('acdbikjl->abcdijkl',D31)
    
    #Contraction 3060; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('cadbikjl->abcdijkl',D31)
    
    #Contraction 3061; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('dacbikjl->abcdijkl',D31)
    
    #Contraction 3062; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bacdikjl->abcdijkl',D31)
    
    #Contraction 3063; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('acdbiljk->abcdijkl',D31)
    
    #Contraction 3064; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('cadbiljk->abcdijkl',D31)
    
    #Contraction 3065; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('dacbiljk->abcdijkl',D31)
    
    #Contraction 3066; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bacdiljk->abcdijkl',D31)
    
    #Contraction 3067; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('acdbjkil->abcdijkl',D31)
    
    #Contraction 3068; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('cadbjkil->abcdijkl',D31)
    
    #Contraction 3069; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('dacbjkil->abcdijkl',D31)
    
    #Contraction 3070; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bacdjkil->abcdijkl',D31)
    
    #Contraction 3071; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('acdbjlik->abcdijkl',D31)
    
    #Contraction 3072; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('cadbjlik->abcdijkl',D31)
    
    #Contraction 3073; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('dacbjlik->abcdijkl',D31)
    
    #Contraction 3074; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bacdjlik->abcdijkl',D31)
    
    #Contraction 3075; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('acdbklij->abcdijkl',D31)
    
    #Contraction 3076; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('cadbklij->abcdijkl',D31)
    
    #Contraction 3077; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('dacbklij->abcdijkl',D31)
    
    #Contraction 3078; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bacdklij->abcdijkl',D31)
    
    del D31
    
    E31 = np.zeros([nvir, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 3079; Tree Level  2; Scaling  4/ 8 Result_size  4/ 4
    E31 += oe.contract('abef,cdefklij->abcdklij',V9,T4, optimize='optimal')
    
    #Contraction 3080; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('abcdklij->abcdijkl',E31)
    
    #Contraction 3081; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('acbdklij->abcdijkl',E31)
    
    #Contraction 3082; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('adbcklij->abcdijkl',E31)
    
    #Contraction 3083; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('bcadklij->abcdijkl',E31)
    
    #Contraction 3084; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('bdacklij->abcdijkl',E31)
    
    #Contraction 3085; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('cdabklij->abcdijkl',E31)
    
    del E31
    
    G31 = np.zeros([nvir, nvir, nvir, nocc], dtype=type_)
    
    #Contraction 3086; Tree Level  3; Scaling  5/ 3 Result_size  1/ 3
    G31 += oe.contract('abmn,mnfi->abfi',T2,I1, optimize='optimal')
    
    del I1
    
    I31 = np.zeros([nvir, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 3087; Tree Level  2; Scaling  4/ 6 Result_size  4/ 4
    I31 += oe.contract('fcdjkl,abfi->cdabjkli',T3,G31, optimize='optimal')
    
    del G31
    
    #Contraction 3088; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('cdabjkli->abcdijkl',I31)
    
    #Contraction 3089; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('bdacjkli->abcdijkl',I31)
    
    #Contraction 3090; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('bcadjkli->abcdijkl',I31)
    
    #Contraction 3091; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('adbcjkli->abcdijkl',I31)
    
    #Contraction 3092; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('acbdjkli->abcdijkl',I31)
    
    #Contraction 3093; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('abcdjkli->abcdijkl',I31)
    
    #Contraction 3094; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('cdabiklj->abcdijkl',I31)
    
    #Contraction 3095; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('bdaciklj->abcdijkl',I31)
    
    #Contraction 3096; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('bcadiklj->abcdijkl',I31)
    
    #Contraction 3097; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('adbciklj->abcdijkl',I31)
    
    #Contraction 3098; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('acbdiklj->abcdijkl',I31)
    
    #Contraction 3099; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('abcdiklj->abcdijkl',I31)
    
    #Contraction 3100; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('cdabijlk->abcdijkl',I31)
    
    #Contraction 3101; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('bdacijlk->abcdijkl',I31)
    
    #Contraction 3102; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('bcadijlk->abcdijkl',I31)
    
    #Contraction 3103; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('adbcijlk->abcdijkl',I31)
    
    #Contraction 3104; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('acbdijlk->abcdijkl',I31)
    
    #Contraction 3105; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('abcdijlk->abcdijkl',I31)
    
    #Contraction 3106; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('cdabijkl->abcdijkl',I31)
    
    #Contraction 3107; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('bdacijkl->abcdijkl',I31)
    
    #Contraction 3108; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('bcadijkl->abcdijkl',I31)
    
    #Contraction 3109; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('adbcijkl->abcdijkl',I31)
    
    #Contraction 3110; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('acbdijkl->abcdijkl',I31)
    
    #Contraction 3111; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('abcdijkl->abcdijkl',I31)
    
    del I31
    
    J31 = np.zeros([nvir, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 3112; Tree Level  2; Scaling  6/ 4 Result_size  4/ 4
    J31 += oe.contract('bcdnkl,anji->bcdaklji',T3,I7, optimize='optimal')
    
    del I7
    
    #Contraction 3113; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('bcdaklji->abcdijkl',J31)
    
    #Contraction 3114; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('acdbklji->abcdijkl',J31)
    
    #Contraction 3115; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('abdcklji->abcdijkl',J31)
    
    #Contraction 3116; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('abcdklji->abcdijkl',J31)
    
    #Contraction 3117; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('bcdajlki->abcdijkl',J31)
    
    #Contraction 3118; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('acdbjlki->abcdijkl',J31)
    
    #Contraction 3119; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('abdcjlki->abcdijkl',J31)
    
    #Contraction 3120; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('abcdjlki->abcdijkl',J31)
    
    #Contraction 3121; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('bcdajkli->abcdijkl',J31)
    
    #Contraction 3122; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('acdbjkli->abcdijkl',J31)
    
    #Contraction 3123; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('abdcjkli->abcdijkl',J31)
    
    #Contraction 3124; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('abcdjkli->abcdijkl',J31)
    
    #Contraction 3125; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('bcdaklij->abcdijkl',J31)
    
    #Contraction 3126; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('acdbklij->abcdijkl',J31)
    
    #Contraction 3127; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('abdcklij->abcdijkl',J31)
    
    #Contraction 3128; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('abcdklij->abcdijkl',J31)
    
    #Contraction 3129; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('bcdailkj->abcdijkl',J31)
    
    #Contraction 3130; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('acdbilkj->abcdijkl',J31)
    
    #Contraction 3131; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('abdcilkj->abcdijkl',J31)
    
    #Contraction 3132; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('abcdilkj->abcdijkl',J31)
    
    #Contraction 3133; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('bcdaiklj->abcdijkl',J31)
    
    #Contraction 3134; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('acdbiklj->abcdijkl',J31)
    
    #Contraction 3135; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('abdciklj->abcdijkl',J31)
    
    #Contraction 3136; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('abcdiklj->abcdijkl',J31)
    
    #Contraction 3137; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('bcdajlik->abcdijkl',J31)
    
    #Contraction 3138; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('acdbjlik->abcdijkl',J31)
    
    #Contraction 3139; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('abdcjlik->abcdijkl',J31)
    
    #Contraction 3140; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('abcdjlik->abcdijkl',J31)
    
    #Contraction 3141; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('bcdailjk->abcdijkl',J31)
    
    #Contraction 3142; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('acdbiljk->abcdijkl',J31)
    
    #Contraction 3143; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('abdciljk->abcdijkl',J31)
    
    #Contraction 3144; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('abcdiljk->abcdijkl',J31)
    
    #Contraction 3145; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('bcdaijlk->abcdijkl',J31)
    
    #Contraction 3146; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('acdbijlk->abcdijkl',J31)
    
    #Contraction 3147; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('abdcijlk->abcdijkl',J31)
    
    #Contraction 3148; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('abcdijlk->abcdijkl',J31)
    
    #Contraction 3149; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('bcdajkil->abcdijkl',J31)
    
    #Contraction 3150; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('acdbjkil->abcdijkl',J31)
    
    #Contraction 3151; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('abdcjkil->abcdijkl',J31)
    
    #Contraction 3152; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('abcdjkil->abcdijkl',J31)
    
    #Contraction 3153; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('bcdaikjl->abcdijkl',J31)
    
    #Contraction 3154; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('acdbikjl->abcdijkl',J31)
    
    #Contraction 3155; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('abdcikjl->abcdijkl',J31)
    
    #Contraction 3156; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('abcdikjl->abcdijkl',J31)
    
    #Contraction 3157; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('bcdaijkl->abcdijkl',J31)
    
    #Contraction 3158; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('acdbijkl->abcdijkl',J31)
    
    #Contraction 3159; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('abdcijkl->abcdijkl',J31)
    
    #Contraction 3160; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('abcdijkl->abcdijkl',J31)
    
    del J31
    
    M31 = np.zeros([nvir, nvir, nvir, nvir, nocc, nocc], dtype=type_)
    
    #Contraction 3161; Tree Level  3; Scaling  2/ 6 Result_size  2/ 4
    M31 += oe.contract('abef,fdjl->abdejl',V9,T2, optimize='optimal')
    
    #del V9
    
    X31 = np.zeros([nvir, nvir, nvir, nocc], dtype=type_)
    
    #Contraction 3162; Tree Level  3; Scaling  3/ 5 Result_size  1/ 3
    X31 += oe.contract('amef,bejm->abfj',V8,T2, optimize='optimal')
    
    #del V8
    
    Y31 = np.zeros([nvir, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 3163; Tree Level  2; Scaling  4/ 6 Result_size  4/ 4
    Y31 += oe.contract('fcdikl,abfj->cdabiklj',T3,X31, optimize='optimal')
    
    del X31
    
    #Contraction 3164; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('cdabiklj->abcdijkl',Y31)
    
    #Contraction 3165; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bdaciklj->abcdijkl',Y31)
    
    #Contraction 3166; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bcadiklj->abcdijkl',Y31)
    
    #Contraction 3167; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('cdbaiklj->abcdijkl',Y31)
    
    #Contraction 3168; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('adbciklj->abcdijkl',Y31)
    
    #Contraction 3169; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('acbdiklj->abcdijkl',Y31)
    
    #Contraction 3170; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bdcaiklj->abcdijkl',Y31)
    
    #Contraction 3171; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('adcbiklj->abcdijkl',Y31)
    
    #Contraction 3172; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('abcdiklj->abcdijkl',Y31)
    
    #Contraction 3173; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bcdaiklj->abcdijkl',Y31)
    
    #Contraction 3174; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('acdbiklj->abcdijkl',Y31)
    
    #Contraction 3175; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('abdciklj->abcdijkl',Y31)
    
    #Contraction 3176; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('cdabjkli->abcdijkl',Y31)
    
    #Contraction 3177; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bdacjkli->abcdijkl',Y31)
    
    #Contraction 3178; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bcadjkli->abcdijkl',Y31)
    
    #Contraction 3179; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('cdbajkli->abcdijkl',Y31)
    
    #Contraction 3180; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('adbcjkli->abcdijkl',Y31)
    
    #Contraction 3181; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('acbdjkli->abcdijkl',Y31)
    
    #Contraction 3182; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bdcajkli->abcdijkl',Y31)
    
    #Contraction 3183; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('adcbjkli->abcdijkl',Y31)
    
    #Contraction 3184; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('abcdjkli->abcdijkl',Y31)
    
    #Contraction 3185; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bcdajkli->abcdijkl',Y31)
    
    #Contraction 3186; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('acdbjkli->abcdijkl',Y31)
    
    #Contraction 3187; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('abdcjkli->abcdijkl',Y31)
    
    #Contraction 3188; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('cdabjilk->abcdijkl',Y31)
    
    #Contraction 3189; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bdacjilk->abcdijkl',Y31)
    
    #Contraction 3190; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bcadjilk->abcdijkl',Y31)
    
    #Contraction 3191; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('cdbajilk->abcdijkl',Y31)
    
    #Contraction 3192; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('adbcjilk->abcdijkl',Y31)
    
    #Contraction 3193; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('acbdjilk->abcdijkl',Y31)
    
    #Contraction 3194; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bdcajilk->abcdijkl',Y31)
    
    #Contraction 3195; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('adcbjilk->abcdijkl',Y31)
    
    #Contraction 3196; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('abcdjilk->abcdijkl',Y31)
    
    #Contraction 3197; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bcdajilk->abcdijkl',Y31)
    
    #Contraction 3198; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('acdbjilk->abcdijkl',Y31)
    
    #Contraction 3199; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('abdcjilk->abcdijkl',Y31)
    
    #Contraction 3200; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('cdabjikl->abcdijkl',Y31)
    
    #Contraction 3201; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bdacjikl->abcdijkl',Y31)
    
    #Contraction 3202; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bcadjikl->abcdijkl',Y31)
    
    #Contraction 3203; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('cdbajikl->abcdijkl',Y31)
    
    #Contraction 3204; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('adbcjikl->abcdijkl',Y31)
    
    #Contraction 3205; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('acbdjikl->abcdijkl',Y31)
    
    #Contraction 3206; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bdcajikl->abcdijkl',Y31)
    
    #Contraction 3207; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('adcbjikl->abcdijkl',Y31)
    
    #Contraction 3208; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('abcdjikl->abcdijkl',Y31)
    
    #Contraction 3209; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bcdajikl->abcdijkl',Y31)
    
    #Contraction 3210; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('acdbjikl->abcdijkl',Y31)
    
    #Contraction 3211; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('abdcjikl->abcdijkl',Y31)
    
    del Y31
    
    A32 = np.zeros([nvir, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 3212; Tree Level  2; Scaling  8/ 4 Result_size  4/ 4
    A32 += oe.contract('mnij,cdabklmn->cdabijkl',V1,T4, optimize='optimal')
    
    #del V1
    
    #Contraction 3213; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('cdabijkl->abcdijkl',A32)
    
    #Contraction 3214; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('cdabikjl->abcdijkl',A32)
    
    #Contraction 3215; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('cdabiljk->abcdijkl',A32)
    
    #Contraction 3216; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('cdabjkil->abcdijkl',A32)
    
    #Contraction 3217; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('cdabjlik->abcdijkl',A32)
    
    #Contraction 3218; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('cdabklij->abcdijkl',A32)
    
    del A32
    
    D32 = np.zeros([nvir, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 3219; Tree Level  2; Scaling  4/ 6 Result_size  4/ 4
    D32 += oe.contract('beji,acdekl->bacdjikl',T2,J23, optimize='optimal')
    
    del J23
    
    #Contraction 3220; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bacdjikl->abcdijkl',D32)
    
    #Contraction 3221; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('cabdjikl->abcdijkl',D32)
    
    #Contraction 3222; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('dabcjikl->abcdijkl',D32)
    
    #Contraction 3223; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('abcdjikl->abcdijkl',D32)
    
    #Contraction 3224; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('cbadjikl->abcdijkl',D32)
    
    #Contraction 3225; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('dbacjikl->abcdijkl',D32)
    
    #Contraction 3226; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('acbdjikl->abcdijkl',D32)
    
    #Contraction 3227; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bcadjikl->abcdijkl',D32)
    
    #Contraction 3228; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('dcabjikl->abcdijkl',D32)
    
    #Contraction 3229; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('adbcjikl->abcdijkl',D32)
    
    #Contraction 3230; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bdacjikl->abcdijkl',D32)
    
    #Contraction 3231; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('cdabjikl->abcdijkl',D32)
    
    #Contraction 3232; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bacdjkil->abcdijkl',D32)
    
    #Contraction 3233; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('cabdjkil->abcdijkl',D32)
    
    #Contraction 3234; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('dabcjkil->abcdijkl',D32)
    
    #Contraction 3235; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('abcdjkil->abcdijkl',D32)
    
    #Contraction 3236; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('cbadjkil->abcdijkl',D32)
    
    #Contraction 3237; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('dbacjkil->abcdijkl',D32)
    
    #Contraction 3238; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('acbdjkil->abcdijkl',D32)
    
    #Contraction 3239; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bcadjkil->abcdijkl',D32)
    
    #Contraction 3240; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('dcabjkil->abcdijkl',D32)
    
    #Contraction 3241; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('adbcjkil->abcdijkl',D32)
    
    #Contraction 3242; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bdacjkil->abcdijkl',D32)
    
    #Contraction 3243; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('cdabjkil->abcdijkl',D32)
    
    #Contraction 3244; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bacdjlik->abcdijkl',D32)
    
    #Contraction 3245; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('cabdjlik->abcdijkl',D32)
    
    #Contraction 3246; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('dabcjlik->abcdijkl',D32)
    
    #Contraction 3247; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('abcdjlik->abcdijkl',D32)
    
    #Contraction 3248; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('cbadjlik->abcdijkl',D32)
    
    #Contraction 3249; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('dbacjlik->abcdijkl',D32)
    
    #Contraction 3250; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('acbdjlik->abcdijkl',D32)
    
    #Contraction 3251; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bcadjlik->abcdijkl',D32)
    
    #Contraction 3252; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('dcabjlik->abcdijkl',D32)
    
    #Contraction 3253; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('adbcjlik->abcdijkl',D32)
    
    #Contraction 3254; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bdacjlik->abcdijkl',D32)
    
    #Contraction 3255; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('cdabjlik->abcdijkl',D32)
    
    #Contraction 3256; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bacdikjl->abcdijkl',D32)
    
    #Contraction 3257; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('cabdikjl->abcdijkl',D32)
    
    #Contraction 3258; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('dabcikjl->abcdijkl',D32)
    
    #Contraction 3259; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('abcdikjl->abcdijkl',D32)
    
    #Contraction 3260; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('cbadikjl->abcdijkl',D32)
    
    #Contraction 3261; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('dbacikjl->abcdijkl',D32)
    
    #Contraction 3262; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('acbdikjl->abcdijkl',D32)
    
    #Contraction 3263; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bcadikjl->abcdijkl',D32)
    
    #Contraction 3264; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('dcabikjl->abcdijkl',D32)
    
    #Contraction 3265; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('adbcikjl->abcdijkl',D32)
    
    #Contraction 3266; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bdacikjl->abcdijkl',D32)
    
    #Contraction 3267; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('cdabikjl->abcdijkl',D32)
    
    #Contraction 3268; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bacdiljk->abcdijkl',D32)
    
    #Contraction 3269; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('cabdiljk->abcdijkl',D32)
    
    #Contraction 3270; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('dabciljk->abcdijkl',D32)
    
    #Contraction 3271; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('abcdiljk->abcdijkl',D32)
    
    #Contraction 3272; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('cbadiljk->abcdijkl',D32)
    
    #Contraction 3273; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('dbaciljk->abcdijkl',D32)
    
    #Contraction 3274; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('acbdiljk->abcdijkl',D32)
    
    #Contraction 3275; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bcadiljk->abcdijkl',D32)
    
    #Contraction 3276; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('dcabiljk->abcdijkl',D32)
    
    #Contraction 3277; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('adbciljk->abcdijkl',D32)
    
    #Contraction 3278; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bdaciljk->abcdijkl',D32)
    
    #Contraction 3279; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('cdabiljk->abcdijkl',D32)
    
    #Contraction 3280; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bacdklji->abcdijkl',D32)
    
    #Contraction 3281; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('cabdklji->abcdijkl',D32)
    
    #Contraction 3282; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('dabcklji->abcdijkl',D32)
    
    #Contraction 3283; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('abcdklji->abcdijkl',D32)
    
    #Contraction 3284; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('cbadklji->abcdijkl',D32)
    
    #Contraction 3285; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('dbacklji->abcdijkl',D32)
    
    #Contraction 3286; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('acbdklji->abcdijkl',D32)
    
    #Contraction 3287; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bcadklji->abcdijkl',D32)
    
    #Contraction 3288; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('dcabklji->abcdijkl',D32)
    
    #Contraction 3289; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('adbcklji->abcdijkl',D32)
    
    #Contraction 3290; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bdacklji->abcdijkl',D32)
    
    #Contraction 3291; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('cdabklji->abcdijkl',D32)
    
    del D32
    
    E32 = np.zeros([nvir, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 3292; Tree Level  2; Scaling  6/ 4 Result_size  4/ 4
    E32 += oe.contract('abcdmjkl,mi->abcdjkli',T4,Y1, optimize='optimal')
    
    del Y1
    
    #Contraction 3293; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('abcdjkli->abcdijkl',E32)
    
    #Contraction 3294; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('abcdiklj->abcdijkl',E32)
    
    #Contraction 3295; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('abcdijlk->abcdijkl',E32)
    
    #Contraction 3296; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('abcdijkl->abcdijkl',E32)
    
    del E32
    
    G32 = np.zeros([nvir, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 3297; Tree Level  2; Scaling  4/ 6 Result_size  4/ 4
    G32 += oe.contract('fbcdijkl,af->bcdaijkl',T4,A16, optimize='optimal')
    
    del A16
    
    #Contraction 3298; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('bcdaijkl->abcdijkl',G32)
    
    #Contraction 3299; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('acdbijkl->abcdijkl',G32)
    
    #Contraction 3300; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('abdcijkl->abcdijkl',G32)
    
    #Contraction 3301; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('abcdijkl->abcdijkl',G32)
    
    del G32
    
    I32 = np.zeros([nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 3302; Tree Level  3; Scaling  6/ 2 Result_size  4/ 2
    I32 += oe.contract('bdnl,mnji->bdmlji',T2,E7, optimize='optimal')
    
    del E7
    
    J32 = np.zeros([nvir, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 3303; Tree Level  2; Scaling  6/ 4 Result_size  4/ 4
    J32 += oe.contract('acmk,bdmlji->acbdklji',T2,I32, optimize='optimal')
    
    del I32
    
    #Contraction 3304; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('acbdklji->abcdijkl',J32)
    
    #Contraction 3305; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('abcdklji->abcdijkl',J32)
    
    #Contraction 3306; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('adcbklji->abcdijkl',J32)
    
    #Contraction 3307; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('cbadklji->abcdijkl',J32)
    
    #Contraction 3308; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('cdabklji->abcdijkl',J32)
    
    #Contraction 3309; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('bdacklji->abcdijkl',J32)
    
    #Contraction 3310; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('acbdlkji->abcdijkl',J32)
    
    #Contraction 3311; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('abcdlkji->abcdijkl',J32)
    
    #Contraction 3312; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('adcblkji->abcdijkl',J32)
    
    #Contraction 3313; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('cbadlkji->abcdijkl',J32)
    
    #Contraction 3314; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('cdablkji->abcdijkl',J32)
    
    #Contraction 3315; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('bdaclkji->abcdijkl',J32)
    
    #Contraction 3316; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('acbdjlki->abcdijkl',J32)
    
    #Contraction 3317; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('abcdjlki->abcdijkl',J32)
    
    #Contraction 3318; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('adcbjlki->abcdijkl',J32)
    
    #Contraction 3319; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('cbadjlki->abcdijkl',J32)
    
    #Contraction 3320; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('cdabjlki->abcdijkl',J32)
    
    #Contraction 3321; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('bdacjlki->abcdijkl',J32)
    
    #Contraction 3322; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('acbdljki->abcdijkl',J32)
    
    #Contraction 3323; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('abcdljki->abcdijkl',J32)
    
    #Contraction 3324; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('adcbljki->abcdijkl',J32)
    
    #Contraction 3325; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('cbadljki->abcdijkl',J32)
    
    #Contraction 3326; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('cdabljki->abcdijkl',J32)
    
    #Contraction 3327; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('bdacljki->abcdijkl',J32)
    
    #Contraction 3328; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('acbdjkli->abcdijkl',J32)
    
    #Contraction 3329; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('abcdjkli->abcdijkl',J32)
    
    #Contraction 3330; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('adcbjkli->abcdijkl',J32)
    
    #Contraction 3331; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('cbadjkli->abcdijkl',J32)
    
    #Contraction 3332; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('cdabjkli->abcdijkl',J32)
    
    #Contraction 3333; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('bdacjkli->abcdijkl',J32)
    
    #Contraction 3334; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('acbdkjli->abcdijkl',J32)
    
    #Contraction 3335; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('abcdkjli->abcdijkl',J32)
    
    #Contraction 3336; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('adcbkjli->abcdijkl',J32)
    
    #Contraction 3337; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('cbadkjli->abcdijkl',J32)
    
    #Contraction 3338; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('cdabkjli->abcdijkl',J32)
    
    #Contraction 3339; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('bdackjli->abcdijkl',J32)
    
    #Contraction 3340; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('acbdklij->abcdijkl',J32)
    
    #Contraction 3341; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('abcdklij->abcdijkl',J32)
    
    #Contraction 3342; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('adcbklij->abcdijkl',J32)
    
    #Contraction 3343; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('cbadklij->abcdijkl',J32)
    
    #Contraction 3344; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('cdabklij->abcdijkl',J32)
    
    #Contraction 3345; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('bdacklij->abcdijkl',J32)
    
    #Contraction 3346; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('acbdlkij->abcdijkl',J32)
    
    #Contraction 3347; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('abcdlkij->abcdijkl',J32)
    
    #Contraction 3348; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('adcblkij->abcdijkl',J32)
    
    #Contraction 3349; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('cbadlkij->abcdijkl',J32)
    
    #Contraction 3350; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('cdablkij->abcdijkl',J32)
    
    #Contraction 3351; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('bdaclkij->abcdijkl',J32)
    
    #Contraction 3352; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('acbdilkj->abcdijkl',J32)
    
    #Contraction 3353; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('abcdilkj->abcdijkl',J32)
    
    #Contraction 3354; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('adcbilkj->abcdijkl',J32)
    
    #Contraction 3355; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('cbadilkj->abcdijkl',J32)
    
    #Contraction 3356; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('cdabilkj->abcdijkl',J32)
    
    #Contraction 3357; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('bdacilkj->abcdijkl',J32)
    
    #Contraction 3358; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('acbdlikj->abcdijkl',J32)
    
    #Contraction 3359; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('abcdlikj->abcdijkl',J32)
    
    #Contraction 3360; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('adcblikj->abcdijkl',J32)
    
    #Contraction 3361; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('cbadlikj->abcdijkl',J32)
    
    #Contraction 3362; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('cdablikj->abcdijkl',J32)
    
    #Contraction 3363; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('bdaclikj->abcdijkl',J32)
    
    #Contraction 3364; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('acbdiklj->abcdijkl',J32)
    
    #Contraction 3365; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('abcdiklj->abcdijkl',J32)
    
    #Contraction 3366; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('adcbiklj->abcdijkl',J32)
    
    #Contraction 3367; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('cbadiklj->abcdijkl',J32)
    
    #Contraction 3368; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('cdabiklj->abcdijkl',J32)
    
    #Contraction 3369; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('bdaciklj->abcdijkl',J32)
    
    #Contraction 3370; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('acbdkilj->abcdijkl',J32)
    
    #Contraction 3371; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('abcdkilj->abcdijkl',J32)
    
    #Contraction 3372; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('adcbkilj->abcdijkl',J32)
    
    #Contraction 3373; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('cbadkilj->abcdijkl',J32)
    
    #Contraction 3374; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('cdabkilj->abcdijkl',J32)
    
    #Contraction 3375; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('bdackilj->abcdijkl',J32)
    
    #Contraction 3376; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('acbdjlik->abcdijkl',J32)
    
    #Contraction 3377; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('abcdjlik->abcdijkl',J32)
    
    #Contraction 3378; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('adcbjlik->abcdijkl',J32)
    
    #Contraction 3379; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('cbadjlik->abcdijkl',J32)
    
    #Contraction 3380; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('cdabjlik->abcdijkl',J32)
    
    #Contraction 3381; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('bdacjlik->abcdijkl',J32)
    
    #Contraction 3382; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('acbdljik->abcdijkl',J32)
    
    #Contraction 3383; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('abcdljik->abcdijkl',J32)
    
    #Contraction 3384; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('adcbljik->abcdijkl',J32)
    
    #Contraction 3385; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('cbadljik->abcdijkl',J32)
    
    #Contraction 3386; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('cdabljik->abcdijkl',J32)
    
    #Contraction 3387; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('bdacljik->abcdijkl',J32)
    
    #Contraction 3388; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('acbdiljk->abcdijkl',J32)
    
    #Contraction 3389; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('abcdiljk->abcdijkl',J32)
    
    #Contraction 3390; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('adcbiljk->abcdijkl',J32)
    
    #Contraction 3391; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('cbadiljk->abcdijkl',J32)
    
    #Contraction 3392; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('cdabiljk->abcdijkl',J32)
    
    #Contraction 3393; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('bdaciljk->abcdijkl',J32)
    
    #Contraction 3394; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('acbdlijk->abcdijkl',J32)
    
    #Contraction 3395; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('abcdlijk->abcdijkl',J32)
    
    #Contraction 3396; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('adcblijk->abcdijkl',J32)
    
    #Contraction 3397; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('cbadlijk->abcdijkl',J32)
    
    #Contraction 3398; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('cdablijk->abcdijkl',J32)
    
    #Contraction 3399; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('bdaclijk->abcdijkl',J32)
    
    #Contraction 3400; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('acbdijlk->abcdijkl',J32)
    
    #Contraction 3401; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('abcdijlk->abcdijkl',J32)
    
    #Contraction 3402; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('adcbijlk->abcdijkl',J32)
    
    #Contraction 3403; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('cbadijlk->abcdijkl',J32)
    
    #Contraction 3404; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('cdabijlk->abcdijkl',J32)
    
    #Contraction 3405; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('bdacijlk->abcdijkl',J32)
    
    #Contraction 3406; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('acbdjilk->abcdijkl',J32)
    
    #Contraction 3407; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('abcdjilk->abcdijkl',J32)
    
    #Contraction 3408; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('adcbjilk->abcdijkl',J32)
    
    #Contraction 3409; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('cbadjilk->abcdijkl',J32)
    
    #Contraction 3410; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('cdabjilk->abcdijkl',J32)
    
    #Contraction 3411; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('bdacjilk->abcdijkl',J32)
    
    #Contraction 3412; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('acbdjkil->abcdijkl',J32)
    
    #Contraction 3413; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('abcdjkil->abcdijkl',J32)
    
    #Contraction 3414; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('adcbjkil->abcdijkl',J32)
    
    #Contraction 3415; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('cbadjkil->abcdijkl',J32)
    
    #Contraction 3416; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('cdabjkil->abcdijkl',J32)
    
    #Contraction 3417; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('bdacjkil->abcdijkl',J32)
    
    #Contraction 3418; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('acbdkjil->abcdijkl',J32)
    
    #Contraction 3419; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('abcdkjil->abcdijkl',J32)
    
    #Contraction 3420; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('adcbkjil->abcdijkl',J32)
    
    #Contraction 3421; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('cbadkjil->abcdijkl',J32)
    
    #Contraction 3422; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('cdabkjil->abcdijkl',J32)
    
    #Contraction 3423; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('bdackjil->abcdijkl',J32)
    
    #Contraction 3424; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('acbdikjl->abcdijkl',J32)
    
    #Contraction 3425; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('abcdikjl->abcdijkl',J32)
    
    #Contraction 3426; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('adcbikjl->abcdijkl',J32)
    
    #Contraction 3427; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('cbadikjl->abcdijkl',J32)
    
    #Contraction 3428; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('cdabikjl->abcdijkl',J32)
    
    #Contraction 3429; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('bdacikjl->abcdijkl',J32)
    
    #Contraction 3430; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('acbdkijl->abcdijkl',J32)
    
    #Contraction 3431; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('abcdkijl->abcdijkl',J32)
    
    #Contraction 3432; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('adcbkijl->abcdijkl',J32)
    
    #Contraction 3433; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('cbadkijl->abcdijkl',J32)
    
    #Contraction 3434; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('cdabkijl->abcdijkl',J32)
    
    #Contraction 3435; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('bdackijl->abcdijkl',J32)
    
    #Contraction 3436; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('acbdijkl->abcdijkl',J32)
    
    #Contraction 3437; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('abcdijkl->abcdijkl',J32)
    
    #Contraction 3438; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('adcbijkl->abcdijkl',J32)
    
    #Contraction 3439; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('cbadijkl->abcdijkl',J32)
    
    #Contraction 3440; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('cdabijkl->abcdijkl',J32)
    
    #Contraction 3441; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('bdacijkl->abcdijkl',J32)
    
    #Contraction 3442; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('acbdjikl->abcdijkl',J32)
    
    #Contraction 3443; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('abcdjikl->abcdijkl',J32)
    
    #Contraction 3444; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('adcbjikl->abcdijkl',J32)
    
    #Contraction 3445; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('cbadjikl->abcdijkl',J32)
    
    #Contraction 3446; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('cdabjikl->abcdijkl',J32)
    
    #Contraction 3447; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('bdacjikl->abcdijkl',J32)
    
    del J32
    
    M32 = np.zeros([nvir, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 3448; Tree Level  2; Scaling  6/ 4 Result_size  4/ 4
    M32 += oe.contract('bdnl,ancijk->bdaclijk',T2,G22, optimize='optimal')
    
    del G22
    
    #Contraction 3449; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bdaclijk->abcdijkl',M32)
    
    #Contraction 3450; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('cdablijk->abcdijkl',M32)
    
    #Contraction 3451; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('cbadlijk->abcdijkl',M32)
    
    #Contraction 3452; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bdcalijk->abcdijkl',M32)
    
    #Contraction 3453; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('adcblijk->abcdijkl',M32)
    
    #Contraction 3454; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('abcdlijk->abcdijkl',M32)
    
    #Contraction 3455; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('cdbalijk->abcdijkl',M32)
    
    #Contraction 3456; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('adbclijk->abcdijkl',M32)
    
    #Contraction 3457; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('acbdlijk->abcdijkl',M32)
    
    #Contraction 3458; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('cbdalijk->abcdijkl',M32)
    
    #Contraction 3459; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('abdclijk->abcdijkl',M32)
    
    #Contraction 3460; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('acdblijk->abcdijkl',M32)
    
    #Contraction 3461; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bdackijl->abcdijkl',M32)
    
    #Contraction 3462; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('cdabkijl->abcdijkl',M32)
    
    #Contraction 3463; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('cbadkijl->abcdijkl',M32)
    
    #Contraction 3464; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bdcakijl->abcdijkl',M32)
    
    #Contraction 3465; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('adcbkijl->abcdijkl',M32)
    
    #Contraction 3466; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('abcdkijl->abcdijkl',M32)
    
    #Contraction 3467; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('cdbakijl->abcdijkl',M32)
    
    #Contraction 3468; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('adbckijl->abcdijkl',M32)
    
    #Contraction 3469; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('acbdkijl->abcdijkl',M32)
    
    #Contraction 3470; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('cbdakijl->abcdijkl',M32)
    
    #Contraction 3471; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('abdckijl->abcdijkl',M32)
    
    #Contraction 3472; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('acdbkijl->abcdijkl',M32)
    
    #Contraction 3473; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bdacjikl->abcdijkl',M32)
    
    #Contraction 3474; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('cdabjikl->abcdijkl',M32)
    
    #Contraction 3475; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('cbadjikl->abcdijkl',M32)
    
    #Contraction 3476; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bdcajikl->abcdijkl',M32)
    
    #Contraction 3477; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('adcbjikl->abcdijkl',M32)
    
    #Contraction 3478; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('abcdjikl->abcdijkl',M32)
    
    #Contraction 3479; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('cdbajikl->abcdijkl',M32)
    
    #Contraction 3480; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('adbcjikl->abcdijkl',M32)
    
    #Contraction 3481; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('acbdjikl->abcdijkl',M32)
    
    #Contraction 3482; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('cbdajikl->abcdijkl',M32)
    
    #Contraction 3483; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('abdcjikl->abcdijkl',M32)
    
    #Contraction 3484; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('acdbjikl->abcdijkl',M32)
    
    #Contraction 3485; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bdacljik->abcdijkl',M32)
    
    #Contraction 3486; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('cdabljik->abcdijkl',M32)
    
    #Contraction 3487; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('cbadljik->abcdijkl',M32)
    
    #Contraction 3488; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bdcaljik->abcdijkl',M32)
    
    #Contraction 3489; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('adcbljik->abcdijkl',M32)
    
    #Contraction 3490; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('abcdljik->abcdijkl',M32)
    
    #Contraction 3491; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('cdbaljik->abcdijkl',M32)
    
    #Contraction 3492; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('adbcljik->abcdijkl',M32)
    
    #Contraction 3493; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('acbdljik->abcdijkl',M32)
    
    #Contraction 3494; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('cbdaljik->abcdijkl',M32)
    
    #Contraction 3495; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('abdcljik->abcdijkl',M32)
    
    #Contraction 3496; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('acdbljik->abcdijkl',M32)
    
    #Contraction 3497; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bdackjil->abcdijkl',M32)
    
    #Contraction 3498; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('cdabkjil->abcdijkl',M32)
    
    #Contraction 3499; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('cbadkjil->abcdijkl',M32)
    
    #Contraction 3500; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bdcakjil->abcdijkl',M32)
    
    #Contraction 3501; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('adcbkjil->abcdijkl',M32)
    
    #Contraction 3502; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('abcdkjil->abcdijkl',M32)
    
    #Contraction 3503; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('cdbakjil->abcdijkl',M32)
    
    #Contraction 3504; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('adbckjil->abcdijkl',M32)
    
    #Contraction 3505; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('acbdkjil->abcdijkl',M32)
    
    #Contraction 3506; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('cbdakjil->abcdijkl',M32)
    
    #Contraction 3507; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('abdckjil->abcdijkl',M32)
    
    #Contraction 3508; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('acdbkjil->abcdijkl',M32)
    
    #Contraction 3509; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bdacijkl->abcdijkl',M32)
    
    #Contraction 3510; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('cdabijkl->abcdijkl',M32)
    
    #Contraction 3511; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('cbadijkl->abcdijkl',M32)
    
    #Contraction 3512; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bdcaijkl->abcdijkl',M32)
    
    #Contraction 3513; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('adcbijkl->abcdijkl',M32)
    
    #Contraction 3514; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('abcdijkl->abcdijkl',M32)
    
    #Contraction 3515; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('cdbaijkl->abcdijkl',M32)
    
    #Contraction 3516; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('adbcijkl->abcdijkl',M32)
    
    #Contraction 3517; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('acbdijkl->abcdijkl',M32)
    
    #Contraction 3518; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('cbdaijkl->abcdijkl',M32)
    
    #Contraction 3519; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('abdcijkl->abcdijkl',M32)
    
    #Contraction 3520; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('acdbijkl->abcdijkl',M32)
    
    #Contraction 3521; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bdaclkij->abcdijkl',M32)
    
    #Contraction 3522; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('cdablkij->abcdijkl',M32)
    
    #Contraction 3523; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('cbadlkij->abcdijkl',M32)
    
    #Contraction 3524; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bdcalkij->abcdijkl',M32)
    
    #Contraction 3525; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('adcblkij->abcdijkl',M32)
    
    #Contraction 3526; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('abcdlkij->abcdijkl',M32)
    
    #Contraction 3527; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('cdbalkij->abcdijkl',M32)
    
    #Contraction 3528; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('adbclkij->abcdijkl',M32)
    
    #Contraction 3529; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('acbdlkij->abcdijkl',M32)
    
    #Contraction 3530; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('cbdalkij->abcdijkl',M32)
    
    #Contraction 3531; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('abdclkij->abcdijkl',M32)
    
    #Contraction 3532; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('acdblkij->abcdijkl',M32)
    
    #Contraction 3533; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bdacjkil->abcdijkl',M32)
    
    #Contraction 3534; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('cdabjkil->abcdijkl',M32)
    
    #Contraction 3535; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('cbadjkil->abcdijkl',M32)
    
    #Contraction 3536; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bdcajkil->abcdijkl',M32)
    
    #Contraction 3537; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('adcbjkil->abcdijkl',M32)
    
    #Contraction 3538; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('abcdjkil->abcdijkl',M32)
    
    #Contraction 3539; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('cdbajkil->abcdijkl',M32)
    
    #Contraction 3540; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('adbcjkil->abcdijkl',M32)
    
    #Contraction 3541; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('acbdjkil->abcdijkl',M32)
    
    #Contraction 3542; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('cbdajkil->abcdijkl',M32)
    
    #Contraction 3543; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('abdcjkil->abcdijkl',M32)
    
    #Contraction 3544; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('acdbjkil->abcdijkl',M32)
    
    #Contraction 3545; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bdacikjl->abcdijkl',M32)
    
    #Contraction 3546; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('cdabikjl->abcdijkl',M32)
    
    #Contraction 3547; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('cbadikjl->abcdijkl',M32)
    
    #Contraction 3548; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bdcaikjl->abcdijkl',M32)
    
    #Contraction 3549; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('adcbikjl->abcdijkl',M32)
    
    #Contraction 3550; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('abcdikjl->abcdijkl',M32)
    
    #Contraction 3551; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('cdbaikjl->abcdijkl',M32)
    
    #Contraction 3552; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('adbcikjl->abcdijkl',M32)
    
    #Contraction 3553; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('acbdikjl->abcdijkl',M32)
    
    #Contraction 3554; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('cbdaikjl->abcdijkl',M32)
    
    #Contraction 3555; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('abdcikjl->abcdijkl',M32)
    
    #Contraction 3556; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('acdbikjl->abcdijkl',M32)
    
    #Contraction 3557; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bdacklij->abcdijkl',M32)
    
    #Contraction 3558; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('cdabklij->abcdijkl',M32)
    
    #Contraction 3559; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('cbadklij->abcdijkl',M32)
    
    #Contraction 3560; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bdcaklij->abcdijkl',M32)
    
    #Contraction 3561; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('adcbklij->abcdijkl',M32)
    
    #Contraction 3562; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('abcdklij->abcdijkl',M32)
    
    #Contraction 3563; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('cdbaklij->abcdijkl',M32)
    
    #Contraction 3564; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('adbcklij->abcdijkl',M32)
    
    #Contraction 3565; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('acbdklij->abcdijkl',M32)
    
    #Contraction 3566; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('cbdaklij->abcdijkl',M32)
    
    #Contraction 3567; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('abdcklij->abcdijkl',M32)
    
    #Contraction 3568; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('acdbklij->abcdijkl',M32)
    
    #Contraction 3569; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bdacjlik->abcdijkl',M32)
    
    #Contraction 3570; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('cdabjlik->abcdijkl',M32)
    
    #Contraction 3571; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('cbadjlik->abcdijkl',M32)
    
    #Contraction 3572; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bdcajlik->abcdijkl',M32)
    
    #Contraction 3573; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('adcbjlik->abcdijkl',M32)
    
    #Contraction 3574; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('abcdjlik->abcdijkl',M32)
    
    #Contraction 3575; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('cdbajlik->abcdijkl',M32)
    
    #Contraction 3576; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('adbcjlik->abcdijkl',M32)
    
    #Contraction 3577; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('acbdjlik->abcdijkl',M32)
    
    #Contraction 3578; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('cbdajlik->abcdijkl',M32)
    
    #Contraction 3579; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('abdcjlik->abcdijkl',M32)
    
    #Contraction 3580; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('acdbjlik->abcdijkl',M32)
    
    #Contraction 3581; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bdaciljk->abcdijkl',M32)
    
    #Contraction 3582; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('cdabiljk->abcdijkl',M32)
    
    #Contraction 3583; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('cbadiljk->abcdijkl',M32)
    
    #Contraction 3584; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bdcailjk->abcdijkl',M32)
    
    #Contraction 3585; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('adcbiljk->abcdijkl',M32)
    
    #Contraction 3586; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('abcdiljk->abcdijkl',M32)
    
    #Contraction 3587; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('cdbailjk->abcdijkl',M32)
    
    #Contraction 3588; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('adbciljk->abcdijkl',M32)
    
    #Contraction 3589; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('acbdiljk->abcdijkl',M32)
    
    #Contraction 3590; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('cbdailjk->abcdijkl',M32)
    
    #Contraction 3591; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('abdciljk->abcdijkl',M32)
    
    #Contraction 3592; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('acdbiljk->abcdijkl',M32)
    
    del M32
    
    X32 = np.zeros([nvir, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 3593; Tree Level  2; Scaling  6/ 4 Result_size  4/ 4
    X32 += oe.contract('abcdnjkl,ni->abcdjkli',T4,Y16, optimize='optimal')
    
    del Y16
    
    #Contraction 3594; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('abcdjkli->abcdijkl',X32)
    
    #Contraction 3595; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('abcdiklj->abcdijkl',X32)
    
    #Contraction 3596; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('abcdijlk->abcdijkl',X32)
    
    #Contraction 3597; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('abcdijkl->abcdijkl',X32)
    
    del X32
    
    Y32 = np.zeros([nvir, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 3598; Tree Level  2; Scaling  4/ 6 Result_size  4/ 4
    Y32 += oe.contract('ceki,abdejl->cabdkijl',T2,M31, optimize='optimal')
    
    del M31
    
    #Contraction 3599; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('cabdkijl->abcdijkl',Y32)
    
    #Contraction 3600; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('dabckijl->abcdijkl',Y32)
    
    #Contraction 3601; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('bacdkijl->abcdijkl',Y32)
    
    #Contraction 3602; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('dacbkijl->abcdijkl',Y32)
    
    #Contraction 3603; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('badckijl->abcdijkl',Y32)
    
    #Contraction 3604; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('cadbkijl->abcdijkl',Y32)
    
    #Contraction 3605; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('abcdkijl->abcdijkl',Y32)
    
    #Contraction 3606; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('dbcakijl->abcdijkl',Y32)
    
    #Contraction 3607; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('abdckijl->abcdijkl',Y32)
    
    #Contraction 3608; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('cbdakijl->abcdijkl',Y32)
    
    #Contraction 3609; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('acdbkijl->abcdijkl',Y32)
    
    #Contraction 3610; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('bcdakijl->abcdijkl',Y32)
    
    #Contraction 3611; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('cabdkjil->abcdijkl',Y32)
    
    #Contraction 3612; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('dabckjil->abcdijkl',Y32)
    
    #Contraction 3613; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('bacdkjil->abcdijkl',Y32)
    
    #Contraction 3614; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('dacbkjil->abcdijkl',Y32)
    
    #Contraction 3615; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('badckjil->abcdijkl',Y32)
    
    #Contraction 3616; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('cadbkjil->abcdijkl',Y32)
    
    #Contraction 3617; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('abcdkjil->abcdijkl',Y32)
    
    #Contraction 3618; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('dbcakjil->abcdijkl',Y32)
    
    #Contraction 3619; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('abdckjil->abcdijkl',Y32)
    
    #Contraction 3620; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('cbdakjil->abcdijkl',Y32)
    
    #Contraction 3621; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('acdbkjil->abcdijkl',Y32)
    
    #Contraction 3622; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('bcdakjil->abcdijkl',Y32)
    
    #Contraction 3623; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('cabdklij->abcdijkl',Y32)
    
    #Contraction 3624; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('dabcklij->abcdijkl',Y32)
    
    #Contraction 3625; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('bacdklij->abcdijkl',Y32)
    
    #Contraction 3626; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('dacbklij->abcdijkl',Y32)
    
    #Contraction 3627; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('badcklij->abcdijkl',Y32)
    
    #Contraction 3628; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('cadbklij->abcdijkl',Y32)
    
    #Contraction 3629; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('abcdklij->abcdijkl',Y32)
    
    #Contraction 3630; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('dbcaklij->abcdijkl',Y32)
    
    #Contraction 3631; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('abdcklij->abcdijkl',Y32)
    
    #Contraction 3632; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('cbdaklij->abcdijkl',Y32)
    
    #Contraction 3633; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('acdbklij->abcdijkl',Y32)
    
    #Contraction 3634; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('bcdaklij->abcdijkl',Y32)
    
    #Contraction 3635; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('cabdijkl->abcdijkl',Y32)
    
    #Contraction 3636; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('dabcijkl->abcdijkl',Y32)
    
    #Contraction 3637; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('bacdijkl->abcdijkl',Y32)
    
    #Contraction 3638; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('dacbijkl->abcdijkl',Y32)
    
    #Contraction 3639; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('badcijkl->abcdijkl',Y32)
    
    #Contraction 3640; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('cadbijkl->abcdijkl',Y32)
    
    #Contraction 3641; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('abcdijkl->abcdijkl',Y32)
    
    #Contraction 3642; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('dbcaijkl->abcdijkl',Y32)
    
    #Contraction 3643; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('abdcijkl->abcdijkl',Y32)
    
    #Contraction 3644; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('cbdaijkl->abcdijkl',Y32)
    
    #Contraction 3645; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('acdbijkl->abcdijkl',Y32)
    
    #Contraction 3646; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('bcdaijkl->abcdijkl',Y32)
    
    #Contraction 3647; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('cabdilkj->abcdijkl',Y32)
    
    #Contraction 3648; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('dabcilkj->abcdijkl',Y32)
    
    #Contraction 3649; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('bacdilkj->abcdijkl',Y32)
    
    #Contraction 3650; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('dacbilkj->abcdijkl',Y32)
    
    #Contraction 3651; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('badcilkj->abcdijkl',Y32)
    
    #Contraction 3652; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('cadbilkj->abcdijkl',Y32)
    
    #Contraction 3653; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('abcdilkj->abcdijkl',Y32)
    
    #Contraction 3654; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('dbcailkj->abcdijkl',Y32)
    
    #Contraction 3655; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('abdcilkj->abcdijkl',Y32)
    
    #Contraction 3656; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('cbdailkj->abcdijkl',Y32)
    
    #Contraction 3657; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('acdbilkj->abcdijkl',Y32)
    
    #Contraction 3658; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('bcdailkj->abcdijkl',Y32)
    
    #Contraction 3659; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('cabdjlki->abcdijkl',Y32)
    
    #Contraction 3660; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('dabcjlki->abcdijkl',Y32)
    
    #Contraction 3661; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('bacdjlki->abcdijkl',Y32)
    
    #Contraction 3662; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('dacbjlki->abcdijkl',Y32)
    
    #Contraction 3663; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('badcjlki->abcdijkl',Y32)
    
    #Contraction 3664; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('cadbjlki->abcdijkl',Y32)
    
    #Contraction 3665; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('abcdjlki->abcdijkl',Y32)
    
    #Contraction 3666; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('dbcajlki->abcdijkl',Y32)
    
    #Contraction 3667; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('abdcjlki->abcdijkl',Y32)
    
    #Contraction 3668; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('cbdajlki->abcdijkl',Y32)
    
    #Contraction 3669; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('acdbjlki->abcdijkl',Y32)
    
    #Contraction 3670; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('bcdajlki->abcdijkl',Y32)
    
    del Y32
    
    A33 = np.zeros([nvir, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 3671; Tree Level  2; Scaling  8/ 4 Result_size  4/ 4
    A33 += oe.contract('bcdmnl,mnaijk->bcdalijk',T3,Y30, optimize='optimal')
    
    del Y30
    
    #Contraction 3672; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('bcdalijk->abcdijkl',A33)
    
    #Contraction 3673; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('acdblijk->abcdijkl',A33)
    
    #Contraction 3674; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('abdclijk->abcdijkl',A33)
    
    #Contraction 3675; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('abcdlijk->abcdijkl',A33)
    
    #Contraction 3676; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('bcdakijl->abcdijkl',A33)
    
    #Contraction 3677; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('acdbkijl->abcdijkl',A33)
    
    #Contraction 3678; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('abdckijl->abcdijkl',A33)
    
    #Contraction 3679; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('abcdkijl->abcdijkl',A33)
    
    #Contraction 3680; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('bcdajikl->abcdijkl',A33)
    
    #Contraction 3681; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('acdbjikl->abcdijkl',A33)
    
    #Contraction 3682; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('abdcjikl->abcdijkl',A33)
    
    #Contraction 3683; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('abcdjikl->abcdijkl',A33)
    
    #Contraction 3684; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('bcdaijkl->abcdijkl',A33)
    
    #Contraction 3685; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('acdbijkl->abcdijkl',A33)
    
    #Contraction 3686; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('abdcijkl->abcdijkl',A33)
    
    #Contraction 3687; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('abcdijkl->abcdijkl',A33)
    
    del A33
    
    D33 = np.zeros([nvir, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 3688; Tree Level  2; Scaling  6/ 4 Result_size  4/ 4
    D33 += oe.contract('bcdmkl,amij->bcdaklij',T3,J4, optimize='optimal')
    
    del J4
    
    #del T3
    
    #Contraction 3689; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bcdaklij->abcdijkl',D33)
    
    #Contraction 3690; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('acdbklij->abcdijkl',D33)
    
    #Contraction 3691; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('abdcklij->abcdijkl',D33)
    
    #Contraction 3692; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('abcdklij->abcdijkl',D33)
    
    #Contraction 3693; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bcdajlik->abcdijkl',D33)
    
    #Contraction 3694; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('acdbjlik->abcdijkl',D33)
    
    #Contraction 3695; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('abdcjlik->abcdijkl',D33)
    
    #Contraction 3696; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('abcdjlik->abcdijkl',D33)
    
    #Contraction 3697; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bcdajkil->abcdijkl',D33)
    
    #Contraction 3698; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('acdbjkil->abcdijkl',D33)
    
    #Contraction 3699; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('abdcjkil->abcdijkl',D33)
    
    #Contraction 3700; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('abcdjkil->abcdijkl',D33)
    
    #Contraction 3701; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bcdaklji->abcdijkl',D33)
    
    #Contraction 3702; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('acdbklji->abcdijkl',D33)
    
    #Contraction 3703; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('abdcklji->abcdijkl',D33)
    
    #Contraction 3704; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('abcdklji->abcdijkl',D33)
    
    #Contraction 3705; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bcdailjk->abcdijkl',D33)
    
    #Contraction 3706; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('acdbiljk->abcdijkl',D33)
    
    #Contraction 3707; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('abdciljk->abcdijkl',D33)
    
    #Contraction 3708; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('abcdiljk->abcdijkl',D33)
    
    #Contraction 3709; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bcdaikjl->abcdijkl',D33)
    
    #Contraction 3710; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('acdbikjl->abcdijkl',D33)
    
    #Contraction 3711; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('abdcikjl->abcdijkl',D33)
    
    #Contraction 3712; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('abcdikjl->abcdijkl',D33)
    
    #Contraction 3713; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bcdajlki->abcdijkl',D33)
    
    #Contraction 3714; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('acdbjlki->abcdijkl',D33)
    
    #Contraction 3715; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('abdcjlki->abcdijkl',D33)
    
    #Contraction 3716; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('abcdjlki->abcdijkl',D33)
    
    #Contraction 3717; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bcdailkj->abcdijkl',D33)
    
    #Contraction 3718; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('acdbilkj->abcdijkl',D33)
    
    #Contraction 3719; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('abdcilkj->abcdijkl',D33)
    
    #Contraction 3720; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('abcdilkj->abcdijkl',D33)
    
    #Contraction 3721; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bcdaijkl->abcdijkl',D33)
    
    #Contraction 3722; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('acdbijkl->abcdijkl',D33)
    
    #Contraction 3723; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('abdcijkl->abcdijkl',D33)
    
    #Contraction 3724; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('abcdijkl->abcdijkl',D33)
    
    #Contraction 3725; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bcdajkli->abcdijkl',D33)
    
    #Contraction 3726; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('acdbjkli->abcdijkl',D33)
    
    #Contraction 3727; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('abdcjkli->abcdijkl',D33)
    
    #Contraction 3728; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('abcdjkli->abcdijkl',D33)
    
    #Contraction 3729; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bcdaiklj->abcdijkl',D33)
    
    #Contraction 3730; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('acdbiklj->abcdijkl',D33)
    
    #Contraction 3731; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('abdciklj->abcdijkl',D33)
    
    #Contraction 3732; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('abcdiklj->abcdijkl',D33)
    
    #Contraction 3733; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bcdaijlk->abcdijkl',D33)
    
    #Contraction 3734; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('acdbijlk->abcdijkl',D33)
    
    #Contraction 3735; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('abdcijlk->abcdijkl',D33)
    
    #Contraction 3736; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('abcdijlk->abcdijkl',D33)
    
    del D33
    
    E33 = np.zeros([nvir, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 3737; Tree Level  2; Scaling  6/ 4 Result_size  4/ 4
    E33 += oe.contract('mi,bcdajklm->bcdaijkl',F1,T4, optimize='optimal')
    
    #del F1
    
    #Contraction 3738; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bcdaijkl->abcdijkl',E33)
    
    #Contraction 3739; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bcdajikl->abcdijkl',E33)
    
    #Contraction 3740; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bcdakijl->abcdijkl',E33)
    
    #Contraction 3741; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bcdalijk->abcdijkl',E33)
    
    del E33
    
    G33 = np.zeros([nvir, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 3742; Tree Level  2; Scaling  6/ 4 Result_size  4/ 4
    G33 += oe.contract('abcdnjkl,ni->abcdjkli',T4,J8, optimize='optimal')
    
    del J8
    
    #Contraction 3743; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('abcdjkli->abcdijkl',G33)
    
    #Contraction 3744; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('abcdiklj->abcdijkl',G33)
    
    #Contraction 3745; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('abcdijlk->abcdijkl',G33)
    
    #Contraction 3746; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('abcdijkl->abcdijkl',G33)
    
    del G33
    
    I33 = np.zeros([nocc, nocc, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 3747; Tree Level  3; Scaling  6/ 6 Result_size  6/ 2
    I33 += oe.contract('mnef,efcdijkl->mncdijkl',V7,T4, optimize='optimal')
    
    #del T4
    
    #del V7
    
    J33 = np.zeros([nvir, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 3748; Tree Level  2; Scaling  8/ 4 Result_size  4/ 4
    J33 += oe.contract('abmn,mncdijkl->abcdijkl',T2,I33, optimize='optimal')
    
    #Contraction 3749; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('abcdijkl->abcdijkl',J33)
    
    #Contraction 3750; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('acbdijkl->abcdijkl',J33)
    
    #Contraction 3751; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('adbcijkl->abcdijkl',J33)
    
    #Contraction 3752; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('bcadijkl->abcdijkl',J33)
    
    #Contraction 3753; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('bdacijkl->abcdijkl',J33)
    
    #Contraction 3754; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('cdabijkl->abcdijkl',J33)
    
    del J33
    
    M33 = np.zeros([nvir, nocc, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 3755; Tree Level  3; Scaling  7/ 3 Result_size  5/ 3
    M33 += oe.contract('bn,mncdijkl->bmcdijkl',T1,I33, optimize='optimal')
    
    del I33
    
    X33 = np.zeros([nvir, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 3756; Tree Level  2; Scaling  6/ 4 Result_size  4/ 4
    X33 += oe.contract('am,bmcdijkl->abcdijkl',T1,M33, optimize='optimal')
    
    del M33
    
    #Contraction 3757; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('abcdijkl->abcdijkl',X33)
    
    #Contraction 3758; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('acbdijkl->abcdijkl',X33)
    
    #Contraction 3759; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('adbcijkl->abcdijkl',X33)
    
    #Contraction 3760; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('bacdijkl->abcdijkl',X33)
    
    #Contraction 3761; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('bcadijkl->abcdijkl',X33)
    
    #Contraction 3762; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('bdacijkl->abcdijkl',X33)
    
    #Contraction 3763; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('cabdijkl->abcdijkl',X33)
    
    #Contraction 3764; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('cbadijkl->abcdijkl',X33)
    
    #Contraction 3765; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('cdabijkl->abcdijkl',X33)
    
    #Contraction 3766; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('dabcijkl->abcdijkl',X33)
    
    #Contraction 3767; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.25 * oe.contract('dbacijkl->abcdijkl',X33)
    
    #Contraction 3768; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.25 * oe.contract('dcabijkl->abcdijkl',X33)
    
    del X33
    
    Y33 = np.zeros([nvir, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 3769; Tree Level  2; Scaling  6/ 4 Result_size  4/ 4
    Y33 += oe.contract('cdml,ambijk->cdablijk',T2,E24, optimize='optimal')
    
    del E24
    
    #Contraction 3770; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('cdablijk->abcdijkl',Y33)
    
    #Contraction 3771; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bdaclijk->abcdijkl',Y33)
    
    #Contraction 3772; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bcadlijk->abcdijkl',Y33)
    
    #Contraction 3773; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('cdbalijk->abcdijkl',Y33)
    
    #Contraction 3774; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('adbclijk->abcdijkl',Y33)
    
    #Contraction 3775; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('acbdlijk->abcdijkl',Y33)
    
    #Contraction 3776; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bdcalijk->abcdijkl',Y33)
    
    #Contraction 3777; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('adcblijk->abcdijkl',Y33)
    
    #Contraction 3778; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('abcdlijk->abcdijkl',Y33)
    
    #Contraction 3779; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bcdalijk->abcdijkl',Y33)
    
    #Contraction 3780; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('acdblijk->abcdijkl',Y33)
    
    #Contraction 3781; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('abdclijk->abcdijkl',Y33)
    
    #Contraction 3782; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('cdabkijl->abcdijkl',Y33)
    
    #Contraction 3783; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bdackijl->abcdijkl',Y33)
    
    #Contraction 3784; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bcadkijl->abcdijkl',Y33)
    
    #Contraction 3785; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('cdbakijl->abcdijkl',Y33)
    
    #Contraction 3786; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('adbckijl->abcdijkl',Y33)
    
    #Contraction 3787; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('acbdkijl->abcdijkl',Y33)
    
    #Contraction 3788; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bdcakijl->abcdijkl',Y33)
    
    #Contraction 3789; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('adcbkijl->abcdijkl',Y33)
    
    #Contraction 3790; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('abcdkijl->abcdijkl',Y33)
    
    #Contraction 3791; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bcdakijl->abcdijkl',Y33)
    
    #Contraction 3792; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('acdbkijl->abcdijkl',Y33)
    
    #Contraction 3793; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('abdckijl->abcdijkl',Y33)
    
    #Contraction 3794; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('cdabjikl->abcdijkl',Y33)
    
    #Contraction 3795; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bdacjikl->abcdijkl',Y33)
    
    #Contraction 3796; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bcadjikl->abcdijkl',Y33)
    
    #Contraction 3797; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('cdbajikl->abcdijkl',Y33)
    
    #Contraction 3798; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('adbcjikl->abcdijkl',Y33)
    
    #Contraction 3799; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('acbdjikl->abcdijkl',Y33)
    
    #Contraction 3800; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bdcajikl->abcdijkl',Y33)
    
    #Contraction 3801; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('adcbjikl->abcdijkl',Y33)
    
    #Contraction 3802; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('abcdjikl->abcdijkl',Y33)
    
    #Contraction 3803; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bcdajikl->abcdijkl',Y33)
    
    #Contraction 3804; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('acdbjikl->abcdijkl',Y33)
    
    #Contraction 3805; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('abdcjikl->abcdijkl',Y33)
    
    #Contraction 3806; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('cdabljik->abcdijkl',Y33)
    
    #Contraction 3807; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bdacljik->abcdijkl',Y33)
    
    #Contraction 3808; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bcadljik->abcdijkl',Y33)
    
    #Contraction 3809; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('cdbaljik->abcdijkl',Y33)
    
    #Contraction 3810; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('adbcljik->abcdijkl',Y33)
    
    #Contraction 3811; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('acbdljik->abcdijkl',Y33)
    
    #Contraction 3812; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bdcaljik->abcdijkl',Y33)
    
    #Contraction 3813; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('adcbljik->abcdijkl',Y33)
    
    #Contraction 3814; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('abcdljik->abcdijkl',Y33)
    
    #Contraction 3815; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bcdaljik->abcdijkl',Y33)
    
    #Contraction 3816; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('acdbljik->abcdijkl',Y33)
    
    #Contraction 3817; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('abdcljik->abcdijkl',Y33)
    
    #Contraction 3818; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('cdabkjil->abcdijkl',Y33)
    
    #Contraction 3819; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bdackjil->abcdijkl',Y33)
    
    #Contraction 3820; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bcadkjil->abcdijkl',Y33)
    
    #Contraction 3821; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('cdbakjil->abcdijkl',Y33)
    
    #Contraction 3822; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('adbckjil->abcdijkl',Y33)
    
    #Contraction 3823; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('acbdkjil->abcdijkl',Y33)
    
    #Contraction 3824; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bdcakjil->abcdijkl',Y33)
    
    #Contraction 3825; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('adcbkjil->abcdijkl',Y33)
    
    #Contraction 3826; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('abcdkjil->abcdijkl',Y33)
    
    #Contraction 3827; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bcdakjil->abcdijkl',Y33)
    
    #Contraction 3828; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('acdbkjil->abcdijkl',Y33)
    
    #Contraction 3829; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('abdckjil->abcdijkl',Y33)
    
    #Contraction 3830; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('cdabijkl->abcdijkl',Y33)
    
    #Contraction 3831; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bdacijkl->abcdijkl',Y33)
    
    #Contraction 3832; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bcadijkl->abcdijkl',Y33)
    
    #Contraction 3833; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('cdbaijkl->abcdijkl',Y33)
    
    #Contraction 3834; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('adbcijkl->abcdijkl',Y33)
    
    #Contraction 3835; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('acbdijkl->abcdijkl',Y33)
    
    #Contraction 3836; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bdcaijkl->abcdijkl',Y33)
    
    #Contraction 3837; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('adcbijkl->abcdijkl',Y33)
    
    #Contraction 3838; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('abcdijkl->abcdijkl',Y33)
    
    #Contraction 3839; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bcdaijkl->abcdijkl',Y33)
    
    #Contraction 3840; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('acdbijkl->abcdijkl',Y33)
    
    #Contraction 3841; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('abdcijkl->abcdijkl',Y33)
    
    #Contraction 3842; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('cdablkij->abcdijkl',Y33)
    
    #Contraction 3843; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bdaclkij->abcdijkl',Y33)
    
    #Contraction 3844; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bcadlkij->abcdijkl',Y33)
    
    #Contraction 3845; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('cdbalkij->abcdijkl',Y33)
    
    #Contraction 3846; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('adbclkij->abcdijkl',Y33)
    
    #Contraction 3847; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('acbdlkij->abcdijkl',Y33)
    
    #Contraction 3848; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bdcalkij->abcdijkl',Y33)
    
    #Contraction 3849; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('adcblkij->abcdijkl',Y33)
    
    #Contraction 3850; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('abcdlkij->abcdijkl',Y33)
    
    #Contraction 3851; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bcdalkij->abcdijkl',Y33)
    
    #Contraction 3852; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('acdblkij->abcdijkl',Y33)
    
    #Contraction 3853; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('abdclkij->abcdijkl',Y33)
    
    #Contraction 3854; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('cdabjkil->abcdijkl',Y33)
    
    #Contraction 3855; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bdacjkil->abcdijkl',Y33)
    
    #Contraction 3856; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bcadjkil->abcdijkl',Y33)
    
    #Contraction 3857; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('cdbajkil->abcdijkl',Y33)
    
    #Contraction 3858; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('adbcjkil->abcdijkl',Y33)
    
    #Contraction 3859; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('acbdjkil->abcdijkl',Y33)
    
    #Contraction 3860; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bdcajkil->abcdijkl',Y33)
    
    #Contraction 3861; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('adcbjkil->abcdijkl',Y33)
    
    #Contraction 3862; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('abcdjkil->abcdijkl',Y33)
    
    #Contraction 3863; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bcdajkil->abcdijkl',Y33)
    
    #Contraction 3864; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('acdbjkil->abcdijkl',Y33)
    
    #Contraction 3865; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('abdcjkil->abcdijkl',Y33)
    
    #Contraction 3866; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('cdabikjl->abcdijkl',Y33)
    
    #Contraction 3867; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bdacikjl->abcdijkl',Y33)
    
    #Contraction 3868; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bcadikjl->abcdijkl',Y33)
    
    #Contraction 3869; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('cdbaikjl->abcdijkl',Y33)
    
    #Contraction 3870; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('adbcikjl->abcdijkl',Y33)
    
    #Contraction 3871; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('acbdikjl->abcdijkl',Y33)
    
    #Contraction 3872; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bdcaikjl->abcdijkl',Y33)
    
    #Contraction 3873; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('adcbikjl->abcdijkl',Y33)
    
    #Contraction 3874; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('abcdikjl->abcdijkl',Y33)
    
    #Contraction 3875; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bcdaikjl->abcdijkl',Y33)
    
    #Contraction 3876; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('acdbikjl->abcdijkl',Y33)
    
    #Contraction 3877; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('abdcikjl->abcdijkl',Y33)
    
    #Contraction 3878; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('cdabklij->abcdijkl',Y33)
    
    #Contraction 3879; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bdacklij->abcdijkl',Y33)
    
    #Contraction 3880; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bcadklij->abcdijkl',Y33)
    
    #Contraction 3881; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('cdbaklij->abcdijkl',Y33)
    
    #Contraction 3882; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('adbcklij->abcdijkl',Y33)
    
    #Contraction 3883; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('acbdklij->abcdijkl',Y33)
    
    #Contraction 3884; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bdcaklij->abcdijkl',Y33)
    
    #Contraction 3885; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('adcbklij->abcdijkl',Y33)
    
    #Contraction 3886; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('abcdklij->abcdijkl',Y33)
    
    #Contraction 3887; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bcdaklij->abcdijkl',Y33)
    
    #Contraction 3888; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('acdbklij->abcdijkl',Y33)
    
    #Contraction 3889; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('abdcklij->abcdijkl',Y33)
    
    #Contraction 3890; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('cdabjlik->abcdijkl',Y33)
    
    #Contraction 3891; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bdacjlik->abcdijkl',Y33)
    
    #Contraction 3892; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bcadjlik->abcdijkl',Y33)
    
    #Contraction 3893; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('cdbajlik->abcdijkl',Y33)
    
    #Contraction 3894; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('adbcjlik->abcdijkl',Y33)
    
    #Contraction 3895; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('acbdjlik->abcdijkl',Y33)
    
    #Contraction 3896; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bdcajlik->abcdijkl',Y33)
    
    #Contraction 3897; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('adcbjlik->abcdijkl',Y33)
    
    #Contraction 3898; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('abcdjlik->abcdijkl',Y33)
    
    #Contraction 3899; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bcdajlik->abcdijkl',Y33)
    
    #Contraction 3900; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('acdbjlik->abcdijkl',Y33)
    
    #Contraction 3901; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('abdcjlik->abcdijkl',Y33)
    
    #Contraction 3902; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('cdabiljk->abcdijkl',Y33)
    
    #Contraction 3903; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bdaciljk->abcdijkl',Y33)
    
    #Contraction 3904; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bcadiljk->abcdijkl',Y33)
    
    #Contraction 3905; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('cdbailjk->abcdijkl',Y33)
    
    #Contraction 3906; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('adbciljk->abcdijkl',Y33)
    
    #Contraction 3907; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('acbdiljk->abcdijkl',Y33)
    
    #Contraction 3908; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bdcailjk->abcdijkl',Y33)
    
    #Contraction 3909; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('adcbiljk->abcdijkl',Y33)
    
    #Contraction 3910; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('abcdiljk->abcdijkl',Y33)
    
    #Contraction 3911; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bcdailjk->abcdijkl',Y33)
    
    #Contraction 3912; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('acdbiljk->abcdijkl',Y33)
    
    #Contraction 3913; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('abdciljk->abcdijkl',Y33)
    
    del Y33
    
    A34 = np.zeros([nvir, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 3914; Tree Level  2; Scaling  6/ 4 Result_size  4/ 4
    A34 += oe.contract('bm,amcdijkl->bacdijkl',T1,X27, optimize='optimal')
    
    del X27
    
    #del T1
    
    #Contraction 3915; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('bacdijkl->abcdijkl',A34)
    
    #Contraction 3916; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('cabdijkl->abcdijkl',A34)
    
    #Contraction 3917; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('dabcijkl->abcdijkl',A34)
    
    #Contraction 3918; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('abcdijkl->abcdijkl',A34)
    
    #Contraction 3919; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('cbadijkl->abcdijkl',A34)
    
    #Contraction 3920; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('dbacijkl->abcdijkl',A34)
    
    #Contraction 3921; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('acbdijkl->abcdijkl',A34)
    
    #Contraction 3922; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('bcadijkl->abcdijkl',A34)
    
    #Contraction 3923; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('dcabijkl->abcdijkl',A34)
    
    #Contraction 3924; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('adbcijkl->abcdijkl',A34)
    
    #Contraction 3925; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -0.5 * oe.contract('bdacijkl->abcdijkl',A34)
    
    #Contraction 3926; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += 0.5 * oe.contract('cdabijkl->abcdijkl',A34)
    
    del A34
    
    D34 = np.zeros([nvir, nvir, nvir, nvir, nocc, nocc, nocc, nocc], dtype=type_)
    
    #Contraction 3927; Tree Level  2; Scaling  6/ 4 Result_size  4/ 4
    D34 += oe.contract('bdml,camjki->bdcaljki',T2,J20, optimize='optimal')
    
    del J20
    
    #del T2
    
    #Contraction 3928; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bdcaljki->abcdijkl',D34)
    
    #Contraction 3929; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('cdbaljki->abcdijkl',D34)
    
    #Contraction 3930; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('cbdaljki->abcdijkl',D34)
    
    #Contraction 3931; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bdacljki->abcdijkl',D34)
    
    #Contraction 3932; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('adbcljki->abcdijkl',D34)
    
    #Contraction 3933; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('abdcljki->abcdijkl',D34)
    
    #Contraction 3934; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('cdabljki->abcdijkl',D34)
    
    #Contraction 3935; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('adcbljki->abcdijkl',D34)
    
    #Contraction 3936; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('acdbljki->abcdijkl',D34)
    
    #Contraction 3937; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('cbadljki->abcdijkl',D34)
    
    #Contraction 3938; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('abcdljki->abcdijkl',D34)
    
    #Contraction 3939; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('acbdljki->abcdijkl',D34)
    
    #Contraction 3940; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bdcakjli->abcdijkl',D34)
    
    #Contraction 3941; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('cdbakjli->abcdijkl',D34)
    
    #Contraction 3942; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('cbdakjli->abcdijkl',D34)
    
    #Contraction 3943; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bdackjli->abcdijkl',D34)
    
    #Contraction 3944; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('adbckjli->abcdijkl',D34)
    
    #Contraction 3945; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('abdckjli->abcdijkl',D34)
    
    #Contraction 3946; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('cdabkjli->abcdijkl',D34)
    
    #Contraction 3947; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('adcbkjli->abcdijkl',D34)
    
    #Contraction 3948; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('acdbkjli->abcdijkl',D34)
    
    #Contraction 3949; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('cbadkjli->abcdijkl',D34)
    
    #Contraction 3950; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('abcdkjli->abcdijkl',D34)
    
    #Contraction 3951; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('acbdkjli->abcdijkl',D34)
    
    #Contraction 3952; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bdcajkli->abcdijkl',D34)
    
    #Contraction 3953; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('cdbajkli->abcdijkl',D34)
    
    #Contraction 3954; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('cbdajkli->abcdijkl',D34)
    
    #Contraction 3955; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bdacjkli->abcdijkl',D34)
    
    #Contraction 3956; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('adbcjkli->abcdijkl',D34)
    
    #Contraction 3957; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('abdcjkli->abcdijkl',D34)
    
    #Contraction 3958; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('cdabjkli->abcdijkl',D34)
    
    #Contraction 3959; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('adcbjkli->abcdijkl',D34)
    
    #Contraction 3960; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('acdbjkli->abcdijkl',D34)
    
    #Contraction 3961; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('cbadjkli->abcdijkl',D34)
    
    #Contraction 3962; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('abcdjkli->abcdijkl',D34)
    
    #Contraction 3963; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('acbdjkli->abcdijkl',D34)
    
    #Contraction 3964; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bdcalikj->abcdijkl',D34)
    
    #Contraction 3965; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('cdbalikj->abcdijkl',D34)
    
    #Contraction 3966; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('cbdalikj->abcdijkl',D34)
    
    #Contraction 3967; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bdaclikj->abcdijkl',D34)
    
    #Contraction 3968; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('adbclikj->abcdijkl',D34)
    
    #Contraction 3969; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('abdclikj->abcdijkl',D34)
    
    #Contraction 3970; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('cdablikj->abcdijkl',D34)
    
    #Contraction 3971; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('adcblikj->abcdijkl',D34)
    
    #Contraction 3972; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('acdblikj->abcdijkl',D34)
    
    #Contraction 3973; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('cbadlikj->abcdijkl',D34)
    
    #Contraction 3974; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('abcdlikj->abcdijkl',D34)
    
    #Contraction 3975; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('acbdlikj->abcdijkl',D34)
    
    #Contraction 3976; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bdcakilj->abcdijkl',D34)
    
    #Contraction 3977; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('cdbakilj->abcdijkl',D34)
    
    #Contraction 3978; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('cbdakilj->abcdijkl',D34)
    
    #Contraction 3979; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bdackilj->abcdijkl',D34)
    
    #Contraction 3980; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('adbckilj->abcdijkl',D34)
    
    #Contraction 3981; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('abdckilj->abcdijkl',D34)
    
    #Contraction 3982; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('cdabkilj->abcdijkl',D34)
    
    #Contraction 3983; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('adcbkilj->abcdijkl',D34)
    
    #Contraction 3984; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('acdbkilj->abcdijkl',D34)
    
    #Contraction 3985; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('cbadkilj->abcdijkl',D34)
    
    #Contraction 3986; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('abcdkilj->abcdijkl',D34)
    
    #Contraction 3987; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('acbdkilj->abcdijkl',D34)
    
    #Contraction 3988; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bdcaiklj->abcdijkl',D34)
    
    #Contraction 3989; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('cdbaiklj->abcdijkl',D34)
    
    #Contraction 3990; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('cbdaiklj->abcdijkl',D34)
    
    #Contraction 3991; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bdaciklj->abcdijkl',D34)
    
    #Contraction 3992; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('adbciklj->abcdijkl',D34)
    
    #Contraction 3993; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('abdciklj->abcdijkl',D34)
    
    #Contraction 3994; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('cdabiklj->abcdijkl',D34)
    
    #Contraction 3995; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('adcbiklj->abcdijkl',D34)
    
    #Contraction 3996; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('acdbiklj->abcdijkl',D34)
    
    #Contraction 3997; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('cbadiklj->abcdijkl',D34)
    
    #Contraction 3998; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('abcdiklj->abcdijkl',D34)
    
    #Contraction 3999; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('acbdiklj->abcdijkl',D34)
    
    #Contraction 4000; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bdcalijk->abcdijkl',D34)
    
    #Contraction 4001; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('cdbalijk->abcdijkl',D34)
    
    #Contraction 4002; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('cbdalijk->abcdijkl',D34)
    
    #Contraction 4003; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bdaclijk->abcdijkl',D34)
    
    #Contraction 4004; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('adbclijk->abcdijkl',D34)
    
    #Contraction 4005; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('abdclijk->abcdijkl',D34)
    
    #Contraction 4006; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('cdablijk->abcdijkl',D34)
    
    #Contraction 4007; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('adcblijk->abcdijkl',D34)
    
    #Contraction 4008; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('acdblijk->abcdijkl',D34)
    
    #Contraction 4009; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('cbadlijk->abcdijkl',D34)
    
    #Contraction 4010; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('abcdlijk->abcdijkl',D34)
    
    #Contraction 4011; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('acbdlijk->abcdijkl',D34)
    
    #Contraction 4012; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bdcajilk->abcdijkl',D34)
    
    #Contraction 4013; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('cdbajilk->abcdijkl',D34)
    
    #Contraction 4014; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('cbdajilk->abcdijkl',D34)
    
    #Contraction 4015; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bdacjilk->abcdijkl',D34)
    
    #Contraction 4016; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('adbcjilk->abcdijkl',D34)
    
    #Contraction 4017; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('abdcjilk->abcdijkl',D34)
    
    #Contraction 4018; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('cdabjilk->abcdijkl',D34)
    
    #Contraction 4019; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('adcbjilk->abcdijkl',D34)
    
    #Contraction 4020; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('acdbjilk->abcdijkl',D34)
    
    #Contraction 4021; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('cbadjilk->abcdijkl',D34)
    
    #Contraction 4022; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('abcdjilk->abcdijkl',D34)
    
    #Contraction 4023; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('acbdjilk->abcdijkl',D34)
    
    #Contraction 4024; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bdcaijlk->abcdijkl',D34)
    
    #Contraction 4025; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('cdbaijlk->abcdijkl',D34)
    
    #Contraction 4026; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('cbdaijlk->abcdijkl',D34)
    
    #Contraction 4027; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bdacijlk->abcdijkl',D34)
    
    #Contraction 4028; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('adbcijlk->abcdijkl',D34)
    
    #Contraction 4029; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('abdcijlk->abcdijkl',D34)
    
    #Contraction 4030; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('cdabijlk->abcdijkl',D34)
    
    #Contraction 4031; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('adcbijlk->abcdijkl',D34)
    
    #Contraction 4032; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('acdbijlk->abcdijkl',D34)
    
    #Contraction 4033; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('cbadijlk->abcdijkl',D34)
    
    #Contraction 4034; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('abcdijlk->abcdijkl',D34)
    
    #Contraction 4035; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('acbdijlk->abcdijkl',D34)
    
    #Contraction 4036; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bdcakijl->abcdijkl',D34)
    
    #Contraction 4037; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('cdbakijl->abcdijkl',D34)
    
    #Contraction 4038; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('cbdakijl->abcdijkl',D34)
    
    #Contraction 4039; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bdackijl->abcdijkl',D34)
    
    #Contraction 4040; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('adbckijl->abcdijkl',D34)
    
    #Contraction 4041; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('abdckijl->abcdijkl',D34)
    
    #Contraction 4042; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('cdabkijl->abcdijkl',D34)
    
    #Contraction 4043; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('adcbkijl->abcdijkl',D34)
    
    #Contraction 4044; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('acdbkijl->abcdijkl',D34)
    
    #Contraction 4045; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('cbadkijl->abcdijkl',D34)
    
    #Contraction 4046; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('abcdkijl->abcdijkl',D34)
    
    #Contraction 4047; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('acbdkijl->abcdijkl',D34)
    
    #Contraction 4048; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bdcajikl->abcdijkl',D34)
    
    #Contraction 4049; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('cdbajikl->abcdijkl',D34)
    
    #Contraction 4050; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('cbdajikl->abcdijkl',D34)
    
    #Contraction 4051; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bdacjikl->abcdijkl',D34)
    
    #Contraction 4052; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('adbcjikl->abcdijkl',D34)
    
    #Contraction 4053; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('abdcjikl->abcdijkl',D34)
    
    #Contraction 4054; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('cdabjikl->abcdijkl',D34)
    
    #Contraction 4055; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('adcbjikl->abcdijkl',D34)
    
    #Contraction 4056; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('acdbjikl->abcdijkl',D34)
    
    #Contraction 4057; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('cbadjikl->abcdijkl',D34)
    
    #Contraction 4058; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('abcdjikl->abcdijkl',D34)
    
    #Contraction 4059; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('acbdjikl->abcdijkl',D34)
    
    #Contraction 4060; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('bdcaijkl->abcdijkl',D34)
    
    #Contraction 4061; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('cdbaijkl->abcdijkl',D34)
    
    #Contraction 4062; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('cbdaijkl->abcdijkl',D34)
    
    #Contraction 4063; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('bdacijkl->abcdijkl',D34)
    
    #Contraction 4064; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('adbcijkl->abcdijkl',D34)
    
    #Contraction 4065; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('abdcijkl->abcdijkl',D34)
    
    #Contraction 4066; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('cdabijkl->abcdijkl',D34)
    
    #Contraction 4067; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('adcbijkl->abcdijkl',D34)
    
    #Contraction 4068; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('acdbijkl->abcdijkl',D34)
    
    #Contraction 4069; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('cbadijkl->abcdijkl',D34)
    
    #Contraction 4070; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += oe.contract('abcdijkl->abcdijkl',D34)
    
    #Contraction 4071; Tree Level  1; Scaling  4/ 4 Result_size  4/ 4
    Z4 += -1.0 * oe.contract('acbdijkl->abcdijkl',D34)
    
    del D34
    
    return([Z0[0], Z1, Z2, Z3, Z4])
    
# end of numpy_tenpi_ccsdtq
