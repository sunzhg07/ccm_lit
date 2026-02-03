"""
Quick diagnostic to check the sp_coeffs from HF
"""
import numpy as np
from read_snt_io import read_snt, generate_m_scheme, decouple_2b
from hf import hartree_fock

snt_file = "sd.snt"
orbits, potential = read_snt(snt_file)
m_scheme = generate_m_scheme(orbits)
v2b_sparse = decouple_2b(potential, m_scheme)

Z_val = 2
N_val = 2

hf_energy, sp_energies, rho, sp_coeffs = hartree_fock(
    m_scheme, potential, Z_val, N_val, v2b_sparse=v2b_sparse, max_iter=100, tol=1e-8
)

print("sp_coeffs shape:", sp_coeffs.shape)
print("sp_coeffs dtype:", sp_coeffs.dtype)
print("\nFirst few rows:")
print(sp_coeffs[:5, :5])
print("\nMax value:", np.max(np.abs(sp_coeffs)))
print("Min value:", np.min(np.abs(sp_coeffs[sp_coeffs != 0])) if np.any(sp_coeffs != 0) else 0)
print("\nAny NaN?", np.any(np.isnan(sp_coeffs)))
print("Any Inf?", np.any(np.isinf(sp_coeffs)))

# Check orthogonality
overlap = sp_coeffs.T @ sp_coeffs
print("\nOverlap matrix (C^T C):")
print(overlap[:5, :5])
print("Max off-diagonal:", np.max(np.abs(overlap - np.diag(np.diag(overlap)))))
