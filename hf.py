import numpy as np
from read_snt_io import decouple_1b, decouple_2b, sparse_to_dense_4d
from scipy.sparse import csr_matrix

class NormalOrderedHamiltonian:
    def __init__(self, E0, f, Gamma):
        self.E0 = E0      # 0-body term (float)
        self.f = f        # 1-body term (Fock matrix)
        self.Gamma = Gamma # 2-body term (Antisymmetrized matrix elements)
        
    def __repr__(self):
        return (f"NormalOrderedHamiltonian(E0={self.E0:.6f}, "
                f"f_shape={self.f.shape}, Gamma_shape={self.Gamma.shape})")


def hartree_fock(m_basis, j_potential, n_p_val, n_n_val, v2b_sparse=None, max_iter=100, tol=1e-8, occ_indices=None, mode='deformed'):
    """
    Performs a Hartree-Fock calculation for a given number of valence nucleons.
    """
    n_states = len(m_basis.n)
    v1b = decouple_1b(j_potential, m_basis).v1b
    
    if v2b_sparse is None:
        v2b_sparse = decouple_2b(j_potential, m_basis)
    
    # Initial density matrix: occupy the lowest energy states from v1b
    energies = np.diag(v1b)
    # Separate protons and neutrons for occupation
    p_indices = np.where(m_basis.tz < 0)[0]
    n_indices = np.where(m_basis.tz > 0)[0]
    
    if occ_indices is not None:
        occupied = np.array(occ_indices)
    else:
        # Sort primarily by absolute value of jz, then by initial single particle energy
        # np.lexsort sorts by the LAST key primarily
        p_jz_abs = np.abs(m_basis.jz[p_indices])
        n_jz_abs = np.abs(m_basis.jz[n_indices])
        
        p_sort_idx = np.lexsort((energies[p_indices], p_jz_abs))
        n_sort_idx = np.lexsort((energies[n_indices], n_jz_abs))
        
        p_sorted = p_indices[p_sort_idx]
        n_sorted = n_indices[n_sort_idx]
        
        occupied = np.concatenate([p_sorted[:n_p_val], n_sorted[:n_n_val]])
    rho = np.zeros((n_states, n_states))
    for i in occupied:
        rho[i, i] = 1.0
    # Determine symmetry sectors and target counts
    symmetry_keys = []
    for i in range(n_states):
        if mode == 'spherical':
            # spherical
            key = (m_basis.l[i] % 2, m_basis.j[i], m_basis.jz[i], m_basis.tz[i])
        else:
            # deformed
            key = (m_basis.l[i] % 2, m_basis.jz[i], m_basis.tz[i])
        symmetry_keys.append(key)
    
    unique_symmetries = sorted(list(set(symmetry_keys)))
    target_counts = {key: 0 for key in unique_symmetries}
    for i in occupied:
        target_counts[symmetry_keys[i]] += 1
    
    sector_indices = {key: [i for i, k in enumerate(symmetry_keys) if k == key] for key in unique_symmetries}
        
    old_energy = 0.0
    print(f"{'Iter':>4} | {'Total Energy':>15} | {'Delta E':>12}")
    print("-" * 38)
    
    for iteration in range(max_iter):
        # Build Fock matrix efficiently using sparsity
        # F_ab = h_ab + sum_cd rho_dc * V_as(acbd)
        # In our sparse representation V_mn = <a b | V | c d> with m = a*N + b, n = c*N + d
        # We need V_as(ac, bd).
        
        # Method: Iterate over non-zeros of v2b_sparse
        # Or better: Reshape rho and do sparse matrix-vector product if we can map indices.
        # Let's use the property that <ac|V|bd> is just a permutation.
        fock = v1b.copy()
        
        # For HF, we only need the contribution sum_cd rho_dc <ac||bd>
        rows, cols = v2b_sparse.nonzero()
        data = v2b_sparse.data
        for idx in range(len(data)):
            a, c = divmod(rows[idx], n_states)
            b, d = divmod(cols[idx], n_states)
            fock[a, b] += data[idx] * rho[d, c]
        
        new_rho = np.zeros((n_states, n_states))
        
        # Process each symmetry sector independently
        for key in unique_symmetries:
            indices = sector_indices[key]
            if not indices:
                continue
                
            sub_fock = fock[np.ix_(indices, indices)]
            e_sub, c_sub = np.linalg.eigh(sub_fock)
            
            # Occupy the lowest 'count' states for this sector
            count = target_counts[key]
            for i in range(count):
                full_v = np.zeros(n_states)
                full_v[indices] = c_sub[:, i]
                new_rho += np.outer(full_v, full_v)
        
        # Calculate HF energy: E = Tr(rho * v1b) + 0.5 * Tr(rho * (F - v1b))
        e1b = np.sum(new_rho * v1b)
        e2b = 0.5 * np.sum(new_rho * (fock - v1b))
        energy = e1b + e2b
        print(f"   E1B: {e1b:.8f}, E2B: {e2b:.8f}")
        
        delta_e = abs(energy - old_energy)
        print(f"{iteration:4d} | {energy:15.8f} | {delta_e:12.4e}")
        
        rho = new_rho
        old_energy = energy
        
        if delta_e < tol:
            print("Converged!")
            break
        
        rho = new_rho
        old_energy = energy
    else:
        print("HF did not converge within max iterations.")
            
    # Reorder states: All states we chose to occupy (Holes) first, then Particles.
    hole_data = [] # List of (energy, coeff_full)
    part_data = []
    
    # Use the last calculated fock and rho
    for key in unique_symmetries:
        indices = sector_indices[key]
        if not indices: continue
        
        sub_fock = fock[np.ix_(indices, indices)]
        e_sub, c_sub = np.linalg.eigh(sub_fock)
        
        count = target_counts[key]
        for i in range(len(e_sub)):
            full_v = np.zeros(n_states)
            full_v[indices] = c_sub[:, i]
            if i < count:
                hole_data.append((e_sub[i], full_v))
            else:
                part_data.append((e_sub[i], full_v))
    
    hole_data.sort(key=lambda x: x[0])
    part_data.sort(key=lambda x: x[0])
    
    final_energies = np.array([d[0] for d in hole_data] + [d[0] for d in part_data])
    final_coeffs = np.array([d[1] for d in hole_data] + [d[1] for d in part_data]).T
    
    return energy, final_energies, rho, final_coeffs


def normal_order(m_basis, j_potential, energy, sp_energies, rho, sp_coeffs, v2b_sparse=None):
    """
    Transforms the Hamiltonian into its normal-ordered form in the HF basis.
    Uses an optimized multi-step transformation for high performance.
    """
    v1b = decouple_1b(j_potential, m_basis).v1b
    if v2b_sparse is None:
        v2b_sparse = decouple_2b(j_potential, m_basis)
    
    # 0-body: HF Energy
    E0 = energy
    
    # 1-body: Fock matrix in HF basis (should be diagonal)
    f_hf = np.diag(sp_energies)
    
    # 2-body: Transform antisymmetrized interaction to HF basis
    # Gamma_abcd = sum_{pqrs} C_pa * C_qb * V_as(pqrs) * C_rc * C_sd
    print("Transforming 2-body interaction to HF basis (N^5 optimized)...")
    
    n_m = len(m_basis.n)
    C = sp_coeffs # (n_old, n_hf)
    
    # 1st step: Gamma1_{a,q,r,s} = sum_p C_pa * V_{p,q,r,s}
    # This is efficient if V is sparse.
    # We can group non-zeros of V by (q,r,s) and do a dot product or just iterate.
    # Since V is in CSR (pq, rs), we can do:
    # Gamma_step1(a, q, r, s)
    
    # Actually, a better way for sparse V:
    # Gamma_mid(a, b, r, s) = sum_{p,q} C_pa * C_qb * V_{p,q,r,s}
    # V is matrix (pq, rs). Let C2 = outer(C, C) be (pq, ab).
    # Then Gamma_mid = C2^T * V  (ab, rs)
    
    # For large n_m, constructing C2 is N^4.
    # Let's do it in 4 steps N^5 to be safe and compatible with memory.
    
    # Step 1: V1(p,q,r,d) = sum_s V(p,q,r,s) * C(s,d)
    # Step 2: V2(p,q,c,d) = sum_r V1(p,q,r,d) * C(r,c)
    # Step 3: V3(p,b,c,d) = sum_q V2(p,q,c,d) * C(q,b)
    # Step 4: V4(a,b,c,d) = sum_p V3(p,b,c,d) * C(p,a)
    
    # Convert sparse to dense 4D once to perform N^5 steps efficiently
    # (Memory allowing - N=100 is ~800MB)
    V_dense = sparse_to_dense_4d(v2b_sparse, n_m)
    
    temp = np.einsum('pqrs,sd->pqrd', V_dense, C)
    temp = np.einsum('pqrd,rc->pqcd', temp, C)
    temp = np.einsum('pqcd,qb->pbcd', temp, C)
    Gamma = np.einsum('pbcd,pa->abcd', temp, C)
    
    return NormalOrderedHamiltonian(E0, f_hf, Gamma)
