import numpy as np
from sympy.physics.quantum.cg import CG
from sympy import S
from opt_einsum import contract
from scipy.sparse import csr_matrix, lil_matrix

_cg_cache = {}

class SingleParticleOrbits:
    def __init__(self, n_p_core, n_n_core, n_p_orbits, n_n_orbits):
        self.n_p_core = n_p_core
        self.n_n_core = n_n_core
        self.n_p_orbits = n_p_orbits
        self.n_n_orbits = n_n_orbits
        
        total_orbits = n_p_orbits + n_n_orbits
        self.n = np.zeros(total_orbits, dtype=int)
        self.l = np.zeros(total_orbits, dtype=int)
        self.j = np.zeros(total_orbits, dtype=int) # Storing 2j (e.g. 3 -> j=3/2)
        self.jz = np.zeros(total_orbits, dtype=int) # Storing 2jz for m-scheme states; j-scheme entries set to 0
        self.tz = np.zeros(total_orbits, dtype=int) # Storing tz (isospin projection: -1 proton, +1 neutron)
        self.coupling_map = np.zeros(total_orbits, dtype=int) # Maps m-scheme to j-scheme orbit index

    def __repr__(self):
        return (f"SingleParticleOrbits(Z_core={self.n_p_core}, N_core={self.n_n_core}, "
                f"p_orbits={self.n_p_orbits}, n_orbits={self.n_n_orbits})")


class Potential:
    def __init__(self, n_orbits):
        self.n_orbits = n_orbits
        self.v1b = np.zeros((n_orbits, n_orbits))
        self.v2b = [] # List of lists: [i, j, k, l, J, value]
        self.hw = 0.0
        
    def __repr__(self):
        return (f"Potential(hw={self.hw}, non-zero 1b: {np.count_nonzero(self.v1b)}, "
                f"non-zero 2b groups: {len(self.v2b)})")


def read_snt(filename):
    with open(filename, 'r') as f:
        # Skip comment lines until we reach the model space definition
        line = f.readline()
        while line and (line.strip().startswith('!') or not line.strip()):
            line = f.readline()
        
        if not line:
            raise ValueError("Could not find model space definition in .snt file")
            
        # First 4 numbers: Z_core, N_core, p_orbits, n_orbits
        parts = line.split()
        if len(parts) < 4:
            raise ValueError(f"Invalid model space header: {line}")
            
        n_p_core = int(parts[2])
        n_n_core = int(parts[3])
        n_p_orbits = int(parts[0])
        n_n_orbits = int(parts[1])
        
        orbits = SingleParticleOrbits(n_p_core, n_n_core, n_p_orbits, n_n_orbits)
        
        total_orbits = n_p_orbits + n_n_orbits
        for i in range(total_orbits):
            line = f.readline()
            while line and (line.strip().startswith('!') or not line.strip()):
                line = f.readline()
            
            # Line format: index n l 2j 2tz
            parts = line.split()
            if len(parts) < 5:
                raise ValueError(f"Invalid orbit definition at line {i+1}: {line}")
            
            # We don't necessarily need the index from the file if we rely on order
            orbits.n[i] = int(parts[1])
            orbits.l[i] = int(parts[2])
            orbits.j[i] = int(parts[3])
            orbits.jz[i] = orbits.j[i]
            orbits.tz[i] = int(parts[4])
            
        # Read 1-body interaction
        line = f.readline()
        while line and (line.strip().startswith('!') or not line.strip()):
            line = f.readline()
            
        if not line:
            return orbits, None
            
        # 1-body header: n_elements scaling_factor hw
        parts = line.split()
        n_elements = int(parts[0])
        
        potential = Potential(total_orbits)
        
        for _ in range(n_elements):
            line = f.readline()
            while line and (line.strip().startswith('!') or not line.strip()):
                line = f.readline()
            parts = line.split()
            # snt format is 1-indexed: i, j, value
            i_idx = int(parts[0]) - 1
            j_idx = int(parts[1]) - 1
            val = float(parts[2])
            potential.v1b[i_idx, j_idx] = val
            if i_idx != j_idx:
                potential.v1b[j_idx, i_idx] = val # Hermitian
                
        # Read 2-body interaction
        line = f.readline()
        while line and (line.strip().startswith('!') or not line.strip()):
            line = f.readline()
            
        if not line:
            return orbits, potential
            
        # 2-body header: n_elements scaling_factor hw
        parts = line.split()
        n_elements_2b = int(parts[0])
        # Scaling factor and hw might be repeated or different
        
        for _ in range(n_elements_2b):
            line = f.readline()
            while line and (line.strip().startswith('!') or not line.strip()):
                line = f.readline()
            parts = line.split()
            # snt format: i, j, k, l, J, value
            i = int(parts[0]) - 1
            j = int(parts[1]) - 1
            k = int(parts[2]) - 1
            l = int(parts[3]) - 1
            J = int(parts[4])
            val = float(parts[5])
            
            potential.v2b.append([i, j, k, l, J, val])
            # Typically shell model interactions are symmetric under (ij) <-> (kl)
            # and may have other symmetries, but we store what's in the file.
            
        return orbits, potential


def generate_m_scheme(coupled_orbits):
    """
    Generates an m-scheme basis from a j-coupled basis.
    Each orbit i is expanded into (2j_i + 1) states with 2jz = -2j, -2j+2, ..., 2j.
    """
    # Identify proton and neutron indices in the coupled basis
    p_indices = np.where(coupled_orbits.tz < 0)[0]
    n_indices = np.where(coupled_orbits.tz > 0)[0]
    
    # Calculate m-scheme orbit counts
    n_p_m = sum(coupled_orbits.j[p_indices] + 1)
    n_n_m = sum(coupled_orbits.j[n_indices] + 1)
    
    m_scheme = SingleParticleOrbits(
        coupled_orbits.n_p_core, 
        coupled_orbits.n_n_core, 
        n_p_m, 
        n_n_m
    )
    
    idx = 0
    # Process orbits in the same order as the coupled basis (usually protons then neutrons or mixed)
    for i in range(len(coupled_orbits.n)):
        two_j = coupled_orbits.j[i]
        # Generate 2jz values sorted by absolute value
        jz_values = sorted(range(-two_j, two_j + 1, 2), key=abs)
        for two_jz in jz_values:
            m_scheme.n[idx] = coupled_orbits.n[i]
            m_scheme.l[idx] = coupled_orbits.l[i]
            m_scheme.j[idx] = coupled_orbits.j[i]
            m_scheme.jz[idx] = two_jz
            m_scheme.tz[idx] = coupled_orbits.tz[i]
            m_scheme.coupling_map[idx] = i  # Store the source j-scheme index
            idx += 1
            
    return m_scheme


def get_j_index(m_scheme, m_idx):
    """Returns the j-scheme orbit index for a given m-scheme state index."""
    return m_scheme.coupling_map[m_idx]


def get_m_indices(m_scheme, j_idx):
    """Returns a list of m-scheme state indices that correspond to a given j-scheme orbit index."""
    return np.where(m_scheme.coupling_map == j_idx)[0]


def get_cg_cached(two_j1, two_m1, two_j2, two_m2, two_J, two_M):
    """Calculates Clebsch-Gordan coefficient with caching."""
    key = (two_j1, two_m1, two_j2, two_m2, two_J, two_M)
    if key in _cg_cache:
        return _cg_cache[key]
    
    if two_m1 + two_m2 != two_M:
        return 0.0
    
    # Sympy CG implementation
    cg_val = float(CG(S(two_j1)/2, S(two_m1)/2, S(two_j2)/2, S(two_m2)/2, S(two_J)/2, S(two_M)/2).doit())
    _cg_cache[key] = cg_val
    return cg_val


def decouple_1b(j_potential, m_scheme):
    """
    Decouples j-scheme 1-body interaction to m-scheme.
    v_ij (m-scheme) = v_rs (j-scheme) where r = get_j_index(i), s = get_j_index(j),
    with the restriction that jz[i] == jz[j] and tz[i] == tz[j].
    """
    m_potential = Potential(len(m_scheme.n))
    
    for i in range(m_potential.n_orbits):
        r = m_scheme.coupling_map[i]
        for j in range(m_potential.n_orbits):
            s = m_scheme.coupling_map[j]
            
            # Physical restriction: l, j, m_j, and isospin must be conserved for 1-body operators
            if (m_scheme.l[i] == m_scheme.l[j] and 
                m_scheme.j[i] == m_scheme.j[j] and 
                m_scheme.tz[i] == m_scheme.tz[j] and 
                m_scheme.jz[i] == m_scheme.jz[j]):
                m_potential.v1b[i, j] = j_potential.v1b[r, s]
                
    return m_potential


def decouple_2b(j_potential, m_scheme):
    """
    Decouples j-scheme 2-body interaction to m-scheme and returns a sparse matrix.
    Uses the Clebsch-Gordan expansion to populate the m-scheme matrix efficiently.
    SNT matrix elements are normalized and antisymmetrized: <(rs)J|V|(tu)J>_NAS
    Formula: <m_a m_b | V | m_c m_d > = sqrt(1+d_rs)*sqrt(1+d_tu) * sum_J C_ab^J * C_cd^J * <J|V|J>_NAS
    """
    n_m = len(m_scheme.n)
    # Using LIL for efficient buildup, will convert to CSR for performance
    v2b_sparse = lil_matrix((n_m**2, n_m**2))
    
    # Pre-calculate CG blocks for each orbit
    # A block for orbit 'r' is a list of (m_idx, jz, CG_coefficient) for each J, M
    # But since CG depends on j1, j2, J, M, we group by (r, s, J, M)
    print(f"Decoupling {len(j_potential.v2b)} TBMEs to sparse M-scheme...")
    
    # Group entries by (r, s, t, u, J) to process sum over J more efficiently?
    # Actually, SNT typically has them grouped anyway.
    
    for entry in j_potential.v2b:
        r, s, t, u, J, val = entry
        if abs(val) < 1e-15: continue
        two_J = 2 * J
        
        m_a_indices = get_m_indices(m_scheme, r)
        m_b_indices = get_m_indices(m_scheme, s)
        m_c_indices = get_m_indices(m_scheme, t)
        m_d_indices = get_m_indices(m_scheme, u)
        
        nas_factor = 1.0
        if r == s: nas_factor *= np.sqrt(2.0)
        if t == u: nas_factor *= np.sqrt(2.0)
        
        # We search for combinations (a,b) and (c,d) that couple to (J, M)
        # Iterate over possible M values for this J
        for two_M in range(-two_J, two_J + 1, 2):
            # Pre-filter m states for this M
            labels_ab = []
            cgs_ab = []
            for a_idx in m_a_indices:
                for b_idx in m_b_indices:
                    if m_scheme.jz[a_idx] + m_scheme.jz[b_idx] == two_M:
                        cg = get_cg_cached(m_scheme.j[a_idx], m_scheme.jz[a_idx], 
                                           m_scheme.j[b_idx], m_scheme.jz[b_idx], two_J, two_M)
                        if abs(cg) > 1e-10:
                            labels_ab.append((a_idx, b_idx))
                            cgs_ab.append(cg)
            
            if not labels_ab: continue
            
            labels_cd = []
            cgs_cd = []
            for c_idx in m_c_indices:
                for d_idx in m_d_indices:
                    if m_scheme.jz[c_idx] + m_scheme.jz[d_idx] == two_M:
                        cg = get_cg_cached(m_scheme.j[c_idx], m_scheme.jz[c_idx], 
                                           m_scheme.j[d_idx], m_scheme.jz[d_idx], two_J, two_M)
                        if abs(cg) > 1e-10:
                            labels_cd.append((c_idx, d_idx))
                            cgs_cd.append(cg)
            
            if not labels_cd: continue
            
            # Outer product of CGs scaled by val and nas_factor
            # V_m(a,b,c,d) += CG_ab * CG_cd * val * nas_factor
            v_base = val*((42./56)**(0.30)) * nas_factor
            for i, (a, b) in enumerate(labels_ab):
                idx_ab = a * n_m + b
                idx_ba = b * n_m + a
                cg_i = cgs_ab[i]
                for j, (c, d) in enumerate(labels_cd):
                    idx_cd = c * n_m + d
                    idx_dc = d * n_m + c
                    cg_j = cgs_cd[j]
                    
                    v_val = cg_i * cg_j * v_base
                    
                    # Store only non-zero unique permutations
                    # This logic handles r!=s and t!=u correctly via the nested loops
                    # We just need to populate the transposed blocks
                    
                    # Direct term
                    v2b_sparse[idx_ab, idx_cd] += v_val
                    
                    # Orbit permutations (if orbits distinct)
                    if r != s: v2b_sparse[idx_ba, idx_cd] -= v_val
                    if t != u: v2b_sparse[idx_ab, idx_dc] -= v_val
                    if r != s and t != u: v2b_sparse[idx_ba, idx_dc] += v_val
                    
                    # Hermiticity
                    if r != t or s != u:
                        v2b_sparse[idx_cd, idx_ab] += v_val
                        if r != s: v2b_sparse[idx_cd, idx_ba] -= v_val
                        if t != u: v2b_sparse[idx_dc, idx_ab] -= v_val
                        if r != s and t != u: v2b_sparse[idx_dc, idx_ba] += v_val
                            
    return v2b_sparse.tocsr()


def sparse_to_dense_4d(v2b_sparse, n_m):
    """Utility to convert flattened sparse interaction to 4D dense array."""
    dense = np.zeros((n_m, n_m, n_m, n_m))
    row, col = v2b_sparse.nonzero()
    data = v2b_sparse.data
    for i in range(len(data)):
        a, b = divmod(row[i], n_m)
        c, d = divmod(col[i], n_m)
        dense[a, b, c, d] = data[i]
    return dense


def recouple_2b(v2b_m, m_scheme, j_potential):
    """
    Inverse: <J|V|J>_NAS = [ 1 / (sqrt(1+d_rs)*sqrt(1+d_tu)) ] * sum_m C_ab^J * C_cd^J * <m_a m_b|V|m_c m_d>
    """
    n_m = len(m_scheme.n)
    j_elements = []
    for entry in j_potential.v2b:
        r, s, t, u, J, _ = entry
        two_J = 2 * J
        
        m_a_indices = get_m_indices(m_scheme, r)
        m_b_indices = get_m_indices(m_scheme, s)
        m_c_indices = get_m_indices(m_scheme, t)
        m_d_indices = get_m_indices(m_scheme, u)
        
        nas_factor = 1.0
        if r == s: nas_factor *= np.sqrt(2.0)
        if t == u: nas_factor *= np.sqrt(2.0)
        
        # Scalar operator: <J||V||J> is independent of M. We can take M=J for simplicity or average.
        # Let's sum over all M and divide by (2J+1)
        total_val = 0.0
        for two_M in range(-two_J, two_J + 1, 2):
            sum_m = 0.0
            for a_idx in m_a_indices:
                two_ma = m_scheme.jz[a_idx]
                for b_idx in m_b_indices:
                    two_mb = m_scheme.jz[b_idx]
                    if two_ma + two_mb != two_M: continue
                    cg_ab = get_cg_cached(m_scheme.j[a_idx], two_ma, m_scheme.j[b_idx], two_mb, two_J, two_M)
                    if abs(cg_ab) < 1e-10: continue
                    
                    for c_idx in m_c_indices:
                        two_mc = m_scheme.jz[c_idx]
                        for d_idx in m_d_indices:
                            two_md = m_scheme.jz[d_idx]
                            if two_mc + two_md != two_M: continue
                            cg_cd = get_cg_cached(m_scheme.j[c_idx], two_mc, m_scheme.j[d_idx], two_md, two_J, two_M)
                            if abs(cg_cd) < 1e-10: continue
                            
                            idx_ab = a_idx * n_m + b_idx
                            idx_cd = c_idx * n_m + d_idx
                            sum_m += cg_ab * cg_cd * v2b_m[idx_ab, idx_cd]
            total_val += sum_m
            
        final_val = (total_val / nas_factor) / (J * 2 + 1)
        j_elements.append([r, s, t, u, J, final_val])
        
    return j_elements
