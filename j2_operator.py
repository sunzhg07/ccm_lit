import numpy as np
from read_snt_io import Potential, SingleParticleOrbits, decouple_1b, decouple_2b, generate_m_scheme

def generate_j2_j_scheme(orbits: SingleParticleOrbits):
    """
    Generates the J^2 operator in j-scheme.
    J^2 = sum_i j_i^2 + 2 sum_{i < j} j_i . j_j
    
    1-body part: j_i(j_i+1)
    2-body part: J(J+1) - j_i(j_i+1) - j_j(j_j+1)
    
    Note: orbits.j[i] stores 2j, and potential.v2b elements use J (not 2J).
    The J^2 operator is isoscalar and only connects states with the same T_z and T.
    """
    n_orbits = len(orbits.n)
    potential = Potential(n_orbits)
    
    # 1-body part: sum_i j_i(j_i+1)
    for i in range(n_orbits):
        ji = orbits.j[i] / 2.0
        potential.v1b[i, i] = ji * (ji + 1.0)
        
    p_indices = np.where(orbits.tz < 0)[0]
    n_indices = np.where(orbits.tz > 0)[0]
    
    def add_tbmes(idx_list1, idx_list2):
        for r in idx_list1:
            for s in idx_list2:
                # For blocks where particles are of the same type, we avoid double counting and handle r=s
                if idx_list1 is idx_list2 and r > s:
                    continue
                
                jr = orbits.j[r] / 2.0
                js = orbits.j[s] / 2.0
                two_jr = orbits.j[r]
                two_js = orbits.j[s]
                
                # J ranges from |jr - js| to jr + js
                for two_J in range(abs(two_jr - two_js), (two_jr + two_js) + 1, 2):
                    J_val = (two_J // 2)
                    
                    # For identical particles in the same orbit, J must be even
                    if r == s and J_val % 2 != 0:
                        continue
                    
                    # TBME value in NAS format
                    # <rs J | V | rs J>_NAS = J(J+1) - j_r(j_r+1) - j_s(j_s+1)
                    val = J_val * (J_val + 1.0) - jr * (jr + 1.0) - js * (js + 1.0)
                    
                    # SNT format: i, j, k, l, J, value
                    potential.v2b.append([r, s, r, s, J_val, val])

    # pp sector
    add_tbmes(p_indices, p_indices)
    # nn sector
    add_tbmes(n_indices, n_indices)
    # pn sector
    add_tbmes(p_indices, n_indices)
    
    return potential

def get_j2_m_scheme(orbits: SingleParticleOrbits):
    """
    Generates J^2 in j-scheme and then transforms it to m-scheme.
    Returns:
        m_scheme: SingleParticleOrbits object for m-scheme
        v1b_m: 1-body part in m-scheme (dense matrix)
        v2b_m: 2-body part in m-scheme (sparse matrix in flattened (n_m^2, n_m^2) form)
    """
    # 1. Generate J^2 in j-scheme
    j_potential = generate_j2_j_scheme(orbits)
    
    # 2. Generate m-scheme basis
    m_scheme = generate_m_scheme(orbits)
    
    # 3. Decouple 1-body part
    m_potential_obj = decouple_1b(j_potential, m_scheme)
    v1b_m = m_potential_obj.v1b
    
    # 4. Decouple 2-body part
    v2b_m_sparse = decouple_2b(j_potential, m_scheme)
    
    return m_scheme, v1b_m, v2b_m_sparse

if __name__ == "__main__":
    from read_snt_io import read_snt
    import os
    
    # Test with sd.snt if it exists
    snt_file = "sd.snt"
    if os.path.exists(snt_file):
        print(f"Testing with {snt_file}...")
        orbits, _ = read_snt(snt_file)
        m_scheme, v1b_m, v2b_m_sparse = get_j2_m_scheme(orbits)
        
        print(f"M-scheme basis size: {len(m_scheme.n)}")
        print(f"M-scheme 1-body non-zero: {np.count_nonzero(v1b_m)}")
        print(f"M-scheme 2-body non-zero: {v2b_m_sparse.count_nonzero()}")
        
        # Verify tracing? 
        # For a closed shell or simple system, we could check eigenvalues.
    else:
        print("sd.snt not found, skipping test.")
