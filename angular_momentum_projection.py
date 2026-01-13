import numpy as np
from sympy.physics.wigner import wigner_d
from sympy import S, N as sympy_N
from opt_einsum import contract


def compute_rotation_matrix(beta, j_values, jz_values):
    """
    Compute the Wigner D-matrix elements for rotation by angle beta around y-axis.
    
    D^j_{m',m}(0, beta, 0) = d^j_{m',m}(beta)
    
    Args:
        beta: Rotation angle (Euler angle)
        j_values: Array of 2j values for each state
        jz_values: Array of 2jz values for each state
        
    Returns:
        D: Rotation matrix (n_states, n_states)
    """
    n_states = len(j_values)
    D = np.zeros((n_states, n_states), dtype=complex)
    
    # Group states by j quantum number
    for i in range(n_states):
        two_j_i = j_values[i]
        two_jz_i = jz_values[i]
        
        for j in range(n_states):
            two_j_j = j_values[j]
            two_jz_j = jz_values[j]
            
            # Only same j can mix under rotation
            if two_j_i != two_j_j:
                continue
            
            # Compute Wigner d-matrix element
            j_val = two_j_i / 2.0
            m_prime = two_jz_i / 2.0
            m = two_jz_j / 2.0
            
            # Use sympy's wigner_d function
            d_elem = wigner_d(S(two_j_i)/2, S(two_jz_i)/2, S(two_jz_j)/2, beta)
            D[i, j] = complex(sympy_N(d_elem))
    
    return D


def project_angular_momentum(m_basis, rho, target_J, target_Jz=0, n_angles=16):
    """
    Project the density matrix onto a state of good angular momentum J.
    
    Uses the Peierls-Yoccoz projection:
    P^J_{MK} = (2J+1)/(8π²) ∫ dΩ D^J*_{MK}(Ω) R(Ω)
    
    For axially symmetric case (K=M), this simplifies to integration over β only.
    
    Args:
        m_basis: M-scheme basis information
        rho: Density matrix in M-scheme (n_states, n_states)
        target_J: Target total angular momentum (2J value)
        target_Jz: Target Jz projection (2Jz value)
        n_angles: Number of quadrature points for integration
        
    Returns:
        rho_proj: Projected density matrix
        norm: Norm of the projected state
    """
    n_states = len(m_basis.n)
    
    # Get quantum numbers for each state
    j_values = m_basis.j
    jz_values = m_basis.jz
    
    # Integration over Euler angle beta (0 to pi)
    # Using Gauss-Legendre quadrature
    beta_points, weights = np.polynomial.legendre.leggauss(n_angles)
    # Map from [-1, 1] to [0, pi]
    beta_points = (beta_points + 1) * np.pi / 2
    weights = weights * np.pi / 2
    
    rho_proj = np.zeros((n_states, n_states), dtype=complex)
    norm = 0.0
    
    J = target_J / 2.0
    M = target_Jz / 2.0
    K = M  # Axially symmetric case
    
    print(f"\n[Angular Momentum Projection]")
    print(f"Target J = {J:.1f}, Jz = {M:.1f}")
    print(f"Using {n_angles} quadrature points")
    
    for idx, (beta, weight) in enumerate(zip(beta_points, weights)):
        # Compute rotation operator R(beta)
        D = compute_rotation_matrix(beta, j_values, jz_values)
        
        # Rotated density: rho' = D^† rho D
        rho_rot = D.conj().T @ rho @ D
        
        # Wigner D-function weight
        d_weight = wigner_d(S(target_J)/2, S(target_Jz)/2, S(target_Jz)/2, beta)
        d_weight_val = complex(sympy_N(d_weight))
        
        # Accumulate projection
        # Factor: (2J+1)/(8π²) → for β-only integration: (2J+1)/(4π)
        factor = (target_J + 1) / (4 * np.pi) * weight
        rho_proj += factor * d_weight_val.conjugate() * rho_rot
        
        if idx % 4 == 0:
            print(f"  β = {beta:.4f}, weight = {weight:.4f}, D-weight = {abs(d_weight_val):.4f}")
    
    # Compute norm
    norm = np.trace(rho_proj).real
    
    # Normalize
    if norm > 1e-10:
        rho_proj = rho_proj / norm
    else:
        print(f"[WARNING] Projection norm is very small: {norm:.6e}")
        print(f"  The reference state may have negligible overlap with J={J}")
    
    return rho_proj, norm


def compute_projected_energy(no_ham, rho_proj, n_occ):
    """
    Compute the energy of the projected state.
    
    E = Tr[rho_proj * h] + 0.5 * Tr[rho_proj * Gamma * rho_proj]
    
    Args:
        no_ham: Normal-ordered Hamiltonian
        rho_proj: Projected density matrix
        n_occ: Number of occupied states
        
    Returns:
        energy: Projected energy
    """
    n_states = no_ham.f.shape[0]
    
    # One-body contribution
    e1 = np.trace(rho_proj @ no_ham.f).real
    
    # Two-body contribution (simplified)
    # E2 = 0.5 * sum_ijkl rho_ij rho_kl <ij||kl>
    # This is expensive for large systems, so we use a trace formula
    
    # For occupied-occupied block
    o = slice(0, n_occ)
    rho_oo = rho_proj[o, o]
    
    # Approximate 2-body energy using occupied block
    e2 = 0.0
    for i in range(n_occ):
        for j in range(n_occ):
            for k in range(n_occ):
                for l in range(n_occ):
                    e2 += 0.5 * rho_oo[i,j] * rho_oo[k,l] * no_ham.Gamma[i,j,k,l]
    
    e2 = e2.real
    
    total_energy = no_ham.E0 + e1 + e2
    
    return total_energy


def analyze_angular_momentum_content(m_basis, rho, max_J=10):
    """
    Analyze the angular momentum content of a density matrix.
    
    Computes the overlap with projected states for different J values.
    
    Args:
        m_basis: M-scheme basis
        rho: Density matrix
        max_J: Maximum J to analyze (2J value)
        
    Returns:
        j_distribution: Dictionary mapping 2J -> overlap
    """
    print("\n[Angular Momentum Analysis]")
    print(f"{'2J':>4} | {'J':>6} | {'Overlap':>12} | {'Percentage':>10}")
    print("-" * 45)
    
    j_distribution = {}
    total_norm = 0.0
    
    for two_J in range(0, max_J + 1, 2):  # Even J for even number of particles
        try:
            _, norm = project_angular_momentum(m_basis, rho, two_J, target_Jz=0, n_angles=8)
            j_distribution[two_J] = norm
            total_norm += norm
        except Exception as e:
            print(f"[WARNING] Failed to project J={two_J/2}: {e}")
            j_distribution[two_J] = 0.0
    
    # Normalize and display
    for two_J, norm in sorted(j_distribution.items()):
        J = two_J / 2.0
        percentage = (norm / total_norm * 100) if total_norm > 0 else 0.0
        print(f"{two_J:4d} | {J:6.1f} | {norm:12.6f} | {percentage:9.2f}%")
    
    return j_distribution


def project_ccsd_state(m_basis, no_ham, t1, t2, l1, l2, n_occ, target_J):
    """
    Project the CCSD correlated state onto good angular momentum.
    
    This uses the one-body density matrix from CCSD as an approximation.
    
    Args:
        m_basis: M-scheme basis
        no_ham: Normal-ordered Hamiltonian
        t1, t2: T amplitudes
        l1, l2: Lambda amplitudes
        n_occ: Number of occupied states
        target_J: Target angular momentum (2J value)
        
    Returns:
        rho_proj: Projected density matrix
        energy_proj: Projected energy
        norm: Projection norm
    """
    n_states = no_ham.f.shape[0]
    
    # Construct approximate one-body density matrix from CCSD
    rho = np.zeros((n_states, n_states))
    
    # Occupied block: delta_ij - corrections
    o = slice(0, n_occ)
    v = slice(n_occ, n_states)
    
    rho[o, o] = np.eye(n_occ)
    
    # Add correlation corrections from Lambda amplitudes
    # rho_ij -= 0.5 * sum_ef l2_ijef * t2_ijef
    if l2 is not None and t2 is not None:
        for i in range(n_occ):
            for j in range(n_occ):
                correction = 0.5 * np.sum(l2[i, j, :, :] * t2[i, j, :, :])
                rho[i, j] -= correction
    
    # Virtual block: sum_ij l2_ijab * t2_ijab
    if l2 is not None and t2 is not None:
        for a in range(n_states - n_occ):
            for b in range(n_states - n_occ):
                rho[n_occ + a, n_occ + b] = 0.5 * np.sum(l2[:, :, a, b] * t2[:, :, a, b])
    
    # Project onto target J
    rho_proj, norm = project_angular_momentum(m_basis, rho, target_J, target_Jz=0, n_angles=16)
    
    # Compute projected energy
    energy_proj = compute_projected_energy(no_ham, rho_proj, n_occ)
    
    return rho_proj, energy_proj, norm
