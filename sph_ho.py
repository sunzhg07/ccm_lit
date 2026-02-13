import numpy as np
from scipy.special import genlaguerre, gammaln
from sympy.physics.wigner import wigner_3j
from math import sqrt, pi, exp, log

# --- 1. Radial Functions ---

def ho_radial_function(r, n, l, b):
    """
    Harmonic Oscillator radial wavefunction R_nl(r).
    n: Radial quantum number (0, 1, 2...) [count of nodes]
    l: Orbital angular momentum
    b: Oscillator length (fm)
    """
    # Normalization constant N_nl calculation using log-gamma for stability
    # N = sqrt( 2 * n! / b^3 / Gamma(n + l + 1.5) )
    log_N = 0.5 * (log(2) + gammaln(n + 1) - 3*log(b) - gammaln(n + l + 1.5))
    N_nl = exp(log_N)
    
    x = (r / b)**2
    # Generalized Laguerre Polynomial L_n^(alpha)
    laguerre = genlaguerre(n, l + 0.5)(x)
    
    return N_nl * np.exp(-x / 2) * (r / b)**l * laguerre

def radial_integral_r2(n1, l1, n2, l2, b):
    """
    Computes radial integral <n1 l1 | r^2 | n2 l2>
    """
    from scipy.integrate import quad
    
    # Integrand = R_n1l1(r) * r^2 * R_n2l2(r) * r^2 (jacobian)
    integrand = lambda r: ho_radial_function(r, n1, l1, b) * \
                          ho_radial_function(r, n2, l2, b) * \
                          r**4  # r^2 operator * r^2 jacobian
    
    # Integrate from 0 to essentially infinity (15*b is sufficient for HO decay)
    result, _ = quad(integrand, 0, 15*b)
    return result

# --- 2. Angular Functions (General Y_LM) ---

def angular_matrix_element(j1, m1, j2, m2, L, M):
    """
    Computes <j1 m1 | Y_LM | j2 m2> using Wigner-Eckart theorem.
    """
    # Selection Rule 1: m1 = m2 + M
    # Floating point comparison tolerance included
    if abs(m1 - (m2 + M)) > 1e-9:
        return 0.0

    # Reduced Matrix Element <j1 || Y_L || j2>
    # Formula: (-1)^(j1+1/2) * sqrt((2j1+1)(2j2+1)(2L+1)/4pi) * (j1 L j2)
    #                                                           (1/2 0 -1/2)
    
    phase_red = (-1)**(int(j1 + 0.5))
    prefactor = sqrt( (2*j1 + 1) * (2*j2 + 1) * (2*L + 1) / (4 * pi) )
    w3j_red = wigner_3j(j1, L, j2, 0.5, 0, -0.5)
    
    reduced_me = phase_red * prefactor * float(w3j_red)
    
    # Full Element: (-1)^(j1-m1) * (j1 L j2) * <j1 || Y_L || j2>
    #                              (-m1 M m2)
    phase_we = (-1)**(int(j1 - m1))
    w3j_we = wigner_3j(j1, L, j2, -m1, M, m2)
    
    return phase_we * float(w3j_we) * reduced_me

# --- 3. Main Wrapper ---

def compute_matrix_elements(n1, l1, j1, m1, n2, l2, j2, m2, hw):
    """
    Returns a dictionary with matrix elements for Q20, Q22, and Q00.
    """
    b= np.sqrt(41.47/hw)
    # Parity Check: All these operators are even parity (L=0, 2).
    # If (-1)^l1 != (-1)^l2, all are zero.
    if (l1 % 2) != (l2 % 2):
        return {"Q20": 0.0, "Q22": 0.0, "Q00": 0.0}

    # Calculate Radial Part (common for Q20 and Q22, same r^2 dependence)
    # Q00 also uses r^2 radial part.
    rad_int = radial_integral_r2(n1, l1, n2, l2, b)
    
    # -- Q20 (L=2, M=0) --
    # Convention: prolate (high |jz|) → positive Q20, oblate (low |jz|) → negative Q20
    ang_20 = angular_matrix_element(j1, m1, j2, m2, 2, 0)
    q20 = -rad_int * ang_20
    
    # -- Q22 (L=2, M=2) --
    ang_22 = angular_matrix_element(j1, m1, j2, m2, 2, 2)
    q22 = rad_int * ang_22
    
    # -- Q00 / Monopole (L=0, M=0) --
    # Note: Angular part for Y_00 is delta_j1j2 * delta_m1m2 / sqrt(4pi)
    # Usually we define Q0 = r^2 (without Y00), but here we calculate r^2 Y_00
    ang_00 = angular_matrix_element(j1, m1, j2, m2, 0, 0)
    q00 = rad_int * ang_00

    return {
        "Q20": q20,
        "Q22": q22,  # Complex component r^2 Y_22
        "Q00": q00   # Monopole r^2 Y_00
    }

# --- Example Execution ---

if __name__ == "__main__":
    # Example: 
    # Bra <1|: 0d3/2, m=3/2  (n=0, l=2, j=1.5, m=1.5)
    # Ket |2>: 0s1/2, m=-1/2 (n=0, l=0, j=0.5, m=-0.5)
    
    # This transition satisfies Q22 selection (Delta m = 2)
    
    res = compute_matrix_elements(
        n1=0, l1=2, j1=1.5, m1=1.5,
        n2=0, l2=0, j2=0.5, m2=-0.5,
        hw=20
    )
    
    print(f"Transition: <0d3/2 m=3/2 | O | 0s1/2 m=-1/2>")
    print("-" * 40)
    print(f"Q20 (Axial)    : {res['Q20']:.6f}  (Should be 0, Delta m != 0)")
    print(f"Q22 (Non-axial): {res['Q22']:.6f}  (Allowed, Delta m = 2)")
    print(f"Q00 (Monopole) : {res['Q00']:.6f}  (Should be 0, Delta l != 0)")
