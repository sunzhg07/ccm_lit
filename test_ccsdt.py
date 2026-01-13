"""
Test script for CCSDT implementation in cc.py

This script demonstrates how to use the CCSDT solver.
For a small test case, you can call it from main.py or use this as a reference.
"""

import numpy as np
from cc import ccsdt, ccsd, mp2

# Example usage (you'll need to integrate with your actual Hamiltonian)
def test_ccsdt():
    """
    Example of how to call CCSDT
    
    You'll need to provide:
    - no_ham: A Hamiltonian object with attributes:
        - .f : Fock matrix (1-body operator)
        - .Gamma : 2-body interaction tensor
    - n_occ: Number of occupied orbitals
    """
    
    print("=" * 60)
    print("CCSDT Test Example")
    print("=" * 60)
    print("\nTo use CCSDT in your main code:")
    print("1. Perform Hartree-Fock to get normal-ordered Hamiltonian")
    print("2. Call: e_corr, t1, t2, t3 = ccsdt(no_ham, n_occ)")
    print("3. Total energy = E_HF + e_corr")
    print("\nIMPORTANT NOTES:")
    print("- CCSDT scales as O(N^8) in memory for T3 amplitudes")
    print("- Only use for small systems (< 10 particles, small basis)")
    print("- For larger systems, use CCSD or CCSD(T)")
    print("- Default damping alpha=0.3 is conservative for stability")
    print("=" * 60)
    
    # Example with dummy small system
    n_states = 6  # 3 occupied, 3 virtual
    n_occ = 3
    
    # Create a simple mock Hamiltonian for testing structure
    class MockHamiltonian:
        def __init__(self, n_states):
            self.f = np.diag(np.arange(n_states))  # Dummy Fock
            self.Gamma = np.random.randn(n_states, n_states, n_states, n_states) * 0.1
            # Make it antisymmetric
            for i in range(n_states):
                for j in range(n_states):
                    for k in range(n_states):
                        for l in range(n_states):
                            self.Gamma[i,j,k,l] = -self.Gamma[j,i,k,l]
                            self.Gamma[i,j,k,l] = -self.Gamma[i,j,l,k]
    
    print(f"\nRunning CCSDT on mock {n_states}-state system ({n_occ} occupied)...")
    no_ham = MockHamiltonian(n_states)
    
    # This will run but won't give physical results (just a test)
    e_corr, t1, t2, t3 = ccsdt(no_ham, n_occ, max_iter=5, tol=1e-6, alpha=0.5)
    
    print(f"\n[RESULT] Correlation Energy: {e_corr:.8f}")
    print(f"[RESULT] T1 shape: {t1.shape}, norm: {np.linalg.norm(t1):.4e}")
    print(f"[RESULT] T2 shape: {t2.shape}, norm: {np.linalg.norm(t2):.4e}")
    print(f"[RESULT] T3 shape: {t3.shape}, norm: {np.linalg.norm(t3):.4e}")


if __name__ == "__main__":
    test_ccsdt()
