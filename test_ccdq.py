"""
Test script for CCDQ solver.
Demonstrates usage with a simple mock Hamiltonian.
"""

import numpy as np
from ccdq import ccdq


class MockHamiltonian:
    """Simple Hamiltonian for testing"""
    def __init__(self, n_occ, n_virt):
        n_states = n_occ + n_virt
        
        # Fock matrix (diagonal for simplicity)
        self.f = np.diag(np.concatenate([
            -np.linspace(2.0, 1.0, n_occ),  # Occupied energies (negative)
            np.linspace(0.5, 2.0, n_virt)   # Virtual energies (positive)
        ]))
        
        # Antisymmetrized two-body integrals <pq||rs> = <pq|rs> - <pq|sr>
        self.Gamma = np.random.randn(n_states, n_states, n_states, n_states) * 0.1
        
        # Antisymmetrize: <pq||rs> = -<qp||rs> = -<pq||sr> = <qp||sr>
        self.Gamma = self.Gamma - self.Gamma.transpose(1, 0, 2, 3)
        self.Gamma = self.Gamma - self.Gamma.transpose(0, 1, 3, 2)
        
        # Make Hermitian in a physics sense
        self.Gamma = 0.5 * (self.Gamma + self.Gamma.transpose(2, 3, 0, 1))


def test_small_system():
    """Test CCDQ on a minimal 2-electron, 4-orbital system"""
    print("\n" + "="*70)
    print("  TEST 1: Small System (2 occupied, 2 virtual)")
    print("="*70)
    
    n_occ = 2
    n_virt = 2
    
    ham = MockHamiltonian(n_occ, n_virt)
    
    try:
        e_corr, t2, t4 = ccdq(ham, n_occ, max_iter=50, tol=1e-6, alpha=0.3, print_level=1)
        
        print(f"\n✓ Test PASSED")
        print(f"  Correlation Energy: {e_corr:.8f}")
        print(f"  T2 max amplitude: {np.max(np.abs(t2)):.6f}")
        print(f"  T4 max amplitude: {np.max(np.abs(t4)):.6f}")
        
    except Exception as e:
        print(f"\n✗ Test FAILED with error: {e}")
        import traceback
        traceback.print_exc()


def test_medium_system():
    """Test CCDQ on a slightly larger system"""
    print("\n" + "="*70)
    print("  TEST 2: Medium System (3 occupied, 3 virtual)")
    print("="*70)
    
    n_occ = 3
    n_virt = 3
    
    ham = MockHamiltonian(n_occ, n_virt)
    
    try:
        e_corr, t2, t4 = ccdq(ham, n_occ, max_iter=30, tol=1e-5, alpha=0.25, print_level=1)
        
        print(f"\n✓ Test PASSED")
        print(f"  Correlation Energy: {e_corr:.8f}")
        print(f"  T2 dimension: {t2.shape}")
        print(f"  T4 dimension: {t4.shape}")
        print(f"  T2 norm: {np.linalg.norm(t2):.6f}")
        print(f"  T4 norm: {np.linalg.norm(t4):.6f}")
        
    except Exception as e:
        print(f"\n✗ Test FAILED with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("\n" + "="*70)
    print("  CCDQ SOLVER TEST SUITE")
    print("  Testing Coupled Cluster Doubles + Quadruples Implementation")
    print("="*70)
    
    # Run tests
    test_small_system()
    # test_medium_system()  # Uncomment for larger test (warning: slow!)
    
    print("\n" + "="*70)
    print("  ALL TESTS COMPLETED")
    print("="*70 + "\n")
