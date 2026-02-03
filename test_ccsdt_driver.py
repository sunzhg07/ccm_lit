
import numpy as np
import time
from ccsdt import ccsdt

class MockHamiltonian:
    def __init__(self, n_orb):
        # Create a random Fock matrix (diagonal for simplicity)
        self.f = np.diag(np.linspace(-10, 10, n_orb))
        
        # Create random antisymmetrized 2-body integrals <pq||rs>
        # Ensure antisymmetry: Gamma[p,q,r,s] = -Gamma[q,p,r,s] = -Gamma[p,q,s,r]
        dim = (n_orb, n_orb, n_orb, n_orb)
        g = np.random.rand(*dim) * 0.1
        g = g - g.transpose(1, 0, 2, 3)
        g = g - g.transpose(0, 1, 3, 2)
        self.Gamma = g

def test_ccsdt_driver():
    print("Testing CCSDT implementation...")
    
    # Small system for fast testing: 4 holes, 4 particles -> 8 orbitals
    n_occ = 2
    n_virt = 2
    n_orb = n_occ + n_virt
    
    ham = MockHamiltonian(n_orb)
    
    start_time = time.time()
    try:
        e_corr, t1, t2, t3 = ccsdt(ham, n_occ, max_iter=5, tol=1e-5)
        print(f"\nTest Completed Successfully!")
        print(f"Final CCDST Correlation Energy: {e_corr:.8f}")
        print(f"Time taken: {time.time() - start_time:.4f}s")
        
        # Basic sanity checks
        assert not np.isnan(e_corr), "Energy is NaN"
        assert t1.shape == (n_occ, n_virt)
        assert t2.shape == (n_occ, n_occ, n_virt, n_virt)
        assert t3.shape == (n_occ, n_occ, n_occ, n_virt, n_virt, n_virt)
        
    except Exception as e:
        print(f"\nTest FAILED with error:")
        print(e)
        raise e

if __name__ == "__main__":
    test_ccsdt_driver()
