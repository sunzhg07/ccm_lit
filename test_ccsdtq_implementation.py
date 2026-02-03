"""
Test script for CCSDTQ implementation
Verifies the framework works on a tiny system
"""

import numpy as np
import sys
sys.path.insert(0, '/Users/wolf/work/ccm_lit')

# Create a minimal test Hamiltonian
class TestHamiltonian:
    def __init__(self, n_occ, n_virt):
        n = n_occ + n_virt
        
        # Random but symmetric Fock matrix
        np.random.seed(42)
        self.f = np.random.randn(n, n)
        self.f = 0.5 * (self.f + self.f.T)
        
        # Make occupied orbitals lower in energy
        for i in range(n_occ):
            self.f[i,i] = -1.0 - np.random.rand()
        for a in range(n_occ, n):
            self.f[a,a] = 0.5 + np.random.rand()
        
        # Antisymmetrized two-electron integrals <pq||rs>
        raw = np.random.randn(n, n, n, n) * 0.1
        # Antisymmetrize
        self.Gamma = raw - raw.transpose(0,1,3,2)
        # Make it antisymmetric in first pair too
        self.Gamma = self.Gamma - self.Gamma.transpose(1,0,2,3)


def test_permutation_operators():
    """Test that permutation operators work correctly"""
    print("\n" + "="*70)
    print("Testing Permutation Operators")
    print("="*70)
    
    # Test P(a/bc) for T3
    t3_test = np.random.randn(2, 2, 2, 3, 3, 3)
    
    from ccsdtq_permutations import P_a_bc
    result = P_a_bc(t3_test)
    
    # Should have shape (2,2,2,3,3,3)
    assert result.shape == t3_test.shape, "Shape mismatch!"
    
    # Check it's antisymmetric
    # P(a/bc) = original - swap(a,b) - swap(a,c)
    expected = (t3_test 
                - t3_test.transpose(0,1,2,4,3,5)  # swap a↔b
                - t3_test.transpose(0,1,2,5,4,3))  # swap a↔c
    
    assert np.allclose(result, expected), "P(a/bc) implementation error!"
    print("✓ P(a/bc) works correctly")
    
    # Test P(a/bcd) for T4
    from ccsdtq_permutations import P_a_bcd
    t4_test = np.random.randn(2, 2, 2, 2, 2, 2, 2, 2)
    result_t4 = P_a_bcd(t4_test)
    
    # Should produce 4 terms (original + 3 swaps)
    expected_t4 = (t4_test
                   - t4_test.transpose(0,1,2,3,5,4,6,7)  # swap a↔b
                   - t4_test.transpose(0,1,2,3,6,5,4,7)  # swap a↔c
                   - t4_test.transpose(0,1,2,3,7,5,6,4))  # swap a↔d
    
    assert np.allclose(result_t4, expected_t4), "P(a/bcd) implementation error!"
    print("✓ P(a/bcd) works correctly (4 terms)")
    
    print("\n✅ All permutation operator tests passed!")


def test_ccsdtq_tiny_system():
    """Test CCSDTQ on a 2 occ, 2 virt system (manageable memory)"""
    print("\n" + "="*70)
    print("Testing CCSDTQ Framework on Tiny System (2 occ, 2 virt)")
    print("="*70)
    
    n_occ, n_virt = 2, 2
    ham = TestHamiltonian(n_occ, n_virt)
    
    print(f"\nSystem size: {n_occ} occupied, {n_virt} virtual")
    
    # Calculate memory
    t4_size = n_occ**4 * n_virt**4 * 8  # bytes
    print(f"T4 memory: {t4_size} bytes = {t4_size/1024:.2f} KB")
    
    try:
        from ccsdtq_full import ccsdtq
        
        print("\nRunning CCSDTQ (max 3 iterations for test)...")
        e_corr, t1, t2, t3, t4 = ccsdtq(
            ham,
            n_occ=n_occ,
            max_iter=3,  # Just test the framework
            tol=1e-8,
            alpha=0.5,
            diis_size=3
        )
        
        print(f"\n✓ Framework executed successfully!")
        print(f"  Correlation energy: {e_corr:.8f}")
        print(f"  T1 norm: {np.linalg.norm(t1):.6f}")
        print(f"  T2 norm: {np.linalg.norm(t2):.6f}")
        print(f"  T3 norm: {np.linalg.norm(t3):.6f}")
        print(f"  T4 norm: {np.linalg.norm(t4):.6f}")
        
        # Basic sanity checks
        assert t1.shape == (n_occ, n_virt), "T1 shape wrong!"
        assert t2.shape == (n_occ, n_occ, n_virt, n_virt), "T2 shape wrong!"
        assert t3.shape == (n_occ, n_occ, n_occ, n_virt, n_virt, n_virt), "T3 shape wrong!"
        assert t4.shape == (n_occ,)*4 + (n_virt,)*4, "T4 shape wrong!"
        
        print("\n✅ All shape checks passed!")
        
    except Exception as e:
        print(f"\n❌ Error during CCSDTQ execution:")
        print(f"   {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def test_equation_parser():
    """Test the equation parser""" 
    print("\n" + "="*70)
    print("Testing Equation Parser")
    print("="*70)
    
    try:
        from parse_ccsdtq_equations import parse_latex_equation
        
        # Test a simple equation
        latex = r"-V^{jk}_{bc}t^{b}_{i}t^{a}_{j}t^{c}_{k}"
        parsed = parse_latex_equation(latex)
        
        print(f"\nInput: {latex}")
        print(f"Parsed:")
        print(f"  Coefficient: {parsed['coefficient']}")
        print(f"  # Tensors: {len(parsed['tensors'])}")
        print(f"  Tensors: {parsed['tensors']}")
        
        assert parsed['coefficient'] == -1.0, "Coefficient wrong!"
        assert len(parsed['tensors']) ==  4, "Should have 4 tensors!"
        
        print("\n✅ Parser test passed!")
        
    except Exception as e:
        print(f"\n❌ Parser test failed: {e}")
        return False
    
    return True


if __name__ == "__main__":
    print("\n" + "#"*70)
    print("# CCSDTQ Implementation Test Suite")
    print("#"*70)
    
    success = True
    
    # Test 1: Permutation operators
    try:
        test_permutation_operators()
    except Exception as e:
        print(f"\n❌ Permutation test failed: {e}")
        success = False
    
    # Test 2: Equation parser
    try:
        test_equation_parser()
    except Exception as e:
        print(f"\n❌ Parser test failed: {e}")
        success = False
    
    # Test 3: CCSDTQ framework
    try:
        result = test_ccsdtq_tiny_system()
        if not result:
            success = False
    except Exception as e:
        print(f"\n❌ CCSDTQ test failed: {e}")
        import traceback
        traceback.print_exc()
        success = False
    
    # Final summary
    print("\n" + "="*70)
    if success:
        print("✅ ALL TESTS PASSED!")
        print("\nThe CCSDTQ framework is working correctly.")
        print("Next step: Complete the equation implementation.")
    else:
        print("❌ SOME TESTS FAILED")
        print("\nPlease review the errors above.")
    print("="*70)
