"""
Test suite for the matcher library
Tests the C++ point matching implementation
"""
import pytest
import numpy as np
import sys
import os

# Add build directory to path to import the compiled module
build_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'build')
if os.path.exists(build_path):
    sys.path.insert(0, build_path)

try:
    import matcher
except ImportError:
    pytest.skip("matcher module not found, skipping tests", allow_module_level=True)


class TestMatcherLibrary:
    """Test cases for the matcher library"""
    
    def test_simple_matching(self):
        """Test basic point matching functionality"""
        # Simple test case
        A = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]], dtype=np.float64)
        B = np.array([[0.0, 1.0], [2.1, 2.1]], dtype=np.float64)
        
        indices, distances = matcher.match_bruteforce(A, B)
        
        # Check return types
        assert isinstance(indices, np.ndarray)
        assert isinstance(distances, np.ndarray)
        assert indices.dtype == np.int32 or indices.dtype == np.int64
        assert distances.dtype == np.float64
        
        # Check shapes
        assert indices.shape == (3,)
        assert distances.shape == (3,)
        
        # Check results
        expected_indices = [0, 0, 1]  # A[0]->B[0], A[1]->B[0], A[2]->B[1]
        assert list(indices) == expected_indices
        
        # Check distances are reasonable
        assert distances[0] == pytest.approx(1.0, abs=1e-6)  # Distance from (0,0) to (0,1)
        assert distances[1] == pytest.approx(1.0, abs=1e-6)  # Distance from (1,1) to (0,1)
        assert distances[2] < 1.0  # Distance from (2,2) to (2.1,2.1) should be small
    
    def test_identical_points(self):
        """Test matching with identical points"""
        A = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
        B = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
        
        indices, distances = matcher.match_bruteforce(A, B)
        
        # Should match perfectly
        assert list(indices) == [0, 1]
        assert np.allclose(distances, [0.0, 0.0], atol=1e-10)
    
    def test_single_points(self):
        """Test with single points"""
        A = np.array([[5.0, 5.0]], dtype=np.float64)
        B = np.array([[6.0, 6.0]], dtype=np.float64)
        
        indices, distances = matcher.match_bruteforce(A, B)
        
        assert list(indices) == [0]
        expected_distance = np.sqrt(2.0)  # sqrt((6-5)^2 + (6-5)^2)
        assert distances[0] == pytest.approx(expected_distance, abs=1e-6)
    
    def test_input_validation(self):
        """Test input validation"""
        # Wrong dimensions for A
        with pytest.raises(RuntimeError, match="A must be Nx2"):
            A = np.array([[1.0, 2.0, 3.0]], dtype=np.float64)  # 3 columns
            B = np.array([[1.0, 2.0]], dtype=np.float64)
            matcher.match_bruteforce(A, B)
        
        # Wrong dimensions for B
        with pytest.raises(RuntimeError, match="B must be Mx2"):
            A = np.array([[1.0, 2.0]], dtype=np.float64)
            B = np.array([[1.0]], dtype=np.float64)  # 1 column
            matcher.match_bruteforce(A, B)


class TestMatcherCGAL:
    """Test cases for the CGAL-based matcher"""
    
    def test_simple_matching_cgal(self):
        """Test basic point matching functionality with CGAL"""
        # Simple test case
        A = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]], dtype=np.float64)
        B = np.array([[0.0, 1.0], [2.1, 2.1]], dtype=np.float64)
        
        indices, distances = matcher.match_cgal(A, B)
        
        # Check return types
        assert isinstance(indices, np.ndarray)
        assert isinstance(distances, np.ndarray)
        assert indices.dtype == np.int32 or indices.dtype == np.int64
        assert distances.dtype == np.float64
        
        # Check shapes
        assert indices.shape == (3,)
        assert distances.shape == (3,)
        
        # Check results
        expected_indices = [0, 0, 1]  # A[0]->B[0], A[1]->B[0], A[2]->B[1]
        assert list(indices) == expected_indices
        
        # Check distances are reasonable
        assert distances[0] == pytest.approx(1.0, abs=1e-6)  # Distance from (0,0) to (0,1)
        assert distances[1] == pytest.approx(1.0, abs=1e-6)  # Distance from (1,1) to (0,1)
        assert distances[2] < 1.0  # Distance from (2,2) to (2.1,2.1) should be small
    
    def test_identical_points_cgal(self):
        """Test matching with identical points using CGAL"""
        A = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
        B = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
        
        indices, distances = matcher.match_cgal(A, B)
        
        # Should match perfectly
        assert list(indices) == [0, 1]
        assert np.allclose(distances, [0.0, 0.0], atol=1e-10)
    
    def test_single_points_cgal(self):
        """Test with single points using CGAL"""
        A = np.array([[5.0, 5.0]], dtype=np.float64)
        B = np.array([[6.0, 6.0]], dtype=np.float64)
        
        indices, distances = matcher.match_cgal(A, B)
        
        assert list(indices) == [0]
        expected_distance = np.sqrt(2.0)  # sqrt((6-5)^2 + (6-5)^2)
        assert distances[0] == pytest.approx(expected_distance, abs=1e-6)
    
    def test_input_validation_cgal(self):
        """Test input validation for CGAL matcher"""
        # Wrong dimensions for A
        with pytest.raises(RuntimeError, match="A must be Nx2"):
            A = np.array([[1.0, 2.0, 3.0]], dtype=np.float64)  # 3 columns
            B = np.array([[1.0, 2.0]], dtype=np.float64)
            matcher.match_cgal(A, B)
        
        # Wrong dimensions for B
        with pytest.raises(RuntimeError, match="B must be Mx2"):
            A = np.array([[1.0, 2.0]], dtype=np.float64)
            B = np.array([[1.0]], dtype=np.float64)  # 1 column
            matcher.match_cgal(A, B)


def test_module_availability():
    """Test that the matcher module is properly imported"""
    assert hasattr(matcher, 'match_bruteforce')
    assert hasattr(matcher, 'match_cgal')
    
    # Check function signatures by calling help
    bruteforce_help = str(matcher.match_bruteforce.__doc__)
    assert 'Find nearest point' in bruteforce_help or bruteforce_help is None
    
    cgal_help = str(matcher.match_cgal.__doc__)
    assert 'Find nearest point' in cgal_help or cgal_help is None


def test_cgal_vs_bruteforce_consistency():
    """Test that CGAL and bruteforce methods produce consistent results"""
    np.random.seed(42)
    A = np.random.rand(20, 2).astype(np.float64) * 10
    B = np.random.rand(15, 2).astype(np.float64) * 10
    
    # Get results from both methods
    indices_bf, distances_bf = matcher.match_bruteforce(A, B)
    indices_cgal, distances_cgal = matcher.match_cgal(A, B)
    
    # Both should find the same nearest neighbors
    assert np.array_equal(indices_bf, indices_cgal), "CGAL and bruteforce should find the same nearest neighbors"
    
    # Distances should be very close (allowing for floating point differences)
    assert np.allclose(distances_bf, distances_cgal, atol=1e-9), "Distances should match between methods"


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v"])
