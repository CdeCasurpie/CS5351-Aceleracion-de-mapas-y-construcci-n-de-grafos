#!/usr/bin/env python3
"""
Example usage of the matcher library
Demonstrates how to use the C++ point matching implementation from Python
"""
import numpy as np
import sys
import os

# Add build directory to path to import the compiled module
build_path = os.path.join(os.path.dirname(__file__), 'build')
if os.path.exists(build_path):
    sys.path.insert(0, build_path)

try:
    import matcher
    print("✓ Matcher library imported successfully!")
except ImportError as e:
    print(f"✗ Failed to import matcher library: {e}")
    print("Make sure to build the library first with 'make build'")
    sys.exit(1)


def test_bruteforce():
    """Test brute-force matching algorithm"""
    print("\n=== Testing Brute-Force Matcher ===")
    
    # Create sample data
    print("\n1. Creating sample point arrays:")
    A = np.array([
        [0.0, 0.0],  # Point 0
        [1.0, 1.0],  # Point 1
        [2.0, 2.0],  # Point 2
        [5.0, 5.0],  # Point 3
    ], dtype=np.float64)
    
    B = np.array([
        [0.1, 0.1],  # Point 0
        [2.1, 2.1],  # Point 1
        [10.0, 10.0], # Point 2
    ], dtype=np.float64)
    
    print(f"Array A (source points):\n{A}")
    print(f"Array B (target points):\n{B}")
    
    # Perform matching
    print("\n2. Performing brute-force matching...")
    indices, distances = matcher.match_bruteforce(A, B)
    
    print(f"Matching indices: {indices}")
    print(f"Distances: {distances}")
    
    # Display results
    print("\n3. Matching results:")
    for i in range(len(A)):
        matched_idx = indices[i]
        distance = distances[i]
        print(f"Point A[{i}] = {A[i]} -> Point B[{matched_idx}] = {B[matched_idx]} (distance: {distance:.4f})")
    
    # Performance test with larger arrays
    print("\n4. Performance test with larger arrays:")
    np.random.seed(42)
    large_A = np.random.rand(1000, 2).astype(np.float64)
    large_B = np.random.rand(500, 2).astype(np.float64)
    
    import time
    start_time = time.time()
    large_indices, large_distances = matcher.match_bruteforce(large_A, large_B)
    end_time = time.time()
    
    print(f"Matched {len(large_A)} points to {len(large_B)} targets in {end_time - start_time:.4f} seconds")
    print(f"Average distance: {np.mean(large_distances):.4f}")
    print(f"Min distance: {np.min(large_distances):.4f}")
    print(f"Max distance: {np.max(large_distances):.4f}")
    
    print("\n✓ Brute-force test completed successfully!")
    return large_A, large_B, large_indices, large_distances


def test_cgal():
    """Test CGAL matching algorithm"""
    print("\n\n=== Testing CGAL Matcher ===")
    
    # Check if match_cgal is available
    if not hasattr(matcher, 'match_cgal'):
        print("⚠ match_cgal not available in this build")
        return None, None, None, None
    
    # Create sample data
    print("\n1. Creating sample point arrays:")
    A = np.array([
        [0.0, 0.0],  # Point 0
        [1.0, 1.0],  # Point 1
        [2.0, 2.0],  # Point 2
        [5.0, 5.0],  # Point 3
    ], dtype=np.float64)
    
    B = np.array([
        [0.1, 0.1],  # Point 0
        [2.1, 2.1],  # Point 1
        [10.0, 10.0], # Point 2
    ], dtype=np.float64)
    
    print(f"Array A (source points):\n{A}")
    print(f"Array B (target points):\n{B}")
    
    # Perform matching
    print("\n2. Performing CGAL matching...")
    indices, distances = matcher.match_cgal(A, B)
    
    print(f"Matching indices: {indices}")
    print(f"Distances: {distances}")
    
    # Display results
    print("\n3. Matching results:")
    for i in range(len(A)):
        matched_idx = indices[i]
        distance = distances[i]
        print(f"Point A[{i}] = {A[i]} -> Point B[{matched_idx}] = {B[matched_idx]} (distance: {distance:.4f})")
    
    # Performance test with larger arrays
    print("\n4. Performance test with larger arrays:")
    np.random.seed(42)
    large_A = np.random.rand(1000, 2).astype(np.float64)
    large_B = np.random.rand(500, 2).astype(np.float64)
    
    import time
    start_time = time.time()
    large_indices, large_distances = matcher.match_cgal(large_A, large_B)
    end_time = time.time()
    
    print(f"Matched {len(large_A)} points to {len(large_B)} targets in {end_time - start_time:.4f} seconds")
    print(f"Average distance: {np.mean(large_distances):.4f}")
    print(f"Min distance: {np.min(large_distances):.4f}")
    print(f"Max distance: {np.max(large_distances):.4f}")
    
    print("\n✓ CGAL test completed successfully!")
    return large_A, large_B, large_indices, large_distances


def compare_algorithms():
    """Compare brute-force and CGAL algorithms"""
    print("\n\n=== Comparing Algorithms ===")
    
    # Check if match_cgal is available
    if not hasattr(matcher, 'match_cgal'):
        print("⚠ match_cgal not available, skipping comparison")
        return
    
    # Create test data
    print("\n1. Creating test point arrays:")
    np.random.seed(123)
    A = np.random.rand(100, 2).astype(np.float64)
    B = np.random.rand(50, 2).astype(np.float64)
    
    print(f"Testing with {len(A)} source points and {len(B)} target points")
    
    # Test brute-force
    print("\n2. Running brute-force algorithm...")
    import time
    start_bf = time.time()
    bf_indices, bf_distances = matcher.match_bruteforce(A, B)
    time_bf = time.time() - start_bf
    print(f"Brute-force completed in {time_bf:.4f} seconds")
    
    # Test CGAL
    print("\n3. Running CGAL algorithm...")
    start_cgal = time.time()
    cgal_indices, cgal_distances = matcher.match_cgal(A, B)
    time_cgal = time.time() - start_cgal
    print(f"CGAL completed in {time_cgal:.4f} seconds")
    
    # Compare results
    print("\n4. Comparing results:")
    indices_match = np.array_equal(bf_indices, cgal_indices)
    distances_match = np.allclose(bf_distances, cgal_distances, rtol=1e-5, atol=1e-8)
    
    print(f"   Indices match: {'✓' if indices_match else '✗'}")
    print(f"   Distances match: {'✓' if distances_match else '✗'}")
    
    if not indices_match:
        diff_count = np.sum(bf_indices != cgal_indices)
        print(f"   Mismatched indices: {diff_count}/{len(A)}")
    
    if not distances_match:
        max_diff = np.max(np.abs(bf_distances - cgal_distances))
        print(f"   Max distance difference: {max_diff:.10f}")
    
    # Performance comparison
    print("\n5. Performance comparison:")
    speedup = time_bf / time_cgal if time_cgal > 0 else float('inf')
    print(f"   Brute-force: {time_bf:.4f}s")
    print(f"   CGAL: {time_cgal:.4f}s")
    print(f"   Speedup: {speedup:.2f}x {'(CGAL faster)' if speedup > 1 else '(Brute-force faster)'}")
    
    print("\n✓ Comparison completed successfully!")


def main():
    """Demonstrate the matcher library functionality"""
    print("\n" + "="*60)
    print("  Point Matching Library - Comprehensive Demo")
    print("="*60)
    
    # Test brute-force algorithm
    test_bruteforce()
    
    # Test CGAL algorithm
    test_cgal()
    
    # Compare both algorithms
    compare_algorithms()
    
    print("\n" + "="*60)
    print("  All tests completed successfully!")
    print("="*60)


if __name__ == "__main__":
    main()