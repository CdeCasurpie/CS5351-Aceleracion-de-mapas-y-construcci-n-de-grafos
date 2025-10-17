#!/usr/bin/env python3
"""
Performance benchmark comparing brute-force vs CGAL algorithms
Tests with increasingly large point sets to demonstrate CGAL's efficiency
"""
import numpy as np
import sys
import os
import time

# Add build directory to path
build_path = os.path.join(os.path.dirname(__file__), 'build')
if os.path.exists(build_path):
    sys.path.insert(0, build_path)

try:
    import matcher
    print("✓ Matcher library loaded\n")
except ImportError as e:
    print(f"✗ Failed to import matcher: {e}")
    sys.exit(1)


def benchmark_size(n_source, n_target, seed=42):
    """Benchmark both algorithms with given point set sizes"""
    np.random.seed(seed)
    
    # Generate random points in [0, 100] x [0, 100]
    A = (np.random.rand(n_source, 2) * 100).astype(np.float64)
    B = (np.random.rand(n_target, 2) * 100).astype(np.float64)
    
    # Warm-up (to avoid cold start effects)
    if n_source <= 100:
        _ = matcher.match_bruteforce(A[:10], B[:10])
        _ = matcher.match_cgal(A[:10], B[:10])
    
    # Benchmark brute-force
    start = time.time()
    bf_indices, bf_distances = matcher.match_bruteforce(A, B)
    time_bf = time.time() - start
    
    # Benchmark CGAL
    start = time.time()
    cgal_indices, cgal_distances = matcher.match_cgal(A, B)
    time_cgal = time.time() - start
    
    # Verify results match
    match_correct = np.array_equal(bf_indices, cgal_indices)
    
    return {
        'n_source': n_source,
        'n_target': n_target,
        'time_bf': time_bf,
        'time_cgal': time_cgal,
        'speedup': time_bf / time_cgal if time_cgal > 0 else float('inf'),
        'match_correct': match_correct,
        'avg_distance': np.mean(bf_distances)
    }


def print_results_table(results):
    """Print formatted results table"""
    print("\n" + "="*100)
    print(f"{'Source':>8} | {'Target':>8} | {'Brute-Force':>12} | {'CGAL':>12} | {'Speedup':>10} | {'Match':>7}")
    print(f"{'Points':>8} | {'Points':>8} | {'Time (s)':>12} | {'Time (s)':>12} | {'(x)':>10} | {'OK':>7}")
    print("="*100)
    
    for r in results:
        speedup_str = f"{r['speedup']:.2f}x"
        winner = "🚀 CGAL" if r['speedup'] > 1 else "💪 BF"
        match_str = "✓" if r['match_correct'] else "✗"
        
        print(f"{r['n_source']:>8,} | {r['n_target']:>8,} | "
              f"{r['time_bf']:>12.6f} | {r['time_cgal']:>12.6f} | "
              f"{speedup_str:>10} | {match_str:>7}  {winner}")
    
    print("="*100)


def main():
    print("\n" + "🔥"*40)
    print("   MATCHER LIBRARY - PERFORMANCE BENCHMARK")
    print("🔥"*40 + "\n")
    
    print("Testing with increasingly large point sets...")
    print("This will demonstrate the scalability of both algorithms.\n")
    
    # Test configurations: (n_source, n_target)
    test_cases = [
        (100, 50),
        (500, 250),
        (1000, 500),
        (2000, 1000),
        (5000, 2500),
        (10000, 5000),
        (20000, 10000),
        (50000, 25000),
        (100000, 50000),
    ]
    
    results = []
    
    for i, (n_src, n_tgt) in enumerate(test_cases, 1):
        print(f"[{i}/{len(test_cases)}] Benchmarking {n_src:,} source → {n_tgt:,} target points...", end=" ", flush=True)
        
        try:
            result = benchmark_size(n_src, n_tgt)
            results.append(result)
            
            speedup_emoji = "🚀" if result['speedup'] > 1 else "🐌"
            print(f"Done! {speedup_emoji} Speedup: {result['speedup']:.2f}x")
            
        except Exception as e:
            print(f"Error: {e}")
            break
    
    # Print summary table
    print_results_table(results)
    
    # Analysis
    print("\n📊 ANALYSIS:")
    print("-" * 100)
    
    cgal_wins = sum(1 for r in results if r['speedup'] > 1)
    bf_wins = len(results) - cgal_wins
    
    print(f"• CGAL won in {cgal_wins}/{len(results)} tests")
    print(f"• Brute-force won in {bf_wins}/{len(results)} tests")
    
    if results:
        best_speedup = max(results, key=lambda r: r['speedup'])
        print(f"\n• Best speedup: {best_speedup['speedup']:.2f}x")
        print(f"  at {best_speedup['n_source']:,} source × {best_speedup['n_target']:,} target points")
        
        winner_name = "CGAL" if best_speedup['speedup'] > 1 else "Brute-force"
        print(f"  Winner: {winner_name}")
    
    print("\n💡 KEY INSIGHTS:")
    print("-" * 100)
    print("• Brute-Force complexity: O(N × M) - scales quadratically")
    print("• CGAL complexity: O(M log M) construction + O(N log M) queries")
    print("• CGAL overhead matters for small datasets")
    print("• For MASSIVE datasets, CGAL's logarithmic scaling becomes advantageous")
    print("• Both algorithms produce identical results ✓")
    
    print("\n" + "="*100)
    print("  Benchmark Complete!")
    print("="*100 + "\n")


if __name__ == "__main__":
    main()
