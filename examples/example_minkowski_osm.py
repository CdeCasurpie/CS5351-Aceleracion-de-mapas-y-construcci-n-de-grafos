#!/usr/bin/env python3
"""
Interactive graph simplification using Minkowski Straight Skeleton.

This example demonstrates:
1. Loading real street network from OpenStreetMap
2. Vectorial simplification using CGAL Minkowski Sums and Straight Skeleton
3. Interactive side-by-side GPU visualization

Configuration: Change CITY_NAME variable to analyze any city
"""

import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "build"))
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import geojac

CITY_NAME = "Barranco, Lima, Peru"
RADIUS_METERS = 5.0


def main():
    print("=" * 80)
    print("ACJ Graph Simplification - Minkowski Straight Skeleton")
    print("=" * 80)
    print(f"Location: {CITY_NAME}")
    print(f"Street Radius: {RADIUS_METERS}m")
    print("=" * 80)
    print()

    cache_dir = "./cache"

    print("[1/3] Loading street network from OpenStreetMap...")
    start_time = time.time()
    try:
        graph_original = geojac.load_map(
            CITY_NAME, cache_dir=cache_dir, network_type="drive"
        )
        load_time = time.time() - start_time
        print(f"      ✓ Network loaded successfully in {load_time:.2f} seconds")
        print(f"      Original Nodes: {len(graph_original.nodes):,}")
        print(f"      Original Segments: {len(graph_original.segments):,}")
    except Exception as e:
        print(f"      ✗ ERROR: Could not load map: {e}")
        return

    print()

    print("[2/3] Simplifying graph using exact CGAL Minkowski engine...")
    print("      Algorithm: Minkowski Sum (Buffer) + Straight Skeleton + Smart Pruning")

    start_time = time.time()
    graph_simplified = geojac.simplify_graph(
        graph_original, threshold_meters=RADIUS_METERS, method="guided_minkowski"
    )
    simp_time = time.time() - start_time

    print(f"      ✓ Simplification completed in {simp_time:.4f} seconds")
    print(f"      Simplified Nodes: {len(graph_simplified.nodes):,}")
    print(f"      Simplified Segments: {len(graph_simplified.segments):,}")

    reduction_pct = 100 * (
        1 - len(graph_simplified.segments) / len(graph_original.segments)
    )
    print(f"      Reduction: {reduction_pct:.1f}% fewer segments")
    print()

    print("[3/3] Launching interactive GPU visualization...")
    print("=" * 80)
    print("COMPARISON: Original (LEFT) vs Minkowski Simplified (RIGHT)")
    print("=" * 80)
    print("Description:")
    print("  • Minkowski merges parallel streets (like avenues) into a single axis.")
    print("  • Uses exact C++ computational geometry (no pixelation errors).")
    print()

    index_original = geojac.MapIndex(graph_original)
    index_simplified = geojac.MapIndex(graph_simplified)

    geojac.render_comparison(
        map_index_left=index_original,
        map_index_right=index_simplified,
        title=f"ACJ Minkowski Comparison - {CITY_NAME}",
        title_left=f"Original ({len(graph_original.segments):,} segments)",
        title_right=f"Minkowski Simplified ({len(graph_simplified.segments):,} segments)",
    )

    print()
    print("=" * 80)
    print("Visualization closed. Example completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
