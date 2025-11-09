#!/usr/bin/env python3
"""
Interactive graph simplification visualization with real OSM data.

This example demonstrates:
1. Loading real street network from OpenStreetMap
2. Topological simplification (removing degree-2 nodes)
3. Geometric simplification (merging nearby nodes)
4. Side-by-side interactive visualization comparing methods

Configuration: Change CITY_NAME variable to analyze any city
"""

import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'build'))
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import acj
import pandas as pd
from collections import defaultdict


# ============================================================================
# CONFIGURATION - Change these values to analyze different cities
# ============================================================================
CITY_NAME = "Barranco, Lima, Peru"  # Change to any city/location
GEOMETRIC_THRESHOLD_METERS = 15.0   # Distance threshold for geometric simplification
# ============================================================================


def print_graph_statistics(graph, name="Graph"):
    """Print detailed statistics about a graph."""
    print(f"\n{name}:")
    print(f"  Nodes: {len(graph.nodes):,}")
    print(f"  Segments: {len(graph.segments):,}")
    
    # Calculate degree distribution
    degrees = defaultdict(int)
    for _, segment in graph.segments.iterrows():
        degrees[segment['node_start']] += 1
        degrees[segment['node_end']] += 1
    
    # Count nodes by degree
    degree_counts = defaultdict(int)
    for node_id, degree in degrees.items():
        degree_counts[degree] += 1
    
    print(f"  Degree distribution:")
    for degree in sorted(degree_counts.keys())[:10]:  # Show first 10 degrees
        count = degree_counts[degree]
        percentage = (count / len(graph.nodes)) * 100
        print(f"    Degree {degree}: {count:4d} nodes ({percentage:5.2f}%)")
    
    if len(degree_counts) > 10:
        remaining = sum(degree_counts[d] for d in sorted(degree_counts.keys())[10:])
        print(f"    Degree >9: {remaining:4d} nodes")
    
    # Calculate average degree
    avg_degree = sum(degrees.values()) / len(degrees) if degrees else 0
    print(f"  Average degree: {avg_degree:.2f}")


def main():
    print("=" * 80)
    print("ACJ Graph Simplification - Real OSM Data")
    print("=" * 80)
    print(f"Location: {CITY_NAME}")
    print(f"Geometric threshold: {GEOMETRIC_THRESHOLD_METERS}m")
    print("=" * 80)
    print()
    
    cache_dir = "./cache"
    
    # Step 1: Load real street network from OSM
    print(f"[1/5] Loading street network from OpenStreetMap...")
    print(f"      Location: {CITY_NAME}")
    
    start_time = time.time()
    try:
        graph_original = acj.load_map(CITY_NAME, cache_dir=cache_dir, network_type="drive")
        load_time = time.time() - start_time
        print(f"      ✓ Network loaded successfully in {load_time:.2f} seconds")
    except Exception as e:
        print(f"      ✗ ERROR: Could not load map: {e}")
        print("      Please check your internet connection and OSMnx installation.")
        return
    
    print_graph_statistics(graph_original, "Original Network")
    print()
    
    # Step 2: Apply topological simplification
    print(f"[2/5] Applying topological simplification...")
    print(f"      Algorithm: Remove degree-2 nodes (intermediate nodes on paths)")
    
    start_time = time.time()
    graph_topo = acj.simplify_graph_topological(graph_original)
    topo_time = time.time() - start_time
    
    print(f"      ✓ Topological simplification completed in {topo_time:.2f} seconds")
    
    print_graph_statistics(graph_topo, "After Topological Simplification")
    
    # Calculate reductions
    node_reduction_topo = (len(graph_original.nodes) - len(graph_topo.nodes)) / len(graph_original.nodes) * 100
    segment_reduction_topo = (len(graph_original.segments) - len(graph_topo.segments)) / len(graph_original.segments) * 100
    
    print(f"\n  Reduction:")
    print(f"    Nodes: {node_reduction_topo:.1f}% ({len(graph_original.nodes):,} → {len(graph_topo.nodes):,})")
    print(f"    Segments: {segment_reduction_topo:.1f}% ({len(graph_original.segments):,} → {len(graph_topo.segments):,})")
    print()
    
    # Step 3: Apply geometric simplification
    print(f"[3/5] Applying geometric simplification...")
    print(f"      Algorithm: Merge nodes within {GEOMETRIC_THRESHOLD_METERS}m using CGAL clustering")
    
    start_time = time.time()
    graph_geo = acj.simplify_graph_geometric(graph_original, threshold_meters=GEOMETRIC_THRESHOLD_METERS)
    geo_time = time.time() - start_time
    
    print(f"      ✓ Geometric simplification completed in {geo_time:.2f} seconds")
    
    print_graph_statistics(graph_geo, f"After Geometric Simplification ({GEOMETRIC_THRESHOLD_METERS}m threshold)")
    
    # Calculate reductions
    node_reduction_geo = (len(graph_original.nodes) - len(graph_geo.nodes)) / len(graph_original.nodes) * 100
    segment_reduction_geo = (len(graph_original.segments) - len(graph_geo.segments)) / len(graph_original.segments) * 100
    
    print(f"\n  Reduction:")
    print(f"    Nodes: {node_reduction_geo:.1f}% ({len(graph_original.nodes):,} → {len(graph_geo.nodes):,})")
    print(f"    Segments: {segment_reduction_geo:.1f}% ({len(graph_original.segments):,} → {len(graph_geo.segments):,})")
    print()
    
    # Step 4: Launch first comparison (Original vs Topological)
    print("[4/5] Launching first interactive comparison...")
    print()
    print("=" * 80)
    print("COMPARISON 1: Original Network (LEFT) vs Topological Simplified (RIGHT)")
    print("=" * 80)
    print()
    print("Description:")
    print("  • Topological simplification removes intermediate nodes (degree-2)")
    print("  • All intersections are preserved (nodes with degree ≠ 2)")
    print("  • Network topology remains unchanged")
    print("  • Best for: Routing, navigation, real-time queries")
    print()
    print("Visual Differences:")
    print(f"  • LEFT (Original):    {len(graph_original.nodes):,} nodes, {len(graph_original.segments):,} segments")
    print(f"  • RIGHT (Topological): {len(graph_topo.nodes):,} nodes, {len(graph_topo.segments):,} segments")
    print(f"  • Reduction:           {node_reduction_topo:.1f}% nodes, {segment_reduction_topo:.1f}% segments")
    print()
    print("Interactive Controls:")
    print("  • Left-click + drag:  Pan/move view")
    print("  • Mouse wheel:        Zoom in/out")
    print("  • Right-click + drag: Rotate view")
    print("  • N key:              Toggle nodes visibility")
    print("  • L key:              Toggle lines/segments visibility")
    print("  • R key:              Reset camera view")
    print("  • Q or ESC:           Close window and continue")
    print()
    print("Close the window to see the next comparison...")
    print("=" * 80)
    
    index_original = acj.MapIndex(graph_original)
    index_topo = acj.MapIndex(graph_topo)
    
    acj.render_comparison(
        index_original,
        index_topo,
        title=f"Graph Simplification - {CITY_NAME}",
        title_left=f"Original Network ({len(graph_original.nodes):,} nodes)",
        title_right=f"Topological Simplified ({len(graph_topo.nodes):,} nodes, {node_reduction_topo:.1f}% reduction)"
    )
    
    # Step 5: Launch second comparison (Topological vs Geometric)
    print()
    print("[5/5] Launching second interactive comparison...")
    print()
    print("=" * 80)
    print("COMPARISON 2: Topological (LEFT) vs Geometric Simplified (RIGHT)")
    print("=" * 80)
    print()
    print("Description:")
    print(f"  • Geometric simplification merges nodes within {GEOMETRIC_THRESHOLD_METERS}m")
    print("  • Uses CGAL Delaunay triangulation for efficient spatial clustering")
    print("  • More aggressive than topological simplification")
    print("  • Best for: Data analysis, visualization, dense urban networks")
    print()
    print("Visual Differences:")
    print(f"  • LEFT (Topological):  {len(graph_topo.nodes):,} nodes, {len(graph_topo.segments):,} segments")
    print(f"  • RIGHT (Geometric):   {len(graph_geo.nodes):,} nodes, {len(graph_geo.segments):,} segments")
    print(f"  • Additional reduction: {((len(graph_topo.nodes) - len(graph_geo.nodes)) / len(graph_topo.nodes) * 100):.1f}% nodes")
    print()
    print("Note: Geometric simplification may slightly alter network topology")
    print("      by merging nearby intersections into single nodes.")
    print()
    print("Close the window to finish...")
    print("=" * 80)
    
    index_geo = acj.MapIndex(graph_geo)
    
    acj.render_comparison(
        index_topo,
        index_geo,
        title=f"Graph Simplification - {CITY_NAME}",
        title_left=f"Topological Simplified ({len(graph_topo.nodes):,} nodes)",
        title_right=f"Geometric Simplified ({len(graph_geo.nodes):,} nodes, {GEOMETRIC_THRESHOLD_METERS}m, {node_reduction_geo:.1f}% total reduction)"
    )
    
    # Final summary
    print()
    print("=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    print()
    print(f"Location: {CITY_NAME}")
    print()
    print("Network Statistics:")
    print(f"  Original:    {len(graph_original.nodes):5,} nodes, {len(graph_original.segments):5,} segments")
    print(f"  Topological: {len(graph_topo.nodes):5,} nodes, {len(graph_topo.segments):5,} segments ({node_reduction_topo:5.1f}% reduction)")
    print(f"  Geometric:   {len(graph_geo.nodes):5,} nodes, {len(graph_geo.segments):5,} segments ({node_reduction_geo:5.1f}% reduction)")
    print()
    print("Processing Times:")
    print(f"  Map loading:                {load_time:.2f}s")
    print(f"  Topological simplification: {topo_time:.2f}s")
    print(f"  Geometric simplification:   {geo_time:.2f}s")
    print(f"  Total:                      {load_time + topo_time + geo_time:.2f}s")
    print()
    print("Use Cases:")
    print("  • Topological: Routing, navigation, preserves exact topology")
    print("  • Geometric:   Visualization, analysis, reduces visual clutter")
    print()
    print("Algorithm Complexity:")
    print("  • Topological: O(n) where n = number of nodes")
    print("  • Geometric:   O(n log n) using CGAL Delaunay triangulation")
    print()
    print("=" * 80)
    print("Example completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()

