#!/usr/bin/env python3
"""
Visual example showing graph simplification with interactive side-by-side comparison.

This example demonstrates:
1. Topological simplification (removing degree-2 nodes)
2. Geometric simplification (merging close nodes)
3. Interactive side-by-side visualization using VisPy
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'build'))
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import acj
import pandas as pd
import numpy as np


def create_complex_test_graph():
    """Create a more complex graph for better visualization."""
    # Create a grid-like structure with some degree-2 nodes
    nodes = pd.DataFrame({
        'node_id': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        'x': [0.0, 50.0, 100.0, 150.0, 200.0,    # Bottom row
              25.0, 75.0, 125.0, 175.0,           # Middle-bottom
              0.0, 100.0, 200.0,                  # Top row
              100.0],                              # Center point
        'y': [0.0, 0.0, 0.0, 0.0, 0.0,            # Bottom row
              50.0, 50.0, 50.0, 50.0,             # Middle-bottom
              100.0, 100.0, 100.0,                # Top row
              75.0]                                # Center point
    })
    
    # Create segments - mix of paths and intersections
    segments_data = []
    segment_id = 0
    
    # Bottom row path (nodes 0-1-2-3-4)
    for i in range(4):
        start_id, end_id = i, i + 1
        segments_data.append({
            'segment_id': segment_id,
            'node_start': start_id,
            'node_end': end_id,
            'x1': nodes.loc[start_id, 'x'],
            'y1': nodes.loc[start_id, 'y'],
            'x2': nodes.loc[end_id, 'x'],
            'y2': nodes.loc[end_id, 'y']
        })
        segment_id += 1
    
    # Middle connections (creating some intersections)
    connections = [(0, 5), (5, 6), (6, 2), (2, 7), (7, 8), (8, 4),
                   (5, 9), (6, 12), (7, 12), (8, 11), (12, 10)]
    
    for start_id, end_id in connections:
        segments_data.append({
            'segment_id': segment_id,
            'node_start': start_id,
            'node_end': end_id,
            'x1': nodes.loc[start_id, 'x'],
            'y1': nodes.loc[start_id, 'y'],
            'x2': nodes.loc[end_id, 'x'],
            'y2': nodes.loc[end_id, 'y']
        })
        segment_id += 1
    
    # Top row connections
    top_connections = [(9, 10), (10, 11)]
    for start_id, end_id in top_connections:
        segments_data.append({
            'segment_id': segment_id,
            'node_start': start_id,
            'node_end': end_id,
            'x1': nodes.loc[start_id, 'x'],
            'y1': nodes.loc[start_id, 'y'],
            'x2': nodes.loc[end_id, 'x'],
            'y2': nodes.loc[end_id, 'y']
        })
        segment_id += 1
    
    segments = pd.DataFrame(segments_data)
    
    return nodes, segments


def print_graph_stats(graph, name="Graph"):
    """Print statistics about a graph."""
    print(f"{name}:")
    print(f"  Nodes: {len(graph.nodes)}")
    print(f"  Segments: {len(graph.segments)}")
    
    # Calculate degree distribution
    from collections import defaultdict
    degrees = defaultdict(int)
    for _, seg in graph.segments.iterrows():
        degrees[seg['node_start']] += 1
        degrees[seg['node_end']] += 1
    
    degree_counts = defaultdict(int)
    for node_id, degree in degrees.items():
        degree_counts[degree] += 1
    
    print(f"  Degree distribution:")
    for degree in sorted(degree_counts.keys()):
        print(f"    Degree {degree}: {degree_counts[degree]} nodes")


def main():
    print("=" * 80)
    print("ACJ Graph Simplification - Interactive Visual Comparison")
    print("=" * 80)
    print()
    
    # Create test graph
    print("[1/4] Creating complex test graph...")
    nodes, segments = create_complex_test_graph()
    graph_original = acj.load_graph(nodes, segments)
    print_graph_stats(graph_original, "Original Graph")
    print()
    
    # Topological simplification
    print("[2/4] Applying topological simplification...")
    graph_topo = acj.simplify_graph_topological(graph_original)
    print_graph_stats(graph_topo, "After Topological Simplification")
    reduction_topo = (len(graph_original.nodes) - len(graph_topo.nodes)) / len(graph_original.nodes) * 100
    print(f"  Reduction: {reduction_topo:.1f}% of nodes removed")
    print()
    
    # Geometric simplification
    print("[3/4] Applying geometric simplification...")
    graph_geo = acj.simplify_graph_geometric(graph_original, threshold_meters=30.0)
    print_graph_stats(graph_geo, "After Geometric Simplification (30m threshold)")
    reduction_geo = (len(graph_original.nodes) - len(graph_geo.nodes)) / len(graph_original.nodes) * 100
    print(f"  Reduction: {reduction_geo:.1f}% of nodes removed")
    print()
    
    # Show comparison 1: Original vs Topological
    print("[4/4] Launching interactive visualizations...")
    print()
    print("=" * 80)
    print("COMPARISON 1: Original (LEFT) vs Topological Simplified (RIGHT)")
    print("=" * 80)
    print("Description: Topological simplification removes degree-2 nodes")
    print("             Preserves all intersections and network topology")
    print()
    print("Controls:")
    print("  - Mouse drag: Pan")
    print("  - Mouse wheel: Zoom")
    print("  - Right-click drag: Rotate")
    print("  - N: Toggle nodes")
    print("  - L: Toggle lines")
    print("  - R: Reset view")
    print("  - Q or ESC: Close window")
    print()
    print("Close the window to see the next comparison...")
    print("=" * 80)
    
    index_original = acj.MapIndex(graph_original)
    index_topo = acj.MapIndex(graph_topo)
    
    acj.render_comparison(
        index_original,
        index_topo,
        title_left=f"Original Graph ({len(graph_original.nodes)} nodes)",
        title_right=f"Topological Simplified ({len(graph_topo.nodes)} nodes, {reduction_topo:.1f}% reduction)"
    )
    
    # Show comparison 2: Topological vs Geometric
    print()
    print("=" * 80)
    print("COMPARISON 2: Topological (LEFT) vs Geometric Simplified (RIGHT)")
    print("=" * 80)
    print("Description: Geometric simplification merges nearby intersections")
    print("             More aggressive than topological simplification")
    print()
    print("Close the window to finish...")
    print("=" * 80)
    
    index_geo = acj.MapIndex(graph_geo)
    
    acj.render_comparison(
        index_topo,
        index_geo,
        title_left=f"Topological Simplified ({len(graph_topo.nodes)} nodes)",
        title_right=f"Geometric Simplified ({len(graph_geo.nodes)} nodes, 30m threshold, {reduction_geo:.1f}% reduction)"
    )
    
    print()
    print("=" * 80)
    print("Example completed successfully!")
    print()
    print("Summary:")
    print(f"  Original:    {len(graph_original.nodes):3d} nodes, {len(graph_original.segments):3d} segments")
    print(f"  Topological: {len(graph_topo.nodes):3d} nodes, {len(graph_topo.segments):3d} segments ({reduction_topo:.1f}% reduction)")
    print(f"  Geometric:   {len(graph_geo.nodes):3d} nodes, {len(graph_geo.segments):3d} segments ({reduction_geo:.1f}% reduction)")
    print()
    print("Use cases:")
    print("  - Topological: Fast queries, routing, preserves exact topology")
    print("  - Geometric:   Data analysis, dense networks, visual clarity")
    print("=" * 80)


if __name__ == "__main__":
    main()

