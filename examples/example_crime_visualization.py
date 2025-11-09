#!/usr/bin/env python3
"""
Interactive crime heatmap visualization with side-by-side comparison.

This example demonstrates:
1. Creating a street network
2. Generating crime hotspots
3. Assigning crimes to nearest junctions
4. Interactive heatmap visualization with comparison
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'build'))
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import acj
import pandas as pd
import numpy as np


def create_sample_network():
    """Create a more complex grid network for better visualization."""
    # Create a 5x5 grid of nodes (25 nodes total)
    nodes_data = []
    node_id = 0
    for x in [0.0, 50.0, 100.0, 150.0, 200.0]:
        for y in [0.0, 50.0, 100.0, 150.0, 200.0]:
            nodes_data.append({'node_id': node_id, 'x': x, 'y': y})
            node_id += 1
    
    nodes = pd.DataFrame(nodes_data)
    
    # Create segments connecting the grid
    segments_data = []
    segment_id = 0
    
    # Horizontal segments
    for row in range(5):
        for col in range(4):
            start_id = row * 5 + col
            end_id = row * 5 + col + 1
            start_node = nodes[nodes['node_id'] == start_id].iloc[0]
            end_node = nodes[nodes['node_id'] == end_id].iloc[0]
            
            segments_data.append({
                'segment_id': segment_id,
                'node_start': start_id,
                'node_end': end_id,
                'x1': start_node['x'],
                'y1': start_node['y'],
                'x2': end_node['x'],
                'y2': end_node['y']
            })
            segment_id += 1
    
    # Vertical segments
    for col in range(5):
        for row in range(4):
            start_id = row * 5 + col
            end_id = (row + 1) * 5 + col
            start_node = nodes[nodes['node_id'] == start_id].iloc[0]
            end_node = nodes[nodes['node_id'] == end_id].iloc[0]
            
            segments_data.append({
                'segment_id': segment_id,
                'node_start': start_id,
                'node_end': end_id,
                'x1': start_node['x'],
                'y1': start_node['y'],
                'x2': end_node['x'],
                'y2': end_node['y']
            })
            segment_id += 1
    
    segments = pd.DataFrame(segments_data)
    
    return nodes, segments


def create_sample_crimes(num_crimes=200):
    """Create sample crime points with realistic clustering."""
    np.random.seed(42)
    
    # Create multiple hotspots with different intensities
    hotspots = [
        (50.0, 50.0, 40, 15),     # Center - high density
        (150.0, 150.0, 60, 20),   # Top-right - very high density
        (50.0, 150.0, 30, 12),    # Top-left - medium density
        (150.0, 50.0, 25, 10),    # Bottom-right - medium density
    ]
    
    crimes_data = []
    crime_id = 0
    
    for center_x, center_y, count, spread in hotspots:
        for _ in range(count):
            # Add noise around hotspot center
            x = center_x + np.random.normal(0, spread)
            y = center_y + np.random.normal(0, spread)
            
            # Clip to bounds
            x = max(0, min(200, x))
            y = max(0, min(200, y))
            
            crimes_data.append({
                'point_id': crime_id,
                'x': x,
                'y': y,
                'type': 'Robbery'
            })
            crime_id += 1
    
    # Add random background crimes
    for _ in range(num_crimes - crime_id):
        x = np.random.uniform(0, 200)
        y = np.random.uniform(0, 200)
        
        crimes_data.append({
            'point_id': crime_id,
            'x': x,
            'y': y,
            'type': 'Theft'
        })
        crime_id += 1
    
    return pd.DataFrame(crimes_data)


def print_assignment_summary(assignments):
    """Print summary of crime assignments."""
    print(f"Crime Assignment Summary:")
    print(f"  Total crimes: {len(assignments)}")
    print(f"  Unique assigned nodes: {assignments['assigned_node_id'].nunique()}")
    
    # Show top 5 nodes with most crimes
    top_nodes = assignments['assigned_node_id'].value_counts().head(5)
    print(f"  Top 5 hotspot nodes:")
    for node_id, count in top_nodes.items():
        print(f"    Node {node_id}: {count} crimes")


def main():
    print("=" * 80)
    print("ACJ Crime Heatmap - Interactive Visual Comparison")
    print("=" * 80)
    print()
    
    # Create sample network
    print("[1/4] Creating sample 5x5 grid network...")
    nodes, segments = create_sample_network()
    graph = acj.load_graph(nodes, segments)
    print(f"  Created network with {len(nodes)} nodes and {len(segments)} segments")
    print()
    
    # Create MapIndex for spatial queries
    print("[2/4] Building spatial index...")
    map_index = acj.MapIndex(graph)
    print("  Spatial index built successfully (using CGAL Delaunay triangulation)")
    print()
    
    # Create sample crimes
    print("[3/4] Generating sample crime data...")
    crimes = create_sample_crimes(num_crimes=200)
    print(f"  Generated {len(crimes)} crime points with 4 hotspots")
    print(f"  Crime types: {crimes['type'].value_counts().to_dict()}")
    print()
    
    # Assign crimes to nearest junctions
    print("[4/4] Assigning crimes to nearest junctions...")
    assignments = map_index.assign_to_endpoints(crimes)
    print_assignment_summary(assignments)
    
    # Show example assignments
    print()
    print("  Example assignments (first 5):")
    for idx, row in assignments.head(5).iterrows():
        print(f"    Crime {int(row['point_id'])} at ({row['x']:.1f}, {row['y']:.1f}) "
              f"-> Node {int(row['assigned_node_id'])} at distance {row['distance']:.2f}m")
    print()
    
    # Calculate statistics
    crime_counts = assignments['assigned_node_id'].value_counts()
    max_crimes = crime_counts.max()
    avg_crimes = crime_counts.mean()
    
    print("  Crime Density Statistics:")
    print(f"    Maximum crimes at one node: {max_crimes}")
    print(f"    Average crimes per active node: {avg_crimes:.1f}")
    print(f"    Nodes with crimes: {len(crime_counts)}/{len(nodes)}")
    print()
    
    # Launch interactive visualization
    print("=" * 80)
    print("INTERACTIVE HEATMAP VISUALIZATION")
    print("=" * 80)
    print()
    print("Description:")
    print("  LEFT:  Original street network (no crime data)")
    print("  RIGHT: Crime density heatmap (with assigned crimes)")
    print()
    print("Heatmap Color Scale:")
    print("  • Gray:   Low/no crimes")
    print("  • Yellow: Medium density")
    print("  • Orange: High density")
    print("  • Red:    Very high density (hotspots)")
    print()
    print("Interactive Controls:")
    print("  • Left-click + drag:  Pan/move view")
    print("  • Mouse wheel:        Zoom in/out")
    print("  • Right-click + drag: Rotate view")
    print("  • N key:              Toggle nodes visibility")
    print("  • L key:              Toggle lines/segments visibility")
    print("  • R key:              Reset camera view")
    print("  • Q or ESC:           Close window")
    print()
    print("Launching visualization...")
    print("=" * 80)
    
    # Create map index without crimes (for left side)
    map_index_plain = acj.MapIndex(graph)
    
    # Create map index with crimes (for right side)
    # The render_comparison will use get_render_data to show the heatmap
    acj.render_comparison(
        map_index_plain,
        map_index,
        assignments_right=assignments,
        title=f"Crime Heatmap Visualization - Synthetic Network",
        title_left=f"Original Network ({len(graph.nodes)} nodes)",
        title_right=f"Crime Heatmap ({len(crimes)} crimes assigned, max {max_crimes}/node)"
    )
    
    print()
    print("=" * 80)
    print("Example completed successfully!")
    print()
    print("Summary:")
    print(f"  Network: {len(nodes)} nodes, {len(segments)} segments")
    print(f"  Crimes: {len(crimes)} total")
    print(f"  Hotspots: {len(crime_counts)} junctions with assigned crimes")
    print(f"  Max density: {max_crimes} crimes at one junction")
    print()
    print("Algorithm Features:")
    print("  - CGAL Delaunay triangulation for O(log n) nearest neighbor queries")
    print("  - Neighbor smoothing for better heatmap visualization")
    print("  - Robust outlier handling using percentile normalization")
    print("  - GPU-accelerated rendering with VisPy/OpenGL")
    print("=" * 80)


if __name__ == "__main__":
    main()

