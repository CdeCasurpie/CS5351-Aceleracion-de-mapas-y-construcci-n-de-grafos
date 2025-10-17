"""
Example usage of the ACJ library.

This example demonstrates how to:
1. Load graph data from pandas DataFrames
2. Create a MapIndex for spatial queries
3. Assign points to nearest graph endpoints (nodes)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import acj
import pandas as pd
import numpy as np


def main():
    print("=" * 70)
    print("ACJ Library - Example Usage")
    print("=" * 70)
    print()
    
    # Step 1: Create sample graph data (simple street network)
    print("[1/4] Creating sample graph data...")
    print()
    
    nodes = pd.DataFrame({
        'node_id': [0, 1, 2, 3, 4],
        'x': [0.0, 100.0, 200.0, 300.0, 150.0],
        'y': [0.0, 0.0, 100.0, 0.0, 100.0]
    })
    
    segments = pd.DataFrame({
        'segment_id': [0, 1, 2, 3],
        'node_start': [0, 1, 2, 3],
        'node_end': [1, 2, 4, 4],
        'x1': [0.0, 100.0, 200.0, 300.0],
        'y1': [0.0, 0.0, 100.0, 0.0],
        'x2': [100.0, 200.0, 150.0, 150.0],
        'y2': [0.0, 100.0, 100.0, 100.0]
    })
    
    print(f"  Nodes: {len(nodes)}")
    print(f"  Segments: {len(segments)}")
    print()
    
    # Step 2: Load graph using ACJ
    print("[2/4] Loading graph data...")
    graph = acj.load_graph(nodes, segments)
    print(f"  Graph loaded: {graph}")
    print()
    
    # Step 3: Create MapIndex
    print("[3/4] Building spatial index...")
    map_index = acj.MapIndex(graph)
    print(f"  MapIndex created: {map_index}")
    print()
    
    # Step 4: Create sample points (e.g., crime locations)
    print("[4/4] Assigning sample points to nearest nodes...")
    print()
    
    points = pd.DataFrame({
        'point_id': [0, 1, 2, 3, 4],
        'x': [10.0, 105.0, 195.0, 290.0, 155.0],
        'y': [5.0, 10.0, 95.0, 5.0, 90.0],
        'crime_type': ['robbery', 'assault', 'theft', 'vandalism', 'burglary']
    })
    
    print("Sample points to assign:")
    print(points)
    print()
    
    # Perform assignment
    print("Performing endpoint assignment...")
    assignments = map_index.assign_to_endpoints(points)
    
    print()
    print("Assignment results:")
    print(assignments[['point_id', 'crime_type', 'assigned_node_id', 'distance']])
    print()
    
    # Summary statistics
    print("=" * 70)
    print("Summary Statistics")
    print("=" * 70)
    print(f"Total points assigned: {len(assignments)}")
    print(f"Average distance: {assignments['distance'].mean():.2f} meters")
    print(f"Max distance: {assignments['distance'].max():.2f} meters")
    print(f"Min distance: {assignments['distance'].min():.2f} meters")
    print()
    
    # Distribution by node
    print("Points per node:")
    node_counts = assignments['assigned_node_id'].value_counts().sort_index()
    for node_id, count in node_counts.items():
        print(f"  Node {node_id}: {count} points")
    print()
    
    print("=" * 70)
    print("Example completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
