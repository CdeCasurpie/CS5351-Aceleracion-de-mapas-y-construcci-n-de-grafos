#!/usr/bin/env python3
"""
Interactive crime heatmap visualization with real OSM data.

This example demonstrates:
1. Loading real street network from OpenStreetMap
2. Generating realistic crime hotspots
3. Assigning crimes to nearest junctions using CGAL
4. Interactive heatmap visualization with side-by-side comparison

Configuration: Change CITY_NAME variable to analyze any city
"""

import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'build'))
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import acj
import pandas as pd
import numpy as np


# ============================================================================
# CONFIGURATION - Change these values to analyze different cities
# ============================================================================
CITY_NAME = "Barranco, Lima, Peru"  # Change to any city/location
N_CRIMES = 1000                      # Number of synthetic crimes to generate
# ============================================================================


def generate_crime_hotspots(graph, n_crimes=1000):
    """Generate realistic crime points based on the street network."""
    np.random.seed(42)
    
    # Get bounding box from graph nodes
    nodes = graph.nodes
    min_x, max_x = nodes['x'].min(), nodes['x'].max()
    min_y, max_y = nodes['y'].min(), nodes['y'].max()
    
    # Define hotspot centers (as percentages of the bounding box)
    # These represent high-crime areas in the district
    hotspot_specs = [
        # (x_percent, y_percent, num_crimes, spread_meters)
        (0.3, 0.7, int(n_crimes * 0.25), 50),   # Northwest hotspot (25%)
        (0.7, 0.6, int(n_crimes * 0.30), 60),   # Northeast hotspot (30%)
        (0.5, 0.3, int(n_crimes * 0.20), 40),   # South-center hotspot (20%)
        (0.4, 0.5, int(n_crimes * 0.15), 35),   # Center hotspot (15%)
    ]
    
    crimes_data = []
    crime_id = 0
    
    # Generate hotspot crimes
    for x_pct, y_pct, count, spread in hotspot_specs:
        center_x = min_x + (max_x - min_x) * x_pct
        center_y = min_y + (max_y - min_y) * y_pct
        
        for _ in range(count):
            # Add Gaussian noise around hotspot center
            x = center_x + np.random.normal(0, spread)
            y = center_y + np.random.normal(0, spread)
            
            # Clip to bounding box
            x = max(min_x, min(max_x, x))
            y = max(min_y, min(max_y, y))
            
            crime_type = np.random.choice(['Robbery', 'Theft', 'Assault'], p=[0.5, 0.3, 0.2])
            
            crimes_data.append({
                'point_id': crime_id,
                'x': x,
                'y': y,
                'type': crime_type
            })
            crime_id += 1
    
    # Generate random background crimes (10% of total)
    remaining = n_crimes - crime_id
    for _ in range(remaining):
        x = np.random.uniform(min_x, max_x)
        y = np.random.uniform(min_y, max_y)
        
        crimes_data.append({
            'point_id': crime_id,
            'x': x,
            'y': y,
            'type': 'Other'
        })
        crime_id += 1
    
    return pd.DataFrame(crimes_data)


def print_crime_statistics(assignments, n_crimes):
    """Print detailed crime assignment statistics."""
    print("\n" + "=" * 80)
    print("CRIME ASSIGNMENT STATISTICS")
    print("=" * 80)
    
    crime_counts = assignments['assigned_node_id'].value_counts()
    
    print(f"\nTotal crimes: {len(assignments)}")
    print(f"Unique nodes with crimes: {len(crime_counts)}")
    print(f"Average crimes per active node: {crime_counts.mean():.1f}")
    print(f"Maximum crimes at one node: {crime_counts.max()}")
    print(f"Minimum crimes at active nodes: {crime_counts.min()}")
    
    print(f"\nTop 10 crime hotspot nodes:")
    for i, (node_id, count) in enumerate(crime_counts.head(10).items(), 1):
        percentage = (count / n_crimes) * 100
        print(f"  {i:2d}. Node {node_id:5d}: {count:4d} crimes ({percentage:5.2f}%)")
    
    # Crime type distribution
    if 'type' in assignments.columns:
        type_counts = assignments['type'].value_counts()
        print(f"\nCrime type distribution:")
        for crime_type, count in type_counts.items():
            percentage = (count / n_crimes) * 100
            print(f"  {crime_type:12s}: {count:5d} ({percentage:5.2f}%)")
    
    # Distance statistics
    print(f"\nAssignment distance statistics:")
    print(f"  Mean distance: {assignments['distance'].mean():.2f} meters")
    print(f"  Median distance: {assignments['distance'].median():.2f} meters")
    print(f"  Max distance: {assignments['distance'].max():.2f} meters")
    print(f"  Min distance: {assignments['distance'].min():.2f} meters")


def main():
    print("=" * 80)
    print("ACJ Crime Heatmap - Real OSM Data")
    print("=" * 80)
    print(f"Location: {CITY_NAME}")
    print(f"Crimes: {N_CRIMES}")
    print("=" * 80)
    print()
    
    cache_dir = "./cache"
    
    # Step 1: Load real street network from OSM
    print(f"[1/5] Loading street network from OpenStreetMap...")
    print(f"      Location: {CITY_NAME}")
    
    start_time = time.time()
    try:
        graph = acj.load_map(CITY_NAME, cache_dir=cache_dir, network_type="drive")
        load_time = time.time() - start_time
        print(f"      ✓ Network loaded successfully in {load_time:.2f} seconds")
        print(f"      Nodes: {len(graph.nodes)}")
        print(f"      Segments: {len(graph.segments)}")
    except Exception as e:
        print(f"      ✗ ERROR: Could not load map: {e}")
        print("      Please check your internet connection and OSMnx installation.")
        return
    
    print()
    
    # Step 2: Build spatial index
    print(f"[2/5] Building spatial index with CGAL Delaunay triangulation...")
    start_time = time.time()
    map_index = acj.MapIndex(graph)
    index_time = time.time() - start_time
    print(f"      ✓ Spatial index built in {index_time:.2f} seconds")
    print()
    
    # Step 3: Generate crime data
    print(f"[3/5] Generating {N_CRIMES} synthetic crime points...")
    print(f"      Distribution: 4 major hotspots + random background crimes")
    crimes = generate_crime_hotspots(graph, n_crimes=N_CRIMES)
    print(f"      ✓ Generated {len(crimes)} crime points")
    print()
    
    # Step 4: Assign crimes to junctions
    print(f"[4/5] Assigning crimes to nearest street junctions...")
    start_time = time.time()
    assignments = map_index.assign_to_endpoints(crimes)
    assign_time = time.time() - start_time
    print(f"      ✓ Assignment completed in {assign_time:.2f} seconds")
    print(f"      Performance: {N_CRIMES/assign_time:.0f} assignments/second")
    
    # Print detailed statistics
    print_crime_statistics(assignments, N_CRIMES)
    
    print()
    
    # Step 5: Launch interactive visualization
    print("[5/5] Launching interactive visualization...")
    print()
    print("=" * 80)
    print("INTERACTIVE HEATMAP VISUALIZATION")
    print("=" * 80)
    print()
    print("Window Layout:")
    print("  LEFT:  Original street network (no crime data)")
    print("         • Gray nodes and white lines")
    print()
    print("  RIGHT: Crime density heatmap (with assigned crimes)")
    print("         • Color gradient shows crime concentration:")
    print("           - Gray:   No crimes or very low density")
    print("           - Yellow: Low-medium density")
    print("           - Orange: Medium-high density")
    print("           - Red:    High density (hotspots)")
    print()
    print("Interactive Controls:")
    print("  • Left-click + drag:  Pan/move view")
    print("  • Mouse wheel:        Zoom in/out")
    print("  • Right-click + drag: Rotate view")
    print("  • N key:              Toggle nodes visibility")
    print("  • L key:              Toggle lines/edges visibility")
    print("  • R key:              Reset camera view")
    print("  • Q or ESC:           Close window")
    print()
    print("Performance:")
    print(f"  • Network loading:    {load_time:.2f}s")
    print(f"  • Index building:     {index_time:.2f}s")
    print(f"  • Crime assignment:   {assign_time:.2f}s ({N_CRIMES/assign_time:.0f} ops/s)")
    print(f"  • Total preprocessing: {load_time + index_time + assign_time:.2f}s")
    print()
    print("Launching window...")
    print("=" * 80)
    
    # Create visualization
    map_index_plain = acj.MapIndex(graph)
    
    # Get statistics for titles
    crime_counts = assignments['assigned_node_id'].value_counts()
    max_crimes = crime_counts.max()
    active_nodes = len(crime_counts)
    
    acj.render_comparison(
        map_index_plain,
        map_index,
        assignments_right=assignments,
        title=f"Crime Heatmap Visualization - {CITY_NAME}",
        title_left=f"Original Network ({len(graph.nodes)} nodes)",
        title_right=f"Crime Heatmap ({N_CRIMES} crimes, {active_nodes} hotspots, max {max_crimes}/node)"
    )
    
    print()
    print("=" * 80)
    print("Visualization closed. Example completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()

