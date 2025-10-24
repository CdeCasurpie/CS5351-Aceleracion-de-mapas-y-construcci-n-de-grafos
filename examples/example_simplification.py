"""
Example: Graph Simplification with ACJ.

This example demonstrates the two graph simplification methods:
1. Topological: Removes degree-2 nodes (preserves topology)
2. Geometric: Merges nearby intersections using CGAL clustering
"""

import sys
import os
import time

# Ensure the 'acj' library (built in the build directory) can be imported
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

try:
    import acj
except ImportError:
    print("Error: Could not import the 'acj' library.")
    print("Make sure you have compiled the project first (e.g. 'make example-realtime').")
    print("This will create the library in the 'build' folder required by this script.")
    sys.exit(1)


def main():
    print("=" * 80)
    print("ACJ Library - Graph Simplification Example")
    print("=" * 80)
    print()
    
    # Configuration
    city_name = "Piedmont, California, USA"
    cache_dir = "./cache"
    
    # --- 1. Load Original Graph ---
    print(f"[1/3] Loading street network for '{city_name}'...")
    try:
        graph_original = acj.load_map(city_name, cache_dir=cache_dir, network_type="drive")
    except Exception as e:
        print(f"ERROR: Could not load map: {e}")
        print("Please check your internet connection and OSMnx installation.")
        return
    
    print(f"  Original: {len(graph_original.nodes)} nodes, {len(graph_original.segments)} segments")
    print()
    
    # --- 2. Topological Simplification ---
    print("[2/3] Topological simplification (preserves topology)...")
    start_time = time.time()
    graph_topo = acj.simplify_graph_topological(graph_original)
    topo_time = time.time() - start_time
    
    print(f"  Result: {len(graph_topo.nodes)} nodes, {len(graph_topo.segments)} segments")
    print(f"  Reduction: {((len(graph_original.nodes) - len(graph_topo.nodes)) / len(graph_original.nodes) * 100):.1f}%")
    print(f"  Time: {topo_time:.3f} seconds")
    print()
    
    # --- 3. Geometric Simplification ---
    print("[3/3] Geometric simplification (CGAL clustering)...")
    start_time = time.time()
    graph_geo = acj.simplify_graph_geometric(graph_original, threshold_meters=15.0)
    geo_time = time.time() - start_time
    
    print(f"  Result: {len(graph_geo.nodes)} nodes, {len(graph_geo.segments)} segments")
    print(f"  Reduction: {((len(graph_original.nodes) - len(graph_geo.nodes)) / len(graph_original.nodes) * 100):.1f}%")
    print(f"  Time: {geo_time:.3f} seconds")
    print()

    # --- 4. Summary ---
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Original:     {len(graph_original.nodes):,} nodes")
    print(f"Topological:  {len(graph_topo.nodes):,} nodes (preserves intersections)")
    print(f"Geometric:    {len(graph_geo.nodes):,} nodes (merges close intersections)")
    print()
    print("Use cases:")
    print("  - Topological: Real-time navigation, routing")
    print("  - Geometric:   Dense urban networks, data analysis")
    print("=" * 80)

    # --- 5. Interactive Visualization ---
    print("Launching interactive comparison...")
    print("Controls: Mouse drag=pan, Mouse wheel=zoom, N=nodes, L=lines, R=reset, Q=quit")
    print()
    
    index_original = acj.MapIndex(graph_original)
    index_geometric = acj.MapIndex(graph_geo)
    
    acj.render_comparison(
        index_original,
        index_geometric,
        title_left="Original Graph",
        title_right="Geometric Simplified (15m threshold)"
    )
    
    print("Example complete!")


if __name__ == "__main__":
    main()