"""
Example: Graph Simplification Comparison with ACJ.

This example demonstrates how to load a graph, apply a (currently placeholder)
simplification function, and visualize both the original and "simplified"
versions side by side using a synchronized camera comparison view.
"""

import sys
import os

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
    print("ACJ Library - Graph Simplification Comparison Example")
    print("=" * 80)
    print()
    
    # Configuration
    # We'll use a small and simple city for testing
    city_name = "Piedmont, California, USA"
    cache_dir = "./cache"
    
    # --- 1. Load the Original Graph ---
    print(f"[1/3] Loading original street network for '{city_name}'...")
    try:
        graph_original = acj.load_map(city_name, cache_dir=cache_dir, network_type="drive")
    except Exception as e:
        print(f"ERROR: Could not load map: {e}")
        print("Please check your internet connection and OSMnx installation.")
        return
    
    print(f"  Original graph: {len(graph_original.nodes)} nodes, {len(graph_original.segments)} segments")
    print()
    
    # --- 2. "Simplify" the Graph ---
    # NOTE: This uses your 'simplify_graph' function from acj/graph.py
    # For now, as you mentioned, it’s a placeholder that just returns the original.
    # Once you implement the logic, this script will use it automatically.
    print("[2/3] Simplifying graph (using placeholder function)...")
    
    # Here is where you call your simplification function.
    # 'threshold_meters=15.0' is just an example value.
    graph_simplified = acj.simplify_graph(graph_original, threshold_meters=15.0)
    
    if graph_original is graph_simplified:
        print("  WARNING: 'simplify_graph' is currently a placeholder. Showing the same graph on both sides.")
    
    print(f"  Simplified graph: {len(graph_simplified.nodes)} nodes, {len(graph_simplified.segments)} segments")
    print()

    # --- 3. Build Indices and Render Comparison ---
    print("[3/3] Building spatial indices and launching comparison tool...")
    
    # Create a MapIndex for each version of the graph
    index_original = acj.MapIndex(graph_original)
    index_simplified = acj.MapIndex(graph_simplified)
    
    print()
    print("=" * 80)
    print("INTERACTIVE COMPARISON CONTROLS:")
    print("=" * 80)
    print("  Mouse drag        : Pan the view (both views move together)")
    print("  Mouse wheel       : Zoom in/out (both views zoom together)")
    print("  'N' key           : Toggle Nodes (ON/OFF)")
    print("  'L' key           : Toggle Segments (Lines) (ON/OFF)")
    print("  'R' key           : Reset camera view")
    print("  'Q' or ESC        : Close window")
    print("=" * 80)

    # Call the rendering function
    acj.render.render_comparison(
        index_original,
        index_simplified,
        title_left="Original Graph",
        title_right="‘Simplified’ Graph (WIP)"
    )
    
    print()
    print("=" * 80)
    print("Comparison example complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
