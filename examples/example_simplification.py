"""
Example: Graph Simplification with ACJ.

This example demonstrates the two graph simplification methods:
1. Topological: Removes degree-2 nodes (preserves topology)
2. Geometric: Merges nearby intersections using CGAL clustering
"""

import os
import sys
import time

import osmnx as ox
from osmnx import utils_graph

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
    city_name = "Barranco, Lima, Peru"

    cache_dir = "./cache"

    # --- 1. Load Original Graph (as MultiDiGraph) ---
    print(f"[1/5] Loading street network for '{city_name}'...")
    try:
        # Asumimos que acj.load_map devuelve el grafo MultiDiGraph original
        graph_original_directed = acj.load_map(city_name, cache_dir=cache_dir, network_type="drive")
    except Exception as e:
        print(f"ERROR: Could not load map: {e}")
        print("Please check your internet connection and OSMnx installation.")
        return

    print(f"  Original (Directed): {len(graph_original_directed.nodes)} nodes, {len(graph_original_directed.edges)} edges")
    print()

    # --- 2. NEW STEP: Consolidate "Ida y Vuelta" ---
    print("[2/5] Consolidating dual carriageways (MultiDiGraph -> MultiGraph)...")
    # Esta función convierte el grafo dirigido en un grafo no dirigido,
    # fusionando las aristas de ida y vuelta en una sola.
    graph_original = utils_graph.get_undirected(graph_original_directed)
    print(f"  Consolidated: {len(graph_original.nodes)} nodes, {len(graph_original.edges)} edges")
    print()

    # --- 3. Topological Simplification ---
    # Ahora 'graph_original' ya está consolidado
    print("[3/5] Topological simplification (preserves topology)...")
    start_time = time.time()
    # Esta función ahora simplificará el grafo consolidado
    graph_topo = acj.simplify_graph_topological(graph_original)
    topo_time = time.time() - start_time

    # NOTA: Puede que necesites ajustar cómo cuentas los segmentos si acj.simplify_...
    # espera una propiedad 'segments'. Usaremos 'edges' (aristas) como un genérico.
    print(f"  Result: {len(graph_topo.nodes)} nodes, {len(graph_topo.edges)} edges")
    reduction_nodes = (len(graph_original.nodes) - len(graph_topo.nodes)) / len(graph_original.nodes) * 100
    print(f"  Reduction: {reduction_nodes:.1f}%")
    print(f"  Time: {topo_time:.3f} seconds")
    print()

    # --- 4. Geometric Simplification ---
    print("[4/5] Geometric simplification (CGAL clustering)...")
    start_time = time.time()
    # La simplificación geométrica también se basa en el grafo consolidado
    graph_geo = acj.simplify_graph_geometric(graph_original, threshold_meters=15.0)
    geo_time = time.time() - start_time

    print(f"  Result: {len(graph_geo.nodes)} nodes, {len(graph_geo.edges)} edges")
    reduction_nodes_geo = (len(graph_original.nodes) - len(graph_geo.nodes)) / len(graph_original.nodes) * 100
    print(f"  Reduction: {reduction_nodes_geo:.1f}%")
    print(f"  Time: {geo_time:.3f} seconds")
    print()

    # --- 5. Summary ---
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Original (Consolidated): {len(graph_original.nodes):,} nodes")
    print(f"Topological:             {len(graph_topo.nodes):,} nodes (preserves intersections)")
    print(f"Geometric:               {len(graph_geo.nodes):,} nodes (merges close intersections)")
    print()
    print("Use cases:")
    print("  - Topological: Real-time navigation, routing")
    print("  - Geometric:   Dense urban networks, data analysis")
    print("=" * 80)

    # --- 6. Interactive Visualization ---
    # Es posible que acj.MapIndex o acj.render_comparison esperen propiedades
    # específicas. Si 'segments' era una propiedad de tu grafo, asegúrate
    # de que 'utils_graph.get_undirected' la preserve correctamente.
    # Si 'segments' es un alias de 'edges', esto debería funcionar.
    print("Launching interactive comparison...")
    print("Controls: Mouse drag=pan, Mouse wheel=zoom, N=nodes, L=lines, R=reset, Q=quit")
    print()

    index_original = acj.MapIndex(graph_original)
    index_geometric = acj.MapIndex(graph_geo)

    acj.render_comparison(
        index_original,
        index_geometric,
        title_left="Original Graph (Consolidated)",
        title_right="Geometric Simplified (15m threshold)"
    )

    print("Example complete!")


if __name__ == "__main__":
    main()
