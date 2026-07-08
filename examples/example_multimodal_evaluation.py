import os
import sys
import osmnx as ox

# Ensure geojac package is accessible
sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
)
from geojac import (
    UrbanNetwork,
    ACJTopologicalEvaluator,
    CompressionRatioMetric,
    SemanticSpeedDistortionMetric,
)


def main():
    print("1. Descargando grafo de OSMnx (Barranco, Lima, Peru)...")
    # Using a small bounding box or place to ensure it runs quickly
    try:
        G = ox.graph_from_place("Barranco, Lima, Peru", network_type="drive")
    except Exception:
        print("Error descargando desde OSMnx. Fallback a un bounding box pequeño...")
        # Fallback coordinate box for Barranco
        G = ox.graph_from_bbox(
            north=-12.13, south=-12.15, east=-77.01, west=-77.03, network_type="drive"
        )

    print("\n2. Parseando grafo a UrbanNetwork...")
    network = UrbanNetwork.from_networkx(G)
    print("Estado inicial del grafo:")
    print(network)

    print("\n3. Instanciando ACJTopologicalEvaluator...")
    metrics = [CompressionRatioMetric(), SemanticSpeedDistortionMetric()]
    evaluator = ACJTopologicalEvaluator(network, metrics)

    print(
        "\n4. Ejecutando la simplificación y evaluación (esto puede tardar unos segundos)..."
    )
    results = evaluator.evaluate()

    print("\n5. Resultados de Métricas:")
    for metric_name, value in results.items():
        print(f"   - {metric_name}: {value:.4f}")

    print("\n6. Trazabilidad Semántica:")
    simplified_network = evaluator.simplified_network
    print(simplified_network)

    if len(simplified_network.edge_metadata) > 0:
        sample_id = list(simplified_network.edge_metadata.keys())[0]
        metadata = simplified_network.edge_metadata[sample_id]
        print(f"\nEjemplo de fusión semántica en nueva arista {sample_id}:")
        for k, v in metadata.items():
            print(f"   - {k}: {v}")
    else:
        print("\nNo se encontraron metadatos en el grafo simplificado.")

    print("\n7. Renderizando comparación de grafos...")
    from geojac import MapIndex, GraphData, render_comparison

    try:

        def add_coords_to_edges(nodes_df, edges_df):
            edges = edges_df.copy()
            nodes = nodes_df.set_index("node_id")
            edges["x1"] = edges["node_start"].map(nodes["x"])
            edges["y1"] = edges["node_start"].map(nodes["y"])
            edges["x2"] = edges["node_end"].map(nodes["x"])
            edges["y2"] = edges["node_end"].map(nodes["y"])
            return edges

        original_edges_df = add_coords_to_edges(network.nodes_df, network.edges_df)
        simplified_edges_df = add_coords_to_edges(
            simplified_network.nodes_df, simplified_network.edges_df
        )

        original_graph_data = GraphData(network.nodes_df, original_edges_df)
        simplified_graph_data = GraphData(
            simplified_network.nodes_df, simplified_edges_df
        )

        map_index_original = MapIndex(original_graph_data)
        map_index_simplified = MapIndex(simplified_graph_data)

        render_comparison(
            map_index_original,
            map_index_simplified,
            title="Comparación: Grafo Original vs. Simplificado",
            title_left="Grafo Original",
            title_right="Simplificación Topológica",
        )
    except Exception as e:
        import traceback

        traceback.print_exc()
        print(
            f"\nNo se pudo renderizar la comparación (es posible que falte pyqt5 o dependencias de UI): {e}"
        )


if __name__ == "__main__":
    main()
