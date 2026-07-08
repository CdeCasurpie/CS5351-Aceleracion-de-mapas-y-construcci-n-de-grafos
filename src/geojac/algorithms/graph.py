"""
Graph simplification and preprocessing utilities.

This module provides functions for simplifying and preprocessing graph data
to improve performance and reduce complexity for spatial queries.
"""

import pandas as pd
import numpy as np
from collections import defaultdict
from typing import List
from geojac.data.io import GraphData, SimplificationResult
from geojac.algorithms.minkowski import simplify_graph_minkowski


def simplify_graph_topological(graph_data: GraphData) -> SimplificationResult:
    """
    Simplify a graph by consolidating nodes of degree 2 (topological simplification).

    This function removes intermediate nodes on paths, keeping only intersections.

    Args:
        graph_data: GraphData object to simplify

    Returns:
        GraphData object with simplified graph (only intersections remain)

    Example:
        >>> graph = geojac.load_graph(nodes_df, segments_df)
        topo_simplified = simplify_graph_topological(graph_data).graph
        >>> print(f"Reduced from {len(graph.nodes)} to {len(simplified.nodes)} nodes")

    Notes:
        - Preserves all intersections and connectivity
        - Dramatically reduces node count for OSM data
        - Maintains spatial accuracy (no coordinate changes)
        - Fast O(n) algorithm suitable for real-time use
    """
    nodes_df = graph_data.nodes.copy()
    segments_df = graph_data.segments.copy()

    if len(nodes_df) == 0 or len(segments_df) == 0:
        return SimplificationResult(graph_data)

    # Construimos listas de adyacencia direccionales (in/out)
    out_edges = defaultdict(list)
    in_edges = defaultdict(list)

    for _, seg in segments_df.iterrows():
        start, end = seg["node_start"], seg["node_end"]
        out_edges[start].append(end)
        in_edges[end].append(start)

    # Identificamos qué nodos conservar
    # Solo borramos si el nodo es estrictamente un punto intermedio (1 entrada, 1 salida)
    nodes_to_keep = set()
    all_nodes = set(nodes_df["node_id"])

    for node_id in all_nodes:
        in_deg = len(in_edges[node_id])
        out_deg = len(out_edges[node_id])

        if not (in_deg == 1 and out_deg == 1):
            nodes_to_keep.add(node_id)

    # Si conservamos todos, devolvemos el original
    if len(nodes_to_keep) == len(nodes_df):
        return SimplificationResult(graph_data)

    # Reconstruimos los segmentos saltando los nodos eliminados
    new_segments = []
    segment_id = 0
    visited_edges = set()

    for start_node in nodes_to_keep:
        for neighbor in out_edges[start_node]:
            edge = (start_node, neighbor)
            if edge in visited_edges:
                continue

            # Trazamos el camino
            path = [start_node]
            current = neighbor

            while current not in nodes_to_keep:
                path.append(current)
                visited_edges.add((path[-2], current))
                # Avanzamos al siguiente nodo
                next_nodes = out_edges[current]
                if len(next_nodes) != 1:
                    break
                current = next_nodes[0]

            # Añadimos el nodo final (que sí está en nodes_to_keep)
            path.append(current)
            visited_edges.add((path[-2], current))

            start_id = path[0]
            end_id = path[-1]

            if start_id != end_id:  # Evitar self-loops
                start_coords = nodes_df[nodes_df["node_id"] == start_id].iloc[0]
                end_coords = nodes_df[nodes_df["node_id"] == end_id].iloc[0]

                new_segments.append(
                    {
                        "segment_id": segment_id,
                        "node_start": start_id,
                        "node_end": end_id,
                        "x1": start_coords["x"],
                        "y1": start_coords["y"],
                        "x2": end_coords["x"],
                        "y2": end_coords["y"],
                    }
                )
                segment_id += 1

    new_nodes_df = (
        nodes_df[nodes_df["node_id"].isin(nodes_to_keep)].copy().reset_index(drop=True)
    )
    new_segments_df = (
        pd.DataFrame(new_segments)
        if new_segments
        else pd.DataFrame(
            columns=["segment_id", "node_start", "node_end", "x1", "y1", "x2", "y2"]
        )
    )

    return SimplificationResult(GraphData(new_nodes_df, new_segments_df))


def simplify_graph_geometric(
    graph_data: GraphData, threshold_meters: float = 10.0
) -> SimplificationResult:
    """
    Simplify a graph by merging nearby nodes using geometric distance threshold.

    This function implements geometric simplification using CGAL for high-performance
    spatial clustering. It first applies topological simplification, then merges
    nearby intersections based on distance threshold.

    Algorithm:
        1. Apply topological simplification (remove degree-2 nodes)
        2. Build spatial index of remaining nodes using CGAL
        3. Cluster nodes within threshold_meters distance
        4. Merge clusters into single nodes at centroids
        5. Reconnect all edges to new cluster nodes

    Args:
        graph_data: GraphData object to simplify
        threshold_meters: Distance threshold for merging nodes (in meters)

    Returns:
        GraphData object with geometrically simplified graph

    Example:
        >>> graph = geojac.load_graph(nodes_df, segments_df)
        >>> simplified = geojac.simplify_graph_geometric(graph, threshold_meters=15.0)

    Notes:
        - More aggressive simplification than topological
        - May change network topology slightly
        - Best for very dense networks with many close intersections
        - Uses CGAL for high-performance spatial operations
    """
    if len(graph_data.nodes) == 0:
        return SimplificationResult(graph_data)

    # Get node coordinates directly (skip topological to avoid removing valid intersections)
    node_coords = graph_data.nodes[["x", "y"]].values
    node_ids = graph_data.nodes["node_id"].values

    # Use slightly smaller threshold to avoid merging exact boundary matches (e.g. exactly 10.0m)
    safe_threshold = max(0.0, threshold_meters - 1e-5)
    clusters = _find_node_clusters(node_coords, safe_threshold)

    # Ensure isolated nodes (not clustered by CGAL) are kept as their own cluster of size 1
    clustered_indices = set()
    for cluster in clusters:
        clustered_indices.update(cluster)

    for i in range(len(node_coords)):
        if i not in clustered_indices:
            clusters.append([i])

    # Create mapping from old node IDs to new cluster IDs
    node_to_cluster = {}
    cluster_centers = {}

    for cluster_id, cluster_nodes in enumerate(clusters):
        cluster_node_ids = [node_ids[i] for i in cluster_nodes]
        cluster_coords = node_coords[cluster_nodes]

        # Calculate cluster centroid
        centroid_x = np.mean(cluster_coords[:, 0])
        centroid_y = np.mean(cluster_coords[:, 1])

        # Use the first node ID as the cluster representative
        cluster_rep_id = int(cluster_node_ids[0])
        cluster_centers[cluster_rep_id] = (centroid_x, centroid_y)

        # Map all nodes in cluster to representative
        for node_id in cluster_node_ids:
            node_to_cluster[int(node_id)] = cluster_rep_id

    # Create new nodes
    new_nodes_data = []
    for cluster_rep_id, (centroid_x, centroid_y) in cluster_centers.items():
        new_nodes_data.append(
            {"node_id": cluster_rep_id, "x": centroid_x, "y": centroid_y}
        )

    new_nodes_df = pd.DataFrame(new_nodes_data)

    # Update segments to use cluster representatives and deduplicate parallel edges
    new_segments_data = []
    segment_id_counter = 0
    seen_edges = set()

    for _, segment in graph_data.segments.iterrows():
        start_id = int(segment["node_start"])
        end_id = int(segment["node_end"])

        if start_id not in node_to_cluster or end_id not in node_to_cluster:
            continue

        start_cluster = node_to_cluster[start_id]
        end_cluster = node_to_cluster[end_id]

        if start_cluster == end_cluster:
            continue

        # Deduplication mechanism to prevent double segments
        edge_key = tuple(sorted([start_cluster, end_cluster]))
        if edge_key in seen_edges:
            continue
        seen_edges.add(edge_key)

        start_center = cluster_centers[start_cluster]
        end_center = cluster_centers[end_cluster]

        new_segments_data.append(
            {
                "segment_id": segment_id_counter,
                "node_start": start_cluster,
                "node_end": end_cluster,
                "x1": start_center[0],
                "y1": start_center[1],
                "x2": end_center[0],
                "y2": end_center[1],
            }
        )
        segment_id_counter += 1

    if len(new_segments_data) == 0:
        new_segments_df = pd.DataFrame(
            columns=["segment_id", "node_start", "node_end", "x1", "y1", "x2", "y2"]
        )
    else:
        new_segments_df = pd.DataFrame(new_segments_data)

    return SimplificationResult(GraphData(new_nodes_df, new_segments_df))


def _find_node_clusters(coords: np.ndarray, threshold: float) -> List[List[int]]:
    """
    Find clusters of nodes within threshold distance using CGAL spatial indexing.

    This function uses the CGAL core module to perform efficient spatial queries
    for clustering nearby nodes. It leverages the same CGAL infrastructure
    used by the MapIndex class for consistent performance.

    Args:
        coords: Nx2 array of node coordinates
        threshold: Distance threshold for clustering (in meters)

    Returns:
        List of clusters, where each cluster is a list of node indices
    """
    try:
        import sys
        import os

        project_root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        )
        build_root = os.path.join(project_root, "build")
        build_core = os.path.join(build_root, "src", "geojac", "core")

        for path in [build_root, build_core]:
            if os.path.exists(path) and path not in sys.path:
                sys.path.insert(0, path)

        import acj_core
    except ImportError as e:
        raise ImportError(
            "CGAL core module not available. Compile the extension first."
        ) from e

    n_nodes = len(coords)
    if n_nodes == 0:
        return []

    clusters = acj_core.find_clusters_cgal(coords, threshold)

    result = []
    for cluster in clusters:
        result.append(list(cluster))

    return result


def simplify_graph_parallel_cgal(
    nodes_df: pd.DataFrame,
    segments_df: pd.DataFrame,
    distance_threshold: float = 10.0,
    angle_threshold_deg: float = 5.0,
) -> SimplificationResult:
    """
    Simplify a graph by merging parallel segments.

    This function merges segments that are parallel and close to each other,
    which is useful for simplifying dual carriageways and parallel roads.

    Args:
        nodes_df: DataFrame with node data
        segments_df: DataFrame with segment data
        distance_threshold: Maximum distance between segments to merge (meters)
        angle_threshold_deg: Maximum angle difference for parallel segments (degrees)

    Returns:
        GraphData object with simplified graph

    Example:
        >>> graph = geojac.load_graph(nodes_df, segments_df)
        >>> simplified = geojac.simplify_graph_parallel_cgal(
        ...     graph.nodes, graph.segments,
        ...     distance_threshold=10.0, angle_threshold_deg=5.0
        ... )

    Note:
        This is a simplified implementation that merges nearby parallel segments.
        For large graphs, consider using the full CGAL implementation.
    """
    if len(nodes_df) == 0 or len(segments_df) == 0:
        return SimplificationResult(GraphData(nodes_df, segments_df))

    nodes = nodes_df.copy()
    segments = segments_df.copy()

    merged_segment_indices = set()
    new_nodes = []
    new_segments = []

    next_node_id = nodes["node_id"].max() + 1
    next_seg_id = segments["segment_id"].max() + 1
    node_replacements = {}

    for i, row1 in segments.iterrows():
        if i in merged_segment_indices:
            continue

        for j, row2 in segments.iterrows():
            if i >= j or j in merged_segment_indices:
                continue

            dx1, dy1 = row1["x2"] - row1["x1"], row1["y2"] - row1["y1"]
            dx2, dy2 = row2["x2"] - row2["x1"], row2["y2"] - row2["y1"]

            len1 = np.hypot(dx1, dy1)
            len2 = np.hypot(dx2, dy2)

            if len1 == 0 or len2 == 0:
                continue

            cos_theta = (dx1 * dx2 + dy1 * dy2) / (len1 * len2)
            cos_theta = max(-1.0, min(1.0, cos_theta))
            angle = np.degrees(np.arccos(abs(cos_theta)))

            if angle <= angle_threshold_deg:
                mid_x1, mid_y1 = (
                    (row1["x1"] + row1["x2"]) / 2,
                    (row1["y1"] + row1["y2"]) / 2,
                )
                mid_x2, mid_y2 = (
                    (row2["x1"] + row2["x2"]) / 2,
                    (row2["y1"] + row2["y2"]) / 2,
                )
                dist = np.hypot(mid_x1 - mid_x2, mid_y1 - mid_y2)

                if dist <= distance_threshold:
                    merged_segment_indices.add(i)
                    merged_segment_indices.add(j)

                    start_x = (row1["x1"] + row2["x1"]) / 2
                    start_y = (row1["y1"] + row2["y1"]) / 2
                    end_x = (row1["x2"] + row2["x2"]) / 2
                    end_y = (row1["y2"] + row2["y2"]) / 2

                    node_start_id = next_node_id
                    next_node_id += 1
                    node_end_id = next_node_id
                    next_node_id += 1

                    new_nodes.extend(
                        [
                            {"node_id": node_start_id, "x": start_x, "y": start_y},
                            {"node_id": node_end_id, "x": end_x, "y": end_y},
                        ]
                    )

                    new_segments.append(
                        {
                            "segment_id": next_seg_id,
                            "node_start": node_start_id,
                            "node_end": node_end_id,
                            "x1": start_x,
                            "y1": start_y,
                            "x2": end_x,
                            "y2": end_y,
                        }
                    )
                    next_seg_id += 1

                    node_replacements[row1["node_start"]] = node_start_id
                    node_replacements[row1["node_end"]] = node_end_id
                    node_replacements[row2["node_start"]] = node_start_id
                    node_replacements[row2["node_end"]] = node_end_id
                    break

    if not merged_segment_indices:
        return SimplificationResult(GraphData(nodes_df.copy(), segments_df.copy()))

    final_segments = [
        row.to_dict()
        for idx, row in segments.iterrows()
        if idx not in merged_segment_indices
    ]
    final_segments.extend(new_segments)

    final_nodes = [
        row.to_dict()
        for idx, row in nodes.iterrows()
        if row["node_id"] not in node_replacements
    ]
    final_nodes.extend(new_nodes)

    return SimplificationResult(
        GraphData(pd.DataFrame(final_nodes), pd.DataFrame(final_segments))
    )


def simplify_graph_guided_minkowski(
    graph: GraphData, radius: float = 5.0
) -> SimplificationResult:
    """
    Simplificación avanzada: Minkowski + Straight Skeleton + Poda Guiada por Topología.
    Utiliza el grafo original como ground-truth para eliminar la cristalización.
    """
    try:
        import acj_core
    except ImportError as e:
        raise ImportError(
            "CGAL core module not available. Compile the extension first."
        ) from e

    nodes_x = graph.nodes["x"].to_numpy(dtype="float64")
    nodes_y = graph.nodes["y"].to_numpy(dtype="float64")

    seg_start = graph.segments["node_start"].to_numpy(dtype="int64")
    seg_end = graph.segments["node_end"].to_numpy(dtype="int64")

    new_nodes_x, new_nodes_y, new_seg_start, new_seg_end = (
        acj_core.minkowski_guided_simplify(nodes_x, nodes_y, seg_start, seg_end, radius)
    )

    new_nodes = pd.DataFrame(
        {"node_id": range(len(new_nodes_x)), "x": new_nodes_x, "y": new_nodes_y}
    )

    x1_coords = new_nodes_x[new_seg_start]
    y1_coords = new_nodes_y[new_seg_start]
    x2_coords = new_nodes_x[new_seg_end]
    y2_coords = new_nodes_y[new_seg_end]

    new_segments = pd.DataFrame(
        {
            "segment_id": range(len(new_seg_start)),
            "node_start": new_seg_start,
            "node_end": new_seg_end,
            "x1": x1_coords,
            "y1": y1_coords,
            "x2": x2_coords,
            "y2": y2_coords,
        }
    )

    return SimplificationResult(GraphData(new_nodes, new_segments))


def simplify_graph(
    graph_data: GraphData, threshold_meters: float = 10.0, method: str = "geometric"
) -> SimplificationResult:
    """
    Simplify a graph using the most appropriate method based on threshold.

    This is the main simplification function that automatically chooses
    between topological and geometric simplification based on the threshold.

    Args:
        graph_data: GraphData object to simplify
        threshold_meters: Distance threshold for merging nodes (in meters),
                          or 'radius' if using minkowski methods.
        method: The simplification method to use ('auto', 'topological', 'geometric', 'parallel', 'minkowski', 'guided_minkowski')

    Returns:
        GraphData object with simplified graph
    """
    if method == "topological" or (method == "auto" and threshold_meters <= 0):
        return simplify_graph_topological(graph_data)
    elif method == "geometric" or (method == "auto" and threshold_meters > 0):
        return simplify_graph_geometric(graph_data, threshold_meters)
    elif method == "parallel":
        return simplify_graph_parallel_cgal(
            graph_data.nodes, graph_data.segments, threshold_meters
        )
    elif method == "minkowski":
        return simplify_graph_minkowski(graph_data, radius=threshold_meters)
    elif method == "guided_minkowski":
        return simplify_graph_guided_minkowski(graph_data, radius=threshold_meters)

    return graph_data
