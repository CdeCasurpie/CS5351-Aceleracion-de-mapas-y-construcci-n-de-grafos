import pandas as pd
import numpy as np
from collections import defaultdict
from geojac.data.io import GraphData, SimplificationResult


def simplify_graph_minkowski(
    graph_data: GraphData, radius: float = 5.0
) -> SimplificationResult:
    """Simplifica un grafo usando sumas de Minkowski y Straight Skeleton (CGAL vectorial)."""
    if len(graph_data.nodes) == 0 or len(graph_data.segments) == 0:
        return SimplificationResult(graph_data)

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

    nodes_array = np.ascontiguousarray(
        graph_data.nodes[["node_id", "x", "y"]].values, dtype=np.float64
    )
    segments_array = np.ascontiguousarray(
        graph_data.segments[["segment_id", "node_start", "node_end"]].values,
        dtype=np.float64,
    )

    cgal_result = acj_core._experimental.simplify_graph_minkowski_cgal(
        nodes_array, segments_array, float(radius)
    )
    nodes_list, segments_list = cgal_result.graph

    if not segments_list or not nodes_list:
        return SimplificationResult(
            GraphData(
                pd.DataFrame(columns=["node_id", "x", "y"]),
                pd.DataFrame(
                    columns=[
                        "segment_id",
                        "node_start",
                        "node_end",
                        "x1",
                        "y1",
                        "x2",
                        "y2",
                    ]
                ),
            )
        )

    node_coords = {n[0]: (n[1], n[2]) for n in nodes_list}
    adj = defaultdict(set)
    for s in segments_list:
        adj[s[1]].add(s[2])
        adj[s[2]].add(s[1])

    changed = True
    while changed:
        changed = False
        leaves = [node for node, neighbors in adj.items() if len(neighbors) == 1]
        for leaf in leaves:
            if leaf not in adj:
                continue
            if len(adj) <= 2:
                break
            neighbor = list(adj[leaf])[0]

            x1, y1 = node_coords[leaf]
            x2, y2 = node_coords[neighbor]
            dist = np.hypot(x2 - x1, y2 - y1)

            if dist <= radius * 2.0:
                adj[neighbor].remove(leaf)
                del adj[leaf]
                changed = True

    valid_nodes = set(adj.keys())

    final_nodes = [
        {"node_id": n[0], "x": n[1], "y": n[2]}
        for n in nodes_list
        if n[0] in valid_nodes
    ]

    final_segments = []
    seg_id_counter = 0
    for s in segments_list:
        if s[1] in valid_nodes and s[2] in valid_nodes:
            final_segments.append(
                {
                    "segment_id": seg_id_counter,
                    "node_start": s[1],
                    "node_end": s[2],
                    "x1": node_coords[s[1]][0],
                    "y1": node_coords[s[1]][1],
                    "x2": node_coords[s[2]][0],
                    "y2": node_coords[s[2]][1],
                }
            )
            seg_id_counter += 1

    minkowski_graph = GraphData(pd.DataFrame(final_nodes), pd.DataFrame(final_segments))
    from geojac.algorithms.graph import simplify_graph_topological

    return simplify_graph_topological(minkowski_graph)
