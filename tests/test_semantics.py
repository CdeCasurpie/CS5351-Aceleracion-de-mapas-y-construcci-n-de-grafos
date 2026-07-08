import os
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from geojac.core.network import UrbanNetwork
from geojac.data.io import SimplificationResult, GraphData
from geojac.core.semantics import resolve_semantics


def test_resolve_semantics():
    original = UrbanNetwork()
    original.node_metadata = {
        1: {"speed": 50, "name": "A", "is_highway": True},
        2: {"speed": 70, "name": "B", "is_highway": False},
        3: {"speed": 60, "name": "A", "is_highway": False},
    }
    original.edge_metadata = {
        100: {"lanes": 2, "surface": "asphalt"},
        101: {"lanes": 4, "surface": "concrete"},
    }

    # Simulate SimplificationResult
    # new node 10 absorbs old nodes 1 and 2
    # new node 11 absorbs old node 3
    node_lineage = {10: [1, 2], 11: [3]}

    # new edge 500 absorbs old edges 100 and 101
    edge_lineage = {500: [100, 101]}

    nodes_df = pd.DataFrame({"node_id": [10, 11], "x": [0.0, 1.0], "y": [0.0, 1.0]})
    segments_df = pd.DataFrame(
        {
            "segment_id": [500],
            "node_start": [10],
            "node_end": [11],
            "x1": [0.0],
            "y1": [0.0],
            "x2": [1.0],
            "y2": [1.0],
        }
    )

    graph = GraphData(nodes_df, segments_df)
    result = SimplificationResult(
        graph, node_lineage=node_lineage, edge_lineage=edge_lineage
    )

    resolved = resolve_semantics(original, result)

    assert len(resolved.node_metadata) == 2
    assert resolved.node_metadata[10]["speed"] == 60.0  # (50+70)/2
    assert resolved.node_metadata[10]["name"] == "A | B"
    assert resolved.node_metadata[10]["is_highway"] is True  # True OR False

    assert resolved.node_metadata[11]["speed"] == 60.0
    assert resolved.node_metadata[11]["name"] == "A"
    assert resolved.node_metadata[11]["is_highway"] is False

    assert len(resolved.edge_metadata) == 1
    assert resolved.edge_metadata[500]["lanes"] == 3.0  # (2+4)/2
    assert resolved.edge_metadata[500]["surface"] == "asphalt | concrete"

    assert len(resolved.nodes_df) == 2
    assert len(resolved.edges_df) == 1
