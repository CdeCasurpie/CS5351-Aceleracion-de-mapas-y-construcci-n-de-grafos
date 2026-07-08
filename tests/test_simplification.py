"""Layer 3: Python simplification API wrappers (geojac.simplify_graph_* via pandas I/O)."""

import pandas as pd
import geojac


def _nodes(*rows):
    return pd.DataFrame(rows, columns=["node_id", "x", "y"])


def _segs(*rows):
    return pd.DataFrame(
        rows, columns=["segment_id", "node_start", "node_end", "x1", "y1", "x2", "y2"]
    )


class TestTopologicalSimplificationAPI:
    def test_linear_chain_collapses_interior_degree2_nodes(self):
        nodes = _nodes([0, 0.0, 0.0], [1, 10.0, 0.0], [2, 20.0, 0.0], [3, 30.0, 0.0])
        segs = _segs(
            [0, 0, 1, 0.0, 0.0, 10.0, 0.0],
            [1, 1, 2, 10.0, 0.0, 20.0, 0.0],
            [2, 2, 3, 20.0, 0.0, 30.0, 0.0],
        )
        graph = geojac.load_graph(nodes, segs)
        simplified = geojac.simplify_graph_topological(graph)
        assert len(simplified.graph.nodes) == 2
        assert len(simplified.graph.segments) == 1

    def test_t_junction_preserves_all_anchor_nodes(self):
        nodes = _nodes(
            [0, 0.0, 0.0],
            [1, 10.0, 0.0],
            [2, 20.0, 0.0],
            [3, 10.0, 10.0],
            [4, 10.0, -10.0],
        )
        segs = _segs(
            [0, 0, 1, 0.0, 0.0, 10.0, 0.0],
            [1, 1, 2, 10.0, 0.0, 20.0, 0.0],
            [2, 1, 3, 10.0, 0.0, 10.0, 10.0],
            [3, 1, 4, 10.0, 0.0, 10.0, -10.0],
        )
        graph = geojac.load_graph(nodes, segs)
        simplified = geojac.simplify_graph_topological(graph)
        assert len(simplified.graph.nodes) == 5

    def test_geometric_nearby_intersections_merge_to_centroid(self):
        nodes = _nodes([0, 0.0, 0.0], [1, 5.0, 0.0], [2, 2.5, 0.0], [3, 2.5, 10.0])
        segs = _segs(
            [0, 0, 2, 0.0, 0.0, 2.5, 0.0],
            [1, 1, 2, 5.0, 0.0, 2.5, 0.0],
            [2, 2, 3, 2.5, 0.0, 2.5, 10.0],
        )
        graph = geojac.load_graph(nodes, segs)
        simplified = geojac.simplify_graph_geometric(graph, threshold_meters=10.0)
        assert len(simplified.graph.nodes) == 2
        assert len(simplified.graph.segments) == 1


class TestMinkowskiSimplificationAPI:
    def test_empty_graph_returns_zero_nodes_and_segments(self):
        nodes = pd.DataFrame(columns=["node_id", "x", "y"])
        segs = pd.DataFrame(
            columns=["segment_id", "node_start", "node_end", "x1", "y1", "x2", "y2"]
        )
        graph = geojac.load_graph(nodes, segs)
        simplified = geojac.simplify_graph(graph, method="minkowski")
        assert len(simplified.graph.nodes) == 0
        assert len(simplified.graph.segments) == 0

    def test_parallel_segments_within_radius_merge_to_single_medial_axis(self):
        nodes = _nodes([0, 0.0, 0.0], [1, 100.0, 0.0], [2, 0.0, 4.0], [3, 100.0, 4.0])
        segs = _segs([0, 0, 1, 0.0, 0.0, 100.0, 0.0], [1, 2, 3, 0.0, 4.0, 100.0, 4.0])
        graph = geojac.load_graph(nodes, segs)
        simplified = geojac.simplify_graph(
            graph, threshold_meters=5.0, method="minkowski"
        )
        assert len(simplified.graph.segments) == 1
        assert len(simplified.graph.nodes) == 2

    def test_parallel_segments_beyond_radius_remain_separate(self):
        nodes = _nodes([0, 0.0, 0.0], [1, 100.0, 0.0], [2, 0.0, 50.0], [3, 100.0, 50.0])
        segs = _segs([0, 0, 1, 0.0, 0.0, 100.0, 0.0], [1, 2, 3, 0.0, 50.0, 100.0, 50.0])
        graph = geojac.load_graph(nodes, segs)
        simplified = geojac.simplify_graph(
            graph, threshold_meters=5.0, method="minkowski"
        )
        assert len(simplified.graph.segments) == 2
        assert len(simplified.graph.nodes) == 4
