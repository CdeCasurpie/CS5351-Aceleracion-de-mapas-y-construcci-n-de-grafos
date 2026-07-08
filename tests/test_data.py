"""Test cases for graph data loading and validation."""
import os
import sys
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import geojac

class TestGraphDataLoading:
    def test_load_graph_valid_data(self):
        nodes = pd.DataFrame({'node_id': [0, 1, 2], 'x': [0.0, 100.0, 200.0], 'y': [0.0, 0.0, 100.0]})
        segments = pd.DataFrame({'segment_id': [0, 1], 'node_start': [0, 1], 'node_end': [1, 2],
                                 'x1': [0.0, 100.0], 'y1': [0.0, 0.0], 'x2': [100.0, 200.0], 'y2': [0.0, 100.0]})
        graph = geojac.load_graph(nodes, segments)
        assert len(graph.nodes) == 3
        assert len(graph.segments) == 2
        assert isinstance(graph, geojac.GraphData)

    def test_load_graph_missing_node_columns(self):
        nodes = pd.DataFrame({'node_id': [0, 1], 'x': [0.0, 100.0]})
        segments = pd.DataFrame({'segment_id': [0], 'node_start': [0], 'node_end': [1],
                                 'x1': [0.0], 'y1': [0.0], 'x2': [100.0], 'y2': [0.0]})
        with pytest.raises(ValueError, match="missing required columns"):
            geojac.load_graph(nodes, segments)

    def test_load_graph_missing_segment_columns(self):
        nodes = pd.DataFrame({'node_id': [0, 1], 'x': [0.0, 100.0], 'y': [0.0, 0.0]})
        segments = pd.DataFrame({'segment_id': [0], 'node_start': [0], 'node_end': [1]})
        with pytest.raises(ValueError, match="missing required columns"):
            geojac.load_graph(nodes, segments)

class TestMapLoading:
    @pytest.mark.skipif('SKIP_OSMNX_TESTS' in os.environ, reason="OSMnx tests skipped")
    def test_load_map_basic(self):
        try:
            graph = geojac.load_map("Liechtenstein", network_type="drive")
            assert len(graph.nodes) > 0
            assert len(graph.segments) > 0
            assert 'node_id' in graph.nodes.columns
            assert 'x' in graph.nodes.columns
            assert 'y' in graph.nodes.columns
        except Exception as e:
            pytest.skip(f"OSMnx test skipped due to: {e}")
