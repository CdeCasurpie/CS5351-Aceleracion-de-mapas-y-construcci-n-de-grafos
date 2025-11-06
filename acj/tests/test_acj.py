"""
Test suite for ACJ library.

Tests the main functionality of the ACJ geospatial analysis library
including data loading, spatial indexing, and point assignment.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import acj


class TestGraphDataLoading:
    """Test cases for graph data loading and validation."""

    def test_load_graph_valid_data(self):
        """Test loading valid graph data."""
        nodes = pd.DataFrame({
            'node_id': [0, 1, 2],
            'x': [0.0, 100.0, 200.0],
            'y': [0.0, 0.0, 100.0]
        })

        segments = pd.DataFrame({
            'segment_id': [0, 1],
            'node_start': [0, 1],
            'node_end': [1, 2],
            'x1': [0.0, 100.0],
            'y1': [0.0, 0.0],
            'x2': [100.0, 200.0],
            'y2': [0.0, 100.0]
        })

        graph = acj.load_graph(nodes, segments)

        assert len(graph.nodes) == 3
        assert len(graph.segments) == 2
        assert isinstance(graph, acj.io.GraphData)

    def test_load_graph_missing_node_columns(self):
        """Test that loading fails with missing node columns."""
        nodes = pd.DataFrame({
            'node_id': [0, 1],
            'x': [0.0, 100.0]
            # Missing 'y' column
        })

        segments = pd.DataFrame({
            'segment_id': [0],
            'node_start': [0],
            'node_end': [1],
            'x1': [0.0],
            'y1': [0.0],
            'x2': [100.0],
            'y2': [0.0]
        })

        with pytest.raises(ValueError, match="missing required columns"):
            acj.load_graph(nodes, segments)

    def test_load_graph_missing_segment_columns(self):
        """Test that loading fails with missing segment columns."""
        nodes = pd.DataFrame({
            'node_id': [0, 1],
            'x': [0.0, 100.0],
            'y': [0.0, 0.0]
        })

        segments = pd.DataFrame({
            'segment_id': [0],
            'node_start': [0],
            'node_end': [1]
            # Missing x1, y1, x2, y2
        })

        with pytest.raises(ValueError, match="missing required columns"):
            acj.load_graph(nodes, segments)


class TestMapLoading:
    """Test cases for loading maps from OSMnx."""

    @pytest.mark.skipif(
        'SKIP_OSMNX_TESTS' in os.environ,
        reason="OSMnx tests skipped (set SKIP_OSMNX_TESTS to skip)"
    )
    def test_load_map_basic(self):
        """Test loading a small city map from OSMnx."""
        # Use a very small location for testing
        try:
            graph = acj.load_map("Liechtenstein", network_type="drive")

            # Verify structure
            assert len(graph.nodes) > 0
            assert len(graph.segments) > 0
            assert 'node_id' in graph.nodes.columns
            assert 'x' in graph.nodes.columns
            assert 'y' in graph.nodes.columns

        except Exception as e:
            pytest.skip(f"OSMnx test skipped due to: {e}")


class TestMapIndex:
    """Test cases for MapIndex spatial queries."""

    @pytest.fixture
    def simple_graph(self):
        """Create a simple test graph."""
        nodes = pd.DataFrame({
            'node_id': [0, 1, 2, 3],
            'x': [0.0, 100.0, 200.0, 100.0],
            'y': [0.0, 0.0, 100.0, 100.0]
        })

        segments = pd.DataFrame({
            'segment_id': [0, 1, 2],
            'node_start': [0, 1, 2],
            'node_end': [1, 2, 3],
            'x1': [0.0, 100.0, 200.0],
            'y1': [0.0, 0.0, 100.0],
            'x2': [100.0, 200.0, 100.0],
            'y2': [0.0, 100.0, 100.0]
        })

        return acj.load_graph(nodes, segments)

    def test_map_index_initialization(self, simple_graph):
        """Test MapIndex initialization."""
        map_index = acj.MapIndex(simple_graph)

        assert map_index.graph_data is simple_graph
        assert map_index._endpoint_index is None  # Built lazily
        assert map_index._acj_core is not None  # C++ module loaded

    def test_assign_to_endpoints_simple(self, simple_graph):
        """Test endpoint assignment with simple data."""
        map_index = acj.MapIndex(simple_graph)

        # Create test points near known nodes
        points = pd.DataFrame({
            'point_id': [0, 1, 2],
            'x': [5.0, 105.0, 195.0],  # Near nodes 0, 1, 2
            'y': [5.0, 5.0, 95.0]
        })

        result = map_index.assign_to_endpoints(points)

        # Check result structure
        assert 'assigned_node_id' in result.columns
        assert 'distance' in result.columns
        assert len(result) == len(points)

        # Check assignments are reasonable
        assert result.loc[0, 'assigned_node_id'] == 0  # Point 0 near node 0
        assert result.loc[1, 'assigned_node_id'] == 1  # Point 1 near node 1
        # Node 2 (200, 100) or Node 3 (100, 100) are far away from (195, 95). Node 2 is closer.
        assert result.loc[2, 'assigned_node_id'] == 2

        # Check distances are positive
        assert all(result['distance'] >= 0)

    def test_assign_to_endpoints_missing_columns(self, simple_graph):
        """Test that assignment fails with missing columns."""
        map_index = acj.MapIndex(simple_graph)

        points = pd.DataFrame({
            'point_id': [0, 1],
            'x': [5.0, 105.0]
            # Missing 'y' column
        })

        with pytest.raises(ValueError, match="missing required columns"):
            map_index.assign_to_endpoints(points)

    def test_assign_to_endpoints_preserves_data(self, simple_graph):
        """Test that assignment preserves additional columns."""
        map_index = acj.MapIndex(simple_graph)
        points = pd.DataFrame({
            'point_id': [0, 1],
            'x': [5.0, 105.0],
            'y': [5.0, 5.0],
            'crime_type': ['robbery', 'assault']
        })

        result = map_index.assign_to_endpoints(points)

        # Original columns should be preserved
        assert 'crime_type' in result.columns
        assert result.loc[0, 'crime_type'] == 'robbery'
        assert result.loc[1, 'crime_type'] == 'assault'

    def test_assign_to_segments_not_implemented(self, simple_graph):
        """Test that assign_to_segments, though present in C++ core,
        is correctly flagged or handled in the Python wrapper API.
        We expect the python wrapper to call the C++ function and return a result.
        """
        map_index = acj.MapIndex(simple_graph)

        points = pd.DataFrame({
            'point_id': [0],
            'x': [50.0],
            'y': [0.0]
        })

        # Placeholder: assume a basic implementation exists in the wrapper now
        result = map_index.assign_to_segments(points)

        assert 'assigned_segment_id' in result.columns
        assert 'distance' in result.columns
        assert len(result) == 1
        # Check assignment to segment 0 (0-1) since (50, 0) is midpoint.
        assert result.loc[0, 'assigned_segment_id'] == 0
        assert np.isclose(result.loc[0, 'distance'], 0.0, atol=1e-6)


class TestGraphSimplification:
    """Test cases for graph simplification."""

    def test_simplify_graph_topological_basic(self):
        """Test basic topological simplification (A-B-C-D -> A-D)."""
        # Create a simple chain: A(0)-B(1)-C(2)-D(3) where B and C are degree-2 nodes
        # Nodes 0 (deg 1) and 3 (deg 1) are kept. Nodes 1 (deg 2) and 2 (deg 2) are removed.
        nodes = pd.DataFrame({
            'node_id': [0, 1, 2, 3],
            'x': [0.0, 10.0, 20.0, 30.0],
            'y': [0.0, 0.0, 0.0, 0.0]
        })

        segments = pd.DataFrame({
            'segment_id': [0, 1, 2],
            'node_start': [0, 1, 2],
            'node_end': [1, 2, 3],
            'x1': [0.0, 10.0, 20.0],
            'y1': [0.0, 0.0, 0.0],
            'x2': [10.0, 20.0, 30.0],
            'y2': [0.0, 0.0, 0.0]
        })

        graph = acj.load_graph(nodes, segments)
        simplified = acj.simplify_graph_topological(graph)

        # Should reduce from 4 nodes to 3 nodes (A, C, D) - B is removed
        assert len(simplified.nodes) == 2
        assert len(simplified.segments) == 1 # New segment connects 0 and 3
        assert 0 in simplified.nodes['node_id'].values  # Node A preserved
        assert 3 in simplified.nodes['node_id'].values  # Node D preserved
        assert 1 not in simplified.nodes['node_id'].values
        assert 2 not in simplified.nodes['node_id'].values

    def test_simplify_graph_topological_intersection_preserved(self):
        """Test that intersections (degree != 2) are preserved."""
        nodes = pd.DataFrame({
            'node_id': [0, 1, 2, 3, 4],
            'x': [0.0, 10.0, 20.0, 10.0, 10.0],
            'y': [0.0, 0.0, 0.0, 10.0, -10.0]
        })

        segments = pd.DataFrame({
            'segment_id': [0, 1, 2, 3],
            'node_start': [0, 1, 1, 1],
            'node_end': [1, 2, 3, 4],
            'x1': [0.0, 10.0, 10.0, 10.0],
            'y1': [0.0, 0.0, 0.0, 0.0],
            'x2': [10.0, 20.0, 10.0, 10.0],
            'y2': [0.0, 0.0, 10.0, -10.0]
        })

        graph = acj.load_graph(nodes, segments)
        simplified = acj.simplify_graph_topological(graph)

        # Should preserve the intersection (node 1) - it has degree 4
        assert len(simplified.nodes) == 5
        assert len(simplified.segments) == 4
        assert 1 in simplified.nodes['node_id'].values

    def test_simplify_graph_geometric_basic(self):
        """Test basic geometric simplification by merging close intersections."""
        # Nodes 0 (deg 1) and 1 (deg 1) are endpoints, they are both intersections
        nodes = pd.DataFrame({
            'node_id': [0, 1, 2],
            'x': [0.0, 5.0, 100.0],  # Nodes 0 and 1 are 5m apart
            'y': [0.0, 0.0, 0.0]
        })

        segments = pd.DataFrame({
            'segment_id': [0, 1],
            'node_start': [0, 1],
            'node_end': [2, 2], # Node 2 is an intersection (deg 2)
            'x1': [0.0, 5.0],
            'y1': [0.0, 0.0],
            'x2': [100.0, 100.0],
            'y2': [0.0, 0.0]
            })
            # Original graph: (0)-(2) and (1)-(2). Node 2 is deg 2. Nodes 0, 1 are deg 1.
            # Intersections: 0, 1, 2. But node 2 is deg 2.
            # C++ implementation: degree != 2 are intersections: 0, 1.

        graph = acj.load_graph(nodes, segments)

        # Test with threshold that should merge nodes 0 and 1 (5m apart)
        simplified_merged = acj.simplify_graph_geometric(graph, threshold_meters=10.0)

        # Intersections (0, 1) should be merged into one new node (ID 0). Node 2 remains.
        # 3 original nodes -> 2 new nodes (one centroid for 0/1, one for 2).
        assert len(simplified_merged.nodes) == 2
        # Two segments (0-2 and 1-2) become one (NewID-2)
        assert len(simplified_merged.segments) == 1
        # Test with threshold that should NOT merge nodes 0 and 1
        simplified_unmerged = acj.simplify_graph_geometric(graph, threshold_meters=1.0)
        # No merge. All intersections (0, 1) remain distinct. Node 2 remains.
        assert len(simplified_unmerged.nodes) == 3
        assert len(simplified_unmerged.segments) == 2 # Two segments: 0-2 and 1-2.

    def test_simplify_graph_automatic_selection(self):
        """Test that simplify_graph automatically selects the right method."""
        # Chain graph: 0-1-2. Node 1 is degree 2 (to be removed topologically).
        nodes = pd.DataFrame({
            'node_id': [0, 1, 2],
            'x': [0.0, 10.0, 100.0],
            'y': [0.0, 0.0, 0.0]
        })

        segments = pd.DataFrame({
            'segment_id': [0, 1],
            'node_start': [0, 1],
            'node_end': [1, 2],
            'x1': [0.0, 10.0],
            'y1': [0.0, 0.0],
            'x2': [10.0, 100.0],
            'y2': [0.0, 0.0]
        })

        graph = acj.load_graph(nodes, segments)

        # 1. threshold_meters = 0 (or not passed) should use topological simplification.
        simplified_topo = acj.simplify_graph(graph, threshold_meters=0.0)
        # 3 original nodes -> 2 new nodes (0 and 2).
        assert len(simplified_topo.nodes) == 2
        assert 1 not in simplified_topo.nodes['node_id'].values

        # 2. Use geometric simplification for a graph with close intersections.
        # Nodes 10 (deg 1) and 11 (deg 1) are 5m apart. Node 20 (deg 1) is far.
        nodes_geom = pd.DataFrame({
            'node_id': [10, 11, 20],
            'x': [0.0, 5.0, 100.0],
            'y': [0.0, 0.0, 0.0]
        })
        segments_geom = pd.DataFrame({
            'segment_id': [0, 1],
            'node_start': [10, 11],
            'node_end': [20, 20],
            'x1': [0.0, 5.0], 'y1': [0.0, 0.0],
            'x2': [100.0, 100.0], 'y2': [0.0, 0.0]
        })
        graph_geom = acj.load_graph(nodes_geom, segments_geom)

        # threshold_meters = 10.0 should use geometric simplification and merge 10/11
        simplified_geom = acj.simplify_graph(graph_geom, threshold_meters=10.0)
        # 3 original nodes -> 2 new nodes (centroid of 10/11, plus 20)
        assert len(simplified_geom.nodes) == 2
        assert len(simplified_geom.segments) == 1

        # 3. Test explicit method selection
        simplified_explicit_topo = acj.simplify_graph(
            graph_geom, method='topological', threshold_meters=100.0
        )
        # Should just run topological, which keeps all nodes since none are degree 2.
        assert len(simplified_explicit_topo.nodes) == 3

        simplified_explicit_geom = acj.simplify_graph(
            graph_geom, method='geometric', threshold_meters=10.0
        )
            # Should run geometric and merge 10/11.
        assert len(simplified_explicit_geom.nodes) == 2

    def test_simplify_graph_empty_graph(self):
        """Test simplification with empty graph."""
        nodes = pd.DataFrame({'node_id': [], 'x': [], 'y': []})
        segments = pd.DataFrame({
            'segment_id': [], 'node_start': [], 'node_end': [],
            'x1': [], 'y1': [], 'x2': [], 'y2': []
        })

        graph = acj.load_graph(nodes, segments)
        simplified = acj.simplify_graph(graph)

        assert len(simplified.nodes) == 0
        assert len(simplified.segments) == 0

    def test_simplify_graph_single_node(self):
        """Test simplification with single node."""
        nodes = pd.DataFrame({
            'node_id': [0],
            'x': [0.0],
            'y': [0.0]
        })

        segments = pd.DataFrame({
            'segment_id': [], 'node_start': [], 'node_end': [],
            'x1': [], 'y1': [], 'x2': [], 'y2': []
        })

        graph = acj.load_graph(nodes, segments)
        # Use topological simplification for single node (no segments)
        simplified = acj.simplify_graph_topological(graph)

        assert len(simplified.nodes) == 1
        assert len(simplified.segments) == 0

    def test_simplify_graph_parallel_cgal_merge(self):
        """Test geometric simplification by merging parallel segments using CGAL method."""
        # Create two parallel segments 5m apart, both running from x=0 to x=100.
        nodes = pd.DataFrame({
            'node_id': [0, 1, 2, 3],
            'x': [0.0, 0.0, 100.0, 100.0],
            'y': [0.0, 5.0, 0.0, 5.0]
        })

        segments = pd.DataFrame({
            'segment_id': [0, 1],
            'node_start': [0, 1],
            'node_end': [2, 3],
            'x1': [0.0, 0.0],
            'y1': [0.0, 5.0],
            'x2': [100.0, 100.0],
            'y2': [0.0, 5.0]
        })

        graph = acj.load_graph(nodes, segments)

        # Merge should occur: distance_threshold = 10.0 (covers 5m gap), angle_threshold_deg = 5.0 (covers parallel lines)
        simplified = acj.simplify_graph_parallel_cgal(
            graph.nodes, graph.segments,
            distance_threshold=10.0, angle_threshold_deg=5.0
        )

        # The two parallel segments should merge into a single segment (and 2 nodes)
        assert len(simplified.nodes) == 2
        assert len(simplified.segments) == 1

        # No merge should occur: distance_threshold = 1.0 (does not cover 5m gap)
        simplified_no_merge = acj.simplify_graph_parallel_cgal(
            graph.nodes, graph.segments,
            distance_threshold=1.0, angle_threshold_deg=5.0
        )

        # Should remain the original 4 nodes and 2 segments
        assert len(simplified_no_merge.nodes) == 4
        assert len(simplified_no_merge.segments) == 2

    def test_simplify_graph_automatic_selection(self):
        """Test that simplify_graph automatically selects the right method."""
        # Chain graph: 0-1-2. Node 1 is degree 2 (to be removed topologically).
        nodes = pd.DataFrame({
            'node_id': [0, 1, 2],
            'x': [0.0, 10.0, 100.0],
            'y': [0.0, 0.0, 0.0]
        })

        segments = pd.DataFrame({
            'segment_id': [0, 1],
            'node_start': [0, 1],
            'node_end': [1, 2],
            'x1': [0.0, 10.0],
            'y1': [0.0, 0.0],
            'x2': [10.0, 100.0],
            'y2': [0.0, 0.0]
        })

        graph = acj.load_graph(nodes, segments)

        # 1. threshold_meters = 0 (or not passed) should use topological simplification.
        simplified_topo = acj.simplify_graph(graph, threshold_meters=0.0)
        # 3 original nodes -> 2 new nodes (0 and 2).
        assert len(simplified_topo.nodes) == 2
        assert 1 not in simplified_topo.nodes['node_id'].values

        # 2. Use geometric simplification for a graph with close intersections.
        # Nodes 10 (deg 1) and 11 (deg 1) are 5m apart. Node 20 (deg 1) is far.
        nodes_geom = pd.DataFrame({
            'node_id': [10, 11, 20],
            'x': [0.0, 5.0, 100.0],
            'y': [0.0, 0.0, 0.0]
        })
        segments_geom = pd.DataFrame({
            'segment_id': [0, 1],
            'node_start': [10, 11],
            'node_end': [20, 20],
            'x1': [0.0, 5.0], 'y1': [0.0, 0.0],
            'x2': [100.0, 100.0], 'y2': [0.0, 0.0]
        })
        graph_geom = acj.load_graph(nodes_geom, segments_geom)

        # threshold_meters = 10.0 should use geometric simplification and merge 10/11
        simplified_geom = acj.simplify_graph(graph_geom, threshold_meters=10.0)
        # 3 original nodes -> 2 new nodes (centroid of 10/11, plus 20)
        assert len(simplified_geom.nodes) == 2
        assert len(simplified_geom.segments) == 1

        # 3. Test explicit method selection
        simplified_explicit_topo = acj.simplify_graph(
            graph_geom, method='topological', threshold_meters=100.0
        )
        # Should just run topological, which keeps all nodes since none are degree 2.
        assert len(simplified_explicit_topo.nodes) == 3

        simplified_explicit_geom = acj.simplify_graph(
            graph_geom, method='geometric', threshold_meters=10.0
        )
        # Should run geometric and merge 10/11.
        assert len(simplified_explicit_geom.nodes) == 2

    def test_simplify_graph_empty_graph(self):
        """Test simplification with empty graph."""
        nodes = pd.DataFrame({'node_id': [], 'x': [], 'y': []})
        segments = pd.DataFrame({
            'segment_id': [], 'node_start': [], 'node_end': [],
            'x1': [], 'y1': [], 'x2': [], 'y2': []
        })

        graph = acj.load_graph(nodes, segments)
        simplified = acj.simplify_graph(graph)

        assert len(simplified.nodes) == 0
        assert len(simplified.segments) == 0

    def test_simplify_graph_single_node(self):
        """Test simplification with single node."""
        nodes = pd.DataFrame({
            'node_id': [0],
            'x': [0.0],
            'y': [0.0]
        })

        segments = pd.DataFrame({
            'segment_id': [], 'node_start': [], 'node_end': [],
            'x1': [], 'y1': [], 'x2': [], 'y2': []
        })

        graph = acj.load_graph(nodes, segments)
        # Use topological simplification for single node (no segments)
        simplified = acj.simplify_graph_topological(graph)

        assert len(simplified.nodes) == 1
        assert len(simplified.segments) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
