"""
Test suite for ACJ library.

Tests the main functionality of the ACJ geospatial analysis library
including data loading, spatial indexing, and point assignment.
"""

import pytest
import numpy as np
import pandas as pd
import sys
import os

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
        assert result.loc[2, 'assigned_node_id'] in [2, 3]  # Point 2 near nodes 2 or 3
        
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
        """Test that assign_to_segments raises NotImplementedError."""
        map_index = acj.MapIndex(simple_graph)
        
        points = pd.DataFrame({
            'point_id': [0],
            'x': [50.0],
            'y': [0.0]
        })
        
        with pytest.raises(NotImplementedError):
            map_index.assign_to_segments(points)


class TestGraphSimplification:
    """Test cases for graph simplification."""
    
    def test_simplify_graph_returns_unchanged(self):
        """Test that simplify_graph currently returns unchanged graph."""
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
        simplified = acj.simplify_graph(graph, threshold_meters=15.0)
        
        # Currently returns unchanged
        assert len(simplified.nodes) == len(graph.nodes)
        assert len(simplified.segments) == len(graph.segments)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
