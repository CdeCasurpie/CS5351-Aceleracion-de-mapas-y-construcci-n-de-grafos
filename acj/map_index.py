"""
Core MapIndex class for spatial queries.

This module provides the main interface for performing spatial queries
on graph data using CGAL-based spatial indexing structures.
"""

import pandas as pd
import numpy as np
from typing import Tuple
from .io import GraphData


class MapIndex:
    """
    Spatial index for efficient point-to-graph assignment queries.
    
    This class wraps CGAL spatial data structures (Delaunay triangulation
    and AABB trees) to provide fast nearest-neighbor queries for assigning
    points to graph elements (nodes and segments).
    
    Attributes:
        graph_data: The underlying GraphData object
        _endpoint_index: Internal CGAL index for endpoint queries (built lazily)
        _segment_index: Internal CGAL index for segment queries (built lazily)
    
    Example:
        >>> graph = acj.load_graph(nodes_df, segments_df)
        >>> map_index = acj.MapIndex(graph)
        >>> assignments = map_index.assign_to_endpoints(crimes_df)
    """
    
    def __init__(self, graph_data: GraphData):
        """
        Initialize MapIndex with graph data.
        
        Args:
            graph_data: GraphData object containing nodes and segments
        
        Notes:
            The spatial indices are built lazily on first query to save
            initialization time if only one query type is needed.
        """
        self.graph_data = graph_data
        self._endpoint_index = None
        self._segment_index = None
        self._acj_core = None
        
        # Try to import the compiled C++ module
        try:
            import sys
            import os
            # Add build directory to path
            build_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'build')
            if os.path.exists(build_path) and build_path not in sys.path:
                sys.path.insert(0, build_path)
            
            import acj_core
            self._acj_core = acj_core
        except ImportError as e:
            raise ImportError(
                "Failed to import acj_core module. "
                "Please ensure the C++ extension is compiled. "
                "Run 'make build' from the project root directory. "
                f"Original error: {e}"
            )
    
    def _build_endpoint_index(self) -> None:
        """
        Build the CGAL Delaunay triangulation index for endpoints (nodes).
        
        This is called lazily on the first call to assign_to_endpoints().
        Constructs a spatial index over all graph nodes for O(log N) queries.
        """
        if self._endpoint_index is not None:
            return
        
        # Extract node coordinates as Nx2 numpy array
        node_coords = self.graph_data.nodes[['x', 'y']].values.astype(np.float64)
        
        # Build the index using Alejandro's CGAL implementation
        # The index is stored internally by the C++ module
        # We just store a reference/flag that it's been built
        self._endpoint_index = node_coords
    
    def _build_segment_index(self) -> None:
        """
        Build the CGAL AABB tree index for segments.
        
        PENDING IMPLEMENTATION: This requires extending the C++ code
        to support AABB tree construction from line segments.
        
        This will be called lazily on the first call to assign_to_segments().
        """
        if self._segment_index is not None:
            return
        
        # PENDING IMPLEMENTATION
        # Extract segment coordinates as Nx4 numpy array (x1, y1, x2, y2)
        # segment_coords = self.graph_data.segments[['x1', 'y1', 'x2', 'y2']].values
        
        # Build AABB tree index (requires new C++ function)
        # self._segment_index = self._acj_core.build_segment_index(segment_coords)
        
        raise NotImplementedError(
            "Segment index building is not yet implemented. "
            "This requires extending the C++ core with AABB tree support."
        )
    
    def assign_to_endpoints(self, points_df: pd.DataFrame) -> pd.DataFrame:
        """
        Assign points to their nearest graph endpoints (nodes).
        
        For each point in points_df, finds the closest node in the graph
        using CGAL's Delaunay triangulation for O(log N) query performance.
        
        Args:
            points_df: DataFrame with point data
                Required columns: ['point_id', 'x', 'y']
                Optional columns: any additional data to preserve
        
        Returns:
            DataFrame with assignment results:
                - point_id: Original point identifier
                - assigned_node_id: ID of nearest node
                - distance: Euclidean distance to assigned node
                - (additional columns from input are preserved)
        
        Example:
            >>> crimes = pd.DataFrame({
            ...     'point_id': [0, 1, 2],
            ...     'x': [50.0, 150.0, 25.0],
            ...     'y': [5.0, 10.0, 90.0]
            ... })
            >>> assignments = map_index.assign_to_endpoints(crimes)
            >>> print(assignments[['point_id', 'assigned_node_id', 'distance']])
        
        Raises:
            ValueError: If required columns are missing from points_df
        """
        # Validate input
        required = ['point_id', 'x', 'y']
        missing = [col for col in required if col not in points_df.columns]
        if missing:
            raise ValueError(f"Points DataFrame missing required columns: {missing}")
        
        # Build endpoint index if not already built
        self._build_endpoint_index()
        
        # Extract point coordinates
        point_coords = points_df[['x', 'y']].values.astype(np.float64)
        
        # Call Alejandro's match_cgal function
        # This returns (indices, distances) where indices[i] is the index
        # into self._endpoint_index of the nearest node for point i
        indices, distances = self._acj_core.match_cgal(
            point_coords,  # Query points
            self._endpoint_index  # Target nodes
        )
        
        # Map indices back to node_ids
        node_ids = self.graph_data.nodes['node_id'].values
        assigned_node_ids = node_ids[indices]
        
        # Build result DataFrame
        result = points_df.copy()
        result['assigned_node_id'] = assigned_node_ids
        result['distance'] = distances
        
        return result
    
    def assign_to_segments(self, points_df: pd.DataFrame) -> pd.DataFrame:
        """
        Assign points to their nearest graph segments (edges).
        
        For each point in points_df, finds the closest segment in the graph.
        This uses point-to-line-segment distance computation.
        
        PENDING IMPLEMENTATION: Requires C++ extension with AABB tree support.
        
        Args:
            points_df: DataFrame with point data
                Required columns: ['point_id', 'x', 'y']
        
        Returns:
            DataFrame with assignment results:
                - point_id: Original point identifier
                - assigned_segment_id: ID of nearest segment
                - distance: Distance to assigned segment
                - projection_x, projection_y: Point on segment closest to input point
        
        Raises:
            NotImplementedError: This function is pending implementation
        
        Example:
            >>> # assignments = map_index.assign_to_segments(crimes)  # PENDING
        """
        # Validate input
        required = ['point_id', 'x', 'y']
        missing = [col for col in required if col not in points_df.columns]
        if missing:
            raise ValueError(f"Points DataFrame missing required columns: {missing}")
        
        # PENDING IMPLEMENTATION
        raise NotImplementedError(
            "assign_to_segments() is not yet implemented. "
            "This requires extending the C++ core with AABB tree and "
            "point-to-segment distance computation."
        )
        
        # TODO: Implementation steps:
        # 1. Build segment index if not already built
        #    self._build_segment_index()
        #
        # 2. Extract point coordinates
        #    point_coords = points_df[['x', 'y']].values.astype(np.float64)
        #
        # 3. Call C++ function for segment assignment
        #    indices, distances, projections = self._acj_core.assign_to_segments(
        #        point_coords,
        #        self._segment_index
        #    )
        #
        # 4. Map indices to segment_ids and build result DataFrame
        #    segment_ids = self.graph_data.segments['segment_id'].values
        #    result = points_df.copy()
        #    result['assigned_segment_id'] = segment_ids[indices]
        #    result['distance'] = distances
        #    result['projection_x'] = projections[:, 0]
        #    result['projection_y'] = projections[:, 1]
        #    return result
    
    def __repr__(self) -> str:
        return (
            f"MapIndex(nodes={len(self.graph_data.nodes)}, "
            f"segments={len(self.graph_data.segments)})"
        )
