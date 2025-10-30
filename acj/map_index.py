"""
Core MapIndex class for spatial queries.

This module provides the main interface for performing spatial queries
on graph data using CGAL-based spatial indexing structures.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict
from .io import GraphData


class MapIndex:
    """
    Spatial index for efficient point-to-graph assignment queries.
    
    This class wraps CGAL spatial data structures to provide fast 
    nearest-neighbor queries.
    """
    
    def __init__(self, graph_data: GraphData):
        """Initialize MapIndex with graph data."""
        self.graph_data = graph_data
        self._endpoint_index = None
        self._segment_index = None
        self._acj_core = None
        self._adjacency_list = None
        
        try:
            import sys
            import os
            build_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'build')
            if os.path.exists(build_path) and build_path not in sys.path:
                sys.path.insert(0, build_path)
            
            import acj_core
            self._acj_core = acj_core
        except ImportError as e:
            raise ImportError(
                "Failed to import acj_core module. "
                "Please ensure the C++ extension is compiled. "
                f"Original error: {e}"
            )

    def _build_adjacency_list(self):
        """Builds an adjacency list for the graph for neighbor lookups."""
        if self._adjacency_list is not None:
            return
        
        print("Building adjacency list for neighbor analysis...")
        adj = {node_id: [] for node_id in self.graph_data.nodes['node_id']}
        for seg in self.graph_data.segments.itertuples():
            adj[seg.node_start].append(seg.node_end)
            adj[seg.node_end].append(seg.node_start)
        self._adjacency_list = adj
    
    def _build_endpoint_index(self) -> None:
        """Build the CGAL Delaunay triangulation index for endpoints (nodes)."""
        if self._endpoint_index is not None:
            return
        
        node_coords = self.graph_data.nodes[['x', 'y']].values.astype(np.float64)
        self._endpoint_index = node_coords
    
    def _build_segment_index(self) -> None:
        """Build the CGAL Segment Delaunay graph index for line segments (edges)."""
        if self._segment_index is not None:
            return
        
        segment_coords = self.graph_data.segments[['x1', 'y1', 'x2', 'y2']].values.astype(np.float64)
        self._segment_index = segment_coords
    
    def assign_to_endpoints(self, points_df: pd.DataFrame) -> pd.DataFrame:
        """Assign points to their nearest graph endpoints (nodes)."""
        required = ['point_id', 'x', 'y']
        if any(col not in points_df.columns for col in required):
            raise ValueError(f"Points DataFrame missing required columns: {required}")
        
        self._build_endpoint_index()
        point_coords = points_df[['x', 'y']].values.astype(np.float64)
        indices, distances = self._acj_core.match_point(point_coords, self._endpoint_index)
        
        node_ids = self.graph_data.nodes['node_id'].values
        result = points_df.copy()
        result['assigned_node_id'] = node_ids[indices]
        result['distance'] = distances
        return result
    
    def assign_to_segments(self, segments_df: pd.DataFrame) -> pd.DataFrame:
        """Assign points to their nearest graph line segments (edges)."""
        required = ['point_id', 'x', 'y']
        if any(col not in segments_df.columns for col in required):
            raise ValueError(f"Points DataFrame missing required columns: {required}")

        self._build_segment_index()
        point_coords = segments_df[['x', 'y']].values.astype(np.float64)
        indices, distances = self._acj_core.match_segment(point_coords, self._segment_index)

        segment_ids = self.graph_data.segments['segment_id'].values
        result = segments_df.copy()
        result['assigned_segment_id'] = segment_ids[indices]
        result['distance'] = distances
        return result

    def get_render_data(self, assignments: pd.DataFrame = None, neighbor_influence: float = 0.5, outlier_percentile: float = 99.0) -> Dict[str, np.ndarray]:
        """
        Pre-compute all data needed for GPU-accelerated real-time rendering.
        
        Includes neighbor smoothing and robust normalization to handle outliers.
        
        Args:
            assignments: DataFrame with assignment results.
            neighbor_influence: Factor (0.0 to 1.0) of neighbor influence.
            outlier_percentile: Percentile (e.g., 98.0) to cap the color scale,
                                making the heatmap robust to extreme outliers.
        """
        node_vertices = self.graph_data.nodes[['x', 'y']].values.astype(np.float32)
        n_nodes = len(node_vertices)
        default_color = [0.4, 0.4, 0.4, 0.8]

        if assignments is not None:
            # --- LÓGICA DE SUAVIZADO Y NORMALIZACIÓN ROBUSTA ---
            original_crime_counts = assignments['assigned_node_id'].value_counts().to_dict()
            self._build_adjacency_list()

            smoothed_crime_counts = {}
            for node_id in self.graph_data.nodes['node_id']:
                original_count = original_crime_counts.get(node_id, 0)
                neighbor_influence_sum = sum(original_crime_counts.get(n_id, 0) for n_id in self._adjacency_list[node_id])
                smoothed_count = original_count + (neighbor_influence * neighbor_influence_sum)
                smoothed_crime_counts[node_id] = smoothed_count
            
            # --- NUEVA LÓGICA DE NORMALIZACIÓN PARA DATOS ATÍPICOS ---
            all_counts = list(smoothed_crime_counts.values())
            if not all_counts:
                max_for_scaling = 1
            else:
                # Usar un percentil para definir el "máximo razonable"
                # Esto evita que un solo outlier domine toda la escala de colores.
                max_for_scaling = np.percentile(all_counts, outlier_percentile)
                if max_for_scaling == 0:
                    max_for_scaling = 1 # Evitar división por cero
            
            node_colors = np.full((n_nodes, 4), default_color, dtype=np.float32)
            node_id_to_idx = {node_id: i for i, node_id in enumerate(self.graph_data.nodes['node_id'])}
            
            for node_id, smoothed_count in smoothed_crime_counts.items():
                if smoothed_count > 0 and node_id in node_id_to_idx:
                    idx = node_id_to_idx[node_id]
                    # La intensidad se calcula contra el máximo del percentil.
                    # Se usa min() para que los outliers > percentil no pasen de 1.0.
                    intensity = min(smoothed_count, max_for_scaling) / max_for_scaling
                    
                    # Gradiente: gris -> amarillo -> naranja -> rojo
                    if intensity < 0.33:
                        t = intensity / 0.33
                        r, g, b = (0.4 + 0.6 * t, 0.4 + 0.6 * t, 0.4 - 0.4 * t)
                    elif intensity < 0.66:
                        t = (intensity - 0.33) / 0.33
                        r, g, b = (1.0, 1.0 - 0.35 * t, 0.0)
                    else:
                        t = (intensity - 0.66) / 0.34
                        r, g, b = (1.0, 0.65 - 0.65 * t, 0.0)
                    
                    node_colors[idx] = [r, g, b, 1.0]
        else:
            node_colors = np.full((n_nodes, 4), default_color, dtype=np.float32)

        # Preparación de datos de segmentos (sin cambios)
        n_segments = len(self.graph_data.segments)
        segment_vertices = np.zeros((n_segments * 2, 2), dtype=np.float32)
        segment_colors = np.zeros((n_segments * 2, 4), dtype=np.float32)
        segment_connectivity = np.zeros((n_segments, 2), dtype=np.int32)
        node_id_to_idx = {node_id: i for i, node_id in enumerate(self.graph_data.nodes['node_id'])}

        for i, seg in enumerate(self.graph_data.segments.itertuples()):
            start_idx, end_idx = i * 2, i * 2 + 1
            segment_vertices[start_idx] = [seg.x1, seg.y1]
            segment_vertices[end_idx] = [seg.x2, seg.y2]
            
            node_start_idx = node_id_to_idx.get(seg.node_start)
            node_end_idx = node_id_to_idx.get(seg.node_end)
            
            if node_start_idx is not None and node_end_idx is not None:
                segment_colors[start_idx] = node_colors[node_start_idx]
                segment_colors[end_idx] = node_colors[node_end_idx]
            
            segment_connectivity[i] = [start_idx, end_idx]
        
        return {
            'node_vertices': node_vertices,
            'node_colors': node_colors,
            'segment_vertices': segment_vertices,
            'segment_colors': segment_colors,
            'segment_connectivity': segment_connectivity
        }
    
    def __repr__(self) -> str:
        return f"MapIndex(nodes={len(self.graph_data.nodes)}, segments={len(self.graph_data.segments)})"