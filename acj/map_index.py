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
    
    This class wraps CGAL spatial data structures (Delaunay triangulation
    and AABB trees) to provide fast nearest-neighbor queries for assigning
    points to graph elements (nodes and segments).
    """
    
    def __init__(self, graph_data: GraphData):
        """
        Initialize MapIndex with graph data.
        """
        self.graph_data = graph_data
        self._endpoint_index = None
        self._segment_index = None
        self._acj_core = None
        
        # Try to import the compiled C++ module
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
    
    def _build_endpoint_index(self) -> None:
        """Build the CGAL Delaunay triangulation index for endpoints (nodes)."""
        if self._endpoint_index is not None:
            return
        
        node_coords = self.graph_data.nodes[['x', 'y']].values.astype(np.float64)
        self._endpoint_index = node_coords
    
    def _build_segment_index(self) -> None:
        """Build the CGAL AABB tree index for segments."""
        raise NotImplementedError(
            "Segment index building is not yet implemented."
        )
    
    def assign_to_endpoints(self, points_df: pd.DataFrame) -> pd.DataFrame:
        """Assign points to their nearest graph endpoints (nodes)."""
        required = ['point_id', 'x', 'y']
        missing = [col for col in required if col not in points_df.columns]
        if missing:
            raise ValueError(f"Points DataFrame missing required columns: {missing}")
        
        self._build_endpoint_index()
        
        point_coords = points_df[['x', 'y']].values.astype(np.float64)
        
        indices, distances = self._acj_core.match_cgal(
            point_coords,
            self._endpoint_index
        )
        
        node_ids = self.graph_data.nodes['node_id'].values
        assigned_node_ids = node_ids[indices]
        
        result = points_df.copy()
        result['assigned_node_id'] = assigned_node_ids
        result['distance'] = distances
        
        return result
    
    def assign_to_segments(self, points_df: pd.DataFrame) -> pd.DataFrame:
        """Assign points to their nearest graph segments (edges)."""
        raise NotImplementedError(
            "assign_to_segments() is not yet implemented."
        )
    
    def get_render_data(self, assignments: pd.DataFrame = None) -> Dict[str, np.ndarray]:
        """Pre-compute all data needed for GPU-accelerated real-time rendering."""
        node_vertices = self.graph_data.nodes[['x', 'y']].values.astype(np.float32)
        n_nodes = len(node_vertices)
        
        # --- FIX 1: CAMBIO DE COLOR BASE DE BLANCO A GRIS ---
        # El color por defecto para nodos sin crímenes ahora es un gris suave.
        default_color = [0.4, 0.4, 0.4, 0.8] # RGBA
        
        if assignments is not None:
            crime_counts = assignments['assigned_node_id'].value_counts()
            node_colors = np.full((n_nodes, 4), default_color, dtype=np.float32)
            
            max_crimes = crime_counts.max() if len(crime_counts) > 0 else 1
            
            # --- FIX 2: BUG DE INDEXACIÓN CORREGIDO ---
            # Mapear node_id a su índice de array (0, 1, 2...) para un acceso rápido y correcto.
            node_id_to_idx = {node_id: i for i, node_id in enumerate(self.graph_data.nodes['node_id'])}
            
            for node_id, count in crime_counts.items():
                if count > 0 and node_id in node_id_to_idx:
                    idx = node_id_to_idx[node_id] # Usar el índice correcto
                    intensity = count / max_crimes
                    
                    # Gradiente: gris -> amarillo -> naranja -> rojo
                    if intensity < 0.33:
                        # Gris (0.4, 0.4, 0.4) a Amarillo (1, 1, 0)
                        t = intensity / 0.33
                        r = 0.4 + 0.6 * t
                        g = 0.4 + 0.6 * t
                        b = 0.4 - 0.4 * t
                    elif intensity < 0.66:
                        # Amarillo (1, 1, 0) a Naranja (1, 0.65, 0)
                        t = (intensity - 0.33) / 0.33
                        r = 1.0
                        g = 1.0 - 0.35 * t
                        b = 0.0
                    else:
                        # Naranja (1, 0.65, 0) a Rojo (1, 0, 0)
                        t = (intensity - 0.66) / 0.34
                        r = 1.0
                        g = 0.65 - 0.65 * t
                        b = 0.0
                    
                    node_colors[idx] = [r, g, b, 1.0]
        else:
            node_colors = np.full((n_nodes, 4), default_color, dtype=np.float32)
        
        n_segments = len(self.graph_data.segments)
        segment_vertices = np.zeros((n_segments * 2, 2), dtype=np.float32)
        segment_colors = np.zeros((n_segments * 2, 4), dtype=np.float32)
        segment_connectivity = np.zeros((n_segments, 2), dtype=np.int32)
        
        # Usar el mismo mapeo para asegurar la consistencia
        node_id_to_idx = {node_id: i for i, node_id in enumerate(self.graph_data.nodes['node_id'])}

        for i, seg in enumerate(self.graph_data.segments.itertuples()):
            start_idx = i * 2
            end_idx = i * 2 + 1
            
            segment_vertices[start_idx] = [seg.x1, seg.y1]
            segment_vertices[end_idx] = [seg.x2, seg.y2]
            
            # Obtener los colores de los nodos por su índice correcto
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
        return (
            f"MapIndex(nodes={len(self.graph_data.nodes)}, "
            f"segments={len(self.graph_data.segments)})"
        )