"""
ACJ - Advanced Geospatial Analysis Library

A Python library for efficient geospatial point-to-graph assignment using CGAL.
Designed for large-scale urban analytics, crime mapping, and network analysis.

Main Components:
    - MapIndex: Core class for spatial indexing and queries
    - load_graph: Load graph data from pandas DataFrames
    - simplify_graph: Graph simplification utilities
    
Author: Cèsar, Alejandro, Jeremy.
"""

from .map_index import MapIndex
from .io import load_graph, load_map
from .graph import simplify_graph
from .render import render_graph, render_heatmap

__version__ = "0.1.0"
__all__ = [
    "MapIndex",
    "load_graph",
    "load_map",
    "simplify_graph",
    "render_graph",
    "render_heatmap",
]
