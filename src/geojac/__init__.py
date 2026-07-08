"""
ACJ - Advanced Geospatial Analysis Library

A Python library for efficient geospatial point-to-graph assignment using CGAL.
Designed for large-scale urban analytics, crime mapping, and network analysis.

Main Components:
    - MapIndex: Core class for spatial indexing and queries
    - load_graph: Load graph data from pandas DataFrames
    - simplify_graph: Graph simplification utilities
    
Author: Cèsar, Alejandro, Jerimy.
"""

# Importaciones con las nuevas rutas de Clean Architecture
from .core.network import UrbanNetwork
from .core.semantics import resolve_semantics
from .algorithms.map_index import MapIndex
from .data.io import load_graph, load_map, GraphData, SimplificationResult
from .algorithms.graph import (simplify_graph, simplify_graph_topological,
                               simplify_graph_geometric)
from .utils.render import render_graph, render_heatmap, render_comparison
from .evaluation.base import BaseEvaluator
from .evaluation.evaluators import ACJTopologicalEvaluator
from .evaluation.metrics import (
    CompressionRatioMetric,
    SemanticSpeedDistortionMetric,
    KeypointDisplacementMetric,
    ReachabilityPreservationMetric,
    PathErrorMedianMetric,
    PathErrorP95Metric,
    PathRatioMedianMetric,
    PathRatioP95Metric,
    NodeCountMetric,
    EdgeCountMetric,
    TotalCoordinatesMetric,
    TotalLengthKmMetric,
    AvgDegreeMetric,
    AvgSinuosityMetric,
    ConnectivityIndexMetric,
)
from .evaluation.reporting import ThesisReportGenerator

__version__ = "0.1.0"
__all__ = [
    "UrbanNetwork",
    "resolve_semantics",
    "SimplificationResult",
    "MapIndex",
    "load_graph",
    "load_map",
    "GraphData",
    "simplify_graph",
    "simplify_graph_topological",
    "simplify_graph_geometric",
    "render_graph",
    "render_heatmap",
    "render_comparison",
    "BaseEvaluator",
    "ACJTopologicalEvaluator",
    "CompressionRatioMetric",
    "SemanticSpeedDistortionMetric",
    "KeypointDisplacementMetric",
    "ReachabilityPreservationMetric",
    "PathErrorMedianMetric",
    "PathErrorP95Metric",
    "PathRatioMedianMetric",
    "PathRatioP95Metric",
    "NodeCountMetric",
    "EdgeCountMetric",
    "TotalCoordinatesMetric",
    "TotalLengthKmMetric",
    "AvgDegreeMetric",
    "AvgSinuosityMetric",
    "ConnectivityIndexMetric",
    "ThesisReportGenerator",
]
