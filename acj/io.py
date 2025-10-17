"""
Input/Output module for loading graph data.

This module handles loading graph data from various sources including
OpenStreetMap (via OSMnx) and pandas DataFrames.

Data Format Standards:
    Nodes DataFrame:
        - Required columns: ['node_id', 'x', 'y']
        - node_id: Unique integer identifier
        - x, y: Projected coordinates in meters (e.g., UTM)
    
    Segments DataFrame:
        - Required columns: ['segment_id', 'node_start', 'node_end', 'x1', 'y1', 'x2', 'y2']
        - segment_id: Unique integer identifier
        - node_start, node_end: References to node_id
        - x1, y1, x2, y2: Endpoint coordinates (must match node coordinates)
"""

import pandas as pd
import os
import pickle
from pathlib import Path


class GraphData:
    """
    Container for graph data (nodes and segments).
    
    Attributes:
        nodes: DataFrame with columns ['node_id', 'x', 'y']
        segments: DataFrame with columns ['segment_id', 'node_start', 'node_end', 'x1', 'y1', 'x2', 'y2']
    """
    
    def __init__(self, nodes: pd.DataFrame, segments: pd.DataFrame):
        """
        Initialize GraphData container.
        
        Args:
            nodes: DataFrame with node information
            segments: DataFrame with segment information
            
        Raises:
            ValueError: If required columns are missing
        """
        self._validate_nodes(nodes)
        self._validate_segments(segments)
        
        self.nodes = nodes.copy()
        self.segments = segments.copy()
    
    def _validate_nodes(self, nodes: pd.DataFrame) -> None:
        """Validate that nodes DataFrame has required columns."""
        required = ['node_id', 'x', 'y']
        missing = [col for col in required if col not in nodes.columns]
        if missing:
            raise ValueError(f"Nodes DataFrame missing required columns: {missing}")
    
    def _validate_segments(self, segments: pd.DataFrame) -> None:
        """Validate that segments DataFrame has required columns."""
        required = ['segment_id', 'node_start', 'node_end', 'x1', 'y1', 'x2', 'y2']
        missing = [col for col in required if col not in segments.columns]
        if missing:
            raise ValueError(f"Segments DataFrame missing required columns: {missing}")
    
    def __repr__(self) -> str:
        return f"GraphData(nodes={len(self.nodes)}, segments={len(self.segments)})"


def load_graph(nodes_df: pd.DataFrame, segments_df: pd.DataFrame) -> GraphData:
    """
    Load graph data from pandas DataFrames.
    
    This is the main entry point for loading custom graph data. DataFrames
    must follow the standard format specified in the module docstring.
    
    Args:
        nodes_df: DataFrame with node data
            Required columns: ['node_id', 'x', 'y']
        segments_df: DataFrame with segment data
            Required columns: ['segment_id', 'node_start', 'node_end', 'x1', 'y1', 'x2', 'y2']
    
    Returns:
        GraphData object containing validated graph data
    
    Raises:
        ValueError: If DataFrames don't meet format requirements
    
    Example:
        >>> nodes = pd.DataFrame({
        ...     'node_id': [0, 1, 2],
        ...     'x': [0.0, 100.0, 200.0],
        ...     'y': [0.0, 0.0, 100.0]
        ... })
        >>> segments = pd.DataFrame({
        ...     'segment_id': [0, 1],
        ...     'node_start': [0, 1],
        ...     'node_end': [1, 2],
        ...     'x1': [0.0, 100.0],
        ...     'y1': [0.0, 0.0],
        ...     'x2': [100.0, 200.0],
        ...     'y2': [0.0, 100.0]
        ... })
        >>> graph = acj.load_graph(nodes, segments)
    """
    return GraphData(nodes_df, segments_df)


def load_map(city_name: str, cache_dir: str = "./cache", network_type: str = "drive") -> GraphData:
    """
    Load map data from OpenStreetMap using OSMnx.
    
    This function downloads street network data for a specified city,
    converts it to the standard GraphData format, and caches the results
    for future use.
    
    Args:
        city_name: Name of the city (e.g., "Manhattan, New York City")
        cache_dir: Directory to cache downloaded data
        network_type: Type of street network ('drive', 'walk', 'bike', 'all')
    
    Returns:
        GraphData object with the street network
    
    Raises:
        ImportError: If OSMnx is not installed
        ValueError: If city cannot be found
    
    Example:
        >>> graph = acj.load_map("Cholula, Puebla, Mexico")
        >>> print(f"Loaded {len(graph.nodes)} nodes and {len(graph.segments)} segments")
    """
    try:
        import osmnx as ox
        import geopandas as gpd
    except ImportError:
        raise ImportError(
            "OSMnx and GeoPandas are required for load_map(). "
            "Install with: pip install osmnx geopandas"
        )
    
    # Create cache directory if it doesn't exist
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    
    # Generate cache filename from city name and network type
    cache_filename = f"{city_name.replace(' ', '_').replace(',', '')}_{network_type}.pkl"
    cache_file = cache_path / cache_filename
    
    # Check if cached data exists
    if cache_file.exists():
        print(f"Loading cached data from {cache_file}")
        with open(cache_file, 'rb') as f:
            cached_data = pickle.load(f)
        return GraphData(cached_data['nodes'], cached_data['segments'])
    
    print(f"Downloading street network for '{city_name}' from OpenStreetMap...")
    
    # Download graph from OSMnx
    try:
        G = ox.graph_from_place(city_name, network_type=network_type, simplify=True)
    except Exception as e:
        raise ValueError(f"Could not download map for '{city_name}': {e}")
    
    print(f"Downloaded graph with {len(G.nodes)} nodes and {len(G.edges)} edges")
    
    # Project to UTM for meter-based coordinates
    G_proj = ox.project_graph(G)
    
    # Extract nodes
    nodes_data = []
    node_id_map = {}  # Map OSM node IDs to sequential integers
    
    for idx, (osm_id, data) in enumerate(G_proj.nodes(data=True)):
        node_id_map[osm_id] = idx
        nodes_data.append({
            'node_id': idx,
            'x': data['x'],
            'y': data['y']
        })
    
    nodes_df = pd.DataFrame(nodes_data)
    
    # Extract segments (edges)
    segments_data = []
    segment_id = 0
    
    for u, v, data in G_proj.edges(data=True):
        # Get node coordinates
        u_node = G_proj.nodes[u]
        v_node = G_proj.nodes[v]
        
        segments_data.append({
            'segment_id': segment_id,
            'node_start': node_id_map[u],
            'node_end': node_id_map[v],
            'x1': u_node['x'],
            'y1': u_node['y'],
            'x2': v_node['x'],
            'y2': v_node['y']
        })
        segment_id += 1
    
    segments_df = pd.DataFrame(segments_data)
    
    print(f"Converted to ACJ format: {len(nodes_df)} nodes, {len(segments_df)} segments")
    
    # Cache the data
    print(f"Caching data to {cache_file}")
    with open(cache_file, 'wb') as f:
        pickle.dump({'nodes': nodes_df, 'segments': segments_df}, f)
    
    return GraphData(nodes_df, segments_df)
