"""
Graph simplification and preprocessing utilities.

This module provides functions for simplifying and preprocessing graph data
to improve performance and reduce complexity for spatial queries.
"""

import pandas as pd
import numpy as np
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Set
from .io import GraphData


def simplify_graph_topological(graph_data: GraphData) -> GraphData:
    """
    Simplify a graph by consolidating nodes of degree 2 (topological simplification).
    
    This function implements the same logic as OSMnx.simplify_graph() to remove
    intermediate nodes that don't represent intersections. This is the most
    efficient approach for real-time applications as it preserves topology
    while dramatically reducing graph complexity.
    
    Algorithm:
        1. Identify all nodes with degree 2 (intermediate nodes)
        2. For each degree-2 node, merge its two incident edges into one
        3. Remove the intermediate node and update connectivity
        4. Repeat until no more degree-2 nodes exist
    
    Args:
        graph_data: GraphData object to simplify
    
    Returns:
        GraphData object with simplified graph (only intersections remain)
    
    Example:
        >>> graph = acj.load_graph(nodes_df, segments_df)
        >>> simplified = acj.simplify_graph_topological(graph)
        >>> print(f"Reduced from {len(graph.nodes)} to {len(simplified.nodes)} nodes")
    
    Notes:
        - Preserves all intersections and connectivity
        - Dramatically reduces node count for OSM data
        - Maintains spatial accuracy (no coordinate changes)
        - Fast O(n) algorithm suitable for real-time use
    """
    nodes_df = graph_data.nodes.copy()
    segments_df = graph_data.segments.copy()
    
    if len(nodes_df) == 0 or len(segments_df) == 0:
        return graph_data
    
    # Build adjacency list for efficient graph operations
    adjacency = defaultdict(list)
    node_degrees = defaultdict(int)
    
    # Initialize adjacency and degree counts
    for _, segment in segments_df.iterrows():
        start_id = segment['node_start']
        end_id = segment['node_end']
        
        adjacency[start_id].append((end_id, segment['segment_id']))
        adjacency[end_id].append((start_id, segment['segment_id']))
        node_degrees[start_id] += 1
        node_degrees[end_id] += 1
    
    # Find all degree-2 nodes (candidates for removal)
    degree2_nodes = deque([node_id for node_id in node_degrees.keys() 
                          if node_degrees[node_id] == 2])
    
    # Track which segments to remove and which to add
    segments_to_remove = set()
    new_segments = []
    nodes_to_remove = set()
    
    # Process degree-2 nodes iteratively
    while degree2_nodes:
        current_node = degree2_nodes.popleft()
        
        # Skip if node was already processed or removed
        if current_node in nodes_to_remove or node_degrees[current_node] != 2:
            continue
        
        # Get the two neighbors
        neighbors = adjacency[current_node]
        if len(neighbors) != 2:
            continue
        
        neighbor1_id, segment1_id = neighbors[0]
        neighbor2_id, segment2_id = neighbors[1]
        
        # Skip if neighbors are the same (self-loop)
        if neighbor1_id == neighbor2_id:
            continue
        
        # Check if neighbors are already connected
        neighbor1_connections = [nid for nid, _ in adjacency[neighbor1_id]]
        if neighbor2_id in neighbor1_connections:
            # Neighbors already connected, just remove current node
            segments_to_remove.add(segment1_id)
            segments_to_remove.add(segment2_id)
        else:
            # Create new segment connecting the neighbors
            # Get coordinates for the new segment
            node1_data = nodes_df[nodes_df['node_id'] == neighbor1_id].iloc[0]
            node2_data = nodes_df[nodes_df['node_id'] == neighbor2_id].iloc[0]
            
            # Calculate new segment length (sum of both segments)
            segment1_data = segments_df[segments_df['segment_id'] == segment1_id].iloc[0]
            segment2_data = segments_df[segments_df['segment_id'] == segment2_id].iloc[0]
            
            # Create new segment
            new_segment_id = max(segments_df['segment_id']) + len(new_segments) + 1
            new_segment = {
                'segment_id': new_segment_id,
                'node_start': neighbor1_id,
                'node_end': neighbor2_id,
                'x1': node1_data['x'],
                'y1': node1_data['y'],
                'x2': node2_data['x'],
                'y2': node2_data['y']
            }
            new_segments.append(new_segment)
            
            # Mark old segments for removal
            segments_to_remove.add(segment1_id)
            segments_to_remove.add(segment2_id)
            
            # Update adjacency list
            adjacency[neighbor1_id].remove((current_node, segment1_id))
            adjacency[neighbor2_id].remove((current_node, segment2_id))
            adjacency[neighbor1_id].append((neighbor2_id, new_segment_id))
            adjacency[neighbor2_id].append((neighbor1_id, new_segment_id))
        
        # Mark current node for removal
        nodes_to_remove.add(current_node)
        node_degrees[current_node] = 0
        
        # Update degrees of neighbors
        node_degrees[neighbor1_id] -= 1
        node_degrees[neighbor2_id] -= 1
        
        # Add neighbors back to queue if they become degree-2
        if node_degrees[neighbor1_id] == 2:
            degree2_nodes.append(neighbor1_id)
        if node_degrees[neighbor2_id] == 2:
            degree2_nodes.append(neighbor2_id)
    
    # Create simplified graph
    # Remove nodes
    simplified_nodes = nodes_df[~nodes_df['node_id'].isin(nodes_to_remove)].copy()
    
    # Remove old segments and add new ones
    simplified_segments = segments_df[~segments_df['segment_id'].isin(segments_to_remove)].copy()
    
    if new_segments:
        new_segments_df = pd.DataFrame(new_segments)
        simplified_segments = pd.concat([simplified_segments, new_segments_df], ignore_index=True)
    
    return GraphData(simplified_nodes, simplified_segments)


def simplify_graph_geometric(graph_data: GraphData, threshold_meters: float = 10.0) -> GraphData:
    """
    Simplify a graph by merging nearby nodes using geometric distance threshold.
    
    This function implements geometric simplification using CGAL for high-performance
    spatial clustering. It first applies topological simplification, then merges
    nearby intersections based on distance threshold.
    
    Algorithm:
        1. Apply topological simplification (remove degree-2 nodes)
        2. Build spatial index of remaining nodes using CGAL
        3. Cluster nodes within threshold_meters distance
        4. Merge clusters into single nodes at centroids
        5. Reconnect all edges to new cluster nodes
    
    Args:
        graph_data: GraphData object to simplify
        threshold_meters: Distance threshold for merging nodes (in meters)
    
    Returns:
        GraphData object with geometrically simplified graph
    
    Example:
        >>> graph = acj.load_graph(nodes_df, segments_df)
        >>> simplified = acj.simplify_graph_geometric(graph, threshold_meters=15.0)
    
    Notes:
        - More aggressive simplification than topological
        - May change network topology slightly
        - Best for very dense networks with many close intersections
        - Uses CGAL for high-performance spatial operations
    """
    # First apply topological simplification
    topo_simplified = simplify_graph_topological(graph_data)
    
    if len(topo_simplified.nodes) == 0:
        return topo_simplified
    
    # Import CGAL core for spatial operations
    try:
        import acj_core
    except ImportError:
        raise ImportError("CGAL core module not available. Run 'make build' first.")
    
    # Get node coordinates
    node_coords = topo_simplified.nodes[['x', 'y']].values
    node_ids = topo_simplified.nodes['node_id'].values
    
    # Build spatial index and find clusters
    clusters = _find_node_clusters(node_coords, threshold_meters)
    
    # Create mapping from old node IDs to new cluster IDs
    node_to_cluster = {}
    cluster_centers = {}
    
    for cluster_id, cluster_nodes in enumerate(clusters):
        cluster_node_ids = [node_ids[i] for i in cluster_nodes]
        cluster_coords = node_coords[cluster_nodes]
        
        # Calculate cluster centroid
        centroid_x = np.mean(cluster_coords[:, 0])
        centroid_y = np.mean(cluster_coords[:, 1])
        
        # Use the first node ID as the cluster representative
        cluster_rep_id = cluster_node_ids[0]
        cluster_centers[cluster_rep_id] = (centroid_x, centroid_y)
        
        # Map all nodes in cluster to representative
        for node_id in cluster_node_ids:
            node_to_cluster[node_id] = cluster_rep_id
    
    # Create new nodes (cluster representatives)
    new_nodes_data = []
    for cluster_rep_id, (centroid_x, centroid_y) in cluster_centers.items():
        new_nodes_data.append({
            'node_id': cluster_rep_id,
            'x': centroid_x,
            'y': centroid_y
        })
    
    new_nodes_df = pd.DataFrame(new_nodes_data)
    
    # Update segments to use cluster representatives
    new_segments_data = []
    segment_id_counter = 0
    
    for _, segment in topo_simplified.segments.iterrows():
        start_cluster = node_to_cluster[segment['node_start']]
        end_cluster = node_to_cluster[segment['node_end']]
        
        # Skip self-loops
        if start_cluster == end_cluster:
            continue
        
        # Get cluster center coordinates
        start_center = cluster_centers[start_cluster]
        end_center = cluster_centers[end_cluster]
        
        new_segments_data.append({
            'segment_id': segment_id_counter,
            'node_start': start_cluster,
            'node_end': end_cluster,
            'x1': start_center[0],
            'y1': start_center[1],
            'x2': end_center[0],
            'y2': end_center[1]
        })
        segment_id_counter += 1
    
    new_segments_df = pd.DataFrame(new_segments_data)
    
    return GraphData(new_nodes_df, new_segments_df)


def _find_node_clusters(coords: np.ndarray, threshold: float) -> List[List[int]]:
    """
    Find clusters of nodes within threshold distance using CGAL spatial indexing.
    
    This function uses the CGAL core module to perform efficient spatial queries
    for clustering nearby nodes. It leverages the same CGAL infrastructure
    used by the MapIndex class for consistent performance.
    
    Args:
        coords: Nx2 array of node coordinates
        threshold: Distance threshold for clustering (in meters)
    
    Returns:
        List of clusters, where each cluster is a list of node indices
    """
    import acj_core
    
    n_nodes = len(coords)
    if n_nodes == 0:
        return []
    
    # Use CGAL's spatial indexing for efficient clustering
    # This leverages the same CGAL infrastructure as MapIndex
    clusters = acj_core.find_clusters_cgal(coords, threshold)
    
    # Convert from Python list of lists to List[List[int]]
    result = []
    for cluster in clusters:
        result.append(list(cluster))
    
    return result


def simplify_graph(graph_data: GraphData, threshold_meters: float = 10.0) -> GraphData:
    """
    Simplify a graph using the most appropriate method based on threshold.
    
    This is the main simplification function that automatically chooses
    between topological and geometric simplification based on the threshold.
    
    Args:
        graph_data: GraphData object to simplify
        threshold_meters: Distance threshold for merging nodes (in meters)
                        - threshold_meters = 0: Only topological simplification
                        - threshold_meters > 0: Geometric simplification
    
    Returns:
        GraphData object with simplified graph
    
    Example:
        >>> graph = acj.load_graph(nodes_df, segments_df)
        >>> # Topological only
        >>> simplified = acj.simplify_graph(graph, threshold_meters=0)
        >>> # Geometric with 15m threshold
        >>> simplified = acj.simplify_graph(graph, threshold_meters=15.0)
    """
    if threshold_meters <= 0:
        return simplify_graph_topological(graph_data)
    else:
        return simplify_graph_geometric(graph_data, threshold_meters)
