import pandas as pd

class UrbanNetwork:
    """
    Core container for topological-semantic decoupling.
    Encapsulates topology for C++ operations and semantics for Python.
    """

    def __init__(self):
        # Topological Attributes (Strictly formatted for C++ invariance)
        self.nodes_df = pd.DataFrame(columns=['node_id', 'x', 'y'])
        self.edges_df = pd.DataFrame(columns=['segment_id', 'node_start', 'node_end'])

        # Semantic Attributes
        self.node_metadata = {}
        self.edge_metadata = {}

        # Lineage/Traceability Attributes
        self.lineage_nodes = {}
        self.lineage_edges = {}

    def __repr__(self) -> str:
        num_nodes = len(self.nodes_df)
        num_edges = len(self.edges_df)
        has_node_meta = len(self.node_metadata) > 0
        has_edge_meta = len(self.edge_metadata) > 0

        return (
            f"<UrbanNetwork | "
            f"Nodes: {num_nodes} (Meta: {has_node_meta}) | "
            f"Edges: {num_edges} (Meta: {has_edge_meta})>"
        )

    @classmethod
    def from_dataframe(cls, nodes_df: pd.DataFrame, edges_df: pd.DataFrame) -> 'UrbanNetwork':
        network = cls()
        
        # Topological
        network.nodes_df = nodes_df[['node_id', 'x', 'y']].copy()
        network.edges_df = edges_df[['segment_id', 'node_start', 'node_end']].copy()
        
        # Semantic
        nodes_meta_cols = [col for col in nodes_df.columns if col not in ['node_id', 'x', 'y']]
        if nodes_meta_cols:
            network.node_metadata = nodes_df.set_index('node_id')[nodes_meta_cols].to_dict('index')
            
        edges_meta_cols = [col for col in edges_df.columns if col not in ['segment_id', 'node_start', 'node_end']]
        if edges_meta_cols:
            network.edge_metadata = edges_df.set_index('segment_id')[edges_meta_cols].to_dict('index')
            
        return network

    @classmethod
    def from_networkx(cls, G) -> 'UrbanNetwork':
        network = cls()
        
        nodes_data = []
        node_metadata = {}
        node_id_map = {}
        
        # Extract Nodes
        for idx, (osm_id, data) in enumerate(G.nodes(data=True)):
            node_id_map[osm_id] = idx
            nodes_data.append({
                'node_id': idx,
                'x': data.get('x', 0.0),
                'y': data.get('y', 0.0)
            })
            
            # Semantic
            meta = {k: v for k, v in data.items() if k not in ['x', 'y']}
            if meta:
                node_metadata[idx] = meta
                
        network.nodes_df = pd.DataFrame(nodes_data)
        network.node_metadata = node_metadata
        
        # Extract Edges
        edges_data = []
        edge_metadata = {}
        segment_id = 0
        
        # Determine whether G is a MultiGraph or standard Graph
        is_multi = G.is_multigraph()
        edge_iter = G.edges(keys=True, data=True) if is_multi else ((u, v, 0, d) for u, v, d in G.edges(data=True))
        
        for u, v, k, data in edge_iter:
            edges_data.append({
                'segment_id': segment_id,
                'node_start': node_id_map[u],
                'node_end': node_id_map[v]
            })
            
            # Semantic
            meta = {key: val for key, val in data.items()}
            if meta:
                edge_metadata[segment_id] = meta
                
            segment_id += 1
            
        network.edges_df = pd.DataFrame(edges_data)
        network.edge_metadata = edge_metadata
        
        return network

