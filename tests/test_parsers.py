import pytest
import pandas as pd
import networkx as nx
from geojac import UrbanNetwork

def test_from_dataframe():
    nodes_data = {
        'node_id': [0, 1],
        'x': [10.0, 20.0],
        'y': [15.0, 25.0],
        'name': ['Intersection A', 'Intersection B']
    }
    edges_data = {
        'segment_id': [0],
        'node_start': [0],
        'node_end': [1],
        'street_name': ['Main St'],
        'speed_limit': [50]
    }
    
    nodes_df = pd.DataFrame(nodes_data)
    edges_df = pd.DataFrame(edges_data)
    
    network = UrbanNetwork.from_dataframe(nodes_df, edges_df)
    
    # Check Topological Invariants
    assert list(network.nodes_df.columns) == ['node_id', 'x', 'y']
    assert list(network.edges_df.columns) == ['segment_id', 'node_start', 'node_end']
    
    # Check Semantic Parsing
    assert network.node_metadata[0]['name'] == 'Intersection A'
    assert network.edge_metadata[0]['street_name'] == 'Main St'
    assert network.edge_metadata[0]['speed_limit'] == 50

def test_from_networkx():
    G = nx.MultiDiGraph()
    G.add_node("A", x=0.0, y=0.0, osmid=1001)
    G.add_node("B", x=10.0, y=0.0, osmid=1002)
    G.add_edge("A", "B", name="Broadway", maxspeed="60 mph")
    
    network = UrbanNetwork.from_networkx(G)
    
    assert len(network.nodes_df) == 2
    assert len(network.edges_df) == 1
    
    # Check node metadata extraction
    assert network.node_metadata[0]['osmid'] == 1001
    
    # Check edge metadata extraction
    assert network.edge_metadata[0]['name'] == 'Broadway'
    assert network.edge_metadata[0]['maxspeed'] == '60 mph'
    
    # Verify sequential mapping of topological edges
    assert network.edges_df.iloc[0]['node_start'] == 0
    assert network.edges_df.iloc[0]['node_end'] == 1
