"""Layer 2: UrbanNetwork domain model — init contracts, factory methods, lineage."""
import networkx as nx
import pandas as pd
import pytest

from geojac.core.network import UrbanNetwork


class TestUrbanNetworkInit:
    def test_nodes_df_initialized_with_correct_schema(self):
        n = UrbanNetwork()
        assert list(n.nodes_df.columns) == ['node_id', 'x', 'y']
        assert len(n.nodes_df) == 0

    def test_edges_df_initialized_with_correct_schema(self):
        n = UrbanNetwork()
        assert list(n.edges_df.columns) == ['segment_id', 'node_start', 'node_end']
        assert len(n.edges_df) == 0

    def test_metadata_dicts_empty_by_default(self):
        n = UrbanNetwork()
        assert n.node_metadata == {}
        assert n.edge_metadata == {}

    def test_lineage_nodes_empty_by_default(self):
        assert UrbanNetwork().lineage_nodes == {}

    def test_lineage_edges_empty_by_default(self):
        assert UrbanNetwork().lineage_edges == {}


class TestUrbanNetworkRepr:
    def test_repr_shows_zero_counts_on_empty_network(self):
        s = repr(UrbanNetwork())
        assert "Nodes: 0" in s
        assert "Edges: 0" in s

    def test_repr_shows_meta_false_when_no_metadata(self):
        assert "Meta: False" in repr(UrbanNetwork())

    def test_repr_shows_meta_true_after_node_metadata_added(self):
        n = UrbanNetwork()
        n.node_metadata[1] = {"highway": "residential"}
        assert "Meta: True" in repr(n)


class TestUrbanNetworkFromDataframe:
    @pytest.fixture
    def nodes_with_extras(self):
        return pd.DataFrame({
            'node_id': [0, 1], 'x': [10.0, 20.0], 'y': [15.0, 25.0],
            'name': ['A', 'B'], 'highway': ['primary', 'secondary'],
        })

    @pytest.fixture
    def edges_with_extras(self):
        return pd.DataFrame({
            'segment_id': [0], 'node_start': [0], 'node_end': [1],
            'street_name': ['Main St'], 'speed_limit': [50],
        })

    def test_topology_columns_isolated_from_semantic_extras(self, nodes_with_extras, edges_with_extras):
        n = UrbanNetwork.from_dataframe(nodes_with_extras, edges_with_extras)
        assert list(n.nodes_df.columns) == ['node_id', 'x', 'y']
        assert list(n.edges_df.columns) == ['segment_id', 'node_start', 'node_end']

    def test_extra_node_columns_stored_in_node_metadata(self, nodes_with_extras, edges_with_extras):
        n = UrbanNetwork.from_dataframe(nodes_with_extras, edges_with_extras)
        assert n.node_metadata[0]['name'] == 'A'
        assert n.node_metadata[1]['highway'] == 'secondary'

    def test_extra_edge_columns_stored_in_edge_metadata(self, nodes_with_extras, edges_with_extras):
        n = UrbanNetwork.from_dataframe(nodes_with_extras, edges_with_extras)
        assert n.edge_metadata[0]['street_name'] == 'Main St'
        assert n.edge_metadata[0]['speed_limit'] == 50

    def test_no_extra_columns_produces_empty_metadata(self):
        nodes = pd.DataFrame({'node_id': [0], 'x': [0.0], 'y': [0.0]})
        edges = pd.DataFrame({'segment_id': [0], 'node_start': [0], 'node_end': [0]})
        n = UrbanNetwork.from_dataframe(nodes, edges)
        assert n.node_metadata == {} or 0 not in n.node_metadata


class TestUrbanNetworkFromNetworkx:
    def test_multigraph_nodes_mapped_to_sequential_integer_ids(self):
        G = nx.MultiDiGraph()
        G.add_node("X", x=0.0, y=0.0, osmid=111)
        G.add_node("Y", x=10.0, y=0.0, osmid=222)
        G.add_edge("X", "Y", name="Broadway")
        n = UrbanNetwork.from_networkx(G)
        assert set(n.nodes_df['node_id']) == {0, 1}
        assert n.edges_df.iloc[0]['node_start'] == 0
        assert n.edges_df.iloc[0]['node_end'] == 1

    def test_multigraph_node_attributes_extracted_to_metadata(self):
        G = nx.MultiDiGraph()
        G.add_node("A", x=0.0, y=0.0, osmid=1001)
        G.add_node("B", x=5.0, y=0.0, osmid=1002)
        G.add_edge("A", "B")
        n = UrbanNetwork.from_networkx(G)
        assert n.node_metadata[0]['osmid'] == 1001

    def test_multigraph_edge_attributes_extracted_to_metadata(self):
        G = nx.MultiDiGraph()
        G.add_node("A", x=0.0, y=0.0)
        G.add_node("B", x=5.0, y=0.0)
        G.add_edge("A", "B", maxspeed="60 mph", lanes=2)
        n = UrbanNetwork.from_networkx(G)
        assert n.edge_metadata[0]['maxspeed'] == "60 mph"
        assert n.edge_metadata[0]['lanes'] == 2

    def test_simple_digraph_parsed_without_error(self):
        G = nx.DiGraph()
        G.add_node(0, x=0.0, y=0.0)
        G.add_node(1, x=1.0, y=0.0)
        G.add_edge(0, 1)
        n = UrbanNetwork.from_networkx(G)
        assert len(n.nodes_df) == 2
        assert len(n.edges_df) == 1

    def test_graph_with_no_extra_attributes_produces_empty_metadata(self):
        G = nx.MultiDiGraph()
        G.add_node("P", x=0.0, y=0.0)
        G.add_node("Q", x=1.0, y=0.0)
        G.add_edge("P", "Q")
        n = UrbanNetwork.from_networkx(G)
        # edges have no extra attrs → metadata may be empty or contain empty dict
        for v in n.edge_metadata.values():
            assert isinstance(v, dict)
