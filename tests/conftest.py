"""Shared fixtures for the ACJ test suite (Layers 0-4)."""
import numpy as np
import pandas as pd
import pytest

from geojac.core.network import UrbanNetwork
from geojac.data.io import GraphData, SimplificationResult


# ── UrbanNetwork builders ──────────────────────────────────────────────────────

def _make_urban_network(node_rows, edge_rows, *,
                        node_meta=None, edge_meta=None,
                        lineage_nodes=None, lineage_edges=None):
    n = UrbanNetwork()
    n.nodes_df      = pd.DataFrame(node_rows, columns=['node_id', 'x', 'y'])
    n.edges_df      = pd.DataFrame(edge_rows,  columns=['segment_id', 'node_start', 'node_end'])
    n.node_metadata = node_meta     or {}
    n.edge_metadata = edge_meta     or {}
    n.lineage_nodes = lineage_nodes or {}
    n.lineage_edges = lineage_edges or {}
    return n


def _make_graph_data(node_rows, seg_rows):
    nodes = pd.DataFrame(node_rows, columns=['node_id', 'x', 'y'])
    segs  = pd.DataFrame(seg_rows,  columns=['segment_id', 'node_start', 'node_end',
                                              'x1', 'y1', 'x2', 'y2'])
    return GraphData(nodes, segs)


# ── UrbanNetwork fixtures ──────────────────────────────────────────────────────

@pytest.fixture
def empty_network():
    return _make_urban_network([], [])


@pytest.fixture
def chain_network():
    """4-node undirected chain 0──1──2──3 (unit 10 m spacing)."""
    return _make_urban_network(
        [[0, 0.0, 0.0], [1, 10.0, 0.0], [2, 20.0, 0.0], [3, 30.0, 0.0]],
        [[0, 0, 1], [1, 1, 2], [2, 2, 3]],
    )


@pytest.fixture
def junction_network():
    """T-junction: node 1 (degree 3) must survive any topological simplification."""
    return _make_urban_network(
        [[0, 0.0, 0.0], [1, 10.0, 0.0], [2, 20.0, 0.0], [3, 10.0, 10.0]],
        [[0, 0, 1], [1, 1, 2], [2, 1, 3]],
    )


@pytest.fixture
def semantic_chain_network():
    """4-node chain with edge speed metadata, used by evaluation tests."""
    return _make_urban_network(
        [[0, 0.0, 0.0], [1, 10.0, 0.0], [2, 20.0, 0.0], [3, 30.0, 0.0]],
        [[0, 0, 1], [1, 1, 2], [2, 2, 3]],
        edge_meta={0: {'maxspeed': 50}, 1: {'maxspeed': 60}, 2: {'maxspeed': 70}},
    )


# ── NumPy array fixtures (Layer 0) ─────────────────────────────────────────────

@pytest.fixture
def chain_nodes_array():
    """Nx3 float64 [id, x, y] for 4-node chain."""
    return np.array(
        [[0, 0.0, 0.0], [1, 10.0, 0.0], [2, 20.0, 0.0], [3, 30.0, 0.0]],
        dtype=np.float64,
    )


@pytest.fixture
def chain_segs_array():
    """Mx3 float64 [seg_id, u, v] for 4-node chain."""
    return np.array([[0, 0, 1], [1, 1, 2], [2, 2, 3]], dtype=np.float64)


@pytest.fixture
def empty_nodes_array():
    return np.empty((0, 3), dtype=np.float64)


@pytest.fixture
def empty_segs_array():
    return np.empty((0, 3), dtype=np.float64)


# ── GraphData fixtures (Layer 1) ───────────────────────────────────────────────

@pytest.fixture
def chain_graph_data():
    return _make_graph_data(
        [[0, 0.0, 0.0], [1, 10.0, 0.0], [2, 20.0, 0.0], [3, 30.0, 0.0]],
        [[0, 0, 1, 0.0, 0.0, 10.0, 0.0],
         [1, 1, 2, 10.0, 0.0, 20.0, 0.0],
         [2, 2, 3, 20.0, 0.0, 30.0, 0.0]],
    )
