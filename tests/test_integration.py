"""Layer 4: End-to-end pipeline tests — IO → Simplification → Evaluation."""
import numpy as np
import pandas as pd
import pytest

acj_core = pytest.importorskip("acj_core", reason="acj_core (.so) not built")

from geojac.core.network import UrbanNetwork
from geojac.evaluation.evaluators import ACJTopologicalEvaluator
from geojac.evaluation.metrics import CompressionRatioMetric, SemanticSpeedDistortionMetric


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _chain(n=4, spacing=10.0):
    net = UrbanNetwork()
    net.nodes_df = pd.DataFrame({
        'node_id': list(range(n)),
        'x': [float(i * spacing) for i in range(n)],
        'y': [0.0] * n,
    })
    net.edges_df = pd.DataFrame({
        'segment_id': list(range(n - 1)),
        'node_start': list(range(n - 1)),
        'node_end':   list(range(1, n)),
    })
    return net


def _junction():
    """T-junction: all 4 nodes have degree ≠ 2 → 0% compression expected."""
    net = UrbanNetwork()
    net.nodes_df = pd.DataFrame({
        'node_id': [0, 1, 2, 3],
        'x': [0.0, 10.0, 20.0, 10.0],
        'y': [0.0, 0.0, 0.0, 10.0],
    })
    net.edges_df = pd.DataFrame({
        'segment_id': [0, 1, 2],
        'node_start': [0, 1, 1],
        'node_end':   [1, 2, 3],
    })
    return net


# ── ACJTopologicalEvaluator pipeline ──────────────────────────────────────────

class TestTopologicalEvaluatorPipeline:
    def test_chain_4_simplification_collapses_to_2_nodes(self):
        evaluator = ACJTopologicalEvaluator(_chain(4), [])
        result = evaluator.simplify()
        assert len(result.graph.nodes) == 2

    def test_chain_4_compression_ratio_equals_half(self):
        evaluator = ACJTopologicalEvaluator(_chain(4), [CompressionRatioMetric()])
        scores = evaluator.evaluate()
        assert 'CompressionRatio' in scores
        assert abs(scores['CompressionRatio'] - 0.5) < 1e-9

    def test_junction_4_compression_ratio_is_zero(self):
        """All nodes are intersections; none may be removed."""
        evaluator = ACJTopologicalEvaluator(_junction(), [CompressionRatioMetric()])
        scores = evaluator.evaluate()
        assert scores['CompressionRatio'] == 0.0

    def test_empty_network_compression_ratio_is_zero(self):
        net = UrbanNetwork()
        evaluator = ACJTopologicalEvaluator(net, [CompressionRatioMetric()])
        scores = evaluator.evaluate()
        assert scores['CompressionRatio'] == 0.0

    def test_chain_4_with_speed_metadata_semantic_distortion_is_float(self):
        net = _chain(4)
        net.edge_metadata = {0: {'maxspeed': 50}, 1: {'maxspeed': 60}, 2: {'maxspeed': 70}}
        evaluator = ACJTopologicalEvaluator(net, [SemanticSpeedDistortionMetric()])
        scores = evaluator.evaluate()
        assert 'SemanticSpeedDistortion' in scores
        assert isinstance(scores['SemanticSpeedDistortion'], float)

    def test_chain_8_compression_ratio_greater_than_chain_4(self):
        """Longer chains have more degree-2 nodes → higher compression."""
        score4 = ACJTopologicalEvaluator(_chain(4), [CompressionRatioMetric()]).evaluate()['CompressionRatio']
        score8 = ACJTopologicalEvaluator(_chain(8), [CompressionRatioMetric()]).evaluate()['CompressionRatio']
        assert score8 >= score4

    def test_evaluate_returns_dict_with_all_requested_metric_keys(self):
        evaluator = ACJTopologicalEvaluator(
            _chain(4),
            [CompressionRatioMetric(), SemanticSpeedDistortionMetric()],
        )
        scores = evaluator.evaluate()
        assert set(scores.keys()) == {'CompressionRatio', 'SemanticSpeedDistortion'}


# ── ACJ Master direct pipeline ─────────────────────────────────────────────────

class TestACJMasterDirectPipeline:
    def test_sinusoidal_200_nodes_reduces_node_count(self):
        n   = 200
        ids = np.arange(n, dtype=np.float64)
        xs  = np.linspace(0, 1000, n)
        ys  = np.sin(xs / 50) * 10
        nodes = np.stack([ids, xs, ys], axis=1)
        segs  = np.stack([np.arange(n - 1, dtype=np.float64), ids[:-1], ids[1:]], axis=1)
        result = acj_core.simplify_graph_acj_master_cgal(
            nodes, segs, 30.0, 1.0, 0.5, False)
        out_n = np.asarray(result.graph[0])
        assert len(out_n) < n

    def test_collinear_200_nodes_collapses_to_2(self):
        n   = 200
        ids = np.arange(n, dtype=np.float64)
        xs  = np.linspace(0, 1000, n)
        ys  = np.zeros(n)
        nodes = np.stack([ids, xs, ys], axis=1)
        segs  = np.stack([np.arange(n - 1, dtype=np.float64), ids[:-1], ids[1:]], axis=1)
        result = acj_core.simplify_graph_acj_master_cgal(
            nodes, segs, 90.0, 1.0, 100.0, False)
        out_n = np.asarray(result.graph[0])
        assert len(out_n) == 2

    def test_output_node_ids_referential_integrity_across_10_random_graphs(self):
        rng = np.random.default_rng(0)
        for _ in range(10):
            n   = rng.integers(5, 30)
            ids = np.arange(n, dtype=np.float64)
            nodes = np.stack([ids, rng.uniform(0, 100, n), rng.uniform(0, 100, n)], axis=1)
            segs  = np.stack([np.arange(n - 1, dtype=np.float64), ids[:-1], ids[1:]], axis=1)
            result = acj_core.simplify_graph_acj_master_cgal(nodes, segs, 30.0, 1.0, 0.5, False)
            out_n = np.asarray(result.graph[0])
            out_e = np.asarray(result.graph[1])
            if len(out_e) == 0:
                continue
            node_ids = set(out_n[:, 0].astype(int))
            for edge in out_e:
                assert int(edge[1]) in node_ids
                assert int(edge[2]) in node_ids

    def test_with_index_true_produces_finite_coordinates(self):
        n   = 20
        ids = np.arange(n, dtype=np.float64)
        xs  = np.linspace(0, 100, n)
        ys  = np.sin(xs) * 5
        nodes = np.stack([ids, xs, ys], axis=1)
        segs  = np.stack([np.arange(n - 1, dtype=np.float64), ids[:-1], ids[1:]], axis=1)
        result = acj_core.simplify_graph_acj_master_cgal(
            nodes, segs, 30.0, 1.0, 0.5, True)
        out_n = np.asarray(result.graph[0])
        out_e = np.asarray(result.graph[1])
        assert np.all(np.isfinite(out_n))
        assert np.all(np.isfinite(out_e))
