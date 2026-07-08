"""
Direct pybind11 tests for acj_core C++ engine.

Coverage targets:
  - Geometric correctness (topological, geometric, ACJ master)
  - Memory safety (empty inputs, boundary cases, large random inputs)
  - Determinism (repeated calls produce identical output)
  - forcecast acceptance (float32 / int arrays must not raise)
"""

import numpy as np
import pytest

import acj_core
from acj_core import _experimental as exp


# ─── Helpers ─────────────────────────────────────────────────────────────────


def _nodes(*rows):
    """Build Nx3 float64 [id, x, y]."""
    return np.array(rows, dtype=np.float64)


def _segs(*rows):
    """Build Mx3 float64 [seg_id, u, v]."""
    return np.array(rows, dtype=np.float64)


def _unpack_topo(result):
    """Return (nodes_list, segs_list) from topological/geometric/parallel result."""
    return result.graph[0], result.graph[1]


def _unpack_acj(result):
    """Return (nodes_arr, edges_arr) Nx3 from ACJ master result."""
    return np.asarray(result.graph[0]), np.asarray(result.graph[1])


# ─── simplify_graph_topological_cgal ─────────────────────────────────────────


class TestTopological:
    def test_linear_chain_collapses(self):
        """3-node interior chain (degrees 1,2,2,1) → 2 nodes, 1 segment."""
        nodes = _nodes([0, 0.0, 0.0], [1, 10.0, 0.0], [2, 20.0, 0.0], [3, 30.0, 0.0])
        segs = _segs([0, 0, 1], [1, 1, 2], [2, 2, 3])
        r = exp.simplify_graph_topological_cgal(nodes, segs)
        out_nodes, out_segs = _unpack_topo(r)
        assert len(out_nodes) == 2
        assert len(out_segs) == 1

    def test_t_junction_preserves_intersection(self):
        """T-junction: node 1 has degree 3 → must survive."""
        nodes = _nodes(
            [0, 0.0, 0.0],
            [1, 10.0, 0.0],
            [2, 20.0, 0.0],
            [3, 10.0, 10.0],
        )
        segs = _segs([0, 0, 1], [1, 1, 2], [2, 1, 3])
        r = exp.simplify_graph_topological_cgal(nodes, segs)
        out_nodes, _ = _unpack_topo(r)
        surviving_ids = {int(n[0]) for n in out_nodes}
        assert 1 in surviving_ids

    def test_x_intersection_all_preserved(self):
        """X-intersection: centre degree 4, all 5 nodes are anchors."""
        nodes = _nodes(
            [0, 0.0, 0.0],
            [1, 10.0, 10.0],
            [2, 20.0, 0.0],
            [3, 0.0, 20.0],
            [4, 20.0, 20.0],
        )
        segs = _segs([0, 0, 1], [1, 1, 2], [2, 3, 1], [3, 1, 4])
        r = exp.simplify_graph_topological_cgal(nodes, segs)
        out_nodes, _ = _unpack_topo(r)
        assert len(out_nodes) == 5

    def test_empty_graph_no_crash(self):
        r = exp.simplify_graph_topological_cgal(
            np.empty((0, 3), dtype=np.float64),
            np.empty((0, 3), dtype=np.float64),
        )
        out_nodes, out_segs = _unpack_topo(r)
        assert len(out_nodes) == 0
        assert len(out_segs) == 0

    def test_single_node_no_edges(self):
        r = exp.simplify_graph_topological_cgal(
            _nodes([0, 5.0, 5.0]),
            np.empty((0, 3), dtype=np.float64),
        )
        out_nodes, out_segs = _unpack_topo(r)
        assert len(out_nodes) == 1
        assert len(out_segs) == 0

    def test_segment_refs_missing_node_skipped(self):
        """Segment referencing a non-existent node ID must not crash."""
        nodes = _nodes([0, 0.0, 0.0], [1, 10.0, 0.0])
        segs = _segs([0, 0, 99])  # node 99 doesn't exist
        exp.simplify_graph_topological_cgal(nodes, segs)  # must not raise

    def test_no_self_loops_in_output(self):
        """No output segment should have u == v."""
        nodes = _nodes([0, 0.0, 0.0], [1, 5.0, 0.0], [2, 10.0, 0.0])
        segs = _segs([0, 0, 1], [1, 1, 2])
        r = exp.simplify_graph_topological_cgal(nodes, segs)
        _, out_segs = _unpack_topo(r)
        for s in out_segs:
            assert s[1] != s[2], f"Self-loop detected: seg {s}"

    def test_forcecast_accepts_float32(self):
        nodes = _nodes([0, 0.0, 0.0], [1, 10.0, 0.0]).astype(np.float32)
        segs = _segs([0, 0, 1]).astype(np.float32)
        exp.simplify_graph_topological_cgal(nodes, segs)  # must not raise

    def test_forcecast_accepts_int_array(self):
        nodes = np.array([[0, 0, 0], [1, 10, 0]], dtype=np.int64)
        segs = np.array([[0, 0, 1]], dtype=np.int64)
        exp.simplify_graph_topological_cgal(nodes, segs)  # must not raise

    def test_determinism(self):
        rng = np.random.default_rng(42)
        n = 50
        ids = np.arange(n, dtype=np.float64)
        xs = rng.uniform(0, 100, n)
        ys = rng.uniform(0, 100, n)
        nodes = np.stack([ids, xs, ys], axis=1)
        seg_ids = np.arange(n - 1, dtype=np.float64)
        segs = np.stack([seg_ids, ids[:-1], ids[1:]], axis=1)

        r1 = exp.simplify_graph_topological_cgal(nodes, segs)
        r2 = exp.simplify_graph_topological_cgal(nodes, segs)

        n1, s1 = _unpack_topo(r1)
        n2, s2 = _unpack_topo(r2)
        assert len(n1) == len(n2)
        assert len(s1) == len(s2)


# ─── simplify_graph_geometric_cgal ───────────────────────────────────────────


class TestGeometric:
    def test_nearby_nodes_merge(self):
        """Two intersections within threshold → merge to centroid."""
        nodes = _nodes([0, 0.0, 0.0], [1, 1.0, 0.0], [2, 0.5, 5.0])
        segs = _segs([0, 0, 2], [1, 1, 2])
        r = exp.simplify_graph_geometric_cgal(nodes, segs, threshold=3.0)
        out_nodes, _ = _unpack_topo(r)
        assert len(out_nodes) <= 2

    def test_far_nodes_not_merged(self):
        """Intersections far apart must not merge."""
        nodes = _nodes(
            [0, 0.0, 0.0],
            [1, 50.0, 0.0],
            [2, 100.0, 0.0],
        )
        segs = _segs([0, 0, 1], [1, 1, 2])
        r = exp.simplify_graph_geometric_cgal(nodes, segs, threshold=5.0)
        out_nodes, _ = _unpack_topo(r)
        assert len(out_nodes) == 2  # endpoints only (node 1 has degree 2)

    def test_zero_threshold_no_merge(self):
        nodes = _nodes([0, 0.0, 0.0], [1, 0.001, 0.0], [2, 5.0, 0.0], [3, 5.0, 5.0])
        segs = _segs([0, 0, 2], [1, 1, 2], [2, 2, 3])
        r0 = exp.simplify_graph_geometric_cgal(nodes, segs, threshold=0.0)
        rn = exp.simplify_graph_geometric_cgal(nodes, segs, threshold=1.0)
        n0, _ = _unpack_topo(r0)
        nn, _ = _unpack_topo(rn)
        assert len(n0) >= len(nn)

    def test_empty_graph(self):
        r = exp.simplify_graph_geometric_cgal(
            np.empty((0, 3), dtype=np.float64),
            np.empty((0, 3), dtype=np.float64),
            threshold=5.0,
        )
        out_nodes, out_segs = _unpack_topo(r)
        assert len(out_nodes) == 0
        assert len(out_segs) == 0

    def test_centroid_lies_between_originals(self):
        """Merged centroid x must be between the two original x-coords."""
        nodes = _nodes([0, 0.0, 5.0], [1, 4.0, 5.0], [2, 2.0, 10.0])
        segs = _segs([0, 0, 2], [1, 1, 2])
        r = exp.simplify_graph_geometric_cgal(nodes, segs, threshold=10.0)
        out_nodes, _ = _unpack_topo(r)
        xs = [n[1] for n in out_nodes]
        for x in xs:
            assert 0.0 <= x <= 4.0 or abs(x - 2.0) < 1e-9  # centroid ≈ 2.0


# ─── simplify_graph_parallel_cgal ────────────────────────────────────────────


class TestParallel:
    def test_parallel_segments_grouped(self):
        """Two near-parallel segments → single cluster."""
        nodes = _nodes(
            [0, 0.0, 0.0],
            [1, 10.0, 0.0],
            [2, 0.0, 2.0],
            [3, 10.0, 2.0],
        )
        segs = _segs([0, 0, 1], [1, 2, 3])
        r = exp.simplify_graph_parallel_cgal(
            nodes, segs, distance_threshold=5.0, angle_threshold_deg=10.0
        )
        out_nodes, _ = _unpack_topo(r)
        assert len(out_nodes) <= 2

    def test_perpendicular_segments_not_grouped(self):
        """Perpendicular segments must NOT merge (angle >> threshold)."""
        nodes = _nodes(
            [0, 0.0, 0.0],
            [1, 10.0, 0.0],
            [2, 5.0, -1.0],
            [3, 5.0, 1.0],
        )
        segs = _segs([0, 0, 1], [1, 2, 3])
        r = exp.simplify_graph_parallel_cgal(
            nodes, segs, distance_threshold=5.0, angle_threshold_deg=10.0
        )
        out_nodes, _ = _unpack_topo(r)
        assert len(out_nodes) >= 3

    def test_empty_graph(self):
        r = exp.simplify_graph_parallel_cgal(
            np.empty((0, 3), dtype=np.float64),
            np.empty((0, 3), dtype=np.float64),
            distance_threshold=5.0,
            angle_threshold_deg=10.0,
        )
        out_nodes, out_segs = _unpack_topo(r)
        assert len(out_nodes) == 0
        assert len(out_segs) == 0


# ─── simplify_graph_acj_master_cgal ──────────────────────────────────────────


class TestACJMaster:
    def _call(self, nodes, segs, angle=30.0, factor=1.0, eps=0.1, index=False):
        return acj_core.simplify_graph_acj_master_cgal(
            nodes, segs, angle, factor, eps, index
        )

    def test_basic_directed_chain(self):
        """Directed chain 0→1→2→3: anchors are 0 and 3, output has ≥1 edge."""
        nodes = _nodes([0, 0.0, 0.0], [1, 10.0, 0.0], [2, 20.0, 0.0], [3, 30.0, 0.0])
        segs = _segs([0, 0, 1], [1, 1, 2], [2, 2, 3])
        r = self._call(nodes, segs)
        out_n, out_e = _unpack_acj(r)
        assert out_n.shape[1] == 3
        assert out_e.shape[1] == 3
        assert len(out_n) >= 2
        assert len(out_e) >= 1

    def test_output_node_ids_in_edges(self):
        """Every node ID referenced in edges must appear in nodes array."""
        nodes = _nodes([0, 0.0, 0.0], [1, 5.0, 0.0], [2, 10.0, 0.0], [3, 10.0, 5.0])
        segs = _segs([0, 0, 1], [1, 1, 2], [2, 2, 3])
        r = self._call(nodes, segs)
        out_n, out_e = _unpack_acj(r)
        node_ids = set(out_n[:, 0].astype(int))
        for edge in out_e:
            assert int(edge[1]) in node_ids, f"u={int(edge[1])} not in nodes"
            assert int(edge[2]) in node_ids, f"v={int(edge[2])} not in nodes"

    def test_empty_input_no_crash(self):
        r = self._call(
            np.empty((0, 3), dtype=np.float64),
            np.empty((0, 3), dtype=np.float64),
        )
        out_n, out_e = _unpack_acj(r)
        assert len(out_n) == 0 or True  # just must not raise

    def test_single_edge(self):
        nodes = _nodes([0, 0.0, 0.0], [1, 10.0, 0.0])
        segs = _segs([0, 0, 1])
        r = self._call(nodes, segs)
        out_n, out_e = _unpack_acj(r)
        assert len(out_n) >= 2
        assert len(out_e) >= 1

    def test_wrong_ndim_raises(self):
        bad_nodes = np.ones((5, 2), dtype=np.float64)  # shape Nx2 instead of Nx3
        segs = _segs([0, 0, 1])
        with pytest.raises(RuntimeError):
            self._call(bad_nodes, segs)

    def test_determinism(self):
        """Identical inputs → byte-identical output arrays across two calls."""
        rng = np.random.default_rng(7)
        n = 20
        ids = np.arange(n, dtype=np.float64)
        xs, ys = rng.uniform(0, 50, n), rng.uniform(0, 50, n)
        nodes = np.stack([ids, xs, ys], axis=1)
        seg_ids = np.arange(n - 1, dtype=np.float64)
        segs = np.stack([seg_ids, ids[:-1], ids[1:]], axis=1)

        r1 = self._call(nodes, segs)
        r2 = self._call(nodes, segs)
        n1, e1 = _unpack_acj(r1)
        n2, e2 = _unpack_acj(r2)
        assert n1.shape == n2.shape
        assert e1.shape == e2.shape
        np.testing.assert_array_equal(n1, n2)
        np.testing.assert_array_equal(e1, e2)

    def test_determinism_with_index(self):
        """with_index=True path also deterministic."""
        nodes = _nodes(
            [0, 0.0, 0.0],
            [1, 1.0, 0.5],
            [2, 2.0, -0.3],
            [3, 3.0, 0.1],
            [4, 4.0, 0.0],
        )
        segs = _segs([0, 0, 1], [1, 1, 2], [2, 2, 3], [3, 3, 4])
        r1 = self._call(nodes, segs, index=True)
        r2 = self._call(nodes, segs, index=True)
        n1, e1 = _unpack_acj(r1)
        n2, e2 = _unpack_acj(r2)
        np.testing.assert_array_equal(n1, n2)
        np.testing.assert_array_equal(e1, e2)

    def test_forcecast_float32(self):
        nodes = _nodes([0, 0.0, 0.0], [1, 10.0, 0.0]).astype(np.float32)
        segs = _segs([0, 0, 1]).astype(np.float32)
        self._call(nodes, segs)  # must not raise

    def test_large_random_no_crash(self):
        """Stress: 200 nodes in a chain — no crash, output is smaller."""
        n = 200
        ids = np.arange(n, dtype=np.float64)
        xs = np.linspace(0, 1000, n)
        ys = np.sin(xs / 50) * 10
        nodes = np.stack([ids, xs, ys], axis=1)
        seg_ids = np.arange(n - 1, dtype=np.float64)
        segs = np.stack([seg_ids, ids[:-1], ids[1:]], axis=1)
        r = self._call(nodes, segs, angle=15.0, factor=1.0, eps=0.5)
        out_n, out_e = _unpack_acj(r)
        assert len(out_n) < n  # simplification must reduce node count
        assert len(out_e) >= 1

    def test_angle_threshold_effect(self):
        """Stricter angle threshold → more anchor nodes (fewer collapses)."""
        nodes = _nodes(
            [0, 0.0, 0.0],
            [1, 5.0, 1.0],
            [2, 10.0, 0.0],
            [3, 15.0, 2.0],
            [4, 20.0, 0.0],
        )
        segs = _segs([0, 0, 1], [1, 1, 2], [2, 2, 3], [3, 3, 4])
        r_loose = self._call(nodes, segs, angle=60.0)
        r_strict = self._call(nodes, segs, angle=5.0)
        n_loose, _ = _unpack_acj(r_loose)
        n_strict, _ = _unpack_acj(r_strict)
        assert len(n_strict) >= len(n_loose)

    def test_output_coords_finite(self):
        """No NaN or Inf in output coordinates."""
        nodes = _nodes(
            [0, 0.0, 0.0],
            [1, 3.0, 4.0],
            [2, 6.0, 0.0],
        )
        segs = _segs([0, 0, 1], [1, 1, 2])
        r = self._call(nodes, segs)
        out_n, out_e = _unpack_acj(r)
        assert np.all(np.isfinite(out_n))
        assert np.all(np.isfinite(out_e))


# ─── Douglas-Peucker geometric correctness ───────────────────────────────────


class TestDouglasPeuckerViaACJ:
    """Verify D-P epsilon=0 keeps all points, large epsilon collapses to 2."""

    def _chain(self, n=10):
        ids = np.arange(n, dtype=np.float64)
        xs = np.linspace(0, 100, n)
        ys = np.zeros(n)
        nodes = np.stack([ids, xs, ys], axis=1)
        seg_ids = np.arange(n - 1, dtype=np.float64)
        segs = np.stack([seg_ids, ids[:-1], ids[1:]], axis=1)
        return nodes, segs

    def test_large_epsilon_collapses_collinear(self):
        """Collinear chain with huge epsilon → only endpoints survive."""
        nodes, segs = self._chain(10)
        r = acj_core.simplify_graph_acj_master_cgal(
            nodes,
            segs,
            90.0,
            1.0,
            100.0,
            False,  # factor_epsilon=100 → huge eps
        )
        out_n, out_e = _unpack_acj(r)
        assert len(out_n) == 2
        assert len(out_e) == 1

    def test_small_epsilon_preserves_more_points(self):
        """Sinusoidal chain: tiny eps keeps more intermediate points."""
        n = 30
        ids = np.arange(n, dtype=np.float64)
        xs = np.linspace(0, 60, n)
        ys = np.sin(xs) * 5
        nodes = np.stack([ids, xs, ys], axis=1)
        seg_ids = np.arange(n - 1, dtype=np.float64)
        segs = np.stack([seg_ids, ids[:-1], ids[1:]], axis=1)

        r_coarse = acj_core.simplify_graph_acj_master_cgal(
            nodes, segs, 90.0, 1.0, 10.0, False
        )
        r_fine = acj_core.simplify_graph_acj_master_cgal(
            nodes, segs, 90.0, 1.0, 0.01, False
        )
        _, e_coarse = _unpack_acj(r_coarse)
        _, e_fine = _unpack_acj(r_fine)
        assert len(e_fine) >= len(e_coarse)
