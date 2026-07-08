"""Metric Strategy implementations for ACJ graph simplification evaluation."""

import abc
import math
from typing import Dict

import networkx as nx
import numpy as np

from geojac.core.network import UrbanNetwork


def _nn_query(ref_xy: np.ndarray, query_xy: np.ndarray):
    """
    Pure-numpy nearest-neighbour: for each row in query_xy find closest row in ref_xy.
    Returns (distances, indices) of shape (len(query_xy),).
    O(Q * R) — acceptable for anchor-node sets (<1 000 pts).
    Falls back to scipy cKDTree when available for large sets.
    """
    try:
        from scipy.spatial import cKDTree

        tree = cKDTree(ref_xy)
        return tree.query(query_xy)
    except ImportError:
        pass
    # numpy fallback
    diff = ref_xy[np.newaxis, :, :] - query_xy[:, np.newaxis, :]  # (Q, R, 2)
    dists = np.sqrt((diff**2).sum(axis=2))  # (Q, R)
    idx = dists.argmin(axis=1)
    return dists[np.arange(len(query_xy)), idx], idx


# ── Abstract base ─────────────────────────────────────────────────────────────


class Metric(abc.ABC):
    @abc.abstractmethod
    def compute(self, original: UrbanNetwork, simplified: UrbanNetwork) -> float:
        pass


# ── Internal NetworkX conversion ──────────────────────────────────────────────


def _to_networkx(network: UrbanNetwork) -> nx.DiGraph:
    """Build a weighted DiGraph from UrbanNetwork; 'length' falls back to Euclidean."""
    G = nx.DiGraph()
    node_xy = {}
    for _, row in network.nodes_df.iterrows():
        nid = int(row["node_id"])
        x, y = float(row["x"]), float(row["y"])
        G.add_node(nid, x=x, y=y)
        node_xy[nid] = (x, y)

    for _, row in network.edges_df.iterrows():
        sid = int(row["segment_id"])
        u, v = int(row["node_start"]), int(row["node_end"])
        meta = dict(network.edge_metadata.get(sid, {}))
        if "length" not in meta and u in node_xy and v in node_xy:
            dx = node_xy[u][0] - node_xy[v][0]
            dy = node_xy[u][1] - node_xy[v][1]
            meta["length"] = math.hypot(dx, dy)
        G.add_edge(u, v, **meta)
    return G


def _anchor_nodes(G: nx.DiGraph):
    """Nodes with degree ≠ 2 (intersections/endpoints). Falls back to all nodes."""
    anchors = [n for n in G.nodes() if G.degree(n) != 2]
    return anchors if len(anchors) >= 2 else list(G.nodes())


def _spatial_map(G_raw: nx.DiGraph, G_model: nx.DiGraph, max_dist_m: float):
    """Map anchor nodes of G_raw to nearest node in G_model within max_dist_m."""
    raw_anchors = _anchor_nodes(G_raw)
    model_nodes = list(G_model.nodes())
    if not model_nodes:
        return [], {}
    model_xy = np.array(
        [[G_model.nodes[n]["x"], G_model.nodes[n]["y"]] for n in model_nodes]
    )
    raw_xy = np.array([[G_raw.nodes[u]["x"], G_raw.nodes[u]["y"]] for u in raw_anchors])
    dists, indices = _nn_query(model_xy, raw_xy)
    valid, mapping = [], {}
    for i, u in enumerate(raw_anchors):
        if dists[i] <= max_dist_m:
            mapping[u] = model_nodes[indices[i]]
            valid.append(u)
    return valid, mapping


# ── Module-level pairwise cache keyed by (id(orig), id(simp), max_dist_m) ────

_PAIRWISE_CACHE: Dict[tuple, dict] = {}

_PAIRWISE_EMPTY = {
    "path_error_abs_median": 0.0,
    "path_error_abs_p95": 0.0,
    "path_ratio_median": 1.0,
    "path_ratio_p95": 1.0,
}


def _pairwise_cached(
    original: UrbanNetwork, simplified: UrbanNetwork, max_dist_m: float
) -> dict:
    key = (id(original), id(simplified), max_dist_m)
    if key in _PAIRWISE_CACHE:
        return _PAIRWISE_CACHE[key]

    G_orig = _to_networkx(original)
    G_simp = _to_networkx(simplified)
    valid_raw, mapping = _spatial_map(G_orig, G_simp, max_dist_m)

    if len(valid_raw) < 2:
        _PAIRWISE_CACHE[key] = _PAIRWISE_EMPTY
        return _PAIRWISE_EMPTY

    abs_diffs, ratios = [], []
    for u_raw in valid_raw:
        u_model = mapping[u_raw]
        try:
            paths_raw = nx.single_source_dijkstra_path_length(
                G_orig, u_raw, weight="length"
            )
        except Exception:
            paths_raw = {}
        try:
            paths_model = nx.single_source_dijkstra_path_length(
                G_simp, u_model, weight="length"
            )
        except Exception:
            paths_model = {}
        for v_raw in valid_raw:
            if u_raw == v_raw:
                continue
            v_model = mapping[v_raw]
            d_raw = paths_raw.get(v_raw)
            d_mod = paths_model.get(v_model)
            if d_raw is not None and d_mod is not None and d_raw > 0:
                abs_diffs.append(abs(d_mod - d_raw))
                ratios.append(d_mod / d_raw)

    if not abs_diffs:
        _PAIRWISE_CACHE[key] = _PAIRWISE_EMPTY
        return _PAIRWISE_EMPTY

    result = {
        "path_error_abs_median": round(float(np.median(abs_diffs)), 4),
        "path_error_abs_p95": round(float(np.percentile(abs_diffs, 95)), 4),
        "path_ratio_median": round(float(np.median(ratios)), 4),
        "path_ratio_p95": round(float(np.percentile(ratios, 95)), 4),
    }
    _PAIRWISE_CACHE[key] = result
    return result


# ── Existing metrics (kept unchanged) ─────────────────────────────────────────


class CompressionRatioMetric(Metric):
    def compute(self, original: UrbanNetwork, simplified: UrbanNetwork) -> float:
        if len(original.nodes_df) == 0:
            return 0.0
        return 1.0 - (len(simplified.nodes_df) / len(original.nodes_df))


class SemanticSpeedDistortionMetric(Metric):
    def compute(self, original: UrbanNetwork, simplified: UrbanNetwork) -> float:
        total, count = 0.0, 0
        for new_id, old_ids in simplified.lineage_edges.items():
            if new_id not in simplified.edge_metadata:
                continue
            new_speed = simplified.edge_metadata[new_id].get("maxspeed")
            if new_speed is None:
                continue
            for old_id in old_ids:
                old_speed = original.edge_metadata.get(old_id, {}).get("maxspeed")
                if old_speed is not None:
                    try:
                        ns = float(str(new_speed).split(" | ")[0])
                        os_ = float(str(old_speed).split(" | ")[0])
                        total += abs(ns - os_)
                        count += 1
                    except (ValueError, TypeError):
                        pass
        return total / count if count else 0.0


# ── New metrics from legacy math ──────────────────────────────────────────────


class KeypointDisplacementMetric(Metric):
    """
    Mean nearest-neighbour displacement (metres) between anchor nodes
    (degree ≠ 2) of original and simplified graphs. Uses cKDTree O(N log N).
    Lower is better. 0 = perfect geometric preservation of intersections.
    """

    def compute(self, original: UrbanNetwork, simplified: UrbanNetwork) -> float:
        G_orig = _to_networkx(original)
        G_simp = _to_networkx(simplified)

        kp_orig = np.array(
            [
                [G_orig.nodes[n]["x"], G_orig.nodes[n]["y"]]
                for n in G_orig.nodes()
                if G_orig.degree(n) != 2
            ]
        )
        kp_simp = np.array(
            [
                [G_simp.nodes[n]["x"], G_simp.nodes[n]["y"]]
                for n in G_simp.nodes()
                if G_simp.degree(n) != 2
            ]
        )

        if len(kp_orig) == 0 or len(kp_simp) == 0:
            return 0.0

        distances, _ = _nn_query(kp_orig, kp_simp)
        return round(float(np.mean(distances)), 4)


class ReachabilityPreservationMetric(Metric):
    """
    Fraction (%) of reachable anchor-node pairs in the original that remain
    reachable after simplification. Spatial alignment via cKDTree within
    max_dist_m. All-pairs BFS. 100 = perfect. Lower = connectivity loss.
    """

    def __init__(self, max_dist_m: float = 25.0):
        self.max_dist_m = max_dist_m

    def compute(self, original: UrbanNetwork, simplified: UrbanNetwork) -> float:
        G_orig = _to_networkx(original)
        G_simp = _to_networkx(simplified)
        valid_raw, mapping = _spatial_map(G_orig, G_simp, self.max_dist_m)
        if len(valid_raw) < 2:
            return 0.0

        total = preserved = 0
        for u in valid_raw:
            reach_orig = nx.single_source_shortest_path_length(G_orig, u)
            reach_simp = nx.single_source_shortest_path_length(G_simp, mapping[u])
            for v in valid_raw:
                if u == v:
                    continue
                if v in reach_orig:
                    total += 1
                    if mapping[v] in reach_simp:
                        preserved += 1

        return round((preserved / total) * 100.0, 2) if total else 100.0


class PathErrorMedianMetric(Metric):
    """
    Median |d_simplified(u,v) - d_original(u,v)| over all valid anchor-node
    pairs (metres, all-pairs Dijkstra). Shares cached computation with P95/ratio
    siblings to avoid redundant O(V² log V) traversals.
    """

    def __init__(self, max_dist_m: float = 25.0):
        self.max_dist_m = max_dist_m

    def compute(self, original: UrbanNetwork, simplified: UrbanNetwork) -> float:
        return _pairwise_cached(original, simplified, self.max_dist_m)[
            "path_error_abs_median"
        ]


class PathErrorP95Metric(Metric):
    """95th-percentile absolute path-length error (metres). Worst-case path distortion."""

    def __init__(self, max_dist_m: float = 25.0):
        self.max_dist_m = max_dist_m

    def compute(self, original: UrbanNetwork, simplified: UrbanNetwork) -> float:
        return _pairwise_cached(original, simplified, self.max_dist_m)[
            "path_error_abs_p95"
        ]


class PathRatioMedianMetric(Metric):
    """Median d_simplified / d_original. 1.0 = perfect; >1 = path inflation."""

    def __init__(self, max_dist_m: float = 25.0):
        self.max_dist_m = max_dist_m

    def compute(self, original: UrbanNetwork, simplified: UrbanNetwork) -> float:
        return _pairwise_cached(original, simplified, self.max_dist_m)[
            "path_ratio_median"
        ]


class PathRatioP95Metric(Metric):
    """95th-percentile path-length ratio. Worst-case path inflation."""

    def __init__(self, max_dist_m: float = 25.0):
        self.max_dist_m = max_dist_m

    def compute(self, original: UrbanNetwork, simplified: UrbanNetwork) -> float:
        return _pairwise_cached(original, simplified, self.max_dist_m)["path_ratio_p95"]


# ── Descriptive metrics (operate on `simplified` only; `original` is ignored) ─


class NodeCountMetric(Metric):
    """Number of nodes in the simplified graph."""

    def compute(self, original: UrbanNetwork, simplified: UrbanNetwork) -> float:
        return float(len(simplified.nodes_df))


class EdgeCountMetric(Metric):
    """Number of edges in the simplified graph."""

    def compute(self, original: UrbanNetwork, simplified: UrbanNetwork) -> float:
        return float(len(simplified.edges_df))


class TotalCoordinatesMetric(Metric):
    """
    Total polyline vertex count across all edges.
    Uses geometry.coords when available; falls back to 2 (endpoints only).
    """

    def compute(self, original: UrbanNetwork, simplified: UrbanNetwork) -> float:
        total = 0
        for _, row in simplified.edges_df.iterrows():
            sid = int(row["segment_id"])
            geom = simplified.edge_metadata.get(sid, {}).get("geometry")
            total += (
                len(geom.coords)
                if (geom is not None and hasattr(geom, "coords"))
                else 2
            )
        return float(total)


class TotalLengthKmMetric(Metric):
    """Total network length in kilometres (sums edge 'length' attribute)."""

    def compute(self, original: UrbanNetwork, simplified: UrbanNetwork) -> float:
        G = _to_networkx(simplified)
        total = sum(data.get("length", 0.0) or 0.0 for _, _, data in G.edges(data=True))
        return round(total / 1000, 4)


class AvgDegreeMetric(Metric):
    """Mean node degree (in + out for directed graphs)."""

    def compute(self, original: UrbanNetwork, simplified: UrbanNetwork) -> float:
        G = _to_networkx(simplified)
        degrees = [d for _, d in G.degree()]
        return round(float(np.mean(degrees)), 4) if degrees else 0.0


class AvgSinuosityMetric(Metric):
    """
    Mean sinuosity = actual_edge_length / straight_line_distance.
    Uses geometry.length when available; otherwise length attribute.
    Value of 1.0 means perfectly straight edges.
    """

    def compute(self, original: UrbanNetwork, simplified: UrbanNetwork) -> float:
        G = _to_networkx(simplified)
        sinuosities = []
        for u, v, data in G.edges(data=True):
            geom = data.get("geometry")
            length = data.get("length", 0.0) or 0.0
            if geom is not None and hasattr(geom, "length") and geom.length > 0:
                length = geom.length
            xu, yu = G.nodes[u]["x"], G.nodes[u]["y"]
            xv, yv = G.nodes[v]["x"], G.nodes[v]["y"]
            straight = math.hypot(xu - xv, yu - yv)
            if straight > 0 and length > 0:
                sinuosities.append(length / straight)
        return round(float(np.mean(sinuosities)), 6) if sinuosities else 1.0


class ConnectivityIndexMetric(Metric):
    """
    Number of (weakly) connected components.
    1 = fully connected. Uses weakly-connected for directed graphs.
    """

    def compute(self, original: UrbanNetwork, simplified: UrbanNetwork) -> float:
        G = _to_networkx(simplified)
        if G.is_directed():
            return float(nx.number_weakly_connected_components(G))
        return float(nx.number_connected_components(G))
