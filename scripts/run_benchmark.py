"""
Thesis benchmark script.
Downloads a city, runs all simplification algorithms, evaluates every metric
via the unified Metric pipeline, and saves plots + CSV via ThesisReportGenerator.

Usage:
    PYTHONPATH=src:build python scripts/run_benchmark.py [city_name]
"""
import math
import os
import sys

import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
from shapely.geometry import LineString, MultiLineString

# ── path bootstrap ────────────────────────────────────────────────────────────
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in [os.path.join(_ROOT, "src"), os.path.join(_ROOT, "build")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from geojac.core.network import UrbanNetwork
from geojac.evaluation.metrics import (
    # descriptive
    NodeCountMetric, EdgeCountMetric, TotalCoordinatesMetric,
    TotalLengthKmMetric, AvgDegreeMetric, AvgSinuosityMetric,
    ConnectivityIndexMetric,
    # comparative
    KeypointDisplacementMetric, ReachabilityPreservationMetric,
    PathErrorMedianMetric, PathErrorP95Metric,
    PathRatioMedianMetric, PathRatioP95Metric,
)
from geojac.evaluation.reporting import ThesisReportGenerator

# ── CONFIG ────────────────────────────────────────────────────────────────────
CITY           = "Barranco, Lima, Peru"
OUTPUT_BASE    = os.path.join(_ROOT, "outputs")
SAVE_PLOTS     = True
SAVE_CSV       = True
MAX_DIST_M     = 25.0

ACJ_ANGLE      = 80.0
ACJ_FACTOR_ADJ = 8.0
ACJ_FACTOR_EPS = 0.1
ACJ_WITH_INDEX = False

# ── Unified metric pipeline ───────────────────────────────────────────────────
PIPELINE: list[tuple[str, object]] = [
    ("nodes",                       NodeCountMetric()),
    ("edges",                       EdgeCountMetric()),
    ("coords",                      TotalCoordinatesMetric()),
    ("total_length_km",             TotalLengthKmMetric()),
    ("avg_degree",                  AvgDegreeMetric()),
    ("avg_sinuosity",               AvgSinuosityMetric()),
    ("connectivity",                ConnectivityIndexMetric()),
    ("keypoint_displacement_m",     KeypointDisplacementMetric()),
    ("reachability_preservation_%", ReachabilityPreservationMetric(MAX_DIST_M)),
    ("path_error_abs_median",       PathErrorMedianMetric(MAX_DIST_M)),
    ("path_error_abs_p95",          PathErrorP95Metric(MAX_DIST_M)),
    ("path_ratio_median",           PathRatioMedianMetric(MAX_DIST_M)),
    ("path_ratio_p95",              PathRatioP95Metric(MAX_DIST_M)),
]

# Neutral values for comparative keys when net IS the raw graph;
# avoids running O(V² log V) Dijkstra on the full unsimplified graph.
_BASELINE_DEFAULTS: dict[str, float] = {
    "keypoint_displacement_m":      0.0,
    "reachability_preservation_%":  100.0,
    "path_error_abs_median":        0.0,
    "path_error_abs_p95":           0.0,
    "path_ratio_median":            1.0,
    "path_ratio_p95":               1.0,
}


def _evaluate(raw: UrbanNetwork, net: UrbanNetwork,
              is_baseline: bool = False) -> dict:
    """Run every metric in PIPELINE and return {key: float}."""
    result: dict[str, float] = {}
    for key, metric in PIPELINE:
        if is_baseline and key in _BASELINE_DEFAULTS:
            result[key] = _BASELINE_DEFAULTS[key]
        else:
            result[key] = metric.compute(raw, net)
    return result


# ── Algorithm runners ─────────────────────────────────────────────────────────

def _run_acj_master(network: UrbanNetwork) -> UrbanNetwork:
    import acj_core

    nodes_arr = np.ascontiguousarray(
        network.nodes_df[['node_id', 'x', 'y']].values, dtype=np.float64)
    segs_arr  = np.ascontiguousarray(
        network.edges_df[['segment_id', 'node_start', 'node_end']].values, dtype=np.float64)

    result = acj_core.simplify_graph_acj_master_cgal(
        nodes_arr, segs_arr,
        float(ACJ_ANGLE), float(ACJ_FACTOR_ADJ), float(ACJ_FACTOR_EPS), ACJ_WITH_INDEX)

    net   = UrbanNetwork()
    out_n = np.asarray(result.graph[0])
    out_e = np.asarray(result.graph[1])
    if len(out_n) > 0:
        net.nodes_df = pd.DataFrame(out_n, columns=['node_id', 'x', 'y'])
    if len(out_e) > 0:
        net.edges_df = pd.DataFrame(out_e, columns=['segment_id', 'node_start', 'node_end'])
    return net


# ── NeatNet adapter ───────────────────────────────────────────────────────────

def _neatnet_simplify(edges_gdf):
    """
    Thin wrapper around neatnet.neatify().

    Expected neatnet API (verified against neatnet 0.3.x / momepy fork):
        import neatnet
        simplified_gdf = neatnet.neatify(gdf: GeoDataFrame) -> GeoDataFrame

    Input:  projected-CRS GeoDataFrame of OSM edges (from ox.graph_to_gdfs).
    Output: simplified GeoDataFrame, same CRS, with LineString geometries and
            OSM attribute columns inherited or aggregated from merged segments.

    TODO: If neatnet changes its entry point (e.g. neatnet.simplify or a
          class-based API), update the call below and verify column names.
    TODO: Check neatnet changelog for breaking changes in output schema before
          each thesis experiment run.
    """
    import neatnet  # raises ImportError if not installed
    return neatnet.neatify(edges_gdf)


def _neatnet_to_urban_network(simplified_gdf) -> UrbanNetwork:
    """
    Adapter: GeoDataFrame (neatnet output) → UrbanNetwork.

    neatnet returns only an edge GeoDataFrame; nodes are not explicit.
    We reconstruct them from edge endpoint coordinates, deduplicating by
    exact (x, y) tuple (valid because neatnet snaps endpoints to a grid).

    Handles:
    - LineString and MultiLineString geometries (MultiLineStrings are
      decomposed into individual segments).
    - NaN-valued metadata columns (dropped per edge to avoid poisoning
      metric computations).
    - Bidirectional edges: a reverse edge is added for any edge where the
      'oneway' attribute is not truthy.

    TODO: If a future neatnet version returns explicit node GeoDataFrames
          alongside the edge GDF, replace the endpoint-reconstruction logic
          with direct node ingestion to avoid floating-point deduplication.
    TODO: Verify that the 'oneway' column exists and has consistent typing
          (bool vs. str "True"/"yes") across neatnet versions.
    """
    nodes_dict: dict[tuple, int] = {}  # (x, y) → node_id
    next_node_id = 0
    node_rows:  list[dict] = []
    edge_rows:  list[dict] = []
    edge_meta:  dict[int, dict] = {}
    next_seg_id = 0

    def _get_or_add_node(pt: tuple) -> int:
        nonlocal next_node_id
        if pt not in nodes_dict:
            nodes_dict[pt] = next_node_id
            node_rows.append({'node_id': float(next_node_id), 'x': pt[0], 'y': pt[1]})
            next_node_id += 1
        return nodes_dict[pt]

    def _safe_meta(row) -> dict:
        """Extract non-geometry metadata, dropping NaN scalars gracefully."""
        meta: dict = {}
        for col, val in row.items():
            if col == 'geometry':
                continue
            if isinstance(val, (list, dict)):
                meta[col] = val
                continue
            try:
                if pd.isna(val):
                    continue
            except (TypeError, ValueError):
                pass
            meta[col] = val
        return meta

    def _is_oneway(meta: dict) -> bool:
        val = meta.get('oneway', False)
        if isinstance(val, (list, pd.Series)):
            val = val[0] if len(val) else False
        return str(val).lower() in ('true', 'yes', '1')

    def _add_edge(u: int, v: int, geom: LineString, base_meta: dict) -> None:
        nonlocal next_seg_id
        meta = {**base_meta, 'geometry': geom, 'length': geom.length}
        edge_rows.append({'segment_id': float(next_seg_id),
                          'node_start': float(u), 'node_end': float(v)})
        edge_meta[next_seg_id] = meta
        next_seg_id += 1

    for _, row in simplified_gdf.iterrows():
        geom = row.get('geometry')
        if geom is None or (hasattr(geom, 'is_empty') and geom.is_empty):
            continue

        # Decompose MultiLineString into individual LineStrings
        segments: list[LineString] = (
            list(geom.geoms) if isinstance(geom, MultiLineString) else [geom]
        )

        base_meta = _safe_meta(row)
        oneway    = _is_oneway(base_meta)

        for seg in segments:
            if not hasattr(seg, 'coords') or len(seg.coords) < 2:
                continue
            u = _get_or_add_node(seg.coords[0])
            v = _get_or_add_node(seg.coords[-1])
            _add_edge(u, v, seg, base_meta)
            if not oneway:
                _add_edge(v, u, LineString(list(seg.coords)[::-1]), base_meta)

    net = UrbanNetwork()
    net.nodes_df = (pd.DataFrame(node_rows)
                    if node_rows
                    else pd.DataFrame(columns=['node_id', 'x', 'y']))
    net.edges_df = (pd.DataFrame(edge_rows)
                    if edge_rows
                    else pd.DataFrame(columns=['segment_id', 'node_start', 'node_end']))
    net.edge_metadata = edge_meta
    return net


def _recalibrate_edge_lengths(network: UrbanNetwork) -> UrbanNetwork:
    """Return copy of network with edge 'length' derived from geometry or Euclidean."""
    net = UrbanNetwork()
    net.nodes_df      = network.nodes_df.copy()
    net.edges_df      = network.edges_df.copy()
    net.node_metadata = network.node_metadata.copy()
    net.lineage_nodes = network.lineage_nodes.copy()
    net.lineage_edges = network.lineage_edges.copy()

    node_xy = network.nodes_df.set_index('node_id')[['x', 'y']]
    calibrated: dict = {}
    for _, row in network.edges_df.iterrows():
        sid  = int(row['segment_id'])
        u, v = int(row['node_start']), int(row['node_end'])
        meta = dict(network.edge_metadata.get(sid, {}))
        geom = meta.get('geometry')
        if geom is not None and hasattr(geom, 'length') and geom.length > 0:
            meta['length'] = geom.length
        elif not meta.get('length') or (
                isinstance(meta.get('length'), float) and math.isnan(meta['length'])):
            if u in node_xy.index and v in node_xy.index:
                dx = node_xy.loc[u, 'x'] - node_xy.loc[v, 'x']
                dy = node_xy.loc[u, 'y'] - node_xy.loc[v, 'y']
                meta['length'] = float(np.hypot(dx, dy))
        calibrated[sid] = meta
    net.edge_metadata = calibrated
    return net


# ── Main benchmark ────────────────────────────────────────────────────────────

def run_benchmark(city_name: str = CITY) -> None:
    reporter = ThesisReportGenerator(city_name, output_base=OUTPUT_BASE)

    # ── 1. Raw graph ──────────────────────────────────────────────────────────
    print(f"\n[1] Downloading {city_name} …")
    G_raw_osm = ox.graph_from_place(city_name, network_type="drive", simplify=False)
    G_raw     = ox.project_graph(G_raw_osm)
    raw_net   = UrbanNetwork.from_networkx(G_raw)
    print(f"    raw: {len(raw_net.nodes_df)} nodes, {len(raw_net.edges_df)} edges")

    # ── 2. Simplifications ────────────────────────────────────────────────────
    print("\n[2] Running simplifications …")

    G_osmnx   = ox.simplify_graph(G_raw.copy())
    osmnx_net = UrbanNetwork.from_networkx(G_osmnx)
    print(f"    OSMnx Standard:    {len(osmnx_net.nodes_df):>5} nodes, "
          f"{len(osmnx_net.edges_df):>5} edges")

    acj_net = _run_acj_master(raw_net)
    print(f"    ACJ Topology+DP:   {len(acj_net.nodes_df):>5} nodes, "
          f"{len(acj_net.edges_df):>5} edges")

    # Ordered so results appear in a consistent column order everywhere.
    competitors: dict[str, UrbanNetwork] = {
        "OSMnx Standard":  osmnx_net,
        "ACJ Topology+DP": acj_net,
    }

    # NeatNet — optional; gracefully skipped when not installed or if it errors.
    try:
        _, edges_gdf   = ox.graph_to_gdfs(G_raw)
        simplified_gdf = _neatnet_simplify(edges_gdf)
        neatnet_net    = _neatnet_to_urban_network(simplified_gdf)
        competitors["NeatNet Morfológico"] = neatnet_net
        print(f"    NeatNet Morfológico: {len(neatnet_net.nodes_df):>5} nodes, "
              f"{len(neatnet_net.edges_df):>5} edges")
    except ImportError:
        print("    NeatNet Morfológico: not installed (pip install neatnet) — skipped.")
    except Exception as exc:
        print(f"    NeatNet Morfológico: failed ({type(exc).__name__}: {exc}) — skipped.")

    # ── 3. Evaluate (sin_blindaje — native OSMnx weights) ────────────────────
    print("\n[3] Evaluating SIN BLINDAJE …")
    sin_b: dict = {"Raw OSM": _evaluate(raw_net, raw_net, is_baseline=True)}
    for name, net in competitors.items():
        print(f"    {name} …")
        sin_b[name] = _evaluate(raw_net, net)

    # ── 4. Evaluate (con_blindaje — geometry-recalibrated weights) ───────────
    print("\n[4] Evaluating CON BLINDAJE …")
    raw_cal = _recalibrate_edge_lengths(raw_net)
    con_b: dict = {"Raw OSM": _evaluate(raw_cal, raw_cal, is_baseline=True)}
    for name, net in competitors.items():
        print(f"    {name} …")
        net_cal = _recalibrate_edge_lengths(net)
        result  = _evaluate(raw_cal, net_cal)
        # Reachability is topology-only — borrow from sin_blindaje (no length dependency).
        result["reachability_preservation_%"] = sin_b[name]["reachability_preservation_%"]
        con_b[name] = result

    dual_results = {"sin_blindaje": sin_b, "con_blindaje": con_b}

    # ── 5. Save outputs ───────────────────────────────────────────────────────
    if SAVE_PLOTS:
        try:
            print("\n[5] Saving plots …")
            reporter.save_graph_comparison({"Raw OSM": raw_net, **competitors}, show=False)
            reporter.save_metrics_plots(dual_results, show=False)
        except ImportError as exc:
            print(f"  [!] Plots skipped — {exc}")

    if SAVE_CSV:
        print("\n[5] Saving CSV …")
        reporter.save_metrics_csv(dual_results)

    print("\n✅ Benchmark complete.")


if __name__ == "__main__":
    city = sys.argv[1] if len(sys.argv) > 1 else CITY
    run_benchmark(city)
