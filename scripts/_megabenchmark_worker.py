"""
Subprocess worker for run_megabenchmark.py.

Runs in complete isolation from the orchestrator process so that a hang
(NeatNet's documented O(N^3) blowup), a crash (CGAL/pybind11 segfault), or
a memory leak in one (city, algorithm) combination can never take down the
orchestrator or contaminate the next run's measurements. The orchestrator
enforces the hard wall-clock timeout externally via subprocess.run(timeout=...),
which SIGKILLs this process if it doesn't finish in time — that works even
if this process is stuck inside a C extension that never checks for Python
signals.

Two modes:
    --mode download   Download+project one city, cache it as .graphml, exit.
    --mode run        Load a cached .graphml, run ONE algorithm ONE time,
                       write metrics as JSON to --result-path.

Usage:
    python3 _megabenchmark_worker.py --mode download \
        --city "Barranco, Lima, Peru" --graph-cache <path>

    python3 _megabenchmark_worker.py --mode run \
        --graph-cache <path> --algorithm GeoJAC --result-path <path>
"""
import argparse
import json
import os
import resource
import sys
import time
import traceback

import numpy as np

# ── path bootstrap: put scripts/ (for run_benchmark.py) and src/ (for geojac,
#    ahead of any stale `pip install`-ed copy) at the front of sys.path ──────
_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_SCRIPTS_DIR)
for _p in [_SCRIPTS_DIR, os.path.join(_ROOT, "src"), os.path.join(_ROOT, "build")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

np.random.seed(42)  # reproducibility for any RNG-dependent metric/algorithm


def _peak_memory_mb() -> float:
    """Peak RSS of this process (and children) since it started, in MB.

    ru_maxrss captures *all* memory this process has touched — including
    CGAL/pybind11 native allocations that tracemalloc (Python-heap-only)
    would miss entirely. On Linux ru_maxrss is reported in KiB.
    """
    kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    kb += resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    return round(kb / 1024, 2)


def _do_download(
    city_name: str, graph_cache: str, bbox: list[float] | None,
    network_type: str = "drive",
) -> None:
    import osmnx as ox

    # HTTP-level (socket) timeout for the Overpass/Nominatim requests this
    # makes internally — bounds the connect+read time of each individual
    # HTTP call, independent of and *underneath* the orchestrator's
    # DOWNLOAD_TIMEOUT_S subprocess timeout. Verified against the installed
    # osmnx==2.1.0 source (.venv/lib/python3.13/site-packages/osmnx/): the
    # setting is `requests_timeout` (NOT `timeout` — osmnx.settings has no
    # such attribute; setting one would silently no-op) and it's threaded
    # straight into `requests.get/post(..., timeout=settings.requests_timeout)`
    # in _overpass.py, _nominatim.py, _http.py, and elevation.py. 180s is
    # already osmnx's own default here — set explicitly so intent doesn't
    # silently depend on that default surviving a future osmnx upgrade.
    #
    # CAVEAT: this bounds the HTTP request/response phase only. It does NOT
    # bound DNS resolution (getaddrinfo()), which happens before the socket
    # timeout applies and is a separate, unbounded blocking libc call in
    # Python's stdlib socket layer. A hang in DNS resolution — plausible
    # kernel "D"-state cause of the earlier 6.5h stall — would not be fixed
    # by this setting. No pure-Python fix exists for that; it would need a
    # bounded/async resolver.
    ox.settings.requests_timeout = 180

    if bbox:
        west, south, east, north = bbox
        G_osm = ox.graph_from_bbox(
            (west, south, east, north), network_type=network_type, simplify=False
        )
        label = f"bbox({west},{south},{east},{north})"
    else:
        G_osm = ox.graph_from_place(city_name, network_type=network_type, simplify=False)
        label = city_name

    G_proj = ox.project_graph(G_osm)
    os.makedirs(os.path.dirname(graph_cache), exist_ok=True)
    ox.save_graphml(G_proj, graph_cache)
    print(
        f"downloaded+cached {label}: "
        f"{len(G_proj.nodes)} nodes, {len(G_proj.edges)} edges -> {graph_cache}"
    )


def _do_run(graph_cache: str, algorithm: str, result_path: str) -> None:
    import osmnx as ox
    import run_benchmark as rb  # reuses PIPELINE / _evaluate / algorithm runners

    from geojac.core.network import UrbanNetwork

    t0 = time.time()
    G_raw = ox.load_graphml(graph_cache)
    raw_net = UrbanNetwork.from_networkx(G_raw)

    if algorithm == "Raw OSM":
        metrics = rb._evaluate(raw_net, raw_net, is_baseline=True)

    elif algorithm == "OSMnx":
        G_osmnx = ox.simplify_graph(G_raw.copy())
        net = UrbanNetwork.from_networkx(G_osmnx)
        metrics = rb._evaluate(raw_net, net)

    elif algorithm == "GeoJAC":
        net = rb._run_geojac_master(raw_net)
        metrics = rb._evaluate(raw_net, net)

    elif algorithm == "NeatNet":
        _, edges_gdf = ox.graph_to_gdfs(G_raw)
        simplified_gdf = rb._neatnet_simplify(edges_gdf)
        net = rb._neatnet_to_urban_network(simplified_gdf)
        metrics = rb._evaluate(raw_net, net)

    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    elapsed = time.time() - t0
    payload = {
        "status": "OK",
        "nodes": metrics["nodes"],
        "edges": metrics["edges"],
        "total_length_km": metrics["total_length_km"],
        "avg_sinuosity": metrics["avg_sinuosity"],
        "keypoint_displacement_m": metrics["keypoint_displacement_m"],
        "reachability_pct": metrics["reachability_preservation_%"],
        "path_error_median": metrics["path_error_abs_median"],
        "path_error_p95": metrics["path_error_abs_p95"],
        "internal_wall_time_s": round(elapsed, 3),
        "peak_memory_mb": _peak_memory_mb(),
        "note": "",
    }
    with open(result_path, "w") as f:
        json.dump(payload, f)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["download", "run"])
    ap.add_argument("--city", help="City name for OSMnx (download mode)")
    ap.add_argument(
        "--bbox", help="west,south,east,north — bbox fallback for download mode"
    )
    ap.add_argument(
        "--network-type", default="drive",
        help="osmnx network_type for download mode (default: drive)",
    )
    ap.add_argument("--graph-cache", required=True, help="Path to .graphml cache")
    ap.add_argument("--algorithm", help="Raw OSM | OSMnx | GeoJAC | NeatNet")
    ap.add_argument("--result-path", help="Where to write JSON metrics (run mode)")
    args = ap.parse_args()

    try:
        if args.mode == "download":
            bbox = None
            if args.bbox:
                bbox = [float(v) for v in args.bbox.split(",")]
                if len(bbox) != 4:
                    raise ValueError("--bbox must be west,south,east,north")
            elif not args.city:
                raise ValueError("--city or --bbox is required for --mode download")
            _do_download(args.city, args.graph_cache, bbox, args.network_type)
        else:
            if not args.algorithm or not args.result_path:
                raise ValueError(
                    "--algorithm and --result-path are required for --mode run"
                )
            _do_run(args.graph_cache, args.algorithm, args.result_path)
        return 0

    except Exception as exc:  # noqa: BLE001 — must never propagate a raw traceback
        # Best-effort: still try to leave a JSON breadcrumb for the orchestrator
        # so a Python-level failure (as opposed to a hard segfault/SIGKILL) is
        # distinguishable and carries an error message in the CSV.
        if args.mode == "run" and args.result_path:
            try:
                with open(args.result_path, "w") as f:
                    json.dump({"status": "FAILED", "note": f"{type(exc).__name__}: {exc}"}, f)
            except OSError:
                pass
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
