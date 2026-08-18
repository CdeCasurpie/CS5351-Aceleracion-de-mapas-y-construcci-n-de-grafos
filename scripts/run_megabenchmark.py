"""
Multi-tier thesis megabenchmark.

Runs Raw OSM / OSMnx / NeatNet / GeoJAC across increasingly large cities,
unattended, for hours, on constrained hardware (i5-1035G1, 7.55 GiB RAM).
Designed to survive crashes, hangs, and timeouts without losing partial
progress: every (city, algorithm, run) result is appended to
outputs/benchmark_results.csv and fsync'd to disk immediately, and every
run happens in its own subprocess so nothing in this process can hang,
leak, or crash the overall benchmark.

Each algorithm run is delegated to _megabenchmark_worker.py under a hard
subprocess timeout — this is a real kill (SIGKILL), not a try/except, so
it works even if the worker is stuck inside a C extension (CGAL, NeatNet's
documented O(N^3) phases) that never checks for Python-level signals.

Usage:
    # Nix env, not Docker:
    nix develop --command python3 scripts/run_megabenchmark.py --tiers 1
    nix develop --command python3 scripts/run_megabenchmark.py --tiers 1,2,3
    nix develop --command python3 scripts/run_megabenchmark.py --tiers 4
    nix develop --command python3 scripts/run_megabenchmark.py --summarize-only

Resumability: on startup, any (city, algorithm, run_number) already logged
with a *terminal* status (OK, WARNING, TIMEOUT, or SKIPPED_RAM) is skipped.
FAILED and DOWNLOAD_TIMEOUT rows are treated as retry-eligible, not done —
they're usually download/environment issues (network, DNS, Nominatim)
rather than a deterministic property of the (city, algorithm) pair, so a
fresh run prunes those rows from the CSV and re-attempts them
automatically. TIMEOUT, SKIPPED_RAM, and WARNING stay terminal on purpose:
TIMEOUT/SKIPPED_RAM usually reflect a genuine, deterministic resource/
complexity limit (e.g. NeatNet's documented O(N^3) blowup on a huge city)
that a retry won't fix, and WARNING (see _sanity_check_scale) flags a
plausibly-wrong download candidate that needs a human decision, not an
automatic retry loop. To force a retry of a TIMEOUT/SKIPPED_RAM/WARNING/OK
row too, delete it from outputs/benchmark_results.csv manually first.
"""
import argparse
import csv
import os
import subprocess
import sys
import time
from datetime import datetime, timezone

import psutil

# ── path bootstrap (for _city_slug reuse; matches run_benchmark.py) ──────────
_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_SCRIPTS_DIR)
for _p in [os.path.join(_ROOT, "src"), os.path.join(_ROOT, "build")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from geojac.evaluation.reporting import _city_slug  # noqa: E402

WORKER = os.path.join(_SCRIPTS_DIR, "_megabenchmark_worker.py")
OUTPUT_BASE = os.path.join(_ROOT, "outputs")
GRAPH_CACHE_DIR = os.path.join(OUTPUT_BASE, "graph_cache")
RESULTS_CSV = os.path.join(OUTPUT_BASE, "benchmark_results.csv")
SUMMARY_CSV = os.path.join(OUTPUT_BASE, "benchmark_summary.csv")
DOWNLOAD_ERROR_LOG = os.path.join(OUTPUT_BASE, "download_errors.log")

ALGORITHMS = ["Raw OSM", "OSMnx", "GeoJAC", "NeatNet"]
CSV_FIELDS = [
    "city", "tier", "algorithm", "run_number", "status",
    "nodes", "edges", "total_length_km", "avg_sinuosity",
    "keypoint_displacement_m", "reachability_pct",
    "path_error_median", "path_error_p95",
    "wall_time_s", "peak_memory_mb",
    "note",  # diagnostic extra, not in the original schema — see report
]

DEFAULT_TIMEOUT_S = 1800
DEFAULT_REPETITIONS = 3
DEFAULT_RAM_GUARD_GB = 1.5
# Separate, shorter timeout for the graph-download step. 10 min is generous
# even for a large metro area's Overpass query — it should never need the
# full 1800s an algorithm run gets. Kept distinct from DEFAULT_TIMEOUT_S so
# the two can be tuned independently.
#
# NOTE: like the algorithm-run timeout, this is enforced via
# subprocess.run(timeout=...), which does kill()+wait() on expiry. If the
# child is blocked in kernel-level uninterruptible sleep (Linux "D" state —
# e.g. a hung DNS/socket syscall with no request-level timeout set), even
# SIGKILL cannot unblock it until the kernel does, so wait() can still take
# far longer than DOWNLOAD_TIMEOUT_S in that specific failure mode. This
# constant bounds the *nominal* wait and gives that failure mode its own
# CSV status (DOWNLOAD_TIMEOUT); it does not eliminate the D-state hazard
# itself — that would need a socket-level timeout inside osmnx/requests
# (ox.settings.requests_timeout).
DOWNLOAD_TIMEOUT_S = 600

# Metropolitan Lima's bounding box (west, south, east, north; EPSG:4326) —
# last-resort fallback that needs no Nominatim boundary resolution at all.
_LIMA_METRO_BBOX = (-77.2, -12.4, -76.7, -11.7)

# Cercado de Lima district's approximate bounding box (west, south, east,
# north; EPSG:4326) — same last-resort role as _LIMA_METRO_BBOX above.
# Approximate, generous margin around the district (centered near Plaza
# Mayor de Lima, ~-12.0464,-77.0428) — not surveyed to the exact admin
# boundary, same precision tier as _LIMA_METRO_BBOX.
_CERCADO_LIMA_BBOX = (-77.08, -12.09, -76.98, -12.00)

# tier -> [(city_name, tier_label, candidates), ...]
# `candidates` is None for the common case (single place-query download,
# network_type="drive"). When set, it's an ordered list of alternates to
# try if the primary `city_name` fails to resolve/download; each item is
# one of:
#   - a place-query string                      (network_type="drive")
#   - ("place", place_query_string, network_type)  (explicit network_type)
#   - ("osmid", "R<id>", network_type)      (explicit OSM relation, via
#                                             geocode_to_gdf(by_osmid=True)
#                                             + graph_from_polygon)
#   - ("bbox", west, south, east, north)         (network_type="drive")
TIERS: dict[int, list[tuple[str, str, list | None]]] = {
    1: [("Barranco, Lima, Peru", "control", None)],
    2: [
        (
            "Cercado de Lima, Lima, Peru", "medium",
            [
                # Primary: verified correct 2026-08-18. Nominatim's
                # free-text search for "Cercado de Lima" never surfaces
                # the actual admin boundary — every variant tried matched
                # unrelated POIs (a bank, a ministry, "Editora Perú" —
                # a newspaper office at a 0.05km x 0.03km address) whose
                # address happens to contain "Cercado de Lima" as an
                # Urbanización name. Peru's district is officially just
                # named "Lima" (homonymous with the city/province), which
                # is why the place-name search never finds it. Found via
                # a direct Overpass query for admin_level=8 boundaries
                # named "Lima" in the metro bbox, then confirmed with a
                # real graph_from_polygon build: 11,550 nodes / 16,845
                # edges, network_type=drive — sane relative to Barranco
                # (2,053) and Eixample (6,663).
                ("osmid", "R1944756", "drive"),
                # Historical fallbacks below, kept only in case the osmid
                # approach ever breaks (e.g. relation renumbered/merged
                # upstream in OSM) — none of these are known-good:
                # a) original query, network_type="drive" — known-bad,
                #    matches the "Editora Perú" office building, not the
                #    district (see above)
                "Cercado de Lima, Lima, Peru",
                # b) same place, network_type="all" — same known-bad tiny
                #    polygon; confirmed via download_errors.log that this
                #    fails identically regardless of network_type
                ("place", "Cercado de Lima, Lima, Peru", "all"),
                # c) more specific disambiguation string — also matches
                #    only POIs, never the boundary, per direct Nominatim
                #    query during diagnosis
                "Cercado de Lima, Lima Province, Peru",
                # d) last resort: explicit bbox — CONFIRMED WRONG in
                #    practice (resolved 61,473 nodes / 101,436 edges vs
                #    the real ~11,550/~16,845 — pulls in a large chunk of
                #    surrounding Lima, not just the district). Left in
                #    only as an absolute last resort; the scale sanity
                #    check below (_EXPECTED_MAX_NODES) is what catches it
                #    if this candidate is ever actually reached again.
                ("bbox", *_CERCADO_LIMA_BBOX),
            ],
        ),
        ("Eixample, Barcelona, Spain", "medium", None),
    ],
    3: [(
        "Lima Metropolitana, Peru", "large",
        [
            "Lima Metropolitana, Peru",
            "Lima, Peru",
            "Lima Province, Peru",
            ("bbox", *_LIMA_METRO_BBOX),
        ],
    )],
    4: [("Cuauhtémoc, Ciudad de México, Mexico", "extreme", None)],
}


# ── CSV helpers ────────────────────────────────────────────────────────────

# Statuses that represent a real, finished attempt and stay skipped on
# resume. Anything else currently logged (FAILED, DOWNLOAD_TIMEOUT) is
# retry-eligible — see the module docstring's Resumability section for why
# the split lands here specifically. WARNING (see _sanity_check_scale
# below) is terminal like OK: re-attempting it wouldn't fix anything on its
# own — the download candidate itself needs a human look, not a retry loop.
_TERMINAL_STATUSES = {"OK", "TIMEOUT", "SKIPPED_RAM", "WARNING"}

# Rough per-tier node-count ceilings for a lightweight scale sanity check —
# a trip-wire, not a hard limit. A graph whose node count blows way past
# its tier's ceiling almost always means the download resolved to the
# wrong place rather than a legitimately huge (or tiny) city — e.g. the
# 2026-08-18 Cercado de Lima incident, where a bbox fallback silently
# produced 61,473 nodes (should be ~11,550) by pulling in a chunk of
# surrounding Lima instead of just the district. This check is what would
# have caught that automatically instead of requiring manual review.
#
# control/medium are calibrated against real data (Barranco=2,053,
# Eixample=6,663, Cercado de Lima=11,550, all network_type=drive raw
# counts). large/extreme are rough placeholder estimates — as of this
# writing neither Lima Metropolitana nor Cuauhtémoc has ever downloaded
# successfully, so there's no real number to calibrate against yet; expect
# to tighten these once real data exists.
_EXPECTED_MAX_NODES = {
    "control": 10_000,
    "medium": 30_000,
    "large": 2_000_000,
    "extreme": 1_000_000,
}


def _sanity_check_scale(row: dict) -> None:
    """Downgrade an OK row to WARNING (in place) if its node count blows
    past its tier's rough ceiling. Doesn't touch any other field — the
    metrics are left as computed, just flagged for a human to look at
    before trusting them."""
    if row.get("status") != "OK":
        return
    ceiling = _EXPECTED_MAX_NODES.get(row.get("tier"))
    if ceiling is None:
        return
    try:
        nodes = float(row.get("nodes") or 0)
    except (TypeError, ValueError):
        return
    if nodes > ceiling:
        row["status"] = "WARNING"
        extra = (
            f"scale sanity check: {int(nodes)} nodes exceeds the "
            f"{row['tier']!r} tier's ~{ceiling} ceiling — likely resolved "
            f"to the wrong place; review before trusting this row"
        )
        row["note"] = f"{extra} | {row['note']}" if row.get("note") else extra


def _load_completed() -> set[tuple[str, str, int]]:
    """(city, algorithm, run_number) triples with a terminal status.

    As a side effect, rewrites RESULTS_CSV to drop any row whose status is
    NOT terminal (FAILED / DOWNLOAD_TIMEOUT) — those are retry-eligible, and
    pruning them here (rather than just excluding them from the returned
    set) keeps the append-only CSV model correct: without pruning, a retried
    (city, algorithm, run_number) would get a *second* row appended for the
    same key alongside the old failed one, corrupting n_runs/n_ok in
    write_summary(). Pruned rows aren't unrecoverably lost — they're still
    in git history from the last checkpoint commit.
    """
    if not os.path.exists(RESULTS_CSV):
        return set()
    with open(RESULTS_CSV, newline="") as f:
        rows = list(csv.DictReader(f))

    keep = [r for r in rows if r["status"] in _TERMINAL_STATUSES]
    pruned = len(rows) - len(keep)
    if pruned:
        with open(RESULTS_CSV, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            writer.writeheader()
            for r in keep:
                writer.writerow({k: r.get(k, "") for k in CSV_FIELDS})
            f.flush()
            os.fsync(f.fileno())
        print(f"  resumability: pruned {pruned} retry-eligible row(s) "
              f"(FAILED/DOWNLOAD_TIMEOUT) from {RESULTS_CSV}")

    return {(r["city"], r["algorithm"], int(r["run_number"])) for r in keep}


def _append_row(row: dict) -> None:
    """Append one result row and force it to disk before returning."""
    is_new = not os.path.exists(RESULTS_CSV)
    os.makedirs(OUTPUT_BASE, exist_ok=True)
    with open(RESULTS_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if is_new:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in CSV_FIELDS})
        f.flush()
        os.fsync(f.fileno())


# ── RAM guard ──────────────────────────────────────────────────────────────

def _available_gb() -> float:
    return psutil.virtual_memory().available / (1024 ** 3)


def _ram_guard_ok(min_gb: float) -> tuple[bool, float]:
    avail = _available_gb()
    return avail >= min_gb, avail


# ── graph download / cache ────────────────────────────────────────────────

def _log_download_error(label: str, cmd: list[str], detail: str) -> None:
    """Append full diagnostic detail (traceback, full stderr/stdout — not
    just the one-line tail that goes in the CSV note) to DOWNLOAD_ERROR_LOG,
    so a failed candidate never requires manually re-running the download to
    see what actually went wrong."""
    os.makedirs(OUTPUT_BASE, exist_ok=True)
    ts = datetime.now(timezone.utc).isoformat()
    with open(DOWNLOAD_ERROR_LOG, "a") as f:
        f.write(
            f"\n{'=' * 70}\n[{ts}] candidate={label!r}\n"
            f"cmd={' '.join(cmd)}\n{detail}\n"
        )
        f.flush()
        os.fsync(f.fileno())


def _try_one_download(
    cmd: list[str], cache_path: str, timeout_s: int, label: str,
) -> tuple[bool, str, bool]:
    """Returns (ok, message, timed_out) — timed_out is True only when this
    specific attempt hit the subprocess-level wall clock, as opposed to
    failing for any other reason (bad place name, network error, ...)."""
    try:
        proc = subprocess.run(cmd, timeout=timeout_s, capture_output=True, text=True)
    except subprocess.TimeoutExpired as exc:
        # subprocess.run's TimeoutExpired carries whatever stdout/stderr the
        # child had already produced before being SIGKILLed — may be empty
        # if it never got that far (e.g. still resolving DNS), but capture
        # it when present.
        _log_download_error(
            label, cmd,
            f"TIMEOUT after {timeout_s}s\n"
            f"--- partial stderr ---\n{(exc.stderr or '(none captured)').strip()}\n"
            f"--- partial stdout ---\n{(exc.stdout or '(none captured)').strip()}",
        )
        return False, f"TIMEOUT after {timeout_s}s", True

    if proc.returncode != 0 or not os.path.exists(cache_path):
        full_stderr = (proc.stderr or "").strip()
        stderr_tail = full_stderr.splitlines()[-1:] or ["(no stderr)"]
        _log_download_error(
            label, cmd,
            f"rc={proc.returncode}\n"
            f"--- full stderr (includes traceback) ---\n{full_stderr or '(empty)'}\n"
            f"--- full stdout ---\n{(proc.stdout or '').strip() or '(empty)'}",
        )
        return False, f"FAILED (rc={proc.returncode}): {stderr_tail[0]}", False

    ok_msg = proc.stdout.strip().splitlines()[-1] if proc.stdout else "downloaded"
    return True, ok_msg, False


def _ensure_graph_cached(
    city_name: str, timeout_s: int, candidates: list | None = None,
) -> tuple[bool, str, bool]:
    """Download+project+cache a city's graph if not already cached.

    `city_name` names the *canonical* CSV/cache identity regardless of which
    candidate query actually resolves. Each candidate is tried in order,
    each in its own subprocess with its own timeout, so a hung
    Nominatim/Overpass call can never stall the whole benchmark. The first
    candidate to succeed wins; if all fail, every attempt's failure reason
    is reported so it's clear why the city was skipped, and the full detail
    (traceback / full stderr) for each failed attempt is appended to
    DOWNLOAD_ERROR_LOG.

    Each candidate is a place-query string (network_type="drive"),
    ("place", place_query_string, network_type) for an explicit
    network_type, ("osmid", "R<id>", network_type) to resolve an explicit
    OSM relation directly (bypassing Nominatim's free-text ranking
    entirely — use when a place-name search doesn't reliably surface the
    boundary you actually want), or ("bbox", west, south, east, north)
    (network_type "drive").

    Returns (ok, message, any_timed_out) — any_timed_out is True if at least
    one candidate specifically hit the download timeout (as opposed to
    failing outright), so the caller can log a DOWNLOAD_TIMEOUT status
    distinct from a plain FAILED.
    """
    slug = _city_slug(city_name)
    cache_path = os.path.join(GRAPH_CACHE_DIR, f"{slug}.graphml")
    if os.path.exists(cache_path):
        return True, f"cache hit: {cache_path}", False

    attempts = candidates if candidates else [city_name]
    failures = []
    any_timed_out = False
    for candidate in attempts:
        if isinstance(candidate, tuple) and candidate[0] == "bbox":
            _, west, south, east, north = candidate
            # NOTE: "--bbox", "value" as two argv tokens breaks argparse when
            # the value starts with a negative number containing commas (it
            # fails the negative-number heuristic and looks like a stray
            # flag) — use the "--bbox=value" single-token form instead.
            cmd = [
                sys.executable, WORKER, "--mode", "download",
                f"--bbox={west},{south},{east},{north}",
                "--graph-cache", cache_path,
            ]
            label = f"bbox({west},{south},{east},{north})"
        elif isinstance(candidate, tuple) and candidate[0] == "place":
            _, place, network_type = candidate
            cmd = [
                sys.executable, WORKER, "--mode", "download",
                "--city", place, "--network-type", network_type,
                "--graph-cache", cache_path,
            ]
            label = f"{place} (network_type={network_type})"
        elif isinstance(candidate, tuple) and candidate[0] == "osmid":
            _, osmid, network_type = candidate
            cmd = [
                sys.executable, WORKER, "--mode", "download",
                "--osmid", osmid, "--network-type", network_type,
                "--graph-cache", cache_path,
            ]
            label = f"{osmid} (network_type={network_type})"
        else:
            cmd = [
                sys.executable, WORKER, "--mode", "download",
                "--city", candidate, "--graph-cache", cache_path,
            ]
            label = candidate

        print(f"  trying download candidate: {label} …")
        ok, msg, timed_out = _try_one_download(cmd, cache_path, timeout_s, label)
        if ok:
            note = f"resolved via {label!r}: {msg}"
            if len(attempts) > 1:
                note = f"[{len(failures)} earlier candidate(s) failed] " + note
            return True, note, False
        any_timed_out = any_timed_out or timed_out
        failures.append(f"{label!r} -> {msg}")

    joined = " | ".join(failures)
    return False, f"all {len(attempts)} download candidate(s) failed: {joined}", any_timed_out


# ── single algorithm run ──────────────────────────────────────────────────

def _run_one(
    city_name: str, tier_label: str, algorithm: str,
    run_number: int, graph_cache: str, timeout_s: int,
) -> dict:
    result_path = os.path.join(
        GRAPH_CACHE_DIR,
        f".result_{_city_slug(city_name)}_{algorithm}_{run_number}.json",
    )
    if os.path.exists(result_path):
        os.remove(result_path)

    row = {
        "city": city_name, "tier": tier_label, "algorithm": algorithm,
        "run_number": run_number,
    }

    cmd = [
        sys.executable, WORKER,
        "--mode", "run",
        "--graph-cache", graph_cache,
        "--algorithm", algorithm,
        "--result-path", result_path,
    ]

    t0 = time.time()
    try:
        proc = subprocess.run(
            cmd, timeout=timeout_s, capture_output=True, text=True,
        )
        wall = time.time() - t0
    except subprocess.TimeoutExpired:
        wall = time.time() - t0
        row["status"] = "TIMEOUT"
        row["wall_time_s"] = round(wall, 3)
        row["note"] = f"exceeded {timeout_s}s hard timeout"
        if os.path.exists(result_path):
            os.remove(result_path)
        return row

    row["wall_time_s"] = round(wall, 3)

    # Prefer whatever the worker managed to write, even on a non-zero exit —
    # a Python-level exception still leaves a FAILED breadcrumb with a message.
    if os.path.exists(result_path):
        import json
        with open(result_path) as f:
            payload = json.load(f)
        os.remove(result_path)
        row.update(payload)
        row.pop("internal_wall_time_s", None)  # superseded by orchestrator wall_time_s
        return row

    # Worker died before writing anything (segfault, SIGKILL by OOM-killer, ...).
    row["status"] = "FAILED"
    signal_hint = ""
    if proc.returncode < 0:
        signal_hint = f" (killed by signal {-proc.returncode}, likely OOM)"
    stderr_tail = (proc.stderr or "").strip().splitlines()[-1:] or ["(no stderr)"]
    row["note"] = f"rc={proc.returncode}{signal_hint}: {stderr_tail[0]}"
    return row


# ── orchestration ──────────────────────────────────────────────────────────

def run_tiers(
    tier_nums: list[int], timeout_s: int, repetitions: int,
    ram_guard_gb: float, algorithms: list[str], tier4_place: str | None,
) -> None:
    tiers = dict(TIERS)
    if tier4_place:
        tiers[4] = [(tier4_place, "extreme", None)]

    completed = _load_completed()
    os.makedirs(GRAPH_CACHE_DIR, exist_ok=True)

    for tier_num in tier_nums:
        for city_name, tier_label, candidates in tiers.get(tier_num, []):
            print(f"\n{'=' * 70}\n[Tier {tier_num}] {city_name}\n{'=' * 70}")

            ok, avail = _ram_guard_ok(ram_guard_gb)
            if not ok:
                print(f"  RAM guard: only {avail:.2f} GiB free (<{ram_guard_gb} GiB) "
                      f"— skipping entire city.")
                for algorithm in algorithms:
                    for run_number in range(1, repetitions + 1):
                        key = (city_name, algorithm, run_number)
                        if key in completed:
                            continue
                        _append_row({
                            "city": city_name, "tier": tier_label,
                            "algorithm": algorithm, "run_number": run_number,
                            "status": "SKIPPED_RAM",
                            "note": f"{avail:.2f} GiB free at city-level guard",
                        })
                        completed.add(key)
                continue

            slug = _city_slug(city_name)
            graph_cache = os.path.join(GRAPH_CACHE_DIR, f"{slug}.graphml")
            dl_ok, dl_msg, dl_timed_out = _ensure_graph_cached(
                city_name, DOWNLOAD_TIMEOUT_S, candidates,
            )
            print(f"  graph: {dl_msg}")
            if not dl_ok:
                # DOWNLOAD_TIMEOUT distinguishes "at least one candidate hit
                # the download wall clock" from a plain FAILED (bad place
                # name, network refused, ...) — same pattern _run_one uses
                # to distinguish TIMEOUT from FAILED at the algorithm level.
                dl_status = "DOWNLOAD_TIMEOUT" if dl_timed_out else "FAILED"
                for algorithm in algorithms:
                    for run_number in range(1, repetitions + 1):
                        key = (city_name, algorithm, run_number)
                        if key in completed:
                            continue
                        _append_row({
                            "city": city_name, "tier": tier_label,
                            "algorithm": algorithm, "run_number": run_number,
                            "status": dl_status, "note": dl_msg,
                        })
                        completed.add(key)
                continue

            for algorithm in algorithms:
                for run_number in range(1, repetitions + 1):
                    key = (city_name, algorithm, run_number)
                    if key in completed:
                        print(f"  [{algorithm} #{run_number}] already logged — skip")
                        continue

                    ok, avail = _ram_guard_ok(ram_guard_gb)
                    if not ok:
                        print(f"  [{algorithm} #{run_number}] RAM guard: "
                              f"{avail:.2f} GiB free — SKIPPED_RAM")
                        _append_row({
                            "city": city_name, "tier": tier_label,
                            "algorithm": algorithm, "run_number": run_number,
                            "status": "SKIPPED_RAM",
                            "note": f"{avail:.2f} GiB free",
                        })
                        completed.add(key)
                        continue

                    print(f"  [{algorithm} #{run_number}] running "
                          f"(timeout={timeout_s}s)…", end=" ", flush=True)
                    row = _run_one(
                        city_name, tier_label, algorithm,
                        run_number, graph_cache, timeout_s,
                    )
                    _sanity_check_scale(row)
                    _append_row(row)
                    completed.add(key)
                    print(f"{row['status']} ({row.get('wall_time_s', '?')}s, "
                          f"{row.get('peak_memory_mb', '?')} MB)")


# ── summary ──────────────────────────────────────────────────────────────

def write_summary() -> None:
    """Group raw results by (city, tier, algorithm); median the timing
    columns across repetitions; write outputs/benchmark_summary.csv."""
    if not os.path.exists(RESULTS_CSV):
        print("No results CSV yet — nothing to summarize.")
        return

    import statistics
    from collections import defaultdict

    groups: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    with open(RESULTS_CSV, newline="") as f:
        for row in csv.DictReader(f):
            groups[(row["city"], row["tier"], row["algorithm"])].append(row)

    numeric_cols = [
        "nodes", "edges", "total_length_km", "avg_sinuosity",
        "keypoint_displacement_m", "reachability_pct",
        "path_error_median", "path_error_p95",
        "wall_time_s", "peak_memory_mb",
    ]
    summary_fields = ["city", "tier", "algorithm", "n_runs", "n_ok",
                       "status_breakdown"] + [f"median_{c}" for c in numeric_cols]

    with open(SUMMARY_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()
        for (city, tier, algorithm), rows in groups.items():
            statuses = [r["status"] for r in rows]
            ok_rows = [r for r in rows if r["status"] == "OK"]
            out = {
                "city": city, "tier": tier, "algorithm": algorithm,
                "n_runs": len(rows), "n_ok": len(ok_rows),
                "status_breakdown": ",".join(
                    f"{s}:{statuses.count(s)}" for s in sorted(set(statuses))
                ),
            }
            for col in numeric_cols:
                vals = []
                for r in ok_rows:
                    try:
                        vals.append(float(r[col]))
                    except (ValueError, KeyError):
                        pass
                out[f"median_{col}"] = round(statistics.median(vals), 4) if vals else ""
            writer.writerow(out)

    print(f"  saved → {SUMMARY_CSV}")


# ── CLI ────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tiers", default="1",
                     help="Comma-separated tier numbers to run, e.g. 1,2,3")
    ap.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_S)
    ap.add_argument("--repetitions", type=int, default=DEFAULT_REPETITIONS)
    ap.add_argument("--ram-guard-gb", type=float, default=DEFAULT_RAM_GUARD_GB)
    ap.add_argument("--algorithms", default=",".join(ALGORITHMS))
    ap.add_argument("--tier4-place", default=None,
                     help="Override the Tier 4 place string (fallback district)")
    ap.add_argument("--summarize-only", action="store_true",
                     help="Skip all runs; just (re)build benchmark_summary.csv")
    args = ap.parse_args()

    if args.summarize_only:
        write_summary()
        return

    tier_nums = [int(t) for t in args.tiers.split(",") if t.strip()]
    algorithms = [a.strip() for a in args.algorithms.split(",") if a.strip()]

    print(f"Tiers: {tier_nums} | timeout={args.timeout}s | "
          f"repetitions={args.repetitions} | ram_guard={args.ram_guard_gb} GiB")
    run_tiers(
        tier_nums, args.timeout, args.repetitions,
        args.ram_guard_gb, algorithms, args.tier4_place,
    )
    write_summary()
    print("\n✅ Megabenchmark pass complete.")


if __name__ == "__main__":
    main()
