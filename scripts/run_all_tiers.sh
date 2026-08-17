#!/usr/bin/env bash
# Unattended multi-tier thesis megabenchmark runner.
#
# Runs Tiers 1, 2, 3, then 4 sequentially (each as its own
# run_megabenchmark.py invocation so resumability/checkpointing applies
# within a tier too), committing outputs/benchmark_{results,summary}.csv
# to `develop` after each tier completes. Never pushes — local commits
# only. On an orchestrator-level failure (not a single algorithm run —
# those are caught internally as FAILED/TIMEOUT/SKIPPED_RAM CSV rows)
# it stops immediately, writes outputs/CRASH_LOG.txt, and does not retry.
#
# Intended to be launched detached (nohup / backgrounded) so it survives
# terminal disconnection; expected total runtime: hours.
set -uo pipefail  # no -e: failures are handled explicitly below

cd "$(dirname "${BASH_SOURCE[0]}")/.."
REPO_ROOT="$(pwd)"
LOG="outputs/megabenchmark_run.log"
CRASH_LOG="outputs/CRASH_LOG.txt"
mkdir -p outputs

log() { echo "[$(date -Is)] $*" | tee -a "$LOG"; }

crash() {
    log "UNRECOVERABLE ERROR: $*"
    {
        echo "=== CRASH at $(date -Is) ==="
        echo "$*"
        echo
        echo "--- last 150 lines of $LOG ---"
        tail -150 "$LOG" 2>/dev/null
    } > "$CRASH_LOG"
    exit 1
}

check_disk() {
    local avail_kb
    avail_kb=$(df -Pk "$REPO_ROOT" | tail -1 | awk '{print $4}')
    if [ -z "$avail_kb" ] || [ "$avail_kb" -lt 1048576 ]; then
        crash "less than 1 GiB free disk space (df reported ${avail_kb:-unknown} KiB) before tier $1"
    fi
}

commit_progress() {
    local msg="$1"
    # outputs/ is gitignored on purpose (large/binary graph_cache, per-city
    # plot dirs) — force-add ONLY the two small result CSVs as an extra
    # durability layer on top of the fsync'd-to-disk checkpointing that
    # run_megabenchmark.py already does per row.
    #
    # IMPORTANT: `git add -f a b` is atomic across its pathspec list — if
    # EITHER file doesn't exist yet, the whole add fails (exit 128) and
    # NEITHER gets staged. Only pass paths that actually exist, or a
    # missing summary.csv would silently skip committing an existing
    # results.csv too.
    local files=()
    [ -f outputs/benchmark_results.csv ] && files+=(outputs/benchmark_results.csv)
    [ -f outputs/benchmark_summary.csv ] && files+=(outputs/benchmark_summary.csv)
    if [ ${#files[@]} -eq 0 ]; then
        log "commit_progress: no result files exist yet ($msg)"
        return
    fi

    if ! git add -f "${files[@]}" 2>>"$LOG"; then
        log "WARN: git add -f failed for '$msg' — continuing anyway (results are still on disk)"
        return
    fi
    if git diff --cached --quiet; then
        log "commit_progress: no changes to commit ($msg)"
    else
        if git commit -m "$msg" >>"$LOG" 2>&1; then
            log "committed: $msg"
        else
            log "WARN: git commit failed for '$msg' — continuing anyway (results are still on disk)"
        fi
    fi
}

run_tier() {
    local tier="$1"
    check_disk "$tier"
    log "=== Tier $tier starting ==="
    if nix develop --command python3 scripts/run_megabenchmark.py --tiers "$tier" >>"$LOG" 2>&1; then
        log "=== Tier $tier finished (exit 0) ==="
    else
        local rc=$?
        # A non-zero exit here means run_megabenchmark.py ITSELF crashed —
        # every per-run algorithm failure/timeout/RAM-skip is already caught
        # internally and logged as a CSV row with exit 0. This is the
        # "something broke the whole script" case.
        crash "run_megabenchmark.py --tiers $tier exited with code $rc (orchestrator-level failure, not a single run failure)"
    fi
    commit_progress "chore(benchmark): checkpoint after Tier $tier"
}

# Testing hook: `MEGABENCH_SOURCE_ONLY=1 bash -c 'source run_all_tiers.sh'`
# defines log/crash/commit_progress/run_tier without executing the tiers —
# lets tests exercise the real functions instead of a re-typed copy of them.
if [ "${MEGABENCH_SOURCE_ONLY:-0}" = "1" ]; then
    return 0 2>/dev/null || exit 0
fi

command -v nix >/dev/null 2>&1 || crash "nix command not found on PATH"

log "########## megabenchmark unattended run starting (pid $$) ##########"

run_tier 1
run_tier 2
run_tier 3

log "=== Tiers 1-3 complete — auto-proceeding to Tier 4 (pre-approved, no further confirmation needed) ==="
run_tier 4

log "########## ALL TIERS COMPLETE — nothing else will run, stopping ##########"
