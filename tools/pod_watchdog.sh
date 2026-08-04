#!/usr/bin/env bash
# Independent kill-switch for a RunPod pod.
#
# setup_runpod.sh terminates the pod from an EXIT trap, which does not fire if that
# script is killed or the pod crashes, and which renews its grace period for as long
# as a training process exists — so a *hung* job keeps the pod billing forever.
#
# This runs detached from everything else and enforces a hard deadline: past MAX_HOURS
# the pod dies no matter what the training is doing. It also dies early when the work
# is clearly over (no train/eval process) or clearly stuck (log stopped growing).
#
# Usage on the pod, started BEFORE the training:
#   export RUNPOD_API_KEY=...            # RUNPOD_POD_ID is injected by RunPod itself
#   nohup bash tools/pod_watchdog.sh > /workspace/watchdog.log 2>&1 &
#
# Env:
#   MAX_HOURS=8        hard deadline from watchdog start (fractional allowed)
#   IDLE_MINUTES=20    no training process, or no log growth, for this long -> terminate
#   WATCH_LOG=path     log whose growth counts as progress (default: results_4_pipeline.log)
#   DRY_RUN=1          log the decision but never actually terminate

set -uo pipefail

MAX_HOURS="${MAX_HOURS:-8}"
IDLE_MINUTES="${IDLE_MINUTES:-20}"
WATCH_LOG="${WATCH_LOG:-$(cd "$(dirname "$0")/.." && pwd)/results_4_pipeline.log}"
JOB_PATTERN='python.*(train|evaluate)\.py'
POLL_SECONDS=60

log() { echo "[watchdog $(date -u +'%H:%M:%S')] $*"; }

if [ -z "${RUNPOD_POD_ID:-}" ]; then
    log "RUNPOD_POD_ID missing — not a RunPod pod. Nothing to guard, exiting."
    exit 0
fi
if [ -z "${RUNPOD_API_KEY:-}" ] && [ -z "${DRY_RUN:-}" ]; then
    log "FATAL: RUNPOD_API_KEY missing — the pod could NOT be terminated. Refusing to run."
    log "Set it, or re-run with DRY_RUN=1 to test the logic."
    exit 1
fi

terminate() {
    log "TERMINATING pod $RUNPOD_POD_ID — reason: $1"
    if [ -n "${DRY_RUN:-}" ]; then
        log "DRY_RUN set: no API call made."
        exit 0
    fi
    # retry: a terminate that silently fails is the whole failure mode we are guarding
    for attempt in 1 2 3 4 5; do
        response=$(curl -s --max-time 30 -X POST https://api.runpod.io/graphql \
            -H "Content-Type: application/json" \
            -H "Authorization: Bearer $RUNPOD_API_KEY" \
            -d "{\"query\":\"mutation { podTerminate(input: {podId: \\\"$RUNPOD_POD_ID\\\"}) }\"}")
        log "attempt $attempt -> ${response:-<empty>}"
        case "$response" in
            *error*|"") sleep $((attempt * 20)) ;;
            *) log "Termination accepted."; exit 0 ;;
        esac
    done
    log "ALL TERMINATION ATTEMPTS FAILED — pod is still billing, kill it by hand."
    exit 1
}

log_size() { [ -f "$WATCH_LOG" ] && wc -c < "$WATCH_LOG" | tr -d ' ' || echo 0; }

# awk, not bc: bc is absent from the RunPod pytorch images, and an empty result here
# would silently make the deadline "now" — the watchdog would kill the pod on its first
# poll. If the budget cannot be computed, refuse to run rather than terminate anything.
budget_seconds=$(awk -v h="$MAX_HOURS" 'BEGIN{printf "%.0f", h*3600}' 2>/dev/null)
case "$budget_seconds" in
    ''|0|*[!0-9]*)
        log "FATAL: MAX_HOURS='$MAX_HOURS' is not a usable number. Exiting without guarding."
        exit 1
        ;;
esac
deadline=$(( $(date +%s) + budget_seconds ))
idle_limit=$(( IDLE_MINUTES * 60 ))
idle_for=0
last_size=$(log_size)

log "guarding pod $RUNPOD_POD_ID | hard deadline in ${MAX_HOURS}h | idle limit ${IDLE_MINUTES}min | log $WATCH_LOG"

while true; do
    sleep "$POLL_SECONDS"

    now=$(date +%s)
    if [ "$now" -ge "$deadline" ]; then
        terminate "hard deadline of ${MAX_HOURS}h reached"
    fi

    size=$(log_size)
    if pgrep -f "$JOB_PATTERN" > /dev/null 2>&1 && [ "$size" != "$last_size" ]; then
        idle_for=0
    else
        idle_for=$(( idle_for + POLL_SECONDS ))
    fi
    last_size=$size

    if [ "$idle_for" -ge "$idle_limit" ]; then
        if pgrep -f "$JOB_PATTERN" > /dev/null 2>&1; then
            terminate "training alive but log frozen for ${IDLE_MINUTES}min — hung"
        fi
        terminate "no training process for ${IDLE_MINUTES}min — work is done"
    fi

    remaining=$(( (deadline - now) / 60 ))
    log "alive | idle ${idle_for}s/${idle_limit}s | ${remaining}min to deadline"
done
