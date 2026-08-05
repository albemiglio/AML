#!/usr/bin/env bash
# Independent kill-switch for a RunPod pod.
#
# setup_runpod.sh terminates the pod from an EXIT trap, which does not fire if that
# script is killed or the pod crashes, and which renews its grace period for as long
# as a training process exists — so a *hung* job keeps the pod billing forever.
#
# This runs detached from everything else and enforces a hard deadline: past MAX_HOURS
# the pod is stopped no matter what the training is doing. It also dies early when the work
# is clearly over (no train/eval process) or clearly stuck (log stopped growing).
#
# Usage on the pod, started BEFORE the training:
#   export RUNPOD_API_KEY=...            # RUNPOD_POD_ID is injected by RunPod itself
#   nohup bash tools/pod_watchdog.sh > /workspace/watchdog.log 2>&1 &
#
# Env:
#   MAX_HOURS=8        hard deadline from watchdog start (fractional allowed)
#   IDLE_MINUTES=20    no training process, or no log growth, for this long -> stop the pod
#   WATCH_LOG=path     log whose growth counts as progress (default: results_4_pipeline.log)
#   DRY_RUN=1          log the decision but never actually stop it

set -uo pipefail

MAX_HOURS="${MAX_HOURS:-8}"
IDLE_MINUTES="${IDLE_MINUTES:-20}"
WATCH_LOG="${WATCH_LOG:-$(cd "$(dirname "$0")/.." && pwd)/results_4_pipeline.log}"
# No \.py suffix: jobs are launched as `python -m package.module`, whose cmdline
# carries no ".py" — the old pattern never matched a single real training.
JOB_PATTERN='python.*(train|evaluate)'
POLL_SECONDS=60

log() { echo "[watchdog $(date -u +'%H:%M:%S')] $*"; }

if [ -z "${RUNPOD_POD_ID:-}" ]; then
    log "RUNPOD_POD_ID missing — not a RunPod pod. Nothing to guard, exiting."
    exit 0
fi
if [ -z "${RUNPOD_API_KEY:-}" ] && [ -z "${DRY_RUN:-}" ]; then
    log "FATAL: RUNPOD_API_KEY missing — the pod could NOT be stopped. Refusing to run."
    log "Set it, or re-run with DRY_RUN=1 to test the logic."
    exit 1
fi

# Stop, never terminate. Terminating destroys the pod disk and takes the best
# checkpoint with it — the watchdog would be deleting exactly the work it exists to
# protect. Stopping ends GPU billing, which is the part that costs, and leaves
# /workspace intact so a finished or crashed run stays recoverable.
stop_pod() {
    log "STOPPING pod $RUNPOD_POD_ID — reason: $1"
    if [ -n "${DRY_RUN:-}" ]; then
        log "DRY_RUN set: no API call made."
        exit 0
    fi
    # retry: a stop that silently fails is the whole failure mode we are guarding
    for attempt in 1 2 3 4 5; do
        code=$(curl -s -o /tmp/wd_stop.json -w '%{http_code}' --max-time 30 -X POST \
            "https://rest.runpod.io/v1/pods/$RUNPOD_POD_ID/stop" \
            -H "Authorization: Bearer $RUNPOD_API_KEY")
        log "attempt $attempt -> HTTP $code $(head -c 120 /tmp/wd_stop.json 2>/dev/null)"
        case "$code" in
            200|201|202|204)
                log "Stop accepted. GPU billing ended, /workspace preserved."
                exit 0
                ;;
        esac
        sleep $((attempt * 20))
    done
    log "ALL STOP ATTEMPTS FAILED — pod is still billing, stop it by hand."
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
        stop_pod "hard deadline of ${MAX_HOURS}h reached"
    fi

    # A growing log is the only liveness signal that matters: requiring a matching
    # process too made setup phases (download/unzip) count as idle and killed a
    # healthy session 9 minutes into training. The process check only labels the
    # stop reason below.
    size=$(log_size)
    if [ "$size" != "$last_size" ]; then
        idle_for=0
    else
        idle_for=$(( idle_for + POLL_SECONDS ))
    fi
    last_size=$size

    if [ "$idle_for" -ge "$idle_limit" ]; then
        if pgrep -f "$JOB_PATTERN" > /dev/null 2>&1; then
            stop_pod "training alive but log frozen for ${IDLE_MINUTES}min — hung"
        fi
        stop_pod "no training process for ${IDLE_MINUTES}min — work is done"
    fi

    remaining=$(( (deadline - now) / 60 ))
    log "alive | idle ${idle_for}s/${idle_limit}s | ${remaining}min to deadline"
done
