#!/usr/bin/env python3
"""RunPod sentinel — runs on the VPS, independent of any laptop or coding session.

Third and last line of defence: the EXIT trap dies with its script, the on-pod watchdog
only exists on pods launched through setup_runpod.sh. This one sees every pod, including
ones started by hand from the dashboard.

It reads the expected runtime from the pod NAME, so it knows what "too long" means for
that specific job instead of applying one blind threshold to everything:

    name convention:  <anything>-eta<N>h        e.g.  aml-phase4-eta3h, aml-calib-eta0.5h

    age <   eta   ->  silent, this is normal
    age >=  eta   ->  warn (job overran its budget)
    age >= 2*eta  ->  stop (never terminate: that would destroy the checkpoints)
    no eta tag    ->  DEFAULT_ETA_H is assumed, and the missing tag is reported

Env:
    RUNPOD_API_KEY   required (or put it in /etc/runpod-sentinel.key, mode 600)
    DISCORD_WEBHOOK  optional; warnings and kills are posted there
    DEFAULT_ETA_H    fallback budget for untagged pods (default 2)
    DRY_RUN=1        report what it would do, stop nothing
"""

import json
import os
import re
import sys
import urllib.request
from datetime import datetime, timezone

API = "https://rest.runpod.io/v1/pods"
GRAPHQL = "https://api.runpod.io/graphql"
DEFAULT_ETA_H = float(os.environ.get("DEFAULT_ETA_H", "2"))
DRY_RUN = bool(os.environ.get("DRY_RUN"))
KEY_FILE = "/etc/runpod-sentinel.key"
ETA_RE = re.compile(r"eta([0-9]+(?:\.[0-9]+)?)h", re.I)
# lastStartedAt is what we want (a restarted pod bills from then); the rest are fallbacks
AGE_FIELDS = ("lastStartedAt", "startedAt", "createdAt", "createdOn")


def api_key():
    key = os.environ.get("RUNPOD_API_KEY")
    if not key and os.path.exists(KEY_FILE):
        with open(KEY_FILE) as f:
            key = f.read().strip()
    if not key:
        sys.exit("RUNPOD_API_KEY not set and %s missing" % KEY_FILE)
    return key


def get_json(url, key):
    req = urllib.request.Request(url, headers={"Authorization": "Bearer " + key})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode())


def notify(text):
    print(text, flush=True)
    hook = os.environ.get("DISCORD_WEBHOOK")
    if not hook:
        return
    body = json.dumps({"content": text}).encode()
    req = urllib.request.Request(hook, data=body, headers={"Content-Type": "application/json"})
    try:
        urllib.request.urlopen(req, timeout=20).read()
    except Exception as exc:  # a failed notification must never stop a termination
        print("  (discord notify failed: %s)" % exc, flush=True)


def parse_time(raw):
    """RunPod does not return ISO 8601. It returns Go's time.Time.String(), e.g.
    '2026-08-04 11:01:54.506 +0000 UTC'. Normalise that (and plain ISO) or give up."""
    text = str(raw).strip().replace("Z", "+00:00")
    text = re.sub(r"\s+[A-Z]{2,5}$", "", text)          # drop trailing zone name (UTC, CEST)
    text = re.sub(r"\s+([+-]\d{2}:?\d{2})$", r"\1", text)  # glue the offset to the time
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def age_hours(pod, now):
    for field in AGE_FIELDS:
        raw = pod.get(field)
        if not raw:
            continue
        started = parse_time(raw)
        if started is None:
            continue
        if started.tzinfo is None:
            started = started.replace(tzinfo=timezone.utc)
        return (now - started).total_seconds() / 3600.0
    return None


def budget_hours(name):
    match = ETA_RE.search(name or "")
    return (float(match.group(1)), True) if match else (DEFAULT_ETA_H, False)


def stop_pod(pod_id, key):
    """Stop, never terminate: terminating destroys the pod disk and every checkpoint
    on it. Stopping ends the GPU billing, which is what a runaway pod actually costs."""
    req = urllib.request.Request(
        "%s/%s/stop" % (API, pod_id),
        data=b"",
        headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        return "HTTP %s" % r.status


def classify(pod, now):
    """-> (level, message). level in {ok, warn, kill, unknown}."""
    name = pod.get("name") or pod.get("id") or "<unnamed>"
    age = age_hours(pod, now)
    eta, tagged = budget_hours(name)
    cost = pod.get("costPerHr") or 0

    if age is None:
        return "unknown", "%s: cannot determine age (fields: %s)" % (name, sorted(pod))

    spent = age * float(cost or 0)
    tag = "" if tagged else " [no eta tag in name, assumed %.2gh]" % eta
    detail = "%s: up %.1fh vs budget %.2gh, ~$%.2f spent%s" % (name, age, eta, spent, tag)

    if age >= 2 * eta:
        return "kill", detail
    if age >= eta:
        return "warn", detail
    return "ok", detail


def main():
    now = datetime.now(timezone.utc)
    key = api_key()
    data = get_json(API, key)
    pods = data if isinstance(data, list) else data.get("data", [])
    running = [p for p in pods if str(p.get("desiredStatus", "RUNNING")).upper() == "RUNNING"]

    if not running:
        print("no running pods")
        return

    for pod in running:
        level, detail = classify(pod, now)
        if level == "ok":
            print("OK   " + detail)
        elif level == "unknown":
            notify("RunPod sentinel: " + detail)
        elif level == "warn":
            notify("RunPod: pod over budget — " + detail)
        else:
            notify("RunPod: STOPPING pod over 2x budget — " + detail)
            if DRY_RUN:
                print("  DRY_RUN: not stopped")
                continue
            try:
                print("  stop -> " + stop_pod(pod["id"], key))
            except Exception as exc:
                notify("RunPod: STOP FAILED for %s (%s) — stop it by hand" % (detail, exc))


def self_test():
    now = datetime.now(timezone.utc)

    def pod(name, hours_up):
        started = now.timestamp() - hours_up * 3600
        return {
            "id": "x",
            "name": name,
            "costPerHr": 0.34,
            "lastStartedAt": datetime.fromtimestamp(started, timezone.utc).isoformat(),
        }

    assert classify(pod("aml-phase4-eta3h", 1.0), now)[0] == "ok"
    assert classify(pod("aml-phase4-eta3h", 3.5), now)[0] == "warn"
    assert classify(pod("aml-phase4-eta3h", 6.5), now)[0] == "kill"
    assert classify(pod("aml-calib-eta0.5h", 0.2), now)[0] == "ok"
    assert classify(pod("aml-calib-eta0.5h", 1.2), now)[0] == "kill"
    # untagged pod falls back to DEFAULT_ETA_H (2h) and is still guarded
    assert classify(pod("some-pod", 1.0), now)[0] == "ok"
    assert classify(pod("some-pod", 5.0), now)[0] == "kill"
    assert "no eta tag" in classify(pod("some-pod", 1.0), now)[1]
    assert classify({"id": "y", "name": "n"}, now)[0] == "unknown"

    # the format RunPod actually returns (Go time.Time.String), not ISO 8601 —
    # this is what silently made every real pod unreadable
    assert parse_time("2026-08-04 11:01:54.506 +0000 UTC") is not None
    assert parse_time("2026-08-04T11:01:54.506Z") is not None
    assert parse_time("2026-08-04 13:01:54 +0200 CEST") is not None
    assert parse_time("not a date") is None
    go_pod = {"id": "z", "name": "aml-calib-eta1h", "costPerHr": 0.22,
              "lastStartedAt": (now - __import__("datetime").timedelta(hours=3))
              .strftime("%Y-%m-%d %H:%M:%S.%f +0000 UTC")}
    assert classify(go_pod, now)[0] == "kill", "pod Go-formattato oltre 2x budget deve essere fermato"
    print("self-test OK")


if __name__ == "__main__":
    self_test() if "--self-test" in sys.argv else main()
