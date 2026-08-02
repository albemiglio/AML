#!/usr/bin/env bash
# Phase 4 training pipeline launcher for RunPod.
#
# PRECONDITIONS on the pod:
#   - Repo cloned (or rsynced) to /workspace/AML and you cd'd into it.
#   - WANDB_API_KEY exported (or wandb will run in offline mode).
#
# Usage:
#   export WANDB_API_KEY=...
#   bash setup_runpod.sh
#
# Tunables via env:
#   NUM_WORKERS=12 (default 8) — match pod vCPU count minus 2
#   SKIP_DATASET=1 to skip dataset download (if already present)
#   AUTO_TERMINATE=1 + RUNPOD_API_KEY=... to terminate the pod when done.
#     (RUNPOD_POD_ID is auto-injected on every RunPod container.)

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_ROOT"

# Defaults tuned for RunPod RTX PRO 4500 32GB Blackwell (28 vCPU, 62 GB RAM).
# Override any of these via env before invoking the script.
export NUM_WORKERS="${NUM_WORKERS:-16}"
export BATCH_SIZE="${BATCH_SIZE:-64}"
export LEARNING_RATE="${LEARNING_RATE:-2e-4}"
export EPOCHS="${EPOCHS:-100}"
export PYTHONIOENCODING=utf-8
export PYTHONUTF8=1

PIPELINE_LOG="$PROJECT_ROOT/results_4_pipeline.log"
log() { echo "[$(date +'%H:%M:%S')] $*" | tee -a "$PIPELINE_LOG"; }

terminate_pod_if_requested() {
    local rc=$?
    if [ -z "${AUTO_TERMINATE:-}" ]; then
        log "Pipeline exited (rc=$rc). AUTO_TERMINATE not set, leaving pod running."
        return $rc
    fi
    if [ -z "${RUNPOD_POD_ID:-}" ]; then
        log "AUTO_TERMINATE set but RUNPOD_POD_ID missing — not a RunPod environment."
        return $rc
    fi
    if [ -z "${RUNPOD_API_KEY:-}" ]; then
        log "AUTO_TERMINATE set but RUNPOD_API_KEY missing — cannot call API."
        return $rc
    fi
    local grace_seconds="${AUTO_TERMINATE_GRACE:-3600}"
    log "Pipeline exited (rc=$rc). Will terminate pod $RUNPOD_POD_ID in ${grace_seconds}s (~$(( grace_seconds / 60 ))min)."
    log "Abort with: pkill -f setup_runpod.sh   (or kill the sleep loop)"
    local end_at=$(($(date +%s) + grace_seconds))
    while [ "$(date +%s)" -lt "$end_at" ]; do
        if pgrep -f 'python.*(train|evaluate)\.py' > /dev/null 2>&1; then
            log "Detected running train/eval job — postponing termination by another ${grace_seconds}s."
            end_at=$(($(date +%s) + grace_seconds))
        fi
        sleep 60
    done
    log "Grace period elapsed with no running jobs. Calling podTerminate..."
    curl -s -X POST https://api.runpod.io/graphql \
        -H "Content-Type: application/json" \
        -H "Authorization: Bearer $RUNPOD_API_KEY" \
        -d "{\"query\":\"mutation { podTerminate(input: {podId: \\\"$RUNPOD_POD_ID\\\"}) }\"}" \
        | tee -a "$PIPELINE_LOG"
    log "Termination request sent."
}
trap terminate_pod_if_requested EXIT

# The trap above dies with this script and renews its grace period forever while a
# training process exists, so it cannot save us from a hung job or a hard kill.
# The watchdog runs detached with a hard deadline. Disable with NO_WATCHDOG=1.
if [ -z "${NO_WATCHDOG:-}" ] && [ -n "${RUNPOD_POD_ID:-}" ]; then
    WATCH_LOG="$PIPELINE_LOG" MAX_HOURS="${MAX_HOURS:-8}" \
        nohup bash "$PROJECT_ROOT/tools/pod_watchdog.sh" > "$PROJECT_ROOT/watchdog.log" 2>&1 &
    log "Watchdog started (pid $!) — hard deadline ${MAX_HOURS:-8}h, see watchdog.log"
fi

# ---------- 0. Detect Blackwell, upgrade PyTorch if needed ----------
log "=== 0/5: pytorch compatibility check ==="
NEEDS_TORCH_UPGRADE=$(python -c "
import torch
if not torch.cuda.is_available():
    print('0'); raise SystemExit(0)
cap = torch.cuda.get_device_capability(0)
supported = torch.cuda.get_arch_list()
needed = f'sm_{cap[0]}{cap[1]}'
print('1' if needed not in supported else '0')
" 2>/dev/null || echo "0")
if [ "$NEEDS_TORCH_UPGRADE" = "1" ]; then
    log "GPU compute capability not supported by current torch — upgrading to cu128 (Blackwell)..."
    pip install --quiet --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cu128
    python -c "import torch; caps=torch.cuda.get_arch_list(); cap=torch.cuda.get_device_capability(0); needed=f'sm_{cap[0]}{cap[1]}'; ok = needed in caps; print('torch', torch.__version__, '| device:', torch.cuda.get_device_name(0), '| needed:', needed, '| caps:', caps, '| OK:', ok); raise SystemExit(0 if ok else 3)" | tee -a "$PIPELINE_LOG"
else
    log "Torch GPU support OK, skipping upgrade."
fi

# ---------- 1. Python deps ----------
log "=== 1/5: install python deps ==="
pip install --quiet --upgrade pip
pip install --quiet \
    kornia \
    albumentations \
    trimesh \
    ultralytics \
    wandb \
    pandas \
    pyyaml \
    opencv-python-headless \
    tqdm \
    gdown \
    scikit-learn

# ---------- 2. Dataset + (optionally) weights from team Drive bundle ----------
log "=== 2/5: dataset + weights bundle ==="
# URL della folder Drive del team (override via env). Stesso bundle usato da
# download_data_and_weights.sh — vedi quello script per la struttura attesa.
AML_BUNDLE_URL="${AML_BUNDLE_URL:-https://drive.google.com/drive/folders/<FOLDER_ID>?usp=sharing}"

if [ -z "${SKIP_DATASET:-}" ] && [ ! -d "datasets/linemod/Linemod_preprocessed" ]; then
    if [[ "$AML_BUNDLE_URL" == *"<FOLDER_ID>"* ]]; then
        log "ERROR: AML_BUNDLE_URL non configurato. Esporta la variabile o modifica setup_runpod.sh."
        exit 1
    fi
    pip install --quiet --upgrade gdown
    STAGING_DIR=".aml_bundle_staging"
    rm -rf "$STAGING_DIR" && mkdir -p "$STAGING_DIR"
    log "downloading team Drive bundle (~9 GB dataset + ~0.5 GB weights)..."
    gdown --folder --remaining-ok "$AML_BUNDLE_URL" -O "$STAGING_DIR"

    BUNDLE_ROOT="$(find "$STAGING_DIR" -mindepth 1 -maxdepth 1 -type d | head -n 1)"
    [ -n "$BUNDLE_ROOT" ] || { log "Bundle download failed (no dir in $STAGING_DIR)."; exit 1; }

    DATASET_ZIP="$BUNDLE_ROOT/datasets/linemod/Linemod_preprocessed.zip"
    if [ -f "$DATASET_ZIP" ]; then
        mkdir -p datasets/linemod
        log "extracting dataset..."
        if command -v unzip >/dev/null 2>&1; then
            unzip -q -o "$DATASET_ZIP" -d datasets/linemod/
        else
            log "unzip not available, falling back to python zipfile..."
            python -m zipfile -e "$DATASET_ZIP" datasets/linemod/
        fi
    else
        log "WARN: $DATASET_ZIP non presente nel bundle, skip dataset."
    fi

    if [ -d "$BUNDLE_ROOT/weights" ]; then
        mkdir -p weights
        cp -R "$BUNDLE_ROOT/weights/." weights/
        log "weights copied to weights/:"
        find weights -type f \( -name "*.pth" -o -name "*.pt" \) | tee -a "$PIPELINE_LOG"
    fi

    rm -rf "$STAGING_DIR"
else
    log "dataset dir present, skipping download/extract."
fi

# ---------- 3. Wandb auth ----------
log "=== 3/5: wandb auth ==="
if [ -n "${WANDB_API_KEY:-}" ]; then
    wandb login --relogin "$WANDB_API_KEY" >/dev/null
    log "wandb authenticated."
else
    log "WARNING: WANDB_API_KEY not set — wandb will run in offline mode."
    export WANDB_MODE=offline
fi

# ---------- 4. GPU sanity check ----------
log "=== 4/5: GPU sanity ==="
python -c "import torch; print('CUDA:', torch.cuda.is_available(), '| GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only', '| bf16:', torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False)" | tee -a "$PIPELINE_LOG"

# ---------- 5. Pipeline: train + eval for both variants ----------
mkdir -p results_4_main
MAIN_TRAIN_LOG="$PROJECT_ROOT/results_4_main/pipeline_train.log"
MAIN_EVAL_LOG="$PROJECT_ROOT/results_4_main/pipeline_eval.log"

run_step() {
    local name="$1" module="$2" logfile="$3"
    log ">>> START $name"
    if python -u -m "$module" 2>&1 | tee -a "$logfile"; then
        log "<<< END   $name (OK)"
        return 0
    else
        local rc=${PIPESTATUS[0]}
        log "<<< END   $name (FAIL rc=$rc)"
        return $rc
    fi
}

log "=== 5/5: pipeline (NUM_WORKERS=$NUM_WORKERS) ==="

if run_step "phase4 MAIN train" "phase4_fusion.main.train" "$MAIN_TRAIN_LOG"; then
    run_step "phase4 MAIN eval"  "phase4_fusion.main.evaluate" "$MAIN_EVAL_LOG" || log "MAIN eval failed (continuing)"
else
    log "MAIN train failed — skipping MAIN eval."
fi

log "=== Pipeline finished. ==="
log "Results in: results_4_main/"
log "Download with:  runpodctl receive <pod-id>:/workspace/AML/results_4_main"
