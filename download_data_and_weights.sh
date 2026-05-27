#!/usr/bin/env bash
#
# Download dataset + model weights from the team's public Google Drive folders.
#
# Layout (two separate public folders, owner: albertomigliorato@gmail.com):
#
#   aml-2026-linemod-dataset/
#   └── Linemod_preprocessed.zip          # ~8.4 GB raw images + GT
#
#   aml-2026-pose-weights/
#   ├── baseline/pose_resnet50_baseline_best.pth    # Phase 3 (may be absent until retrained)
#   ├── fusion_main/pose_rgbd_fusion_best.pth       # Phase 4 main (~145 MB)
#   └── yolo/best.pt                                # YOLO11n fine-tuned (~5 MB)
#
# Override URLs via env if you fork the bundles.
#

set -euo pipefail

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
AML_DATASET_URL="${AML_DATASET_URL:-https://drive.google.com/drive/folders/1H1OnlAOX9_C1TeYiNEcrdq-BL-Ka4Cxc?usp=sharing}"
AML_WEIGHTS_URL="${AML_WEIGHTS_URL:-https://drive.google.com/drive/folders/1RzuBBxHeq-yfW2AAU7n60WEzEEhdkSyL?usp=sharing}"

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_ROOT"

echo "============================================="
echo "   SETUP PIPELINE: LINEMOD + WEIGHTS"
echo "============================================="

# 0. Preflight
if [ ! -f "phase2_detection/prepare_yolo_data.py" ]; then
    echo "ERROR: cannot find 'phase2_detection/prepare_yolo_data.py' — wrong working dir?"
    exit 1
fi

# 1. gdown
echo "[1/5] gdown..."
if ! command -v gdown >/dev/null 2>&1; then
    pip install --quiet --upgrade gdown
else
    pip install --quiet --upgrade gdown >/dev/null 2>&1 || true
fi

STAGING_DIR=".aml_bundle_staging"
rm -rf "$STAGING_DIR"
mkdir -p "$STAGING_DIR/dataset" "$STAGING_DIR/weights"

# 2. Dataset
echo "[2/5] Downloading dataset folder..."
echo "      $AML_DATASET_URL"
gdown --folder --remaining-ok "$AML_DATASET_URL" -O "$STAGING_DIR/dataset"

DATASET_ROOT="$(find "$STAGING_DIR/dataset" -mindepth 1 -maxdepth 1 -type d | head -n 1)"
if [ -z "$DATASET_ROOT" ]; then
    echo "ERROR: dataset download produced no folder in $STAGING_DIR/dataset."
    exit 1
fi

DATASET_ZIP="$(find "$DATASET_ROOT" -maxdepth 2 -name "Linemod_preprocessed.zip" | head -n 1)"
echo "[3/5] Extracting dataset..."
if [ -n "$DATASET_ZIP" ] && [ -f "$DATASET_ZIP" ]; then
    mkdir -p datasets/linemod
    if command -v unzip >/dev/null 2>&1; then
        unzip -q -o "$DATASET_ZIP" -d datasets/linemod/
    else
        python -m zipfile -e "$DATASET_ZIP" datasets/linemod/
    fi
    echo "      extracted to datasets/linemod/"
else
    echo "WARNING: Linemod_preprocessed.zip not found in dataset folder — skip."
fi

# 3. Weights
echo "[4/5] Downloading weights folder..."
echo "      $AML_WEIGHTS_URL"
gdown --folder --remaining-ok "$AML_WEIGHTS_URL" -O "$STAGING_DIR/weights"

WEIGHTS_ROOT="$(find "$STAGING_DIR/weights" -mindepth 1 -maxdepth 1 -type d | head -n 1)"
if [ -n "$WEIGHTS_ROOT" ] && [ -d "$WEIGHTS_ROOT" ]; then
    mkdir -p weights
    cp -R "$WEIGHTS_ROOT/." weights/
    echo "      installed weights:"
    find weights -type f \( -name "*.pth" -o -name "*.pt" \) | sed 's/^/        /'
else
    echo "WARNING: weights download produced nothing — skip."
fi

# 4. Cleanup + YOLO prep
echo "[5/5] Cleanup + prepare_yolo_data..."
rm -rf "$STAGING_DIR"
python phase2_detection/prepare_yolo_data.py

echo "============================================="
echo "DONE."
echo "============================================="
