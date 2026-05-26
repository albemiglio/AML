#!/usr/bin/env bash
#
# Download dataset + model weights from the team's public Google Drive folder.
#
# Il bundle è una folder Drive pubblica (read-only) gestita dal team, con
# questa struttura interna:
#
#   AML_release/
#   ├── datasets/
#   │   └── linemod/
#   │       └── Linemod_preprocessed.zip
#   └── weights/
#       ├── baseline/pose_resnet50_baseline_best.pth
#       ├── fusion_main/pose_rgbd_fusion_best.pth
#       └── yolo/best.pt
#
# Override the URL via env if you fork the bundle:
#   export AML_BUNDLE_URL="https://drive.google.com/drive/folders/<your-id>"
#

set -euo pipefail

# ---------------------------------------------------------------------------
# CONFIG — sostituire <FOLDER_ID> con l'ID della cartella Drive del team.
# ---------------------------------------------------------------------------
AML_BUNDLE_URL="${AML_BUNDLE_URL:-https://drive.google.com/drive/folders/<FOLDER_ID>?usp=sharing}"

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_ROOT"

echo "============================================="
echo "   SETUP PIPELINE: LINEMOD + WEIGHTS"
echo "============================================="

# 0. Verifica preliminare ----------------------------------------------------
if [ ! -f "phase2_detection/prepare_yolo_data.py" ]; then
    echo "❌ ERRORE: Non trovo 'phase2_detection/prepare_yolo_data.py'!"
    exit 1
fi

if [[ "$AML_BUNDLE_URL" == *"<FOLDER_ID>"* ]]; then
    echo "❌ ERRORE: AML_BUNDLE_URL non configurato."
    echo "   Modifica la variabile in cima a $(basename "$0") oppure esporta:"
    echo "     export AML_BUNDLE_URL=\"https://drive.google.com/drive/folders/<id>?usp=sharing\""
    exit 1
fi

# 1. Assicura gdown aggiornato ----------------------------------------------
echo "🐍 [1/5] Verifica gdown..."
if ! command -v gdown >/dev/null 2>&1; then
    echo "   gdown non trovato, installo..."
    pip install --quiet --upgrade gdown
else
    pip install --quiet --upgrade gdown >/dev/null 2>&1 || true
fi

# 2. Download dell'intera folder Drive --------------------------------------
STAGING_DIR=".aml_bundle_staging"
rm -rf "$STAGING_DIR"
mkdir -p "$STAGING_DIR"

echo "⬇️  [2/5] Download cartella Drive (dataset + pesi)..."
echo "   URL: $AML_BUNDLE_URL"
gdown --folder --remaining-ok "$AML_BUNDLE_URL" -O "$STAGING_DIR"

# gdown --folder crea STAGING_DIR/<nome_cartella_drive>/... — risali a quello.
BUNDLE_ROOT="$(find "$STAGING_DIR" -mindepth 1 -maxdepth 1 -type d | head -n 1)"
if [ -z "$BUNDLE_ROOT" ]; then
    echo "❌ Download fallito: nessuna sotto-cartella in $STAGING_DIR."
    exit 1
fi
echo "   Bundle scaricato in: $BUNDLE_ROOT"

# 3. Estrazione dataset ------------------------------------------------------
DATASET_ZIP="$BUNDLE_ROOT/datasets/linemod/Linemod_preprocessed.zip"
echo "📦 [3/5] Estrazione dataset LineMod..."
if [ -f "$DATASET_ZIP" ]; then
    mkdir -p datasets/linemod
    if command -v unzip >/dev/null 2>&1; then
        unzip -q -o "$DATASET_ZIP" -d datasets/linemod/
    else
        echo "   unzip non disponibile, uso python zipfile..."
        python -m zipfile -e "$DATASET_ZIP" datasets/linemod/
    fi
    echo "   ✓ Estratto in datasets/linemod/"
else
    echo "   ⚠️  $DATASET_ZIP non trovato nella folder Drive — skip dataset."
fi

# 4. Copia weights nel layout del progetto ----------------------------------
echo "🏋️  [4/5] Installazione pesi modelli..."
if [ -d "$BUNDLE_ROOT/weights" ]; then
    mkdir -p weights
    # cp -R src/. dst/  copia il CONTENUTO di src dentro dst (no nesting extra)
    cp -R "$BUNDLE_ROOT/weights/." weights/
    echo "   ✓ Pesi copiati in weights/:"
    find weights -type f \( -name "*.pth" -o -name "*.pt" \) | sed 's/^/     /'
else
    echo "   ⚠️  $BUNDLE_ROOT/weights non trovato — skip pesi."
fi

# 5. Pulizia + prepare YOLO data --------------------------------------------
echo "🧹 [5/5] Cleanup staging + prepare YOLO data..."
rm -rf "$STAGING_DIR"
python phase2_detection/prepare_yolo_data.py

echo "============================================="
echo "✅ SETUP COMPLETATO!"
echo "============================================="
