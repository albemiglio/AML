#!/bin/bash

# Interrompi se c'è un errore
set -e

echo "============================================="
echo "   SETUP PIPELINE: LINEMOD, YOLO & WEIGHTS"
echo "============================================="

# 0. Verifica preliminare
if [ ! -f "phase2_detection/prepare_yolo_data.py" ]; then
    echo "❌ ERRORE: Non trovo 'phase2_detection/prepare_yolo_data.py'!"
    exit 1
fi

# 1. Crea la cartella per i datasets
echo "📂 [1/6] Creazione cartella datasets/linemod..."
mkdir -p datasets/linemod

# 2. Scarica i dati del dataset
echo "⬇️  [2/6] Scaricamento dataset..."
if [ ! -f "datasets/linemod/Linemod_preprocessed.zip" ]; then
    gdown --fuzzy "https://drive.google.com/file/d/1qQ8ZjUI6QauzFsiF8EpaaI2nKFWna_kQ/view?usp=sharing" -O datasets/linemod/Linemod_preprocessed.zip
else
    echo "   Archivio dataset già presente, salto."
fi

# 3. Estrazione Dataset
echo "📦 [3/6] Estrazione archivio..."
unzip -q -o datasets/linemod/Linemod_preprocessed.zip -d datasets/linemod/

# 4. Pulizia Dataset
echo "🧹 [4/6] Rimozione file zip temporaneo..."
rm -f datasets/linemod/Linemod_preprocessed.zip

# 5. Download dei Pesi (Weights)
echo "🏋️  [5/6] Scaricamento pesi dei modelli..."

# Definiamo i percorsi delle cartelle pesi
WEIGHTS_ROOT="weights"
mkdir -p "$WEIGHTS_ROOT/fusion_ext/5layer_cnn" \
         "$WEIGHTS_ROOT/fusion_ext/resnet10" \
         "$WEIGHTS_ROOT/fusion_main" \
         "$WEIGHTS_ROOT/baseline" \
         "$WEIGHTS_ROOT/yolo"

echo "   -> Scaricamento pesi in corso..."

# 1. Pesi 5-layer CNN (archived, alternative backbone)
gdown --fuzzy "https://drive.google.com/file/d/1oMGwPRnoMcQx5kUbokZaxkA1cNOybp2u/view?usp=drive_link" -O "$WEIGHTS_ROOT/fusion_ext/5layer_cnn/pose_rgbd_custom_1ch_best.pth" || echo "⚠️  Fallito download CNN"

# 2. Pesi custom ResNet-10 (phase4_fusion/extension)
gdown --fuzzy "https://drive.google.com/file/d/1rBXKMibpuEOz3eX1uIJcWiItgY-lbx1N/view?usp=drive_link" -O "$WEIGHTS_ROOT/fusion_ext/resnet10/pose_rgbd_custom_1ch_best.pth" || echo "⚠️  Fallito download Resnet10"

# 3. Pesi ResNet50+ResNet18 (phase4_fusion/main)
gdown --fuzzy "https://drive.google.com/file/d/1K2uOzuRh4HCnHc0pHMGryWBRcXgQMRXZ/view?usp=drive_link" -O "$WEIGHTS_ROOT/fusion_main/pose_rgbd_fusion_best.pth" || echo "⚠️  Fallito download Resnet50/18"

# 4. Pesi baseline
gdown --fuzzy "https://drive.google.com/file/d/1dvU2vq0fWRbVnVBO0RfM_IEXhQUXLXQE/view?usp=drive_link" -O "$WEIGHTS_ROOT/baseline/pose_resnet50_baseline_best.pth" || echo "⚠️  Fallito download Baseline"

# 5. Pesi Yolo
gdown --fuzzy "https://drive.google.com/file/d/1TWGOZI667ZZIYBzRVLM7Ma02KlGy9JCR/view?usp=drive_link" -O "$WEIGHTS_ROOT/yolo/best.pt" || echo "⚠️  Fallito download YOLO"

# 6. Esecuzione Script Python
echo "⚙️  [6/6] Esecuzione phase2_detection/prepare_yolo_data.py..."
python phase2_detection/prepare_yolo_data.py

echo "============================================="
echo "✅ SETUP COMPLETATO!"
echo "============================================="