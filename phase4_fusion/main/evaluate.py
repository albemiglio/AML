import torch
import numpy as np
import cv2
import os
import yaml
import trimesh
import pandas as pd
from ultralytics import YOLO
from tqdm import tqdm

from phase4_fusion.main.model import RGBD_FusionPredictor
from common.data_split import prepare_data_and_splits
from phase4_fusion.main.rgbd_utils import (
    load_info_cache,
    fetch_sample_info,
    convert_depth_to_meters,
    square_crop_coords,
    prepare_rgb_tensor,
    prepare_depth_tensor,
    build_meta_tensor,
)
from common.yolo_metadata import select_detection_for_object, get_object_metadata
from common.pose_metrics import pose_error

def get_annotation(gt_cache, obj_id, sample_id):
    ann_list = gt_cache[obj_id][sample_id]
    return next((ann for ann in ann_list if ann['obj_id'] == obj_id), ann_list[0])


def generate_fusion_report(test_samples, gt_cache, info_cache, model, yolo_model, models_info, ROOT_DATASET, DEVICE):
    class_names = {
        1: "ape", 2: "benchvise", 3: "bowl", 4: "camera", 5: "can",
        6: "cat", 7: "cup", 8: "driller", 9: "duck", 10: "eggbox",
        11: "glue", 12: "holepuncher", 13: "iron", 14: "lamp", 15: "phone"
    }
    # Load point clouds for ADD metric
    ply_cache = {}
    for oid in class_names.keys():
        ply_path = os.path.join(ROOT_DATASET, 'models', f'obj_{oid:02d}.ply')
        if os.path.exists(ply_path):
            ply_cache[oid] = trimesh.load(ply_path).sample(2000)
    
    results = []
    model.eval()
    detection_misses = 0
    missing_info = 0
    processed = 0

    print("\nEvaluating RGB-D Fusion model...")
    with torch.no_grad():
        for obj_id, sample_id in tqdm(test_samples):
            if obj_id not in class_names:
                continue

            if obj_id not in ply_cache:
                continue

            ann = get_annotation(gt_cache, obj_id, sample_id)
            info_entry = fetch_sample_info(info_cache, obj_id, sample_id)
            if info_entry is None:
                missing_info += 1
                continue

            model_info = get_object_metadata(models_info, obj_id)
            if model_info is None:
                missing_info += 1
                continue

            img_path = os.path.join(
                ROOT_DATASET, 'data', f"{obj_id:02d}", 'rgb', f"{sample_id:04d}.png"
            )
            depth_path = os.path.join(
                ROOT_DATASET, 'data', f"{obj_id:02d}", 'depth', f"{sample_id:04d}.png"
            )

            img_bgr = cv2.imread(img_path)
            depth_raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            if img_bgr is None or depth_raw is None:
                missing_info += 1
                continue

            depth_meters = convert_depth_to_meters(depth_raw, info_entry.get('depth_scale', 1.0))
            cam_K = np.array(info_entry['cam_K'], dtype=np.float32).reshape(3, 3)

            result = yolo_model(img_bgr, verbose=False)[0]
            box = select_detection_for_object(result, obj_id)
            if box is None:
                detection_misses += 1
                continue

            xyxy = box.xyxy.cpu().numpy()[0]
            yolo_bbox = [
                float(xyxy[0]), float(xyxy[1]),
                float(xyxy[2] - xyxy[0]), float(xyxy[3] - xyxy[1])
            ]

            crop_coords = square_crop_coords(yolo_bbox, img_bgr.shape)
            if crop_coords is None:
                detection_misses += 1
                continue

            rgb_tensor = prepare_rgb_tensor(img_bgr, crop_coords)
            depth_tensor = prepare_depth_tensor(depth_meters, crop_coords)
            meta_tensor = build_meta_tensor(yolo_bbox, cam_K, img_bgr.shape)

            if rgb_tensor is None or depth_tensor is None or meta_tensor is None:
                detection_misses += 1
                continue

            rgb = rgb_tensor.to(DEVICE)
            depth = depth_tensor.to(DEVICE)
            meta = meta_tensor.to(DEVICE)

            pred_T, pred_R = model(rgb, depth, meta)
            R_pred = pred_R[0].cpu().numpy()         # (3, 3) from SO(3)
            T_pred = pred_T[0].cpu().numpy() * 1000.0  # metres -> mm

            R_gt = np.array(ann['cam_R_m2c'], dtype=np.float32).reshape(3, 3)
            T_gt = np.array(ann['cam_t_m2c'], dtype=np.float32)
            pts = ply_cache[obj_id]
            diameter = model_info['diameter']
            threshold = 0.10 * diameter

            add_score = pose_error(pts, R_gt, T_gt, R_pred, T_pred, obj_id)
            results.append({
                "Classe": class_names[obj_id],
                "ADD_mm": add_score,
                "Success": add_score < threshold
            })
            processed += 1

    df = pd.DataFrame(results)
    if df.empty:
        print("No samples evaluated. Check YOLO detections or missing data.")
        print(f"Samples skipped due to missing info: {missing_info}")
        print(f"Samples skipped due to YOLO/crop: {detection_misses}")
        return

    summary = df.groupby("Classe").agg(
        Media_ADD=("ADD_mm", "mean"),
        Accuracy=("Success", lambda x: x.mean() * 100)
    ).reset_index()

    print("\n" + "="*60)
    print(f"{'CLASSE':<15} | {'ADD/ADD-S (mm)':<15} | {'ACCURACY (%)':<12}")
    print("-" * 60)
    for _, r in summary.iterrows():
        sym = "*" if r['Classe'] in ("eggbox", "glue") else " "
        print(f"{r['Classe']:<14}{sym}| {r['Media_ADD']:<15.2f} | {r['Accuracy']:<12.1f}")
    print("="*60)
    print("* ADD-S metric (symmetric objects)")
    print(f"GLOBAL AVERAGE -> Accuracy (ADD/ADD-S < 0.1d): {summary['Accuracy'].mean():.1f}%")
    total_samples = len([item for item in test_samples if item[0] in class_names])
    print(f"Samples processed: {processed}/{total_samples}")
    print(f"Samples skipped (missing info): {missing_info}")
    print(f"Samples skipped (YOLO/crop): {detection_misses}")

    os.makedirs("results", exist_ok=True)
    out_csv = "results/phase4_main_perclass.csv"
    summary.to_csv(out_csv, index=False, float_format="%.2f")
    print(f"\n[saved] per-class summary -> {out_csv}")


def main():
    ROOT_DATASET = "datasets/linemod/Linemod_preprocessed"
    MODEL_PATH = "results_4_main/pose_rgbd_fusion_best.pth"
    YOLO_PATH = 'weights/yolo/best.pt'
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    with open(os.path.join(ROOT_DATASET, 'models', 'models_info.yml'), 'r') as f:
        models_info = yaml.safe_load(f)

    _, _, test_samples, gt_cache = prepare_data_and_splits(ROOT_DATASET)
    info_cache = load_info_cache(ROOT_DATASET, sorted(gt_cache.keys()))
    
    model = RGBD_FusionPredictor().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    yolo_model = YOLO(YOLO_PATH)

    generate_fusion_report(test_samples, gt_cache, info_cache, model, yolo_model, models_info, ROOT_DATASET, DEVICE)

if __name__ == "__main__":
    main()