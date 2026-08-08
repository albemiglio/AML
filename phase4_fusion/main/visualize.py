import torch
import numpy as np
import cv2
import os
import yaml
import random
import argparse
import trimesh
from ultralytics import YOLO

from phase4_fusion.main.model import RGBD_FusionPredictor
from phase3_baseline.dataset import LineModDataset
from phase4_fusion.main.dataset import LineModDatasetRGBD
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

def get_all_models_info(root_path):
    """Load global models_info.yml database from models/."""
    info_path = os.path.join(root_path, 'models', 'models_info.yml')
    with open(info_path, 'r') as f:
        return yaml.safe_load(f)

def project_3d_box(img, R, T, K, obj_info, color=(0, 255, 0), thickness=2):
    """Project 3D box using real min and size data."""
    min_x, max_x = obj_info['min_x'], obj_info['min_x'] + obj_info['size_x']
    min_y, max_y = obj_info['min_y'], obj_info['min_y'] + obj_info['size_y']
    min_z, max_z = obj_info['min_z'], obj_info['min_z'] + obj_info['size_z']

    pts = np.array([
        [min_x, min_y, min_z], [min_x, min_y, max_z],
        [min_x, max_y, min_z], [min_x, max_y, max_z],
        [max_x, min_y, min_z], [max_x, min_y, max_z],
        [max_x, max_y, min_z], [max_x, max_y, max_z]
    ], dtype=np.float32)

    if R.shape == (3, 3):
        rvec, _ = cv2.Rodrigues(R)
    else:
        rvec = R
    T_vec = T.reshape(3, 1)

    pts_2d, _ = cv2.projectPoints(pts, rvec, T_vec, K, None)
    pts_2d = pts_2d.reshape(-1, 2).astype(int)

    edges = [(0,1), (0,2), (1,3), (2,3), (4,5), (4,6), (5,7), (6,7), (0,4), (1,5), (2,6), (3,7)]
    for i, j in edges:
        cv2.line(img, tuple(pts_2d[i]), tuple(pts_2d[j]), color, thickness)
    return img

def main():
    parser = argparse.ArgumentParser(description="Visualize Phase 4 RGB-D Fusion predictions.")
    parser.add_argument("--save-dir", type=str, default=None,
                        help="If set, save annotated frames as PNG to this directory "
                             "(no interactive window).")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Stop after this many samples (default: process all).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for sample shuffling (for reproducible figure picks).")
    args = parser.parse_args()

    ROOT_DATASET = "datasets/linemod/Linemod_preprocessed"
    DEPTH_MODE = os.environ.get("DEPTH_MODE", "norm")
    assert DEPTH_MODE in ("raw", "norm"), "visualize supporta raw|norm"
    RGBD_MODEL_PATH = os.environ.get("CKPT", f"results_4_main/pose_rgbd_fusion_best_{DEPTH_MODE}.pth")
    YOLO_PATH = os.environ.get("YOLO_WEIGHTS", "weights/yolo/best.pt")
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"depth_mode={DEPTH_MODE}  ckpt={RGBD_MODEL_PATH}  yolo={YOLO_PATH}")

    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        print(f"Save mode: writing PNG frames to {args.save_dir}")

    print("Loading models...")
    yolo_model = YOLO(YOLO_PATH)

    pose_model = RGBD_FusionPredictor().to(DEVICE)
    pose_model.load_state_dict(torch.load(RGBD_MODEL_PATH, map_location=DEVICE))
    pose_model.eval()

    models_info = get_all_models_info(ROOT_DATASET)
    _, _, test_samples, gt_cache = prepare_data_and_splits(ROOT_DATASET)
    test_dataset = LineModDataset(ROOT_DATASET, test_samples, gt_cache)
    object_ids = sorted(gt_cache.keys())
    info_cache = load_info_cache(ROOT_DATASET, object_ids)

    ply_cache = {}
    for oid in object_ids:
        ply_path = os.path.join(ROOT_DATASET, 'models', f'obj_{oid:02d}.ply')
        if os.path.exists(ply_path):
            ply_cache[oid] = trimesh.load(ply_path).sample(800)
        else:
            print(f"Warning: PLY file missing for obj {oid:02d} ({ply_path}).")

    window_name = "Green=GT, Red=Pred (YOLO+RGBD Fusion)"
    print("Pipeline ready. Press 'q' to exit." if not args.save_dir else "Pipeline ready.")

    saved_count = 0
    with torch.no_grad():
        indices = list(range(len(test_dataset)))
        random.Random(args.seed).shuffle(indices)

        for idx in indices:
            if args.max_samples is not None and saved_count >= args.max_samples:
                break
            sample = test_dataset[idx]
            obj_id = int(sample["obj_id"])
            sample_id = int(sample["sample_id"])
            obj_info = get_object_metadata(models_info, obj_id)
            if obj_info is None:
                print(f"Warning: Metadata missing for obj {obj_id:02d}.")
                continue

            pts_model = ply_cache.get(obj_id)
            if pts_model is None:
                print(f"Warning: Point cloud not available for obj {obj_id:02d}.")
                continue

            info_entry = fetch_sample_info(info_cache, obj_id, sample_id)
            if info_entry is None:
                print(f"Warning: Camera info missing for obj {obj_id:02d} sample {sample_id:04d}.")
                continue
            cam_K = np.array(info_entry['cam_K'], dtype=np.float32).reshape(3, 3)
            depth_scale = info_entry.get('depth_scale', 1.0)
            img_path = os.path.join(ROOT_DATASET, 'data', f"{obj_id:02d}", 'rgb', f"{sample['sample_id']:04d}.png")
            depth_path = os.path.join(ROOT_DATASET, 'data', f"{obj_id:02d}", 'depth', f"{sample['sample_id']:04d}.png")

            img_bgr = cv2.imread(img_path)
            depth_raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            if img_bgr is None or depth_raw is None:
                print(f"Warning: Images missing for obj {obj_id:02d} sample {sample_id:04d}.")
                continue

            depth_meters = convert_depth_to_meters(depth_raw, depth_scale)

            results = yolo_model(img_bgr, verbose=False)[0]
            box = select_detection_for_object(results, obj_id)
            if box is None:
                print(f"Warning: YOLO did not find obj {obj_id:02d} in {img_path}")
                continue

            xyxy = box.xyxy.cpu().numpy()[0]
            yolo_bbox = [float(xyxy[0]), float(xyxy[1]), float(xyxy[2]-xyxy[0]), float(xyxy[3]-xyxy[1])] # [x, y, w, h]

            crop_coords = square_crop_coords(yolo_bbox, img_bgr.shape)
            if crop_coords is None:
                print(f"Warning: Invalid crop for obj {obj_id:02d}.")
                continue

            rgb_tensor = prepare_rgb_tensor(img_bgr, crop_coords)
            meta_tensor = build_meta_tensor(yolo_bbox, cam_K, img_bgr.shape)
            anchor = np.zeros(3, dtype=np.float32)
            if DEPTH_MODE == "norm":
                # stessa pipeline del training: depth centrata sull'ancora della bbox
                anchor = LineModDatasetRGBD._bbox_anchor(depth_meters, yolo_bbox, cam_K)
                l, t, r, b = crop_coords
                d_crop = cv2.resize(depth_meters[t:b, l:r], (224, 224), interpolation=cv2.INTER_NEAREST)
                dn = np.where(d_crop > 0, (d_crop - anchor[2]) / LineModDatasetRGBD.DEPTH_NORM_SCALE, 0.0)
                dn = np.clip(dn, -4.0, 4.0).astype(np.float32)
                depth_tensor = torch.from_numpy(np.repeat(dn[None], 3, axis=0)).unsqueeze(0)
            else:
                depth_tensor = prepare_depth_tensor(depth_meters, crop_coords)

            if rgb_tensor is None or depth_tensor is None or meta_tensor is None:
                print(f"Warning: Preprocessing failed for obj {obj_id:02d}.")
                continue

            pred_T, pred_R = pose_model(
                rgb_tensor.to(DEVICE),
                depth_tensor.to(DEVICE),
                meta_tensor.to(DEVICE)
            )
            R_pred = pred_R[0].cpu().numpy()  # (3, 3) from SO(3)
            # la testa predice il residuo rispetto all'ancora (metri) -> millimetri
            T_pred = ((pred_T[0].cpu().numpy() + anchor) * 1000.0).astype(np.float32)

            R_gt = sample["R"].cpu().numpy() if torch.is_tensor(sample["R"]) else np.array(sample["R"], dtype=np.float32)
            T_gt = sample["T"].cpu().numpy() if torch.is_tensor(sample["T"]) else np.array(sample["T"], dtype=np.float32)

            pts_gt = np.dot(pts_model, R_gt.T) + T_gt
            pts_pred = np.dot(pts_model, R_pred.T) + T_pred
            add_error = float(np.mean(np.linalg.norm(pts_gt - pts_pred, axis=1)))
            trans_error = float(np.linalg.norm(T_gt - T_pred))
            rot_trace = np.trace(np.dot(R_pred, R_gt.T))
            rot_angle = np.degrees(np.arccos(np.clip((rot_trace - 1.0) / 2.0, -1.0, 1.0)))
            error_summary = f"ADD {add_error:.2f}mm | dT {trans_error:.2f}mm | dR {rot_angle:.2f} deg"
            print(f"OBJ {obj_id:02d} sample {sample_id:04d} -> {error_summary}")

            vis_img = project_3d_box(
                img_bgr.copy(),
                sample["R"].numpy(),
                sample["T"].numpy(),
                cam_K,
                obj_info,
                color=(0, 255, 0)
            )
            vis_img = project_3d_box(vis_img, R_pred, T_pred, cam_K, obj_info, color=(0, 0, 255))

            cv2.rectangle(vis_img, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (255, 0, 0), 1)
            cv2.putText(vis_img, f"OBJ {obj_id} - YOLO BBox", (int(xyxy[0]), int(xyxy[1]-5)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

            overlay_lines = error_summary.split(" | ")
            for idx_line, line in enumerate(overlay_lines):
                y_pos = 20 + idx_line * 22
                cv2.putText(vis_img, line, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(vis_img, line, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

            if args.save_dir:
                out_name = f"obj{obj_id:02d}_sample{sample_id:04d}_add{add_error:05.2f}mm.png"
                out_path = os.path.join(args.save_dir, out_name)
                cv2.imwrite(out_path, vis_img)
                saved_count += 1
                if saved_count % 10 == 0:
                    print(f"  saved {saved_count} frames so far...")
            else:
                cv2.imshow(window_name, vis_img)
                try:
                    cv2.setWindowTitle(window_name, f"{window_name} | {error_summary}")
                except (cv2.error, AttributeError):
                    pass
                if cv2.waitKey(0) & 0xFF == ord('q'): break

    if args.save_dir:
        print(f"Done. Saved {saved_count} frames to {args.save_dir}/")
    else:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()