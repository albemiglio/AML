import os
import random
import argparse
import cv2
import numpy as np
import torch
import yaml
import trimesh
from ultralytics import YOLO

from phase4_fusion.extension.model import FusionResNetCustom





MODEL_PATH = "weights/fusion_ext/resnet10/pose_rgbd_custom_1ch_best.pth"
print("Using custom ResNet-10 model.")

from common.data_split import prepare_data_and_splits
from phase4_fusion.extension.rgbd_utils import (
    load_info_cache,
    fetch_sample_info,
    convert_depth_to_meters,
    square_crop_coords,
    prepare_rgb_tensor,
    prepare_depth_tensor,
    build_meta_tensor,
)
from common.yolo_metadata import get_object_metadata, select_detection_for_object

import albumentations as A
from albumentations.pytorch import ToTensorV2

# Pipeline coerente con l'addestramento
INFERENCE_TRANSFORM = A.Compose([
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
], additional_targets={'depth': 'mask'})




def get_all_models_info(root_path):
    info_path = os.path.join(root_path, "models", "models_info.yml")
    with open(info_path, "r") as f:
        return yaml.safe_load(f)


def get_annotation(gt_cache, obj_id, sample_id):
    obj_gt = gt_cache.get(obj_id)
    if obj_gt is None:
        return None

    ann_list = None
    for key in (sample_id, str(sample_id), f"{sample_id:04d}"):
        if key in obj_gt:
            ann_list = obj_gt[key]
            break

    if ann_list is None and isinstance(obj_gt, list) and sample_id < len(obj_gt):
        ann_list = obj_gt[sample_id]

    if ann_list is None:
        return None
    if isinstance(ann_list, list):
        return next((ann for ann in ann_list if ann.get("obj_id") == obj_id), ann_list[0])
    return ann_list


def project_3d_box(img, R, T, K, obj_info, color=(0, 255, 0), thickness=2):
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

    edges = [(0, 1), (0, 2), (1, 3), (2, 3), (4, 5), (4, 6), (5, 7), (6, 7), (0, 4), (1, 5), (2, 6), (3, 7)]
    for i, j in edges:
        cv2.line(img, tuple(pts_2d[i]), tuple(pts_2d[j]), color, thickness)
    return img

def project_model_points(img, R, T, K, points_3d, color=(255, 255, 0)):
    """Project PLY model points onto image."""
    T_vec = T.reshape(3, 1)
    
    # Proiezione dei punti
    pts_2d, _ = cv2.projectPoints(points_3d, R, T_vec, K, None)
    pts_2d = pts_2d.reshape(-1, 2).astype(int)

    # Disegna ogni punto come un piccolo cerchio
    for p in pts_2d:
        cv2.circle(img, tuple(p), 1, color, -1)
    return img


def main():
    parser = argparse.ArgumentParser(description="Visualize Phase 4 Extension predictions.")
    parser.add_argument("--save-dir", type=str, default=None,
                        help="If set, save annotated frames as PNG to this directory.")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ROOT_DATASET = "datasets/linemod/Linemod_preprocessed"
    YOLO_PATH = "weights/yolo/best.pt"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        print(f"Save mode: writing PNG frames to {args.save_dir}")

    print("Loading RGBD custom model...")
    pose_model = FusionResNetCustom().to(DEVICE)
    pose_model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    pose_model.eval()

    print("Loading YOLO...")
    yolo_model = YOLO(YOLO_PATH)

    _, _, test_samples, gt_cache = prepare_data_and_splits(ROOT_DATASET)
    object_ids = sorted(gt_cache.keys())
    info_cache = load_info_cache(ROOT_DATASET, object_ids)
    models_info = get_all_models_info(ROOT_DATASET)

    ply_cache = {}
    for oid in object_ids:
        ply_path = os.path.join(ROOT_DATASET, 'models', f'obj_{oid:02d}.ply')
        if os.path.exists(ply_path):
            ply_cache[oid] = trimesh.load(ply_path).sample(500)
        else:
            print(f"Warning: PLY file missing for obj {oid:02d} ({ply_path}).")

    window_name = "Green=GT, Red=Pred (RGBD custom)"
    print("Pipeline ready. Press 'q' to exit.")

    with torch.no_grad():
        indices = list(range(len(test_samples)))
        random.shuffle(indices)

        for idx in indices:
            obj_id, sample_id = test_samples[idx]
            obj_info = get_object_metadata(models_info, obj_id)
            if obj_info is None:
                print(f"Warning: Metadata missing for obj {obj_id:02d}.")
                continue

            pts_model = ply_cache.get(obj_id)
            if pts_model is None:
                print(f"Warning: Point cloud not available for obj {obj_id:02d}.")
                continue

            ann = get_annotation(gt_cache, obj_id, sample_id)
            if ann is None:
                print(f"Warning: Annotation missing for obj {obj_id:02d} sample {sample_id:04d}.")
                continue

            info_entry = fetch_sample_info(info_cache, obj_id, sample_id)
            if info_entry is None:
                print(f"Warning: Camera info missing for obj {obj_id:02d} sample {sample_id:04d}.")
                continue

            bbox_gt = ann.get('obj_bb')

            cam_K = np.array(info_entry['cam_K'], dtype=np.float32).reshape(3, 3)
            depth_scale = info_entry.get('depth_scale', 1.0)

            img_path = os.path.join(ROOT_DATASET, 'data', f"{obj_id:02d}", 'rgb', f"{sample_id:04d}.png")
            depth_path = os.path.join(ROOT_DATASET, 'data', f"{obj_id:02d}", 'depth', f"{sample_id:04d}.png")

            img_bgr = cv2.imread(img_path)
            depth_raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            if img_bgr is None or depth_raw is None:
                print(f"Warning: Images missing for obj {obj_id:02d} sample {sample_id:04d}.")
                continue

            depth_meters = convert_depth_to_meters(depth_raw, depth_scale)

            yolo_result = yolo_model(img_bgr, verbose=False)[0]
            box = select_detection_for_object(yolo_result, obj_id)
            if box is None:
                print(f"Warning: YOLO did not find obj {obj_id:02d} in {img_path}")
                continue

            xyxy = box.xyxy.cpu().numpy()[0]
            yolo_bbox = [
                float(xyxy[0]),
                float(xyxy[1]),
                float(xyxy[2] - xyxy[0]),
                float(xyxy[3] - xyxy[1])
            ]

            crop_coords = square_crop_coords(yolo_bbox, img_bgr.shape)
            if crop_coords is None:
                print(f"Warning: Invalid crop (YOLO) for obj {obj_id:02d} sample {sample_id:04d}.")
                continue

            left, top, right, bottom = crop_coords

            rgb_crop = cv2.resize(img_bgr[top:bottom, left:right], (224, 224))
            rgb_crop = cv2.cvtColor(rgb_crop, cv2.COLOR_BGR2RGB)

            depth_crop = cv2.resize(
                depth_meters[top:bottom, left:right], 
                (224, 224), 
                interpolation=cv2.INTER_NEAREST
            )

            transformed = INFERENCE_TRANSFORM(image=rgb_crop, depth=depth_crop)

            rgb_tensor = transformed['image'].unsqueeze(0).to(DEVICE)
            depth_tensor = transformed['depth'].float().unsqueeze(0).unsqueeze(0).to(DEVICE)


            meta_tensor = build_meta_tensor(yolo_bbox, cam_K, img_bgr.shape)

            if rgb_tensor is None or depth_tensor is None or meta_tensor is None:
                print(f"Warning: Preprocessing failed for obj {obj_id:02d} sample {sample_id:04d}.")
                continue

            pred_T, pred_R = pose_model(rgb_tensor, depth_tensor, meta_tensor.to(DEVICE))
            R_pred = pred_R[0].cpu().numpy()  # (3, 3) from SO(3)
            T_pred = (pred_T[0].cpu().numpy() * 1000.0).astype(np.float32)



            R_gt = np.array(ann['cam_R_m2c'], dtype=np.float32).reshape(3, 3)
            T_gt = np.array(ann['cam_t_m2c'], dtype=np.float32)




            vis_points = img_bgr.copy()

            pts_mm = pts_model * 1000.0 if pts_model.max() < 10.0 else pts_model

            project_model_points(vis_points, R_gt, T_gt, cam_K, pts_mm, color=(0, 255, 0))
            project_model_points(vis_points, R_pred, T_pred, cam_K, pts_mm, color=(255, 255, 0))

            pts_gt = np.dot(pts_model, R_gt.T) + T_gt
            pts_pred = np.dot(pts_model, R_pred.T) + T_pred
            add_error = float(np.mean(np.linalg.norm(pts_gt - pts_pred, axis=1)))
            trans_error = float(np.linalg.norm(T_gt - T_pred))
            rot_trace = np.trace(np.dot(R_pred, R_gt.T))
            rot_angle = np.degrees(np.arccos(np.clip((rot_trace - 1.0) / 2.0, -1.0, 1.0)))
            error_summary = f"ADD {add_error:.2f}mm | dT {trans_error:.2f}mm | dR {rot_angle:.4f} deg"
            print(f"OBJ {obj_id:02d} sample {sample_id:04d} -> {error_summary}")

            vis_img = project_3d_box(
                img_bgr.copy(),
                R_gt,
                T_gt,
                cam_K,
                obj_info,
                color=(0, 255, 0)
            )
            vis_img = project_3d_box(vis_img, R_pred, T_pred, cam_K, obj_info, color=(0, 0, 255))

            if bbox_gt is not None:
                x_gt, y_gt, w_gt, h_gt = bbox_gt
                cv2.rectangle(vis_img, (int(x_gt), int(y_gt)), (int(x_gt + w_gt), int(y_gt + h_gt)), (255, 255, 0), 1)
                cv2.putText(vis_img, f"OBJ {obj_id} - GT BBox", (int(x_gt), max(15, int(y_gt) - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

            x_det, y_det, w_det, h_det = yolo_bbox
            cv2.rectangle(vis_img, (int(x_det), int(y_det)), (int(x_det + w_det), int(y_det + h_det)), (255, 0, 0), 1)
            cv2.putText(vis_img, "YOLO BBox", (int(x_det), max(15, int(y_det) - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

            overlay_lines = error_summary.split(" | ")
            for idx_line, line in enumerate(overlay_lines):
                y_pos = 20 + idx_line * 22
                cv2.putText(vis_img, line, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(vis_img, line, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

            comparison = np.hstack((vis_img.copy(), vis_points))
            if args.save_dir:
                out_name = f"obj{obj_id:02d}_sample{sample_id:04d}_add{add_error:05.2f}mm.png"
                cv2.imwrite(os.path.join(args.save_dir, out_name), comparison)
                if args.max_samples is not None:
                    args.max_samples -= 1
                    if args.max_samples <= 0:
                        break
            else:
                cv2.imshow("Left: Box | Right: Point Cloud", comparison)
                if cv2.waitKey(0) & 0xFF == ord('q'):
                    break

    if not args.save_dir:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
