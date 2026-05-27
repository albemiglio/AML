import argparse
import torch
import numpy as np
import cv2
import os
import yaml
import trimesh
import pandas as pd
from ultralytics import YOLO
from scipy.spatial.transform import Rotation as R_conv
from torchvision import transforms
import matplotlib.pyplot as plt
from tqdm import tqdm


from common.pose_metrics import pose_error
from phase3_baseline.model import PosePredictor
from phase3_baseline.dataset import LineModDataset
from common.data_split import prepare_data_and_splits


def project_3d_box(img, R, T, K, obj_info, color=(0, 255, 0)):
    """Draw 3D box based on YAML data."""
    min_x, max_x = obj_info['min_x'], obj_info['min_x'] + obj_info['size_x']
    min_y, max_y = obj_info['min_y'], obj_info['min_y'] + obj_info['size_y']
    min_z, max_z = obj_info['min_z'], obj_info['min_z'] + obj_info['size_z']
    pts = np.array([
        [min_x, min_y, min_z], [min_x, min_y, max_z], [min_x, max_y, min_z], [min_x, max_y, max_z],
        [max_x, min_y, min_z], [max_x, min_y, max_z], [max_x, max_y, min_z], [max_x, max_y, max_z]
    ], dtype=np.float32)
    pts_2d, _ = cv2.projectPoints(pts, R, T, K, None)
    pts_2d = pts_2d.reshape(-1, 2).astype(int)
    edges = [(0,1), (0,2), (1,3), (2,3), (4,5), (4,6), (5,7), (6,7), (0,4), (1,5), (2,6), (3,7)]
    for i, j in edges:
        cv2.line(img, tuple(pts_2d[i]), tuple(pts_2d[j]), color, 2)
    return img

def project_ply_points(img, R, T, K, pts_3d, color=(0, 255, 0)):
    """Project .ply file points."""
    pts_2d, _ = cv2.projectPoints(pts_3d, R, T, K, None)
    pts_2d = pts_2d.reshape(-1, 2).astype(int)
    for p in pts_2d:
        if 0 <= p[0] < img.shape[1] and 0 <= p[1] < img.shape[0]:
            cv2.circle(img, (p[0], p[1]), 1, color, -1)
    return img

def compute_translation(bbox, intrinsics, diameter):
    """Compute T(X, Y, Z) via Pinhole."""
    x, y, w, h = bbox
    pixel_size = max(w, h)
    Z = (intrinsics['fx'] * diameter) / pixel_size 
    X = ((x + w/2) - intrinsics['cx']) * Z / intrinsics['fx']
    Y = ((y + h/2) - intrinsics['cy']) * Z / intrinsics['fy']
    return np.array([X, Y, Z])

def preprocess_yolo_crop(img_rgb, box_yolo):
    """Prepare YOLO crop in the same way as training."""
    x1, y1, x2, y2 = box_yolo
    w, h = x2 - x1, y2 - y1
    center_x, center_y = x1 + w/2, y1 + h/2
    side = max(w, h)
    left, top = int(center_x - side/2), int(center_y - side/2)
    right, bottom = int(center_x + side/2), int(center_y + side/2)
    crop = img_rgb[max(0, top):min(480, bottom), max(0, left):min(640, right)]
    if crop.size == 0: return None
    crop_resized = cv2.resize(crop, (224, 224))
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(crop_resized).unsqueeze(0)


def generate_terminal_report(val_dataset, pose_net, yolo_model, models_info, ROOT_DATASET, DEVICE, intrinsics):
    class_names = {1: "ape", 2: "benchvise", 4: "camera", 5: "can", 6: "cat", 8: "driller", 
                   9: "duck", 10: "eggbox", 11: "glue", 12: "holepuncher", 13: "iron", 14: "lamp", 15: "phone"}
    
    ply_cache = {oid: trimesh.load(os.path.join(ROOT_DATASET, 'models', f'obj_{oid:02d}.ply')).sample(500) for oid in class_names.keys()}
    results = []

    print("\nStatistical analysis: GT CROP | YOLO+T_gt | YOLO+T_pinhole | YOLO+T_pred ...")
    with torch.no_grad():
        for batch in tqdm(val_dataset):
            obj_id = int(batch["obj_id"])
            if obj_id not in class_names: continue

            R_gt, T_gt = batch["R"].numpy(), batch["T"].numpy()
            pts = ply_cache[obj_id]
            diameter = models_info[obj_id]['diameter']
            threshold = 0.1 * diameter

            # Mode 1: GT crop — isolates rotation quality
            pred_quat_gt, _ = pose_net(batch["rgb"].unsqueeze(0).to(DEVICE))
            pred_quat_gt = pred_quat_gt.detach().cpu().numpy()[0]
            R_pred_gt = R_conv.from_quat(pred_quat_gt / np.linalg.norm(pred_quat_gt)).as_matrix()
            add_gt_crop = pose_error(pts, R_gt, T_gt, R_pred_gt, T_gt, obj_id)

            # Modes 2/3/4: YOLO crop
            img_path = os.path.join(ROOT_DATASET, 'data', f"{obj_id:02d}", 'rgb', f"{batch['sample_id']:04d}.png")
            img_bgr = cv2.imread(img_path)
            yolo_res = yolo_model(img_bgr, verbose=False)[0]

            add_yolo_rot, add_pinhole, add_pred_t = 999.0, 999.0, 999.0
            if len(yolo_res.boxes) > 0:
                box = yolo_res.boxes[0].xyxy.cpu().numpy()[0]
                input_yolo = preprocess_yolo_crop(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), box)

                if input_yolo is not None:
                    pred_quat_y, pred_tvec_y = pose_net(input_yolo.to(DEVICE))
                    pred_quat_y  = pred_quat_y.detach().cpu().numpy()[0]
                    pred_tvec_mm = pred_tvec_y.detach().cpu().numpy()[0] * 1000.0  # m → mm
                    R_pred_y = R_conv.from_quat(pred_quat_y / np.linalg.norm(pred_quat_y)).as_matrix()

                    bbox_xywh = [box[0], box[1], box[2]-box[0], box[3]-box[1]]
                    T_pinhole = compute_translation(bbox_xywh, intrinsics, diameter)

                    # Mode 2: YOLO R + GT T (isolates effect of YOLO crop on rotation)
                    add_yolo_rot = pose_error(pts, R_gt, T_gt, R_pred_y, T_gt, obj_id)
                    # Mode 3: YOLO R + pinhole T (geometric full system)
                    add_pinhole  = pose_error(pts, R_gt, T_gt, R_pred_y, T_pinhole, obj_id)
                    # Mode 4: YOLO R + predicted T (full learned system)
                    add_pred_t   = pose_error(pts, R_gt, T_gt, R_pred_y, pred_tvec_mm, obj_id)

            results.append({
                "Classe": class_names[obj_id],
                "ADD_GT":    add_gt_crop,  "Acc_GT":    add_gt_crop  < threshold,
                "ADD_Y_Rot": add_yolo_rot, "Acc_Y_Rot": add_yolo_rot < threshold,
                "ADD_Full":  add_pinhole,  "Acc_Full":  add_pinhole  < threshold,
                "ADD_Pred":  add_pred_t,   "Acc_Pred":  add_pred_t   < threshold,
            })

    df = pd.DataFrame(results)
    summary = df.groupby("Classe").agg(
        Media_GT    =("ADD_GT",    "mean"), Acc_GT    =("Acc_GT",    lambda x: x.mean()*100),
        Media_Y_Rot =("ADD_Y_Rot", "mean"), Acc_Y_Rot =("Acc_Y_Rot", lambda x: x.mean()*100),
        Media_Full  =("ADD_Full",  "mean"), Acc_Full  =("Acc_Full",  lambda x: x.mean()*100),
        Media_Pred  =("ADD_Pred",  "mean"), Acc_Pred  =("Acc_Pred",  lambda x: x.mean()*100),
    ).reset_index()

    sym_note = " (ADD-S for eggbox/glue)"
    print("\n" + "="*155)
    print(f"{'CLASSE':<14} | {'GT Crop':^22} | {'YOLO+T_gt':^22} | {'YOLO+T_pinhole':^22} | {'YOLO+T_pred':^22}")
    print(f"{'':14} | {'ADD(mm)':<10} {'Acc%':<10} | {'ADD(mm)':<10} {'Acc%':<10} | {'ADD(mm)':<10} {'Acc%':<10} | {'ADD(mm)':<10} {'Acc%':<10}")
    print("-" * 155)
    for _, r in summary.iterrows():
        sym = "*" if r['Classe'] in ("eggbox", "glue") else " "
        print(
            f"{r['Classe']:<14}{sym}| {r['Media_GT']:<10.2f} {r['Acc_GT']:<10.1f}"
            f"| {r['Media_Y_Rot']:<10.2f} {r['Acc_Y_Rot']:<10.1f}"
            f"| {r['Media_Full']:<10.2f} {r['Acc_Full']:<10.1f}"
            f"| {r['Media_Pred']:<10.2f} {r['Acc_Pred']:<10.1f}"
        )
    print("="*155)
    print(f"* ADD-S metric (symmetric objects)")
    print(
        f"GLOBAL AVERAGE -> GT: {summary['Acc_GT'].mean():.1f}%"
        f" | YOLO+T_gt: {summary['Acc_Y_Rot'].mean():.1f}%"
        f" | YOLO+T_pinhole: {summary['Acc_Full'].mean():.1f}%"
        f" | YOLO+T_pred: {summary['Acc_Pred'].mean():.1f}%"
    )

    os.makedirs("results", exist_ok=True)
    out_csv = "results/phase3_perclass.csv"
    summary.to_csv(out_csv, index=False, float_format="%.2f")
    print(f"\n[saved] per-class summary -> {out_csv}")


def run_inspector(report_only=False):
    ROOT_DATASET = "datasets/linemod/Linemod_preprocessed"
    YOLO_PATH = 'weights/yolo/best.pt'
    RESNET_PATH = "weights/baseline/pose_resnet50_baseline.pth"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    K = np.array([[572.4114, 0, 325.2611], [0, 573.5704, 242.0489], [0, 0, 1]], dtype=np.float32)
    intrinsics = {'fx': K[0,0], 'fy': K[1,1], 'cx': K[0,2], 'cy': K[1,2]}

    yolo = YOLO(YOLO_PATH)
    pose_net = PosePredictor().to(DEVICE)
    pose_net.load_state_dict(torch.load(RESNET_PATH, map_location=DEVICE), strict=False)
    pose_net.eval()

    with open(os.path.join(ROOT_DATASET, 'models', 'models_info.yml'), 'r') as f:
        models_info = yaml.safe_load(f)
    _, _, test_samples, gt_cache = prepare_data_and_splits(ROOT_DATASET)
    test_dataset = LineModDataset(ROOT_DATASET, test_samples, gt_cache)

    # 1. Report su terminale con tre confronti
    generate_terminal_report(test_dataset, pose_net, yolo, models_info, ROOT_DATASET, DEVICE, intrinsics)

    if report_only:
        return

    # 2. Visualizzazione con GT verde e Pred rosso
    while True:
        idx = np.random.randint(len(test_dataset))
        batch = test_dataset[idx]
        obj_id = int(batch["obj_id"])
        img_rgb = cv2.cvtColor(cv2.imread(os.path.join(ROOT_DATASET, 'data', f"{obj_id:02d}", 'rgb', f"{batch['sample_id']:04d}.png")), cv2.COLOR_BGR2RGB)
        
        results = yolo(img_rgb, verbose=False)[0]
        if not results.boxes: continue
        box_y, box_gt = results.boxes[0].xyxy.cpu().numpy()[0], batch["bbox_originale"].numpy()
        
        R_gt, T_gt = batch["R"].numpy(), batch["T"].numpy()
        pred_quat, pred_tvec = pose_net(batch["rgb"].unsqueeze(0).to(DEVICE))
        pred_quat = pred_quat.detach().cpu().numpy()[0]
        R_p = R_conv.from_quat(pred_quat / np.linalg.norm(pred_quat)).as_matrix()
        T_p = compute_translation([box_y[0], box_y[1], box_y[2]-box_y[0], box_y[3]-box_y[1]], intrinsics, models_info[obj_id]['diameter'])

        plt.figure(figsize=(25, 5))
        titles = ["Rotazione (T reale)", "Traslazione (R reale)", "BBox Detection", "Punti PLY", "Full 6D Pose"]
        for i in range(5):
            ax = plt.subplot(1, 5, i+1); viz = img_rgb.copy()
            if i == 0: # Rotazione isolata
                viz = project_3d_box(viz, R_gt, T_gt, K, models_info[obj_id], (0,255,0))
                viz = project_3d_box(viz, R_p, T_gt, K, models_info[obj_id], (255,0,0))
            elif i == 1: # Traslazione isolata
                viz = project_3d_box(viz, R_gt, T_gt, K, models_info[obj_id], (0,255,0))
                viz = project_3d_box(viz, R_gt, T_p, K, models_info[obj_id], (255,0,0))
            elif i == 2: # BBox
                cv2.rectangle(viz, (int(box_gt[0]), int(box_gt[1])), (int(box_gt[0]+box_gt[2]), int(box_gt[1]+box_gt[3])), (0,255,0), 2)
                cv2.rectangle(viz, (int(box_y[0]), int(box_y[1])), (int(box_y[2]), int(box_y[3])), (255,0,0), 2)
            elif i == 3: # PLY
                pts = trimesh.load(os.path.join(ROOT_DATASET, 'models', f'obj_{obj_id:02d}.ply')).sample(400)
                viz = project_ply_points(viz, R_gt, T_gt, K, pts, (0,255,0))
                viz = project_ply_points(viz, R_p, T_p, K, pts, (255,0,0))
            elif i == 4: # Full Pose
                viz = project_3d_box(viz, R_gt, T_gt, K, models_info[obj_id], (0,255,0))
                viz = project_3d_box(viz, R_p, T_p, K, models_info[obj_id], (255,0,0))
            plt.imshow(viz); plt.title(titles[i])
        plt.tight_layout(); plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-only", action="store_true",
                        help="Stampa solo la tabella ADD e termina, senza il visualizer interattivo.")
    args = parser.parse_args()
    run_inspector(report_only=args.report_only)