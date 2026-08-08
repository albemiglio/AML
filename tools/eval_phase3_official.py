"""Lean Phase-3 evaluation on the official LineMod test split.

The legacy evaluator loops frame-by-frame and builds matplotlib figures along the
way, which puts a full pass over the 13,407-image test set beyond any reasonable
GPU-hour budget. This one batches the pose forward passes and skips plotting:
same four configurations, same metric, a fraction of the wall clock.

Configs reported (ADD(-S), threshold 10% of diameter):
  gt_rot    -- GT crop, predicted rotation, GT translation (rotation quality)
  yolo_tgt  -- YOLO crop, predicted rotation, GT translation
  pinhole   -- YOLO crop, predicted rotation, pinhole translation
  regressed -- YOLO crop, predicted rotation, regressed translation

Run from the repo root:  python -m tools.eval_phase3_official
Env: CKPT (default results_3/pose_resnet50_baseline_best.pth), YOLO_WEIGHTS.
"""
import os
import sys
from collections import defaultdict

import cv2
import numpy as np
import torch
import yaml
from scipy.spatial.transform import Rotation as R_conv
from ultralytics import YOLO

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.data_split import prepare_data_and_splits
from common.pose_metrics import pose_error, SYMMETRIC_OBJ_IDS
from common.yolo_metadata import select_detection_for_object
from phase3_baseline.model import PosePredictor
from phase4_fusion.main.rgbd_utils import load_info_cache, fetch_sample_info

ROOT = "datasets/linemod/Linemod_preprocessed"
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
NAMES = {1: "ape", 2: "benchvise", 4: "camera", 5: "can", 6: "cat", 8: "driller", 9: "duck",
         10: "eggbox", 11: "glue", 12: "holepuncher", 13: "iron", 14: "lamp", 15: "phone"}


def crop_tensor(img_rgb, bbox):
    x, y, w, h = [int(v) for v in bbox]
    side = max(w, h)
    cx, cy = x + w // 2, y + h // 2
    x0, y0 = max(0, cx - side // 2), max(0, cy - side // 2)
    x1, y1 = min(img_rgb.shape[1], x0 + side), min(img_rgb.shape[0], y0 + side)
    crop = img_rgb[y0:y1, x0:x1]
    if crop.size == 0:
        return None
    crop = cv2.resize(crop, (224, 224), interpolation=cv2.INTER_CUBIC).astype(np.float32) / 255.0
    crop = (crop - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(crop.transpose(2, 0, 1))


def main():
    ckpt = os.environ.get("CKPT", "results_3/pose_resnet50_baseline_best.pth")
    yolo_w = os.environ.get("YOLO_WEIGHTS", "weights/yolo/best.pt")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"ckpt={ckpt}  yolo={yolo_w}", flush=True)

    _, _, test_samples, gt_cache = prepare_data_and_splits(ROOT)
    assert len(test_samples) == 13407
    info = load_info_cache(ROOT, sorted(gt_cache.keys()))
    models_info = yaml.safe_load(open(f"{ROOT}/models/models_info.yml"))

    import trimesh
    rng = np.random.RandomState(0)
    pts_cache = {}
    for oid in sorted(gt_cache.keys()):
        v = np.asarray(trimesh.load(f"{ROOT}/models/obj_{oid:02d}.ply").vertices, dtype=np.float64)
        pts_cache[oid] = (v[rng.choice(len(v), 500, replace=False)] if len(v) > 500 else v) / 1000.0

    net = PosePredictor().to(device)
    net.load_state_dict(torch.load(ckpt, map_location=device), strict=False)
    net.eval()
    yolo = YOLO(yolo_w)

    stats = {k: defaultdict(lambda: [0, 0]) for k in ("gt_rot", "yolo_tgt", "pinhole", "regressed")}
    BATCH = 64
    buf, meta = [], []

    def flush():
        if not buf:
            return
        with torch.no_grad():
            q, t = net(torch.stack(buf).to(device))
        q = q.float().cpu().numpy(); t = t.float().cpu().numpy()
        for i, m in enumerate(meta):
            oid, R_gt, T_gt, kind, T_pin, thr = m
            Rp = R_conv.from_quat(q[i] / np.linalg.norm(q[i])).as_matrix()
            pts = pts_cache[oid]
            if kind == "gt":
                e = pose_error(pts, R_gt, T_gt, Rp, T_gt, oid)
                _acc(stats["gt_rot"], oid, e, thr)
            else:
                _acc(stats["yolo_tgt"], oid, pose_error(pts, R_gt, T_gt, Rp, T_gt, oid), thr)
                if T_pin is not None:
                    _acc(stats["pinhole"], oid, pose_error(pts, R_gt, T_gt, Rp, T_pin, oid), thr)
                _acc(stats["regressed"], oid, pose_error(pts, R_gt, T_gt, Rp, t[i], oid), thr)
        buf.clear(); meta.clear()

    def _acc(d, oid, e, thr):
        d[oid][1] += 1
        if e < thr:
            d[oid][0] += 1

    done = 0
    for oid, img_id in test_samples:
        ann = next(a for a in gt_cache[oid][img_id] if a["obj_id"] == oid)
        R_gt = np.array(ann["cam_R_m2c"], dtype=np.float64).reshape(3, 3)
        T_gt = np.array(ann["cam_t_m2c"], dtype=np.float64) / 1000.0
        thr = 0.1 * models_info[oid]["diameter"] / 1000.0
        entry = fetch_sample_info(info, oid, img_id)
        K = np.array(entry["cam_K"], dtype=np.float32).reshape(3, 3)
        img = cv2.imread(f"{ROOT}/data/{oid:02d}/rgb/{img_id:04d}.png")
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        tb = crop_tensor(rgb, ann["obj_bb"])
        if tb is not None:
            buf.append(tb); meta.append((oid, R_gt, T_gt, "gt", None, thr))

        det = select_detection_for_object(yolo(img, verbose=False)[0], oid)
        if det is not None:
            xy = det.xyxy.cpu().numpy()[0]
            bb = [float(xy[0]), float(xy[1]), float(xy[2] - xy[0]), float(xy[3] - xy[1])]
            tb = crop_tensor(rgb, bb)
            if tb is not None:
                intr = {"fx": K[0, 0], "fy": K[1, 1], "cx": K[0, 2], "cy": K[1, 2]}
                d_m = models_info[oid]["diameter"] / 1000.0
                px = max(bb[2], bb[3])
                Z = intr["fx"] * d_m / px
                T_pin = np.array([((bb[0] + bb[2] / 2) - intr["cx"]) * Z / intr["fx"],
                                  ((bb[1] + bb[3] / 2) - intr["cy"]) * Z / intr["fy"], Z])
                buf.append(tb); meta.append((oid, R_gt, T_gt, "yolo", T_pin, thr))
        if len(buf) >= BATCH:
            flush()
        done += 1
        if done % 1000 == 0:
            print(f"  {done}/13407", flush=True)
    flush()

    print("\n" + "=" * 66)
    print(f"{'object':<14}{'gt_rot':>10}{'yolo_tgt':>10}{'pinhole':>10}{'regressed':>11}")
    print("-" * 66)
    means = defaultdict(list)
    for oid in sorted(NAMES):
        row = [NAMES[oid] + ("*" if oid in SYMMETRIC_OBJ_IDS else "")]
        for k in ("gt_rot", "yolo_tgt", "pinhole", "regressed"):
            ok, tot = stats[k][oid]
            acc = 100.0 * ok / tot if tot else 0.0
            means[k].append(acc)
            row.append(f"{acc:.2f}")
        print(f"{row[0]:<14}{row[1]:>10}{row[2]:>10}{row[3]:>10}{row[4]:>11}")
    print("-" * 66)
    print(f"{'MEAN':<14}" + "".join(
        f"{np.mean(means[k]):>10.2f}" if k != "regressed" else f"{np.mean(means[k]):>11.2f}"
        for k in ("gt_rot", "yolo_tgt", "pinhole", "regressed")))
    print("=" * 66)
    print("PHASE3 TABLE DONE", flush=True)


if __name__ == "__main__":
    main()
