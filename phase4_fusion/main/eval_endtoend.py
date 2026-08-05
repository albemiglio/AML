"""End-to-end ADD(-S) on the OFFICIAL LineMod test split: YOLO boxes, not GT.

The honest pipeline number: the crop, the meta tensor AND the 3D anchor all come
from the detector's box. A detection miss counts as a FAILURE for that sample —
skipping it (as the old evaluate.py did) silently inflates accuracy.

Env:
    DEPTH_MODE    raw | norm | xyz          (default norm)
    CKPT          pose checkpoint           (default results_4_main/pose_rgbd_fusion_best_<mode>.pth)
    YOLO_WEIGHTS  detector weights          (default runs/detect/linemod_yolo_run/weights/best.pt)
    CONF          YOLO confidence threshold (default 0.25)
"""
import os
import sys
from collections import defaultdict

import cv2
import numpy as np
import torch
import yaml
from ultralytics import YOLO

if __package__ in {None, ""}:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from common.data_split import prepare_data_and_splits
from common.gpu_augment import GPUAugmentation
from common.pose_metrics import pose_error, SYMMETRIC_OBJ_IDS
from common.yolo_metadata import select_detection_for_object
from phase4_fusion.main.dataset import LineModDatasetRGBD
from phase4_fusion.main.model import RGBD_FusionPredictor
from phase4_fusion.main.rgbd_utils import (
    load_info_cache, fetch_sample_info, convert_depth_to_meters,
    square_crop_coords, build_meta_tensor,
)

ROOT = "datasets/linemod/Linemod_preprocessed"
NAMES = {1: "ape", 2: "benchvise", 4: "camera", 5: "can", 6: "cat", 8: "driller", 9: "duck",
         10: "eggbox", 11: "glue", 12: "holepuncher", 13: "iron", 14: "lamp", 15: "phone"}
SCALE = LineModDatasetRGBD.DEPTH_NORM_SCALE


def build_inputs(rgb_img, depth_meters, bbox, K, depth_mode, anchor_fn):
    """Replicates LineModDatasetRGBD preprocessing for an arbitrary (predicted) bbox.
    Returns (rgb_u8, depth_tensor, meta, t_anchor) or None if the crop is degenerate."""
    crop = square_crop_coords(bbox, rgb_img.shape)
    if crop is None:
        return None
    l, t, r, b = crop
    rgb_crop = cv2.resize(rgb_img[t:b, l:r], (224, 224), interpolation=cv2.INTER_LINEAR)
    depth_crop = cv2.resize(depth_meters[t:b, l:r], (224, 224), interpolation=cv2.INTER_NEAREST)
    rgb_u8 = torch.from_numpy(np.ascontiguousarray(rgb_crop.transpose(2, 0, 1)))

    t_anchor = torch.zeros(3)
    if depth_mode == "raw":
        depth_tensor = torch.from_numpy(depth_crop).float().unsqueeze(0)
    else:
        anchor = anchor_fn(depth_meters, bbox, K)
        t_anchor = torch.from_numpy(anchor)
        if depth_mode == "norm":
            dn = np.where(depth_crop > 0, (depth_crop - anchor[2]) / SCALE, 0.0)
            depth_tensor = torch.from_numpy(np.clip(dn, -4.0, 4.0).astype(np.float32)).unsqueeze(0)
        else:  # xyz
            fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
            u = l + (np.arange(224, dtype=np.float32) + 0.5) * (r - l) / 224
            v = t + (np.arange(224, dtype=np.float32) + 0.5) * (b - t) / 224
            uu, vv = np.meshgrid(u, v)
            valid = depth_crop > 0
            xyz = np.stack([(uu - cx) * depth_crop / fx, (vv - cy) * depth_crop / fy, depth_crop])
            xyz = np.clip((xyz - anchor.reshape(3, 1, 1)) / SCALE, -4.0, 4.0) * valid[None]
            depth_tensor = torch.from_numpy(xyz.astype(np.float32))

    meta = build_meta_tensor(bbox, K, rgb_img.shape)
    if meta is None:
        return None
    return rgb_u8, depth_tensor, meta.squeeze(0), t_anchor


def main():
    depth_mode = os.environ.get("DEPTH_MODE", "norm")
    ckpt = os.environ.get("CKPT", f"results_4_main/pose_rgbd_fusion_best_{depth_mode}.pth")
    yolo_w = os.environ.get("YOLO_WEIGHTS", "runs/detect/linemod_yolo_run/weights/best.pt")
    conf_thr = float(os.environ.get("CONF", "0.25"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"depth_mode={depth_mode}  ckpt={ckpt}  yolo={yolo_w}  conf={conf_thr}")

    _, _, test_samples, gt_cache = prepare_data_and_splits(ROOT)
    assert len(test_samples) == 13407, f"non-official test set: {len(test_samples)}"
    limit = int(os.environ.get("TEST_LIMIT", "0"))  # smoke tests only; 0 = full set
    if limit:
        test_samples = test_samples[:limit]
        print(f"SMOKE MODE: {limit} samples — not a measurement")
    info = load_info_cache(ROOT, sorted(gt_cache.keys()))
    models_info = yaml.safe_load(open(f"{ROOT}/models/models_info.yml"))

    # model points from the shared dataset cache: meters, same sampling as training
    ds = LineModDatasetRGBD(ROOT, test_samples, gt_cache, info, n_points=500, is_train=False,
                            depth_mode=depth_mode)
    anchor_fn = ds._bbox_anchor
    pts_cache = {oid: ds.model_points_cache[oid].numpy() for oid in ds.model_points_cache}

    pose_net = RGBD_FusionPredictor().to(device)
    pose_net.load_state_dict(torch.load(ckpt, map_location=device))
    pose_net.eval()
    gpu_aug = GPUAugmentation().to(device)
    yolo = YOLO(yolo_w)

    ok, tot, errs = defaultdict(int), defaultdict(int), defaultdict(list)
    misses = defaultdict(int)
    batch, batch_meta = [], []

    def flush():
        if not batch:
            return
        rgb = gpu_aug(torch.stack([b[0] for b in batch]).to(device), training=False)
        depth = torch.stack([b[1] for b in batch]).to(device).expand(-1, 3, -1, -1)
        meta = torch.stack([b[2] for b in batch]).to(device)
        anchor = torch.stack([b[3] for b in batch]).to(device)
        with torch.no_grad():
            pT, pR = pose_net(rgb, depth, meta)
        pT = (pT.float() + anchor).cpu().numpy()
        pR = pR.view(-1, 3, 3).float().cpu().numpy()
        for k, (oid, R_gt, T_gt) in enumerate(batch_meta):
            e = pose_error(pts_cache[oid], R_gt, T_gt, pR[k], pT[k], oid)
            thr = 0.1 * models_info[oid]["diameter"] / 1000.0
            tot[oid] += 1
            errs[oid].append(e * 1000.0)
            if e < thr:
                ok[oid] += 1
        batch.clear()
        batch_meta.clear()

    for n, (oid, iid) in enumerate(test_samples):
        ann = next(a for a in gt_cache[oid][iid] if a["obj_id"] == oid)
        entry = fetch_sample_info(info, oid, iid)
        folder = f"{oid:02d}"
        img_bgr = cv2.imread(f"{ROOT}/data/{folder}/rgb/{iid:04d}.png")
        depth_raw = cv2.imread(f"{ROOT}/data/{folder}/depth/{iid:04d}.png", cv2.IMREAD_UNCHANGED)
        depth_m = convert_depth_to_meters(depth_raw, entry["depth_scale"])
        K = np.array(entry["cam_K"], dtype=np.float32).reshape(3, 3)

        det = select_detection_for_object(yolo(img_bgr, conf=conf_thr, verbose=False)[0], oid)
        built = None
        if det is not None:
            x1, y1, x2, y2 = det.xyxy.cpu().numpy()[0]
            built = build_inputs(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), depth_m,
                                 [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                                 K, depth_mode, anchor_fn)
        if built is None:
            # a sample the detector loses IS a pipeline failure, not a skip
            tot[oid] += 1
            misses[oid] += 1
            continue

        batch.append(built)
        batch_meta.append((oid,
                           np.array(ann["cam_R_m2c"], dtype=np.float64).reshape(3, 3),
                           np.array(ann["cam_t_m2c"], dtype=np.float64) / 1000.0))
        if len(batch) == 64:
            flush()
        if (n + 1) % 2000 == 0:
            print(f"  {n + 1}/{len(test_samples)}", flush=True)
    flush()

    print("\n" + "=" * 78)
    print(f"{'object':<14}{'thr mm':>8}{'mean err mm':>13}{'det miss':>10}{'ADD(-S) %':>12}")
    print("-" * 78)
    accs = []
    for oid in sorted(tot):
        acc = 100.0 * ok[oid] / tot[oid]
        accs.append(acc)
        star = "*" if oid in SYMMETRIC_OBJ_IDS else " "
        me = np.mean(errs[oid]) if errs[oid] else float("nan")
        print(f"{NAMES.get(oid, oid) + star:<14}{0.1 * models_info[oid]['diameter']:>8.1f}"
              f"{me:>13.1f}{misses[oid]:>10}{acc:>12.2f}")
    print("-" * 78)
    print(f"{'MEAN':<14}{'':>8}{'':>13}{sum(misses.values()):>10}{np.mean(accs):>12.2f}")
    print("=" * 78)
    print(f"samples: {sum(tot.values())} official test | detection misses count as failures")


if __name__ == "__main__":
    main()
