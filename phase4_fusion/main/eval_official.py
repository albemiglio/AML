"""ADD(-S) accuracy on the OFFICIAL LineMod test split (13,407 images).

Pose network only: crops come from ground-truth boxes, so the number measures the
pose head without a detector in the loop — the row comparable with DenseFusion
per-pixel (86.2) on the same dataset and split.

Env:
    DEPTH_MODE  raw | norm | xyz   (must match the checkpoint being evaluated)
    CKPT        checkpoint path, default results_4_main/pose_rgbd_fusion_best_<mode>.pth

Units: model points and translations in meters (dataset divides by 1000);
models_info.yml diameters in millimeters. Threshold = 10% of diameter.
"""
import os
import sys
from collections import defaultdict

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

if __package__ in {None, ""}:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from common.data_split import prepare_data_and_splits
from common.gpu_augment import GPUAugmentation
from common.pose_metrics import pose_error, SYMMETRIC_OBJ_IDS
from phase4_fusion.main.dataset import LineModDatasetRGBD
from phase4_fusion.main.model import RGBD_FusionPredictor
from phase4_fusion.main.rgbd_utils import load_info_cache

ROOT = "datasets/linemod/Linemod_preprocessed"
NAMES = {1: "ape", 2: "benchvise", 4: "camera", 5: "can", 6: "cat", 8: "driller", 9: "duck",
         10: "eggbox", 11: "glue", 12: "holepuncher", 13: "iron", 14: "lamp", 15: "phone"}


def main():
    depth_mode = os.environ.get("DEPTH_MODE", "raw")
    ckpt = os.environ.get("CKPT", f"results_4_main/pose_rgbd_fusion_best_{depth_mode}.pth")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"depth_mode={depth_mode}  ckpt={ckpt}")

    _, _, test_samples, gt_cache = prepare_data_and_splits(ROOT)
    assert len(test_samples) == 13407, f"non-official test set: {len(test_samples)}"
    info = load_info_cache(ROOT, sorted(gt_cache.keys()))
    models_info = yaml.safe_load(open(f"{ROOT}/models/models_info.yml"))

    ds = LineModDatasetRGBD(ROOT, test_samples, gt_cache, info, n_points=500, is_train=False,
                            depth_mode=depth_mode)
    dl = DataLoader(ds, batch_size=64, shuffle=False, num_workers=12, pin_memory=True)

    model = RGBD_FusionPredictor().to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    # Same input path as training: /255 then ImageNet normalization. Skipping it
    # feeds the network a distribution it has never seen (measured: 6.78 -> 1.81).
    gpu_aug = GPUAugmentation().to(device)

    ok, tot, errs = defaultdict(int), defaultdict(int), defaultdict(list)
    done = 0
    # PER_SAMPLE_CSV: righe (obj_id,img_id,err_mm,thr_mm,hit) per le analisi a valle
    # (accuracy vs distanza di viewpoint). L'ordine del loader e' quello di test_samples
    # (shuffle=False), quindi l'indice progressivo identifica il campione.
    csv_path = os.environ.get("PER_SAMPLE_CSV", "")
    rows = []
    with torch.no_grad():
        for b in dl:
            rgb = gpu_aug(b["rgb"].to(device), training=False)
            depth = b["depth"].to(device).expand(-1, 3, -1, -1)
            meta = b["meta_info"].to(device)
            pT, pR = model(rgb, depth, meta)
            pT = (pT.float() + b["t_anchor"].to(device)).cpu().numpy()
            pR = pR.view(-1, 3, 3).float().cpu().numpy()
            gR = b["R_matrix"].numpy()
            gT = b["translation_3d"].numpy()
            pts = b["model_points"].numpy()
            ids = b["obj_id"].numpy()
            for i in range(len(ids)):
                oid = int(ids[i])
                e = pose_error(pts[i], gR[i], gT[i], pR[i], pT[i], oid)   # meters
                thr = 0.1 * models_info[oid]["diameter"] / 1000.0          # mm -> m
                tot[oid] += 1
                errs[oid].append(e * 1000.0)                               # report in mm
                if e < thr:
                    ok[oid] += 1
                if csv_path:
                    _, img_id = test_samples[done + i]
                    rows.append(f"{oid},{img_id},{e*1000.0:.2f},{thr*1000.0:.2f},{int(e < thr)}")
            done += len(ids)
            if done % 3200 == 0:
                print(f"  {done}/{len(ds)}", flush=True)

    if csv_path:
        with open(csv_path, "w") as f:
            f.write("obj_id,img_id,err_mm,thr_mm,hit\n")
            f.write("\n".join(rows) + "\n")
        print(f"per-sample csv -> {csv_path} ({len(rows)} rows)")

    print("\n" + "=" * 72)
    print(f"{'object':<14}{'diam mm':>9}{'thr mm':>9}{'mean err mm':>13}{'ADD(-S) %':>12}")
    print("-" * 72)
    accs = []
    for oid in sorted(tot):
        acc = 100.0 * ok[oid] / tot[oid]
        accs.append(acc)
        star = "*" if oid in SYMMETRIC_OBJ_IDS else " "
        print(f"{NAMES.get(oid, oid) + star:<14}{models_info[oid]['diameter']:>9.1f}"
              f"{0.1 * models_info[oid]['diameter']:>9.1f}{np.mean(errs[oid]):>13.1f}{acc:>12.2f}")
    print("-" * 72)
    print(f"{'MEAN':<14}{'':>9}{'':>9}"
          f"{np.mean([np.mean(errs[o]) for o in errs]):>13.1f}{np.mean(accs):>12.2f}")
    print("=" * 72)
    print(f"samples: {sum(tot.values())} (official test)   * = symmetric, ADD-S")


if __name__ == "__main__":
    main()
