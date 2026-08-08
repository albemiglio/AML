"""Memorization bounds for the official LineMod split.

Two questions the 95.1% result must survive:

1. feat-NN  — honest memorization: for each test frame, retrieve the most similar
   TRAIN frame (32x32 grayscale of the GT crop, L2) and copy its pose. This is what
   "fancy nearest neighbour" would score.
2. oracle-NN — the CEILING of any memorization: copy the pose of the train frame
   closest in pose space (min geodesic rotation + translation term). No retrieval
   method can beat this while still just copying poses.

If oracle-NN << 95, the learned result cannot be explained by memorization.

Also reports the viewpoint-gap distribution (min geodesic rotation from each test
frame to any train frame): a property of the benchmark itself.

Runs locally, no GPU. Uses only gt.yml poses + images + CAD models.
"""
import os, sys, yaml, csv
import numpy as np
import cv2

sys.path.insert(0, os.path.expanduser("~/PycharmProjects/AML"))
from common.pose_metrics import pose_error, SYMMETRIC_OBJ_IDS
import trimesh

ROOT = os.path.expanduser("~/PycharmProjects/AML/datasets/linemod/Linemod_preprocessed")
OUT_DIR = os.path.expanduser("~/StudioAML/04_settembre")
NAMES = {1: "ape", 2: "benchvise", 4: "camera", 5: "can", 6: "cat", 8: "driller", 9: "duck",
         10: "eggbox", 11: "glue", 12: "holepuncher", 13: "iron", 14: "lamp", 15: "phone"}
RNG = np.random.RandomState(0)

models_info = yaml.safe_load(open(f"{ROOT}/models/models_info.yml"))


def load_points(oid, n=500):
    mesh = trimesh.load(f"{ROOT}/models/obj_{oid:02d}.ply")
    pts = np.asarray(mesh.vertices, dtype=np.float64)
    if len(pts) > n:
        pts = pts[RNG.choice(len(pts), n, replace=False)]
    return pts / 1000.0


def crop_feat(img_path, bbox, size=32):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    x, y, w, h = [int(v) for v in bbox]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(img.shape[1], x + w), min(img.shape[0], y + h)
    if x1 - x0 < 4 or y1 - y0 < 4:
        return None
    c = cv2.resize(img[y0:y1, x0:x1], (size, size)).astype(np.float32).ravel()
    n = np.linalg.norm(c)
    return c / n if n > 0 else c


def geodesic_deg(Ra, Rb):
    # batch: Ra (N,3,3), Rb (M,3,3) -> (N,M) angles in degrees
    tr = np.einsum("nij,mij->nm", Ra, Rb)
    cos = np.clip((tr - 1.0) / 2.0, -1.0, 1.0)
    return np.degrees(np.arccos(cos))


summary, all_rows = [], []
for oid in sorted(NAMES):
    d = f"{ROOT}/data/{oid:02d}"
    gt = yaml.safe_load(open(f"{d}/gt.yml"))
    train_ids = [int(l) for l in open(f"{d}/train.txt").read().split()]
    test_ids = [int(l) for l in open(f"{d}/test.txt").read().split()]
    pts = load_points(oid)
    thr = 0.1 * models_info[oid]["diameter"] / 1000.0

    def pose_of(i):
        ann = next(a for a in gt[i] if a["obj_id"] == oid)
        R = np.array(ann["cam_R_m2c"], dtype=np.float64).reshape(3, 3)
        T = np.array(ann["cam_t_m2c"], dtype=np.float64) / 1000.0
        return R, T, ann["obj_bb"]

    train = [pose_of(i) for i in train_ids]
    Rtr = np.stack([p[0] for p in train]); Ttr = np.stack([p[1] for p in train])

    feats_tr = []
    for fid, (R, T, bb) in zip(train_ids, train):
        f = crop_feat(f"{d}/rgb/{fid:04d}.png", bb)
        feats_tr.append(f if f is not None else np.zeros(1024, np.float32))
    Ftr = np.stack(feats_tr)

    hit_feat = hit_oracle = 0
    gaps = []
    for tid in test_ids:
        R, T, bb = pose_of(tid)
        # honest retrieval NN
        f = crop_feat(f"{d}/rgb/{tid:04d}.png", bb)
        j = int(np.argmax(Ftr @ f)) if f is not None else 0
        e = pose_error(pts, R, T, Rtr[j], Ttr[j], oid)
        hit_feat += int(e < thr)
        # oracle NN: closest train pose (rotation geodesic + translation in comparable units)
        ang = geodesic_deg(R[None], Rtr)[0]                    # degrees
        dt = np.linalg.norm(Ttr - T, axis=1)                   # meters
        k = int(np.argmin(np.radians(ang) + dt / 0.1))         # 0.1 m ~ 1 rad weighting
        e2 = pose_error(pts, R, T, Rtr[k], Ttr[k], oid)
        hit_oracle += int(e2 < thr)
        gap = float(ang.min())
        gaps.append(gap)
        all_rows.append((oid, tid, round(e * 1000, 2), round(e2 * 1000, 2), round(gap, 2)))

    n = len(test_ids)
    gaps = np.array(gaps)
    summary.append((NAMES[oid], 100 * hit_feat / n, 100 * hit_oracle / n,
                    float(np.median(gaps)), float(np.percentile(gaps, 90))))
    print(f"{NAMES[oid]:<12} featNN {100*hit_feat/n:6.2f}  oracleNN {100*hit_oracle/n:6.2f}"
          f"  gap med {np.median(gaps):5.1f}° p90 {np.percentile(gaps,90):5.1f}°", flush=True)

print("\n" + "=" * 74)
print(f"{'object':<12}{'featNN %':>10}{'oracleNN %':>12}{'gap med°':>10}{'gap p90°':>10}")
for name, a, b, g1, g2 in summary:
    print(f"{name:<12}{a:>10.2f}{b:>12.2f}{g1:>10.1f}{g2:>10.1f}")
mf = np.mean([s[1] for s in summary]); mo = np.mean([s[2] for s in summary])
print("-" * 74)
print(f"{'MEAN':<12}{mf:>10.2f}{mo:>12.2f}")
print("=" * 74)
print("reference: trained model (norm, end-to-end) = 95.10")

with open(f"{OUT_DIR}/nn_baseline_persample.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["obj_id", "img_id", "featNN_err_mm", "oracleNN_err_mm", "min_gap_deg"])
    w.writerows(all_rows)
print(f"csv -> {OUT_DIR}/nn_baseline_persample.csv")
