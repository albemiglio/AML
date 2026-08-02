import os
import random

import yaml

# LineMod ships the official split per object in data/<obj>/{train,test}.txt (15/85).
# It must be used as-is: the frames are consecutive video samples, so reshuffling them
# puts near-duplicate views on both sides of the split. Validation is carved out of the
# official train set; the official test set is never touched.
VAL_RATIO_OF_TRAIN = 0.15


def _read_ids(path):
    with open(path, "r", encoding="utf-8") as f:
        return [int(line) for line in f if line.strip()]


def prepare_data_and_splits(dataset_root, random_seed=42, val_ratio=VAL_RATIO_OF_TRAIN):
    """Return official train/val/test splits plus cached ground-truth annotations."""
    if not 0 <= val_ratio < 1:
        raise ValueError("val_ratio must be in [0, 1)")

    train_samples, val_samples, test_samples = [], [], []
    gt_cache = {}

    data_path = os.path.join(dataset_root, "data")
    obj_dirs = sorted(d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d)))

    print(f"Scanning {len(obj_dirs)} objects...")

    for obj_dir in obj_dirs:
        try:
            obj_id = int(obj_dir)
        except ValueError:
            continue

        obj_path = os.path.join(data_path, obj_dir)
        gt_file = os.path.join(obj_path, "gt.yml")
        train_file = os.path.join(obj_path, "train.txt")
        test_file = os.path.join(obj_path, "test.txt")
        if not all(os.path.exists(p) for p in (gt_file, train_file, test_file)):
            continue

        with open(gt_file, "r", encoding="utf-8") as f:
            gt_cache[obj_id] = yaml.safe_load(f)

        official_train = _read_ids(train_file)

        # deterministic per-object carve-out, independent of how objects are ordered
        shuffled = official_train[:]
        random.Random(random_seed + obj_id).shuffle(shuffled)
        n_val = round(len(shuffled) * val_ratio)

        val_samples += [(obj_id, img_id) for img_id in shuffled[:n_val]]
        train_samples += [(obj_id, img_id) for img_id in shuffled[n_val:]]
        test_samples += [(obj_id, img_id) for img_id in _read_ids(test_file)]

    if not test_samples:
        raise RuntimeError(f"No official split files found under {data_path}")

    fitted = set(train_samples) | set(val_samples)
    leaked = fitted & set(test_samples)
    if leaked:
        raise RuntimeError(f"{len(leaked)} test samples leaked into train/val")

    total = len(fitted) + len(test_samples)
    print("Split completed (official LineMod 15/85):")
    print(f"   - Train: {len(train_samples)} samples ({len(train_samples) / total * 100:.1f}%)")
    print(f"   - Val:   {len(val_samples)} samples ({len(val_samples) / total * 100:.1f}%)")
    print(f"   - Test:  {len(test_samples)} samples ({len(test_samples) / total * 100:.1f}%)")
    print(f"   - Fitted on (train+val): {len(fitted)} ({len(fitted) / total * 100:.1f}%)")

    return train_samples, val_samples, test_samples, gt_cache


if __name__ == "__main__":
    root = os.path.join(os.path.dirname(__file__), "..", "datasets", "linemod", "Linemod_preprocessed")
    train, val, test, gt = prepare_data_and_splits(root)

    # invariants of the official split: reshuffling the frames breaks all of these
    assert len(train) + len(val) == 2373, f"official train is 2373, got {len(train) + len(val)}"
    assert len(test) == 13407, f"official test is 13407, got {len(test)}"
    assert not (set(train) | set(val)) & set(test)
    assert set(gt) == {obj_id for obj_id, _ in test}, "gt cache and split disagree on objects"
    print("\nOK: official 15/85 split, no train/test overlap.")
