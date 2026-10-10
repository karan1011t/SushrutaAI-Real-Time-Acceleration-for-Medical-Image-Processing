"""
Figshare Brain-Tumor (Cheng et al., T1-weighted CE-MRI) preprocessing pipeline.

Reads MATLAB v7.3 (.mat/HDF5) slices from data/external/figshare/, decodes the
patient PID, normalizes intensity with p1-p99.5 windowing, resizes to 256x256,
and writes PNGs + a PID-level stratified split manifest to data/processed/figshare/.

Layout mirrors BraTS2DDataset expectations:
  data/processed/figshare/{train,val,test}/{images,masks}/<name>.png
  data/processed/figshare/split_manifest.json
  data/processed/figshare/metadata.csv

Labels: 1=meningioma, 2=glioma, 3=pituitary. Masks are binary whole-tumor {0,1}.
Splits are patient-disjoint (no PID appears in more than one split).
"""

import csv
import glob
import json
import os
import random
import sys
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

import cv2
import h5py
import numpy as np
from PIL import Image

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

LABEL_NAMES = {1: "meningioma", 2: "glioma", 3: "pituitary"}
TARGET_SIZE = 256


def decode_pid(pid_arr: np.ndarray) -> str:
    """PID is stored as uint16 char codes, e.g. [49,48,...] -> '100860'."""
    return "".join(chr(int(c)) for c in np.asarray(pid_arr).flatten())


def read_figshare_mat(path: str) -> Tuple[np.ndarray, np.ndarray, int, str]:
    """Returns (image_int16, mask_uint8, label_int, pid_str)."""
    with h5py.File(path, "r") as f:
        cj = f["cjdata"]
        image = np.asarray(cj["image"])
        mask = np.asarray(cj["tumorMask"])
        label = int(np.asarray(cj["label"]).flatten()[0])
        pid = decode_pid(np.asarray(cj["PID"]))
    mask = (mask > 0).astype(np.uint8)
    return image, mask, label, pid


def normalize_window(image: np.ndarray) -> np.ndarray:
    """p1-p99.5 windowing -> uint8 0..255. Robust to outlier spikes."""
    img = image.astype(np.float32)
    lo, hi = np.percentile(img, (1.0, 99.5))
    if hi <= lo:
        hi = lo + 1.0
    img = np.clip((img - lo) / (hi - lo), 0.0, 1.0)
    return (img * 255.0).round().astype(np.uint8)


def run_figshare_preprocessing(
    raw_dir: str = None,
    out_dir: str = None,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    seed: int = 42,
) -> Dict:
    raw_dir = raw_dir or os.path.join(project_root, "data", "external", "figshare")
    out_dir = out_dir or os.path.join(project_root, "data", "processed", "figshare")

    mat_files = sorted(
        p for p in glob.glob(os.path.join(raw_dir, "*.mat"))
        if os.path.splitext(os.path.basename(p))[0].isdigit()
    )
    if not mat_files:
        raise FileNotFoundError(f"No .mat files in {raw_dir}")
    print(f"[FIGSHARE] Found {len(mat_files)} .mat slices in {raw_dir}", flush=True)

    # --- Pass 1: index PID -> files, majority class per patient ---
    pid_files: Dict[str, List[str]] = defaultdict(list)
    pid_labels: Dict[str, List[int]] = defaultdict(list)
    for path in mat_files:
        with h5py.File(path, "r") as f:
            cj = f["cjdata"]
            pid = decode_pid(np.asarray(cj["PID"]))
            label = int(np.asarray(cj["label"]).flatten()[0])
        pid_files[pid].append(path)
        pid_labels[pid].append(label)

    patients = sorted(pid_files)
    print(f"[FIGSHARE] {len(patients)} unique patients", flush=True)
    file_label_counts = Counter(l for ls in pid_labels.values() for l in ls)
    print("[FIGSHARE] Slice class distribution:",
          {f"{k}={LABEL_NAMES[k]}": file_label_counts[k] for k in sorted(file_label_counts)}, flush=True)

    # --- PID-level stratified split (by each patient's majority class) ---
    rng = random.Random(seed)
    by_class: Dict[int, List[str]] = defaultdict(list)
    for pid in patients:
        majority = Counter(pid_labels[pid]).most_common(1)[0][0]
        by_class[majority].append(pid)
    splits: Dict[str, List[str]] = {"train": [], "val": [], "test": []}
    for cls in sorted(by_class):
        members = by_class[cls][:]
        rng.shuffle(members)
        n = len(members)
        n_train = int(n * train_frac)
        n_val = int(n * val_frac)
        splits["train"].extend(members[:n_train])
        splits["val"].extend(members[n_train:n_train + n_val])
        splits["test"].extend(members[n_train + n_val:])
    for name in splits:
        rng.shuffle(splits[name])

    # Safety: patient disjointness
    assert not (set(splits["train"]) & set(splits["val"]) & set(splits["test"])), "PID leak across splits!"
    assert not (set(splits["train"]) & set(splits["val"])), "PID leak train/val!"
    assert not (set(splits["train"]) & set(splits["test"])), "PID leak train/test!"
    assert not (set(splits["val"]) & set(splits["test"])), "PID leak val/test!"

    # --- Pass 2: convert + write ---
    manifest = {"seed": seed, "splits": {}, "label_names": LABEL_NAMES}
    metadata_rows = []
    for split_name, pids in splits.items():
        img_dir = os.path.join(out_dir, split_name, "images")
        mask_dir = os.path.join(out_dir, split_name, "masks")
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(mask_dir, exist_ok=True)
        count = 0
        for pid in pids:
            for path in sorted(pid_files[pid]):
                stem = os.path.splitext(os.path.basename(path))[0]
                image, mask, label, _ = read_figshare_mat(path)
                img_u8 = normalize_window(image)
                img_r = cv2.resize(img_u8, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_LINEAR)
                mask_r = cv2.resize(mask, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_NEAREST)
                mask_r = (mask_r > 0).astype(np.uint8)
                out_name = f"fig_{stem}_pid{pid}.png"
                Image.fromarray(img_r).save(os.path.join(img_dir, out_name))
                Image.fromarray(mask_r * 255).save(os.path.join(mask_dir, out_name))
                metadata_rows.append({
                    "file": out_name, "split": split_name, "pid": pid,
                    "label": label, "label_name": LABEL_NAMES[label],
                    "tumor_pixels": int(mask_r.sum()), "src": os.path.basename(path),
                })
                count += 1
        manifest["splits"][split_name] = pids
        print(f"[FIGSHARE] split '{split_name}': {len(pids)} patients, {count} slices", flush=True)

    with open(os.path.join(out_dir, "split_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    with open(os.path.join(out_dir, "metadata.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(metadata_rows[0].keys()))
        writer.writeheader()
        writerows = writer.writerows(metadata_rows)
    print(f"[FIGSHARE] Wrote manifest + metadata.csv ({len(metadata_rows)} rows) to {out_dir}", flush=True)
    return {"out_dir": out_dir, "manifest": manifest, "n_slices": len(metadata_rows)}


if __name__ == "__main__":
    run_figshare_preprocessing()
