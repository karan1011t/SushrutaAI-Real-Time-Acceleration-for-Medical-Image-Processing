"""
Deep Forensic Audit Script for Sushruta Dataset Provenance and Model Validation.
Read-Only: Analyzes intensity distributions, segmentation masks, split membership, array hashes,
header affines, and metadata across raw cases and processed directories.
Outputs structured findings to stdout.
"""

import json
import csv
import hashlib
from pathlib import Path
import numpy as np
import nibabel as nib

def sha256_array(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()

def get_brain_mask():
    X, Y, Z = np.ogrid[:240, :240, :155]
    return ((X - 120)**2 / 70**2 + (Y - 120)**2 / 80**2 + (Z - 77)**2 / 50**2) <= 1.0

def main():
    root = Path(".")
    raw_dir = root / "data" / "raw" / "brats"
    processed_dir = root / "data" / "processed" / "brats"
    manifest_path = processed_dir / "split_manifest.json"
    metadata_path = processed_dir / "metadata.csv"

    brain_mask = get_brain_mask()

    print("=== SUSHRUTA DEEP FORENSIC ANALYSIS ===")

    # 1. Manifest Analysis
    manifest = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    train_manifest_set = set(manifest.get("train", []))
    val_manifest_set = set(manifest.get("val", []))
    test_manifest_set = set(manifest.get("test", []))

    # 2. Metadata.csv Analysis
    metadata_rows = []
    metadata_patient_splits = {}
    metadata_patient_slices = {}
    if metadata_path.exists():
        with open(metadata_path, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for r in reader:
                metadata_rows.append(r)
                pid = r["patient_id"].strip()
                s = r["split"].strip().lower()
                metadata_patient_splits.setdefault(pid, set()).add(s)
                metadata_patient_slices[pid] = metadata_patient_slices.get(pid, 0) + 1

    # 3. Analyze all 360 raw cases
    cases_24 = [f"BraTS2021_{i:05d}" for i in range(24)]

    print("\n--- FORENSIC AUDIT OF THE 24 UNMATCHED CASES (BraTS2021_00000 to BraTS2021_00023) ---")

    seg_hashes_24 = {}
    flair_hashes_24 = {}
    intensity_stats_24 = []

    for cid in cases_24:
        folder = raw_dir / cid
        if not folder.exists(): continue
        flair_files = list(folder.glob("*flair.nii*"))
        seg_files = list(folder.glob("*seg.nii*"))

        if not flair_files or not seg_files: continue

        flair_img = nib.load(str(flair_files[0]))
        seg_img = nib.load(str(seg_files[0]))

        flair_arr = np.asarray(flair_img.dataobj, dtype=np.float32)
        seg_arr = np.asarray(seg_img.dataobj)

        f_hash = sha256_array(flair_arr)
        s_hash = sha256_array(seg_arr)

        flair_hashes_24[cid] = f_hash
        seg_hashes_24[cid] = s_hash

        nonzero = flair_arr[flair_arr != 0]
        p = np.percentile(nonzero, [0, 1, 25, 50, 75, 99, 100]) if nonzero.size > 0 else np.zeros(7)
        outside_nonzero = np.count_nonzero((flair_arr != 0) & (~brain_mask))
        seg_labels = dict(zip(*np.unique(seg_arr, return_counts=True)))

        intensity_stats_24.append({
            "case": cid,
            "shape": flair_arr.shape,
            "dtype": str(flair_img.get_data_dtype()),
            "affine_is_identity": np.array_equal(flair_img.affine, np.eye(4)),
            "nonzero_count": nonzero.size,
            "outside_nonzero_brain": outside_nonzero,
            "min_max_mean": (flair_arr.min(), flair_arr.max(), flair_arr.mean()),
            "percentiles": p.round(2).tolist(),
            "seg_labels": seg_labels,
            "flair_hash": f_hash[:12],
            "seg_hash": s_hash[:12]
        })

    if intensity_stats_24:
        unique_seg_hashes_24 = set(seg_hashes_24.values())
        unique_flair_hashes_24 = set(flair_hashes_24.values())

        print(f"Total unmatched cases: {len(intensity_stats_24)}")
        print(f"Unique FLAIR array hashes among 24 cases: {len(unique_flair_hashes_24)}")
        print(f"Unique SEG array hashes among 24 cases: {len(unique_seg_hashes_24)}")
        print(f"Affine matrix identity check: All cases identity? {all(x['affine_is_identity'] for x in intensity_stats_24)}")

    print("\n=== FORENSIC ANALYSIS COMPLETE ===")

if __name__ == "__main__":
    main()
