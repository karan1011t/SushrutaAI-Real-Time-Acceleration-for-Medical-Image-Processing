import os
import json
import csv
import hashlib
from pathlib import Path
import numpy as np
import nibabel as nib

def get_brain_mask():
    X, Y, Z = np.ogrid[:240, :240, :155]
    return ((X - 120)**2 / 70**2 + (Y - 120)**2 / 80**2 + (Z - 77)**2 / 50**2) <= 1.0

def main():
    root = Path(r"C:\Users\HP\OneDrive\Desktop\Projects\Sushruta")
    raw_dir = root / "data" / "raw" / "brats"
    processed_dir = root / "data" / "processed" / "brats"
    manifest_path = processed_dir / "split_manifest.json"
    metadata_path = processed_dir / "metadata.csv"
    out_csv = root / "results" / "audit" / "DATASET_CASE_INVENTORY.csv"

    manifest_splits = {}
    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            mdata = json.load(f)
            for s in ["train", "val", "test"]:
                for case_id in mdata.get(s, []):
                    manifest_splits[case_id] = s

    metadata_splits = {}
    slice_counts = {}
    if metadata_path.exists():
        with open(metadata_path, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                pid = row["patient_id"].strip()
                split = row["split"].strip()
                metadata_splits.setdefault(pid, set()).add(split)
                slice_counts[pid] = slice_counts.get(pid, 0) + 1

    brain_mask = get_brain_mask()
    patient_dirs = sorted([d for d in raw_dir.iterdir() if d.is_dir()])
    rows = []

    for folder in patient_dirs:
        case_id = folder.name
        flair_files = list(folder.glob("*flair.nii*"))
        seg_files = list(folder.glob("*seg.nii*"))
        t1_files = list(folder.glob("*t1.nii*"))
        t1ce_files = list(folder.glob("*t1ce.nii*"))
        t2_files = list(folder.glob("*t2.nii*"))

        has_flair = len(flair_files) > 0
        has_seg = len(seg_files) > 0
        has_t1 = len(t1_files) > 0
        has_t1ce = len(t1ce_files) > 0
        has_t2 = len(t2_files) > 0

        mods = []
        if has_flair: mods.append("FLAIR")
        if has_t1: mods.append("T1")
        if has_t1ce: mods.append("T1CE")
        if has_t2: mods.append("T2")
        if has_seg: mods.append("SEG")
        mod_str = "+".join(mods)

        if not (has_flair and has_seg):
            rows.append({
                "case_id": case_id, "status": "Incomplete", "available_modalities": mod_str,
                "has_flair": has_flair, "has_t1": has_t1, "has_t1ce": has_t1ce, "has_t2": has_t2, "has_seg": has_seg,
                "volume_shape": "N/A", "datatype": "N/A", "qform_code": -1, "sform_code": -1,
                "flair_hash": "N/A", "seg_hash": "N/A",
                "split_manifest_assignment": manifest_splits.get(case_id, "Unlisted"),
                "processed_metadata_splits": "|".join(sorted(metadata_splits.get(case_id, ["None"]))),
                "processed_slice_count": slice_counts.get(case_id, 0),
                "outside_nonzero_voxels": -1, "non_tumor_flair_is_exact_400": False,
                "fingerprint_result": "Incomplete", "audit_classification": "Incomplete Case (Missing SEG)"
            })
            continue

        flair_img = nib.load(str(flair_files[0]))
        seg_img = nib.load(str(seg_files[0]))

        flair_arr = np.asarray(flair_img.dataobj, dtype=np.float32)
        seg_arr = np.asarray(seg_img.dataobj)

        shape = flair_arr.shape
        dtype_str = str(flair_img.get_data_dtype())
        qform = int(flair_img.header["qform_code"]) if "qform_code" in flair_img.header else -1
        sform = int(flair_img.header["sform_code"]) if "sform_code" in flair_img.header else -1

        f_hash = hashlib.sha256(np.ascontiguousarray(flair_arr).tobytes()).hexdigest()[:16]
        s_hash = hashlib.sha256(np.ascontiguousarray(seg_arr).tobytes()).hexdigest()[:16]

        outside_nonzero = int(np.count_nonzero((flair_arr != 0) & (~brain_mask)))
        non_tumor_voxels = flair_arr[brain_mask & (seg_arr == 0)]
        non_tumor_exact_400 = bool(np.all(non_tumor_voxels == 400.0)) if non_tumor_voxels.size > 0 else False

        is_exact_match = (outside_nonzero == 0 and non_tumor_exact_400)
        fp_res = "EXACT_SYNTHETIC_GENERATOR_MATCH" if is_exact_match else "NO_EXACT_GENERATOR_MATCH"
        audit_cls = "Definitive Synthetic (Matches generator math exactly)" if is_exact_match else "Unmatched Provenance (Detailed investigation needed)"

        rows.append({
            "case_id": case_id, "status": "Complete", "available_modalities": mod_str,
            "has_flair": has_flair, "has_t1": has_t1, "has_t1ce": has_t1ce, "has_t2": has_t2, "has_seg": has_seg,
            "volume_shape": "x".join(map(str, shape)), "datatype": dtype_str, "qform_code": qform, "sform_code": sform,
            "flair_hash": f_hash, "seg_hash": s_hash,
            "split_manifest_assignment": manifest_splits.get(case_id, "Unlisted"),
            "processed_metadata_splits": "|".join(sorted(metadata_splits.get(case_id, ["None"]))),
            "processed_slice_count": slice_counts.get(case_id, 0),
            "outside_nonzero_voxels": outside_nonzero,
            "non_tumor_flair_is_exact_400": non_tumor_exact_400,
            "fingerprint_result": fp_res, "audit_classification": audit_cls
        })

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        f.flush()
        os.fsync(f.fileno())

    print(f"SUCCESS: Wrote {len(rows)} patient rows to {out_csv.resolve()}")

if __name__ == "__main__":
    main()
