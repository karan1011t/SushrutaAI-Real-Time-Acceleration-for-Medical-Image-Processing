"""
SUSHRUTA Two-Stage Pipeline & Dynamic Explanation Verification Test Script
"""

import requests
import os
import json

BASE_URL = "http://localhost:8000"

def test_prediction(slice_name, slice_path):
    print(f"\n==================================================")
    print(f"TESTING: {slice_name}")
    print(f"Path: {slice_path}")
    print(f"==================================================")

    if not os.path.exists(slice_path):
        print(f"[SKIPPED] Test file not found: {slice_path}")
        return

    with open(slice_path, "rb") as f:
        mime = "image/png" if slice_path.endswith(".png") else "image/jpeg"
        res = requests.post(
            f"{BASE_URL}/predict",
            files={"file": (os.path.basename(slice_path), f, mime)}
        )

    assert res.status_code == 200, f"Expected 200, got {res.status_code}: {res.text}"
    data = res.json()

    print(f"[ROOT] tumor_detected: {data.get('tumor_detected')} ({type(data.get('tumor_detected')).__name__})")

    cls = data.get("classification", {})
    print(f"[STAGE 1 CLASSIFICATION]")
    print(f"  - Label:               {cls.get('classification_label')}")
    print(f"  - Tumor Probability:   {cls.get('tumor_probability') * 100:.2f}%")
    print(f"  - Tumor Logit:         {cls.get('tumor_logit'):.4f}")
    print(f"  - Decision Threshold:  {cls.get('decision_threshold') * 100:.0f}%")
    print(f"  - Explanation:         \"{cls.get('explanation')}\"")

    xai = cls.get("xai", {})
    print(f"[CLASSIFIER LAYERCAM XAI]")
    print(f"  - Target Layer:        {xai.get('target_layer')}")
    print(f"  - Region Location:     {xai.get('strongest_region_location')}")
    print(f"  - Centroid:            {xai.get('centroid')}")
    print(f"  - Bounding Box:        {xai.get('bounding_box')}")

    seg = data.get("segmentation", {})
    print(f"[STAGE 2 SEGMENTATION]")
    print(f"  - Performed:           {seg.get('performed')}")

    meas = data.get("measurements", {})
    print(f"  - Area Pixels:         {meas.get('area_pixels')} px ({meas.get('area_percentage')}%)")
    print(f"  - Connected Islands:   {meas.get('num_connected_components')}")
    print(f"  - Confidence:          {seg.get('confidence')}")

    outputs = data.get("outputs", {})
    print(f"[VISUAL ARTIFACTS]")
    print(f"  - Preprocessed MRI:    {outputs.get('preprocessed_mri')}")
    print(f"  - Classifier LayerCAM: {outputs.get('classifier_layercam')}")
    print(f"  - Tumor Mask:          {outputs.get('tumor_mask')}")
    print(f"  - Segmentation Overlay:{outputs.get('overlay')}")
    print(f"  - LayerCAM Overlay:    {outputs.get('layercam_overlay')}")

if __name__ == "__main__":
    test_prediction("POSITIVE CONTROL (slice_063)", "data/processed/brats/val/images/BraTS2021_00002_slice_063.png")
    test_prediction("NEGATIVE CONTROL (slice_028)", "data/processed/brats/val/images/BraTS2021_00002_slice_028.png")

    # Optional external OOD image test case (skipped gracefully if fixture is absent)
    ood_sample_path = os.path.join("data", "external", "sample_brain.jpg")
    if os.path.exists(ood_sample_path):
        test_prediction("EXTERNAL OOD (sample_brain.jpg)", ood_sample_path)
    else:
        print("\n==================================================")
        print("TESTING: EXTERNAL OOD (sample_brain.jpg)")
        print(f"[SKIPPED] Optional OOD image fixture not present at {ood_sample_path}")
        print("==================================================")
