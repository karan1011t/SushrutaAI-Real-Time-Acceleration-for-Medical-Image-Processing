"""
Live API numerical consistency verification script.
Tests both positive and negative slices against http://127.0.0.1:8000/predict.
"""

import os
import requests
import cv2
import numpy as np

BASE_URL = "http://127.0.0.1:8000"

def verify_slice(name, rel_path):
    print("=" * 60)
    print(f"VERIFYING SLICE: {name} ({rel_path})")
    print("=" * 60)

    assert os.path.exists(rel_path), f"File not found: {rel_path}"
    filename = os.path.basename(rel_path)

    with open(rel_path, "rb") as f:
        resp = requests.post(
            f"{BASE_URL}/predict",
            files={"file": (filename, f, "image/png")}
        )

    assert resp.status_code == 200, f"Error {resp.status_code}: {resp.text}"
    data = resp.json()

    cls = data["classification"]
    meas = data["measurements"]
    inf = data["inference"]
    outputs = data["outputs"]
    xai = data["explainability"]

    print(f"[API RESPONSE SUMMARY]")
    print(f"  Root tumor_detected:          {data['tumor_detected']}")
    print(f"  Classifier tumor_detected:    {cls['tumor_detected']} (P = {cls['tumor_probability']*100:.2f}%, logit = {cls['tumor_logit']})")
    print(f"  Segmentation performed:       {data['segmentation']['performed']}")
    print(f"  Area pixels:                  {meas['area_pixels']} px")
    print(f"  Area percentage:              {meas['area_percentage']}%")
    print(f"  Num connected components:     {meas['num_connected_components']}")
    print(f"  Bounding box:                 {meas['bounding_box']}")
    print(f"  Centroid:                     {meas['centroid']}")
    print(f"  Confidence:                   {data['segmentation_confidence']}")
    print(f"  U-Net forward latency:        {inf['segmentation_inference_time_ms']} ms")
    print(f"  Total request latency:        {inf['total_request_time_ms']} ms")

    # 1. Verify area percentage formula: (area_pixels / (256 * 256)) * 100
    expected_pct = round((meas["area_pixels"] / (256 * 256)) * 100.0, 2)
    assert meas["area_percentage"] == expected_pct, f"Area % mismatch: {meas['area_percentage']} vs {expected_pct}"
    print(f"  [CHECK 1] Area % formula verified: {meas['area_percentage']}% == ({meas['area_pixels']} / 65536) * 100")

    # 2. Download returned binary mask image and count nonzero pixels
    mask_url = outputs["tumor_mask"]
    assert mask_url is not None, "Tumor mask URL is None"
    mask_resp = requests.get(mask_url)
    assert mask_resp.status_code == 200, f"Failed to download mask from {mask_url}"

    mask_bytes = np.frombuffer(mask_resp.content, np.uint8)
    mask_img = cv2.imdecode(mask_bytes, cv2.IMREAD_GRAYSCALE)
    assert mask_img.shape == (256, 256), f"Unexpected mask shape: {mask_img.shape}"

    nonzero_pixels = int(np.count_nonzero(mask_img > 127))
    assert nonzero_pixels == meas["area_pixels"], f"Mask pixel count mismatch: {nonzero_pixels} vs {meas['area_pixels']}"
    print(f"  [CHECK 2] Exact nonzero mask pixels: {nonzero_pixels} == area_pixels ({meas['area_pixels']})")

    # 3. Model agreement calculation check
    cls_flag = cls["tumor_detected"]
    seg_flag = meas["area_pixels"] > 0
    agreement = (cls_flag == seg_flag)
    print(f"  [CHECK 3] Model agreement: classifier={cls_flag}, seg={seg_flag} => agreement={agreement}")

    # 4. Check LayerCAM availability vs empty-mask
    if seg_flag:
        assert outputs["layercam_overlay"] is not None, "Positive mask must produce layercam_overlay"
        print(f"  [CHECK 4] Positive mask produced LayerCAM: {outputs['layercam_overlay']}")
    else:
        assert outputs["layercam"] is None, "Empty mask must have outputs.layercam == None"
        assert outputs["layercam_overlay"] is None, "Empty mask must have outputs.layercam_overlay == None"
        assert "Unavailable" in xai["target_definition"]
        print(f"  [CHECK 4] Empty mask produced safe unavailable LayerCAM status: {xai['target_definition']}")

    # 5. Check all image URLs are accessible
    for key in ["preprocessed_mri", "tumor_mask", "overlay", "probability_map", "classifier_layercam"]:
        url = outputs.get(key)
        assert url is not None, f"Missing output URL for {key}"
        img_r = requests.get(url)
        assert img_r.status_code == 200, f"Broken image URL for {key}: {url}"
    print(f"  [CHECK 5] All generated image URLs return HTTP 200 OK")

    return data

if __name__ == "__main__":
    pos_data = verify_slice("POSITIVE CONTROL", "data/processed/brats/val/images/BraTS2021_00002_slice_063.png")
    neg_data = verify_slice("NEGATIVE CONTROL", "data/processed/brats/val/images/BraTS2021_00002_slice_028.png")
    print("\n" + "=" * 60)
    print("ALL NUMERICAL AND URL CONSISTENCY CHECKS PASSED ON LIVE RUNNING API")
    print("=" * 60)
