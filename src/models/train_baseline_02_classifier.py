"""
SUSHRUTA — BASELINE-02: Binary Tumor Classification Training & Evaluation Engine
Trains a 2D CNN classifier on FLAIR MRI slices to predict Tumor (1) vs No Tumor (0).
"""

import os
import sys
import time
import json
import csv
import random
from typing import Dict, Any, Tuple, List

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add project root to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.models.classifier import BinaryTumorClassifier, count_parameters
from src.models.unet import UNet2D
from src.models.dataset import BraTS2DDataset
from src.models.train_unet import set_seed


def calculate_classification_metrics(preds: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
    """
    Computes binary classification metrics: Accuracy, Precision, Recall/Sensitivity, Specificity, F1.
    """
    p = (preds >= 0.5).long().view(-1)
    t = (targets >= 0.5).long().view(-1)

    tp = int(((p == 1) & (t == 1)).sum().item())
    tn = int(((p == 0) & (t == 0)).sum().item())
    fp = int(((p == 1) & (t == 0)).sum().item())
    fn = int(((p == 0) & (t == 1)).sum().item())

    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    denom_f1 = precision + recall
    f1 = (2.0 * precision * recall / denom_f1) if denom_f1 > 0 else 0.0

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "f1": float(f1),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn
    }


def plot_confusion_matrix(tn: int, fp: int, fn: int, tp: int, title: str, save_path: str):
    """
    Generates a clean 2x2 confusion matrix heatmap visualization.
    """
    cm = np.array([[tn, fp], [fn, tp]])
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(2),
        yticks=np.arange(2),
        xticklabels=["No Tumor (0)", "Tumor (1)"],
        yticklabels=["No Tumor (0)", "Tumor (1)"],
        title=title,
        ylabel="Actual Label",
        xlabel="Predicted Label"
    )

    plt.setp(ax.get_xticklabels(), rotation=0, ha="center")

    # Format numbers inside matrix cells
    fmt = "d"
    thresh = cm.max() / 2.0
    for i in range(2):
        for j in range(2):
            ax.text(
                j, i, format(cm[i, j], fmt),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=14, fontweight="bold"
            )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def run_baseline_02_classifier_pipeline():
    print("=" * 60, flush=True)
    print("SUSHRUTA — BASELINE-02: BINARY TUMOR CLASSIFICATION", flush=True)
    print("=" * 60, flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"Device: {device} ({gpu_name})", flush=True)

    set_seed(42)

    # Directories
    results_models_dir = os.path.join(project_root, "results", "models")
    results_metrics_dir = os.path.join(project_root, "results", "metrics")
    results_figures_dir = os.path.join(project_root, "results", "figures")
    data_dir = os.path.join(project_root, "data", "processed", "brats")

    os.makedirs(results_models_dir, exist_ok=True)
    os.makedirs(results_metrics_dir, exist_ok=True)
    os.makedirs(results_figures_dir, exist_ok=True)

    # -------------------------------------------------------------------------
    # STEP 1 — DATASET LABEL DEFINITION & DISTRIBUTION VERIFICATION
    # -------------------------------------------------------------------------
    print("\n" + "=" * 50, flush=True)
    print("STEP 1 & 2: DATASET LABEL DEFINITION & PATIENT ISOLATION", flush=True)
    print("=" * 50, flush=True)

    train_dataset = BraTS2DDataset(data_dir=data_dir, split="train", augment=False, preload=True)
    val_dataset = BraTS2DDataset(data_dir=data_dir, split="val", augment=False, preload=True)
    test_dataset = BraTS2DDataset(data_dir=data_dir, split="test", augment=False, preload=True)

    def compute_split_distribution(dataset: BraTS2DDataset) -> Tuple[int, int, float, float, torch.Tensor]:
        # Derive label strictly from GT mask: label = 1 if GT tumor pixels > 0 else 0
        mask_sums = dataset.masks_tensor.view(len(dataset), -1).sum(dim=1)
        labels = (mask_sums > 0).float()
        tumor_cnt = int(labels.sum().item())
        no_tumor_cnt = len(dataset) - tumor_cnt
        no_tumor_pct = (no_tumor_cnt / len(dataset)) * 100.0
        tumor_pct = (tumor_cnt / len(dataset)) * 100.0
        return no_tumor_cnt, tumor_cnt, no_tumor_pct, tumor_pct, labels

    tr_no_t, tr_t, tr_no_pct, tr_t_pct, train_labels = compute_split_distribution(train_dataset)
    val_no_t, val_t, val_no_pct, val_t_pct, val_labels = compute_split_distribution(val_dataset)
    te_no_t, te_t, te_no_pct, te_t_pct, test_labels = compute_split_distribution(test_dataset)

    print("\n--- DATASET CLASS DISTRIBUTIONS ---", flush=True)
    print(f"Train Split ({len(train_dataset):,} slices):", flush=True)
    print(f"  Class 0 (No Tumor): {tr_no_t:,} slices ({tr_no_pct:.2f}%)", flush=True)
    print(f"  Class 1 (Tumor):    {tr_t:,} slices ({tr_t_pct:.2f}%)\n", flush=True)

    print(f"Validation Split ({len(val_dataset):,} slices):", flush=True)
    print(f"  Class 0 (No Tumor): {val_no_t:,} slices ({val_no_pct:.2f}%)", flush=True)
    print(f"  Class 1 (Tumor):    {val_t:,} slices ({val_t_pct:.2f}%)\n", flush=True)

    print(f"Test Split ({len(test_dataset):,} slices):", flush=True)
    print(f"  Class 0 (No Tumor): {te_no_t:,} slices ({te_no_pct:.2f}%)", flush=True)
    print(f"  Class 1 (Tumor):    {te_t:,} slices ({te_t_pct:.2f}%)\n", flush=True)

    # Patient isolation verification
    tr_patients = set([f.split("_slice")[0] for f in train_dataset.image_filenames])
    val_patients = set([f.split("_slice")[0] for f in val_dataset.image_filenames])
    te_patients = set([f.split("_slice")[0] for f in test_dataset.image_filenames])

    print(f"Patient Counts: Train={len(tr_patients)}, Val={len(val_patients)}, Test={len(te_patients)}", flush=True)
    tr_val_overlap = tr_patients.intersection(val_patients)
    tr_te_overlap = tr_patients.intersection(te_patients)
    val_te_overlap = val_patients.intersection(te_patients)

    if len(tr_val_overlap) > 0 or len(tr_te_overlap) > 0 or len(val_te_overlap) > 0:
        raise RuntimeError("PATIENT ISOLATION FAILED: Patient ID overlap detected across splits!")

    print("Zero Patient Overlap Verification: PASSED", flush=True)

    # Fast Tensor Dataloaders for preloaded RAM tensors
    from torch.utils.data import TensorDataset

    batch_size = 32
    train_tensor_ds = TensorDataset(train_dataset.images_tensor, train_labels.unsqueeze(1))
    val_tensor_ds = TensorDataset(val_dataset.images_tensor, val_labels.unsqueeze(1))
    test_tensor_ds = TensorDataset(test_dataset.images_tensor, test_labels.unsqueeze(1))

    fast_train_loader = DataLoader(train_tensor_ds, batch_size=batch_size, shuffle=True, pin_memory=True)
    fast_val_loader = DataLoader(val_tensor_ds, batch_size=batch_size, shuffle=False, pin_memory=True)
    fast_test_loader = DataLoader(test_tensor_ds, batch_size=batch_size, shuffle=False, pin_memory=True)

    # -------------------------------------------------------------------------
    # STEP 3 — CLASSIFIER DESIGN & PRE-TRAINING REPORT
    # -------------------------------------------------------------------------
    print("\n" + "=" * 50, flush=True)
    print("STEP 3: CLASSIFIER DESIGN & HYPERPARAMETERS", flush=True)
    print("=" * 50, flush=True)

    set_seed(42)
    classifier = BinaryTumorClassifier(in_channels=1, base_channels=32).to(device)
    num_params = count_parameters(classifier)

    lr = 1e-3
    epochs = 5
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    print(f"Model Architecture:   BinaryTumorClassifier (4-Block 2D CNN)", flush=True)
    print(f"Total Parameters:     {num_params:,}", flush=True)
    print(f"Input Shape:          1 x 256 x 256", flush=True)
    print(f"Output Shape:         1 (Raw Logit)", flush=True)
    print(f"Loss Function:        BCEWithLogitsLoss", flush=True)
    print(f"Optimizer:            AdamW (lr={lr}, weight_decay=1e-4)", flush=True)
    print(f"Batch Size:           {batch_size}", flush=True)
    print(f"Epochs:               {epochs}", flush=True)
    print(f"Device:               {device} ({gpu_name})", flush=True)

    # -------------------------------------------------------------------------
    # STEP 4 — TRAINING & MODEL SELECTION
    # -------------------------------------------------------------------------
    print("\n" + "=" * 50, flush=True)
    print("STEP 4: CLASSIFIER TRAINING & VALIDATION", flush=True)
    print("=" * 50, flush=True)

    best_ckpt_path = os.path.join(results_models_dir, "baseline_02_classifier_best.pt")
    latest_ckpt_path = os.path.join(results_models_dir, "baseline_02_classifier_latest.pt")

    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_precision": [],
        "val_recall": [],
        "val_specificity": [],
        "val_f1": []
    }

    best_val_f1 = -1.0
    best_epoch = 0

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        # Training loop
        classifier.train()
        running_train_loss = torch.tensor(0.0, device=device)
        train_samples = 0

        for imgs, labels in fast_train_loader:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            if device.type == "cuda":
                with torch.amp.autocast("cuda"):
                    logits = classifier(imgs)
                    loss = criterion(logits, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = classifier(imgs)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

            running_train_loss += loss.detach() * imgs.size(0)
            train_samples += imgs.size(0)

        epoch_train_loss = (running_train_loss / max(1, train_samples)).item()

        # Validation loop
        classifier.eval()
        running_val_loss = torch.tensor(0.0, device=device)
        val_samples = 0
        all_val_probs = []
        all_val_labels = []

        with torch.no_grad():
            for imgs, labels in fast_val_loader:
                imgs = imgs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                if device.type == "cuda":
                    with torch.amp.autocast("cuda"):
                        logits = classifier(imgs)
                        loss = criterion(logits, labels)
                else:
                    logits = classifier(imgs)
                    loss = criterion(logits, labels)

                running_val_loss += loss.detach() * imgs.size(0)
                val_samples += imgs.size(0)

                probs = torch.sigmoid(logits)
                all_val_probs.append(probs.cpu())
                all_val_labels.append(labels.cpu())

        epoch_val_loss = (running_val_loss / max(1, val_samples)).item()
        val_probs_tensor = torch.cat(all_val_probs, dim=0)
        val_labels_tensor = torch.cat(all_val_labels, dim=0)

        val_m = calculate_classification_metrics(val_probs_tensor, val_labels_tensor)

        history["epoch"].append(epoch)
        history["train_loss"].append(epoch_train_loss)
        history["val_loss"].append(epoch_val_loss)
        history["val_accuracy"].append(val_m["accuracy"])
        history["val_precision"].append(val_m["precision"])
        history["val_recall"].append(val_m["recall"])
        history["val_specificity"].append(val_m["specificity"])
        history["val_f1"].append(val_m["f1"])

        elapsed = time.time() - t0

        print(f"Epoch {epoch}/{epochs} ({elapsed:.1f}s) - Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f} | Val F1: {val_m['f1']:.4f} | Val Acc: {val_m['accuracy']:.4f}", flush=True)

        ckpt_dict = {
            "epoch": epoch,
            "model_state_dict": classifier.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": epoch_val_loss,
            "val_f1": val_m["f1"],
            "val_accuracy": val_m["accuracy"],
            "val_precision": val_m["precision"],
            "val_recall": val_m["recall"],
            "val_specificity": val_m["specificity"]
        }

        torch.save(ckpt_dict, latest_ckpt_path)

        if val_m["f1"] > best_val_f1:
            best_val_f1 = val_m["f1"]
            best_epoch = epoch
            torch.save(ckpt_dict, best_ckpt_path)
            print(f" -> NEW BEST MODEL SAVED (Epoch {epoch}, Val F1: {val_m['f1']:.4f})", flush=True)

    # Save History CSV
    csv_path = os.path.join(results_metrics_dir, "baseline_02_classifier_history.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "val_accuracy", "val_precision", "val_recall", "val_specificity", "val_f1"])
        for i in range(len(history["epoch"])):
            writer.writerow([
                history["epoch"][i], history["train_loss"][i], history["val_loss"][i],
                history["val_accuracy"][i], history["val_precision"][i], history["val_recall"][i],
                history["val_specificity"][i], history["val_f1"][i]
            ])
    print(f"Saved classifier training history to: {csv_path}", flush=True)

    # Save Config JSON
    config_dict = {
        "experiment_name": "BASELINE-02: Binary Tumor Classification",
        "model_name": "BinaryTumorClassifier",
        "total_parameters": num_params,
        "in_channels": 1,
        "base_channels": 32,
        "batch_size": batch_size,
        "epochs": epochs,
        "best_epoch": best_epoch,
        "best_val_f1": best_val_f1,
        "learning_rate": lr,
        "optimizer": "AdamW",
        "loss_function": "BCEWithLogitsLoss",
        "seed": 42,
        "device": str(device),
        "gpu_name": gpu_name,
        "dataset_summary": {
            "train_slices": len(train_dataset),
            "val_slices": len(val_dataset),
            "test_slices": len(test_dataset)
        }
    }
    config_path = os.path.join(results_metrics_dir, "baseline_02_classifier_config.json")
    with open(config_path, "w") as f:
        json.dump(config_dict, f, indent=2)
    print(f"Saved configuration to: {config_path}", flush=True)

    # Plot Training Curves
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history["epoch"], history["train_loss"], label="Train Loss", color="#1f77b4", linewidth=2, marker="o")
    plt.plot(history["epoch"], history["val_loss"], label="Val Loss", color="#ff7f0e", linewidth=2, marker="o")
    plt.title("Classifier Loss (5 Epochs)", fontsize=12, fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history["epoch"], history["val_f1"], label="Val F1 Score", color="#2ca02c", linewidth=2, marker="o")
    plt.plot(history["epoch"], history["val_accuracy"], label="Val Accuracy", color="#d62728", linewidth=2, marker="o")
    plt.title("Classifier Validation Performance", fontsize=12, fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()

    plt.tight_layout()
    curves_path = os.path.join(results_figures_dir, "baseline_02_classifier_training_curves.png")
    plt.savefig(curves_path, dpi=300)
    plt.close()
    print(f"Saved training curves to: {curves_path}", flush=True)

    # -------------------------------------------------------------------------
    # STEP 5 — VALIDATION METRICS & CONFUSION MATRIX
    # -------------------------------------------------------------------------
    print("\n" + "=" * 50, flush=True)
    print("STEP 5: BEST MODEL VALIDATION METRICS & CONFUSION MATRIX", flush=True)
    print("=" * 50, flush=True)

    best_ckpt = torch.load(best_ckpt_path, map_location=device)
    classifier.load_state_dict(best_ckpt["model_state_dict"])
    classifier.eval()

    val_probs_list = []
    val_labels_list = []
    with torch.no_grad():
        for imgs, labels in fast_val_loader:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if device.type == "cuda":
                with torch.amp.autocast("cuda"):
                    logits = classifier(imgs)
            else:
                logits = classifier(imgs)

            probs = torch.sigmoid(logits)
            val_probs_list.append(probs.cpu())
            val_labels_list.append(labels.cpu())

    val_probs = torch.cat(val_probs_list, dim=0)
    val_labels = torch.cat(val_labels_list, dim=0)

    val_final_m = calculate_classification_metrics(val_probs, val_labels)

    print(f"Best Validation Epoch:     {best_epoch}", flush=True)
    print(f"Validation Accuracy:       {val_final_m['accuracy']:.4f}", flush=True)
    print(f"Validation Precision:      {val_final_m['precision']:.4f}", flush=True)
    print(f"Validation Recall/Sens:    {val_final_m['recall']:.4f}", flush=True)
    print(f"Validation Specificity:    {val_final_m['specificity']:.4f}", flush=True)
    print(f"Validation F1 Score:       {val_final_m['f1']:.4f}\n", flush=True)

    print(f"Confusion Matrix (Val):    TN={val_final_m['tn']}, FP={val_final_m['fp']}, FN={val_final_m['fn']}, TP={val_final_m['tp']}", flush=True)

    val_cm_path = os.path.join(results_figures_dir, "baseline_02_classifier_val_confusion_matrix.png")
    plot_confusion_matrix(
        val_final_m["tn"], val_final_m["fp"], val_final_m["fn"], val_final_m["tp"],
        f"Validation Confusion Matrix (Epoch {best_epoch})", val_cm_path
    )
    print(f"Saved validation confusion matrix to: {val_cm_path}", flush=True)

    # -------------------------------------------------------------------------
    # STEP 6 — HELD-OUT TEST EVALUATION
    # -------------------------------------------------------------------------
    print("\n" + "=" * 50, flush=True)
    print("STEP 6: HELD-OUT TEST EVALUATION", flush=True)
    print("=" * 50, flush=True)

    test_probs_list = []
    test_labels_list = []

    with torch.no_grad():
        for imgs, labels in fast_test_loader:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if device.type == "cuda":
                with torch.amp.autocast("cuda"):
                    logits = classifier(imgs)
            else:
                logits = classifier(imgs)

            probs = torch.sigmoid(logits)
            test_probs_list.append(probs.cpu())
            test_labels_list.append(labels.cpu())

    test_probs = torch.cat(test_probs_list, dim=0)
    test_labels = torch.cat(test_labels_list, dim=0)

    test_m = calculate_classification_metrics(test_probs, test_labels)

    print(f"Test Accuracy:            {test_m['accuracy']:.4f}", flush=True)
    print(f"Test Precision:           {test_m['precision']:.4f}", flush=True)
    print(f"Test Recall/Sensitivity: {test_m['recall']:.4f}", flush=True)
    print(f"Test Specificity:         {test_m['specificity']:.4f}", flush=True)
    print(f"Test F1 Score:            {test_m['f1']:.4f}\n", flush=True)

    print(f"Confusion Matrix (Test):  TN={test_m['tn']}, FP={test_m['fp']}, FN={test_m['fn']}, TP={test_m['tp']}", flush=True)

    test_cm_path = os.path.join(results_figures_dir, "baseline_02_classifier_test_confusion_matrix.png")
    plot_confusion_matrix(
        test_m["tn"], test_m["fp"], test_m["fn"], test_m["tp"],
        "Held-Out Test Confusion Matrix (BASELINE-02 Classifier)", test_cm_path
    )
    print(f"Saved test confusion matrix to: {test_cm_path}", flush=True)

    # Save Test Metrics JSON & CSV
    test_results_dict = {
        "milestone": "BASELINE-02: Binary Tumor Classification",
        "best_epoch": best_epoch,
        "val_f1": val_final_m["f1"],
        "test_dataset": {
            "test_slices": len(test_dataset),
            "no_tumor_slices": te_no_t,
            "tumor_slices": te_t
        },
        "test_confusion_matrix": {
            "tn": test_m["tn"],
            "fp": test_m["fp"],
            "fn": test_m["fn"],
            "tp": test_m["tp"]
        },
        "test_metrics": {
            "accuracy": test_m["accuracy"],
            "precision": test_m["precision"],
            "recall_sensitivity": test_m["recall"],
            "specificity": test_m["specificity"],
            "f1": test_m["f1"]
        }
    }

    test_json_path = os.path.join(results_metrics_dir, "baseline_02_classifier_test_metrics.json")
    with open(test_json_path, "w") as f:
        json.dump(test_results_dict, f, indent=2)
    print(f"Saved test metrics JSON to: {test_json_path}", flush=True)

    test_csv_path = os.path.join(results_metrics_dir, "baseline_02_classifier_test_metrics.csv")
    with open(test_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric_category", "metric_name", "value"])
        writer.writerow(["checkpoint", "best_epoch", best_epoch])
        writer.writerow(["confusion_matrix", "tn", test_m["tn"]])
        writer.writerow(["confusion_matrix", "fp", test_m["fp"]])
        writer.writerow(["confusion_matrix", "fn", test_m["fn"]])
        writer.writerow(["confusion_matrix", "tp", test_m["tp"]])
        writer.writerow(["test_metrics", "accuracy", test_m["accuracy"]])
        writer.writerow(["test_metrics", "precision", test_m["precision"]])
        writer.writerow(["test_metrics", "recall_sensitivity", test_m["recall"]])
        writer.writerow(["test_metrics", "specificity", test_m["specificity"]])
        writer.writerow(["test_metrics", "f1", test_m["f1"]])
    print(f"Saved test metrics CSV to:  {test_csv_path}", flush=True)

    # -------------------------------------------------------------------------
    # STEP 7 — CLASSIFICATION + SEGMENTATION CONSISTENCY CHECK
    # -------------------------------------------------------------------------
    print("\n" + "=" * 50, flush=True)
    print("STEP 7: CLASSIFICATION + SEGMENTATION CONSISTENCY CHECK", flush=True)
    print("=" * 50, flush=True)

    unet_ckpt_path = os.path.join(results_models_dir, "baseline_01_clean_v2_best.pt")
    if not os.path.exists(unet_ckpt_path):
        raise FileNotFoundError(f"Missing BASELINE-01 checkpoint at {unet_ckpt_path}")

    unet_model = UNet2D(in_channels=1, out_channels=1, base_channels=32).to(device)
    unet_ckpt = torch.load(unet_ckpt_path, map_location=device)
    unet_model.load_state_dict(unet_ckpt["model_state_dict"])
    unet_model.eval()

    # Sample 12 representative test slices (6 Tumor, 6 No Tumor) across distinct patients
    test_filenames = test_dataset.image_filenames
    sample_indices = []
    seen_patients = set()
    tumor_samples = []
    empty_samples = []

    for idx in range(len(test_dataset)):
        fname = test_filenames[idx]
        patient_id = fname.split("_slice")[0]
        mask_sum = float(torch.sum(test_dataset.masks_tensor[idx] > 0.5).item())
        
        if patient_id not in seen_patients:
            if mask_sum > 100 and len(tumor_samples) < 6:
                tumor_samples.append(idx)
                seen_patients.add(patient_id)
            elif mask_sum == 0 and len(empty_samples) < 6:
                empty_samples.append(idx)
                seen_patients.add(patient_id)

        if len(tumor_samples) >= 6 and len(empty_samples) >= 6:
            break

    selected_indices = tumor_samples + empty_samples

    fig, axes = plt.subplots(12, 5, figsize=(18, 38))
    fig.suptitle("BASELINE-02: Classification + Segmentation Consistency Comparison", fontsize=16, fontweight="bold", y=0.995)

    headers = [
        "1. FLAIR Image",
        "2. GT Mask",
        "3. U-Net Pred Mask",
        "4. Classifier Prediction",
        "5. Classification GT"
    ]
    for col_idx, text in enumerate(headers):
        axes[0, col_idx].set_title(text, fontsize=11, fontweight="bold")

    with torch.no_grad():
        for row_idx, idx in enumerate(selected_indices):
            img_t, mask_t = test_dataset[idx]
            fname = test_filenames[idx]
            patient_id = fname.split("_slice")[0]

            img_input = img_t.unsqueeze(0).to(device)

            if device.type == "cuda":
                with torch.amp.autocast("cuda"):
                    unet_logits = unet_model(img_input)
                    cls_logits = classifier(img_input)
            else:
                unet_logits = unet_model(img_input)
                cls_logits = classifier(img_input)

            unet_prob = torch.sigmoid(unet_logits).squeeze().cpu().numpy()
            unet_pred = (unet_prob >= 0.5).astype(np.float32)

            cls_prob = float(torch.sigmoid(cls_logits).item())
            cls_pred_label = 1 if cls_prob >= 0.5 else 0
            gt_cls_label = 1 if (mask_t > 0.5).sum() > 0 else 0

            flair_np = img_t.squeeze().numpy()
            gt_np = mask_t.squeeze().numpy()

            # Visual panel renderings
            axes[row_idx, 0].imshow(flair_np, cmap="gray")
            axes[row_idx, 0].set_ylabel(f"{patient_id}", fontsize=9, fontweight="bold")

            axes[row_idx, 1].imshow(gt_np, cmap="gray")
            axes[row_idx, 2].imshow(unet_pred, cmap="gray")

            # Classification prediction visual banner
            cls_color = "green" if cls_pred_label == gt_cls_label else "red"
            axes[row_idx, 3].text(
                0.5, 0.5,
                f"Pred Class: {cls_pred_label}\n({cls_prob * 100:.1f}% Tumor Prob)",
                ha="center", va="center", fontsize=11, fontweight="bold", color=cls_color,
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.5)
            )
            axes[row_idx, 3].set_xlim(0, 1)
            axes[row_idx, 3].set_ylim(0, 1)
            axes[row_idx, 3].axis("off")

            # GT Classification label banner
            gt_str = "Class 1 (Tumor)" if gt_cls_label == 1 else "Class 0 (No Tumor)"
            axes[row_idx, 4].text(
                0.5, 0.5,
                f"GT: {gt_str}",
                ha="center", va="center", fontsize=11, fontweight="bold", color="black",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightcyan", alpha=0.5)
            )
            axes[row_idx, 4].set_xlim(0, 1)
            axes[row_idx, 4].set_ylim(0, 1)
            axes[row_idx, 4].axis("off")

            for c in range(3):
                axes[row_idx, c].set_xticks([])
                axes[row_idx, c].set_yticks([])

    plt.tight_layout()
    comparison_fig_path = os.path.join(results_figures_dir, "baseline_02_classification_segmentation_comparison.png")
    plt.savefig(comparison_fig_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved comparison figure to: {comparison_fig_path}", flush=True)

    # -------------------------------------------------------------------------
    # STEP 8 — SANITY CHECK & FINAL SUMMARY OUTPUT
    # -------------------------------------------------------------------------
    print("\n" + "=" * 50, flush=True)
    print("STEP 8: SANITY CHECK & FINAL SUMMARY REPORT", flush=True)
    print("=" * 50, flush=True)

    # Verification assertions before final output
    assert test_m["tp"] > 0 and test_m["tn"] > 0, "SANITY CHECK FAILED: Classifier collapsed to single class predictions!"
    print("Sanity Check: Non-trivial multi-class prediction confirmed. PASSED.", flush=True)

    final_report_text = f"""==================================================
SUSHRUTA — BASELINE-02
BINARY TUMOR CLASSIFICATION
==================================================

DATA
Train slices: {len(train_dataset):,}
Validation slices: {len(val_dataset):,}
Test slices: {len(test_dataset):,}

TRAIN CLASS DISTRIBUTION
No Tumor: {tr_no_t:,} ({tr_no_pct:.2f}%)
Tumor: {tr_t:,} ({tr_t_pct:.2f}%)

VALIDATION CLASS DISTRIBUTION
No Tumor: {val_no_t:,} ({val_no_pct:.2f}%)
Tumor: {val_t:,} ({val_t_pct:.2f}%)

TEST CLASS DISTRIBUTION
No Tumor: {te_no_t:,} ({te_no_pct:.2f}%)
Tumor: {te_t:,} ({te_t_pct:.2f}%)

MODEL
Parameters: {num_params:,}
Input: 1 x 256 x 256
Output: 1 (Raw Logit)
Loss: BCEWithLogitsLoss
Optimizer: AdamW (lr={lr})
Device: {device}
GPU: {gpu_name}

BEST EPOCH: {best_epoch}
Validation F1: {best_val_f1:.4f}

HELD-OUT TEST CONFUSION MATRIX

                 Predicted
              No Tumor  Tumor

Actual
No Tumor        {test_m['tn']:<8} {test_m['fp']:<8}
Tumor           {test_m['fn']:<8} {test_m['tp']:<8}

TEST METRICS
Accuracy: {test_m['accuracy']:.4f}
Precision: {test_m['precision']:.4f}
Recall/Sensitivity: {test_m['recall']:.4f}
Specificity: {test_m['specificity']:.4f}
F1: {test_m['f1']:.4f}

==================================================
"""

    print("\n" + final_report_text, flush=True)


if __name__ == "__main__":
    run_baseline_02_classifier_pipeline()
