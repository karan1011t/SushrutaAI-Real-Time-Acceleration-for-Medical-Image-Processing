"""
Figshare image-level tumor classifier training.

BinaryTumorClassifier (same arch as Baseline-02) trained IN-DOMAIN on
Figshare T1 slices: tumor PNGs (label 1) + healthy PNGs (label 0) from
data/processed/figshare/. Handles ~5:1 imbalance with BCE pos_weight.
Saves results/models/figshare_classifier_best.pt (val-F1 selection).

Outputs:
  results/models/figshare_classifier_best.pt
  results/metrics/figshare_classifier_history.csv
  results/metrics/figshare_classifier_config.json
  results/metrics/figshare_classifier_test_metrics.json
"""

import csv
import glob
import json
import os
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.models.classifier import BinaryTumorClassifier, count_parameters
from src.models.train_unet import set_seed


class FigshareClsDataset(Dataset):
    """Grayscale PNGs -> (1,256,256) float + {0,1} label. healthy_*.png = 0."""

    def __init__(self, img_dir: str, augment: bool = False, seed: int = 42):
        self.files = sorted(glob.glob(os.path.join(img_dir, "*.png")))
        assert self.files, f"no PNGs in {img_dir}"
        self.augment = augment
        self.seed = seed

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        img = np.array(Image.open(path).convert("L"), dtype=np.float32) / 255.0
        label = 0.0 if os.path.basename(path).startswith("healthy_") else 1.0
        t = torch.from_numpy(img).unsqueeze(0)
        if self.augment:
            rng = np.random.default_rng(self.seed * 1_000_003 + idx)
            if rng.random() > 0.5:
                t = torch.flip(t, dims=[2])
            k = int(rng.integers(0, 4))
            if k:
                t = torch.rot90(t, k, dims=[1, 2])
        return t.float(), torch.tensor([label], dtype=torch.float32)


def cls_metrics(y_true: np.ndarray, y_prob: np.ndarray, thr: float = 0.5) -> dict:
    pred = (y_prob >= thr).astype(int)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    n = len(y_true)
    return {"accuracy": (tp + tn) / n, "precision": tp / max(1, tp + fp),
            "recall": tp / max(1, tp + fn), "specificity": tn / max(1, tn + fp),
            "f1": 2 * tp / max(1, 2 * tp + fp + fn),
            "tn": tn, "fp": fp, "fn": fn, "tp": tp, "n": n}


@torch.no_grad()
def evaluate(model, loader, device, criterion):
    model.eval()
    loss_sum, n, probs, trues = 0.0, 0, [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss_sum += criterion(logits, y).item() * x.size(0)
        n += x.size(0)
        probs.append(torch.sigmoid(logits).cpu().numpy())
        trues.append(y.cpu().numpy())
    probs = np.concatenate(probs).flatten()
    trues = np.concatenate(trues).flatten()
    return loss_sum / n, cls_metrics(trues, probs)


def run_figshare_classifier(
    data_dir: str = None, epochs: int = 10, batch_size: int = 32,
    learning_rate: float = 1e-3, patience: int = 4, seed: int = 42,
) -> dict:
    set_seed(seed)
    data_dir = data_dir or os.path.join(project_root, "data", "processed", "figshare")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[FIGSHARE-CLS] device={device} epochs={epochs} batch={batch_size} lr={learning_rate}", flush=True)

    train_ds = FigshareClsDataset(os.path.join(data_dir, "train", "images"), augment=True, seed=seed)
    val_ds = FigshareClsDataset(os.path.join(data_dir, "val", "images"), seed=seed)
    test_ds = FigshareClsDataset(os.path.join(data_dir, "test", "images"), seed=seed)
    pin = torch.cuda.is_available()
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=pin)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=pin)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=pin)

    n_pos = sum(1 for f in train_ds.files if not os.path.basename(f).startswith("healthy_"))
    n_neg = len(train_ds) - n_pos
    pos_weight = torch.tensor([n_pos / max(1, n_neg)], device=device)
    print(f"[FIGSHARE-CLS] train tumor={n_pos} healthy={n_neg} pos_weight={pos_weight.item():.2f}", flush=True)

    model = BinaryTumorClassifier().to(device)
    print(f"[FIGSHARE-CLS] params: {count_parameters(model):,}", flush=True)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    models_dir = os.path.join(project_root, "results", "models")
    metrics_dir = os.path.join(project_root, "results", "metrics")
    best_ckpt = os.path.join(models_dir, "figshare_classifier_best.pt")
    best_f1, bad, history = -1.0, 0, []
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        run_loss, run_n = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            if use_amp:
                with torch.amp.autocast("cuda"):
                    loss = criterion(model(x), y)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss = criterion(model(x), y)
                loss.backward()
                optimizer.step()
            run_loss += loss.item() * x.size(0)
            run_n += x.size(0)
        val_loss, vm = evaluate(model, val_loader, device, criterion)
        row = {"epoch": epoch, "train_loss": run_loss / run_n, "val_loss": val_loss, **vm}
        history.append(row)
        tag = ""
        if vm["f1"] > best_f1:
            best_f1, bad = vm["f1"], 0
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(), **vm}, best_ckpt)
            tag = "NEW BEST"
        else:
            bad += 1
            tag = f"no-improve {bad}/{patience}"
        print(f"Epoch {epoch:02d}/{epochs} [{time.time()-t0:.1f}s] train={row['train_loss']:.4f} "
              f"val_loss={val_loss:.4f} acc={vm['accuracy']:.4f} F1={vm['f1']:.4f} "
              f"rec={vm['recall']:.4f} spec={vm['specificity']:.4f} -> {tag}", flush=True)
        if bad >= patience:
            print("[FIGSHARE-CLS] early stop", flush=True)
            break

    with open(os.path.join(metrics_dir, "figshare_classifier_history.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        w.writeheader()
        w.writerows(history)
    best = max(history, key=lambda r: r["f1"])
    config = {"experiment": "figshare_classifier", "seed": seed, "epochs_run": len(history),
              "batch_size": batch_size, "lr": learning_rate, "pos_weight": pos_weight.item(),
              "best_epoch": best["epoch"], "best_val_f1": best["f1"],
              "params": count_parameters(model), "device": str(device)}
    json.dump(config, open(os.path.join(metrics_dir, "figshare_classifier_config.json"), "w"), indent=2)

    ckpt = torch.load(best_ckpt, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    test_loss, tm = evaluate(model, test_loader, device, criterion)
    report = {"test_loss": round(test_loss, 4),
              **{k: (round(float(v), 4) if isinstance(v, float) else v) for k, v in tm.items()},
              "best_epoch": best["epoch"]}
    json.dump(report, open(os.path.join(metrics_dir, "figshare_classifier_test_metrics.json"), "w"), indent=2)
    print(f"[FIGSHARE-CLS] TEST acc={tm['accuracy']:.4f} F1={tm['f1']:.4f} "
          f"rec={tm['recall']:.4f} spec={tm['specificity']:.4f} {tm}", flush=True)
    return {"config": config, "test": report}


if __name__ == "__main__":
    t0 = time.time()
    out = run_figshare_classifier()
    print(f"[FIGSHARE-CLS] done in {(time.time()-t0)/60:.1f} min: {out['test']}", flush=True)
