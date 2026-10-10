"""
Figshare classifier v2 — calibrated, agreement-focused retrain.

v1 problems: pos_weight=5.33 inflated every output (voided the 0.5 boundary),
no calibration (healthy maxed at 0.999), gate/mask disagreements (empty-mask
" detections", sub-10% scores next to positive verdicts).

v2 protocol (same BinaryTumorClassifier arch):
  - Balanced WeightedRandomSampler instead of pos_weight (cleaner probabilities)
  - Label smoothing alpha=0.05 (anti-overconfidence)
  - 12 epochs, patience 5 on val F1, lr 1e-3, batch 32, AMP
  - Temperature scaling fitted on VAL (minimize NLL); T saved in checkpoint
Saves results/models/figshare_classifier_v2_best.pt (+ history/config/test).
"""

import csv
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.models.classifier import BinaryTumorClassifier, count_parameters
from src.models.train_unet import set_seed
from src.models.train_figshare_classifier import (
    FigshareClsDataset, cls_metrics, evaluate,
)


@torch.no_grad()
def collect_logits(model, loader, device):
    model.eval()
    logits, trues = [], []
    for x, y in loader:
        logits.append(model(x.to(device)).flatten().cpu())
        trues.append(y.flatten())
    return torch.cat(logits), torch.cat(trues)


def fit_temperature(logits: torch.Tensor, trues: torch.Tensor) -> float:
    """Grid-search T minimizing val NLL: p_cal = sigmoid(logit / T)."""
    best_t, best_nll = 1.0, float("inf")
    for t in np.arange(0.5, 5.01, 0.05):
        nll = nn.functional.binary_cross_entropy_with_logits(logits / t, trues).item()
        if nll < best_nll:
            best_nll, best_t = nll, float(t)
    return best_t


def run_figshare_classifier_v2(
    data_dir: str = None, epochs: int = 12, batch_size: int = 32,
    learning_rate: float = 1e-3, patience: int = 5, seed: int = 42,
    smooth: float = 0.05,
) -> dict:
    set_seed(seed)
    data_dir = data_dir or os.path.join(project_root, "data", "processed", "figshare")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[FIGSHARE-CLS-V2] device={device} smooth={smooth} seed={seed}", flush=True)

    train_ds = FigshareClsDataset(os.path.join(data_dir, "train", "images"), augment=True, seed=seed)
    val_ds = FigshareClsDataset(os.path.join(data_dir, "val", "images"), seed=seed)
    test_ds = FigshareClsDataset(os.path.join(data_dir, "test", "images"), seed=seed)

    labels = np.array([0.0 if f.startswith("healthy_") else 1.0
                       for f in (os.path.basename(p) for p in train_ds.files)])
    class_counts = np.bincount(labels.astype(int))
    weights = len(labels) / (2.0 * class_counts[labels.astype(int)])
    sampler = torch.utils.data.WeightedRandomSampler(weights, len(labels), replacement=True)
    pin = torch.cuda.is_available()
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                              num_workers=0, pin_memory=pin)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=pin)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=pin)
    print(f"[FIGSHARE-CLS-V2] train tumor={(labels == 1).sum()} healthy={(labels == 0).sum()} "
          f"(balanced sampler, plain BCE + LS)", flush=True)

    model = BinaryTumorClassifier().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    def smooth_targets(y):
        return y * (1.0 - smooth) + 0.5 * smooth

    models_dir = os.path.join(project_root, "results", "models")
    metrics_dir = os.path.join(project_root, "results", "metrics")
    best_ckpt = os.path.join(models_dir, "figshare_classifier_v2_best.pt")
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
                    loss = criterion(model(x), smooth_targets(y))
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss = criterion(model(x), smooth_targets(y))
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
            print("[FIGSHARE-CLS-V2] early stop", flush=True)
            break

    with open(os.path.join(metrics_dir, "figshare_classifier_v2_history.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        w.writeheader()
        w.writerows(history)

    ckpt = torch.load(best_ckpt, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    val_logits, val_trues = collect_logits(model, val_loader, device)
    temperature = fit_temperature(val_logits, val_trues.float())
    ckpt["temperature"] = temperature
    torch.save(ckpt, best_ckpt)
    print(f"[FIGSHARE-CLS-V2] fitted temperature T={temperature:.2f} (val NLL)", flush=True)

    test_loss, tm = evaluate(model, test_loader, device, criterion)
    best = max(history, key=lambda r: r["f1"])
    config = {"experiment": "figshare_classifier_v2", "seed": seed, "epochs_run": len(history),
              "batch_size": batch_size, "lr": learning_rate, "label_smoothing": smooth,
              "temperature": temperature, "best_epoch": best["epoch"], "best_val_f1": best["f1"],
              "params": count_parameters(model), "device": str(device)}
    json.dump(config, open(os.path.join(metrics_dir, "figshare_classifier_v2_config.json"), "w"), indent=2)
    report = {"test_loss": round(test_loss, 4),
              **{k: (round(float(v), 4) if isinstance(v, float) else v) for k, v in tm.items()},
              "best_epoch": best["epoch"], "temperature": temperature}
    json.dump(report, open(os.path.join(metrics_dir, "figshare_classifier_v2_test_metrics.json"), "w"), indent=2)
    print(f"[FIGSHARE-CLS-V2] TEST acc={tm['accuracy']:.4f} F1={tm['f1']:.4f} "
          f"rec={tm['recall']:.4f} spec={tm['specificity']:.4f} {tm}", flush=True)
    return {"config": config, "test": report}


if __name__ == "__main__":
    t0 = time.time()
    out = run_figshare_classifier_v2()
    print(f"[FIGSHARE-CLS-V2] done in {(time.time()-t0)/60:.1f} min: {out['test']}", flush=True)
