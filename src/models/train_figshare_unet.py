"""
Figshare T1-CE brain-tumor U-Net training.

Single-channel 2D binary whole-tumor segmentation on preprocessed Figshare PNGs
(data/processed/figshare/). Reuses UNet2D + UNetTrainer (CombinedDiceBCELoss,
AMP on CUDA, early stopping on val tumor Dice) for full comparability with the
BraTS Baseline-01 protocol.

Outputs (never overwrites BraTS checkpoints):
  results/models/figshare_unet_best.pt
  results/metrics/figshare_unet_history.csv
  results/metrics/figshare_unet_config.json
  results/metrics/figshare_unet_test_metrics.json
  results/figures/figshare_unet_curves.png
"""

import csv
import json
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.models.unet import UNet2D, count_parameters
from src.models.dataset import create_brats_dataloaders
from src.models.train_unet import UNetTrainer, set_seed


def run_figshare_training(
    data_dir: str = None,
    epochs: int = 15,
    batch_size: int = 8,
    learning_rate: float = 1e-3,
    patience: int = 5,
    seed: int = 42,
) -> dict:
    set_seed(seed)
    data_dir = data_dir or os.path.join(project_root, "data", "processed", "figshare")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[FIGSHARE-TRAIN] device={device} ({torch.cuda.get_device_name(0) if device.type=='cuda' else 'cpu'})",
          flush=True)
    if device.type != "cuda":
        print("[FIGSHARE-TRAIN] WARNING: CUDA not available, falling back to CPU", flush=True)

    loaders = create_brats_dataloaders(data_dir=data_dir, batch_size=batch_size,
                                       num_workers=0, augment_train=True)
    for name, loader in loaders.items():
        print(f"[FIGSHARE-TRAIN] {name}: {len(loader.dataset)} slices, {len(loader)} batches", flush=True)

    model = UNet2D(in_channels=1, out_channels=1, base_channels=32)
    print(f"[FIGSHARE-TRAIN] UNet2D params: {count_parameters(model):,}", flush=True)

    models_dir = os.path.join(project_root, "results", "models")
    trainer = UNetTrainer(model, loaders["train"], loaders["val"], lr=learning_rate,
                          checkpoint_dir=models_dir, best_ckpt_name="figshare_unet_best.pt",
                          device=device, seed=seed)
    history = trainer.fit(epochs=epochs, patience=patience)

    metrics_dir = os.path.join(project_root, "results", "metrics")
    figs_dir = os.path.join(project_root, "results", "figures")
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(figs_dir, exist_ok=True)

    # History CSV
    csv_path = os.path.join(metrics_dir, "figshare_unet_history.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_loss", "val_loss", "val_global_dice", "val_global_iou",
                    "val_precision", "val_recall", "val_tumor_dice", "val_tumor_iou"])
        for i in range(len(history["train_loss"])):
            w.writerow([i + 1] + [history[k][i] for k in
                        ["train_loss", "val_loss", "val_global_dice", "val_global_iou",
                         "val_precision", "val_recall", "val_tumor_dice", "val_tumor_iou"]])

    # Curves
    ep = list(range(1, len(history["train_loss"]) + 1))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(ep, history["train_loss"], label="train loss", marker="o")
    axes[0].plot(ep, history["val_loss"], label="val loss", marker="o")
    axes[0].set_title("Figshare U-Net loss"); axes[0].legend(); axes[0].grid(True)
    axes[1].plot(ep, history["val_tumor_dice"], label="val tumor dice", marker="o")
    axes[1].plot(ep, history["val_global_dice"], label="val global dice", marker="o")
    axes[1].set_title("Figshare U-Net Dice"); axes[1].legend(); axes[1].grid(True)
    fig.tight_layout()
    fig.savefig(os.path.join(figs_dir, "figshare_unet_curves.png"), dpi=120)
    plt.close(fig)

    # Config
    best_idx = int(max(range(len(history["val_tumor_dice"])),
                       key=lambda i: history["val_tumor_dice"][i]))
    config = {"experiment": "figshare_unet_2d", "data_dir": data_dir, "seed": seed,
              "epochs_run": len(ep), "batch_size": batch_size, "lr": learning_rate,
              "best_epoch": best_idx + 1,
              "best_val_tumor_dice": history["val_tumor_dice"][best_idx],
              "params": count_parameters(model),
              "device": str(device), "disclaimer": "Research-prototype result on Figshare CE-T1 slices."}
    with open(os.path.join(metrics_dir, "figshare_unet_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    # Held-out test (after model selection)
    best_ckpt = torch.load(os.path.join(models_dir, "figshare_unet_best.pt"), map_location=device)
    model.load_state_dict(best_ckpt["model_state_dict"])
    model.eval()
    from src.evaluation.segmentation_metrics import ConfusionMatrixAccumulator
    from src.models.train_unet import CombinedDiceBCELoss
    criterion = CombinedDiceBCELoss()
    acc = ConfusionMatrixAccumulator()
    running, total = 0.0, 0
    with torch.no_grad():
        for images, masks in loaders["test"]:
            images, masks = images.to(device), masks.to(device)
            logits = model(images)
            running += criterion(logits, masks).item() * images.size(0)
            total += images.size(0)
            acc.update(torch.sigmoid(logits), masks, threshold=0.5)
    test_loss = running / max(1, total)
    test_metrics = acc.get_metrics()
    report = {"test_loss": round(test_loss, 4),
              **{k: round(float(v), 4) for k, v in test_metrics.items()},
              "test_slices": len(loaders["test"].dataset),
              "best_epoch": best_idx + 1, "disclaimer": config["disclaimer"]}
    with open(os.path.join(metrics_dir, "figshare_unet_test_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"[FIGSHARE-TRAIN] TEST loss={test_loss:.4f} tumor_dice={test_metrics['val_tumor_dice']:.4f} "
          f"global_dice={test_metrics['val_global_dice']:.4f}", flush=True)
    return {"config": config, "test": report}


if __name__ == "__main__":
    t0 = time.time()
    out = run_figshare_training()
    print(f"[FIGSHARE-TRAIN] done in {(time.time()-t0)/60:.1f} min: {out['test']}", flush=True)
