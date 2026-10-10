"""
Figshare UNet v2 — regularized retrain.

v1 (lr 1e-3, 15 epochs, flip/rot90 only) overfit: train loss fell to 0.17 while
val loss oscillated 0.22-0.29 and val tumor Dice swung +/-0.06 late in training.

v2 counter-measures (same UNet2D(1,1,32) + CombinedDiceBCELoss + AMP + CUDA):
  - lr 5e-4 with ReduceLROnPlateau (factor 0.5, patience 3 on val loss)
  - AdamW weight_decay 1e-4 -> 3e-4
  - Extra intensity augmentation (seeded): brightness/contrast jitter + Gaussian noise
  - 30 epochs, early-stopping patience 8 on val tumor Dice
  - Outputs to figshare_unet_v2_* (v1 artifacts untouched)
"""

import csv
import json
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.models.unet import UNet2D, count_parameters
from src.models.dataset import BraTS2DDataset, create_brats_dataloaders
from src.models.train_unet import UNetTrainer, set_seed, CombinedDiceBCELoss
from src.evaluation.segmentation_metrics import ConfusionMatrixAccumulator


class IntensityAugmentWrapper(Dataset):
    """Seeded per-sample intensity jitter over a base (image, mask) dataset.

    Masks are untouched. Augmentation (train only):
      - brightness scale ~ U[0.9, 1.1], contrast scale ~ U[0.9, 1.1]
      - additive Gaussian noise sigma=0.01
    Deterministic given (seed, index): reproducible across runs/workers.
    """

    def __init__(self, base: BraTS2DDataset, seed: int = 42):
        self.base = base
        self.seed = seed

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, mask = self.base[idx]
        rng = np.random.default_rng(self.seed * 1_000_003 + idx)
        b = float(rng.uniform(0.9, 1.1))
        c = float(rng.uniform(0.9, 1.1))
        img = torch.clamp((img - 0.5) * c * b + 0.5, 0.0, 1.0).float()
        noise = torch.from_numpy(rng.normal(0.0, 0.01, size=img.shape).astype(np.float32))
        img = torch.clamp(img + noise, 0.0, 1.0)
        return img, mask


def run_figshare_v2(
    data_dir: str = None,
    epochs: int = 30,
    batch_size: int = 8,
    learning_rate: float = 5e-4,
    weight_decay: float = 3e-4,
    patience: int = 8,
    seed: int = 42,
) -> dict:
    set_seed(seed)
    data_dir = data_dir or os.path.join(project_root, "data", "processed", "figshare")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[FIGSHARE-V2] device={device} lr={learning_rate} wd={weight_decay} "
          f"epochs={epochs} patience={patience} seed={seed}", flush=True)

    base = create_brats_dataloaders(data_dir=data_dir, batch_size=1)  # probe counts
    for name, loader in base.items():
        print(f"[FIGSHARE-V2] {name}: {len(loader.dataset)} slices", flush=True)

    train_base = BraTS2DDataset(data_dir=data_dir, split="train", augment=True, seed=seed)
    val_base = BraTS2DDataset(data_dir=data_dir, split="val", augment=False, seed=seed)
    test_base = BraTS2DDataset(data_dir=data_dir, split="test", augment=False, seed=seed)
    pin = torch.cuda.is_available()
    train_loader = DataLoader(IntensityAugmentWrapper(train_base, seed), batch_size=batch_size,
                              shuffle=True, num_workers=0, pin_memory=pin)
    val_loader = DataLoader(val_base, batch_size=batch_size, shuffle=False,
                            num_workers=0, pin_memory=pin)
    test_loader = DataLoader(test_base, batch_size=batch_size, shuffle=False,
                             num_workers=0, pin_memory=pin)

    model = UNet2D(in_channels=1, out_channels=1, base_channels=32)
    print(f"[FIGSHARE-V2] params: {count_parameters(model):,}", flush=True)
    models_dir = os.path.join(project_root, "results", "models")
    trainer = UNetTrainer(model, train_loader, val_loader, lr=learning_rate,
                          checkpoint_dir=models_dir, best_ckpt_name="figshare_unet_v2_best.pt",
                          device=device, seed=seed)
    # Stronger regularization + LR schedule (trainer has no scheduler hook)
    for g in trainer.optimizer.param_groups:
        g["weight_decay"] = weight_decay
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        trainer.optimizer, mode="min", factor=0.5, patience=3)

    history = {k: [] for k in ["train_loss", "val_loss", "val_global_dice", "val_global_iou",
                               "val_precision", "val_recall", "val_tumor_dice", "val_tumor_iou"]}
    best_dice, bad, lr_now = -1.0, 0, learning_rate
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tr_loss = trainer.train_epoch()
        val_loss, vm = trainer.validate_epoch()
        scheduler.step(val_loss)
        for k in history:
            history[k].append(tr_loss if k == "train_loss" else (val_loss if k == "val_loss" else vm[k]))
        new_lr = trainer.optimizer.param_groups[0]["lr"]
        improved = vm["val_tumor_dice"] > best_dice
        if improved:
            best_dice = vm["val_tumor_dice"]
            bad = 0
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": trainer.optimizer.state_dict(),
                        **{k: vm[k] for k in ["val_global_dice", "val_global_iou", "val_precision",
                                              "val_recall", "val_tumor_dice", "val_tumor_iou"]},
                        "val_loss": val_loss},
                       os.path.join(models_dir, "figshare_unet_v2_best.pt"))
            tag = "NEW BEST"
        else:
            bad += 1
            tag = f"no-improve {bad}/{patience}"
        print(f"Epoch {epoch:02d}/{epochs} [{time.time()-t0:.1f}s] train={tr_loss:.4f} val={val_loss:.4f} "
              f"tumorDice={vm['val_tumor_dice']:.4f} globDice={vm['val_global_dice']:.4f} "
              f"lr={new_lr:.1e} -> {tag}", flush=True)
        if new_lr != lr_now:
            print(f"  [SCHEDULER] lr {lr_now:.1e} -> {new_lr:.1e}", flush=True)
            lr_now = new_lr
        if bad >= patience:
            print(f"[FIGSHARE-V2] early stop at epoch {epoch}", flush=True)
            break

    metrics_dir = os.path.join(project_root, "results", "metrics")
    figs_dir = os.path.join(project_root, "results", "figures")
    with open(os.path.join(metrics_dir, "figshare_unet_v2_history.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["epoch"] + list(history.keys()))
        for i in range(len(history["train_loss"])):
            w.writerow([i + 1] + [history[k][i] for k in history])
    ep = list(range(1, len(history["train_loss"]) + 1))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(ep, history["train_loss"], label="train", marker="o")
    axes[0].plot(ep, history["val_loss"], label="val", marker="o")
    axes[0].set_title("v2 loss"); axes[0].legend(); axes[0].grid(True)
    axes[1].plot(ep, history["val_tumor_dice"], label="val tumor dice", marker="o")
    axes[1].plot(ep, history["val_global_dice"], label="val global dice", marker="o")
    axes[1].set_title("v2 Dice"); axes[1].legend(); axes[1].grid(True)
    fig.tight_layout()
    fig.savefig(os.path.join(figs_dir, "figshare_unet_v2_curves.png"), dpi=120)
    plt.close(fig)

    best_idx = int(max(range(len(history["val_tumor_dice"])),
                       key=lambda i: history["val_tumor_dice"][i]))
    config = {"experiment": "figshare_unet_v2", "seed": seed, "epochs_run": len(ep),
              "batch_size": batch_size, "lr_init": learning_rate, "weight_decay": weight_decay,
              "best_epoch": best_idx + 1, "best_val_tumor_dice": history["val_tumor_dice"][best_idx],
              "params": count_parameters(model), "device": str(device),
              "disclaimer": "Research-prototype result on Figshare CE-T1 slices."}
    json.dump(config, open(os.path.join(metrics_dir, "figshare_unet_v2_config.json"), "w"), indent=2)

    best_ckpt = torch.load(os.path.join(models_dir, "figshare_unet_v2_best.pt"), map_location=device)
    model.load_state_dict(best_ckpt["model_state_dict"])
    model.eval()
    criterion, acc = CombinedDiceBCELoss(), ConfusionMatrixAccumulator()
    running, total = 0.0, 0
    with torch.no_grad():
        for images, masks in test_loader:
            images, masks = images.to(device), masks.to(device)
            logits = model(images)
            running += criterion(logits, masks).item() * images.size(0)
            total += images.size(0)
            acc.update(torch.sigmoid(logits), masks, threshold=0.5)
    tm = acc.get_metrics()
    report = {"test_loss": round(running / total, 4),
              **{k: round(float(v), 4) for k, v in tm.items()},
              "test_slices": len(test_loader.dataset), "best_epoch": best_idx + 1,
              "disclaimer": config["disclaimer"]}
    json.dump(report, open(os.path.join(metrics_dir, "figshare_unet_v2_test_metrics.json"), "w"), indent=2)
    print(f"[FIGSHARE-V2] TEST loss={report['test_loss']} tumor_dice={report['val_tumor_dice']} "
          f"global_dice={report['val_global_dice']}", flush=True)
    return {"config": config, "test": report}


if __name__ == "__main__":
    t0 = time.time()
    out = run_figshare_v2()
    print(f"[FIGSHARE-V2] done in {(time.time()-t0)/60:.1f} min: {out['test']}", flush=True)
