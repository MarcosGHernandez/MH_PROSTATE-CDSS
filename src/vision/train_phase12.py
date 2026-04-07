"""
Phase 12 - Hardened Precision Training
=======================================
Architecture fixes applied after diagnosing 0.06% foreground ratio:

1. DiceFocalLoss (0.8 Dice + 0.2 Focal, gamma=2.0) — ignores easy background
2. SlidingWindowInferer — evaluates FULL volume in validation
3. 0.5mm in-plane spacing — preserves tiny ROIs
4. 4 lesion-centered crops per volume — 75% positive hit rate
5. CosineAnnealingLR (LR=2e-4) + Early Stopping (patience=40)
6. num_workers=0 — prevents Windows RAM collapse
7. Periodic safety checkpoints every 10 epochs
"""
import os
import sys
import time
import json
import logging
import traceback
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from monai.networks.nets import UNet
from monai.losses import DiceFocalLoss
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.inferers import SlidingWindowInferer
from monai.data import DataLoader, decollate_batch, CacheDataset, list_data_collate
from monai.utils import set_determinism
from monai.transforms import AsDiscrete

import transforms

# ─── Logging ────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("phase12")

BASE_DIR  = Path(__file__).resolve().parent.parent.parent
MODEL_DIR = BASE_DIR / "models" / "vision"
LOG_DIR   = BASE_DIR / "runs" / "v12_focal_precision"


def get_dataloaders():
    dataset_json = BASE_DIR / "data" / "kaggle_dataset_split.json"
    with open(dataset_json, "r") as f:
        data = json.load(f)

    train_files = [
        {"image_t2": d["image_t2"], "image_adc": d["image_adc"], "label": d["label"]}
        for d in data if d["split"] == "train"
    ]
    val_files = [
        {"image_t2": d["image_t2"], "image_adc": d["image_adc"], "label": d["label"]}
        for d in data if d["split"] == "val"
    ]

    log.info(f"Dataset: {len(train_files)} Train | {len(val_files)} Val")

    # cache_rate=1.0 — all volumes in RAM for speed (uses ~4-6 GB RAM)
    train_ds = CacheDataset(
        data=train_files, transform=transforms.get_train_transforms(),
        cache_rate=1.0, num_workers=0,
    )
    val_ds = CacheDataset(
        data=val_files, transform=transforms.get_val_transforms(),
        cache_rate=1.0, num_workers=0,
    )

    # list_data_collate handles the 4 sub-patches from RandCropByPosNegLabeld
    train_loader = DataLoader(
        train_ds, batch_size=1, shuffle=True,
        pin_memory=True, num_workers=0,
        collate_fn=list_data_collate,
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False,
        pin_memory=True, num_workers=0,
    )
    return train_loader, val_loader


def main():
    log.info("=" * 70)
    log.info("Phase 12 - Hardened Precision Training (Fresh Start from Epoch 0)")
    log.info("=" * 70)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    set_determinism(seed=42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type == "cuda":
        torch.cuda.empty_cache()
        log.info(f"GPU: {torch.cuda.get_device_name(0)} | VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.2f} GB")
    else:
        log.critical("NO CUDA DETECTED. Aborting.")
        return

    # ─── Model ──────────────────────────────────────────────────
    model = UNet(
        spatial_dims=3,
        in_channels=2,
        out_channels=1,
        channels=(32, 64, 128, 256),
        strides=(2, 2, 2),
        num_res_units=2,
        dropout=0.2,
    ).to(device)
    log.info(f"Model: {sum(p.numel() for p in model.parameters())/1e6:.1f}M parameters")

    # ─── Loss: 0.8 Dice + 0.2 Focal (gamma=2.0) ────────────────
    loss_function = DiceFocalLoss(
        sigmoid=True,
        lambda_dice=0.8,
        lambda_focal=0.2,
        gamma=2.0,
    )

    # ─── Optimizer + Scheduler ──────────────────────────────────
    optimizer = AdamW(model.parameters(), lr=2e-4, weight_decay=1e-5)

    max_epochs = 100
    patience   = 40
    scheduler  = CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1e-6)

    # ─── Metrics & Inferer ──────────────────────────────────────
    scaler     = torch.amp.GradScaler("cuda")
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    hd_metric   = HausdorffDistanceMetric(include_background=False, percentile=95.0, reduction="mean")
    post_pred   = AsDiscrete(threshold=0.5)

    # SlidingWindowInferer: evaluates FULL volume, not a center crop
    inferer = SlidingWindowInferer(
        roi_size=(128, 128, 32),
        sw_batch_size=2,
        overlap=0.5,
        mode="gaussian",
    )

    # ─── DataLoaders ────────────────────────────────────────────
    train_loader, val_loader = get_dataloaders()

    acc_steps = 8   # Effective batch = 4 patches × acc_steps=8 / 4 = 8 updates

    best_metric      = -1
    best_metric_epoch = -1
    patience_counter  = 0

    log.info(f"Loss: DiceFocalLoss(dice=0.8, focal=0.2, gamma=2.0)")
    log.info(f"LR: 2e-4 → CosineAnnealingLR | Patience(ES): {patience}")
    log.info(f"batch=1 × 4patches × acc={acc_steps} | Validation: SlidingWindowInferer")
    log.info("Initiating Training Loop...")

    for epoch in range(1, max_epochs + 1):
        model.train()
        epoch_loss = 0.0
        step = 0
        start_time = time.time()
        optimizer.zero_grad()

        for batch_data in train_loader:
            step += 1
            inputs = batch_data["image"].to(device, non_blocking=True)
            labels = batch_data["label"].to(device, non_blocking=True)

            try:
                with torch.amp.autocast("cuda"):
                    outputs = model(inputs)
                    loss = loss_function(outputs, labels) / acc_steps

                scaler.scale(loss).backward()

                if step % acc_steps == 0 or step == len(train_loader):
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

                epoch_loss += loss.item() * acc_steps

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    log.warning(f"OOM at step {step}. Skipping batch.")
                    torch.cuda.empty_cache()
                    optimizer.zero_grad()
                    continue
                raise e

        if step > 0:
            epoch_loss /= step
        scheduler.step()
        train_time = time.time() - start_time
        current_lr = optimizer.param_groups[0]["lr"]

        # ─── Validation: Full-Volume Sliding Window ──────────────
        model.eval()
        with torch.no_grad():
            for val_data in val_loader:
                v_in  = val_data["image"].to(device)
                v_lab = val_data["label"].to(device)
                try:
                    with torch.amp.autocast("cuda"):
                        v_out = inferer(v_in, model)
                    v_out_sig = torch.sigmoid(v_out)
                    if torch.isnan(v_out_sig).any():
                        continue
                    v_out_list = [post_pred(i) for i in decollate_batch(v_out_sig)]
                    v_lab_list = [i for i in decollate_batch(v_lab)]
                    dice_metric(y_pred=v_out_list, y=v_lab_list)
                    hd_metric(y_pred=v_out_list, y=v_lab_list)
                except RuntimeError:
                    torch.cuda.empty_cache()
                    continue

        try:
            metric = dice_metric.aggregate().item()
            hd_val = hd_metric.aggregate().item()
        except Exception:
            metric, hd_val = 0.0, 999.0

        if metric != metric: metric = 0.0   # NaN guard
        if hd_val != hd_val: hd_val = 999.0

        dice_metric.reset()
        hd_metric.reset()

        log.info(
            f"Epoch {epoch}/{max_epochs} | Loss: {epoch_loss:.4f} | "
            f"Val Dice: {metric:.4f} | Val HD95: {hd_val:.2f} | "
            f"LR: {current_lr:.2e} | Time: {train_time:.1f}s"
        )

        # ─── Checkpoint ─────────────────────────────────────────
        if metric > best_metric:
            best_metric      = metric
            best_metric_epoch = epoch
            torch.save(model.state_dict(), str(MODEL_DIR / "best_unet_prostate.pth"))
            log.info(f"*** NEW BEST: Dice {best_metric:.4f} at Epoch {epoch} ***")
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 10 == 0:
            torch.save(model.state_dict(), str(MODEL_DIR / f"ckpt_ep{epoch}.pth"))
            log.info(f"Safety checkpoint saved: ckpt_ep{epoch}.pth")

        torch.cuda.empty_cache()

        if patience_counter >= patience:
            log.info(f"Early Stopping at Epoch {epoch}. Best Dice: {best_metric:.4f}")
            break

    log.info("=" * 70)
    log.info(f"Phase 12 Complete. Best Dice: {best_metric:.4f} at Epoch {best_metric_epoch}")
    log.info(f"Target Dice > 0.70: {'ACHIEVED' if best_metric > 0.70 else 'NOT YET'}")
    log.info("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.critical(f"FATAL: {e}")
        logging.critical(traceback.format_exc())
        torch.cuda.empty_cache()
        sys.exit(1)
