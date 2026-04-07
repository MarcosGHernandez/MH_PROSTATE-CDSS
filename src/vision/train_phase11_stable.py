"""
Phase 11b - Stable Hardware Saturation
======================================
Optimized to prevent RAM/VRAM exhaustion on Laptop CPUs.
batch_size=1, acc_steps=8 (effective 8), num_workers=0.
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
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter

from monai.networks.nets import UNet
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.data import DataLoader, decollate_batch, CacheDataset
from monai.utils import set_determinism
from monai.transforms import AsDiscrete

import transforms

# ─── Configuration ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger("phase11b")

BASE_DIR   = Path(__file__).resolve().parent.parent.parent
MODEL_DIR  = BASE_DIR / "models" / "vision"
TB_DIR     = BASE_DIR / "runs" / "prostate_3d_unet"

def get_dataloaders(batch_size=1):
    """Load data with num_workers=0 to prevent RAM crashes on Windows."""
    dataset_json = BASE_DIR / "data" / "kaggle_dataset_split.json"
    
    with open(dataset_json, 'r') as f:
        data = json.load(f)

    train_files = [{"image_t2": d["image_t2"], "image_adc": d["image_adc"], "label": d["label"]}
                   for d in data if d["split"] == "train"]
    val_files   = [{"image_t2": d["image_t2"], "image_adc": d["image_adc"], "label": d["label"]}
                   for d in data if d["split"] == "val"]

    log.info(f"Dataset: {len(train_files)} Train | {len(val_files)} Val")

    # num_workers=0 saves 50%+ of System RAM. It's slower to start but 100% stable.
    train_ds = CacheDataset(data=train_files, transform=transforms.get_train_transforms(),
                            cache_rate=1.0, num_workers=0)
    val_ds   = CacheDataset(data=val_files, transform=transforms.get_val_transforms(),
                            cache_rate=1.0, num_workers=0)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=0)
    val_loader   = DataLoader(val_ds, batch_size=1, shuffle=False, pin_memory=True, num_workers=0)
    return train_loader, val_loader


def main():
    log.info("Phase 11b - Stable Recovery Routine (VRAM & RAM Safe)")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    TB_DIR.mkdir(parents=True, exist_ok=True)

    set_determinism(seed=42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.empty_cache()

    model = UNet(
        spatial_dims=3, in_channels=2, out_channels=1,
        channels=(16, 32, 64, 128, 256), strides=(2, 2, 2, 2),
        num_res_units=2,
    ).to(device)

    loss_function = DiceCELoss(sigmoid=True, lambda_dice=1.5, lambda_ce=1.0)
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)

    max_epochs = 100
    patience = 20
    best_metric = -1
    best_metric_epoch = -1
    patience_counter = 0

    scaler = torch.cuda.amp.GradScaler()
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    hd_metric   = HausdorffDistanceMetric(include_background=False, percentile=95.0, reduction="mean")
    post_pred   = AsDiscrete(threshold=0.5)

    writer = SummaryWriter(log_dir=str(TB_DIR / "phase11b"))

    checkpoint_path = MODEL_DIR / "best_unet_prostate.pth"
    start_epoch = 1
    if checkpoint_path.exists():
        model.load_state_dict(torch.load(str(checkpoint_path), map_location=device, weights_only=True))
        start_epoch = 62  # Resuming where we left off logically
        log.info(f"Loaded weights. Resuming from Epoch {start_epoch}.")

    scheduler = CosineAnnealingLR(optimizer, T_max=max(max_epochs - start_epoch + 1, 10))

    # BATCH=1, ACC_STEPS=8 keeps the 8GB VRAM completely safe, but learns like BATCH=8
    train_loader, val_loader = get_dataloaders(batch_size=1)
    acc_steps = 8  

    for epoch in range(start_epoch, max_epochs + 1):
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
                with torch.cuda.amp.autocast():
                    outputs = model(inputs)
                    loss = loss_function(outputs, labels) / acc_steps
                
                scaler.scale(loss).backward()

                if step % acc_steps == 0 or step == len(train_loader):
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                
                epoch_loss += loss.item() * acc_steps
            except RuntimeError as e:
                log.warning("OOM Caught! Skipping...")
                torch.cuda.empty_cache()
                optimizer.zero_grad()

        if step > 0: epoch_loss /= step
        scheduler.step()
        train_time = time.time() - start_time
        writer.add_scalar("Train/Loss", epoch_loss, epoch)

        # Validation
        model.eval()
        with torch.no_grad():
            for val_data in val_loader:
                v_in = val_data["image"].to(device, non_blocking=True)
                v_lab = val_data["label"].to(device, non_blocking=True)
                with torch.cuda.amp.autocast():
                    v_out = model(v_in)
                
                v_out_sig = torch.sigmoid(v_out)
                if torch.isnan(v_out_sig).any(): continue

                v_out_list = [post_pred(i) for i in decollate_batch(v_out_sig)]
                v_lab_list = [i for i in decollate_batch(v_lab)]
                dice_metric(y_pred=v_out_list, y=v_lab_list)
                hd_metric(y_pred=v_out_list, y=v_lab_list)

        try:
            metric = dice_metric.aggregate().item()
            hd_val = hd_metric.aggregate().item()
        except Exception:
            metric = 0.0
            hd_val = 999.0

        if metric != metric: metric = 0.0
        dice_metric.reset()
        hd_metric.reset()

        log.info(f"Epoch {epoch}/{max_epochs} | Loss: {epoch_loss:.4f} | Val Dice: {metric:.4f} | Val HD95: {hd_val:.2f} | Time: {train_time:.1f}s")

        if metric > best_metric:
            best_metric = metric
            best_metric_epoch = epoch
            torch.save(model.state_dict(), str(MODEL_DIR / "best_unet_prostate.pth"))
            log.info(f"New Best Model: Dice {best_metric:.4f}!")
            patience_counter = 0
        else:
            patience_counter += 1

        torch.cuda.empty_cache()

        if patience_counter >= patience:
            log.info("Early Stopping Triggered.")
            break

if __name__ == '__main__':
    main()
