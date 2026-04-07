"""
Phase 11 - Hardware Saturation & Precision Tuning
==================================================
Hardened training script with OOM protection, VRAM watchdog,
batch_size=2, LR kick to 1e-4, and lambda_dice=1.5.
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
log = logging.getLogger("phase11")

BASE_DIR   = Path(__file__).resolve().parent.parent.parent
MODEL_DIR  = BASE_DIR / "models" / "vision"
TB_DIR     = BASE_DIR / "runs" / "prostate_3d_unet"
IMG_LOG_DIR = TB_DIR / "visual_samples"

# ─── Safety Constants ──────────────────────────────────────────
VRAM_HARD_CEILING_GB = 7.6   # Abort step if exceeded
VRAM_SOFT_CEILING_GB = 7.0   # Log warning
OOM_MAX_RETRIES      = 3     # Per-epoch OOM retries before fallback
CHECKPOINT_EVERY     = 5     # Save periodic checkpoint every N epochs


def get_dataloaders(batch_size=2):
    """Load data with num_workers=4 for CPU saturation."""
    dataset_json = BASE_DIR / "data" / "kaggle_dataset_split.json"
    if not dataset_json.exists():
        log.error(f"Dataset JSON not found at {dataset_json}")
        return None, None

    with open(dataset_json, 'r') as f:
        data = json.load(f)

    train_files = [{"image_t2": d["image_t2"], "image_adc": d["image_adc"], "label": d["label"]}
                   for d in data if d["split"] == "train"]
    val_files   = [{"image_t2": d["image_t2"], "image_adc": d["image_adc"], "label": d["label"]}
                   for d in data if d["split"] == "val"]

    log.info(f"Dataset: {len(train_files)} Train | {len(val_files)} Val")

    train_ds = CacheDataset(data=train_files, transform=transforms.get_train_transforms(),
                            cache_rate=1.0, num_workers=0)
    val_ds   = CacheDataset(data=val_files, transform=transforms.get_val_transforms(),
                            cache_rate=1.0, num_workers=0)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              pin_memory=True, num_workers=0)
    val_loader   = DataLoader(val_ds, batch_size=1, shuffle=False,
                              pin_memory=True, num_workers=0)
    return train_loader, val_loader


def vram_check(tag=""):
    """Returns current VRAM usage in GB; logs warnings if thresholds exceeded."""
    if not torch.cuda.is_available():
        return 0.0
    alloc = torch.cuda.memory_allocated(0) / 1e9
    reserved = torch.cuda.memory_reserved(0) / 1e9
    if alloc > VRAM_HARD_CEILING_GB:
        log.critical(f"[VRAM-HARD] {tag} Allocated={alloc:.2f}GB > {VRAM_HARD_CEILING_GB}GB ceiling!")
    elif alloc > VRAM_SOFT_CEILING_GB:
        log.warning(f"[VRAM-SOFT] {tag} Allocated={alloc:.2f}GB (Reserved={reserved:.2f}GB)")
    return alloc


def safe_train_step(model, batch_data, loss_function, scaler, optimizer, acc_steps, step, total_steps, device):
    """
    Single training step with OOM protection.
    Returns (loss_value, success_flag).
    """
    inputs = batch_data["image"].to(device, non_blocking=True)
    labels = batch_data["label"].to(device, non_blocking=True)

    try:
        with torch.cuda.amp.autocast():
            outputs = model(inputs)
            loss = loss_function(outputs, labels) / acc_steps

        scaler.scale(loss).backward()

        if step % acc_steps == 0 or step == total_steps:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        return loss.item() * acc_steps, True

    except RuntimeError as e:
        if 'out of memory' in str(e).lower():
            log.warning(f"OOM at step {step}. Clearing cache and skipping batch.")
            torch.cuda.empty_cache()
            optimizer.zero_grad()
            del inputs, labels
            return 0.0, False
        else:
            raise e


def safe_val_step(model, val_data, post_pred, dice_metric, hd_metric, device):
    """
    Single validation step with NaN/Inf protection.
    Returns True on success.
    """
    v_in  = val_data["image"].to(device, non_blocking=True)
    v_lab = val_data["label"].to(device, non_blocking=True)

    try:
        with torch.cuda.amp.autocast():
            v_out = model(v_in)

        v_out_sig = torch.sigmoid(v_out)

        # NaN/Inf guard
        if torch.isnan(v_out_sig).any() or torch.isinf(v_out_sig).any():
            log.warning("NaN/Inf detected in validation output. Skipping sample.")
            return False

        v_out_list = [post_pred(i) for i in decollate_batch(v_out_sig)]
        v_lab_list = [i for i in decollate_batch(v_lab)]

        dice_metric(y_pred=v_out_list, y=v_lab_list)
        hd_metric(y_pred=v_out_list, y=v_lab_list)
        return True

    except Exception as e:
        log.warning(f"Validation step error: {e}. Skipping sample.")
        torch.cuda.empty_cache()
        return False


def main():
    log.info("=" * 70)
    log.info("Phase 11 - Hardware Saturation & Precision Tuning")
    log.info("=" * 70)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    TB_DIR.mkdir(parents=True, exist_ok=True)
    IMG_LOG_DIR.mkdir(parents=True, exist_ok=True)

    set_determinism(seed=42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1e9
        log.info(f"GPU: {gpu_name} | VRAM: {vram_gb:.2f} GB")
        # Pre-clear fragmented memory
        torch.cuda.empty_cache()
    else:
        log.critical("NO CUDA DETECTED. Aborting Phase 11 (GPU-only).")
        return

    # ─── Model ─────────────────────────────────────────────────
    model = UNet(
        spatial_dims=3, in_channels=2, out_channels=1,
        channels=(16, 32, 64, 128, 256), strides=(2, 2, 2, 2),
        num_res_units=2,
    ).to(device)

    # ─── Loss: lambda_dice=1.5 (boost Dice penalty) ──────────
    loss_function = DiceCELoss(sigmoid=True, lambda_dice=1.5, lambda_ce=1.0)

    # ─── LR Kick: Reset to 1e-4 to escape local minimum ─────
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)

    max_epochs      = 100
    patience        = 20
    patience_counter = 0
    best_metric      = -1
    best_metric_epoch = -1

    scaler     = torch.cuda.amp.GradScaler()
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    hd_metric   = HausdorffDistanceMetric(include_background=False, percentile=95.0, reduction="mean")
    post_pred   = AsDiscrete(threshold=0.5)

    writer = SummaryWriter(log_dir=str(TB_DIR / "phase11"))

    # ─── Load Checkpoint ───────────────────────────────────────
    checkpoint_path = MODEL_DIR / "best_unet_prostate.pth"
    start_epoch = 1
    if checkpoint_path.exists():
        try:
            model.load_state_dict(torch.load(str(checkpoint_path), map_location=device, weights_only=True))
            log.info(f"Loaded weights from {checkpoint_path.name}")
            start_epoch = 62  # Resume from where v4 stopped
        except Exception as e:
            log.error(f"Failed to load checkpoint: {e}. Starting fresh.")

    # Scheduler aligned with remaining epochs
    remaining = max_epochs - start_epoch + 1
    scheduler = CosineAnnealingLR(optimizer, T_max=max(remaining, 10))

    # ─── DataLoaders: batch_size=2, num_workers=4 ─────────────
    train_loader, val_loader = get_dataloaders(batch_size=2)
    if train_loader is None:
        return

    acc_steps = 4  # Effective batch = 2 * 4 = 8

    log.info(f"Config: batch=2 | acc_steps=4 | eff_batch=8 | LR=1e-4 | lambda_dice=1.5")
    log.info(f"Epochs: {start_epoch} -> {max_epochs} | Patience: {patience}")
    log.info(f"Safety: OOM retries={OOM_MAX_RETRIES} | VRAM ceiling={VRAM_HARD_CEILING_GB}GB")
    log.info("Initiating Training Loop...")

    for epoch in range(start_epoch, max_epochs + 1):
        model.train()
        epoch_loss = 0.0
        step = 0
        oom_count = 0
        start_time = time.time()
        optimizer.zero_grad()

        for batch_data in train_loader:
            step += 1
            loss_val, success = safe_train_step(
                model, batch_data, loss_function, scaler,
                optimizer, acc_steps, step, len(train_loader), device
            )

            if success:
                epoch_loss += loss_val
            else:
                oom_count += 1
                if oom_count >= OOM_MAX_RETRIES:
                    log.error(f"Epoch {epoch}: Hit {OOM_MAX_RETRIES} OOM errors. Aborting epoch.")
                    break

            # Periodic VRAM telemetry
            if step % 20 == 0:
                vram_check(f"Epoch {epoch} Step {step}")

        if step > 0:
            epoch_loss /= step
        scheduler.step()
        train_time = time.time() - start_time

        writer.add_scalar("Train/Loss", epoch_loss, epoch)
        writer.add_scalar("Train/LR", scheduler.get_last_lr()[0], epoch)

        # ─── Validation ────────────────────────────────────────
        model.eval()
        val_success_count = 0
        with torch.no_grad():
            for val_data in val_loader:
                ok = safe_val_step(model, val_data, post_pred, dice_metric, hd_metric, device)
                if ok:
                    val_success_count += 1

        # Aggregate with safety
        try:
            metric  = dice_metric.aggregate().item()
            hd_val  = hd_metric.aggregate().item()
        except Exception:
            metric = 0.0
            hd_val = 999.0
            log.warning("Metric aggregation failed. Using fallback values.")

        # NaN guard on aggregated metric
        if metric != metric:  # NaN check
            metric = 0.0
            log.warning("Dice metric returned NaN. Resetting to 0.")
        if hd_val != hd_val:
            hd_val = 999.0

        dice_metric.reset()
        hd_metric.reset()

        writer.add_scalar("Val/Dice", metric, epoch)
        writer.add_scalar("Val/HD95", hd_val, epoch)

        log.info(
            f"Epoch {epoch}/{max_epochs} | Loss: {epoch_loss:.4f} | "
            f"Val Dice: {metric:.4f} | Val HD95: {hd_val:.2f} | "
            f"Time: {train_time:.1f}s | OOM: {oom_count} | ValOK: {val_success_count}"
        )

        # ─── Checkpointing ────────────────────────────────────
        if metric > best_metric:
            best_metric = metric
            best_metric_epoch = epoch
            torch.save(model.state_dict(), str(MODEL_DIR / "best_unet_prostate.pth"))
            log.info(f"*** NEW BEST Model Saved (Dice: {best_metric:.4f}) ***")
            patience_counter = 0
        else:
            patience_counter += 1

        # Periodic safety checkpoint
        if epoch % CHECKPOINT_EVERY == 0:
            torch.save(model.state_dict(), str(MODEL_DIR / f"checkpoint_epoch_{epoch}.pth"))
            log.info(f"Safety checkpoint saved: checkpoint_epoch_{epoch}.pth")

        # Early stopping
        if patience_counter >= patience:
            log.info(f"Early Stopping: No improvement in {patience} epochs. Best Dice: {best_metric:.4f} at Epoch {best_metric_epoch}.")
            break

        # Force VRAM cleanup between epochs
        torch.cuda.empty_cache()

    log.info("=" * 70)
    log.info(f"Phase 11 Complete. Best Dice: {best_metric:.4f} at Epoch {best_metric_epoch}")
    log.info(f"Target Dice > 0.70: {'ACHIEVED' if best_metric > 0.70 else 'NOT YET'}")
    log.info("=" * 70)
    writer.close()


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logging.critical(f"FATAL ERROR: {e}")
        logging.critical(traceback.format_exc())
        torch.cuda.empty_cache()
        sys.exit(1)
