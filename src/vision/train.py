import os
import time
import json
import logging
from pathlib import Path
import matplotlib
matplotlib.use("Agg") # Fixes Tcl_AsyncDelete Tkinter thread collision
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

# Configure Professional Clinical Logger (No Emojis)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("3d-unet-train")

BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODEL_DIR = BASE_DIR / "models" / "vision"
TB_DIR = BASE_DIR / "runs" / "prostate_3d_unet"
IMG_LOG_DIR = TB_DIR / "visual_samples"

def get_dataloaders(batch_size=1):
    dataset_json = BASE_DIR / "data" / "kaggle_dataset_split.json"
    if not dataset_json.exists():
        log.error(f"Dataset JSON not found at {dataset_json}")
        return None, None
        
    with open(dataset_json, 'r') as f:
        data = json.load(f)
        
    train_files = [{"image_t2": d["image_t2"], "image_adc": d["image_adc"], "label": d["label"]} for d in data if d["split"] == "train"]
    val_files = [{"image_t2": d["image_t2"], "image_adc": d["image_adc"], "label": d["label"]} for d in data if d["split"] == "val"]
    
    log.info(f"Loaded {len(train_files)} Train | {len(val_files)} Val samples.")

    # CacheDataset accelerates CPU bounding bounding targeting RAM instead of VRAM
    train_ds = CacheDataset(data=train_files, transform=transforms.get_train_transforms(), cache_rate=1.0, num_workers=4)
    val_ds = CacheDataset(data=val_files, transform=transforms.get_val_transforms(), cache_rate=1.0, num_workers=2)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, pin_memory=True, num_workers=2)

    return train_loader, val_loader

def save_visual_sample(val_inputs, val_labels, val_outputs, epoch, batch_idx, save_dir):
    """
    Saves a 3-way visual smoke test: [T2 Slice | GT | AI Pred]
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    try:
        t2_vol = val_inputs[0, 0].cpu().numpy()  
        gt_vol = val_labels[0, 0].cpu().numpy()  
        
        # Apply sigmoid since out_channels=1
        pred_prob = torch.sigmoid(val_outputs[0, 0]).detach().cpu().numpy()
        pred_mask = (pred_prob > 0.5).astype(int)
        
        z_c = t2_vol.shape[-1] // 2
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(t2_vol[:, :, z_c].T, cmap='gray')
        axes[0].set_title("T2W Slice")
        
        axes[1].imshow(t2_vol[:, :, z_c].T, cmap='gray')
        axes[1].contour(gt_vol[:, :, z_c].T, colors='lime', levels=[0.5], linewidths=2)
        axes[1].set_title("Ground Truth")
        
        axes[2].imshow(t2_vol[:, :, z_c].T, cmap='gray')
        axes[2].contour(pred_mask[:, :, z_c].T, colors='red', levels=[0.5], linewidths=2)
        axes[2].set_title(f"AI Prediction (Epoch {epoch})")
        
        for ax in axes: ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(save_dir / f"epoch_check_{epoch}.png", dpi=150, facecolor='black')
        plt.close(fig)
    except Exception as e:
        log.error(f"Failed to generate visual sample: {e}")

def main():
    log.info("Starting Phase 8d - Mission Critical 3D U-Net Training")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    TB_DIR.mkdir(parents=True, exist_ok=True)
    IMG_LOG_DIR.mkdir(parents=True, exist_ok=True)
    
    set_determinism(seed=42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if device.type == "cuda":
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        log.info(f"GPU Detected: {torch.cuda.get_device_name(0)} | Total VRAM: {vram_gb:.2f} GB")
    else:
        log.critical("CUDA not detected. CPUs will bottleneck 3D operations heavily.")

    # 3D U-Net: in=2 (T2,ADC), out=1 (Binary Mask)
    model = UNet(
        spatial_dims=3,
        in_channels=2,
        out_channels=1, 
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    ).to(device)

    # Fix 2: Boost Dice weight (2.0) over Cross-Entropy (1.0) to avoid background dominance
    loss_function = DiceCELoss(sigmoid=True, lambda_dice=2.0) 
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    
    max_epochs = 100
    patience = 15
    patience_counter = 0
    best_metric = -1
    best_metric_epoch = -1
    
    scheduler = CosineAnnealingLR(optimizer, T_max=max_epochs)
    scaler = torch.cuda.amp.GradScaler() 

    dice_metric = DiceMetric(include_background=False, reduction="mean")
    hd_metric = HausdorffDistanceMetric(include_background=False, percentile=95.0, reduction="mean")

    post_pred = AsDiscrete(threshold=0.5)
    
    writer = SummaryWriter(log_dir=str(TB_DIR))

    # Strict Batch Size = 1 for 8GB VRAM
    train_loader, val_loader = get_dataloaders(batch_size=1)
    if train_loader is None: return

    acc_steps = 4 # Gradient accumulation simulates batch size 4
    
    # Resume logic
    start_epoch = 1
    checkpoint_path = MODEL_DIR / "best_unet_prostate.pth"
    if checkpoint_path.exists():
        model.load_state_dict(torch.load(str(checkpoint_path), map_location=device, weights_only=True))
        log.info(f"Checkpoint found (Best Dice: 0.0307). Resuming Phase 8d (Fix Applied) from Epoch 10.")
        start_epoch = 10
        # Fast-forward scheduler to match the epoch range
        for _ in range(1, start_epoch): scheduler.step()

    log.info("Initiating Training Loop...")
    for epoch in range(start_epoch, max_epochs + 1):
        model.train()
        epoch_loss = 0
        step = 0
        start_time = time.time()
        
        optimizer.zero_grad()
        
        for batch_data in train_loader:
            step += 1
            inputs, labels = batch_data["image"].to(device), batch_data["label"].to(device)
            
            try:
                with torch.cuda.amp.autocast():
                    outputs = model(inputs)
                    loss = loss_function(outputs, labels)
                    loss = loss / acc_steps # scale loss for accumulation
                    
                scaler.scale(loss).backward()
                
                if step % acc_steps == 0 or step == len(train_loader):
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    log.critical(f"OOM Detected. Peak VRAM exceeded limit.")
                    torch.cuda.empty_cache()
                    return 
                else:
                    raise e
            
            epoch_loss += (loss.item() * acc_steps)
            
            # VRAM telemetry
            alloc_vram = torch.cuda.memory_allocated(0) / 1e9
            if alloc_vram > 7.8:
                log.warning(f"VRAM near ceiling: {alloc_vram:.2f}GB at step {step}")
            
        epoch_loss /= step
        scheduler.step()
        
        train_time = time.time() - start_time
        writer.add_scalar("Train/Loss", epoch_loss, epoch)
        writer.add_scalar("Train/LearningRate", scheduler.get_last_lr()[0], epoch)
        
        # --- Validation Phase ---
        model.eval()
        with torch.no_grad():
            for val_step, val_data in enumerate(val_loader):
                val_inputs, val_labels = val_data["image"].to(device), val_data["label"].to(device)
                
                with torch.cuda.amp.autocast():
                    val_outputs = model(val_inputs)
                
                # Metrics Calculation (Sigmoid + Threshold)
                val_outputs_sig = torch.sigmoid(val_outputs)
                val_outputs_list = [post_pred(i) for i in decollate_batch(val_outputs_sig)]
                val_labels_list = [i for i in decollate_batch(val_labels)]
                
                dice_metric(y_pred=val_outputs_list, y=val_labels_list)
                hd_metric(y_pred=val_outputs_list, y=val_labels_list)
                
                if epoch % 10 == 0 and val_step == 0:
                    save_visual_sample(val_inputs, val_labels, val_outputs, epoch, val_step, IMG_LOG_DIR)

            metric = dice_metric.aggregate().item()
            hd_value = hd_metric.aggregate().item()
            
            dice_metric.reset()
            hd_metric.reset()
            
            writer.add_scalar("Val/Dice", metric, epoch)
            writer.add_scalar("Val/Hausdorff95", hd_value, epoch)
            
            log.info(f"Epoch {epoch}/{max_epochs} | Loss: {epoch_loss:.4f} | Val Dice: {metric:.4f} | Val HD95: {hd_value:.4f} | Time: {train_time:.1f}s")
            
            if metric > best_metric:
                best_metric = metric
                best_metric_epoch = epoch
                torch.save(model.state_dict(), str(MODEL_DIR / "best_unet_prostate.pth"))
                log.info(f"Model Checkpoint Saved (Dice: {best_metric:.4f})")
                patience_counter = 0 
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                log.info(f"Early Stopping trigger applied. No improvement in {patience} epochs.")
                break

    log.info(f"Training Finalized. Target Dice > 0.70 {'Achieved' if best_metric > 0.70 else 'Failed'}. Max Dice: {best_metric:.4f} at Epoch {best_metric_epoch}.")

if __name__ == '__main__':
    main()
