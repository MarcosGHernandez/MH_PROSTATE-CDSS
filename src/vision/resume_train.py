import os
import time
import json
import logging
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

# Phase 10b - Clinical Recovery Logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("3d-unet-resume")

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
    train_ds = CacheDataset(data=train_files, transform=transforms.get_train_transforms(), cache_rate=1.0, num_workers=4)
    val_ds = CacheDataset(data=val_files, transform=transforms.get_val_transforms(), cache_rate=1.0, num_workers=2)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, pin_memory=True, num_workers=2)

    return train_loader, val_loader

def main():
    log.info("Starting Phase 10b - Training Recovery (Manual Resume at Epoch 39)")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    set_determinism(seed=42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = UNet(
        spatial_dims=3,
        in_channels=2,
        out_channels=1, 
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    ).to(device)

    # Refined LR: 5e-5 (Half of previous)
    initial_lr = 5e-5
    loss_function = DiceCELoss(sigmoid=True, lambda_dice=2.0) 
    optimizer = AdamW(model.parameters(), lr=initial_lr, weight_decay=1e-5)
    
    max_epochs = 100
    patience = 20 # Increased patience for 10b
    patience_counter = 0
    best_metric = -1
    best_metric_epoch = -1
    
    scaler = torch.cuda.amp.GradScaler() 
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    hd_metric = HausdorffDistanceMetric(include_background=False, percentile=95.0, reduction="mean")
    post_pred = AsDiscrete(threshold=0.5)
    
    writer = SummaryWriter(log_dir=str(TB_DIR / "resume_10b"))

    # Load Weights
    checkpoint_path = MODEL_DIR / "best_unet_prostate.pth"
    if checkpoint_path.exists():
        model.load_state_dict(torch.load(str(checkpoint_path), map_location=device, weights_only=True))
        log.info(f"Successfully loaded weights from {checkpoint_path.name}.")
    else:
        log.warning("No best_unet_prostate.pth found. Starting from scratch (not recommended for 10b).")

    # Start from 39
    start_epoch = 39
    
    # Scheduler aligned with remaining epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=(max_epochs - start_epoch + 1))

    train_loader, val_loader = get_dataloaders(batch_size=1)
    acc_steps = 4 

    log.info(f"Initiating Recovery Loop (Epoch {start_epoch} to {max_epochs})...")
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
                    loss = loss_function(outputs, labels) / acc_steps
                scaler.scale(loss).backward()
                
                if step % acc_steps == 0 or step == len(train_loader):
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    log.critical("OOM Error. Clearing cache.")
                    torch.cuda.empty_cache()
                    return 
                raise e
            epoch_loss += (loss.item() * acc_steps)
            
        epoch_loss /= step
        scheduler.step()
        train_time = time.time() - start_time
        
        # Validation
        model.eval()
        with torch.no_grad():
            for val_data in val_loader:
                v_in, v_lab = val_data["image"].to(device), val_data["label"].to(device)
                with torch.cuda.amp.autocast():
                    v_out = model(v_in)
                v_out_sig = torch.sigmoid(v_out)
                v_out_list = [post_pred(i) for i in decollate_batch(v_out_sig)]
                v_lab_list = [i for i in decollate_batch(v_lab)]
                dice_metric(y_pred=v_out_list, y=v_lab_list)
                hd_metric(y_pred=v_out_list, y=v_lab_list)

            metric = dice_metric.aggregate().item()
            hd_val = hd_metric.aggregate().item()
            dice_metric.reset()
            hd_metric.reset()
            
            log.info(f"Epoch {epoch}/{max_epochs} | Loss: {epoch_loss:.4f} | Val Dice: {metric:.4f} | Val HD95: {hd_val:.4f} | Time: {train_time:.1f}s")
            
            if metric > best_metric:
                best_metric = metric
                best_metric_epoch = epoch
                torch.save(model.state_dict(), str(MODEL_DIR / "best_unet_prostate.pth"))
                log.info(f"Saved Improved Model (Dice: {best_metric:.4f})")
                patience_counter = 0 
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                log.info("Early stopping triggered.")
                break

if __name__ == '__main__':
    main()
