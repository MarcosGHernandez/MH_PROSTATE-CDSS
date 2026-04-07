import argparse
import sys
import json
import logging
from pathlib import Path
import traceback

import torch
import numpy as np

# Suppress matplotlib GUI requirements
import matplotlib
matplotlib.use("Agg")

from monai.networks.nets import UNet
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.inferers import SlidingWindowInferer
from monai.data import DataLoader, decollate_batch, CacheDataset
from monai.utils import set_determinism

# We'll use the validation transforms directly
from transforms import get_val_transforms
from post_processing import ClinicalPostProcessor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("eval-phase13")

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models" / "vision"

def load_validation_data():
    dataset_json = BASE_DIR / "data" / "kaggle_dataset_split.json"
    if not dataset_json.exists():
        log.error(f"Cannot find dataset at {dataset_json}")
        raise FileNotFoundError(f"File not found: {dataset_json}")
        
    with open(dataset_json, "r") as f:
        data = json.load(f)

    val_files = []
    for d in data:
        if d["split"] == "val":
            # Paths in JSON are already absolute Windows paths
            t2p = Path(d["image_t2"])
            adcp = Path(d["image_adc"])
            lp = Path(d["label"])
            
            if t2p.exists() and adcp.exists() and lp.exists():
                val_files.append({"image_t2": str(t2p), "image_adc": str(adcp), "label": str(lp)})
            else:
                log.warning(f"File missing for {d['patient_id']} - {d['finding_id']}")
    log.info(f"Loaded {len(val_files)} validation cases.")
    return val_files

def evaluate_model():
    set_determinism(seed=42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if device.type == "cuda":
        torch.cuda.empty_cache()
        log.info(f"Evaluating on {torch.cuda.get_device_name(0)}")

    val_files = load_validation_data()

    # Create dataset and loader
    val_ds = CacheDataset(
        data=val_files, transform=get_val_transforms(),
        cache_rate=1.0, num_workers=0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False,
        pin_memory=True, num_workers=0,
    )

    # Initialize model and load weights
    model = UNet(
        spatial_dims=3,
        in_channels=2,
        out_channels=1,
        channels=(32, 64, 128, 256),
        strides=(2, 2, 2),
        num_res_units=2,
        dropout=0.2, # Keep same architecture as training
    ).to(device)

    model_path = MODEL_DIR / "best_unet_prostate.pth"
    if not model_path.exists():
         log.error(f"Model weights not found at: {model_path}")
         return

    model.load_state_dict(torch.load(str(model_path), map_location=device))
    model.eval()

    # Evaluators
    inferer = SlidingWindowInferer(
        roi_size=(128, 128, 32),
        sw_batch_size=2,
        overlap=0.5,
        mode="gaussian",
    )

    # We use two sets of metrics to show the impact of the post-processor
    dice_raw = DiceMetric(include_background=False, reduction="mean")
    hd_raw = HausdorffDistanceMetric(include_background=False, percentile=95.0, reduction="mean")
    
    dice_clean = DiceMetric(include_background=False, reduction="mean")
    hd_clean = HausdorffDistanceMetric(include_background=False, percentile=95.0, reduction="mean")

    # Post processing pipeline
    post_processor = ClinicalPostProcessor(
        threshold=0.38, 
        min_volume_mm3=50.0, 
        spacing=(0.5, 0.5, 3.0) 
    )

    log.info("Starting evaluation ...")
    with torch.no_grad():
        for idx, val_data in enumerate(val_loader):
            v_in = val_data["image"].to(device)
            v_lab = val_data["label"].to(device)

            try:
                # AMP evaluation
                with torch.amp.autocast("cuda"):
                    v_out = inferer(v_in, model)

                # Raw predictions
                v_out_sig_raw = torch.sigmoid(v_out)
                
                # We need outputs on CPU for the numpy-based post-processor
                np_pred_logits = v_out.squeeze(0).squeeze(0).cpu().numpy()
                np_label = v_lab.squeeze(0).squeeze(0).cpu().numpy()

                # Clean predictions
                np_pred_clean = post_processor(np_pred_logits)

                # Prepare raw tensor
                raw_binary_tensor = (v_out_sig_raw > 0.5).int()
                
                # Prepare cleaned tensor
                clean_binary_tensor = torch.from_numpy(np_pred_clean).unsqueeze(0).unsqueeze(0).to(device)

                # Collect metrics for valid slices (skip NaN inputs if any)
                if not torch.isnan(v_out_sig_raw).any():
                     # Decollate batches
                     raw_list = decollate_batch(raw_binary_tensor)
                     clean_list = decollate_batch(clean_binary_tensor)
                     lab_list = decollate_batch(v_lab)

                     dice_raw(y_pred=raw_list, y=lab_list)
                     hd_raw(y_pred=raw_list, y=lab_list)
                     
                     dice_clean(y_pred=clean_list, y=lab_list)
                     hd_clean(y_pred=clean_list, y=lab_list)
                     
                     log.info(f"Processed case {idx+1}/{len(val_loader)}")

            except RuntimeError as e:
                log.error(f"Runtime error during inference case {idx}: {e}")
                torch.cuda.empty_cache()
                continue
            except Exception as e:
                log.error(f"Error during inference case {idx}: {e}")
                traceback.print_exc()
                continue

    # Aggregate and print results
    try:
        final_dice_raw = dice_raw.aggregate().item()
        final_hd_raw = hd_raw.aggregate().item()
        
        final_dice_clean = dice_clean.aggregate().item()
        final_hd_clean = hd_clean.aggregate().item()
        
        log.info("=" * 60)
        log.info(f"EVALUATION COMPLETE (VALIDATION SET: {len(val_loader)} CASES)")
        log.info("=" * 60)
        log.info(f"RAW PREDICTION (Threshold 0.5, No Post-proc)")
        log.info(f" - Dice Score: {final_dice_raw:.4f}")
        log.info(f" - HD95 (mm):  {final_hd_raw:.2f}")
        log.info("-" * 60)
        log.info(f"CLEAN PREDICTION (Phase 13 Post-proc)")
        log.info(f" - Threshold:  {post_processor.threshold}")
        log.info(f" - CCA Filter: > {post_processor.min_volume_mm3} mm^3")
        log.info(f" - Dice Score: {final_dice_clean:.4f}  (Delta {final_dice_clean - final_dice_raw:+.4f})")
        log.info(f" - HD95 (mm):  {final_hd_clean:.2f}  (Delta {final_hd_clean - final_hd_raw:+.2f})")
        log.info("=" * 60)
        
    except Exception as e:
         log.error(f"Could not calculate final metrics: {e}")

if __name__ == "__main__":
    evaluate_model()
