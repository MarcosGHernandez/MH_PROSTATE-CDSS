"""
Phase 14 Evaluation Script
============================
Compares three configurations on the validation set:
  1. RAW         — direct U-Net output, threshold=0.5
  2. Phase 13    — CCA + Morphological (no ROI mask)
  3. Phase 14    — ROI Masking + CCA + Morphological (FULL pipeline)

Expected: Phase 14 Dice > Phase 13 Dice ~= Phase 12 RAW Dice,
because extra-prostatic false positives are eliminated BEFORE CCA.
"""
import sys
import json
import logging
import traceback
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import torch
import numpy as np

from monai.networks.nets import UNet
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.inferers import SlidingWindowInferer
from monai.data import DataLoader, decollate_batch, CacheDataset
from monai.utils import set_determinism

from transforms import get_val_transforms
from post_processing import ClinicalPostProcessor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("eval-phase14")

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models" / "vision"
GLAND_MAP  = BASE_DIR / "data" / "gland_mask_mapping.json"


def load_validation_data():
    dataset_json = BASE_DIR / "data" / "kaggle_dataset_split.json"
    if not dataset_json.exists():
        raise FileNotFoundError(f"Dataset JSON not found: {dataset_json}")

    data = json.loads(dataset_json.read_text())
    gland_map = json.loads(GLAND_MAP.read_text()) if GLAND_MAP.exists() else {}

    val_files = []
    for d in data:
        if d["split"] != "val":
            continue
        t2p  = Path(d["image_t2"])
        adcp = Path(d["image_adc"])
        lp   = Path(d["label"])
        if t2p.exists() and adcp.exists() and lp.exists():
            entry = {
                "image_t2":  str(t2p),
                "image_adc": str(adcp),
                "label":     str(lp),
                "patient_id": d["patient_id"],
                "gland_mask": gland_map.get(d["patient_id"]),   # may be None
            }
            val_files.append(entry)
        else:
            log.warning(f"Files missing for {d['patient_id']}")

    gland_avail = sum(1 for v in val_files if v["gland_mask"] is not None)
    log.info(f"Loaded {len(val_files)} val cases | {gland_avail} have gland masks")
    return val_files


def evaluate():
    set_determinism(seed=42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type == "cuda":
        torch.cuda.empty_cache()
        log.info(f"GPU: {torch.cuda.get_device_name(0)}")

    val_files = load_validation_data()

    # MONAI dataset — strip gland_mask/patient_id so CacheDataset doesn't choke
    monai_files = [
        {"image_t2": v["image_t2"], "image_adc": v["image_adc"], "label": v["label"]}
        for v in val_files
    ]
    val_ds = CacheDataset(data=monai_files, transform=get_val_transforms(),
                          cache_rate=1.0, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False,
                            pin_memory=True, num_workers=0)

    # Model
    model = UNet(
        spatial_dims=3, in_channels=2, out_channels=1,
        channels=(32, 64, 128, 256), strides=(2, 2, 2),
        num_res_units=2, dropout=0.2,
    ).to(device)

    model_path = MODEL_DIR / "best_unet_prostate.pth"
    if not model_path.exists():
        log.error(f"Model weights not found: {model_path}")
        return

    model.load_state_dict(torch.load(str(model_path), map_location=device))
    model.eval()

    inferer = SlidingWindowInferer(roi_size=(128, 128, 32), sw_batch_size=2,
                                   overlap=0.5, mode="gaussian")

    # Three metric accumulators
    metrics = {
        "raw":    {"dice": DiceMetric(include_background=False, reduction="mean"),
                   "hd":   HausdorffDistanceMetric(include_background=False, percentile=95.0, reduction="mean")},
        "ph13":   {"dice": DiceMetric(include_background=False, reduction="mean"),
                   "hd":   HausdorffDistanceMetric(include_background=False, percentile=95.0, reduction="mean")},
        "ph14":   {"dice": DiceMetric(include_background=False, reduction="mean"),
                   "hd":   HausdorffDistanceMetric(include_background=False, percentile=95.0, reduction="mean")},
    }

    # Post-processors
    proc_ph13 = ClinicalPostProcessor(threshold=0.38, min_volume_mm3=50.0,
                                      spacing=(0.5, 0.5, 3.0), use_roi_masking=False)
    proc_ph14 = ClinicalPostProcessor(threshold=0.38, min_volume_mm3=50.0,
                                      spacing=(0.5, 0.5, 3.0), use_roi_masking=True)

    log.info("Starting Phase 14 evaluation ...")
    with torch.no_grad():
        for idx, (val_data, val_meta) in enumerate(zip(val_loader, val_files)):
            v_in  = val_data["image"].to(device)
            v_lab = val_data["label"].to(device)
            pid   = val_meta["patient_id"]
            gland_path = val_meta["gland_mask"]

            try:
                with torch.amp.autocast("cuda"):
                    v_out = inferer(v_in, model)

                sig = torch.sigmoid(v_out)
                logits_np = v_out.squeeze(0).squeeze(0).cpu().numpy()

                if torch.isnan(sig).any():
                    log.warning(f"NaN in output for {pid}, skipping")
                    continue

                # RAW
                raw_t = (sig > 0.5).int()

                # Phase 13 (CCA, no gland mask)
                p13_np = proc_ph13(logits_np)
                p13_t  = torch.from_numpy(p13_np).unsqueeze(0).unsqueeze(0).to(device)

                # Phase 14 (ROI Masking + CCA)
                p14_np = proc_ph14(logits_np, gland_mask_path=gland_path)
                p14_t  = torch.from_numpy(p14_np).unsqueeze(0).unsqueeze(0).to(device)

                lab_list = decollate_batch(v_lab)
                metrics["raw"]["dice"](y_pred=decollate_batch(raw_t), y=lab_list)
                metrics["raw"]["hd"](y_pred=decollate_batch(raw_t),   y=lab_list)
                metrics["ph13"]["dice"](y_pred=decollate_batch(p13_t), y=lab_list)
                metrics["ph13"]["hd"](y_pred=decollate_batch(p13_t),   y=lab_list)
                metrics["ph14"]["dice"](y_pred=decollate_batch(p14_t), y=lab_list)
                metrics["ph14"]["hd"](y_pred=decollate_batch(p14_t),   y=lab_list)

                gland_status = "With gland mask" if gland_path else "Ellipsoid fallback"
                log.info(f"  [{idx+1:02d}/{len(val_files)}] {pid} | {gland_status}")

            except RuntimeError as e:
                log.error(f"RuntimeError on {pid}: {e}")
                torch.cuda.empty_cache()
            except Exception as e:
                log.error(f"Error on {pid}: {e}")
                traceback.print_exc()

    # Results
    def agg(m, key):
        try:
            return m[key].aggregate().item()
        except Exception:
            return float("nan")

    dice_raw  = agg(metrics["raw"],  "dice"); hd_raw  = agg(metrics["raw"],  "hd")
    dice_p13  = agg(metrics["ph13"], "dice"); hd_p13  = agg(metrics["ph13"], "hd")
    dice_p14  = agg(metrics["ph14"], "dice"); hd_p14  = agg(metrics["ph14"], "hd")

    sep = "=" * 65
    log.info(sep)
    log.info(f"PHASE 14 EVALUATION — {len(val_files)} VALIDATION CASES")
    log.info(sep)
    log.info(f"{'Pipeline':<30} {'Dice':>10} {'HD95 (mm)':>12}")
    log.info("-" * 55)
    log.info(f"{'RAW (thresh=0.5)':<30} {dice_raw:>10.4f} {hd_raw:>12.2f}")
    log.info(f"{'Phase 13 (CCA, no ROI)':<30} {dice_p13:>10.4f} {hd_p13:>12.2f}")
    log.info(f"{'Phase 14 (ROI + CCA)':<30} {dice_p14:>10.4f} {hd_p14:>12.2f}")
    log.info(sep)
    log.info(f"Phase 14 improvement vs RAW:   Dice {dice_p14 - dice_raw:+.4f} | HD {hd_p14 - hd_raw:+.2f}mm")
    log.info(f"Phase 14 improvement vs Ph13:  Dice {dice_p14 - dice_p13:+.4f} | HD {hd_p14 - hd_p13:+.2f}mm")
    log.info(sep)


if __name__ == "__main__":
    evaluate()
