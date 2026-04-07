import json
import random
import logging
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("align-verify")

BASE_DIR = Path(__file__).resolve().parent.parent.parent
OUTPUT_IMG = BASE_DIR / "runs" / "prostate_3d_unet" / "alignment_check.png"

def load_center_slice(path):
    img = nib.load(str(path))
    arr = img.get_fdata()
    # arr shape is usually (X, Y, Z). Get the center slice in Z
    z_center = arr.shape[-1] // 2
    return arr[:, :, z_center].T # Transposed for visualization (X,Y) -> (Y,X)

def run():
    json_path = BASE_DIR / "data" / "kaggle_dataset_split.json"
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    OUTPUT_IMG.parent.mkdir(parents=True, exist_ok=True)
    
    # Pick 3 random
    random.seed(42)
    samples = random.sample(data, 3)
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    fig.suptitle("Visual Smoke Test: T2 vs ADC vs Mask Alignment", fontsize=16)
    
    for row, sample in enumerate(samples):
        log.info(f"Visualizando: {sample['patient_id']} - Finding {sample['finding_id']}")
        
        t2_slice = load_center_slice(sample['image_t2'])
        adc_slice = load_center_slice(sample['image_adc'])
        mask_slice = load_center_slice(sample['label'])
        
        # Normalize T2 and ADC for better visualization
        t2_norm = (t2_slice - np.min(t2_slice)) / (np.max(t2_slice) - np.min(t2_slice) + 1e-8)
        adc_norm = (adc_slice - np.min(adc_slice)) / (np.max(adc_slice) - np.min(adc_slice) + 1e-8)
        
        # T2
        axes[row, 0].imshow(t2_norm, cmap='gray')
        axes[row, 0].set_title(f"{sample['patient_id']} - T2W")
        axes[row, 0].axis('off')
        
        # ADC
        axes[row, 1].imshow(adc_norm, cmap='gray')
        axes[row, 1].set_title(f"{sample['patient_id']} - ADC")
        axes[row, 1].axis('off')
        
        # T2 + Mask Overlay
        axes[row, 2].imshow(t2_norm, cmap='gray')
        axes[row, 2].contour(mask_slice, colors='red', linewidths=1.5, levels=[0.5])
        axes[row, 2].set_title("T2 + Lesion Mask")
        axes[row, 2].axis('off')
        
    plt.tight_layout()
    plt.savefig(str(OUTPUT_IMG), dpi=150, bbox_inches='tight', facecolor='black')
    log.info(f"Guardado exitosamente en: {OUTPUT_IMG}")

if __name__ == '__main__':
    run()
