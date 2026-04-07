import json, numpy as np, nibabel as nib
from pathlib import Path
BASE = Path(r"c:\Users\Predator Pro\OneDrive\Documents\Proyectos\Marcos_proyects\Health_can")
with open(BASE / "data" / "kaggle_dataset_split.json") as f:
    data = json.load(f)
ratios = []
for d in data[:20]:
    img = nib.load(d["label"])
    arr = img.get_fdata()
    fg = int(np.sum(arr > 0))
    ratio = fg / arr.size
    ratios.append(ratio)
    print(f"{d['patient_id']}: shape={arr.shape}, fg={fg}, ratio={ratio:.6f}")
print(f"\nMean fg ratio: {np.mean(ratios):.6f}")
