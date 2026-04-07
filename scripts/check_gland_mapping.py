"""
Check how many validation cases have whole-gland masks in PI-CAI anatomical delineations.
Outputs a JSON mapping: { patient_id -> gland_mask_path }
"""
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
WG_DIR = BASE / "data/raw/picai_labels/anatomical_delineations/whole_gland/AI/Bosma22b"
MAPPING_FILE = BASE / "data/raw/picai_labels/additional_resources/ProstateX-mapping.json"
SPLIT_JSON = BASE / "data/kaggle_dataset_split.json"

pid_map = json.loads(MAPPING_FILE.read_text())
data = json.loads(SPLIT_JSON.read_text())
val_pids = set(d["patient_id"] for d in data if d["split"] == "val")

# Keys in mapping are like 'ProstateX-0000_07-07-2011'
# Values like '11149_1001172'  <-- PICAI numeric ID
# WG files named like '10002_1000002.nii.gz'

matched = {}
for key, picai_id in pid_map.items():
    base_pid = key.split("_")[0]   # 'ProstateX-XXXX'
    if base_pid in val_pids:
        patient_num = picai_id.split("_")[0]  # e.g. '11149'
        candidate = WG_DIR / f"{picai_id}.nii.gz"
        if candidate.exists():
            matched[base_pid] = str(candidate)
        else:
            # Try with glob in case naming differs slightly
            candidates = list(WG_DIR.glob(f"{patient_num}_*.nii*"))
            if candidates:
                matched[base_pid] = str(candidates[0])

print(f"Matched: {len(matched)}/{len(val_pids)} validation patients")
for pid, path in list(matched.items())[:5]:
    print(f"  {pid} -> {Path(path).name}")

# Save mapping for use in evaluation
out = BASE / "data" / "gland_mask_mapping.json"
json.dump(matched, open(out, "w"), indent=2)
print(f"\nSaved to: {out}")
