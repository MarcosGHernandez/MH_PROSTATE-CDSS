"""
Phase 12 - Data Quality Audit
Scans all label masks to find which ones have foreground pixels
after the preprocessing pipeline is applied.
"""
import json, sys
from pathlib import Path
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent.parent

def main():
    dataset_json = BASE_DIR / "data" / "kaggle_dataset_split.json"
    with open(dataset_json, 'r') as f:
        data = json.load(f)

    import nibabel as nib

    empty_count = 0
    valid_count = 0
    total = len(data)
    empty_ids = []

    for i, d in enumerate(data):
        label_path = d["label"]
        pid = d["patient_id"]
        fid = d["finding_id"]
        split = d["split"]
        try:
            img = nib.load(label_path)
            arr = img.get_fdata()
            fg = np.sum(arr > 0)
            total_vox = arr.size
            if fg == 0:
                empty_count += 1
                empty_ids.append(f"{pid}-{fid} ({split})")
                status = "EMPTY"
            else:
                valid_count += 1
                status = f"OK (fg={fg}, ratio={fg/total_vox:.6f})"
            print(f"[{i+1}/{total}] {pid}-{fid} [{split}]: {status} | shape={arr.shape}")
        except Exception as e:
            empty_count += 1
            empty_ids.append(f"{pid}-{fid} ({split}) - LOAD ERROR: {e}")
            print(f"[{i+1}/{total}] {pid}-{fid} [{split}]: LOAD ERROR - {e}")

    print("\n" + "="*60)
    print(f"AUDIT SUMMARY")
    print(f"Total samples: {total}")
    print(f"Valid (has foreground): {valid_count}")
    print(f"Empty (no foreground): {empty_count}")
    print(f"Empty ratio: {empty_count/total*100:.1f}%")
    if empty_ids:
        print(f"\nEmpty/broken masks:")
        for eid in empty_ids:
            print(f"  - {eid}")

if __name__ == '__main__':
    main()
