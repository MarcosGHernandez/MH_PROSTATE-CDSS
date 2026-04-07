import os
import json
import logging
import pandas as pd
import numpy as np
import SimpleITK as sitk
from pathlib import Path
from sklearn.model_selection import StratifiedGroupKFold

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("map-data")

BASE_DIR = Path(__file__).resolve().parent.parent.parent
NIFTI_DIR = BASE_DIR / "data" / "processed_nifti"
MASKS_DIR = BASE_DIR / "data" / "masks_nifti"
CLINICAL_DIR = BASE_DIR / "data" / "clinical"

def create_spherical_mask(image, center_mm, radius_mm=5.0):
    """
    Creates a 3D spherical binary mask in the same physical space as the input image.
    center_mm: tuple (x,y,z) in physical coordinates (mm) 
               Note: ProstateX CSV 'pos' is often (L, P, S) or (R, A, S). Needs alignment check.
    """
    # Create empty mask matching the physical space
    mask = sitk.Image(image.GetSize(), sitk.sitkUInt8)
    mask.CopyInformation(image)
    
    # Convert physical center to voxel indices
    try:
        center_idx = image.TransformPhysicalPointToIndex(center_mm)
    except Exception as e:
        log.warning(f"Coordenadas fuera de limites: {center_mm} -> {e}")
        return None

    # Calculate radius in voxels (approximate based on spacing)
    spacing = image.GetSpacing()
    rx, ry, rz = [max(1, int(radius_mm / s)) for s in spacing]
    
    # Generate sphere
    cx, cy, cz = center_idx
    nx, ny, nz = image.GetSize()
    
    # Simple bounding box iteration
    for z in range(max(0, cz-rz), min(nz, cz+rz+1)):
        for y in range(max(0, cy-ry), min(ny, cy+ry+1)):
            for x in range(max(0, cx-rx), min(nx, cx+rx+1)):
                # Distance in mm
                pt = image.TransformIndexToPhysicalPoint((x,y,z))
                dist = np.sqrt(sum((p - c)**2 for p, c in zip(pt, center_mm)))
                if dist <= radius_mm:
                    mask[x,y,z] = 1
                    
    return mask

def parse_findings():
    findings_csv = CLINICAL_DIR / "ProstateX-Findings-Train.csv"
    images_csv = CLINICAL_DIR / "ProstateX-Images-Train.csv"
    
    if not findings_csv.exists() or not images_csv.exists():
        log.error(f"Faltan CSVs clinicos en {CLINICAL_DIR}. Descarguelos desde TCIA.")
        return None
        
    df_find = pd.read_csv(findings_csv)
    df_img = pd.read_csv(images_csv)
    
    # Merge on ProxID and fid
    df_merged = pd.merge(df_find, df_img, on=['ProxID', 'fid'], how='inner')
    
    MASKS_DIR.mkdir(parents=True, exist_ok=True)
    dataset = []
    
    for idx, row in df_merged.iterrows():
        pid = row['ProxID']
        # Parse physical position (usually space separated floats)
        pos = tuple(map(float, str(row['pos']).split()))
        clin_sig = row.get('ClinSig', False)
        
        # Determine target T2W image path
        t2_file = list(NIFTI_DIR.glob(f"{pid}/*t2*sag*.nii.gz")) # Example fallback, needs exact UID match from Images.csv
        if not t2_file:
            t2_file = list(NIFTI_DIR.glob(f"{pid}/*t2*.nii.gz"))
            
        if not t2_file:
            continue
            
        t2_path = t2_file[0]
        
        # Load Image
        try:
            img = sitk.ReadImage(str(t2_path))
            mask = create_spherical_mask(img, pos, radius_mm=5.0)
            if mask is None: continue
            
            mask_filename = MASKS_DIR / f"{pid}_finding_{row['fid']}_mask.nii.gz"
            sitk.WriteImage(mask, str(mask_filename))
            
            dataset.append({
                "patient_id": pid,
                "finding_id": row['fid'],
                "image": str(t2_path),
                "label": str(mask_filename),
                "clin_sig": 1 if clin_sig else 0
            })
            log.info(f"OK: {pid} Finding {row['fid']} -> Mask generada.")
        except Exception as e:
            log.error(f"FAIL: {pid} {e}")
            
    return pd.DataFrame(dataset)

def split_dataset(df):
    if df is None or df.empty: return
    
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    # Stratify by classification target, Group by PatientID to prevent leakage
    train_idx, valtest_idx = next(sgkf.split(df['image'], df['clin_sig'], df['patient_id']))
    
    df_train = df.iloc[train_idx].copy()
    df_valtest = df.iloc[valtest_idx].copy()
    
    # Split the remaining 20% into 10% Val / 10% Test
    sgkf_val = StratifiedGroupKFold(n_splits=2, shuffle=True, random_state=42)
    val_idx, test_idx = next(sgkf_val.split(df_valtest['image'], df_valtest['clin_sig'], df_valtest['patient_id']))
    
    df_val = df_valtest.iloc[val_idx].copy()
    df_test = df_valtest.iloc[test_idx].copy()
    
    df_train['split'] = 'train'
    df_val['split'] = 'val'
    df_test['split'] = 'test'
    
    final_df = pd.concat([df_train, df_val, df_test])
    
    out_json = BASE_DIR / "data" / "dataset_split.json"
    final_df.to_json(out_json, orient='records', indent=4)
    log.info(f"Split finalizado: {len(df_train)} Train | {len(df_val)} Val | {len(df_test)} Test. Guardado en {out_json}")

if __name__ == '__main__':
    df = parse_findings()
    split_dataset(df)
