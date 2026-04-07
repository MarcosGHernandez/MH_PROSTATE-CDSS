import os
import json
import logging
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedGroupKFold

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("kaggle-bridge")

BASE_DIR = Path(__file__).resolve().parent.parent.parent
KAGGLE_DIR = BASE_DIR / "data" / "clinical"

def run():
    log.info("Iniciando Phase 8c - Data Remapping (Kaggle -> Local)")
    
    mapping_csv = KAGGLE_DIR / "image_mask_mapping.csv"
    img_list_csv = KAGGLE_DIR / "lesions" / "lesions" / "Image_list.csv"
    classes_csv = KAGGLE_DIR / "lesions" / "lesions" / "PROSTATEx_Classes.csv"
    
    df_map = pd.read_csv(mapping_csv)
    df_list = pd.read_csv(img_list_csv)
    df_class = pd.read_csv(classes_csv)
    
    # Preprocess classes for quick lookup. ID format: "ProstateX-0000_Finding1"
    class_dict = {}
    for _, row in df_class.iterrows():
        class_dict[row['ID']] = 1 if row['Clinically Significant'] else 0
        
    dataset = []
    missing_count = 0
    
    for _, row in df_map.iterrows():
        # Kaggle Path Example: /kaggle/input/prostatex/lesions/lesions/Images/T2/ProstateX-0133_t2_tse_tra_4.nii
        t2_kaggle = Path(row['image'])
        mask_kaggle = Path(row['mask'])
        finding_id = row['finding']
        
        # Extract filename to build local paths
        t2_filename = t2_kaggle.name
        mask_filename = mask_kaggle.name
        
        t2_local = KAGGLE_DIR / "lesions" / "lesions" / "Images" / "T2" / t2_filename
        mask_local = KAGGLE_DIR / "lesions" / "lesions" / "Masks" / "T2" / mask_filename
        
        # Find ADC using Image_list.csv (where T2 sequence name matches T2 filename without .nii)
        t2_seq_name = t2_kaggle.stem
        adc_matches = df_list[df_list['T2'] == t2_seq_name]['ADC'].values
        
        if len(adc_matches) == 0:
            log.debug(f"Missing ADC map for {t2_seq_name}")
            missing_count += 1
            continue
            
        adc_filename = adc_matches[0] + ".nii"
        adc_local = KAGGLE_DIR / "lesions" / "lesions" / "Images" / "ADC" / adc_filename
        
        # Validate existence
        if not t2_local.exists():
            t2_local = t2_local.with_suffix(".nii.gz")
            if not t2_local.exists():
                missing_count += 1
                continue
                
        if not mask_local.exists():
            mask_local = mask_local.with_suffix(".nii.gz")
            if not mask_local.exists():
                missing_count += 1
                continue
                
        if not adc_local.exists():
            adc_local = adc_local.with_suffix(".nii.gz")
            if not adc_local.exists():
                missing_count += 1
                continue
                
        # Extract Patient ID (e.g., ProstateX-XXXX)
        pid = t2_filename.split("_")[0]
        
        class_key = f"{pid}_{finding_id}"
        clin_sig = class_dict.get(class_key, 0)
        
        dataset.append({
            "patient_id": pid,
            "finding_id": finding_id,
            "image_t2": str(t2_local),
            "image_adc": str(adc_local),
            "label": str(mask_local),
            "clin_sig": clin_sig
        })
        
    log.info(f"Mapeo completado. Triplets validados (T2+ADC+Mask): {len(dataset)} | Faltantes/Errores: {missing_count}")
    
    # Train / Val / Test Split (Stratified by Patient to prevent leak)
    df = pd.DataFrame(dataset)
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    
    # 80% Train, 20% ValTest
    train_idx, valtest_idx = next(sgkf.split(df['image_t2'], df['clin_sig'], df['patient_id']))
    df_train = df.iloc[train_idx].copy()
    df_valtest = df.iloc[valtest_idx].copy()
    
    # 10% Val, 10% Test
    sgkf_val = StratifiedGroupKFold(n_splits=2, shuffle=True, random_state=42)
    val_idx, test_idx = next(sgkf_val.split(df_valtest['image_t2'], df_valtest['clin_sig'], df_valtest['patient_id']))
    df_val = df_valtest.iloc[val_idx].copy()
    df_test = df_valtest.iloc[test_idx].copy()
        
    df_train['split'] = 'train'
    df_val['split'] = 'val'
    df_test['split'] = 'test'
    
    final_df = pd.concat([df_train, df_val, df_test])
    out_json = BASE_DIR / "data" / "kaggle_dataset_split.json"
    final_df.to_json(out_json, orient='records', indent=4)
    log.info(f"Split asegurado (Zero-Leakage): {len(df_train)} Train | {len(df_val)} Val | {len(df_test)} Test. -> {out_json.name}")

if __name__ == '__main__':
    run()
