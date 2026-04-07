# MH PROSTATE-CDSS: Data Scaling and Transfer Learning Protocol

## 1. Technical Framework
This guide provides the standardized protocol for scaling the **MH PROSTATE-CDSS** Vision Engine with newly acquired clinical datasets. The primary objective is to facilitate **Transfer Learning (TL)**, allowing the model to adapt to new diagnostic patterns without degrading existing weights.

---

## 2. Directory Standardization
To ensure compatibility with the existing ingestion pipelines, new MRI cases should be structured consistently:

```text
data/
└── raw/
    └── new_cases/
        ├── images/
        │   ├── CaseNNN_T2.nii.gz
        │   └── CaseNNN_ADC.nii.gz
        └── masks/
            └── CaseNNN_label.nii.gz (Expert Delineation)
```

### 2.1 Imaging Compliance
Volumes must be provided as NIfTI (.nii or .nii.gz). T2W and ADC sequences must be co-registered, ensuring that the (X, Y, Z) coordinates and anatomical origins (World Coordinates) are identical.

---

## 3. Transfer Learning Protocol
To update the 3D U-Net weights while preserving pre-existing spatial patterns, use the following retraining protocol:

### Command Execution:
```bash
$env:PYTHONPATH = "src/vision"
python scripts/train_phase15_scaling.py --weights "models/vision/best_unet_prostate.pth" --lr 5e-6 --epochs 50
```

### 3.1 Rationale for Selection:
- **Learning Rate (5e-6):** A reduced learning rate prevents "Catastrophic Forgetting" during weight optimization.
- **Weights Pre-load:** Loading the baseline `best_unet_prostate.pth` provides a diagnostic foundation (14.25% Dice), substantially reducing convergence time.

---

## 4. Dataset Splitting Protocol
All new data must follow a **Patient-Level Grouping** strategy. It is imperative that all MRI sequences and masks corresponding to a single patient ID are maintained within the same dataset split (Training or Validation) to avoid data leakage and skewed metrics.

---

## 5. Performance Monitoring
Consistent with validation standards, monitor the **Val Dice** and **Val HD95** metrics. An upward trend in Dice or a further reduction in HD95 confirms successful scaling across the integrated dataset.

---
**Technical Reference:** MH PROSTATE-CDSS Suite  
**Compliance Standard:** Zero-Egress | EBM 2026
**Project Lead:** Marcos Hernandez
