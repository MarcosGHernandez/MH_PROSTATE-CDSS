# MH PROSTATE-CDSS: Phase 1 Preliminary Dataset Analysis Report

## 1. Introduction
This report provides a comprehensive analysis of the raw clinical datasets utilized for the development of the **MH PROSTATE-CDSS** (Clinical Decision Support System). The data, primary stored in the `data/raw/` directory, encompasses biochemical markers and general clinical history. This document details the specific findings and proposes an unified schema for FHIR-ready normalization.

---

## 2. Dataset Identification and Benchmarking

### 2.1 Clinical Biochemical Dataset (`prostat_ca_veri_seti_duzeltilmis_v2.csv`)
- **Source:** Kaggle Clinical Cohort (Turkish Diagnostic).
- **Primary Features:** Systemic inflammatory markers (NLR, Albumin, CRP), which serve as foundational pillars for the risk stratification model.

### 2.2 Predictive General Dataset (`prostate_cancer_prediction.csv`)
- **Source:** Kaggle General Prediction Cohort.
- **Primary Features:** Measured prostate volume (cm³), essential for calculating PSA Density (PSAd).

### 2.3 PI-CAI Institutional Dataset (`marksheet.csv`)
- **Source:** PI-CAI Challenge (Radboud University). [Reference: https://pi-cai.grand-challenge.org/]
- **Primary Features:** High-fidelity clinical reference data associated with multiparametric MRI (mpMRI).

---

## 3. Proposed Schema for Multi-Source Normalization

The following unification schema is established to ensure compatibility with the FHIR-ready pipeline within `src/data_pipeline/`:

| MH PROSTATE-CDSS Variable | Source A | Source B | Source C | Data Type |
| :--- | :--- | :--- | :--- | :--- |
| `age` | `Yas` | `Age` | `patient_age` | Integer |
| `psa` | `PSA_Tani` | `PSA_Level` | `psa` | Float |
| `prostate_volume` | N/A | `Prostate_Volume` | `prostate_volume` | Float |
| `psad` | Calculated | Calculated | `psad` | Float |
| `nlr` | `NLR` | N/A | N/A | Float |
| `albumin` | `Albumin` | N/A | N/A | Float |
| `isup_grade` | Mapped | N/A | `case_ISUP` | Category (0-5) |
| `target_cspca` | Mapped | `Biopsy_Result`| `case_csPCa` | Boolean |

---

## 4. Technical Ingestion Protocol

1.  **PSA Density Calculation:** PSA Density ($PSAd = psa / volume$) will be programmatically derived for all records where prostate volume is reported.
2.  **Dataset Pre-processing:** Dataset A requires header translation and standardization of decimal separators to ensure consistent numerical ingestion.
3.  **Class Imbalance Management:** Observed prevalence of csPCa varies significantly across sources. The implementation of SMOTE-based augmentation is verified as essential for model training stability.

---
**Standard Compliance:** Zero-Egress | EBM 2026
**Project Lead:** Marcos Hernandez
