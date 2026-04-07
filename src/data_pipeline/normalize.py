"""
MH PROSTATE-CDSS | Data Normalization Pipeline
==============================================
Autor: Senior AI Architect
Objetivo: Ingestar, limpiar, normalizar y unificar los tres datasets crudos
          en un único archivo master_normalized.csv en data/processed/.

Fuentes:
  - data/raw/prostat_ca_veri_seti_duzeltilmis_v2.csv  (Dataset Turco)
  - data/raw/prostate_cancer_prediction.csv           (Dataset General Kaggle)
  - data/raw/picai_labels/clinical_information/marksheet.csv (PI-CAI)

Salida:
  - data/processed/master_normalized.csv
"""

import os
import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer

# ──────────────────────────────────────────────────────────── Configuración ─────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("mh-prostate-normalizer")

BASE_DIR = Path(__file__).resolve().parent.parent.parent
RAW_DIR  = BASE_DIR / "data" / "raw"
OUT_DIR  = BASE_DIR / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PSAD_ALERT_THRESHOLD = 0.15
PSA_OUTLIER_UPPER    = 200.0
PSA_OUTLIER_LOWER    = 0.01


# ═══════════════════════════════════ 1. CARGA ═════════════════════════════════

def load_turkish_dataset(path: Path) -> pd.DataFrame:
    """Turco → esquema Viko-Health. Incluye marcadores inflamatorios."""
    log.info("Cargando dataset turco: %s", path.name)
    df = pd.read_csv(path, sep=",", low_memory=False)

    col_map = {
        "Hasta_ID"       : "patient_id",
        "Yas"            : "age",
        "Tani_Tarihi"    : "diagnosis_date",
        "PSA_Tani"       : "psa",
        "Klinik_Evre"    : "clinical_stage",
        "Biyopsi_Gleason": "biopsy_gleason",
        "Risk_Grubu"     : "risk_group",
        "Albumin"        : "albumin",
        "Lenfosit"       : "lymphocytes",
        "CRP"            : "crp",
        "NLR"            : "nlr",
        "CALLY_Index"    : "cally_index",
        "BCR_Durum"      : "bcr_status",
        "Patolojik_Evre" : "pathological_stage",
        "Final_Gleason"  : "final_gleason",
    }
    df.rename(columns={k: v for k, v in col_map.items() if k in df.columns}, inplace=True)

    # Convertir posibles comas decimales en columnas numéricas
    numeric_cols = ["psa", "albumin", "nlr", "crp", "lymphocytes", "cally_index"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = (
                df[col].astype(str)
                       .str.replace(",", ".", regex=False)
                       .pipe(pd.to_numeric, errors="coerce")
            )

    # Target: Risk_Grubu ≥ 2 = csPCa significativo (Alto/Muy Alto riesgo)
    if "risk_group" in df.columns:
        df["target_cspca"] = (pd.to_numeric(df["risk_group"], errors="coerce") >= 2).astype(int)

    df["source"] = "turkish"
    log.info("Dataset turco cargado: %d filas.", len(df))
    return df


def load_general_dataset(path: Path) -> pd.DataFrame:
    """Dataset Kaggle de predicción general (27k+ registros)."""
    log.info("Cargando dataset general: %s", path.name)
    df = pd.read_csv(path, sep=",", low_memory=False)

    col_map = {
        "Patient_ID"      : "patient_id",
        "Age"             : "age",
        "PSA_Level"       : "psa",
        "Prostate_Volume" : "prostate_volume",
        "DRE_Result"      : "dre_result",
        "Biopsy_Result"   : "biopsy_result_raw",
        "Family_History"  : "family_history",
        "Cancer_Stage"    : "cancer_stage",
    }
    df.rename(columns={k: v for k, v in col_map.items() if k in df.columns}, inplace=True)

    # Target: Malignant → 1
    if "biopsy_result_raw" in df.columns:
        df["target_cspca"] = (
            df["biopsy_result_raw"].str.strip().str.lower() == "malignant"
        ).astype(int)

    # Binarias
    if "family_history" in df.columns:
        df["family_history"] = df["family_history"].str.strip().str.lower().map({"yes": 1, "no": 0})
    if "dre_result" in df.columns:
        df["dre_result"] = (df["dre_result"].str.strip().str.lower() == "abnormal").astype(int)

    # Coerce numéricos
    for col in ["psa", "prostate_volume", "age"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["source"] = "general"
    log.info("Dataset general cargado: %d filas.", len(df))
    return df


def load_picai_dataset(path: Path) -> pd.DataFrame:
    """PI-CAI — pivote de validación clínica de alta fidelidad."""
    log.info("Cargando dataset PI-CAI: %s", path.name)
    df = pd.read_csv(path, sep=",", low_memory=False)

    col_map = {
        "patient_id"     : "patient_id",
        "patient_age"    : "age",
        "psa"            : "psa",
        "psad"           : "psad",
        "prostate_volume": "prostate_volume",
        "case_ISUP"      : "isup_grade",
        "case_csPCa"     : "cspca_raw",
        "histopath_type" : "biopsy_method",
    }
    df.rename(columns={k: v for k, v in col_map.items() if k in df.columns}, inplace=True)

    # Target: YES → 1
    if "cspca_raw" in df.columns:
        df["target_cspca"] = (df["cspca_raw"].str.strip().str.upper() == "YES").astype(int)

    for col in ["psa", "psad", "prostate_volume", "age", "isup_grade"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["source"] = "picai"
    log.info("PI-CAI cargado: %d filas.", len(df))
    return df


# ═══════════════════════════════════ 2. SANITIZACIÓN ══════════════════════════

def sanitize_psa(df: pd.DataFrame) -> pd.DataFrame:
    """Descartar PSA imposibles convirtiéndolos a NaN para posterior imputación."""
    if "psa" not in df.columns:
        return df
    df["psa"] = pd.to_numeric(df["psa"], errors="coerce")
    outlier = (df["psa"] < PSA_OUTLIER_LOWER) | (df["psa"] > PSA_OUTLIER_UPPER)
    n = outlier.sum()
    if n:
        src = df["source"].iloc[0] if "source" in df.columns else "?"
        log.warning("[%s] %d PSA outliers → NaN (rango %.2f–%.2f).", src, n, PSA_OUTLIER_LOWER, PSA_OUTLIER_UPPER)
        df.loc[outlier, "psa"] = np.nan
    return df


# ═══════════════════════════════════ 3. FEATURE ENGINEERING ═══════════════════

def calculate_psad(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula PSAd donde no esté disponible.
    Guarda contra columnas faltantes para ser seguro en cualquier dataset.
    """
    has_psa    = "psa" in df.columns
    has_volume = "prostate_volume" in df.columns

    if "psad" not in df.columns:
        df["psad"] = np.nan

    if has_psa and has_volume:
        mask = df["psad"].isna() & df["psa"].notna() & df["prostate_volume"].notna()
        df.loc[mask, "psad"] = df.loc[mask, "psa"] / df.loc[mask, "prostate_volume"]
        n = mask.sum()
        if n:
            src = df["source"].iloc[0] if "source" in df.columns else "?"
            log.info("[%s] PSAd calculado para %d registros.", src, n)

    df["psad_alert"] = (df["psad"] >= PSAD_ALERT_THRESHOLD).astype("Int64")
    return df


# ═══════════════════════════════════ 4. UNIFICACIÓN ═══════════════════════════

MASTER_COLS = [
    "source", "patient_id", "age", "psa", "prostate_volume",
    "psad", "psad_alert", "nlr", "albumin", "crp",
    "dre_result", "family_history", "isup_grade",
    "clinical_stage", "biopsy_method", "target_cspca",
]


def unify_datasets(dfs: list) -> pd.DataFrame:
    log.info("Unificando %d datasets...", len(dfs))
    frames = []
    for df in dfs:
        sub = pd.DataFrame(index=df.index)
        for col in MASTER_COLS:
            sub[col] = df[col] if col in df.columns else np.nan
        frames.append(sub)
    master = pd.concat(frames, ignore_index=True)
    log.info("Dataset unificado: %d filas totales.", len(master))
    return master


# ═══════════════════════════════════ 5. IMPUTACIÓN ════════════════════════════

def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    """MICE (IterativeImputer) en columnas numéricas críticas."""
    impute_cols = ["age", "psa", "prostate_volume", "psad", "nlr", "albumin", "crp"]
    available   = [c for c in impute_cols if c in df.columns]

    n_before = df[available].isna().sum().sum()
    log.info("Imputando %d valores faltantes con MICE en: %s", n_before, available)

    imputer = IterativeImputer(
        max_iter=10,
        random_state=42,
        initial_strategy="median",
        add_indicator=False,
    )
    df[available] = imputer.fit_transform(df[available])

    n_after = df[available].isna().sum().sum()
    log.info("Nulos restantes tras imputación: %d", n_after)

    # Recalcular psad_alert post-imputación
    df["psad_alert"] = (df["psad"] >= PSAD_ALERT_THRESHOLD).astype("Int64")
    return df


# ═══════════════════════════════════ 6. VALIDACIÓN ════════════════════════════

def validate_and_report(df: pd.DataFrame) -> None:
    log.info("══════════════════════════════════════")
    log.info("RESUMEN DATASET MAESTRO FINAL")
    log.info("══════════════════════════════════════")
    log.info("Filas totales       : %d", len(df))
    log.info("Columnas            : %d", len(df.columns))
    log.info("Fuentes             : %s", df["source"].value_counts().to_dict())
    log.info("Target csPCa dist.  : %s", df["target_cspca"].value_counts(dropna=False).to_dict())
    log.info("PSAd alerta (≥0.15) : %d", int(df["psad_alert"].sum()))
    nulos = df.isna().sum()
    nulos = nulos[nulos > 0]
    if not nulos.empty:
        log.info("Nulos restantes:\n%s", nulos.to_string())
    key = [c for c in ["age", "psa", "psad", "nlr", "albumin"] if c in df.columns]
    print("\n📊 Estadísticas de biomarcadores clave:")
    print(df[key].describe().round(3).to_string())


# ═══════════════════════════════════ MAIN ═════════════════════════════════════

def run_pipeline():
    log.info("=== MH PROSTATE-CDSS | Data Normalization Pipeline Initiated ===")

    turkish_path = RAW_DIR / "prostat_ca_veri_seti_duzeltilmis_v2.csv"
    general_path = RAW_DIR / "prostate_cancer_prediction.csv"
    picai_path   = RAW_DIR / "picai_labels" / "clinical_information" / "marksheet.csv"

    for p in [turkish_path, general_path, picai_path]:
        if not p.exists():
            log.error("Archivo no encontrado: %s", p)
            sys.exit(1)

    df_t = load_turkish_dataset(turkish_path)
    df_g = load_general_dataset(general_path)
    df_p = load_picai_dataset(picai_path)

    for df in [df_t, df_g, df_p]:
        sanitize_psa(df)
        df = calculate_psad(df)

    # Recalculate on cleaned copies
    df_t = calculate_psad(sanitize_psa(df_t))
    df_g = calculate_psad(sanitize_psa(df_g))
    df_p = calculate_psad(sanitize_psa(df_p))

    master = unify_datasets([df_t, df_g, df_p])
    master = impute_missing(master)

    # Tipos finales
    master["age"]          = master["age"].round(0).astype("Int64")
    master["isup_grade"]   = pd.to_numeric(master["isup_grade"],  errors="coerce")
    master["target_cspca"] = pd.to_numeric(master["target_cspca"], errors="coerce").astype("Int64")
    master["psad_alert"]   = pd.to_numeric(master["psad_alert"],   errors="coerce").astype("Int64")

    output_path = OUT_DIR / "master_normalized.csv"
    master.to_csv(output_path, index=False)
    log.info("✅ Dataset maestro guardado en: %s", output_path)

    validate_and_report(master)
    log.info("=== Pipeline finalizado con éxito ===")
    return master


if __name__ == "__main__":
    run_pipeline()
