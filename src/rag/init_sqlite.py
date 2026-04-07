"""
Viko-Health | Phase 3 — SQLite Relational Database Initializer
==============================================================
Autor: Senior AI Architect
Objetivo: Cargar el cohort de alta fidelidad (PI-CAI + Turkish) en un
          base de datos SQLite local para consultas SQL rápidas sin
          necesidad de cargar CSVs masivos en memoria.

Salida:
  - data/database/viko_patients.db  → tabla: patients_history

Schema de la tabla:
  Todos los campos de master_normalized.csv filtrados por fuente HF,
  más columnas de autoincremento y timestamp de ingesta.

Zero-Egress: 100% local. SQLite nativo, sin dependencias externas.
"""

import logging
import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("viko-sqlite")

BASE_DIR   = Path(__file__).resolve().parent.parent.parent
CSV_PATH   = BASE_DIR / "data" / "processed" / "master_normalized.csv"
DB_DIR     = BASE_DIR / "data" / "database"
DB_PATH    = DB_DIR / "viko_patients.db"
TABLE_NAME = "patients_history"
HF_SOURCES = ["picai", "turkish"]


def load_hf_cohort() -> pd.DataFrame:
    """Lee el CSV maestro y filtra el cohort de alta fidelidad."""
    log.info("Leyendo CSV maestro: %s", CSV_PATH)
    df = pd.read_csv(CSV_PATH)
    hf = df[df["source"].isin(HF_SOURCES)].copy()
    hf = hf.dropna(subset=["target_cspca"])
    hf["ingested_at"] = datetime.utcnow().isoformat()
    log.info("Cohort HF cargado: %d filas | Fuentes: %s",
             len(hf), hf["source"].value_counts().to_dict())
    return hf


def init_sqlite(df: pd.DataFrame) -> None:
    """
    Inicializa la base de datos SQLite y carga el cohort HF.
    Aplica IF EXISTS REPLACE para idempotencia (re-ejecución segura).
    """
    DB_DIR.mkdir(parents=True, exist_ok=True)
    log.info("Conectando a SQLite: %s", DB_PATH)

    with sqlite3.connect(DB_PATH) as conn:
        # Cargar dataframe → tabla SQLite (reemplaza si ya existe)
        df.to_sql(TABLE_NAME, conn, if_exists="replace", index=False)

        # Crear índices para consultas frecuentes
        cursor = conn.cursor()
        cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_source     ON {TABLE_NAME}(source)")
        cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_target     ON {TABLE_NAME}(target_cspca)")
        cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_psa        ON {TABLE_NAME}(psa)")
        cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_psad_alert ON {TABLE_NAME}(psad_alert)")
        conn.commit()

        # Verificación post-carga
        count = conn.execute(f"SELECT COUNT(*) FROM {TABLE_NAME}").fetchone()[0]
        cols  = [r[1] for r in conn.execute(f"PRAGMA table_info({TABLE_NAME})")]

        log.info("═══════════════════════════════════")
        log.info("SQLite inicializado correctamente")
        log.info("  Archivo : %s", DB_PATH)
        log.info("  Tabla   : %s", TABLE_NAME)
        log.info("  Filas   : %d", count)
        log.info("  Columnas: %d → %s", len(cols), cols)
        log.info("═══════════════════════════════════")

        # Muestra de consulta SQL de validación
        sample = conn.execute(
            f"SELECT source, COUNT(*) as n, AVG(psa) as avg_psa, "
            f"SUM(target_cspca) as cspca_pos "
            f"FROM {TABLE_NAME} GROUP BY source"
        ).fetchall()
        log.info("Resumen por fuente:")
        for row in sample:
            log.info("  %s", row)


def validate_queries() -> None:
    """Ejecuta consultas de validación clínica de referencia."""
    with sqlite3.connect(DB_PATH) as conn:
        # Consulta 1: Pacientes con PSAd alerta y csPCa positivo
        q1 = conn.execute(
            f"SELECT COUNT(*) FROM {TABLE_NAME} "
            f"WHERE psad_alert = 1 AND target_cspca = 1"
        ).fetchone()[0]
        log.info("Pacientes PSAd≥0.15 Y csPCa=1: %d", q1)

        # Consulta 2: Estadísticas PSA por resultado
        q2 = conn.execute(
            f"SELECT target_cspca, ROUND(AVG(psa),2), ROUND(AVG(psad),3) "
            f"FROM {TABLE_NAME} GROUP BY target_cspca"
        ).fetchall()
        log.info("PSA/PSAd medio por resultado clínico: %s", q2)

        # Consulta 3: Distribución edades por grupo
        q3 = conn.execute(
            f"SELECT CAST(age/10*10 AS INT) as decade, COUNT(*) "
            f"FROM {TABLE_NAME} GROUP BY decade ORDER BY decade"
        ).fetchall()
        log.info("Distribución por década de edad: %s", q3)


def run_init():
    log.info("=== Viko-Health | Phase 3 — SQLite Init ===")
    df = load_hf_cohort()
    init_sqlite(df)
    validate_queries()
    log.info("=== SQLite listo para consultas clínicas ===")


if __name__ == "__main__":
    run_init()
