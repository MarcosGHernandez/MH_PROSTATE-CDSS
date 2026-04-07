"""
MH PROSTATE-CDSS | Phase 4 — Advanced Orchestration
===================================================
Objetivo: Orquestador definitivo de MH PROSTATE-CDSS. Integra:
  1. XGBoost: Predicción de % de riesgo csPCa
  2. ChromaDB: Recuperación de Guías Clínicas (Semantic RAG)
  3. ChromaDB: Comparativa de Casos Históricos similares
  4. Ollama: Síntesis final con LLM (Llama 4 Scout o fallback)

Hardware: Optimizado para NVIDIA RTX 3070 Ti (8GB VRAM) -> num_gpu=1
Zero-Egress: 100% ejecución local.
"""

import json
import logging
import gc
import subprocess
import time
import sys
from pathlib import Path

BASE_DIR        = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(BASE_DIR))

try:
    import ollama
    import pandas as pd
    import xgboost as xgb
    import chromadb
    import shap
    from typing import Dict, Any
except ImportError:
    raise ImportError("Falta una librería. Ejecute: pip install ollama pandas xgboost chromadb shap")

from src.rag.prompts import MH_PROSTATE_CDSS_SYSTEM_PROMPT

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(name)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("mh-orchestrator")

BASE_DIR        = Path(__file__).resolve().parent.parent.parent
MODEL_PATH      = BASE_DIR / "models" / "predictive" / "xgboost_cspca_v1.json"
META_PATH       = BASE_DIR / "models" / "predictive" / "model_metadata.json"
VECTOR_DB_DIR   = BASE_DIR / "data" / "vector_db"
REPORTS_DIR     = BASE_DIR / "reports"

PRIMARY_MODEL   = "llama4-scout"  # Fallback a llama3.1:8b si no existe
FALLBACK_MODEL  = "llama3.1:8b"


# ══════════════════════════════════════════ 1. LLM CHECK ════════════════════

def get_ollama_model() -> str:
    """
    Verifica si llama4-scout está disponible en Ollama.
    Si no, verifica llama3.1:8b. Si ninguno está, intenta hacer pull del fallback.
    """
    log.info("Verificando modelos disponibles en Ollama local...")
    try:
        models = [m['model'] for m in ollama.list()['models']]
    except Exception as e:
        log.error("Ollama no parece estar ejecutándose. ¿Inició el servicio Ollama?")
        raise e

    if any(PRIMARY_MODEL in m for m in models):
        log.info("✅ Modelo primario encontrado: %s", PRIMARY_MODEL)
        return PRIMARY_MODEL
    elif any(FALLBACK_MODEL in m for m in models):
        log.warning("⚠️ %s no encontrado. Usando fallback: %s", PRIMARY_MODEL, FALLBACK_MODEL)
        return FALLBACK_MODEL
    else:
        log.warning("⚠️ Ningún modelo encontrado. Intentando descargar %s ...", FALLBACK_MODEL)
        try:
            ollama.pull(FALLBACK_MODEL)
            log.info("✅ Descarga completada: %s", FALLBACK_MODEL)
            return FALLBACK_MODEL
        except Exception as pull_err:
            log.error("Error descargando el modelo: %s", pull_err)
            raise RuntimeError(f"No se pudo inicializar ningún modelo en Ollama.")


# ══════════════════════════════════════════ 2. ML / RAG PIPELINE ═════════════

class MHProstateOrchestrator:
    def __init__(self):
        log.info("Inicializando MH Prostate-CDSS Orchestrator...")
        
        # 1. Cargar ML Model
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Modelo XGBoost no encontrado en {MODEL_PATH}")
        self.xgb_model = xgb.XGBClassifier()
        self.xgb_model.load_model(str(MODEL_PATH))
        
        with open(META_PATH, "r") as f:
            self.metadata   = json.load(f)
            self.features   = self.metadata["features"]
            self.threshold  = self.metadata["threshold"]
            
        log.info("Modelo XGBoost cargado. Features esperadas: %s (Threshold: %.3f)", 
                 len(self.features), self.threshold)

        # 2. Conectar ChromaDB Data
        self.chroma = chromadb.PersistentClient(path=str(VECTOR_DB_DIR))
        self.col_guidelines = self.chroma.get_collection("clinical_guidelines")
        self.col_historical = self.chroma.get_collection("historical_cases")
        log.info("ChromaDB conectado. Guidelines: %d | Historical: %d", 
                 self.col_guidelines.count(), self.col_historical.count())
                 
        # 3. Determinar LLM
        self.llm_model = get_ollama_model()

    def predict_risk(self, patient_data: dict) -> Dict[str, Any]:
        """Calcula el riesgo ML (XGBoost) basado en la data tabular."""
        # Convertir a DataFrame con las columnas exactas del modelo
        df = pd.DataFrame([patient_data])
        
        # Llenar faltantes temporales con 0 
        for col in self.features:
            if col not in df.columns:
                df[col] = 0.0
                
        X = df[self.features]
        proba = float(self.xgb_model.predict_proba(X)[0][1]) 
        
        # ─── NUEVO: SHAP Local Explainability (XAI) ───
        explainer = shap.TreeExplainer(self.xgb_model)
        shap_values = explainer.shap_values(X)
        
        # Obtener los 2 features con más impacto (valor absoluto)
        import numpy as np
        patient_shaps = shap_values[0]
        top_indices = np.argsort(np.abs(patient_shaps))[-2:][::-1]
        
        shap_explanation = []
        for idx in top_indices:
            feat_name = self.features[idx]
            feat_impact = float(patient_shaps[idx])
            direction = "Aumentó el riesgo" if feat_impact > 0 else "Redujo el riesgo"
            shap_explanation.append(f"{feat_name} ({direction}, impacto: {feat_impact:.3f})")

        return {
            "ml_risk_percent": round(proba * 100, 2),
            "ml_threshold": round(self.threshold * 100, 2),
            "ml_prediction": "ALTO RIESGO" if proba >= self.threshold else "Bajo riesgo relativo",
            "top_2_shap_features": shap_explanation
        }

    def retrieve_guidelines(self, patient_data: dict) -> list[str]:
        """Busca top 3 fragmentos semánticos en las guías AUA/EAU 2025."""
        psa = patient_data.get('psa', 'N/D')
        psad = patient_data.get('psad', 'N/D')
        
        # Formulamos la query basada en los parámetros más críticos
        query = f"Prostate cancer PSA level {psa} ng/mL and PSA density {psad}. Biopsy recommendations and risk stratification."
        
        results = self.col_guidelines.query(
            query_texts=[query],
            n_results=2,  # Reducido de 3 a 2 para conservar ventana de contexto
            include=["documents", "metadatas"]
        )
        
        snippets = []
        if results and results["documents"] and results["documents"][0]:
            for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
                src = meta.get("source", "Guía Médica")
                page = meta.get("page", "?")
                snippets.append(f"[{src} - p.{page}]: {doc}")
                
        return snippets

    def retrieve_historical(self, patient_data: dict) -> list[str]:
        """Busca top 2 casos clínicos históricos más parecidos."""
        psa = patient_data.get('psa', 'N/D')
        psad = patient_data.get('psad', 'N/D')
        age = patient_data.get('age', 'N/D')
        
        query = f"Paciente de {age} años con PSA {psa} ng/mL y densidad prostática {psad}."
        
        results = self.col_historical.query(
            query_texts=[query],
            n_results=1,  # Reducido de 2 a 1 para conservar ventana de contexto
            include=["documents", "metadatas"]
        )
        
        cases = []
        if results and results["documents"] and results["documents"][0]:
            for doc in results["documents"][0]:
                cases.append(doc)
                
        return cases


# ══════════════════════════════════════════ 3. CORE ORCHESTRATION ════════════

def run_analysis(patient_data: dict) -> str:
    """
    Flujo principal de orquestación (ML -> RAG -> LLM).
    Instancia el orquestador, evalúa las tres ramas y pasa el JSON a Ollama.
    """
    orchestrator = MHProstateOrchestrator()
    
    log.info("1. XGBoost ML Risk Step...")
    ml_results = orchestrator.predict_risk(patient_data)
    log.info("Riesgo calculado: %.2f%% (%s)", ml_results['ml_risk_percent'], ml_results['ml_prediction'])
    
    log.info("2. Guías Clínicas (Semantic Search)...")
    guidelines = orchestrator.retrieve_guidelines(patient_data)
    
    log.info("3. Casos Históricos (Semantic Search)...")
    history = orchestrator.retrieve_historical(patient_data)
    
    # Ensamblar el Payload JSON estructurado para el LLM
    payload = {
        "paciente_actual": patient_data,
        "evaluacion_ml_xgboost": ml_results,
        "evidencia_guias_clinicas": guidelines,
        "casos_historicos_similares": history
    }
    
    log.info("4. LLM Synthesis Step (Ollama: %s)...", orchestrator.llm_model)
    payload_str = json.dumps(payload, ensure_ascii=False, indent=2)
    
    # Garbage collection antes de la inferencia LLM
    gc.collect()
    log.info("Memoria liberada (gc.collect). Iniciando inferencia Ollama...")
    
    # Petición a Ollama con Error Handling
    # num_gpu=8 — sweet spot para RTX 3070 Ti 8GB (OS+Streamlit en memoria)
    start_llm = time.time()
    try:
        response = ollama.chat(
            model=orchestrator.llm_model,
            messages=[
                {'role': 'system', 'content': MH_PROSTATE_CDSS_SYSTEM_PROMPT},
                {'role': 'user', 'content': f"Genera el reporte médico final basado en estos datos:\n\n{payload_str}"}
            ],
            options={
                "num_gpu":    0,     # CPU/RAM -- garantiza ejecucion en cualquier estado de VRAM
                "num_predict": 1024,
                "temperature": 0.2,
                "top_p":       0.9
            }
        )
        report = response['message']['content']
    except Exception as e:
        log.error("Error durante la inferencia Ollama: %s", e)
        # ── Deterministic RAG Fallback (Zero-VRAM Error Handling) ──
        risk = ml_results.get("ml_risk_percent", "N/D")
        prediction = ml_results.get("ml_prediction", "N/D")
        shap = ml_results.get("top_2_shap_features", [])
        shap_text = "\n".join([f"- {s}" for s in shap]) if shap else "N/D"
        g_text = "\n".join([f"- {g}" for g in guidelines]) if guidelines else "Sin evidencia recuperada."
        h_text = "\n".join([f"- {h}" for h in history]) if history else "Sin historial similar."
        
        report = (
            "SISTEMA EN MODO FALLBACK ESTATICO (Baja Memoria)\n"
            "El analizador de lenguaje estendido no pudo instanciarse por limite de RAM/VRAM en el host. "
            "A continuacion se presenta la extraccion deterministica estructurada del modelo predictivo y la base de conocimiento.\n\n"
            "### 1. Resumen de Riesgo Predictivo (XGBoost)\n"
            f"- **Riesgo Calculado:** {risk}%\n"
            f"- **Interpretacion ML:** {prediction}\n"
            "**Drivers Principales (SHAP Extractions):**\n"
            f"{shap_text}\n\n"
            "### 2. Evidencia Clinica Directa (EAU/AUA 2025 RAG)\n"
            "Fragmentos semanticos recuperados:\n"
            f"{g_text}\n\n"
            "### 3. Historico de Casos Similares\n"
            f"{h_text}\n\n"
            "### 4. Siguiente Paso\n"
            "Evaluar el riesgo predictivo en conjunto con los fragmentos de la guia para determinar necesidad de biopsia o RM multiparametrica.\n\n"
            "### 5. Disclaimer\n"
            "Nota Legal y Etica: MH Prostate-CDSS es una herramienta de soporte analitico predictivo. No constituye diagnostico medico definitivo. La responsabilidad final recae en el urologo tratante."
        )
        
    llm_time = time.time() - start_llm
    log.info("LLM Synthesis completado en %.1f segundos.", llm_time)
    
    return report


# ══════════════════════════════════════════ MAIN (VALIDATION BLOCK) ══════════

def test_execution():
    log.info("=== MH Prostate-CDSS | Test Execution ===")
    REPORTS_DIR.mkdir(exist_ok=True)
    
    # Paciente de alto riesgo dictado en el prompt
    mock_patient = {
        "age": 70,
        "psa": 9.5,
        "prostate_volume": 43.0,    # 9.5 / 0.22 ≈ 43.1 cm3
        "psad": 0.22,
        "nlr": 3.5,
        "albumin": 4.1,
        "crp": 3.0,
        "family_history": 1,
        "dre_result": 1             # Tacto rectal sospechoso
    }
    
    log.info("Iniciando análisis para paciente mock de alto riesgo...")
    final_report = run_analysis(mock_patient)
    
    # Imprimir a terminal y guardar
    print("\n\n" + "="*60)
    print("🏥 REPORTE CLÍNICO FINAL (MH Prostate-CDSS)")
    print("="*60)
    print(final_report)
    print("="*60 + "\n")
    
    report_file = REPORTS_DIR / "final_ebm_report_v1.md"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("# MH Prostate-CDSS Clinical Report (v1 EBM)\n\n")
        f.write(f"**Fecha:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(final_report)
        
    log.info("✅ Reporte guardado exitosamente en: %s", report_file)


if __name__ == "__main__":
    test_execution()
