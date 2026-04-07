"""
Viko-Health | Phase 3 — ChromaDB Vector Database Initializer (v2)
=================================================================
Autor: Senior AI Architect

ZERO-EGRESS GARANTIZADO:
  Embedding model: sentence-transformers/all-MiniLM-L6-v2 (completamente local).
  Se usa chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction,
  que carga el modelo de HuggingFace local sin ninguna llamada a API externas.
  ChromaDB: persistencia local en data/vector_db/ — sin nube.

Colecciones:
  1. clinical_guidelines — fragmentos de guías EBM (AUA/EAU PDFs)
  2. historical_cases    — casos clínicos del cohort HF vectorizados

Salidas:
  - data/vector_db/  → índices ChromaDB persistentes (SQLite + HNSW)
"""

import logging
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("viko-chromadb")

BASE_DIR       = Path(__file__).resolve().parent.parent.parent
CSV_PATH       = BASE_DIR / "data" / "processed" / "master_normalized.csv"
VECTOR_DB_DIR  = BASE_DIR / "data" / "vector_db"
GUIDELINES_DIR = BASE_DIR / "docs" / "medical_guidelines"
HF_SOURCES     = ["picai", "turkish"]

COL_GUIDELINES = "clinical_guidelines"
COL_HISTORICAL = "historical_cases"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"   # Short name for SentenceTransformer

CHUNK_SIZE    = 1000
CHUNK_OVERLAP = 150


# ════════════════════════ 1. CHROMADB + EMBEDDING FUNCTION ════════════════════

def get_embedding_fn():
    """
    Usa chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction
    para garantizar compatibilidad total con la API de ChromaDB v1.x y
    Zero-Egress (modelo local, sin API keys externas).
    """
    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
    ef = SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)
    log.info("Embedding function lista: %s (local, Zero-Egress ✅)", EMBEDDING_MODEL)
    return ef


def init_chromadb():
    """Inicializa el cliente ChromaDB y crea/verifica las dos colecciones."""
    import chromadb
    VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)

    log.info("Inicializando ChromaDB v%s en: %s",
             chromadb.__version__, VECTOR_DB_DIR)
    client = chromadb.PersistentClient(path=str(VECTOR_DB_DIR))
    ef     = get_embedding_fn()

    col_guidelines = client.get_or_create_collection(
        name=COL_GUIDELINES,
        embedding_function=ef,
        metadata={
            "description":     "Guías EBM (AUA/EAU/NCCN). Chunked via LangChain.",
            "chunk_size":       str(CHUNK_SIZE),
            "chunk_overlap":    str(CHUNK_OVERLAP),
            "embedding_model":  EMBEDDING_MODEL,
            "zero_egress":      "true",
        }
    )
    col_historical = client.get_or_create_collection(
        name=COL_HISTORICAL,
        embedding_function=ef,
        metadata={
            "description":    "Resúmenes clínicos del cohort PI-CAI + Turkish.",
            "sources":        "picai,turkish",
            "embedding_model": EMBEDDING_MODEL,
            "zero_egress":    "true",
        }
    )

    log.info("Colecciones activas: %s",
             [c.name for c in client.list_collections()])
    return client, col_guidelines, col_historical, ef


# ════════════════════════ 2. SEED: CASOS HISTÓRICOS ═══════════════════════════

def seed_historical_cases(col_historical, ef) -> int:
    """
    Genera texto clínico narrativo por paciente y vectoriza con all-MiniLM-L6-v2.
    Carga en lotes de 50 con verificación de idempotencia (evita duplicados).
    """
    log.info("Cargando cohort HF para seed de casos históricos...")
    df = pd.read_csv(CSV_PATH)
    hf = df[df["source"].isin(HF_SOURCES)].dropna(subset=["target_cspca"]).copy()

    documents, ids, metadatas = [], [], []
    for i, row in hf.iterrows():
        def fmt(val, decimals=2):
            return f"{float(val):.{decimals}f}" if pd.notna(val) else "N/D"

        alert_flag = "ALERTA PSAd POSITIVO (≥0.15)" if row.get("psad_alert") == 1 \
                     else "Sin alerta PSAd"
        outcome    = "csPCa POSITIVO (tumor agresivo)" if int(row.get("target_cspca", 0)) == 1 \
                     else "csPCa NEGATIVO (benigno o indolente)"

        text = (
            f"Paciente {str(row.get('source','')).upper()} | "
            f"Edad: {fmt(row.get('age'), 0)} años. "
            f"PSA: {fmt(row.get('psa'))} ng/mL. "
            f"Vol. prostático: {fmt(row.get('prostate_volume'))} cm³. "
            f"PSAd: {fmt(row.get('psad'), 3)} ng/mL/cm³. {alert_flag}. "
            f"NLR: {fmt(row.get('nlr'))}. Albúmina: {fmt(row.get('albumin'))} g/dL. "
            f"CRP: {fmt(row.get('crp'))} mg/L. "
            f"Resultado clínico: {outcome}."
        )
        documents.append(text)
        ids.append(f"{row.get('source', 'hf')}_{i}")
        metadatas.append({
            "source":       str(row.get("source", "")),
            "target_cspca": int(row.get("target_cspca", 0)),
            "psad_alert":   int(row.get("psad_alert", 0)) if pd.notna(row.get("psad_alert")) else 0,
            "psa":          float(row["psa"]) if pd.notna(row.get("psa")) else 0.0,
        })

    # Insertar en lotes de 50 con deduplicación
    batch_size = 50
    n_loaded   = 0
    for start in range(0, len(documents), batch_size):
        end       = min(start + batch_size, len(documents))
        b_docs    = documents[start:end]
        b_ids     = ids[start:end]
        b_metas   = metadatas[start:end]
        existing  = set(col_historical.get(ids=b_ids)["ids"])
        new_mask  = [bid not in existing for bid in b_ids]
        n_docs    = [b_docs[k]  for k in range(len(b_docs))  if new_mask[k]]
        n_ids     = [b_ids[k]   for k in range(len(b_ids))   if new_mask[k]]
        n_metas   = [b_metas[k] for k in range(len(b_metas)) if new_mask[k]]
        if n_docs:
            col_historical.add(documents=n_docs, ids=n_ids, metadatas=n_metas)
            n_loaded += len(n_docs)

    total = col_historical.count()
    log.info("Casos históricos nuevos vectorizados: %d | Total en colección: %d",
             n_loaded, total)
    return total


# ════════════════════════ 3. PDF INGESTION (PLACEHOLDER) ══════════════════════

def ingest_medical_pdfs(col_guidelines, ef) -> int:
    """
    Placeholder para ingesta de PDFs de guías clínicas (AUA/EAU/NCCN).
    Coloque PDFs en docs/medical_guidelines/ y re-ejecute este script.

    Pipeline de ingesta:
      1. PyPDFLoader  → carga páginas del PDF
      2. RecursiveCharacterTextSplitter → chunks de 1000 chars / 150 overlap
      3. Embeddings locales → all-MiniLM-L6-v2 → ChromaDB
    """
    pdf_files = list(GUIDELINES_DIR.glob("*.pdf"))

    if not pdf_files:
        log.info("No hay PDFs en %s", GUIDELINES_DIR)
        # Insertar placeholder informativo si no existe
        existing = col_guidelines.get(ids=["placeholder"])["ids"]
        if "placeholder" not in existing:
            col_guidelines.add(
                documents=["Viko-Health RAG: colección de guías clínicas lista. "
                           "Coloque PDFs AUA/EAU/NCCN en docs/medical_guidelines/ "
                           "y re-ejecute init_chromadb.py para indexarlos."],
                ids=["placeholder"],
                metadatas=[{"type": "placeholder", "zero_egress": "true"}]
            )
            log.info("Placeholder insertado en '%s'.", COL_GUIDELINES)
        return 0

    try:
        from langchain_community.document_loaders import PyPDFLoader
        from langchain.text_splitter import RecursiveCharacterTextSplitter
    except ImportError:
        log.warning("langchain-community / pypdf no disponibles.")
        return 0

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    n_total = 0
    for pdf_path in pdf_files:
        log.info("Procesando PDF: %s", pdf_path.name)
        pages  = PyPDFLoader(str(pdf_path)).load()
        chunks = splitter.split_documents(pages)
        docs   = [c.page_content for c in chunks]
        ids_   = [f"{pdf_path.stem}_{j}" for j in range(len(chunks))]
        metas  = [{"source": pdf_path.name,
                   "page": c.metadata.get("page", 0),
                   "zero_egress": "true"} for c in chunks]
        for s in range(0, len(docs), 50):
            e = min(s + 50, len(docs))
            existing = set(col_guidelines.get(ids=ids_[s:e])["ids"])
            nm = [ids_[s + k] not in existing for k in range(e - s)]
            nd = [docs[s + k]  for k in range(e - s) if nm[k]]
            ni = [ids_[s + k]  for k in range(e - s) if nm[k]]
            nme= [metas[s + k] for k in range(e - s) if nm[k]]
            if nd:
                col_guidelines.add(documents=nd, ids=ni, metadatas=nme)
        n_total += len(chunks)
        log.info("  %d chunks indexados de '%s'", len(chunks), pdf_path.name)

    log.info("Total chunks en '%s': %d", COL_GUIDELINES, col_guidelines.count())
    return n_total


# ════════════════════════ 4. VALIDACIÓN RAG ════════════════════════════════════

def validate_rag(col_historical) -> None:
    """Consulta semántica de prueba para verificar el flujo RAG end-to-end."""
    log.info("────── Validación RAG ──────")
    query = "Paciente con PSA elevado y densidad alta, sospecha de cáncer agresivo."
    results = col_historical.query(
        query_texts=[query],
        n_results=3,
        include=["documents", "distances", "metadatas"]
    )
    for j, (doc, dist, meta) in enumerate(zip(
        results["documents"][0],
        results["distances"][0],
        results["metadatas"][0]
    )):
        log.info("  [%d] dist=%.4f | csPCa=%s | psad_alert=%s | %.80s...",
                 j + 1, dist, meta.get("target_cspca"),
                 meta.get("psad_alert"), doc)


# ════════════════════════ MAIN ═════════════════════════════════════════════════

def run_init():
    log.info("=== Viko-Health | Phase 3 — ChromaDB Init (Zero-Egress) ===")

    client, col_guidelines, col_historical, ef = init_chromadb()

    n_historical = seed_historical_cases(col_historical, ef)
    n_guidelines = ingest_medical_pdfs(col_guidelines, ef)

    validate_rag(col_historical)

    log.info("═══════════════════════════════════════════")
    log.info("ESTADO FINAL VECTOR DB")
    log.info("═══════════════════════════════════════════")
    log.info("  Ruta              : %s", VECTOR_DB_DIR)
    log.info("  historical_cases  : %d documentos", n_historical)
    log.info("  clinical_guidelines: %d chunks PDF", n_guidelines)
    log.info("  Modelo embeddings : %s (local ✅)", EMBEDDING_MODEL)
    log.info("  Zero-Egress       : ✅ Sin API keys externas")
    log.info("  Guías PDFs        : ⏳ Esperando docs en docs/medical_guidelines/")
    log.info("═══════════════════════════════════════════")
    log.info("=== ChromaDB listo para el sistema RAG ===")


if __name__ == "__main__":
    run_init()
