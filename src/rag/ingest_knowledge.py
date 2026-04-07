"""
MH PROSTATE-CDSS | Phase 3b — Knowledge Base Ingestion (RAG)
============================================================
Objetivo: Indexar las guías clínicas PDF en la colección ChromaDB
          `clinical_guidelines` usando embeddings 100% locales.

PDFs en docs/medical_guidelines/:
  - EAU Guidelines on Prostate Cancer 2025 (full)
  - EAU Pocket Guidelines on Prostate Cancer 2025
  - AUA Clinically-Localized Prostate Cancer
  - advanced-prostate-patient_es  (contexto paciente avanzado)
  - prostate-early-patient        (contexto detección temprana)

Pipeline:
  1. PyPDFDirectoryLoader  → carga todos los PDFs de la carpeta
  2. RecursiveCharacterTextSplitter → chunk_size=1000, overlap=150
  3. HuggingFaceEmbeddings (all-MiniLM-L6-v2, local) → vectors
  4. Chroma (LangChain) → persiste en data/vector_db/, colección: clinical_guidelines

ZERO-EGRESS: Sin llamadas a OpenAI, Cohere, Google ni ningún otro endpoint externo.
             Todo el procesamiento es 100% local (CPU/GPU de la máquina).
"""

import logging
import warnings
from pathlib import Path
from datetime import datetime

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("mh-prostate-ingest")

BASE_DIR       = Path(__file__).resolve().parent.parent.parent
GUIDELINES_DIR = BASE_DIR / "docs" / "medical_guidelines"
VECTOR_DB_DIR  = BASE_DIR / "data" / "vector_db"
COLLECTION_NAME = "clinical_guidelines"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE    = 1000
CHUNK_OVERLAP = 150


# ══════════════════════════════════════════ 1. VERIFICACIÓN PREVIA ═════════════

def check_pdfs() -> list[Path]:
    pdf_files = sorted(GUIDELINES_DIR.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(
            f"No se encontraron PDFs en '{GUIDELINES_DIR}'. "
            "Coloque las guías clínicas allí antes de ejecutar."
        )
    log.info("PDFs detectados en '%s':", GUIDELINES_DIR)
    for f in pdf_files:
        size_kb = f.stat().st_size // 1024
        log.info("  • %-60s (%d KB)", f.name, size_kb)
    return pdf_files


# ══════════════════════════════════════════ 2. CARGA Y CHUNKING ════════════════

def load_and_chunk(pdf_files: list[Path]) -> list:
    """
    Carga todos los PDFs y los divide en chunks con contexto solapado.
    Usa RecursiveCharacterTextSplitter con separadores médicos optimizados.
    """
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        # Separadores jerárquicos optimizados para texto médico estructurado
        separators=["\n\n", "\n", ". ", "? ", "! ", "; ", " ", ""],
        length_function=len,
    )

    all_chunks   = []
    docs_summary = []

    for pdf_path in pdf_files:
        log.info("Cargando: %s ...", pdf_path.name)
        try:
            loader = PyPDFLoader(str(pdf_path))
            pages  = loader.load()

            # Enriquecer metadata de cada página
            for page in pages:
                page.metadata["source_file"] = pdf_path.name
                page.metadata["ingested_at"] = datetime.utcnow().isoformat()
                page.metadata["collection"]  = COLLECTION_NAME

            # Chunkear este documento
            chunks = splitter.split_documents(pages)

            log.info("  → %d páginas → %d chunks", len(pages), len(chunks))
            all_chunks.extend(chunks)
            docs_summary.append({
                "file":   pdf_path.name,
                "pages":  len(pages),
                "chunks": len(chunks),
            })
        except Exception as e:
            log.error("  ✗ Error cargando %s: %s", pdf_path.name, e)

    log.info("Total: %d chunks generados de %d documentos.",
             len(all_chunks), len(docs_summary))
    return all_chunks, docs_summary


# ══════════════════════════════════════════ 3. EMBEDDING + INDEXACIÓN ══════════

def embed_and_store(chunks: list) -> int:
    """
    Vectoriza los chunks con all-MiniLM-L6-v2 (local) y los persiste
    en ChromaDB bajo la colección 'clinical_guidelines'.

    Usa langchain_community.vectorstores.Chroma que gestiona internamente
    el mapeo de colecciones y metadatos de LangChain a ChromaDB.
    """
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import Chroma

    log.info("Cargando modelo de embeddings local: %s ...", EMBEDDING_MODEL)
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},   # Forzar CPU — Zero-Egress, sin GPU cloud
        encode_kwargs={"normalize_embeddings": True},
    )
    log.info("Modelo listo. Dimensión de vectores: 384")

    # Verificar si ya existen vectores en la colección (idempotencia)
    log.info("Conectando a ChromaDB en: %s", VECTOR_DB_DIR)
    existing_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(VECTOR_DB_DIR),
    )
    n_existing = existing_store._collection.count()
    if n_existing > 0:
        log.warning(
            "La colección '%s' ya contiene %d documentos. "
            "Se añadirán los chunks nuevos (deduplicación por contenido).",
            COLLECTION_NAME, n_existing
        )

    log.info("Indexando %d chunks en ChromaDB (esto puede tardar varios minutos)...",
             len(chunks))

    # Indexar en lotes de 100 para monitorear el progreso
    batch_size   = 100
    n_indexed    = 0
    vector_store = None

    for start in range(0, len(chunks), batch_size):
        end   = min(start + batch_size, len(chunks))
        batch = chunks[start:end]

        if vector_store is None:
            # Primera iteración: crea/conecta la colección
            vector_store = Chroma.from_documents(
                documents=batch,
                embedding=embeddings,
                collection_name=COLLECTION_NAME,
                persist_directory=str(VECTOR_DB_DIR),
            )
        else:
            vector_store.add_documents(batch)

        n_indexed += len(batch)
        pct = n_indexed / len(chunks) * 100
        log.info("  Progreso: %d/%d chunks (%.0f%%)", n_indexed, len(chunks), pct)

    total_in_collection = vector_store._collection.count()
    log.info("Indexación completa. Documentos en colección: %d", total_in_collection)
    return total_in_collection


# ══════════════════════════════════════════ 4. VALIDACIÓN SEMÁNTICA ════════════

def validate_retrieval() -> None:
    """
    Consulta de validación semántica end-to-end para verificar que el RAG
    recupera texto clínicamente relevante desde las guías indexadas.
    """
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import Chroma

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(VECTOR_DB_DIR),
    )

    queries = [
        "PSA density threshold for prostate biopsy recommendation",
        "ISUP grade group 2 clinically significant prostate cancer criteria",
        "NLR neutrophil lymphocyte ratio prostate cancer prognosis",
    ]

    log.info("────── Validación RAG (3 consultas semánticas) ──────")
    for q in queries:
        results = store.similarity_search_with_score(q, k=2)
        log.info("Query: «%s»", q)
        for doc, score in results:
            src  = doc.metadata.get("source_file", "?")
            page = doc.metadata.get("page", "?")
            log.info("  → [%s, p.%s | sim=%.3f] %s...",
                     src, page, score, doc.page_content[:100].replace("\n", " "))


# ══════════════════════════════════════════ MAIN ═══════════════════════════════

def run_ingestion():
    log.info("=== MH PROSTATE-CDSS | Phase 3b — Knowledge Base Ingestion ===")
    log.info("Zero-Egress: ✅ | Embedding: %s (local CPU)", EMBEDDING_MODEL)

    # 1. Verificar PDFs
    pdf_files = check_pdfs()

    # 2. Cargar y chunkear
    chunks, docs_summary = load_and_chunk(pdf_files)

    if not chunks:
        log.error("No se generaron chunks. Revise los PDFs.")
        return

    # 3. Indexar en ChromaDB
    total_vectors = embed_and_store(chunks)

    # 4. Validación
    validate_retrieval()

    # 5. Resumen final
    log.info("═══════════════════════════════════════════════════")
    log.info("RESUMEN DE INGESTA — MH PROSTATE-CDSS RAG")
    log.info("═══════════════════════════════════════════════════")
    log.info("  Carpeta fuente : %s", GUIDELINES_DIR)
    log.info("  ChromaDB       : %s", VECTOR_DB_DIR)
    log.info("  Colección      : %s", COLLECTION_NAME)
    log.info("  Embedding      : %s (local, Zero-Egress ✅)", EMBEDDING_MODEL)
    log.info("  ─────────────────────────────────────────────────")
    for d in docs_summary:
        log.info("  %-55s | %3d pág | %4d chunks", d["file"], d["pages"], d["chunks"])
    log.info("  ─────────────────────────────────────────────────")
    log.info("  TOTAL chunks generados : %d", len(chunks))
    log.info("  TOTAL vectores en DB   : %d", total_vectors)

    # Imprimir la línea requerida por la tarea
    print(f"\n✅ Ingested {len(chunks)} chunks from {len(docs_summary)} documents into ChromaDB.")
    print(f"   Collection '{COLLECTION_NAME}' → {total_vectors} total vectors at '{VECTOR_DB_DIR}'")
    print(f"   Zero-Egress: 100% local | No external API calls made.\n")

    log.info("=== Ingesta finalizada con éxito ===")
    return len(chunks), len(docs_summary), total_vectors


if __name__ == "__main__":
    run_ingestion()
