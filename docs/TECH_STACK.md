# MH PROSTATE-CDSS: Technology Stack & Validation

The technological infrastructure of the MH PROSTATE-CDSS integrates robust frameworks across Data Engineering, Computer Vision, and Generative Artificial Intelligence. This document outlines the technological stack, versioning, and the rationale behind each component. 

---

## 1. Core Data Science & Biochemical Engine
**Purpose:** Handle structured clinical data (tabular) and optimize the risk stratification model using interpretable machine learning.

*   **Pandas (2.2.0) & NumPy (1.26.4):** Data manipulation and matrix operations underlying all clinical arrays.
*   **Scikit-Learn (1.4.1) & Imbalanced-learn (0.12.0):** Data preprocessing, evaluation metrics, and resampling. `SVMSMOTE` was explicitly chosen to address class imbalance for malignant labels (k=3 clustering).
*   **XGBoost (2.0.3):** The primary inference engine for the biochemical risk stratification. Chosen for its unparalleled performance on tabular data and capability to handle missing values natively.
*   **SHAP (0.45.0):** Shapley Additive Explanations provide mathematical interpretability, transforming black-box XGBoost predictions into granular feature-importance graphs for clinical transparency.

## 2. Deep Learning & Spatial Vision Engine (3D U-Net)
**Purpose:** Accurately segment significant prostate cancer (csPCa) lesions in volumetric mpMRI images (NIfTI format).

*   **PyTorch (2.3.0):** The foundational deep learning framework, enabling dynamic computational graphs and Automatic Mixed Precision (AMP) for RTX VRAM optimization.
*   **MONAI (1.3.0):** (Medical Open Network for AI). The backbone of the spatial vision pipeline. 
    *   **Rationale:** Standard frameworks struggle with 16-bit 3D spatial contexts. MONAI provides specialized medical image transformations (e.g., `RandCropByPosNegLabeld`, `SlidingWindowInferer`) and the `DiceFocalLoss` criteria necessary to solve the severe 99.9% background-to-lesion voxel imbalance.

## 3. RAG Architecture & Offline LLM (Zero-Egress)
**Purpose:** Generate Evidence-Based Medicine (EBM) clinical reports via semantic grounding without exposing patient data to external networks.

*   **LangChain (0.1.13):** Orchestrates the Retrieval-Augmented Generation (RAG) pipeline connecting ChromaDB with the local LLM.
*   **ChromaDB (0.4.24):** High-speed local Vector Database utilizing Hierarchical Navigable Small World (HNSW) indexing to embed and retrieve medical guidelines (EAU/AUA localized index).
*   **Sentence-Transformers (2.5.1):** Generates high-density vector embeddings (`all-MiniLM-L6-v2`) for semantic search of the clinical database.
*   **Ollama (0.1.7):** Serves the *meta-llama/Meta-Llama-3.1-8B-Instruct* model entirely on localhost (Zero-Egress). Selected for its powerful edge-compute reasoning capabilities.

## 4. UI/UX & Interoperability
**Purpose:** Deploy the dual-engine pipeline into a unified, secure, and user-friendly interface.

*   **Streamlit (1.32.2):** High-performance framework used for the clinical dashboard (`src/ui/dashboard.py`). Optimized with `@st.cache_resource` to allow the 3D U-Net to persist in VRAM across sessions.
*   **Plotly (5.20.0):** Generates interactive gauge and bar charts for risk stratification visualization.
*   **FPDF2 (2.7.7):** High-speed generation of physical evidence reports directly into secure PDF format.
*   **FHIR.resources (7.1.0):** Provides rigorous standardisation frameworks aligning the extracted data structures with Fast Healthcare Interoperability Resources guidelines.
