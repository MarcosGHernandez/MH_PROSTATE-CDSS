# MH PROSTATE-CDSS: Clinical Decision Support System for Prostate Cancer Detection

**Organization:** VIKOTECH Solutions  
**Academic Context:** Final Project for Computer Engineering, Universidad Tecnológica de la Mixteca (UTM)  

---

## 1. Executive Context & Value Proposition / Contexto Ejecutivo y Propuesta de Valor

### English
**MH PROSTATE-CDSS** is a high-fidelity Clinical Decision Support System (CDSS) specifically engineered to minimize false-positive biopsies in prostate cancer screening. By synthesizing both structured biochemical biomarkers and complex 3D volumetric images (mpMRI) into a unified diagnostic funnel, the system assists urologists in identifying clinically significant prostate cancer (csPCa). Built upon a strict Evidence-Based Medicine (EBM) framework, it offers absolute patient data sovereignty through a fully offline, Zero-Egress local execution architecture.

### Español
El **MH PROSTATE-CDSS** es un Sistema de Soporte a la Decisión Clínica de alta fidelidad, diseñado específicamente para minimizar las biopsias innecesarias en el tamizaje del cáncer de próstata. Al sintetizar biomarcadores bioquímicos estructurados con complejas imágenes volumétricas 3D (mpMRI) en un embudo diagnóstico unificado, el sistema asiste a los urólogos en la identificación de cáncer de próstata clínicamente significativo (csPCa). Construido bajo el estricto marco de la Medicina Basada en Evidencia (EBM), ofrece soberanía absoluta sobre los datos del paciente mediante una arquitectura local 100% offline (Zero-Egress).

---

## 2. Platform Architecture: The Dual-Engine Model / Arquitectura: Modelo de Motor Dual

The system is constructed with a cascaded funnel approach, transitioning from lightweight scalable tabular risk stratification towards computationally complex spatial analysis. / El sistema está construido con un enfoque de embudo en cascada, transicionando desde una estratificación de riesgo basada en tablas ligeras hacia un análisis espacial computacionalmente complejo.

### Motor A: Biochemical Stratification (XGBoost)
Provides a high-sensitivity initial risk assessment combining variables such as Prostate-Specific Antigen (PSA) density, Neutrophil-to-Lymphocyte Ratio (NLR), and C-Reactive Protein (CRP). The predictions are backed by SHAP (SHapley Additive exPlanations) values to guarantee fully explainable AI functionality for clinical professionals.

### Motor B: mpMRI Spatial Vision (3D U-Net)
Patients crossing the biochemical threshold progress to the Vision phase. A 3D U-Net specifically trained utilizing MONAI handles the volumetric 16-bit T2W and ADC imaging. To overcome the extreme 99.9% background-to-lesion imbalance, the network relies on `DiceFocalLoss`. Final spatial outputs pass through an advanced Cascaded ROI Masking filter (Phase 14), reducing anatomical Hausdorff Distance errors by over 30mm. 

### Motor C: Evidence-Based RAG Orchestration
To culminate the pipeline, context from localized medical guidelines (such as the European Association of Urology - EAU) is dynamically embedded via a ChromaDB Vector Space and processed by a local offline LLM (Llama 3.1 8B), auto-generating an interoperable clinical report directly matching the patient's individual profile.

---

## 3. Technology Stack & Hardware Optimization

The system ensures democratic clinical deployment by remaining functional on advanced consumer-grade hardware constraints.

- **Stack:** Python 3.13 | PyTorch 2.3 | MONAI 1.3 | XGBoost 2.0 | Streamlit 1.32. (See `docs/TECH_STACK.md` for full implementation logic).
- **Optimization Strategy:** Precision deep learning relies on Automatic Mixed Precision (AMP) and localized Gradient Accumulation logic, allowing stable convergence and spatial inference upon a baseline standard of an **NVIDIA RTX 3070 Ti (8GB VRAM)**.
- **Privacy Standard:** *Zero-Egress Protocol*. Institutional medical data is guaranteed to never leave the host system boundaries.

---

## 4. Repository Operations & File Mapping

The application directory separates logic seamlessly:

- `src/data_pipeline/`: Data cleaning, normalization, and SVMSMOTE class balancing modules.
- `src/rag/`: Knowledge ingestion scripts, vector database index management, and LLM orchestration.
- `src/vision/`: Core deep learning architecture, 3D transformations, and U-Net training protocols.
- `src/ui/`: Presentation layer containing the centralized Clinical Streamlit Dashboard (`dashboard.py`).
- `models/`: Destination storage for processed weights (`.pth`) and machine learning pipelines (`.pkl`).
- `docs/`: Expanded institutional documentation mapping execution constraints, development iterations, and data scaling protocols.

---

## 5. Dataset Citations & External Validation

The mathematical integrity of the system represents a convergence of multiple highly verified public research dataset arrays:
- **PI-CAI Challenge:** Multiparametric MRI training resources managed by Radboud University Medical Center.
- **ProstateX:** TCIA database subsets for spatial correlation in significance classification.
- **Turkish Diagnostic Cohort:** Multi-source biochemical metadata serving as the root parameter for the XGBoost engine.

---

## 6. Execution & System Start

To deploy the MH PROSTATE-CDSS securely and verify that network interfaces are strictly isolated offline, use the automated batch initialization directly in the root directory.

**Command (Windows):**
```cmd
.\run_mhprostate.bat
```
*(The protocol actively purges orphaned background processes under localhost port 8501 and guarantees correct Python scoping prior to UI rendering).*
