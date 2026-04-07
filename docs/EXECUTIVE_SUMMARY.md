# MH PROSTATE-CDSS: Executive Summary / Resumen Ejecutivo

## 1. The Clinical Problem / El Problema Clínico
**[English]**  
Prostate cancer diagnostic pathways currently suffer from high false-positive referral rates driven by PSA variability and subjective MRI interpretations. Unnecessary invasive biopsies increase patient morbidity and inflate healthcare costs. The **MH PROSTATE-CDSS** provides a highly specialized, bias-free analytical layer to mathematically mitigate these clinical uncertainties.

**[Español]**  
El diagnóstico de cáncer de próstata sufre de altas tasas de falsos positivos en las derivaciones, impulsadas por la variabilidad del PSA y la interpretación subjetiva de RM. Las biopsias invasivas y no necesarias aumentan la morbilidad del paciente y los costos médicos. El **MH PROSTATE-CDSS** proporciona una capa analítica altamente especializada y libre de sesgos para mitigar matemáticamente estas incertidumbres clínicas.

---

## 2. The Solution: Dual-Engine CDSS / La Solución: CDSS de Motor Dual
**[English]**  
The **MH PROSTATE-CDSS** (Clinical Decision Support System) is an enterprise-grade platform built specifically for strict **Zero-Egress local execution**. It functions as a diagnostic funnel: first filtering patients through an explainable Machine Learning model analyzing metabolic and biochemical indicators, and subsequently validating physical anomalies using an advanced 3D U-Net capable of processing volumetric NIfTI images.

**[Español]**  
El **MH PROSTATE-CDSS** (Sistema de Soporte a la Decisión Clínica) es una plataforma de grado empresarial construida específicamente para **ejecución local Zero-Egress**. Funciona como un embudo diagnóstico: filtrando primero a los pacientes a través de un modelo explicable de Machine Learning que analiza indicadores metabólicos y bioquímicos, y posteriormente validando anomalías físicas utilizando una avanzada 3D U-Net capaz de procesar imágenes volumétricas NIfTI.

---

## 3. Data Integrity & Global Benchmarking / Integridad de Datos y Benchmarking Global
**[English]**  
To guarantee diagnostic alignment with Evidence-Based Medicine (EBM), the system's foundational models have been trained and validated against gold-standard international datasets:
- **PI-CAI (Radboud University):** High-fidelity multiparametric MRI (mpMRI) volumetric benchmarks.
- **ProstateX (TCIA):** Diagnostic classification datasets mapping clinical significance.
- **Kaggle Clinical Cohort:** Multi-institutional biochemical data enabling robust tabular analysis.

**[Español]**  
Para garantizar la alineación diagnóstica con la Medicina Basada en Evidencia (EBM), los modelos base del sistema han sido entrenados y validados frente a conjuntos de datos internacionales de referencia:
- **PI-CAI (Radboud University):** Benchmarks volumétricos de resonancia magnética multiparamétrica (mpMRI) de alta fidelidad.
- **ProstateX (TCIA):** Conjuntos de datos de clasificación diagnóstica que mapean el grado de significancia clínica.
- **Kaggle Clinical Cohort:** Datos bioquímicos multi-institucionales que permiten un análisis tabular robusto.

---

## 4. Core System Outcomes / Desempeño Principal del Sistema
**[English]**  
- **0.897 AUC-ROC:** Unparalleled predictive triage based exclusively on non-invasive biomarkers.
- **-30mm HD95 Reduction:** Precision anatomical localization ensuring AI attention remains bound strictly to the prostate gland via Cascaded ROI Masking (Phase 14 validation).
- **Offline Medical Security:** Complete adherence to HIPAA and European GDPR standards. The system does not possess internet ingress or egress paths.

**[Español]**  
- **0.897 AUC-ROC:** Triaje predictivo incomparable basado exclusivamente en biomarcadores no invasivos.
- **Reducción de -30mm en HD95:** Localización anatómica de precisión que garantiza que la red neuronal permanezca limitada a la próstata a través de enmascaramiento ROI en cascada (fase 14).
- **Seguridad Médica Offline:** Adhesión completa a los estándares HIPAA y GDPR europeo. El sistema no genera, ni requiere, conexiones de salida o entrada a internet.

---
**Technical Standard:** MH PROSTATE-CDSS Suite 1.0  
**Compliance Reference:** EBM 2026 | Zero-Egress Protocol  
**Project Lead:** Marcos Hernandez (*Universidad Tecnológica de la Mixteca / VIKOTECH Solutions*)
