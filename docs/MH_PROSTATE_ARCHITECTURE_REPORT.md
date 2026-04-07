# MH PROSTATE-CDSS: Clinical Decision Support System Architecture Report (Institutional Version 2026)

## 1. System Overview / Resumen del Sistema
**[English]**  
Integrated AI diagnostic platform for non-invasive detection of clinically significant prostate cancer (csPCa), utilizing a dual-engine architectural strategy for maximized evidentiary value.

**[Español]**  
Plataforma de diagnóstico de IA integrada para la detección no invasiva del cáncer de próstata clínicamente significativo (csPCa), utilizando una estrategia arquitectónica de doble motor para maximizar el valor de la evidencia.

### 1.1 Biomarker Prediction Engine (BPE) / Motor de Predicción de Biomarcadores
- **English:** XGBoost-based risk assessment with SHAP interpretability for clinical biomarkers.
- **Español:** Evaluación de riesgo basada en XGBoost con interpretabilidad SHAP para biomarcadores clínicos.

### 1.2 Vision Analytics Engine (VAE) / Motor de Análisis de Visión
- **English:** 3D U-Net (32-256 blocks) with MONAI framework, DiceFocalLoss, and deep spatial analytics.
- **Español:** U-Net 3D (bloques 32-256) con framework MONAI, DiceFocalLoss y analítica espacial profunda.

---

## 2. The Clinical Trade-Off / El Compromiso Clinico
**[English]**  
Prioritization of high anatomical specificity through ROI Masking and CCA filtering. The reduction of HD95 from 92mm to 62mm (-30mm error) ensures diagnostic reliability.

**[Español]**  
Priorización de una alta especificidad anatómica mediante enmascaramiento ROI y filtrado CCA. La reducción de HD95 de 92mm a 62mm (-30mm de error) asegura confiabilidad diagnóstica.

---

## 3. Hardware Engineering / Ingeniería de Hardware
**[English]**  
Optimization for consistent 3D inference using Mixed Precision (AMP), Gradient Accumulation (Steps=8), and Sliding Window Inference (128x128x32).

**[Español]**  
Optimización para inferencia 3D consistente mediante Precisión Mixta (AMP), Acumulación de Gradientes (Pasos=8) e Inferencia por Ventana Deslizante (128x128x32).

---
**Standard Compliance:** Zero-Egress | EBM 2026
**Project Lead:** Marcos Hernandez
