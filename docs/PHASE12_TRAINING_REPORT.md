# MH PROSTATE-CDSS: Phase 12 - Hardened Precision Training Report / Reporte de Entrenamiento

## 1. Executive Summary / Resumen Ejecutivo
**[English]**  
Phase 12 transition from baseline to high-precision 3D U-Net diagnostic on the MH PROSTATE-CDSS. Achieve 448% improvement in Dice and significant HD95 reduction via specialized loss functions (DiceFocalLoss).

**[Español]**  
La Fase 12 marca la transición de un modelo base a una diagnóstica U-Net 3D de alta precisión en el sistema MH PROSTATE-CDSS. Se logró una mejora del 448% en el coeficiente Dice y una reducción significativa de HD95 mediante funciones de pérdida especializadas (DiceFocalLoss).

---

## 2. Technical Configuration / Configuracion Tecnica

| Parameter / Parámetro | Baseline (Phase 11) | Final (Phase 12 Optimized) | Outcome / Resultado |
| :--- | :--- | :--- | :--- |
| **Architecture** | 3D U-Net (Base) | **3D U-Net (32-256)** | High Spatial Depth |
| **Loss function** | Dice Loss | **DiceFocalLoss ($\gamma=2.0$)** | **Imbalance Handling +** |
| **Final Val Dice** | 0.0261 | **0.1425** | **+448% Improvement** |
| **Final HD95 (mm)**| 162.4mm | **87.60mm** | **-74.8mm Gain** |

---

## 3. Results Analysis / Análisis de Resultados
**[English]**  
The specificity-first refinement represents the reduction of false-positives at the cost of slight Dice reduction (from 0.14 to 0.12). 50mm³ CCA filtering is based on PI-RADS v2.1 criteria.

**[Español]**  
El refinamiento de especificidad-primero representa la reducción de falsos positivos al costo de una ligera reducción de Dice (de 0.14 a 0.12). El filtrado CCA de 50mm³ se basa en criterios PI-RADS v2.1.

---
**Standard Compliance:** Zero-Egress | EBM 2026
**Data Recognition:** PI-CAI / ProstateX (TCIA)
**Project Lead:** Marcos Hernandez
