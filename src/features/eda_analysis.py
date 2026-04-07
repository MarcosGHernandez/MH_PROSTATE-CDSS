"""
Viko-Health | Exploratory Data Analysis (EDA)
==============================================
Autor: Senior AI Architect
Objetivo: Analizar el dataset normalizado master_normalized.csv para:
          A. Desbalance de clases (target_cspca)
          B. Validación de umbral clínico PSAd >= 0.15
          C. Correlación de biomarcadores (Spearman)
          D. Análisis PSA vs NLR
"""

import os
from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Directorios
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_PATH = BASE_DIR / "data" / "processed" / "master_normalized.csv"
EDA_OUT_DIR = BASE_DIR / "data" / "processed" / "eda"

def run_eda():
    print("=== Iniciando Exploratory Data Analysis (EDA) ===")
    EDA_OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    if not DATA_PATH.exists():
        print(f"Error: No se encontró el dataset en {DATA_PATH}")
        return
        
    df = pd.read_csv(DATA_PATH)
    print(f"Dataset cargado: {df.shape[0]} filas, {df.shape[1]} columnas\n")
    
    # 1. Resumen Estadístico Clínico
    print("--- Resumen Estadístico Clínico ---")
    stats_cols = ['age', 'psa', 'prostate_volume', 'psad', 'nlr', 'albumin']
    available_cols = [c for c in stats_cols if c in df.columns]
    print(df[available_cols].describe().round(3))
    print("\n")
    
    # Configurar estilo de gráficos
    sns.set_theme(style="whitegrid")
    
    # A. Visualización del Target (Desbalance)
    plt.figure(figsize=(8, 5))
    ax = sns.countplot(x='target_cspca', data=df, palette='viridis')
    plt.title('Distribución de Cáncer Clínicamente Significativo (csPCa)')
    plt.xlabel('Cáncer Significativo (0 = No, 1 = Sí)')
    plt.ylabel('Cantidad de Pacientes')
    
    # Calcular porcentajes
    total = len(df)
    for p in ax.patches:
        height = p.get_height()
        ax.text(p.get_x() + p.get_width()/2., height + 3,
                f'{height} ({height/total:.1%})', ha="center")
                
    img_path = EDA_OUT_DIR / "A_class_imbalance.png"
    plt.savefig(img_path, dpi=300, bbox_inches='tight')
    print(f"Guardado: {img_path}")
    plt.close()
    
    # B. Validación del umbral PSAd >= 0.15 
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='target_cspca', y='psad', data=df, palette='Set2')
    plt.axhline(0.15, ls='--', color='red', label='Umbral Clínico (0.15)', linewidth=2)
    plt.yscale('log') # PSA suele ser logarítmico
    plt.title('Densidad de PSA (PSAd) vs Resultado Clínico (csPCa)')
    plt.xlabel('Cáncer Significativo (0 = No, 1 = Sí)')
    plt.ylabel('PSAd (ng/mL/cm³) - Escala Logarítmica')
    plt.legend()
    
    img_path = EDA_OUT_DIR / "B_psad_validation.png"
    plt.savefig(img_path, dpi=300, bbox_inches='tight')
    print(f"Guardado: {img_path}")
    plt.close()
    
    # C. Correlación de variables críticas (Spearman)
    plt.figure(figsize=(10, 8))
    corr_cols = ['age', 'psa', 'psad', 'nlr', 'albumin', 'prostate_volume', 'target_cspca']
    corr_available = [c for c in corr_cols if c in df.columns]
    
    corr = df[corr_available].corr(method='spearman')
    sns.heatmap(corr, annot=True, cmap='RdBu_r', center=0, fmt='.2f', 
                square=True, linewidths=.5, cbar_kws={"shrink": .8})
    plt.title('Matriz de Correlación de Biomarcadores (Spearman)')
    
    img_path = EDA_OUT_DIR / "C_correlation_heatmap.png"
    plt.savefig(img_path, dpi=300, bbox_inches='tight')
    print(f"Guardado: {img_path}")
    plt.close()
    
    # D. Análisis de Marcadores Inflamatorios (PSA vs NLR)
    plt.figure(figsize=(10, 6))
    
    # Filtramos outliers extremos de NLR para una mejor visualización si es necesario
    # Usaremos el logaritmo de ambos ejes para manejar el sesgo
    sns.scatterplot(x='psa', y='nlr', hue='target_cspca', data=df, 
                    alpha=0.6, palette='coolwarm', edgecolor=None)
    plt.xscale('log')
    plt.yscale('log')
    plt.title('Relación Inflamatoria: PSA vs NLR (Escala Logarítmica)')
    plt.xlabel('PSA (ng/mL)')
    plt.ylabel('NLR (Ratio Neutrófilo-Linfocito)')
    plt.legend(title='csPCa')
    
    img_path = EDA_OUT_DIR / "D_psa_vs_nlr.png"
    plt.savefig(img_path, dpi=300, bbox_inches='tight')
    print(f"Guardado: {img_path}")
    plt.close()
    
    print("\n=== EDA finalizado exitosamente ===")
    print(f"Tasa de positividad (csPCa = 1): {df['target_cspca'].mean():.2%}")
    print("Gráficos generados en data/processed/eda/")

if __name__ == "__main__":
    run_eda()
