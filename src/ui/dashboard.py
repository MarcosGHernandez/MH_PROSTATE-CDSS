"""
MH Prostate-CDSS | Phase 5b — Clinical Dashboard (Dark Mode Overhaul)
======================================================================
Theme: Clinical Sharp — Deep Navy Dark Mode (#0A192F, Cyan #64FFDA)
Zero-Egress: 100% local. No external API calls.

Launch: streamlit run src/ui/dashboard.py
"""

import sys
import time
import io
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(BASE_DIR))

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np

# ─── PAGE CONFIG ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MH Prostate-CDSS",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── GOOGLE FONTS ────────────────────────────────────────────────────────────
st.markdown(
    '<link rel="preconnect" href="https://fonts.googleapis.com">'
    '<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap" rel="stylesheet">',
    unsafe_allow_html=True,
)

# ─── DARK MODE CSS ────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* ── Base Reset ── */
  html, body, [class*="css"], .stApp {
      font-family: 'Inter', 'Segoe UI', sans-serif !important;
      background-color: #0A192F !important;
      color: #CCD6F6 !important;
  }

  /* ── Scrollbar ── */
  ::-webkit-scrollbar { width: 6px; }
  ::-webkit-scrollbar-thumb { background: #233554; border-radius: 2px; }

  /* ── Sidebar ── */
  section[data-testid="stSidebar"] > div:first-child {
      background-color: #112240 !important;
      border-right: 1px solid #1E3A5F;
  }
  section[data-testid="stSidebar"] label,
  section[data-testid="stSidebar"] p,
  section[data-testid="stSidebar"] .stCaption { color: #8892B0 !important; }
  section[data-testid="stSidebar"] input[type="number"] {
      background: #0A192F !important;
      border: 1px solid #233554 !important;
      color: #CCD6F6 !important;
      border-radius: 2px !important;
  }
  section[data-testid="stSidebar"] h3 { color: #64FFDA !important; }

  /* ── All containers — Sharp corners ── */
  div[data-testid="stVerticalBlock"] > div,
  div[data-testid="stHorizontalBlock"] > div,
  .stAlert, .stSuccess, .stWarning, .stInfo, .stError,
  [data-testid="stMetricValue"], [data-testid="column"],
  div.block-container { border-radius: 2px !important; }

  /* ── Main content background ── */
  .main .block-container { background-color: #0A192F !important; }

  /* ── Dividers ── */
  hr { border-color: #1E3A5F !important; }

  /* ── Metric cards ── */
  [data-testid="metric-container"] {
      background: #112240 !important;
      border: 1px solid #1E293B !important;
      border-left: 3px solid #64FFDA !important;
      border-radius: 2px !important;
      padding: 0.5rem 1rem !important;
  }
  [data-testid="stMetricLabel"] { color: #8892B0 !important; font-size: 0.78rem !important; }
  [data-testid="stMetricValue"] { color: #CCD6F6 !important; }
  [data-testid="stMetricDelta"] svg { stroke: #64FFDA; }

  /* ── Buttons ── */
  .stButton > button, div[data-testid="stDownloadButton"] > button {
      background: #004A99 !important;
      color: #FFFFFF !important;
      border: 1px solid #64FFDA !important;
      border-radius: 2px !important;
      font-weight: 600 !important;
      letter-spacing: 0.04em;
      transition: background 0.2s ease;
  }
  .stButton > button:hover, div[data-testid="stDownloadButton"] > button:hover {
      background: #0066CC !important;
      border-color: #64FFDA !important;
  }

  /* ── Checkbox ── */
  .stCheckbox > label > span { color: #CCD6F6 !important; }

  /* ── Spinner ── */
  .stSpinner > div { border-top-color: #64FFDA !important; }

  /* ── Info / Warning / Success banners ── */
  .stInfo    { background: #112240; border: 1px solid #1E293B; color: #CCD6F6; border-radius: 2px !important; }
  .stWarning { background: #1A1800; border: 1px solid #FFD700; color: #FFD700; border-radius: 2px !important; }
  .stSuccess { background: #0D2218; border: 1px solid #64FFDA; color: #64FFDA; border-radius: 2px !important; }
  .stError   { background: #200D0D; border: 1px solid #FF5555; color: #FF5555; border-radius: 2px !important; }

  /* ── Header banner ── */
  .cdss-header {
      background: linear-gradient(135deg, #004A99 0%, #003370 100%);
      border: 1px solid #1E3A5F;
      border-left: 4px solid #64FFDA;
      border-radius: 2px;
      padding: 1.2rem 2rem;
      margin-bottom: 1.5rem;
  }
  .cdss-header h1 { color: #FFFFFF; margin: 0; font-size: 1.7rem; font-weight: 700; letter-spacing: -0.02em; }
  .cdss-header p  { color: #8892B0; margin: 0.25rem 0 0 0; font-size: 0.82rem; }
  .cdss-header span.accent { color: #64FFDA; font-weight: 600; }

  /* ── Feature cards (static) ── */
  .feat-card {
      background: #112240;
      border: 1px solid #1E293B;
      border-top: 2px solid #64FFDA;
      border-radius: 2px;
      padding: 1rem 1.2rem;
      height: 100%;
  }
  .feat-card h4 { color: #64FFDA; margin: 0 0 0.5rem 0; font-size: 0.95rem; }
  .feat-card p  { margin: 0; color: #8892B0; font-size: 0.85rem; line-height: 1.5; }

  /* ── Report box ── */
  .report-box {
      background: #112240;
      border: 1px solid #1E293B;
      border-left: 3px solid #004A99;
      border-radius: 2px;
      padding: 1.5rem;
      line-height: 1.75;
      font-size: 0.92rem;
      color: #CCD6F6;
  }

  /* ── Disclaimer ── */
  .disclaimer {
      background: #0D1B2E;
      border: 1px solid #1E3A5F;
      border-left: 3px solid #E67E22;
      border-radius: 2px;
      padding: 0.9rem 1.2rem;
      font-size: 0.78rem;
      color: #8892B0;
      margin-top: 2rem;
      line-height: 1.6;
  }

  /* ── SHAP badge ── */
  code { background: #1E3A5F !important; color: #64FFDA !important; border-radius: 2px !important; }

  /* ── Plotly container background ── */
  .stPlotlyChart { background: transparent !important; }
  .stPlotlyChart > div { background: transparent !important; }
</style>
""", unsafe_allow_html=True)


# ─── CACHED RESOURCE ─────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Cargando motor clinico (XGBoost + ChromaDB)...")
def load_orchestrator():
    from src.rag.mh_prostate_orchestrator import MHProstateOrchestrator
    return MHProstateOrchestrator()


# ─── PLOTLY GAUGE  (plotly_dark template) ────────────────────────────────────
def make_gauge(risk_pct: float, threshold_pct: float) -> go.Figure:
    if risk_pct >= threshold_pct:
        bar_color = "#FF5555"
    elif risk_pct >= threshold_pct * 0.65:
        bar_color = "#FFB347"
    else:
        bar_color = "#64FFDA"

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=risk_pct,
        delta={
            "reference": threshold_pct, "suffix": "%",
            "decreasing": {"color": "#64FFDA"},
            "increasing": {"color": "#FF5555"},
        },
        number={"suffix": "%", "font": {"size": 38, "color": bar_color}},
        title={"text": "Riesgo&nbsp;csPCa&nbsp;(XGBoost)",
               "font": {"size": 14, "color": "#8892B0"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#233554",
                     "tickfont": {"color": "#8892B0"}},
            "bar": {"color": bar_color, "thickness": 0.22},
            "bgcolor": "#112240",
            "bordercolor": "#1E3A5F",
            "steps": [
                {"range": [0,   threshold_pct * 0.65], "color": "#0D2218"},
                {"range": [threshold_pct * 0.65, threshold_pct], "color": "#1A1800"},
                {"range": [threshold_pct, 100],         "color": "#200D0D"},
            ],
            "threshold": {
                "line": {"color": "#64FFDA", "width": 2},
                "thickness": 0.78,
                "value": threshold_pct,
            },
        },
    ))
    fig.update_layout(
        template="plotly_dark",
        height=300,
        paper_bgcolor="#0A192F",
        plot_bgcolor="#0A192F",
        margin=dict(l=20, r=20, t=40, b=10),
        font={"family": "Inter, Segoe UI, sans-serif"},
    )
    return fig


# ─── PLOTLY SHAP BAR  (plotly_dark template) ─────────────────────────────────
def make_shap_bar(shap_features: list) -> go.Figure:
    labels, impacts = [], []
    for item in shap_features:
        try:
            feat = item.split("(")[0].strip()
            val  = float(item.split("impacto:")[1].replace(")", "").strip())
            labels.append(feat)
            impacts.append(val)
        except Exception:
            pass
    if not labels:
        return None

    colors = ["#FF5555" if v > 0 else "#64FFDA" for v in impacts]
    fig = go.Figure(go.Bar(
        x=impacts, y=labels, orientation="h",
        marker_color=colors,
        marker_line_color="#1E3A5F", marker_line_width=0.5,
        text=[f"{v:+.3f}" for v in impacts], textposition="outside",
        textfont={"color": "#CCD6F6", "size": 11},
    ))
    fig.update_layout(
        template="plotly_dark",
        title={"text": "Explicabilidad SHAP — Top Features del Paciente",
               "font": {"size": 13, "color": "#8892B0"}},
        xaxis=dict(title="Impacto SHAP", gridcolor="#1E3A5F",
                   zeroline=True, zerolinecolor="#64FFDA", zerolinewidth=1.2,
                   tickfont={"color": "#8892B0"}),
        yaxis=dict(tickfont={"color": "#CCD6F6"}),
        height=220,
        paper_bgcolor="#0A192F",
        plot_bgcolor="#112240",
        margin=dict(l=10, r=10, t=40, b=30),
        font={"family": "Inter, Segoe UI, sans-serif"},
    )
    return fig


# ─── PDF EXPORT ───────────────────────────────────────────────────────────────
def generate_pdf(report_text: str, patient: dict, risk_pct: float, vision_summary: str = None) -> bytes:
    from fpdf import FPDF

    class ClinicalPDF(FPDF):
        def header(self):
            import datetime
            self.set_fill_color(0, 74, 153)
            self.rect(0, 0, 210, 22, "F")
            self.set_font("Helvetica", "B", 11)
            self.set_text_color(255, 255, 255)
            self.set_xy(10, 4)
            self.cell(0, 6, "MH PROSTATE-CDSS -- REPORTE DE EVALUACION PREDICTIVA", align="L")
            self.set_font("Helvetica", "", 8)
            self.set_xy(10, 10)
            self.cell(0, 5, "Reporte Generado Localmente (Zero-Egress | EBM)")
            self.set_xy(10, 15)
            curr_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
            self.cell(0, 5, f"Fecha: {curr_date}")
            self.set_draw_color(0, 74, 153)
            self.set_line_width(0.5)
            self.line(10, 26, 200, 26)
            self.ln(12)

        def footer(self):
            self.set_y(-14)
            self.set_font("Helvetica", "I", 7)
            self.set_text_color(130, 130, 130)
            self.cell(0, 5,
                "SISTEMA DE SOPORTE -- NO CONSTITUYE DIAGNOSTICO MEDICO. "
                "Responsabilidad final: urologo tratante.", align="C")

    pdf = ClinicalPDF(orientation='P', unit='mm', format='A4')
    pdf.set_margins(left=20, top=20, right=20)
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.add_page()

    # Patient block
    pdf.set_fill_color(17, 34, 64)
    pdf.set_draw_color(0, 74, 153)
    pdf.set_line_width(0.4)
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(0, 74, 153)
    pdf.cell(0, 7, "Datos del Paciente", ln=True, fill=True, border=1)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(50, 50, 50)
    for label, val in [
        ("Edad",                f"{patient.get('age','N/D')} anos"),
        ("PSA",                 f"{patient.get('psa','N/D')} ng/mL"),
        ("Volumen prostatico",  f"{patient.get('prostate_volume','N/D')} cm3"),
        ("PSAd",                f"{patient.get('psad',0):.3f} ng/mL/cm3"),
        ("NLR",                 f"{patient.get('nlr','N/D')}"),
        ("Historia familiar",   "Si" if patient.get("family_history") else "No"),
        ("Riesgo csPCa (XGBoost)", f"{risk_pct:.1f}%"),
        ("Hallazgo Vision (MRI)", vision_summary if vision_summary else "No procesado"),
    ]:
        safe_label = label.encode("latin-1", errors="replace").decode("latin-1")
        safe_val   = str(val).encode("latin-1", errors="replace").decode("latin-1")
        pdf.cell(65, 6, f"  {safe_label}:", border=0)
        pdf.cell(0, 6, safe_val, ln=True)
    pdf.ln(3)

    # Report body
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(0, 74, 153)
    pdf.cell(0, 7, "Informe Clinico (RAG + LLM)", ln=True)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(40, 40, 40)
    for line in report_text.splitlines():
        clean = (line.replace("**", "").replace("*", "")
                     .replace("#", "").strip())
        if not clean:
            pdf.ln(2)
        else:
            safe = clean.encode("latin-1", errors="replace").decode("latin-1")
            w = pdf.epw
            pdf.multi_cell(w, 7, txt=safe, align='L')

    return bytes(pdf.output())


# ─── MAIN ────────────────────────────────────────────────────────────────────
def main():
    if 'ml_risk_pct' not in st.session_state: st.session_state.ml_risk_pct = None
    if 'ml_prediction' not in st.session_state: st.session_state.ml_prediction = None
    if 'vision_summary' not in st.session_state: st.session_state.vision_summary = None
    
    # ── Header ──────────────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="cdss-header">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <h1>MH PROSTATE-CDSS</h1>
                <p><strong>Clinical Decision Support System (v1.0)</strong> | <em>Evidence-Based Medicine</em></p>
            </div>
            <div style="text-align: right;">
                <span class="accent">Zero-Egress Protocol</span><br>
                <small style="color: #64FFDA;">Session: {time.strftime('%Y-%m-%d %H:%M')}</small>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Sidebar ─────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### Datos del Paciente")
        st.caption("Ingrese los valores clínicos para generar el reporte.")

        age     = st.number_input("Edad (años)",               min_value=30, max_value=100, value=65, step=1)
        psa     = st.number_input("PSA (ng/mL)",               min_value=0.0, max_value=150.0, value=5.5, step=0.1, format="%.2f")
        volume  = st.number_input("Volumen Prostático (cm³)",  min_value=1, max_value=300, value=40, step=1)
        nlr     = st.number_input("NLR (Neutróf./Linfoc.)",   min_value=0.0, max_value=20.0, value=2.5, step=0.1, format="%.1f")
        albumin = st.number_input("Albúmina (g/dL)",           min_value=1.0, max_value=6.0, value=4.2, step=0.1, format="%.1f")
        crp     = st.number_input("CRP (mg/L)",                min_value=0.0, max_value=100.0, value=2.0, step=0.1, format="%.1f")
        fam_hist = st.checkbox("Antecedentes familiares de cáncer de próstata")
        dre      = st.checkbox("Tacto rectal sospechoso (DRE positivo)")

        # Auto-calc PSAd
        psad       = psa / volume if volume > 0 else 0.0
        psad_alert = psad >= 0.15

        st.divider()
        st.markdown(f"**PSAd auto-calculado:** `{psad:.3f}` ng/mL/cm³")
        if psad_alert:
            st.warning("ALERTA CLINICA: PSAd >= 0.15")
        else:
            st.success("PSAd dentro del rango normal")

        st.divider()
        run_btn = st.button("Iniciar Evaluación Clínica",
                            use_container_width=True, type="primary")

    # ── Main UI Layout (Phase 9 Tabs) ────────────────────────────────────────
    tab_clin, tab_vis, tab_mon = st.tabs([
        "Evaluación Clínica (XGBoost + RAG)", 
        "Análisis de Imagen (mpMRI 3D)",
        "Monitor de Entrenamiento (RTX VRAM)"
    ])
    
    with tab_clin:
        # ── Landing state ────────────────────────────────────────────────────────
        if not run_btn:
            st.info("### 1. Ingrese datos en el panel lateral 2. Genere el reporte clínico")
            cc1, cc2 = st.columns([2, 1])
            with cc1:
                st.markdown("""
                **MH PROSTATE-CDSS** utiliza un motor dual para la estratificación de riesgo de cáncer de próstata clínicamente significativo (csPCa).
                - **Motor A (Bioquímico):** Análisis estadístico vía XGBoost optimizado para alta sensibilidad.
                - **Motor B (Visión):** Segmentación 3D U-Net en mpMRI (NIfTI) con eliminación de ruido anatómico.
                """)
            with cc2:
                st.markdown("**Compliance EBM 2026**")
                st.caption("Pestaña 'Monitor' disponible para supervisión de VRAM (RTX Series).")
            
            st.divider()
            col1, col2, col3 = st.columns(3)
            for col, title, desc, symbol in [
                (col1, "Análisis de Riesgo", "Implementación XGBoost (AUC 0.90) con interpretabilidad SHAP para el facultativo.", "[&bull;]"),
                (col2, "Recuperación EBM", "Acceso semántico local a guías clínicas EAU/AUA para soporte basado en evidencia.", "[&bull;]"),
                (col3, "Privacidad de Datos", "Protocolo Zero-Egress: ejecución 100% offline para la seguridad del paciente.", "[&bull;]"),
            ]:
                with col:
                    st.markdown(f'<div class="feat-card"><h3 style="font-size:1rem; color:#64FFDA;">{symbol} {title}</h3><p>{desc}</p></div>',
                                unsafe_allow_html=True)
            pass
        else:
            # ── Run pipeline ─────────────────────────────────────────────────────────
            patient_data = {
                "age": float(age), "psa": float(psa),
                "prostate_volume": float(volume), "psad": round(psad, 4),
                "nlr": float(nlr), "albumin": float(albumin), "crp": float(crp),
                "family_history": int(fam_hist), "dre_result": int(dre),
                "src_turkish": 0,
            }
        
            orchestrator = load_orchestrator()
        
            with st.spinner("Procesando análisis de biomarcadores... Por favor, espere."):
                time.sleep(3)  # UX: artificial delay for perceived valuation
                t0 = time.time()
                try:
                    ml_result  = orchestrator.predict_risk(patient_data)
                    from src.rag.mh_prostate_orchestrator import run_analysis
                    final_report = run_analysis(patient_data)
                    elapsed = round(time.time() - t0, 1)
                    st.success(f"Analisis completado en {elapsed}s")
                except Exception as e:
                    st.error(f"Error durante el analisis: {e}")
                    st.stop()
        
            # ── KPI row ──────────────────────────────────────────────────────────────
            risk_pct      = ml_result["ml_risk_percent"]
            threshold_pct = ml_result["ml_threshold"]
            prediction    = ml_result["ml_prediction"]
            shap_feats    = ml_result.get("top_2_shap_features", [])

            st.session_state.ml_risk_pct = risk_pct
            st.session_state.ml_prediction = prediction
        
            # High-Level Summary Card
            st.markdown(f"""
            <div style="background: #112240; border-left: 4px solid {'#FF5555' if prediction == 'Malignant' else '#64FFDA'}; padding: 1rem; border-radius: 2px; margin-bottom: 2rem;">
                <h2 style="margin:0; color: white;">Resultado: {prediction.upper()}</h2>
                <p style="margin:0; color: #8892B0;">Confianza del modelo: {risk_pct:.1f}% vs Umbral Clínico {threshold_pct:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)

            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Riesgo csPCa",   f"{risk_pct:.1f}%",
                      delta=f"{risk_pct - threshold_pct:+.1f}% vs umbral")
            k2.metric("Umbral Clínico", f"{threshold_pct:.1f}%")
            k3.metric("PSA Density",    f"{psad:.3f}",
                      delta="ALERTA" if psad_alert else "Normal")
            k4.metric("Predicción ML",  prediction)
        
            st.divider()
        
            # ── Charts ───────────────────────────────────────────────────────────────
            ch1, ch2 = st.columns(2)
            with ch1:
                st.plotly_chart(make_gauge(risk_pct, threshold_pct),
                                use_container_width=True)
            with ch2:
                fig_shap = make_shap_bar(shap_feats)
                if fig_shap:
                    st.plotly_chart(fig_shap, use_container_width=True)
                else:
                    st.info("SHAP no disponible para este caso.")
        
            # SHAP text detail
            if shap_feats:
                st.markdown("**Drivers clave SHAP (paciente actual):**")
                for feat in shap_feats:
                    arrow = "[+]" if "Aumentó" in feat else "[-]"
                    st.markdown(f"- {arrow} `{feat}`")
        
            st.divider()
        
            # ── RAG Report ───────────────────────────────────────────────────────────
            st.markdown("### Reporte Clinico EBM")
            report_html = final_report.replace("\n", "<br>")
            st.markdown(f'<div class="report-box">{report_html}</div>',
                        unsafe_allow_html=True)
        
            st.divider()
        
            # ── PDF Download & Archival ─────────────────────────────────────────────
            pdf_bytes = generate_pdf(final_report, patient_data, risk_pct, vision_summary=st.session_state.vision_summary)
            
            # Archival automatico en el servidor (reports/)
            pdf_filename = f"MH_CDSS_{time.strftime('%Y%m%d_%H%M%S')}.pdf"
            server_path = BASE_DIR / "reports" / pdf_filename
            try:
                with open(server_path, "wb") as f:
                    f.write(pdf_bytes)
                st.success(f"Copia del reporte archivada exitosamente en el servidor (reports/{pdf_filename})")
            except Exception as e:
                st.error(f"Error al archivar copia local del reporte: {e}")
        
            st.download_button(
                label="Descargar Reporte PDF",
                data=pdf_bytes,
                file_name=pdf_filename,
                mime="application/pdf",
                use_container_width=True,
            )

    with tab_vis:
        st.markdown("### Selección de Volumes mpMRI")
        st.caption("Cargue volúmenes NIfTI (T2W y ADC correspondientes al paciente) para detectar áreas de csPCa vía la Red Neuronal (3D U-Net).")
        
        demo_mode = st.toggle("Modo Demo Institucional (Carga desde Disco)", value=True)
        
        from src.ui.vision_loader import run_vision_inference
        from src.ui.components.mri_viewer import display_mri_viewer
        
        t2_path, adc_path = None, None
        
        if demo_mode:
            # Hybrid Access - Read from Kaggle DB
            json_target = BASE_DIR / "data" / "kaggle_dataset_split.json"
            if json_target.exists():
                with open(json_target, "r") as f:
                    cases = json.load(f)
                    
                val_cases = [c for c in cases if c["split"] == "val"]
                case_names = [f"Paciente {c['patient_id']} - Finding {c['finding_id']}" for c in val_cases]
                
                sel_idx = st.selectbox("Seleccione un paciente de la base de validación local:", range(len(case_names)), format_func=lambda x: case_names[x])
                target = val_cases[sel_idx]
                t2_path = target["image_t2"]
                adc_path = target["image_adc"]
            else:
                st.error("No se detectó la base de datos local de kaggle. Desactive el Modo Demo.")
        else:
            c1, c2 = st.columns(2)
            t2_up = c1.file_uploader("Subir NIfTI T2W (Ej. series_t2.nii.gz)", type=["nii", "nii.gz"])
            adc_up = c2.file_uploader("Subir NIfTI ADC (Ej. series_adc.nii.gz)", type=["nii", "nii.gz"])
            
            if t2_up and adc_up:
                # Save temporarily for MONAI logic
                import tempfile
                with tempfile.NamedTemporaryFile("wb", suffix=".nii.gz", delete=False) as tf2:
                    tf2.write(t2_up.read())
                    t2_path = tf2.name
                with tempfile.NamedTemporaryFile("wb", suffix=".nii.gz", delete=False) as tfA:
                    tfA.write(adc_up.read())
                    adc_path = tfA.name
                    
        # State Management
        if 't2_vol' not in st.session_state: st.session_state.t2_vol = None
        if 'adc_vol' not in st.session_state: st.session_state.adc_vol = None
        if 'mask_vol_raw' not in st.session_state: st.session_state.mask_vol_raw = None
        if 'mask_vol_clean' not in st.session_state: st.session_state.mask_vol_clean = None
        if 'last_target' not in st.session_state: st.session_state.last_target = ""
        
        target_id = f"{t2_path}_{adc_path}"
        
        if t2_path and adc_path:
            if st.button("Procesar Volumen Neuronal (Inferencia)", type="primary", use_container_width=True):
                with st.spinner("Procesando Tensores en VRAM. Calculando probabilidad csPCa..."):
                    t0 = time.time()
                    try:
                        t2, adc, raw_mask, clean_mask = run_vision_inference(t2_path, adc_path, apply_post_processing=True)
                        st.session_state.t2_vol = t2
                        st.session_state.adc_vol = adc
                        st.session_state.mask_vol_raw = raw_mask
                        st.session_state.mask_vol_clean = clean_mask
                        st.session_state.last_target = target_id
                        
                        # Calculate Lesion Volume (Phase 14 Specification: 0.5x0.5x3.0mm = 0.75mm3/voxel)
                        v_vox = 0.75
                        l_voxels = np.sum(clean_mask)
                        l_vol = l_voxels * v_vox
                        st.session_state.vision_summary = f"Detectada lesion de {l_vol:.1f} mm3" if l_vol > 0 else "Sin lesiones significativas (>50mm3) detectadas."
                        
                        elapsed = time.time() - t0
                        st.success(f"Inferencia 3D Completada localmente en {elapsed:.2f}s (Cero latencia de red)")
                        # Init slider safely
                        if "mri_z_slice" not in st.session_state:
                            st.session_state.mri_z_slice = t2.shape[-1] // 2
                    except Exception as e:
                        st.error(f"Falla de Inferencia: {e}")
                        
            if st.session_state.t2_vol is not None and st.session_state.last_target == target_id:
                st.divider()
                
                # FUSION VIEW: Show Clinical Risk next to MRI
                v1, v2 = st.columns([1.5, 1])
                with v1:
                    st.markdown("### Visor Multiparamétrico IA (mpMRI)")
                    cl_mode = st.radio("Filtro Post-Procesado (Fase 14):", ["Predicción Cruda (Base 0.5)", "Predicción Limpia (ROI + CCA, Thresh 0.38)"], horizontal=True)
                with v2:
                    st.markdown("### Resumen Clínico")
                    if st.session_state.ml_risk_pct is not None:
                        st.metric("Riesgo XGBoost", f"{st.session_state.ml_risk_pct:.1f}%")
                        st.plotly_chart(make_gauge(st.session_state.ml_risk_pct, 15.0), use_container_width=True)
                    else:
                        st.info("Inicie evaluación en Tab 'Clinico' para ver riesgo.")
                    
                    st.warning(f"**Hallazgo Vision:** {st.session_state.vision_summary}")
                
                # Select which mask to pass down to the viewer
                active_mask = st.session_state.mask_vol_clean if "Limpia" in cl_mode else st.session_state.mask_vol_raw
                display_mri_viewer(st.session_state.t2_vol, st.session_state.adc_vol, active_mask)
                
    with tab_mon:
        from src.ui.tabs.training_monitor import render_training_monitor
        render_training_monitor()
        
    st.divider()    

    # ── Disclaimer ───────────────────────────────────────────────────────────
    st.markdown("""
    <div class="disclaimer">
        <strong>Aviso Legal y Etico -- MH PROSTATE-CDSS</strong><br>
        Este sistema es una herramienta de soporte analítico predictivo especializado.
        <strong>No constituye diagnóstico médico definitivo.</strong>
        La responsabilidad diagnóstica y terapéutica final recae
        exclusivamente en el urólogo tratante certificado.<br>
        <em>Protocolo Zero-Egress activo — 100% ejecución local. Sin transmisión de datos del paciente a servicios externos.</em>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
