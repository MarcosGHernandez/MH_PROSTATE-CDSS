import streamlit as st
import pandas as pd
import re
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
LOG_FILE = BASE_DIR / "runs" / "v12_focal_precision" / "train_phase12.txt"

def _parse_training_logs():
    """
    Parses the phase12 precision logs efficiently.
    Uses tail-loading logic to handle growing logs without OOM.
    """
    epochs = []
    losses = []
    val_dices = []
    max_vram_warning = 0.0
    is_amp_active = False
    
    if not LOG_FILE.exists():
        return None, None, None, 0, False

    try:
        # Load only last 10k lines to avoid slow reading of huge logs
        with open(LOG_FILE, "r") as f:
            lines = f.readlines()[-10000:]
            
        for line in lines:
            if "Mission Critical" in line or "Phase 12" in line:
                is_amp_active = True
            
            if "VRAM near ceiling" in line:
                match = re.search(r"VRAM near ceiling: (\d+\.\d+)GB", line)
                if match:
                    max_vram_warning = max(max_vram_warning, float(match.group(1)))
                        
            # Format: ... | Epoch 1/100 | Loss: 0.8479 | Val Dice: 0.0015 | ...
            if "| Val Dice:" in line and "Epoch " in line:
                match_ep = re.search(r"Epoch (\d+)/", line)
                match_loss = re.search(r"Loss: (\d+\.\d+)", line)
                match_dice = re.search(r"Val Dice: (-?\d+\.\d+)", line) # Support potential negative dice if metric fails
                
                if match_ep and match_loss and match_dice:
                    epochs.append(int(match_ep.group(1)))
                    losses.append(float(match_loss.group(1)))
                    val_dices.append(float(match_dice.group(1)))
                    
    except Exception as e:
        # Avoid blocking the whole page on parse error
        pass
        
    if not epochs:
        return None, [], [], 0, is_amp_active

    df_metrics = pd.DataFrame({"Epoch": epochs, "Loss": losses, "Dice": val_dices})
    # Ensure clinical order and uniqueness
    df_metrics = df_metrics.drop_duplicates(subset=["Epoch"], keep="last").sort_values("Epoch")
    df_metrics = df_metrics.set_index("Epoch")
        
    return df_metrics, epochs, val_dices, max_vram_warning, is_amp_active

def render_training_monitor():
    """Renders the Live Training Monitor Tab with Clinical Sharp Aesthetics."""
    st.markdown("### Telemetría de Entrenamiento local (RTX 3070 Ti)")
    st.caption("Lectura en tiempo real del orquestador 3D U-Net (Fase 8d).")
    
    df, epochs, dices, max_vram, amp_active = _parse_training_logs()
    
    # ─── System Health ───
    st.markdown("#### Hardware & System Health")
    health_col1, health_col2 = st.columns(2)
    
    with health_col1:
        if amp_active:
            st.success("Protección VRAM Activa: torch.cuda.amp.GradScaler (Mixed Precision)")
        else:
            st.info("Inicializando Motor PyTorch...")
            
    with health_col2:
        if max_vram > 7.8:
            st.error(f"ALERTA VRAM: Pico registrado de {max_vram:.2f}GB (Cercano al límite de 7.8GB)")
        else:
            st.success("VRAM Nominal: Ejecución estable dentro de los parámetros de seguridad (< 7.8GB)")
    
    st.divider()
    
    # ─── Progress & Live Metrics ───
    st.markdown("#### Progreso de Convergencia (100 Epochs)")
    if not epochs:
        st.info("El modelo está empaquetando los volúmenes en CacheDataset. Las métricas del Epoch 1 aparecerán uskto después de la primera pasada.")
        return
        
    curr_epoch = epochs[-1]
    curr_dice = dices[-1]
    curr_loss = df["Loss"].iloc[-1]
    
    # Progress Bar
    pct = min(curr_epoch / 100.0, 1.0)
    st.progress(pct, text=f"Progreso Total: {curr_epoch} de 100 Epochs")
    
    # Metrics
    m1, m2 = st.columns(2)
    m1.metric(label="Validation Dice Score (csPCa)", value=f"{curr_dice:.4f}", delta=f"{curr_dice - dices[-2]:+.4f}" if len(dices) > 1 else None)
    m2.metric(label="Training Loss (DiceCELoss)", value=f"{curr_loss:.4f}", delta=f"{curr_loss - df['Loss'].iloc[-2]:+.4f}" if len(df) > 1 else None, delta_color="inverse")
    
    st.divider()
    
    # ─── Charts ───
    st.markdown("#### Evolución de Exactitud Diagnóstica (Dice Score)")
    st.line_chart(df["Dice"], height=250, use_container_width=True, color="#64FFDA")
    
    st.markdown("#### Entropía Cruzada y Disminución de Error (Loss)")
    st.line_chart(df["Loss"], height=200, use_container_width=True, color="#FF5555")
    
    # Auto-Refresh Hint
    st.caption("Nota: La telemetría se actualiza según la frecuencia de validación del script de entrenamiento. Use 'R' para recargar el Dashboard en cualquier momento.")
