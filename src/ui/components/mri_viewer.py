import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import io

def _norm(vol):
    """Normalizes a volume to 0-1 for visualization."""
    mx, mn = np.max(vol), np.min(vol)
    if mx == mn: return vol
    return (vol - mn) / (mx - mn)
    
def display_mri_viewer(t2_vol, adc_vol, mask_vol):
    """
    Renders an interactive 3D MRI viewer inside a Streamlit container.
    Provides slice navigation (Z-axis) and Alpha-blended mask overlays.
    
    Args:
        t2_vol (np.ndarray): T2W 3D array (X,Y,Z).
        adc_vol (np.ndarray): ADC 3D array (X,Y,Z).
        mask_vol (np.ndarray): Binary AI prediction mask (X,Y,Z).
    """
    if t2_vol is None or adc_vol is None or mask_vol is None:
        st.error("No se detectó un volumen válido para visualizar.")
        return
        
    # Validations to prevent out-of-index errors
    z_max = t2_vol.shape[-1]
    
    st.markdown("### Visor Multiparamétrico IA (mpMRI)")
    
    st.markdown("""
        <style>
        .mri-toolbox {
            background:#112240; padding:15px; border-radius:5px;
            border-left:4px solid #64FFDA; margin-bottom:10px;
        }
        </style>
        <div class="mri-toolbox">
           Navega a través del eje axial. La IA detectara automáticamente zonas de alta probabilidad (>0.5)
           de Cáncer de Próstata Clínicamente Significativo (csPCa).
        </div>
    """, unsafe_allow_html=True)
    
    # Toolbars
    scol1, scol2, scol3 = st.columns([2, 1, 1])
    
    with scol1:
        # Z slice selection
        slider_key = "mri_z_slice" # we use a session state key
        st.slider("Corte Axial (Z-Axis)", min_value=0, max_value=z_max-1, value=z_max // 2, key=slider_key)
        
    with scol2:
        view_type = st.radio("Modalidad:", ["T2W", "ADC"], horizontal=True)
        
    with scol3:
        show_mask = st.toggle("Mostrar IA", value=True)
        
    z_idx = st.session_state[slider_key]
    
    # Extract 2D Slice
    # Transposing to align properly in matplotlib, mapping standard RAS config
    t2_slice = _norm(t2_vol[:, :, z_idx].T)
    adc_slice = _norm(adc_vol[:, :, z_idx].T)
    mask_slice = mask_vol[:, :, z_idx].T
    
    # ── Render via Matplotlib ──
    # We use a 1-by-2 figure layout. Left: Raw modality, Right: AI Overlay
    fig, axes = plt.subplots(1, 2, figsize=(14, 7), facecolor='#0A192F')
    
    base_slice = t2_slice if view_type == "T2W" else adc_slice
    
    # Base View
    axes[0].imshow(base_slice, cmap='gray')
    axes[0].set_title(f"{view_type} Puro", color='#CCD6F6', pad=15, fontsize=14)
    axes[0].axis('off')
    
    # UI Overlay View
    axes[1].imshow(base_slice, cmap='gray')
    axes[1].set_title("Visión Artificial U-Net", color='#64FFDA', pad=15, fontsize=14)
    axes[1].axis('off')
    
    if show_mask and np.any(mask_slice):
        # Clinical Red (#FF5555 mapped to RGBA)
        red_rgba = np.array([1.0, 0.33, 0.33, 0.4]) # Alpha blending 0.4 directly via cmap or manually
        
        # Build masked layer
        overlay = np.zeros((*mask_slice.shape, 4)) 
        overlay[mask_slice == 1] = red_rgba
        
        axes[1].imshow(overlay)
        # Additionally draw high-contrast contour
        axes[1].contour(mask_slice, colors='#FF5555', linewidths=1.5, levels=[0.5])
    
    st.pyplot(fig)
    plt.close(fig)
