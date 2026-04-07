import sys
import torch
import logging
import streamlit as st
import numpy as np
from pathlib import Path

import monai.transforms as mt
from monai.networks.nets import UNet
from monai.data import decollate_batch

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(BASE_DIR))

# Import transforms to ensure exact match with training
try:
    from src.vision import transforms as vision_transforms
except ImportError:
    pass

log = logging.getLogger("vision-loader")

@st.cache_resource(show_spinner="Inicializando Motor mpMRI 3D en GPU...")
def load_vision_model():
    """
    Loads the trained PyTorch 3D U-Net into VRAM. Cached to prevent 
    re-initialization latency and OOM on tab switches.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = BASE_DIR / "models" / "vision" / "best_unet_prostate.pth"
    
    # Phase 12 hardened precision architecture
    model = UNet(
        spatial_dims=3,
        in_channels=2, # T2W + ADC
        out_channels=1, # Mask Binary Output (1 channel)
        channels=(32, 64, 128, 256),
        strides=(2, 2, 2),
        num_res_units=2,
        dropout=0.2,
    ).to(device)
    
    if model_path.exists():
        model.load_state_dict(torch.load(str(model_path), map_location=device))
        log.info(f"Modelo U-Net cargado exitosamente desde {model_path} en {device}")
    else:
        log.warning(f"No se encontró el modelo local en {model_path}. Operando con pesos aleatorios para DEMO de UI.")
        
    model.eval()
    return model, device

def build_inference_transforms():
    """
    Builds the exact MONAI validation transforms to ensure consistency.
    Target Spacing: 0.5mm x 0.5mm x 3.0mm (Voxel Vol = 0.75 mm3)
    """
    return mt.Compose([
        mt.LoadImaged(keys=["image_t2", "image_adc"]),
        mt.EnsureChannelFirstd(keys=["image_t2", "image_adc"]),
        mt.Spacingd(keys=["image_t2", "image_adc"], 
                    pixdim=(0.5, 0.5, 3.0), 
                    mode=("bilinear", "bilinear")),
        mt.Orientationd(keys=["image_t2", "image_adc"], axcodes="RAS"),
        mt.ResizeWithPadOrCropd(keys=["image_t2", "image_adc"], 
                                spatial_size=(256, 256, 32), 
                                mode="constant"),
        mt.ScaleIntensityRangePercentilesd(keys=["image_t2", "image_adc"], 
                                          lower=1, upper=99, b_min=0.0, b_max=1.0, clip=True),
        mt.ConcatItemsd(keys=["image_t2", "image_adc"], name="image", dim=0),
        mt.EnsureTyped(keys=["image"])
    ])

def run_vision_inference(t2_path: str, adc_path: str, apply_post_processing=True):
    """
    Executes a memory-optimized (AMP, no_grad) forward pass on the MP-MRI pair.
    Uses SlidingWindowInferer for full-volume evaluation.
    
    Args:
        t2_path: Path to T2 NIfTI
        adc_path: Path to ADC NIfTI
        apply_post_processing: Whether to apply CCA and morphological filters
        
    Returns:
        t2_vol: (X, Y, Z) anatomical numpy array.
        adc_vol: (X, Y, Z) diffusion numpy array.
        mask_vol_raw: (X, Y, Z) binary prediction without post-processing (threshold 0.5)
        mask_vol_clean: (X, Y, Z) binary prediction with post-processing (threshold 0.38 + CCA)
    """
    from monai.inferers import SlidingWindowInferer
    import torch
    import sys
    from pathlib import Path
    
    # Needs to find post_processing
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    if str(BASE_DIR / "src" / "vision") not in sys.path:
        sys.path.append(str(BASE_DIR / "src" / "vision"))
    
    try:
        from post_processing import ClinicalPostProcessor
    except ImportError:
         st.error("Error loading ClinicalPostProcessor.")
         raise
    
    model, device = load_vision_model()
    inference_transforms = build_inference_transforms()
    
    # 1. Prepare Data dict for MONAI transforms
    data_dict = {"image_t2": t2_path, "image_adc": adc_path}
    try:
        processed = inference_transforms(data_dict)
    except Exception as e:
        raise ValueError(f"Error procesando volumenes NIfTI: {e}")
        
    # The 'image' tensor has shape (2, X, Y, Z)
    input_tensor = processed["image"].unsqueeze(0).to(device) # adds batch dim -> (1, 2, X, Y, Z)
    
    inferer = SlidingWindowInferer(
        roi_size=(128, 128, 32),
        sw_batch_size=2,
        overlap=0.5,
        mode="gaussian",
    )
    
    # 2. Optimized Forward Pass
    with torch.no_grad():
        with torch.amp.autocast("cuda"):
            output_tensor = inferer(input_tensor, model)
            
    # 3. Post-Process
    # output is (1, 1, X, Y, Z) logits.
    pred_prob = torch.sigmoid(output_tensor).squeeze(0).squeeze(0).cpu().numpy() # -> (X, Y, Z) array
    
    # Raw mask (0.5 threshold, no cleaning)
    mask_vol_raw = (pred_prob > 0.5).astype(np.uint8)
    
    # Clinical post-processing
    if apply_post_processing:
         processor = ClinicalPostProcessor(threshold=0.38, min_volume_mm3=50.0)
         mask_vol_clean = processor(pred_prob)
    else:
         mask_vol_clean = mask_vol_raw
    
    # We extract T2 and ADC from the input tensor directly to ensure spatial matching
    t2_vol_cpu = input_tensor[0, 0].cpu().numpy()
    adc_vol_cpu = input_tensor[0, 1].cpu().numpy()
    
    return t2_vol_cpu, adc_vol_cpu, mask_vol_raw, mask_vol_clean

