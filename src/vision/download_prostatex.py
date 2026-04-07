import os
import shutil
import logging
from pathlib import Path
import tcia_utils.nbia as nbia

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("download-prostatex")

BASE_DIR = Path(__file__).resolve().parent.parent.parent
RAW_DIR = BASE_DIR / "data" / "raw_dicom"

def check_disk_space(required_gb=2):
    free_bytes = shutil.disk_usage(RAW_DIR).free
    free_gb = free_bytes / (1024**3)
    if free_gb < required_gb:
        log.error(f"HARDWARE SAFETY: Espacio insuficiente. Requerido: {required_gb}GB, Disponible: {free_gb:.1f}GB")
        return False
    log.info(f"HARDWARE SAFETY: Espacio disponible: {free_gb:.1f}GB")
    return True

def download_prostatex():
    log.info("Iniciando conexion interactiva con TCIA (NBIA) para PROSTATEx...")
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Fetch Manifest
    try:
        series_data = nbia.getSeries(collection="PROSTATEx")
    except Exception as e:
        log.error(f"Error consultando TCIA: {e}")
        return
        
    if not series_data:
        log.warning("No se encontraron series para PROSTATEx")
        return
        
    log.info(f"Se encontraron {len(series_data)} series en TCIA para la coleccion PROSTATEx.")
    
    target_keywords = ["T2", "ADC", "DWI", "BVAL", "B", "diff"]
    
    # Agrupar series por paciente
    patient_series = {}
    for s in series_data:
        pid = s.get('PatientID')
        if not pid: continue
        if pid not in patient_series:
            patient_series[pid] = []
        patient_series[pid].append(s)
        
    log.info(f"Total de pacientes unicos en la coleccion: {len(patient_series)}")
    
    downloaded = 0
    # 2. Bucle de Descarga
    total_patients = len(patient_series)
    for idx, (pid, series_list) in enumerate(patient_series.items()):
        log.info(f"Procesando {idx+1}/{total_patients} | Paciente: {pid}")
        
        patient_path = RAW_DIR / pid
        
        # Resume Logic (Check if Patient folder already exists and isn't bare)
        if patient_path.exists() and any(patient_path.iterdir()):
            log.info(f"Resume Logic: El paciente {pid} ya existe. Omitiendo descarga...")
            downloaded += 1
            continue
            
        # Hardware Safety Check before every patient
        if not check_disk_space(required_gb=1.0):
            break
            
        log.info(f"--- Procesando Paciente {pid} ---")
        
        selected_series = []
        for s in series_list:
            desc = str(s.get('SeriesDescription', '')).upper()
            if any(k in desc for k in target_keywords):
                selected_series.append(s)
                
        if not selected_series:
            log.warning(f"Paciente {pid} no tiene series target (T2W, ADC, DWI). Saltando...")
            continue
            
        log.info(f"Descargando {len(selected_series)} series priorizadas para {pid}...")
        try:
            # Download directly to patient path
            nbia.downloadSeries(selected_series, path=str(patient_path))
            log.info(f"Descarga de {pid} completada exitosamente.")
            downloaded += 1
        except Exception as e:
            log.error(f"Error descargando {pid}: {e}")

if __name__ == '__main__':
    download_prostatex()
