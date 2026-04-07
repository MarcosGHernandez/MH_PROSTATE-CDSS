import tcia_utils.nbia as nbia
import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("download-missing")

def run():
    RAW_DIR = Path("data/raw_dicom")
    log.info("Fetching PROSTATEx manifest...")
    series = nbia.getSeries(collection='PROSTATEx')
    
    manifest_p = sorted(list(set(item.get('PatientID') for item in series if item.get('PatientID'))))
    local_p = set(os.listdir(RAW_DIR))
    missing = [pid for pid in manifest_p if pid not in local_p]
    
    log.info(f"Total en Manifiesto: {len(manifest_p)}")
    log.info(f"Faltantes en Local: {len(missing)}")
    
    target_keywords = ['T2', 'ADC', 'DWI', 'BVAL', 'B', 'diff']
    
    for pid in missing:
        log.info(f"--- PROCESANDO {pid} ---")
        patient_path = RAW_DIR / pid
        
        series_list = [item for item in series if item.get('PatientID') == pid]
        selected = [sm for sm in series_list if any(k in str(sm.get('SeriesDescription','')).upper() for k in target_keywords)]
        
        if not selected:
            log.warning(f"SKIPPING {pid}: No tiene secuencias prioritarias (T2/ADC/DWI). Creando carpeta vacia para marcar como procesada.")
            patient_path.mkdir(parents=True, exist_ok=True)
            continue
            
        log.info(f"Descargando {len(selected)} series para {pid}...")
        try:
            nbia.downloadSeries(selected, path=str(patient_path))
            log.info(f"OK: {pid} descargado.")
        except Exception as e:
            log.error(f"FAIL: {pid} error: {e}")

if __name__ == '__main__':
    run()
