import os
import shutil
import logging
from pathlib import Path
import dicom2nifti

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("dicom-to-nifti")

BASE_DIR = Path(__file__).resolve().parent.parent.parent
RAW_DIR = BASE_DIR / "data" / "raw_dicom"
NIFTI_DIR = BASE_DIR / "data" / "processed_nifti"

def convert_all():
    log.info("Iniciando capa de estandarizacion: DICOM -> NIfTI")
    NIFTI_DIR.mkdir(parents=True, exist_ok=True)
    
    if not RAW_DIR.exists():
        log.error(f"Directorio de entrada no existe: {RAW_DIR}")
        return
        
    patients = [d for d in RAW_DIR.iterdir() if d.is_dir()]
    log.info(f"Se encontraron {len(patients)} pacientes en {RAW_DIR}")
    
    for pat_dir in patients:
        pid = pat_dir.name
        pat_out_dir = NIFTI_DIR / pid
        
        # Resume logic
        if pat_out_dir.exists() and any(pat_out_dir.glob('*.nii.gz')):
            log.info(f"Resume Logic: Paciente {pid} ya convertido a NIfTI. Saltando...")
            continue
            
        pat_out_dir.mkdir(parents=True, exist_ok=True)
        log.info(f"Convirtiendo paciente {pid}...")
        
        # En TCIA, las series suelen estar en subcarpetas anidadas.
        # Buscamos directorios hoja que contengan archivos DICOM.
        for root, dirs, files in os.walk(pat_dir):
            if files and not dirs:
                # Intento de conversion por directorio de serie
                series_name = Path(root).name
                log.info(f"  -> Procesando directorio de serie: {series_name}")
                try:
                    # dicom2nifti falla silenciosamente o da Exception si no es un bloque 3D valido
                    dicom2nifti.convert_directory(root, str(pat_out_dir), compression=True, reorient=True)
                except Exception as e:
                    # Las secuencias Topograma o SR (Structured Reports) no se pueden convertir a NIfTI
                    log.warning(f"  -> Omision: La serie {series_name} no pudo convertirse a NIfTI 3D. ({e})")

if __name__ == '__main__':
    # Modificacion global segura para ignorar validaciones estrictas de slice increment
    import dicom2nifti.settings as settings
    settings.disable_validate_slice_increment()
    
    convert_all()
    log.info("Proceso completado.")
