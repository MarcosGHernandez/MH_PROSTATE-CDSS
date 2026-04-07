import tcia_utils.nbia as nbia
import os
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("check-missing")

def run():
    log.info("Fetching PROSTATEx manifest...")
    series = nbia.getSeries(collection='PROSTATEx')
    
    manifest_p = set(item.get('PatientID') for item in series if item.get('PatientID'))
    local_p = set(os.listdir('data/raw_dicom'))
    missing = sorted(list(manifest_p - local_p))
    
    log.info(f"Missing: {len(missing)}")
    
    target_keywords = ['T2', 'ADC', 'DWI', 'BVAL', 'B', 'DIFF']
    
    for pid in missing:
        patient_series = [s for s in series if s.get('PatientID') == pid]
        descriptions = [s.get('SeriesDescription', '') for s in patient_series]
        
        # Check if it has REAL diagnostic T2 (not just localizer) and ADC/DWI
        has_t2 = any('T2' in d.upper() and 'LOC' not in d.upper() and 'SAG' not in d.upper() and 'COR' not in d.upper() for d in descriptions)
        has_adc = any('ADC' in d.upper() for d in descriptions)
        has_dwi = any('DWI' in d.upper() or 'BVAL' in d.upper() or 'DIFF' in d.upper() for d in descriptions)
        
        log.info(f"{pid}: T2={has_t2}, ADC={has_adc}, DWI={has_dwi} | All={descriptions}")

if __name__ == '__main__':
    run()
