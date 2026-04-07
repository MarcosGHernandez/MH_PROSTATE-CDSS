import re
import pandas as pd
from pathlib import Path

BASE_DIR = Path(r"c:\Users\Predator Pro\OneDrive\Documents\Proyectos\Marcos_proyects\Health_can")
LOG_FILE = BASE_DIR / "runs" / "v12_focal_precision" / "train_phase12.txt"

def parse_logs():
    data = []
    if not LOG_FILE.exists():
        return None
    
    with open(LOG_FILE, "r") as f:
        for line in f:
            if "| Val Dice:" in line and "Epoch" in line:
                try:
                    ep_match = re.search(r"Epoch (\d+)/(\d+)", line)
                    loss_match = re.search(r"Loss: (\d+\.\d+)", line)
                    dice_match = re.search(r"Val Dice: (-?\d+\.\d+)", line)
                    hd_match = re.search(r"Val HD95: (\d+\.\d+)", line)
                    lr_match = re.search(r"LR: (\d+\.\d+e[+-]\d+)", line)
                    
                    if ep_match and loss_match and dice_match:
                        data.append({
                            "Epoch": int(ep_match.group(1)),
                            "Loss": float(loss_match.group(1)),
                            "Dice": float(dice_match.group(1)),
                            "HD95": float(hd_match.group(1)) if hd_match else 999.0,
                            "LR": float(lr_match.group(1)) if lr_match else 0.0
                        })
                except:
                    continue
    return pd.DataFrame(data)

df = parse_logs()
if df is not None:
    df.to_csv(BASE_DIR / "runs" / "v12_focal_precision" / "metrics_history.csv", index=False)
    print(f"Parsed {len(df)} epochs")
    print(df.tail(5))
    print(df.describe())
else:
    print("Log file not found")
