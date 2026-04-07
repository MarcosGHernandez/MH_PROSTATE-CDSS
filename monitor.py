import time
import sys

def main():
    log_file = "runs/prostate_3d_unet/train_clinical_log.txt"
    print("Monitoring up to Epoch 5...")
    
    with open(log_file, "r") as f:
        f.seek(0, 2)  # Go to end
        
        while True:
            line = f.readline()
            if not line:
                time.sleep(2)
                continue
                
            if "| INFO |" in line:
                print(line.strip())
                if "Epoch 5/" in line:
                    print("Reached Epoch 5. Exiting monitor.")
                    break

if __name__ == "__main__":
    main()
