import transforms
from monai.data import Dataset
import json
import traceback

def main():
    try:
        data = json.load(open('../../data/kaggle_dataset_split.json'))
        train_files = [{'image_t2':d['image_t2'], 'image_adc':d['image_adc'], 'label':d['label']} for d in data]
        ds = Dataset(data=train_files, transform=transforms.get_train_transforms())
        
        # Test the first sample
        print("Testing sample 0...")
        ds[0]
        print("Success! No Error.")
    except Exception as e:
        print("FAILED WITH EXCEPTION:")
        traceback.print_exc()

if __name__ == '__main__':
    main()
