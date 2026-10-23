import os
import pandas as pd
import numpy as np
from PIL import Image

df = pd.read_csv("data/processed/metadata.csv")
print(f"Total metadata records: {len(df)}")
print(df.head(2).T)

# Check 5 random pairs
np.random.seed(42)
sample_rows = df.sample(5, random_state=42)

for idx, row in sample_rows.iterrows():
    img_p = row["image_path"]
    mask_p = row["mask_path"]
    crop_p = row["crop_path"]
    
    print("\n" + "="*50)
    print(f"Case: {row['case_id']} | Abnormality: {row['abnormality_type']} | Pathology: {row['pathology']} (Label: {row['label']})")
    print(f"Split: {row['split']} | Patient: {row['patient_id']}")
    
    if os.path.exists(img_p):
        with Image.open(img_p) as im:
            print(f"  Full Image Size: {im.size}, Mode: {im.mode}")
    else:
        print(f"  Full Image NOT FOUND: {img_p}")
        
    if os.path.exists(mask_p):
        with Image.open(mask_p) as mm:
            arr = np.array(mm)
            print(f"  Mask Size: {mm.size}, Mode: {mm.mode}, Min={arr.min()}, Max={arr.max()}, NonZeroPixels={np.count_nonzero(arr)}")
    else:
        print(f"  Mask NOT FOUND: {mask_p}")
        
    if os.path.exists(crop_p):
        with Image.open(crop_p) as cr:
            print(f"  Crop Size: {cr.size}, Mode: {cr.mode}")
    else:
        print(f"  Crop NOT FOUND: {crop_p}")
