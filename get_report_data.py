import os
import pandas as pd
import json
import yaml

meta = pd.read_csv('data/processed/verified_metadata.csv')
train = pd.read_csv('data/processed/verified_train.csv')
val = pd.read_csv('data/processed/verified_val.csv')
test = pd.read_csv('data/processed/verified_test.csv')

print('--- TOTAL RECORDS ---')
print(f'Total in metadata: {len(meta)} (Patients: {meta.patient_id.nunique()})')
print(f'Valid for segmentation: {meta.is_valid_for_segmentation.sum()}')
print(f'Invalid for segmentation: {(~meta.is_valid_for_segmentation).sum()}')

print('\n--- SPLIT BREAKDOWN (Total / Segmentation Valid / Patients) ---')
for name, df in [('Train', train), ('Val', val), ('Test', test)]:
    print(f'{name}: Total={len(df)}, Seg_Valid={df.is_valid_for_segmentation.sum()}, Patients={df.patient_id.nunique()}')

print('\n--- ABNORMALITY TYPE BREAKDOWN (Segmentation Valid) ---')
valid_meta = meta[meta.is_valid_for_segmentation == True]
print(valid_meta['abnormality_type'].value_counts())

print('\n--- PATHOLOGY BREAKDOWN (Segmentation Valid) ---')
print(valid_meta['pathology'].value_counts())

print('\n--- ABNORMALITY x PATHOLOGY (Segmentation Valid) ---')
print(pd.crosstab(valid_meta['abnormality_type'], valid_meta['pathology']))

print('\n--- INVALID RECORDS DETAILS ---')
invalid_df = meta[meta.is_valid_for_segmentation == False]
for _, r in invalid_df.iterrows():
    print(f"Case: {r['case_id']}, Patient: {r['patient_id']}, Split: {r['split']}, Status: {r['mask_status']}, Mask Exists: {r['mask_exists']}")
