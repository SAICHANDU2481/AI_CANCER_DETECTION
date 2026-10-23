import os, sys, logging, torch, cv2, numpy as np, pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from torch.utils.data import Dataset
from torchvision import transforms

from classification.roi_generator_v3b import MultiScaleROIGeneratorV3B

logger = logging.getLogger('DatasetV3B')
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

class MammographyMultiScaleDatasetV3B(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        split_name: str = 'train',
        is_training: bool = False,
        local_size: Tuple[int, int] = (224, 224),
        context_size: Tuple[int, int] = (224, 224),
        preload_to_ram: bool = True,
        roi_generator: Optional[MultiScaleROIGeneratorV3B] = None,
        roi_meta_df: Optional[pd.DataFrame] = None
    ):
        if split_name == 'test' and is_training:
            raise RuntimeError('CRITICAL GOVERNANCE VIOLATION: Test cohort cannot be loaded in training mode!')

        self.df = df.reset_index(drop=True)
        self.split_name = split_name
        self.is_training = is_training
        self.local_size = local_size
        self.context_size = context_size
        self.preload_to_ram = preload_to_ram
        self.roi_generator = roi_generator
        self.norm_transform = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

        # Build ROI lookup dictionary from metadata if provided
        self.roi_meta_map = {}
        if roi_meta_df is not None:
            for _, r in roi_meta_df.iterrows():
                cid = str(r['case_id'])
                self.roi_meta_map[cid] = {
                    'center_y': float(r['center_y']),
                    'center_x': float(r['center_x']),
                    'local_box_size': int(r['local_box_size']),
                    'med_box_size': int(r['med_box_size']),
                    'lrg_box_size': int(r['lrg_box_size']),
                    'roi_source': str(r['roi_source']),
                    'fallback_reason': str(r.get('fallback_reason', 'none')),
                    'mask_pixel_count': int(r.get('mask_pixel_count', 0))
                }

        self.ram_cache = {}
        if self.preload_to_ram:
            self._preload_dataset()

        logger.info(f'Initialized DatasetV3B ({split_name}): {len(self.df)} samples | Local Size: {local_size}')

    def _preload_dataset(self):
        for idx in range(len(self.df)):
            row = self.df.iloc[idx]
            case_id = str(row['case_id'])
            npz_path = os.path.join('data/cache/segmentation_512', f'{case_id}.npz')
            if os.path.exists(npz_path):
                npz_data = np.load(npz_path)
                image_512 = npz_data['image'].astype(np.float32)
                prob_mask = npz_data['mask'].astype(np.float32)
            else:
                image_512 = np.zeros((512, 512), dtype=np.float32)
                prob_mask = np.zeros((512, 512), dtype=np.float32)

            if case_id in self.roi_meta_map:
                meta = self.roi_meta_map[case_id]
            elif self.roi_generator is not None:
                meta = self.roi_generator.extract_roi_metadata(image_512, prob_mask)
            else:
                meta = {
                    'center_y': 256.0, 'center_x': 256.0,
                    'local_box_size': 180, 'med_box_size': 270, 'lrg_box_size': 360,
                    'roi_source': 'fallback', 'fallback_reason': 'no_meta', 'mask_pixel_count': 0
                }

            self.ram_cache[case_id] = {
                'image_512': image_512,
                'meta': meta
            }

    def __len__(self) -> int:
        return len(self.df)

    def _apply_training_augmentation(self, image_512: np.ndarray, meta: Dict[str, Any]) -> Tuple[np.ndarray, float, float, int, int, int]:
        H, W = image_512.shape[:2]
        if np.random.rand() > 0.5:
            gamma = np.random.uniform(0.95, 1.05)
            img_aug = np.power(np.clip(image_512, 0.0, 1.0), gamma)
        else:
            img_aug = image_512.copy()

        cy = meta['center_y']
        cx = meta['center_x']
        loc_sz = meta['local_box_size']
        med_sz = meta['med_box_size']
        lrg_sz = meta['lrg_box_size']

        dy = np.random.uniform(-0.05, 0.05) * loc_sz
        dx = np.random.uniform(-0.05, 0.05) * loc_sz
        scale_f = np.random.uniform(0.95, 1.05)

        cy_aug = float(np.clip(cy + dy, 0, H))
        cx_aug = float(np.clip(cx + dx, 0, W))
        loc_aug = max(32, int(round(loc_sz * scale_f)))
        med_aug = max(48, int(round(med_sz * scale_f)))
        lrg_aug = max(64, int(round(lrg_sz * scale_f)))

        return img_aug, cy_aug, cx_aug, loc_aug, med_aug, lrg_aug

    def _to_tensor(self, crop: np.ndarray) -> torch.Tensor:
        tensor = torch.from_numpy(crop).float().unsqueeze(0).repeat(3, 1, 1)
        return self.norm_transform(tensor)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        case_id = str(row['case_id'])
        patient_id = str(row.get('patient_id', ''))
        abnormality_type = str(row.get('abnormality_type', '')).upper()
        pathology = str(row.get('pathology', '')).upper()
        label = 1 if 'MALIGNANT' in pathology else 0

        if self.preload_to_ram and case_id in self.ram_cache:
            item = self.ram_cache[case_id]
            image_512 = item['image_512']
            meta = item['meta']
        else:
            npz_path = os.path.join('data/cache/segmentation_512', f'{case_id}.npz')
            if os.path.exists(npz_path):
                npz_data = np.load(npz_path)
                image_512 = npz_data['image'].astype(np.float32)
                prob_mask = npz_data['mask'].astype(np.float32)
            else:
                image_512 = np.zeros((512, 512), dtype=np.float32)
                prob_mask = np.zeros((512, 512), dtype=np.float32)

            if case_id in self.roi_meta_map:
                meta = self.roi_meta_map[case_id]
            elif self.roi_generator is not None:
                meta = self.roi_generator.extract_roi_metadata(image_512, prob_mask)
            else:
                meta = {
                    'center_y': 256.0, 'center_x': 256.0,
                    'local_box_size': 180, 'med_box_size': 270, 'lrg_box_size': 360,
                    'roi_source': 'fallback', 'fallback_reason': 'no_meta', 'mask_pixel_count': 0
                }

        if self.is_training:
            img_in, cy, cx, loc_sz, med_sz, lrg_sz = self._apply_training_augmentation(image_512, meta)
        else:
            img_in = image_512
            cy, cx = meta['center_y'], meta['center_x']
            loc_sz, med_sz, lrg_sz = meta['local_box_size'], meta['med_box_size'], meta['lrg_box_size']

        local_crop = MultiScaleROIGeneratorV3B.extract_crop(img_in, cy, cx, loc_sz, self.local_size[0])
        med_crop = MultiScaleROIGeneratorV3B.extract_crop(img_in, cy, cx, med_sz, self.context_size[0])
        lrg_crop = MultiScaleROIGeneratorV3B.extract_crop(img_in, cy, cx, lrg_sz, self.context_size[0])

        return {
            'local': self._to_tensor(local_crop),
            'medium': self._to_tensor(med_crop),
            'large': self._to_tensor(lrg_crop),
            'label': torch.tensor(label, dtype=torch.long),
            'case_id': case_id,
            'patient_id': patient_id,
            'abnormality_type': abnormality_type,
            'pathology': pathology,
            'roi_source': meta['roi_source'],
            'fallback_reason': meta.get('fallback_reason', 'none'),
            'mask_pixel_count': meta.get('mask_pixel_count', 0)
        }