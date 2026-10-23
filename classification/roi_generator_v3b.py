import os, sys, cv2, numpy as np, torch, torch.nn.functional as F
from typing import Dict, Any, Tuple, Optional, List
from segmentation.ukan_v2b import MultiScaleUKAN

class MultiScaleROIGeneratorV3B:
    def __init__(self, checkpoint_path='models/v2b/segmentation/ukan_v2b_best.pth', device=None, prob_threshold=0.50, min_area=10, local_target_size=224, context_target_size=224):
        self.device = device or (torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        self.prob_threshold = prob_threshold
        self.min_area = min_area
        self.local_target_size = local_target_size
        self.context_target_size = context_target_size
        self.segmentor = MultiScaleUKAN(in_channels=1, out_channels=1).to(self.device)
        chk = torch.load(checkpoint_path, map_location=self.device)
        if isinstance(chk, dict) and 'model_state_dict' in chk:
            self.segmentor.load_state_dict(chk['model_state_dict'])
        elif isinstance(chk, dict) and 'state_dict' in chk:
            self.segmentor.load_state_dict(chk['state_dict'])
        else:
            self.segmentor.load_state_dict(chk)
        self.segmentor.eval()
        for p in self.segmentor.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def segment_batch(self, batch_tensors: torch.Tensor) -> np.ndarray:
        batch_tensors = batch_tensors.to(self.device, dtype=torch.float32)
        out = self.segmentor(batch_tensors)
        logits = out[0] if isinstance(out, tuple) else out
        probs = torch.sigmoid(logits).squeeze(1).cpu().numpy()
        return probs

    @torch.no_grad()
    def segment_image(self, image_tensor: torch.Tensor) -> np.ndarray:
        if image_tensor.dim() == 2:
            image_tensor = image_tensor.unsqueeze(0).unsqueeze(0)
        elif image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)
        image_tensor = image_tensor.to(self.device, dtype=torch.float32)
        out = self.segmentor(image_tensor)
        logits = out[0] if isinstance(out, tuple) else out
        prob = torch.sigmoid(logits).squeeze().cpu().numpy()
        return prob

    @staticmethod
    def extract_crop(img: np.ndarray, cy: float, cx: float, crop_size: int, target_size: int) -> np.ndarray:
        H, W = img.shape[:2]
        half = crop_size / 2.0
        y1, y2 = int(round(cy - half)), int(round(cy + half))
        x1, x2 = int(round(cx - half)), int(round(cx + half))
        pad_t, pad_b = max(0, -y1), max(0, y2 - H)
        pad_l, pad_r = max(0, -x1), max(0, x2 - W)
        if pad_t > 0 or pad_b > 0 or pad_l > 0 or pad_r > 0:
            padded = cv2.copyMakeBorder(img, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_REFLECT_101)
            crop = padded[y1 + pad_t : y2 + pad_t, x1 + pad_l : x2 + pad_l]
        else:
            crop = img[y1:y2, x1:x2]
        if crop.shape[0] != target_size or crop.shape[1] != target_size:
            crop = cv2.resize(crop, (target_size, target_size), interpolation=cv2.INTER_CUBIC)
        return crop

    def extract_roi_metadata(self, image_np: np.ndarray, prob_map: np.ndarray) -> dict:
        binary_mask = (prob_map >= self.prob_threshold).astype(np.uint8)
        mask_pixel_count = int(np.sum(binary_mask))
        if mask_pixel_count >= self.min_area:
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask)
            if num_labels > 1:
                best_label, best_score = 1, -1.0
                for lbl in range(1, num_labels):
                    area = stats[lbl, cv2.CC_STAT_AREA]
                    if area < self.min_area:
                        continue
                    mean_prob = float(np.mean(prob_map[labels == lbl]))
                    score = mean_prob * np.log1p(area)
                    if score > best_score:
                        best_score, best_label = score, lbl
                x = stats[best_label, cv2.CC_STAT_LEFT]
                y = stats[best_label, cv2.CC_STAT_TOP]
                w = stats[best_label, cv2.CC_STAT_WIDTH]
                h = stats[best_label, cv2.CC_STAT_HEIGHT]
                cx, cy = centroids[best_label]
                roi_source, fallback_reason = 'machine_roi', 'none'
            else:
                roi_source, fallback_reason = 'fallback', 'low_confidence'
                tissue_mask = (image_np > (image_np.min() + 0.05 * (image_np.max() - image_np.min()))).astype(np.uint8)
                M = cv2.moments(tissue_mask)
                cx = float(M['m10'] / M['m00']) if M['m00'] > 0 else image_np.shape[1] / 2.0
                cy = float(M['m01'] / M['m00']) if M['m00'] > 0 else image_np.shape[0] / 2.0
                w, h = int(image_np.shape[1] * 0.35), int(image_np.shape[0] * 0.35)
                x, y = int(cx - w / 2), int(cy - h / 2)
        else:
            roi_source = 'fallback'
            fallback_reason = 'no_prediction' if mask_pixel_count == 0 else 'low_confidence'
            tissue_mask = (image_np > (image_np.min() + 0.05 * (image_np.max() - image_np.min()))).astype(np.uint8)
            M = cv2.moments(tissue_mask)
            cx = float(M['m10'] / M['m00']) if M['m00'] > 0 else image_np.shape[1] / 2.0
            cy = float(M['m01'] / M['m00']) if M['m00'] > 0 else image_np.shape[0] / 2.0
            w, h = int(image_np.shape[1] * 0.35), int(image_np.shape[0] * 0.35)
            x, y = int(cx - w / 2), int(cy - h / 2)
        base_size = max(w, h, 64)
        local_box_size = int(round(base_size * 1.2))
        med_box_size = int(round(local_box_size * 1.5))
        lrg_box_size = int(round(local_box_size * 2.0))
        return {
            'roi_source': roi_source,
            'fallback_reason': fallback_reason,
            'center_y': float(cy),
            'center_x': float(cx),
            'ymin': int(y), 'xmin': int(x),
            'ymax': int(y + h), 'xmax': int(x + w),
            'local_box_size': local_box_size,
            'med_box_size': med_box_size,
            'lrg_box_size': lrg_box_size,
            'mask_pixel_count': mask_pixel_count
        }

    def generate_rois(self, image_np: np.ndarray, prob_map: Optional[np.ndarray] = None, local_size_override: Optional[int] = None) -> dict:
        local_target_size = local_size_override or self.local_target_size
        if prob_map is None:
            tensor = torch.from_numpy(image_np).float().unsqueeze(0).unsqueeze(0)
            prob_map = self.segment_image(tensor)
        meta = self.extract_roi_metadata(image_np, prob_map)
        cy, cx = meta['center_y'], meta['center_x']
        meta['local_crop'] = self.extract_crop(image_np, cy, cx, meta['local_box_size'], local_target_size)
        meta['medium_crop'] = self.extract_crop(image_np, cy, cx, meta['med_box_size'], self.context_target_size)
        meta['large_crop'] = self.extract_crop(image_np, cy, cx, meta['lrg_box_size'], self.context_target_size)
        return meta