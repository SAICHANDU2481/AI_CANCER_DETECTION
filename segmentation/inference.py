import logging
from typing import Tuple, Dict, Any, Optional
import numpy as np
import cv2
import torch
from PIL import Image

from utils.seed import get_device
from preprocessing.normalization import normalize_intensity
from segmentation.ukan import UKAN

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")
logger = logging.getLogger("ROIInference")


def extract_lesion_roi(
    image: np.ndarray,
    predicted_mask: np.ndarray,
    threshold: float = 0.5,
    min_area_pixels: int = 50,
    margin_ratio: float = 0.20,
    target_roi_size: Tuple[int, int] = (224, 224),
    fallback_strategy: str = "center_crop"
) -> Dict[str, Any]:
    """
    Extracts high-resolution Region of Interest (ROI) from mammogram using predicted segmentation mask.
    Performs morphological filtering, connected component analysis, bounding box extraction with margin.

    Args:
        image: 2D numpy array [0, 1] of the preprocessed mammogram.
        predicted_mask: 2D probability map [0, 1] from U-KAN.
        threshold: Binarization probability threshold.
        min_area_pixels: Minimum blob area in pixels to reject spurious noise.
        margin_ratio: Context padding percentage around lesion bounding box.
        target_roi_size: Output resolution for ViT classification (H, W).
        fallback_strategy: "center_crop" or "full_resize" if no ROI detected.

    Returns:
        Dictionary containing:
        - "roi_detected": bool (True if valid lesion found, False otherwise)
        - "roi": 2D numpy array [0, 1] resized to target_roi_size
        - "binary_mask": 2D binary uint8 mask
        - "bounding_box": (ymin, xmin, ymax, xmax) or None
        - "lesion_area": int (pixels)
        - "status_message": str
    """
    h, w = image.shape[:2]
    
    # 1. Binarize predicted probability mask
    binary_mask = (predicted_mask >= threshold).astype(np.uint8)

    # 2. Connected Component Analysis to remove small spurious artifacts
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)

    best_bbox = None
    max_area = 0
    clean_mask = np.zeros_like(binary_mask)

    # Label 0 is background; iterate through detected foreground components
    for label_idx in range(1, num_labels):
        area = stats[label_idx, cv2.CC_STAT_AREA]
        if area >= min_area_pixels:
            if area > max_area:
                max_area = area
                xmin = stats[label_idx, cv2.CC_STAT_LEFT]
                ymin = stats[label_idx, cv2.CC_STAT_TOP]
                bw = stats[label_idx, cv2.CC_STAT_WIDTH]
                bh = stats[label_idx, cv2.CC_STAT_HEIGHT]
                best_bbox = (ymin, xmin, ymin + bh, xmin + bw)
                
            # Keep all sufficiently large components in the clean mask
            clean_mask[labels == label_idx] = 1

    # 3. Fallback handling if no reliable ROI detected
    if best_bbox is None or max_area < min_area_pixels:
        msg = "No reliable ROI was detected."
        logger.info(f"Segmentation inference: {msg} Applying {fallback_strategy} fallback.")

        if fallback_strategy == "center_crop":
            # Extract central 50% region as context fallback
            ch, cw = int(h * 0.5), int(w * 0.5)
            y_start = (h - ch) // 2
            x_start = (w - cw) // 2
            roi_crop = image[y_start:y_start + ch, x_start:x_start + cw]
            bbox = (y_start, x_start, y_start + ch, x_start + cw)
        else:
            roi_crop = image
            bbox = (0, 0, h, w)

        roi_resized = cv2.resize(roi_crop, target_roi_size, interpolation=cv2.INTER_AREA)
        
        return {
            "roi_detected": False,
            "roi": normalize_intensity(roi_resized, 0.0, 1.0),
            "binary_mask": clean_mask,
            "bounding_box": bbox,
            "lesion_area": 0,
            "status_message": msg
        }

    # 4. Add margin padding around lesion bounding box
    ymin, xmin, ymax, xmax = best_bbox
    bw = xmax - xmin
    bh = ymax - ymin
    
    pad_y = int(bh * margin_ratio)
    pad_x = int(bw * margin_ratio)

    ymin_padded = max(0, ymin - pad_y)
    xmin_padded = max(0, xmin - pad_x)
    ymax_padded = min(h, ymax + pad_y)
    xmax_padded = min(w, xmax + pad_x)

    roi_crop = image[ymin_padded:ymax_padded, xmin_padded:xmax_padded]
    roi_resized = cv2.resize(roi_crop, target_roi_size, interpolation=cv2.INTER_AREA)

    return {
        "roi_detected": True,
        "roi": normalize_intensity(roi_resized, 0.0, 1.0),
        "binary_mask": clean_mask,
        "bounding_box": (ymin_padded, xmin_padded, ymax_padded, xmax_padded),
        "lesion_area": int(max_area),
        "status_message": f"Reliable lesion ROI localized (Area: {max_area} px)."
    }


def segment_and_crop(
    model: Optional[UKAN],
    preprocessed_image: np.ndarray,
    device: Optional[torch.device] = None,
    threshold: float = 0.5,
    min_area_pixels: int = 50,
    margin_ratio: float = 0.20,
    target_roi_size: Tuple[int, int] = (224, 224)
) -> Dict[str, Any]:
    """
    Executes model-based segmentation forward pass and extracts ROI.
    """
    h_orig, w_orig = preprocessed_image.shape[:2]

    if model is not None:
        if device is None:
            device = get_device("auto")
        model.eval()
        
        # Prepare tensor (1, 1, H, W)
        input_resized = cv2.resize(preprocessed_image, (512, 512), interpolation=cv2.INTER_AREA)
        tensor_in = torch.from_numpy(input_resized).unsqueeze(0).unsqueeze(0).float().to(device)

        with torch.no_grad():
            logits = model(tensor_in)
            probs = torch.sigmoid(logits).squeeze().cpu().numpy()

        # Resize probability mask back to original resolution
        prob_mask = cv2.resize(probs, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)
    else:
        # Fallback if model not instantiated / untrain mode
        prob_mask = np.zeros((h_orig, w_orig), dtype=np.float32)

    roi_result = extract_lesion_roi(
        image=preprocessed_image,
        predicted_mask=prob_mask,
        threshold=threshold,
        min_area_pixels=min_area_pixels,
        margin_ratio=margin_ratio,
        target_roi_size=target_roi_size
    )

    roi_result["probability_mask"] = prob_mask
    return roi_result
