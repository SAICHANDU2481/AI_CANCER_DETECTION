import logging
from typing import Dict, Any, Optional
import numpy as np
import cv2
import torch

from utils.seed import get_device
from classification.vit_model import VisionTransformerClassifier

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")
logger = logging.getLogger("ClassificationInference")


def classify_roi(
    roi: np.ndarray,
    model: Optional[VisionTransformerClassifier] = None,
    device: Optional[torch.device] = None,
    class_names: list = ["Benign", "Malignant"]
) -> Dict[str, Any]:
    """
    Performs classification on an extracted lesion ROI image.

    Args:
        roi: 2D numpy array [0, 1] of shape (H, W).
        model: Instantiated VisionTransformerClassifier or None.
        device: Torch compute device.
        class_names: List of class string names.

    Returns:
        Dictionary containing:
        - "predicted_class": str ("Benign" or "Malignant")
        - "class_id": int (0 or 1)
        - "confidence": float (0.0 to 100.0%)
        - "probabilities": {"Benign": float, "Malignant": float}
        - "status": str
    """
    # 1. Resize to ViT standard resolution (224, 224)
    roi_resized = cv2.resize(roi, (224, 224), interpolation=cv2.INTER_AREA)

    # 2. Build 3-channel normalized tensor
    roi_3ch = np.stack([roi_resized, roi_resized, roi_resized], axis=0).astype(np.float32)
    tensor_roi = torch.from_numpy(roi_3ch).float()
    
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    tensor_roi = ((tensor_roi - mean) / std).unsqueeze(0) # (1, 3, 224, 224)

    if model is None:
        logger.warning("No ViT model supplied to inference. Returning uncalibrated diagnostic status.")
        return {
            "status": "model_missing",
            "predicted_class": "Model Not Loaded",
            "class_id": -1,
            "confidence": 0.0,
            "probabilities": {"Benign": 0.0, "Malignant": 0.0}
        }

    if device is None:
        device = get_device("auto")

    model.eval()
    tensor_roi = tensor_roi.to(device)

    with torch.no_grad():
        logits = model(tensor_roi)
        probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
        pred_idx = int(np.argmax(probs))
        confidence = float(probs[pred_idx] * 100.0)

    pred_name = class_names[pred_idx] if pred_idx < len(class_names) else f"Class_{pred_idx}"

    return {
        "status": "success",
        "predicted_class": pred_name,
        "class_id": pred_idx,
        "confidence": confidence,
        "probabilities": {
            "Benign": float(probs[0]),
            "Malignant": float(probs[1]) if len(probs) > 1 else 0.0
        }
    }


def predict_lesion(image_roi: np.ndarray, model: VisionTransformerClassifier) -> str:
    """Convenience helper returning the predicted diagnosis string."""
    res = classify_roi(image_roi, model=model)
    return res["predicted_class"]
