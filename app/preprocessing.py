"""
================================================================================
Breast Cancer AI Analysis - Preprocessing & Image Ingestion Module
================================================================================
Pipeline:
1. Multi-format Image Ingestion (PNG, JPG, JPEG, BMP, DICOM)
2. Bit-depth normalization & Dynamic Range Scaling
3. Outlier Percentile Clipping (0.5% - 99.5%)
4. Adaptive Wiener Filtering (Speckle & Sensor Noise Denoising)
5. CLAHE (Contrast Limited Adaptive Histogram Equalization)
6. Coordinate standardization & Multi-Resolution Scaling
================================================================================
"""

import os
import io
import logging
from typing import Dict, Any, Optional, Tuple, Union
import numpy as np
from PIL import Image
import cv2

# Optional DICOM support via pydicom
try:
    import pydicom
    PYDICOM_AVAILABLE = True
except ImportError:
    PYDICOM_AVAILABLE = False

from preprocessing.enhancement import enhance_mammogram as core_enhance_mammogram
from app.config import resolve_path

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")
logger = logging.getLogger("App_Preprocessing")


def load_image_from_upload(uploaded_file) -> Optional[np.ndarray]:
    """
    Reads an uploaded mammogram file into a standardized float32 2D grayscale array in [0.0, 1.0].
    Supports PNG, JPG, JPEG, BMP, and DICOM formats.
    """
    if uploaded_file is None:
        return None

    filename = uploaded_file.name.lower()
    file_bytes = uploaded_file.read()
    uploaded_file.seek(0)

    # 1. DICOM Handling
    if filename.endswith(('.dcm', '.dicom')) or file_bytes[:132].endswith(b'DICM'):
        if not PYDICOM_AVAILABLE:
            raise ImportError(
                "pydicom is required to parse native DICOM mammography files. "
                "Please upload a standard PNG/JPG scan or install pydicom."
            )
        try:
            dcm = pydicom.dcmread(io.BytesIO(file_bytes), force=True)
            pixel_array = dcm.pixel_array.astype(np.float32)

            # Invert MONOCHROME1 if necessary (where zero is white)
            photometric = getattr(dcm, 'PhotometricInterpretation', 'MONOCHROME2')
            if photometric == 'MONOCHROME1':
                pixel_array = np.max(pixel_array) - pixel_array

            # Normalize to [0.0, 1.0]
            p_min, p_max = np.min(pixel_array), np.max(pixel_array)
            if p_max > p_min:
                normalized = (pixel_array - p_min) / (p_max - p_min)
            else:
                normalized = np.zeros_like(pixel_array)
            return normalized.astype(np.float32)
        except Exception as e:
            logger.error(f"Failed to parse DICOM stream: {e}")
            raise ValueError(f"Could not decode DICOM mammogram: {e}")

    # 2. Standard Raster Images (PNG, JPG, JPEG, BMP)
    try:
        pil_img = Image.open(io.BytesIO(file_bytes)).convert("L")
        img_np = np.array(pil_img, dtype=np.float32) / 255.0
        if img_np.size == 0 or img_np.shape[0] < 32 or img_np.shape[1] < 32:
            raise ValueError("Uploaded image resolution is too small or invalid.")
        return img_np
    except Exception as e:
        logger.warning(f"PIL failed ({e}), attempting OpenCV fallback...")
        nparr = np.frombuffer(file_bytes, np.uint8)
        img_np = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)

    if img_np is None:
        raise ValueError("Unsupported or corrupted image file format.")

    # Convert to single-channel 2D
    if img_np.ndim == 3:
        if img_np.shape[2] == 1:
            img_np = img_np.squeeze(2)
        elif img_np.shape[2] in (3, 4):
            img_np = cv2.cvtColor(img_np[:, :, :3], cv2.COLOR_RGB2GRAY)

    img_f32 = img_np.astype(np.float32)
    p_min, p_max = np.min(img_f32), np.max(img_f32)
    if p_max > p_min:
        normalized = (img_f32 - p_min) / (p_max - p_min)
    else:
        normalized = np.zeros_like(img_f32)

    return normalized.astype(np.float32)


def load_demo_scan_image(case_data: Dict[str, Any]) -> Optional[np.ndarray]:
    """
    Loads curated benchmark case mammogram from available file paths.
    """
    candidates = [
        case_data.get("image_path", ""),
        case_data.get("alt_path", ""),
        resolve_path(case_data.get("image_path", "")),
        resolve_path(case_data.get("alt_path", ""))
    ]

    for p in candidates:
        if p and os.path.exists(p):
            try:
                pil_img = Image.open(p).convert("L")
                return np.array(pil_img, dtype=np.float32) / 255.0
            except Exception as e:
                logger.warning(f"Failed loading image from {p}: {e}")
                pass
    return None


def preprocess_mammogram_full(raw_image: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Runs the complete verified preprocessing transformation sequence:
    1. Base Normalization [0, 1]
    2. Adaptive Wiener Filtering
    3. CLAHE Contrast Enhancement
    4. Multi-scale standard working dimensions (512x512, 768x768, 224x224)
    """
    prep_dict = core_enhance_mammogram(raw_image)
    orig_norm = prep_dict["original"]
    wiener_img = prep_dict["wiener"]
    clahe_img = prep_dict["enhanced"]

    # Generate standard working resolutions
    img_512 = cv2.resize(clahe_img, (512, 512), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    img_768 = cv2.resize(img_512, (768, 768), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    img_224 = cv2.resize(img_512, (224, 224), interpolation=cv2.INTER_AREA).astype(np.float32)

    return {
        "original": orig_norm,
        "wiener": wiener_img,
        "enhanced": clahe_img,
        "input_512": img_512,
        "input_768": img_768,
        "input_224": img_224,
        "original_shape": raw_image.shape
    }
