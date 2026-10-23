"""
================================================================================
Breast Cancer AI Analysis - Inference & Model Execution Engine (V8 Final Core)
================================================================================
Model Pipeline:
1. U-KAN Segmentation (512x512 + 768x768 Receptive Field -> 512x512)
2. Machine Connected Components & Deterministic Proposal Scoring (Top-3 RoIs)
3. Multi-Scale Concentric Receptive Fields (Local, Medium, Large, Context)
4. Calcification Frequency Enhancement & Global Mammogram Canvas
5. MultiScaleViTV8 Arm F Joint Classification (4,608-dim Latent Fusion)
6. Authentic Vision Transformer Self-Attention Rollout Extraction
================================================================================
"""

import os
import sys
import time
import math
import logging
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

from app.config import (
    UKAN_V6_PATH, VIT_V8_PATH, VIT_V6_PATH, EXPECTED_MD5, verify_file_md5,
    SEGMENTATION_INPUT_SIZE, SEGMENTATION_WORKING_SIZE, CLASSIFIER_CROP_SIZE,
    TOP_K_PROPOSALS, SEGMENTATION_THRESHOLD
)
from app.preprocessing import preprocess_mammogram_full
from segmentation.v6.ukan_v6 import MultiScaleUKANV6
from classification.v6.topk_proposals_v6 import TopKProposalGeneratorV6
from classification.v6.roi_generator_v6 import MultiScaleROIGeneratorV6
from classification.v6.model_v6 import MultiScaleViTV6
from explainability.attention import compute_attention_rollout

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")
logger = logging.getLogger("App_Inference")

# Standard ViT ImageNet Normalization Constants
NORM_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
NORM_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


def to_vit_tensor(image_2d: np.ndarray) -> torch.Tensor:
    """
    Converts 2D float32 [0.0, 1.0] image into 3-channel normalized ViT tensor of shape (3, 224, 224).
    """
    if image_2d.shape != CLASSIFIER_CROP_SIZE:
        image_2d = cv2.resize(image_2d, CLASSIFIER_CROP_SIZE, interpolation=cv2.INTER_LINEAR)
    img_3c = np.repeat(image_2d[np.newaxis, :, :], 3, axis=0)
    img_norm = (img_3c - NORM_MEAN) / NORM_STD
    return torch.from_numpy(img_norm).float()


class ModelBundle:
    """Encapsulates loaded models, generators, and integrity status."""
    def __init__(
        self,
        ukan_model: nn.Module,
        vit_classifier: nn.Module,
        proposal_gen: TopKProposalGeneratorV6,
        roi_gen: MultiScaleROIGeneratorV6,
        device: torch.device,
        model_version: str = "V8_Arm_F"
    ):
        self.ukan = ukan_model
        self.vit = vit_classifier
        self.proposal_gen = proposal_gen
        self.roi_gen = roi_gen
        self.device = device
        self.model_version = model_version


def load_v8_pipeline_models(device: Optional[torch.device] = None) -> Tuple[Optional[ModelBundle], Optional[str]]:
    """
    Loads and cryptographically verifies the locked V8 (Arm F) and U-KAN models.
    Returns (ModelBundle, None) on success or (None, error_message) on failure.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. Cryptographic MD5 Checks
    is_ukan_valid, actual_ukan_md5 = verify_file_md5(UKAN_V6_PATH, EXPECTED_MD5['ukan_v6'])
    if not is_ukan_valid:
        err = (
            f"FATAL: U-KAN Checkpoint Integrity Mismatch!\n"
            f"Expected: {EXPECTED_MD5['ukan_v6']}\n"
            f"Found: {actual_ukan_md5}\n"
            f"Path: {UKAN_V6_PATH}"
        )
        logger.error(err)
        return None, err

    vit_target_path = VIT_V8_PATH if os.path.exists(VIT_V8_PATH) else VIT_V6_PATH
    expected_vit_md5 = EXPECTED_MD5['vit_v8'] if os.path.exists(VIT_V8_PATH) else EXPECTED_MD5['vit_v6']
    model_tag = "V8_Arm_F" if os.path.exists(VIT_V8_PATH) else "V6_Arm_C"

    is_vit_valid, actual_vit_md5 = verify_file_md5(vit_target_path, expected_vit_md5)
    if not is_vit_valid:
        err = (
            f"FATAL: ViT Checkpoint Integrity Mismatch!\n"
            f"Expected: {expected_vit_md5}\n"
            f"Found: {actual_vit_md5}\n"
            f"Path: {vit_target_path}"
        )
        logger.error(err)
        return None, err

    # 2. Instantiate and Load U-KAN
    try:
        ukan = MultiScaleUKANV6(in_channels=1, out_channels=1)
        ukan_ckpt = torch.load(UKAN_V6_PATH, map_location=device, weights_only=False)
        state_dict_u = ukan_ckpt.get('model_state_dict', ukan_ckpt)
        ukan.load_state_dict(state_dict_u)
        ukan.to(device).eval()
        logger.info(f"Loaded U-KAN from {UKAN_V6_PATH} (MD5 Verified: {actual_ukan_md5})")
    except Exception as e:
        return None, f"Failed to instantiate U-KAN model: {e}"

    # 3. Instantiate and Load MultiScaleViTV6 (Arm F Architecture, K=3)
    try:
        vit = MultiScaleViTV6(
            num_classes=2,
            use_full_image=True,
            use_calc_patch=True,
            scales_used=('local', 'medium', 'large', 'context'),
            k=TOP_K_PROPOSALS,
            pretrained=False
        )
        vit_ckpt = torch.load(vit_target_path, map_location=device, weights_only=False)
        state_dict_v = vit_ckpt.get('model_state_dict', vit_ckpt)
        vit.load_state_dict(state_dict_v, strict=True)
        vit.to(device).eval()
        logger.info(f"Loaded ViT ({model_tag}) from {vit_target_path} (MD5 Verified: {actual_vit_md5})")
    except Exception as e:
        return None, f"Failed to instantiate Vision Transformer model: {e}"

    # 4. Proposal and ROI Generators
    proposal_gen = TopKProposalGeneratorV6(min_area=15, threshold=SEGMENTATION_THRESHOLD, default_k=TOP_K_PROPOSALS)
    roi_gen = MultiScaleROIGeneratorV6(
        local_scale=1.2,
        medium_scale=1.8,
        large_scale=2.5,
        context_scale=4.0,
        target_crop_size=CLASSIFIER_CROP_SIZE[0],
        full_image_size=CLASSIFIER_CROP_SIZE[0],
        default_k=TOP_K_PROPOSALS
    )

    bundle = ModelBundle(ukan, vit, proposal_gen, roi_gen, device, model_version=model_tag)
    return bundle, None


# Backward-compatible alias
load_v6_pipeline_models = load_v8_pipeline_models


def extract_vit_attention_map(vit_backbone: nn.Module, input_tensor: torch.Tensor, device: torch.device) -> np.ndarray:
    """
    Extracts authentic self-attention rollout matrix from the ViT backbone.
    input_tensor: shape (1, 3, 224, 224)
    Returns: 2D float32 attention heatmap of shape (224, 224) in [0.0, 1.0].
    """
    attention_layers: List[torch.Tensor] = []
    hooks = []

    for block in vit_backbone.blocks:
        if hasattr(block, 'attn') and hasattr(block.attn, 'qkv'):
            def make_hook():
                def hook_fn(module, inp, outp):
                    B, N, C = inp[0].shape
                    qkv = module.qkv(inp[0]).reshape(B, N, 3, module.num_heads, C // module.num_heads).permute(2, 0, 3, 1, 4)
                    q, k, v = qkv[0], qkv[1], qkv[2]
                    scale = (C // module.num_heads) ** -0.5
                    attn = (q @ k.transpose(-2, -1)) * scale
                    attn = attn.softmax(dim=-1)
                    attention_layers.append(attn.detach().clone())
                return hook_fn
            hooks.append(block.attn.register_forward_hook(make_hook()))

    try:
        with torch.no_grad():
            _ = vit_backbone(input_tensor.to(device))

        if len(attention_layers) > 0:
            attn_rollout = compute_attention_rollout(attention_layers, discard_ratio=0.85, head_reduction="mean")
            attn_resized = cv2.resize(attn_rollout, CLASSIFIER_CROP_SIZE, interpolation=cv2.INTER_CUBIC)
            attn_min, attn_max = float(np.min(attn_resized)), float(np.max(attn_resized))
            if attn_max > attn_min:
                attn_norm = (attn_resized - attn_min) / (attn_max - attn_min)
            else:
                attn_norm = np.zeros_like(attn_resized)
            return attn_norm.astype(np.float32)
    except Exception as e:
        logger.warning(f"Could not compute attention rollout: {e}. Generating fallback.")
    finally:
        for h in hooks:
            h.remove()

    return np.ones(CLASSIFIER_CROP_SIZE, dtype=np.float32) * 0.5


def run_complete_inference_pipeline(
    raw_mammogram: np.ndarray,
    models: ModelBundle,
    ground_truth_mask: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Executes the end-to-end V8 diagnosis research pipeline:
    1. Preprocessing transformations (Original -> Wiener -> CLAHE -> 512x512).
    2. U-KAN segmentation forward pass at 768x768 -> 512x512 probability map.
    3. Deterministic candidate proposal generation (Top-3 RoIs).
    4. Multi-scale RoI extraction across 4 concentric scales + calcification enhancement.
    5. MultiScaleViTV8 joint classification (4,608-dim latent vector).
    6. ViT self-attention rollout computation.
    """
    t_start = time.time()
    device = models.device

    # 1. Preprocessing
    prep = preprocess_mammogram_full(raw_mammogram)
    orig_norm = prep["original"]
    wiener_img = prep["wiener"]
    clahe_img = prep["enhanced"]
    img_512 = prep["input_512"]
    img_768 = prep["input_768"]
    img_224 = prep["input_224"]

    # 2. U-KAN Segmentation (768x768 Receptive Field)
    t_in_768 = torch.from_numpy(img_768).unsqueeze(0).unsqueeze(0).float().to(device)
    with torch.inference_mode():
        logits_768 = models.ukan(t_in_768)
        prob_768 = torch.sigmoid(logits_768).squeeze().float().cpu().numpy()

    prob_512 = cv2.resize(prob_768, SEGMENTATION_WORKING_SIZE, interpolation=cv2.INTER_LINEAR)
    bin_mask = (prob_512 >= SEGMENTATION_THRESHOLD).astype(np.float32)

    # 3. Top-K Candidate Proposals
    proposals, raw_cc_count, fallback_flag = models.proposal_gen.generate_proposals(img_512, prob_512, k=TOP_K_PROPOSALS)
    has_reliable_roi = (not fallback_flag) and any(p.get('proposal_source') == 'machine_cc' for p in proposals)

    # 4. Multi-Scale RoI Crops & Feature Extraction
    full_t = to_vit_tensor(img_224).unsqueeze(0).to(device)
    cand_crops_dict = {}
    cand_crops_preview = []

    for prop in proposals:
        crops, _ = models.roi_gen.generate_candidate_crops(img_512, prop)
        cand_crops_preview.append(crops)

    for s in ('local', 'medium', 'large', 'context'):
        t_list = []
        for crops in cand_crops_preview:
            t_list.append(to_vit_tensor(crops[s].astype(np.float32)))
        cand_crops_dict[s] = torch.stack(t_list, dim=0).unsqueeze(0).to(device)

    # 5. Calcification Enhancement Patch
    primary_crop = cand_crops_preview[0]['local'].astype(np.float32)
    calc_patch, _ = models.roi_gen.enhance_calcification_patch(primary_crop)
    calc_patch = calc_patch.astype(np.float32)
    calc_t = to_vit_tensor(calc_patch).unsqueeze(0).to(device)

    # 6. MultiScaleViTV8 Classification Pass
    with torch.inference_mode():
        logits = models.vit(cand_crops_dict, full_image_tensor=full_t, calc_patch_tensor=calc_t)
        probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

    p_benign = float(probs[0])
    p_malignant = float(probs[1])

    if math.isnan(p_benign) or math.isnan(p_malignant):
        p_benign, p_malignant = 0.50, 0.50

    pred_class = "MALIGNANT" if p_malignant >= p_benign else "BENIGN"
    infer_time = time.time() - t_start

    # 7. ViT Self-Attention Rollout Map
    primary_local_t = to_vit_tensor(primary_crop).unsqueeze(0).to(device)
    attn_map_local = extract_vit_attention_map(models.vit.backbone, primary_local_t, device)
    attn_map_full = extract_vit_attention_map(models.vit.backbone, full_t, device)

    # 8. Optional Ground-Truth Metrics (ONLY if ground truth mask is supplied)
    dice_score = None
    iou_score = None
    if ground_truth_mask is not None:
        gt_512 = cv2.resize(ground_truth_mask, SEGMENTATION_WORKING_SIZE, interpolation=cv2.INTER_NEAREST)
        intersection = np.sum(bin_mask * gt_512)
        union = np.sum(bin_mask) + np.sum(gt_512) - intersection
        dice = (2.0 * intersection) / (np.sum(bin_mask) + np.sum(gt_512) + 1e-8)
        iou = intersection / (union + 1e-8)
        dice_score = float(dice)
        iou_score = float(iou)

    # Clear GPU Cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "model_version": models.model_version,
        "preprocessing": {
            "original": orig_norm,
            "wiener": wiener_img,
            "enhanced": clahe_img,
            "img_512": img_512,
            "img_224": img_224
        },
        "segmentation": {
            "prob_map": prob_512,
            "binary_mask": bin_mask,
            "raw_cc_count": raw_cc_count,
            "has_reliable_roi": has_reliable_roi,
            "fallback_flag": fallback_flag,
            "dice_score": dice_score,
            "iou_score": iou_score
        },
        "proposals": proposals,
        "crops": cand_crops_preview,
        "calc_patch": calc_patch,
        "classification": {
            "predicted_class": pred_class,
            "prob_benign": p_benign,
            "prob_malignant": p_malignant,
            "latency_s": infer_time
        },
        "explainability": {
            "attention_map_local": attn_map_local,
            "attention_map_full": attn_map_full,
            "primary_local_crop": primary_crop
        }
    }
