import math
import argparse
import logging
from typing import List, Tuple, Optional
import numpy as np
import torch
import cv2

from classification.vit_model import VisionTransformerClassifier

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")
logger = logging.getLogger("ExplainabilityAttention")


def compute_attention_rollout(
    all_layer_attentions: List[torch.Tensor],
    discard_ratio: float = 0.9,
    head_reduction: str = "mean"
) -> np.ndarray:
    """
    Computes Attention Rollout (Abnar & Zuidema) across transformer layers.
    Quantifies the information flow from input patch tokens to the final [CLS] diagnostic token.

    Args:
        all_layer_attentions: List of attention tensors per block of shape (B, Heads, N_tokens, N_tokens).
        discard_ratio: Ratio of lowest attention weights to zero out as background noise.
        head_reduction: "mean", "min", or "max" reduction across attention heads.

    Returns:
        2D numpy array of normalized attention weights of shape (grid_h, grid_w).
    """
    if not all_layer_attentions:
        logger.warning("Empty attention layers passed to rollout. Returning uniform map.")
        return np.ones((14, 14), dtype=np.float32)

    # Initialize Rollout matrix as identity I
    # N_tokens includes [CLS] token + spatial patches
    first_layer = all_layer_attentions[0]
    batch_size, num_heads, num_tokens, _ = first_layer.shape
    
    # Work with first sample in batch
    result = torch.eye(num_tokens, device=first_layer.device)

    with torch.no_grad():
        for attn in all_layer_attentions:
            attn_sample = attn[0] # (num_heads, num_tokens, num_tokens)

            # Reduce heads
            if head_reduction == "mean":
                attn_heads = torch.mean(attn_sample, dim=0)
            elif head_reduction == "max":
                attn_heads = torch.max(attn_sample, dim=0)[0]
            elif head_reduction == "min":
                attn_heads = torch.min(attn_sample, dim=0)[0]
            else:
                attn_heads = torch.mean(attn_sample, dim=0)

            # Discard lowest attention noise
            if 0.0 < discard_ratio < 1.0:
                flat = attn_heads.view(-1)
                k = int(flat.size(0) * discard_ratio)
                if k > 0:
                    val, _ = torch.kthvalue(flat, k)
                    attn_heads = torch.where(attn_heads < val, torch.zeros_like(attn_heads), attn_heads)

            # Add identity matrix to account for residual connections
            I = torch.eye(num_tokens, device=attn.device)
            A_prime = 0.5 * attn_heads + 0.5 * I
            # Normalize rows to sum to 1
            A_prime = A_prime / (A_prime.sum(dim=-1, keepdim=True) + 1e-8)

            # Recursive matrix multiplication
            result = torch.matmul(A_prime, result)

    # Extract attention from [CLS] token (index 0) to all spatial patch tokens (indices 1 to N)
    cls_attention = result[0, 1:].cpu().numpy()

    # Calculate patch grid dimension: sqrt(N_patches)
    num_patches = len(cls_attention)
    grid_size = int(math.isqrt(num_patches))
    
    if grid_size * grid_size != num_patches:
        # Pad or trim if irregular
        grid_size = int(math.ceil(math.sqrt(num_patches)))
        padded = np.zeros(grid_size * grid_size, dtype=np.float32)
        padded[:num_patches] = cls_attention
        attention_map = padded.reshape(grid_size, grid_size)
    else:
        attention_map = cls_attention.reshape(grid_size, grid_size)

    # Normalize to [0, 1]
    attn_min, attn_max = np.min(attention_map), np.max(attention_map)
    if attn_max > attn_min:
        attention_map = (attention_map - attn_min) / (attn_max - attn_min)
    else:
        attention_map = np.zeros_like(attention_map)

    return attention_map.astype(np.float32)


def extract_vit_attention(
    model: VisionTransformerClassifier,
    roi_image: np.ndarray,
    device: Optional[torch.device] = None,
    target_size: Tuple[int, int] = (224, 224),
    discard_ratio: float = 0.9
) -> np.ndarray:
    """
    Extracts self-attention rollout heatmap from ViT for a given lesion ROI image.

    Args:
        model: ViT model instance.
        roi_image: 2D numpy array [0, 1] representing the lesion ROI.
        device: Torch compute device.
        target_size: Desired 2D output resolution matching ROI (e.g., 224, 224).
        discard_ratio: Ratio for noise reduction in rollout.

    Returns:
        High-resolution 2D attention map in range [0, 1] interpolated to target_size.
    """
    if device is None:
        device = next(model.parameters()).device

    # 1. Prepare tensor (1, 3, 224, 224)
    roi_resized = cv2.resize(roi_image, (224, 224), interpolation=cv2.INTER_AREA)
    roi_3ch = np.stack([roi_resized, roi_resized, roi_resized], axis=0).astype(np.float32)
    tensor_roi = torch.from_numpy(roi_3ch).float()
    
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    tensor_roi = ((tensor_roi - mean) / std).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        _, attn_layers = model.forward_with_attention(tensor_roi)

    if not attn_layers:
        logger.warning("No attention weights captured from ViT backbone.")
        return np.ones(target_size, dtype=np.float32) * 0.5

    # 2. Compute rollout
    coarse_map = compute_attention_rollout(attn_layers, discard_ratio=discard_ratio)

    # 3. Interpolate coarse patch grid (e.g. 14x14) to high-res ROI resolution (e.g. 224x224) using bi-cubic
    high_res_map = cv2.resize(coarse_map, target_size, interpolation=cv2.INTER_CUBIC)
    
    # Clip and renormalize
    high_res_map = np.clip(high_res_map, 0.0, 1.0)
    h_min, h_max = float(np.min(high_res_map)), float(np.max(high_res_map))
    if h_max > h_min:
        high_res_map = (high_res_map - h_min) / (h_max - h_min)

    return high_res_map.astype(np.float32)
