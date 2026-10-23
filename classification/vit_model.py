import argparse
import logging
from typing import Optional, Dict, Any, Tuple
import torch
import torch.nn as nn
import timm

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")
logger = logging.getLogger("ViTModel")


class VisionTransformerClassifier(nn.Module):
    """
    Vision Transformer (ViT) Classifier tailored for Breast Cancer Malignancy Prediction (Benign vs Malignant).
    Supports extraction of intermediate multi-head self-attention maps for explainability.
    """
    def __init__(
        self,
        model_name: str = "vit_base_patch16_224",
        num_classes: int = 2,
        pretrained: bool = True,
        in_channels: int = 3,
        drop_rate: float = 0.1,
        drop_path_rate: float = 0.1
    ):
        super().__init__()
        self.model_name = model_name
        self.num_classes = num_classes

        # Attempt to create model via timm
        try:
            self.backbone = timm.create_model(
                model_name,
                pretrained=pretrained,
                in_chans=in_channels,
                num_classes=num_classes,
                drop_rate=drop_rate,
                drop_path_rate=drop_path_rate
            )
            logger.info(f"Initialized ViT model '{model_name}' (pretrained={pretrained}).")
        except Exception as e:
            logger.warning(f"Could not load pretrained weights ({e}). Initializing randomly without pretrained weights.")
            self.backbone = timm.create_model(
                model_name,
                pretrained=False,
                in_chans=in_channels,
                num_classes=num_classes,
                drop_rate=drop_rate,
                drop_path_rate=drop_path_rate
            )

        self.attn_matrices = []
        self._hooks = []
        self._register_attention_hooks()

    def _register_attention_hooks(self):
        """
        Registers forward hooks on self-attention modules across all transformer blocks.
        """
        self._clear_hooks()
        if hasattr(self.backbone, "blocks"):
            for blk in self.backbone.blocks:
                attn_mod = getattr(blk, "attn", None)
                if attn_mod is not None:
                    hook = attn_mod.register_forward_hook(self._make_attn_hook(attn_mod))
                    self._hooks.append(hook)

    def _make_attn_hook(self, attn_module):
        def hook_fn(module, input_tensor, output_tensor):
            try:
                x = input_tensor[0]
                B, N, C = x.shape
                num_heads = getattr(module, "num_heads", 12)
                head_dim = C // num_heads
                
                # Check qkv layer
                if hasattr(module, "qkv"):
                    qkv = module.qkv(x).reshape(B, N, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
                    q, k, v = qkv[0], qkv[1], qkv[2]
                    scale = getattr(module, "scale", head_dim ** -0.5)
                    attn = (q @ k.transpose(-2, -1)) * scale
                    attn = attn.softmax(dim=-1)
                    self.attn_matrices.append(attn.detach())
            except Exception as e:
                pass
        return hook_fn

    def _clear_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass yielding classification logits (Batch, NumClasses).
        """
        return self.backbone(x)

    def forward_with_attention(self, x: torch.Tensor) -> Tuple[torch.Tensor, list]:
        """
        Executes forward pass while collecting attention matrices from all transformer blocks.
        """
        self.attn_matrices = []
        logits = self.backbone(x)
        captured = list(self.attn_matrices)
        self.attn_matrices = []
        return logits, captured


def build_vit_from_config(config_dict: Optional[dict] = None) -> VisionTransformerClassifier:
    """Instantiates a ViT model according to configuration parameters."""
    if config_dict is None:
        return VisionTransformerClassifier()

    cls_cfg = config_dict.get("classification", {})
    return VisionTransformerClassifier(
        model_name=cls_cfg.get("model_name", "vit_base_patch16_224"),
        num_classes=cls_cfg.get("num_classes", 2),
        pretrained=cls_cfg.get("pretrained", True),
        in_channels=cls_cfg.get("in_channels", 3),
        drop_rate=cls_cfg.get("drop_rate", 0.1),
        drop_path_rate=cls_cfg.get("drop_path_rate", 0.1)
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test ViT model instantiation and forward pass.")
    parser.add_argument("--model", type=str, default="vit_base_patch16_224")
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()

    model = VisionTransformerClassifier(model_name=args.model, pretrained=False)
    dummy_input = torch.randn(args.batch_size, 3, 224, 224)
    
    logits, attns = model.forward_with_attention(dummy_input)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("=" * 60)
    print("VISION TRANSFORMER ARCHITECTURE VERIFICATION")
    print(f"Model Name:   {args.model}")
    print(f"Input shape:  {dummy_input.shape}")
    print(f"Logits shape: {logits.shape}")
    print(f"Attention blocks captured: {len(attns)}")
    if attns:
        print(f"Attention map shape per block: {attns[0].shape}")
    print(f"Trainable parameters: {num_params:,}")
    print("=" * 60)
