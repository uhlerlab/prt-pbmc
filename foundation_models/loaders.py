"""Loaders + preprocessing for the frozen foundation-model feature extractors.

Both models are used *frozen* (no finetuning): we only run them as feature
extractors on our single-channel 32x32 chromatin crops, save the embeddings,
and train the usual leave-one-plate-out ``FeatClassifier`` on top (mirroring the
pretrained DINO-ViT path in Supp. Fig. 2e/f).

Model code is vendored under ``foundation_models/vendor/`` (see provenance
headers there). Run in the dedicated ``pbmc5-fm`` conda env (SubCell needs
``transformers==4.45.1``; DinoBloom uses the vendored DINOv2, which is py3.9
compatible unlike the current torch.hub DINOv2).

Preprocessing (verified against each model's own source):

* DinoBloom-S (DINOv2 ViT-S/14, CLS token, 384-d): grayscale -> 3ch, resize to
  224 (bicubic), ImageNet mean/std. Fed the raw crop (like the existing DINO-ViT
  path) so only the extractor differs.
* SubCell `bg` (2-channel DNA+Protein ViT, gated-attention pool, 1536-d): mask
  the nucleus, then place it into a 640x640 frame via *integer* bilinear
  upsampling (k=10 -> nucleus channel ~33%, k=20 -> protein channel 66%, an exact
  32->640 upsample), zero-padded and centered, then global min-max to [0,1]
  (matches SubCellPortable ``min_max_norm_fn``). Channel order is (b=nuclei,
  g=protein) per the `bg` model; we duplicate our single chromatin stain into
  both, at the two scales, to emulate a compact nucleus inside a larger cell body.
"""

import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

_HERE = os.path.dirname(os.path.abspath(__file__))
_VENDOR = os.path.join(_HERE, "vendor")
if _VENDOR not in sys.path:
    sys.path.insert(0, _VENDOR)

# ---------------------------------------------------------------------------
# DinoBloom-S (DINOv2 ViT-S/14)
# ---------------------------------------------------------------------------
DINOBLOOM_S_DIM = 384
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def build_dinobloom_s(ckpt_path, device="cuda:0"):
    """Build a frozen DinoBloom-S encoder from the Zenodo checkpoint.

    Construction matches the DINOv2 hub ``dinov2_vits14`` factory; the DinoBloom
    weights (a DINOv2 SSL checkpoint) live under the ``teacher.backbone.`` prefix
    and were trained with a 224-input positional embedding (257 tokens).
    """
    from dinov2.models import vision_transformer as vits

    model = vits.vit_small(
        img_size=518,
        patch_size=14,
        init_values=1.0,
        ffn_layer="mlp",
        block_chunks=0,
        num_register_tokens=0,
        interpolate_antialias=False,
        interpolate_offset=0.1,
    )
    ckpt = torch.load(ckpt_path, map_location="cpu")["teacher"]
    state = {k.replace("backbone.", ""): v for k, v in ckpt.items() if k.startswith("backbone.")}
    model.pos_embed = nn.Parameter(state["pos_embed"])  # (1, 257, 384)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"DinoBloom load mismatch: missing={missing} unexpected={unexpected}")
    return model.to(device).eval()


_dinobloom_resize = T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC, antialias=True)
_dinobloom_norm = T.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD)


def dinobloom_preprocess(imgs):
    """(B,1,32,32) or (B,32,32) float [0,1] -> (B,3,224,224) ImageNet-normalized."""
    if imgs.dim() == 3:
        imgs = imgs[:, None]
    x = imgs.repeat(1, 3, 1, 1)          # grayscale -> RGB
    x = _dinobloom_resize(x)             # 32 -> 224 bicubic
    return _dinobloom_norm(x)


# ---------------------------------------------------------------------------
# SubCell `bg` (2-channel DNA-Protein ViT, gated-attention pool)
# ---------------------------------------------------------------------------
SUBCELL_BG_DIM = 1536

# `bg` (vit_supcon_model) config, copied from SubCellPortable
# models/bg/vit_supcon_model/model_config.yaml.
_SUBCELL_BG_CONFIG = {
    "vit_model": {
        "hidden_size": 768,
        "num_hidden_layers": 12,
        "num_attention_heads": 12,
        "intermediate_size": 3072,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.0,
        "attention_probs_dropout_prob": 0.0,
        "initializer_range": 0.02,
        "layer_norm_eps": 1e-12,
        "image_size": 448,
        "patch_size": 16,
        "num_channels": 2,
        "qkv_bias": True,
    },
    "pool_model": {"dim": 768, "int_dim": 512, "num_heads": 2},
    "num_classes": 31,
}


def build_subcell_bg(encoder_path, device="cuda:0"):
    """Build the frozen SubCell `bg` encoder (embeddings only, no classifier)."""
    from subcell_vit import ViTPoolClassifier

    model = ViTPoolClassifier(_SUBCELL_BG_CONFIG)
    model.load_model_dict(encoder_path, [])  # empty classifier list -> embeddings only
    return model.to(device).eval()


@torch.no_grad()
def subcell_embed(model, x):
    """Return the (B, 1536) gated-attention-pool embedding for input x.

    Calls the encoder with ``output_attentions=False``; the vendored
    ``ViTPoolClassifier.forward`` hardcodes ``output_attentions=True`` (to build
    attention maps we don't need), which materializes every layer's
    1601x1601 attention and OOMs at 640x640. This reproduces exactly the pooled
    embedding it returns as ``pool_op``, without that memory cost.
    """
    outputs = model.encoder(x, output_attentions=False, interpolate_pos_encoding=True)
    pool_op, _ = model.pool_model(outputs.last_hidden_state)
    return pool_op


def subcell_preprocess(imgs, masks, k_nuc=10, k_prot=20, canvas=640):
    """Compose our chromatin crop into SubCell's 2-channel 640x640 input.

    imgs/masks: (B,1,32,32) or (B,32,32), float [0,1] image and binary mask.
    Returns (B,2,canvas,canvas) with ch0=b(nuclei, x``k_nuc``) and
    ch1=g(protein, x``k_prot``), global min-max normalized per image.
    """
    if imgs.dim() == 4:
        imgs = imgs[:, 0]
    if masks.dim() == 4:
        masks = masks[:, 0]
    masked = imgs * masks  # (B,32,32), zero background

    def place(x, k):
        up = F.interpolate(x[:, None], scale_factor=k, mode="bilinear", align_corners=False)[:, 0]
        s = up.shape[-1]
        out = torch.zeros(x.shape[0], canvas, canvas, dtype=up.dtype, device=up.device)
        o = (canvas - s) // 2
        out[:, o:o + s, o:o + s] = up
        return out

    x = torch.stack([place(masked, k_nuc), place(masked, k_prot)], dim=1).clamp(0, 1)
    b = x.shape[0]
    flat = x.reshape(b, -1)
    mn = flat.min(1)[0].view(b, 1, 1, 1)
    mx = flat.max(1)[0].view(b, 1, 1, 1)
    return (x - mn) / (mx - mn + 1e-8)
