"""Frozen foundation-model feature extractors (DinoBloom, SubCell) used to
benchmark against the pretrained DINO-ViT embeddings in Supp. Fig. 2e/f.

The vendored model code lives under ``foundation_models/vendor/`` (see the
provenance header at the top of each vendored file). Public entry points are
the loader/preprocessing helpers in ``foundation_models.loaders``.
"""

from .loaders import (
    DINOBLOOM_S_DIM,
    SUBCELL_BG_DIM,
    build_dinobloom_s,
    build_subcell_bg,
    dinobloom_preprocess,
    subcell_preprocess,
    subcell_embed,
)
