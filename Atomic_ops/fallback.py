"""
PATCH (P0.1): both entry points now pass config.wy_eps and config.clip
into gdn2_chunked_wy_reference, instead of silently taking the reference
module's defaults (wy_eps=0.0, clip=1e4). Before this patch, every
KAGGLE_* preset (wy_eps=1e-3) solved a DIFFERENT damped system on
CPU/fallback than on TPU -- meaning any correctness check, debugging
session, or CPU-only development against these entry points was silently
validating against the wrong numerical model. See test_fallback_config_plumbing.py.
"""
from __future__ import annotations
import jax.numpy as jnp
from .utils import is_tpu_available
from .gdn2_fwd import gdn2_pallas_forward as _pallas_fwd
from .gdn2_pipeline import gdn2_pallas_forward_trainable as _pallas_trainable
from .reference import gdn2_chunked_wy_reference


def gdn2_forward(q, k, v, w, b, g, scale, h0=None, config=None):
    from .configs import DEFAULT_CONFIG
    config = config or DEFAULT_CONFIG
    if is_tpu_available() and q.shape[-1] == 128:
        return _pallas_fwd(q, k, v, w, b, g, scale, h0=h0, config=config)
    return gdn2_chunked_wy_reference(
        q, k, v, g, b, w, scale, chunk_size=config.bt, h0=h0,
        wy_eps=config.wy_eps, clip=config.clip,
    )


def gdn2_forward_trainable(q, k, v, w, b, g, scale, h0=None, config=None):
    from .configs import DEFAULT_CONFIG
    config = config or DEFAULT_CONFIG
    if is_tpu_available() and q.shape[-1] == 128:
        return _pallas_trainable(q, k, v, w, b, g, scale, h0=h0, config=config)
    return gdn2_chunked_wy_reference(
        q, k, v, g, b, w, scale, chunk_size=config.bt, h0=h0,
        wy_eps=config.wy_eps, clip=config.clip,
    )
