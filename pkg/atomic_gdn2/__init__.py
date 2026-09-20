"""atomic_gdn2 — Gated DeltaNet-2 with Block-Local Rescaling Pallas kernels (TPU v5e)."""
from .config import BLRConfig, make_cfg, PROD
from .pipeline import make_blr_trainable
from . import domain, reference, layout
__version__ = "0.2.0"
