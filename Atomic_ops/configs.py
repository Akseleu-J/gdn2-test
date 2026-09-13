"""
Central configuration, numerical-safety helpers, and shape validation
shared across the forward/backward Pallas kernels.
"""
from __future__ import annotations

import dataclasses as dc
import os

import jax
import jax.numpy as jnp

_HIGHEST = jax.lax.Precision.HIGHEST


@dc.dataclass(frozen=True)
class KernelConfig:
    bt: int = 256
    bc: int = 128
    mb: int = 16
    clip: float = 1e4
    wy_eps: float = 0.0
    b_batch_group: int | None = None
    use_centering: bool = False
    # PATCH (P0.3, root-caused): use_centering=True requires explicit
    # opt-in again. Root cause of the leak (confirmed via sweep_centering_
    # leak.py, CPU interpret=True vs token-serial reference, and
    # independently corroborated on TPU by Gate 1 (B) in test1.py/test2.py
    # -- extreme_strong_decay: 4.26e+00, extreme_mixed_sign_g: 4.13e+00):
    #
    # The per-pair local centering in _kernel_a_body/_kernel_b4_body
    # decomposes the true decay gap into three legs and clips EACH leg
    # separately before exponentiating and multiplying:
    #     edecay = exp(clip(leg1)) * exp(clip(leg2)) * exp(clip(leg3))
    # Non-centered instead clips the SUM once:
    #     edecay = exp(clip(leg1 + leg2 + leg3))
    # These are NOT equivalent whenever any single leg approaches +-20:
    # e.g. leg1~+22 (clips to +20), leg2~-25 (clips to -20) -- true
    # combined value ~-3 (mild decay), but clip-then-multiply gives
    # exp(20)*exp(-20)=exp(0)=1, i.e. the pair is treated as UNDECAYED
    # when it should have decayed by exp(-3). This is an information
    # leak, not rounding noise.
    #
    # The "local legs span at most bc tokens (always safe to clip)"
    # assumption in the original comment is only true if bc tokens' worth
    # of cumulative decay stays under ~20. For g ~ -|N(0,1)|*g_scale,
    # expected span ~ bc*g_scale*0.798. Confirmed empirically (bc=64:
    # leak onset ~g_scale=0.3-0.5; bc=32: leak onset shifts to
    # ~g_scale=0.75-1.0, i.e. threshold scales with bc as predicted).
    # For production bc=128, this crosses 20 around g_scale~0.2 -- a
    # moderate, plausible forget-gate magnitude for a trained model, not
    # an exotic edge case.
    #
    # A correctness-preserving fix that keeps the MXU-factorization speed
    # benefit (the reason centering exists) needs a clip-consistent
    # re-derivation, not just "clip the sum instead" (that would collapse
    # back to the O(bc^2) non-centered computation and lose the speedup).
    # Until that exists and passes deep-correctness, use_centering=True
    # requires this explicit acknowledgement.
    unsafe_allow_centering: bool = False

    @property
    def n_sub(self) -> int:
        return self.bt // self.bc

    @property
    def n_micro(self) -> int:
        return self.bc // self.mb

    def __post_init__(self):
        if self.bt % self.bc != 0:
            raise ValueError(f"bt={self.bt} must be divisible by bc={self.bc}")
        if self.bt != 2 * self.bc:
            raise ValueError(
                f"bt={self.bt} must equal 2*bc (top-level WY-solve split "
                f"supports only the 2-block case); got bc={self.bc}."
            )
        if self.bc % self.mb != 0:
            raise ValueError(f"bc={self.bc} must be divisible by mb={self.mb}")
        if not (0.0 <= self.wy_eps < 1.0):
            raise ValueError(f"wy_eps={self.wy_eps} must be in [0, 1)")

        if self.use_centering and not self.unsafe_allow_centering:
            raise NotImplementedError(
                "use_centering=True requires explicit unsafe_allow_centering=True. "
                "The current per-pair centering formula has a confirmed information "
                "leak for decay magnitudes plausible in a trained model (see the long "
                "comment on KernelConfig.unsafe_allow_centering / sweep_centering_leak.py). "
                "Do not set this in a KAGGLE_* preset or in production training until "
                "the clip-consistent fix has been implemented and re-validated."
            )


# PATCH (P0.3): defaults reverted to use_centering=False pending the fix
# above. Centering presets kept available, explicitly named, for
# controlled experiments only (isolated kernel benchmarks, work on the
# fix itself) -- NOT for production training.
KAGGLE_SMALL = KernelConfig(bt=128, bc=64, mb=16, clip=1e4, wy_eps=1e-3, use_centering=False)
KAGGLE_MEDIUM = KernelConfig(bt=256, bc=128, mb=16, clip=1e4, wy_eps=1e-3, use_centering=False)
KAGGLE_LARGE = KernelConfig(bt=256, bc=128, mb=16, clip=5e3, wy_eps=1e-3, use_centering=False)
DEFAULT_CONFIG = KAGGLE_MEDIUM

# Centering variants -- CONTROLLED EXPERIMENTS ONLY. Requires
# unsafe_allow_centering=True to construct (enforced in __post_init__).
# Do not wire these into gdn2_pallas_forward_trainable's default path.
KAGGLE_SMALL_CENTERED = KernelConfig(bt=128, bc=64, mb=16, clip=1e4, wy_eps=1e-3,
                                      use_centering=True, unsafe_allow_centering=True)
KAGGLE_MEDIUM_CENTERED = KernelConfig(bt=256, bc=128, mb=16, clip=1e4, wy_eps=1e-3,
                                       use_centering=True, unsafe_allow_centering=True)
KAGGLE_LARGE_CENTERED = KernelConfig(bt=256, bc=128, mb=16, clip=5e3, wy_eps=1e-3,
                                      use_centering=True, unsafe_allow_centering=True)


def sanitize(x, config: KernelConfig = DEFAULT_CONFIG):
    c = config.clip
    return jnp.nan_to_num(jnp.clip(x, -c, c), nan=0.0, posinf=c, neginf=-c)


def sanitize_h0(h0, config: KernelConfig = DEFAULT_CONFIG):
    return sanitize(h0, config)


def clip_acc(x, config: KernelConfig = DEFAULT_CONFIG):
    return sanitize(x, config)


def _reshape_to_chunks(t, bsz, n_chunks, H, D, bt):
    t = t.reshape(bsz, n_chunks, bt, H, D)
    return jnp.moveaxis(t, (1, 3), (2, 1))


def _reshape_from_chunks(t, bsz, n_chunks, bt, H, D):
    t2 = jnp.moveaxis(t, (1, 2, 3), (3, 1, 2))
    return t2.reshape(bsz, n_chunks * bt, H, D)


_GDN2_FWD_DIAG = os.environ.get("GDN2_FWD_DIAG", "0") == "1"
_LARGE_THRESHOLD = 1e6


def _stage_diag(tag: str, x):
    if not _GDN2_FWD_DIAG:
        return x
    finite_mask = jnp.isfinite(x)
    all_finite = jnp.all(finite_mask)
    n_nonfinite = jnp.sum(jnp.logical_not(finite_mask))
    safe_x = jnp.where(finite_mask, x, 0.0)
    max_abs = jnp.max(jnp.abs(safe_x))

    def _report_nonfinite():
        jax.debug.print(
            "[GDN2-FWD-DIAG] non-finite at " + tag + ": n_nonfinite={n} max_abs_finite={m:.3e}",
            n=n_nonfinite, m=max_abs,
        )

    def _report_large():
        jax.debug.print(
            "[GDN2-FWD-DIAG] suspiciously large (still finite) at " + tag + ": max_abs={m:.3e}",
            m=max_abs,
        )

    jax.lax.cond(
        jnp.logical_not(all_finite),
        _report_nonfinite,
        lambda: jax.lax.cond(max_abs > _LARGE_THRESHOLD, _report_large, lambda: None),
    )
    return x


def validate_inputs(q, k, v, w, b, g, scale, h0, config: KernelConfig):
    if q.ndim != 4:
        raise ValueError(f"q must be (batch, seq_len, heads, d_head); got shape {q.shape}")
    bsz, L, H, D = q.shape
    if D != 128:
        raise ValueError(f"Kernels assume d_head=128 (MXU tile); got D={D}.")
    if L % config.bt != 0:
        raise ValueError(f"seq_len={L} must be divisible by config.bt={config.bt}.")

    for name, t in (("k", k), ("v", v), ("w", w), ("b", b), ("g", g)):
        if t.shape != q.shape:
            raise ValueError(f"{name}.shape={t.shape} must match q.shape={q.shape}")

    if h0 is not None:
        expected_h0 = (bsz, H, D, D)
        if h0.shape != expected_h0:
            raise ValueError(f"h0.shape={h0.shape} must be {expected_h0}")

    n_chunks = L // config.bt
    return bsz, L, H, D, n_chunks
