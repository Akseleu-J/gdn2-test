"""
Central configuration, numerical-safety helpers, and shape validation
shared across the forward/backward Pallas kernels.
"""
from __future__ import annotations

import contextvars
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
    wy_eps: float = 0.0          # NEW: Tikhonov damping strength, 0 = off (current behavior)
    use_centering: bool = False  # NEW: midpoint-centered decay factorization in Kernel A/B4

    @property
    def n_sub(self) -> int:
        return self.bt // self.bc

    @property
    def n_micro(self) -> int:
        return self.bc // self.mb

    def __post_init__(self):
        if self.bt % self.bc != 0:
            raise ValueError(f"bt={self.bt} must be divisible by bc={self.bc}")
        # FIX (найдено в grid_bt_bc_condition_diag.py, Часть 3 -- bc<bt/2
        # давал max|diff vs exact|=1.0 РОВНО, т.е. НЕ численную
        # неточность, а структурно нулевые блоки): wy_solve_pallas'
        # top-level solve (Kernel B, _kernel_b_body) реализует ТОЛЬКО
        # двухблочный split (T00/T11/T10) и жёстко предполагает
        # bt == 2*bc. При bc < bt/2 три четверти матрицы решения A
        # никогда не записываются и остаются нулями из инициализации --
        # это не деградация точности, а отсутствие вычисления вообще.
        # `bc` НЕ является ручкой точности/устойчивости решателя --
        # экспериментально подтверждено (grid_bt_bc_condition_diag.py,
        # Часть 3, повторный корректный прогон), что `mb` (granularity
        # внутри block-recursive forward substitution) не влияет на
        # точность решения вообще -- только на скорость. Проверяем
        # инвариант здесь, чтобы невалидный KernelConfig нельзя было
        # создать вообще, включая экспериментальные/диагностические
        # скрипты.
        if self.bt != 2 * self.bc:
            raise ValueError(
                f"bt={self.bt} must equal 2*bc (top-level WY-solve split "
                f"supports only the 2-block case); got bc={self.bc}. "
                f"Vary `mb` instead of `bc` to change solver granularity -- "
                f"bc/mb do not affect numerical accuracy of the solve, only "
                f"its speed (see grid_bt_bc_condition_diag.py Part 3)."
            )
        if self.bc % self.mb != 0:
            raise ValueError(f"bc={self.bc} must be divisible by mb={self.mb}")
        if not (0.0 <= self.wy_eps < 1.0):
            raise ValueError(f"wy_eps={self.wy_eps} must be in [0, 1)")


KAGGLE_SMALL = KernelConfig(bt=128, bc=64, mb=16, clip=1e4, wy_eps=1e-3)
KAGGLE_MEDIUM = KernelConfig(bt=256, bc=128, mb=16, clip=1e4, wy_eps=1e-3)
KAGGLE_LARGE = KernelConfig(bt=256, bc=128, mb=16, clip=5e3, wy_eps=1e-3)
DEFAULT_CONFIG = KAGGLE_MEDIUM


# FIX: sanitize()/clip_acc() were called from *inside* Pallas kernel
# bodies (_kernel_a_body, _kernel_b_body, _kernel_b3_body, _kernel_b4_body,
# gdn2_bwd.py step functions, etc.) WITHOUT passing `config`, so they
# silently always clipped to DEFAULT_CONFIG.clip=1e4 regardless of which
# KernelConfig the user actually selected (e.g. KAGGLE_LARGE's clip=5e3
# was never honored inside any kernel body -- only in the few call sites
# outside pallas_call that happened to pass config explicitly).
#
# Rather than touch every one of the ~40 call sites inside kernel bodies
# (error-prone, easy to miss one), the active clip is threaded through a
# contextvar set once, at the entry-point function (build_chunk_scores_
# pallas, wy_solve_pallas, recompute_wy_pallas, gdn2_inter_chunk_combine*,
# and the corresponding gdn2_bwd.py entry points), via `clip_scope`. This
# is safe because Pallas kernel bodies are invoked synchronously by
# Python during tracing/lowering inside the same `pl.pallas_call` call --
# there is no thread hop, so the contextvar set just before pallas_call
# is still active when the kernel body (and any nested reference/helper
# it calls, e.g. _micro_forward_substitution) executes.
_current_clip: contextvars.ContextVar = contextvars.ContextVar(
    "_gdn2_current_clip", default=None
)


class clip_scope:
    """Context manager: sets the clip bound used by sanitize()/clip_acc()
    calls that don't receive an explicit `config` (i.e. calls made from
    inside Pallas kernel bodies). Wrap every pallas_call entry point:

        with clip_scope(config):
            ... = pl.pallas_call(...)(...)
    """
    __slots__ = ("_config", "_token")

    def __init__(self, config: "KernelConfig"):
        self._config = config

    def __enter__(self):
        self._token = _current_clip.set(self._config.clip)
        return self

    def __exit__(self, *exc):
        _current_clip.reset(self._token)
        return False


def sanitize(x, config: KernelConfig = None):
    """Standard clip + nan_to_num defense used at every kernel boundary.

    If `config` is omitted (the common case for calls made from inside a
    Pallas kernel body), the clip bound comes from the nearest enclosing
    `clip_scope(config)` set by the entry-point function, falling back to
    DEFAULT_CONFIG.clip only if no scope is active (e.g. calling this
    helper directly outside any kernel entry point).
    """
    if config is not None:
        c = config.clip
    else:
        c = _current_clip.get()
        if c is None:
            c = DEFAULT_CONFIG.clip
    return jnp.nan_to_num(jnp.clip(x, -c, c), nan=0.0, posinf=c, neginf=-c)


def sanitize_h0(h0, config: KernelConfig = None):
    return sanitize(h0, config)


def clip_acc(x, config: KernelConfig = None):
    """Same as sanitize; separate name kept for call-site clarity in
    read-modify-write accumulation loops (see gdn2_bwd.py, Kernel B4)."""
    return sanitize(x, config)


def _reshape_to_chunks(t, bsz, n_chunks, H, D, bt):
    t = t.reshape(bsz, n_chunks, bt, H, D)
    return jnp.moveaxis(t, (1, 3), (2, 1))


def _reshape_from_chunks(t, bsz, n_chunks, bt, H, D):
    t2 = jnp.moveaxis(t, (1, 2, 3), (3, 1, 2))
    return t2.reshape(bsz, n_chunks * bt, H, D)


_GDN2_FWD_DIAG = os.environ.get("GDN2_FWD_DIAG", "0") == "1"
_LARGE_THRESHOLD = 1e6  # suspiciously large but still-finite trigger level


def _stage_diag(tag: str, x):
    """No-op unless GDN2_FWD_DIAG=1. Diagnostic only -- never changes x."""
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
    """Shape/dtype sanity checks shared by forward and backward entry points."""
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
