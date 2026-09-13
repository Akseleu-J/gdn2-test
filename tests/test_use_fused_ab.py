"""
CPU-only (interpret=True) smoke test for the use_fused_ab flag patch.
Does NOT replace Gate 1 (A2) on real TPU -- interpret=True only proves
"math unchanged by the flag", same caveat gate1_wy_solve_batched.py
already states for its own (A2) section.
"""
from __future__ import annotations

import dataclasses as dc

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from Atomic_ops.configs import KernelConfig
from Atomic_ops.gdn2_fwd import gdn2_pallas_forward


def _make_inputs(seed, bsz=1, L=256, H=1, D=128):
    rng = np.random.default_rng(seed)
    shape = (bsz, L, H, D)
    q = rng.normal(size=shape).astype(np.float32) * 0.1
    k = rng.normal(size=shape).astype(np.float32) * 0.1
    v = rng.normal(size=shape).astype(np.float32) * 0.1
    w = np.ones(shape, dtype=np.float32)
    b = (0.5 * rng.uniform(0.5, 1.0, size=shape)).astype(np.float32)
    g = -np.abs(rng.normal(size=shape)).astype(np.float32) * 0.05
    return tuple(jnp.asarray(x) for x in (q, k, v, w, b, g))


def _rel_err(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.linalg.norm((a - b).ravel()) / max(np.linalg.norm(b.ravel()), 1e-12))


@pytest.fixture
def base_config():
    return KernelConfig(bt=128, bc=64, mb=16, wy_eps=1e-3, b_batch_group=2)


def test_default_flag_preserves_old_behavior(base_config):
    q, k, v, w, b, g = _make_inputs(seed=0)
    o1, h1 = gdn2_pallas_forward(q, k, v, w, b, g, scale=1.0, config=base_config, interpret=True)
    o2, h2 = gdn2_pallas_forward(q, k, v, w, b, g, scale=1.0, config=base_config,
                                  interpret=True, use_fused_ab=False)
    assert jnp.allclose(o1, o2, atol=0.0, rtol=0.0)
    assert jnp.allclose(h1, h2, atol=0.0, rtol=0.0)


def test_fused_matches_nonfused_on_cpu(base_config):
    q, k, v, w, b, g = _make_inputs(seed=1)
    o_nonfused, h_nonfused = gdn2_pallas_forward(
        q, k, v, w, b, g, scale=1.0, config=base_config,
        interpret=True, use_fused_ab=False,
    )
    o_fused, h_fused = gdn2_pallas_forward(
        q, k, v, w, b, g, scale=1.0, config=base_config,
        interpret=True, use_fused_ab=True,
    )
    err_o = _rel_err(o_fused, o_nonfused)
    err_h = _rel_err(h_fused, h_nonfused)
    assert err_o < 1e-5, f"fused vs non-fused output diverges: rel_err={err_o:.2e}"
    assert err_h < 1e-5, f"fused vs non-fused h_final diverges: rel_err={err_h:.2e}"
    assert jnp.all(jnp.isfinite(o_fused))
    assert jnp.all(jnp.isfinite(h_fused))


def test_fused_requires_explicit_group():
    config_no_group = KernelConfig(bt=128, bc=64, mb=16, wy_eps=1e-3)  # b_batch_group=None
    q, k, v, w, b, g = _make_inputs(seed=2)
    with pytest.raises(ValueError, match="b_batch_group"):
        gdn2_pallas_forward(q, k, v, w, b, g, scale=1.0, config=config_no_group,
                             interpret=True, use_fused_ab=True)


@pytest.mark.parametrize("seed,g_scale,k_scale,b_scale", [
    (10, 3.0, 1.0, 0.5),    # extreme_strong_decay
    (11, 0.3, 8.0, 4.0),    # extreme_large_kb
])
def test_fused_matches_nonfused_extreme_seeds(base_config, seed, g_scale, k_scale, b_scale):
    rng = np.random.default_rng(seed)
    shape = (1, 256, 1, 128)
    q = rng.normal(size=shape).astype(np.float32) * 0.1
    k = rng.normal(size=shape).astype(np.float32) * 0.1 * k_scale
    v = rng.normal(size=shape).astype(np.float32) * 0.1
    w = np.ones(shape, dtype=np.float32)
    b = (b_scale * rng.uniform(0.5, 1.0, size=shape)).astype(np.float32)
    g = -np.abs(rng.normal(size=shape)).astype(np.float32) * g_scale
    q, k, v, w, b, g = (jnp.asarray(x) for x in (q, k, v, w, b, g))

    o_nonfused, h_nonfused = gdn2_pallas_forward(
        q, k, v, w, b, g, scale=1.0, config=base_config, interpret=True, use_fused_ab=False,
    )
    o_fused, h_fused = gdn2_pallas_forward(
        q, k, v, w, b, g, scale=1.0, config=base_config, interpret=True, use_fused_ab=True,
    )
    assert jnp.all(jnp.isfinite(o_fused)) and jnp.all(jnp.isfinite(h_fused)), (
        f"non-finite fused output on extreme seed={seed} "
        f"(g_scale={g_scale}, k_scale={k_scale}, b_scale={b_scale})"
    )
    assert _rel_err(o_fused, o_nonfused) < 1e-4
