"""
Gate 1 (A)-equivalent for the B2 batching prototype (Atomic_ops/gdn2_bwd_batched_b2.py).
Same discipline as gate1_wy_solve_batched.py's section (A): batched vs
non-batched must be near bit-identical (same math, different Pallas
pattern), multi-seed including extreme g/k/b, checked separately from
any comparison against token-serial ground truth.

Run on CPU first (interpret=True, default here). Before wiring
dav_backward_pallas_batched into gdn2_pipeline.py's _gdn2_core_bwd,
re-run with interpret=False on real TPU -- interpret=True only proves
the math didn't change, not that it lowers on Mosaic (same caveat as
gate1_wy_solve_batched.py's (A2) section).
"""
from __future__ import annotations

import numpy as np
import jax.numpy as jnp
import pytest

from Atomic_ops.configs import KernelConfig
from Atomic_ops.gdn2_fwd import (
    build_chunk_scores_pallas, wy_solve_pallas, recompute_wy_pallas,
    gdn2_inter_chunk_combine_with_state,
)
from Atomic_ops.gdn2_bwd import dav_backward_pallas
from Atomic_ops.gdn2_bwd_batched_b2 import dav_backward_pallas_batched

INTERPRET = True

CFG_BASE = dict(bt=128, bc=64, mb=16, clip=1e4, wy_eps=1e-3)
BSZ, H, N_CHUNKS = 2, 2, 4
D = 128
L = N_CHUNKS * CFG_BASE["bt"]

SEEDS = [
    dict(name="benign_seed0", seed=0, g_scale=0.05, k_scale=1.0, b_scale=0.5),
    dict(name="extreme_strong_decay", seed=2, g_scale=3.0, k_scale=1.0, b_scale=0.5),
    dict(name="extreme_large_kb", seed=4, g_scale=0.3, k_scale=8.0, b_scale=4.0),
    dict(name="extreme_mixed_sign_g", seed=6, g_scale=1.5, k_scale=2.0, b_scale=0.7),
]

GROUP_CHOICES = [1, N_CHUNKS]


def make_inputs(seed, g_scale, k_scale, b_scale):
    rng = np.random.default_rng(seed)
    shape = (BSZ, L, H, D)
    q = rng.normal(size=shape).astype(np.float32) * 0.1
    k = rng.normal(size=shape).astype(np.float32) * 0.1 * k_scale
    v = rng.normal(size=shape).astype(np.float32) * 0.1
    w = np.ones(shape, dtype=np.float32)
    b = (b_scale * rng.uniform(0.5, 1.0, size=shape)).astype(np.float32)
    g = -np.abs(rng.normal(size=shape)).astype(np.float32) * g_scale
    return tuple(jnp.asarray(x) for x in (q, k, v, w, b, g))


def rel_err(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.linalg.norm((a - b).ravel()) / max(np.linalg.norm(b.ravel()), 1e-12))


def _build_real_residuals(q, k, v, w, b, g, scale, config):
    """Builds real Aqk/v_new/do (via a synthetic upstream cotangent) so
    the B2 comparison runs on actual forward residuals, not toy arrays --
    matching the honest-benchmarking principle already applied elsewhere
    in this codebase (full_fwd_bwd_efficiency.py's measure_all)."""
    Aqk, Akk = build_chunk_scores_pallas(q, k, b, g, scale, config, interpret=INTERPRET)
    A = wy_solve_pallas(Akk, config, interpret=INTERPRET)
    w_pseudo, u, kg, qg, gc_last = recompute_wy_pallas(
        q, k, v, w, b, g, A, config, interpret=INTERPRET,
    )
    o, h_final, h_pre_all, v_new_all = gdn2_inter_chunk_combine_with_state(
        Aqk, w_pseudo, u, kg, qg, gc_last, scale, config=config,
    )
    v_new_all = jnp.moveaxis(v_new_all, 0, 2)
    do = jnp.moveaxis(
        jnp.asarray(np.random.default_rng(999).normal(size=o.shape).astype(np.float32)) * 0.01,
        0, 0,
    )
    return Aqk, v_new_all, do


@pytest.mark.parametrize("spec", SEEDS, ids=lambda s: s["name"])
@pytest.mark.parametrize("group", GROUP_CHOICES)
def test_b2_batched_matches_nonbatched(spec, group):
    config = KernelConfig(**CFG_BASE, b_batch_group=group)
    scale = 1.0 / np.sqrt(D)
    q, k, v, w, b, g = make_inputs(spec["seed"], spec["g_scale"], spec["k_scale"], spec["b_scale"])

    Aqk, v_new, do = _build_real_residuals(q, k, v, w, b, g, scale, config)

    dAqk_nb, dv_nb = dav_backward_pallas(Aqk, v_new, do, config=config)
    dAqk_b, dv_b = dav_backward_pallas_batched(Aqk, v_new, do, config=config,
                                               group=group, interpret=INTERPRET)

    err_daqk = rel_err(dAqk_nb, dAqk_b)
    err_dv = rel_err(dv_nb, dv_b)

    assert np.all(np.isfinite(np.asarray(dAqk_b))), f"non-finite dAqk on {spec['name']}, group={group}"
    assert np.all(np.isfinite(np.asarray(dv_b))), f"non-finite dv_new on {spec['name']}, group={group}"
    assert err_daqk < 1e-4, f"dAqk diverges: rel_err={err_daqk:.2e} ({spec['name']}, group={group})"
    assert err_dv < 1e-4, f"dv_new diverges: rel_err={err_dv:.2e} ({spec['name']}, group={group})"
