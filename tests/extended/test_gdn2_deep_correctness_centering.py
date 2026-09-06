"""
test_gdn2_deep_correctness_centering.py

Полный deep-correctness suite для целого пайплайна
(gdn2_pallas_forward_trainable, весь custom_vjp: Kernel A/B/C/D + B1-B5)
с use_centering=True. Дополняет изолированные тесты Kernel A/B4.
"""
from __future__ import annotations

import dataclasses as dc
import sys

import jax
import jax.numpy as jnp

from Atomic_ops.configs import KernelConfig, KAGGLE_SMALL, KAGGLE_MEDIUM
from Atomic_ops.gdn2_pipeline import gdn2_pallas_forward_trainable
from Atomic_ops.gdn2_fwd import gdn2_pallas_forward
from Atomic_ops.reference import gdn2_token_serial_reference

_HIGHEST = jax.lax.Precision.HIGHEST
_FAILURES = []


def _check(name, rel_err, tol, extra=""):
    status = "PASS" if rel_err <= tol else "FAIL"
    print(f"[{status}] {name}: rel_err={rel_err:.3e}  (tol={tol:.1e}) {extra}")
    if rel_err > tol:
        _FAILURES.append(name)
    return rel_err <= tol


def _rel_err(a, b):
    a = jnp.asarray(a, dtype=jnp.float32)
    b = jnp.asarray(b, dtype=jnp.float32)
    num = jnp.max(jnp.abs(a - b))
    den = jnp.maximum(jnp.max(jnp.abs(b)), 1e-8)
    return float(num / den)


def _centered_config(base: KernelConfig, **overrides) -> KernelConfig:
    fields = {f.name: getattr(base, f.name) for f in dc.fields(base)}
    fields.update(overrides)
    fields["use_centering"] = True
    fields["unsafe_allow_centering"] = True
    return KernelConfig(**fields)


def _make_inputs(key, bsz, n_chunks, bt, H, D, decay_scale, h0_nonzero=False, dtype=jnp.float32):
    L = n_chunks * bt
    k1, k2, k3, k4, k5 = jax.random.split(key, 5)
    shape = (bsz, L, H, D)

    q = jax.random.normal(k1, shape)
    k = jax.random.normal(k2, shape)
    q = q / (jnp.linalg.norm(q, axis=-1, keepdims=True) + 1e-6)
    k = k / (jnp.linalg.norm(k, axis=-1, keepdims=True) + 1e-6)
    v = jax.random.normal(k3, shape) * 0.5
    w = jax.random.uniform(k4, shape, minval=0.2, maxval=1.0)
    b = jax.random.uniform(jax.random.fold_in(k4, 1), shape, minval=0.2, maxval=1.0)

    if decay_scale <= 0.0:
        g = jnp.zeros(shape, dtype=jnp.float32)
    else:
        g = -jnp.abs(jax.random.normal(k5, shape)) * decay_scale

    h0 = None
    if h0_nonzero:
        h0 = jax.random.normal(jax.random.fold_in(key, 99), (bsz, H, D, D)) * 0.1

    q, k, v, w, b = (t.astype(dtype) for t in (q, k, v, w, b))
    g = g.astype(jnp.float32)
    return q, k, v, w, b, g, h0


def test_multiseed_sweep_centered(cfg):
    print("\n--- C2: multi-seed sweep, use_centering=True (fwd+bwd vs token-serial) ---")
    bt = cfg["bt"]
    config = _centered_config(KAGGLE_MEDIUM, bt=bt, bc=bt // 2, mb=min(16, bt // 2), wy_eps=0.0)
    n_chunks = 2

    for seed in range(cfg["n_seeds"]):
        key = jax.random.PRNGKey(5000 + seed)
        q, k, v, w, b, g, h0 = _make_inputs(key, cfg["bsz"], n_chunks, bt,
                                             cfg["H"], cfg["D"], decay_scale=0.15,
                                             h0_nonzero=True)

        o_pallas, h_final_pallas = gdn2_pallas_forward(q, k, v, w, b, g, scale=1.0, h0=h0, config=config)
        o_ref, h_final_ref = gdn2_token_serial_reference(q, k, v, g, b, w, scale=1.0, h0=h0)
        _check(f"centered.sweep[seed={seed}].fwd.o", _rel_err(o_pallas, o_ref), 2e-2)
        _check(f"centered.sweep[seed={seed}].fwd.h_final", _rel_err(h_final_pallas, h_final_ref), 2e-2)

        rkey = jax.random.fold_in(key, 777)
        r1, r2 = jax.random.split(rkey)
        do_rand = jax.random.normal(r1, o_pallas.shape)
        dh_rand = jax.random.normal(r2, h_final_pallas.shape)

        def honest_loss(q_, k_, v_, w_, b_, g_, h0_):
            o, hf = gdn2_pallas_forward_trainable(q_, k_, v_, w_, b_, g_, scale=1.0, h0=h0_, config=config)
            return jnp.sum(o * do_rand) + jnp.sum(hf * dh_rand)

        def ref_loss(q_, k_, v_, w_, b_, g_, h0_):
            o, hf = gdn2_token_serial_reference(q_, k_, v_, g_, b_, w_, scale=1.0, h0=h0_)
            return jnp.sum(o * do_rand) + jnp.sum(hf * dh_rand)

        hg = jax.grad(honest_loss, argnums=(0, 1, 2, 3, 4, 5, 6))(q, k, v, w, b, g, h0)
        rg = jax.grad(ref_loss, argnums=(0, 1, 2, 3, 4, 5, 6))(q, k, v, w, b, g, h0)
        for name, h, r in zip(["dq", "dk", "dv", "dw", "db", "dg", "dh0"], hg, rg):
            _check(f"centered.sweep[seed={seed}].bwd.{name}", _rel_err(h, r), 5e-2)


def test_wy_eps_damping_centered(cfg):
    print("\n--- C3: wy_eps > 0 damping, use_centering=True ---")
    bt = cfg["bt"]
    for wy_eps in (1e-3, 1e-2):
        config = _centered_config(KAGGLE_MEDIUM, bt=bt, bc=bt // 2, mb=min(16, bt // 2), wy_eps=wy_eps)
        n_chunks = 2
        key = jax.random.PRNGKey(6000 + int(wy_eps * 1e5))
        q, k, v, w, b, g, h0 = _make_inputs(key, cfg["bsz"], n_chunks, bt, cfg["H"], cfg["D"],
                                             decay_scale=0.1, h0_nonzero=True)

        o_pallas, h_final_pallas = gdn2_pallas_forward(q, k, v, w, b, g, scale=1.0, h0=h0, config=config)
        o_ref, h_final_ref = gdn2_token_serial_reference(q, k, v, g, b, w, scale=1.0, h0=h0)
        tol = max(2e-2, wy_eps * 5)
        _check(f"centered.wy_eps={wy_eps}.fwd.o", _rel_err(o_pallas, o_ref), tol)


def test_bf16_inputs_centered(cfg):
    print("\n--- C4: bf16 inputs, use_centering=True ---")
    bt = cfg["bt"]
    config = _centered_config(KAGGLE_MEDIUM, bt=bt, bc=bt // 2, mb=min(16, bt // 2), wy_eps=0.0)
    n_chunks = 2
    key = jax.random.PRNGKey(7111)
    q, k, v, w, b, g, h0 = _make_inputs(key, cfg["bsz"], n_chunks, bt, cfg["H"], cfg["D"],
                                         decay_scale=0.1, h0_nonzero=True, dtype=jnp.bfloat16)

    o_pallas, h_final_pallas = gdn2_pallas_forward_trainable(q, k, v, w, b, g, scale=1.0, h0=h0, config=config)
    o_ref, h_final_ref = gdn2_token_serial_reference(
        q.astype(jnp.float32), k.astype(jnp.float32), v.astype(jnp.float32),
        g, b.astype(jnp.float32), w.astype(jnp.float32), scale=1.0, h0=h0,
    )
    _check("centered.bf16.fwd.o_vs_fp32_token_serial",
           _rel_err(o_pallas.astype(jnp.float32), o_ref), 8e-2)


def test_kaggle_small_config_centered(cfg):
    print("\n--- C5: KAGGLE_SMALL (bt=128), use_centering=True ---")
    config = _centered_config(KAGGLE_SMALL)
    n_chunks = 2
    key = jax.random.PRNGKey(8131)
    q, k, v, w, b, g, h0 = _make_inputs(key, cfg["bsz"], n_chunks, config.bt, cfg["H"], cfg["D"],
                                         decay_scale=0.1, h0_nonzero=True)

    o_pallas, h_final_pallas = gdn2_pallas_forward(q, k, v, w, b, g, scale=1.0, h0=h0, config=config)
    o_ref, h_final_ref = gdn2_token_serial_reference(q, k, v, g, b, w, scale=1.0, h0=h0)
    tol = max(2e-2, config.wy_eps * 5)
    _check("centered.kaggle_small.fwd.o", _rel_err(o_pallas, o_ref), tol)
    _check("centered.kaggle_small.fwd.h_final", _rel_err(h_final_pallas, h_final_ref), tol)


RUN_CONFIG = dict(bsz=2, H=2, D=128, bt=256, n_seeds=5)


def main(cfg=RUN_CONFIG):
    test_multiseed_sweep_centered(cfg)
    test_wy_eps_damping_centered(cfg)
    test_bf16_inputs_centered(cfg)
    test_kaggle_small_config_centered(cfg)

    print("\n" + "=" * 78)
    if _FAILURES:
        print(f"РЕЗУЛЬТАТ: {len(_FAILURES)} провал(ов):")
        for name in _FAILURES:
            print(f"  - {name}")
        sys.exit(1)
    else:
        print("РЕЗУЛЬТАТ: ВСЕ проверки прошли -- use_centering=True корректен "
              "через весь custom_vjp пайплайн.")
        sys.exit(0)


if __name__ == "__main__":
    main()
