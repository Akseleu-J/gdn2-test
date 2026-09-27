"""T4: every dot-mode / bs2 parameter must change the output (else it is a phantom parameter)."""
import jax, jax.numpy as jnp, numpy as np, pytest
from atomic_gdn2 import make_cfg, make_blr_trainable
from conftest import mk_inputs


def grads(cfg, seed=0, gscale=0.2):
    args = mk_inputs(1, 128, 1, 64, seed, g=gscale, layout="bhld")
    fn = make_blr_trainable(cfg, 1 / 8.0)
    def loss(*a):
        o, hf = fn(*a); return jnp.sum(o * o) + jnp.sum(hf * hf)
    return jax.grad(loss, argnums=(0, 1, 2, 3, 4, 5))(*args)


def test_b4_and_b3_dot_modes_are_live(small_cfg):
    base = grads(small_cfg)
    for field in ("b4_dot_mode", "b3_dot_mode"):
        other = grads(small_cfg.with_(**{field: "bf16x3" if getattr(small_cfg, field) == "highest" else "highest"}))
        d = max(float(jnp.max(jnp.abs(a - b))) for a, b in zip(base, other))
        assert d > 1e-9, f"{field} has no effect -> phantom parameter"


def test_bs2_changes_result_slightly(small_cfg):
    a = grads(small_cfg.with_(bs2=32)); b = grads(small_cfg.with_(bs2=16, g_max=10.0))
    d = max(float(jnp.max(jnp.abs(x - y))) for x, y in zip(a, b))
    assert 0 < d < 1e-2
