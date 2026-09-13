"""
Regression bundle covering Current_state.md / What_next.md "срочное" +
closed-P0 items that had no regression test attached yet:

  - Открыто #1: `_MAX_VALIDATED_GROUP = 16` устарел -> formula guard
  - Закрыто #3/#4 (P0.1/P0.2): wy_eps/clip plumbing fallback<->reference
  - Закрыто #5 (P0.3) + N3: centering gate restored, DEFAULT_CONFIG
    reverted to use_centering=False
  - N5: extreme_large_kb should no longer be nan after _sanitize_ref fix

Run: pytest tests/test_regressions_bundle.py -v
All tests run on CPU (interpret=True where relevant) -- no TPU needed.
"""
from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
import pytest

from Atomic_ops.configs import (
    KernelConfig, DEFAULT_CONFIG, KAGGLE_SMALL, KAGGLE_MEDIUM, KAGGLE_LARGE,
    KAGGLE_MEDIUM_CENTERED,
)
from Atomic_ops.fallback import gdn2_forward
from Atomic_ops.reference import gdn2_chunked_wy_reference
from Atomic_ops.gdn2_fwd_batched import _max_validated_group, wy_solve_pallas_batched


# ===========================================================================
# 1. Group guard formula (Current_state.md "срочное" #1)
# ===========================================================================
class TestGroupGuardFormula:
    @pytest.mark.parametrize("bt,expected", [
        (128, 64),
        (256, 16),   # <-- единственная реально измеренная на TPU точка
        (512, 4),
    ])
    def test_formula_matches_known_points(self, bt, expected):
        assert _max_validated_group(bt) == expected, (
            f"_max_validated_group({bt}) diverges from the MB8-measured "
            f"anchor point (bt=256 -> group=16). If this fails after an "
            f"edit, check whether the formula's bt=256 anchor changed "
            f"without a new TPU measurement to back it."
        )

    def test_at_limit_no_warning(self, recwarn):
        config = KernelConfig(bt=256, bc=128, mb=16, wy_eps=1e-3)
        Akk = jnp.zeros((1, 1, 16, config.bt, config.bt), dtype=jnp.float32)
        wy_solve_pallas_batched(Akk, config, group=16, interpret=True)
        assert not any(issubclass(w.category, RuntimeWarning) for w in recwarn.list)

    def test_over_limit_warns(self, recwarn):
        config = KernelConfig(bt=256, bc=128, mb=16, wy_eps=1e-3)
        Akk = jnp.zeros((1, 1, 32, config.bt, config.bt), dtype=jnp.float32)
        wy_solve_pallas_batched(Akk, config, group=32, interpret=True)
        assert any(issubclass(w.category, RuntimeWarning) for w in recwarn.list), (
            "group=32 > _max_validated_group(256)=16 should warn -- if this "
            "fails, the guard inside wy_solve_pallas_batched still checks "
            "against the old hardcoded constant instead of the formula."
        )


# ===========================================================================
# 2. fallback.py wy_eps/clip plumbing (P0.1/P0.2, closed -- regression guard)
# ===========================================================================
def _make_inputs(seed=0, bt=128):
    key = jax.random.PRNGKey(seed)
    ks = jax.random.split(key, 6)
    shape = (2, bt, 2, 128)
    q, k, v, w, b, g = (jax.random.normal(ks[i], shape) * 0.3 for i in range(6))
    g = -jnp.abs(g) * 0.3
    return q, k, v, w, b, g


class TestFallbackConfigPlumbing:
    @pytest.mark.parametrize("wy_eps,clip", [(1e-3, 1e4), (5e-2, 5e3)])
    def test_fallback_uses_config_wy_eps_and_clip(self, wy_eps, clip):
        config = KernelConfig(bt=128, bc=64, mb=16, wy_eps=wy_eps, clip=clip)
        q, k, v, w, b, g = _make_inputs(bt=config.bt)

        out_fallback, h_fallback = gdn2_forward(q, k, v, w, b, g, scale=1.0, config=config)
        out_ref, h_ref = gdn2_chunked_wy_reference(
            q, k, v, g, b, w, scale=1.0, chunk_size=config.bt,
            wy_eps=config.wy_eps, clip=config.clip,
        )
        assert jnp.allclose(out_fallback, out_ref, atol=0.0, rtol=0.0)
        assert jnp.allclose(h_fallback, h_ref, atol=0.0, rtol=0.0)

        wrong_eps = 0.0 if wy_eps != 0.0 else 1e-2
        out_wrong, _ = gdn2_chunked_wy_reference(
            q, k, v, g, b, w, scale=1.0, chunk_size=config.bt,
            wy_eps=wrong_eps, clip=config.clip,
        )
        assert not jnp.allclose(out_fallback, out_wrong, atol=1e-6), (
            "wy_eps has no visible effect on this input -- test cannot "
            "distinguish correct plumbing from broken plumbing; use a "
            "chunk with worse Akk conditioning."
        )

    def test_sanitize_ref_clips_not_just_nan_to_num_N5(self):
        """N5: extreme_large_kb should no longer produce nan after the
        real jnp.clip was added to _sanitize_ref (not just nan_to_num)."""
        key = jax.random.PRNGKey(1)
        ks = jax.random.split(key, 6)
        shape = (1, 128, 1, 128)
        scale_large = 50.0
        q = jax.random.normal(ks[0], shape) * 0.3
        k = jax.random.normal(ks[1], shape) * scale_large
        v = jax.random.normal(ks[2], shape) * 0.3
        w = jax.random.uniform(ks[3], shape, minval=0.5, maxval=1.0)
        b = jax.random.uniform(ks[4], shape, minval=0.5, maxval=1.0) * scale_large
        g = -jnp.abs(jax.random.normal(ks[5], shape)) * 0.1

        config = KernelConfig(bt=128, bc=64, mb=16, wy_eps=1e-3, clip=1e4)
        out, h_final = gdn2_forward(q, k, v, w, b, g, scale=1.0, config=config)

        assert jnp.all(jnp.isfinite(out)), (
            "N5 not closed: extreme_large_kb still produces nan/inf -- "
            "_sanitize_ref is not clipping w_pseudo/u the way the "
            "docstring in reference.py claims."
        )
        assert jnp.all(jnp.isfinite(h_final))
        assert jnp.max(jnp.abs(out)) <= config.clip * 10


# ===========================================================================
# 3. Centering gate (P0.3) + N3 (DEFAULT_CONFIG behavioral change)
# ===========================================================================
class TestCenteringGateAndDefault:
    def test_default_config_is_noncentered(self):
        assert DEFAULT_CONFIG.use_centering is False

    @pytest.mark.parametrize("preset", [KAGGLE_SMALL, KAGGLE_MEDIUM, KAGGLE_LARGE])
    def test_kaggle_presets_are_noncentered(self, preset):
        assert preset.use_centering is False

    def test_centering_requires_explicit_ack(self):
        with pytest.raises(NotImplementedError):
            KernelConfig(bt=256, bc=128, mb=16, wy_eps=1e-3, use_centering=True)

    def test_centering_with_ack_succeeds(self):
        cfg = KernelConfig(bt=256, bc=128, mb=16, wy_eps=1e-3,
                            use_centering=True, unsafe_allow_centering=True)
        assert cfg.use_centering is True

    def test_centered_preset_already_has_ack(self):
        assert KAGGLE_MEDIUM_CENTERED.use_centering is True
        assert KAGGLE_MEDIUM_CENTERED.unsafe_allow_centering is True
