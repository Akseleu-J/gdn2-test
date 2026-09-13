"""
Batched Kernel B: same math as gdn2_fwd.wy_solve_pallas, but with an
extra batch axis (group) inside the Pallas kernel. This is the version
validated on MB8 (45.0ms -> 7.94ms on TPU, group=16, train_shape) --
see MB8_status_report.md.

Contract (must match gdn2_fwd.py):
    wy_solve_pallas_batched(Akk, config, group=None, interpret=False)
        Akk: (bsz, H, n_chunks, bt, bt)  -- RAW (undamped) matrix.
        Returns A with the same shape.

    _block_solve_batched(T_full, config, group)
        T_full: (group, N, N) -- batched, N = config.bc.
        Returns (group, N, N).

Math parity with the non-batched path is bit-for-bit by construction
(same operations, same order; only the axis layout changes). See
test_gate1_wy_solve_batched.py for the CPU regression test.
"""
from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from .configs import KernelConfig, DEFAULT_CONFIG, sanitize

_HIGHEST = jax.lax.Precision.HIGHEST

# Largest `group` for which VMEM safety has actually been measured
# (train_shape, MB8_status_report.md). This is NOT a hard ceiling --
# it's the highest value that has been *validated on a real TPU*. If you
# set b_batch_group explicitly and exceed this, you get a RuntimeWarning
# (not an assert) -- you are on your own to check vmem_limit_bytes.
def _max_validated_group(bt: int) -> int:
    """Largest `group` for which VMEM safety has actually been measured,
    as a FUNCTION of `bt` (Current_state.md "срочное" #1 -- replaces the
    single hardcoded constant that goes stale whenever bt changes). The
    one real anchor point is bt=256 -> group=16 (MB8_status_report.md,
    train_shape). The rest follows the same group*bt*bt ~ const scaling
    implied by the report's own VMEM-OOM curve (n_chunks=64, group=128
    OOMs at bt=256). This is a WARNING threshold, not a hard assert --
    same semantics as the old constant, just shape-aware now.
    """
    return max(1, (64 * 128 * 128) // (bt * bt))


# Kept for anyone importing the old name directly; now derived from the
# formula above at its one TPU-measured anchor point (bt=256), not
# hand-maintained as a separate literal.
_MAX_VALIDATED_GROUP = _max_validated_group(256)


# ---------- batched micro forward-substitution ----------
def _micro_forward_substitution_batched(T_mb, mb: int, eps: float, config: KernelConfig, group: int):
    """T_mb: (group, mb, mb). Returns (group, mb, mb). Bit-for-bit the
    same recursion as gdn2_fwd._micro_forward_substitution, applied
    independently per batch element."""
    idx = jnp.arange(mb)
    T_mb = T_mb * (1.0 - eps)

    def body(i, A):
        onehot_i = (idx == i).astype(jnp.float32)              # (mb,)
        # t_row[g, :] = row i of T_mb[g]
        t_row = jnp.sum(T_mb * onehot_i[None, :, None], axis=1)  # (group, mb)
        # contrib[g, :] = t_row[g] @ A[g]
        contrib = jnp.sum(t_row[:, :, None] * A, axis=1)         # (group, mb)
        new_row = onehot_i[None, :] - contrib                    # (group, mb)
        new_row = sanitize(new_row, config)
        mask_col = onehot_i[:, None]                             # (mb, 1)
        A = A * (1.0 - mask_col[None, :, :]) + mask_col[None, :, :] * new_row[:, None, :]
        return A

    A0 = jnp.zeros((group, mb, mb), dtype=jnp.float32)
    return jax.lax.fori_loop(0, mb, body, A0)


# ---------- batched block solve ----------
def _block_solve_batched(T_full, config: KernelConfig, group: int):
    """T_full: (group, N, N). Same block-recursive WY solve as
    gdn2_fwd._block_solve, batched over the leading axis."""
    N_MICRO = config.n_micro
    MB = config.mb
    eps = config.wy_eps
    blocks = [[None] * N_MICRO for _ in range(N_MICRO)]

    for m in range(N_MICRO):
        T_mm = T_full[:, m * MB:(m + 1) * MB, m * MB:(m + 1) * MB]      # (group, MB, MB)
        A_mm = sanitize(
            _micro_forward_substitution_batched(T_mm, MB, eps, config, group),
            config,
        )
        blocks[m][m] = A_mm

        for n in range(m - 1, -1, -1):
            acc = jnp.zeros((group, MB, MB), dtype=jnp.float32)
            for k in range(n, m):
                T_mk = T_full[:, m * MB:(m + 1) * MB, k * MB:(k + 1) * MB]
                A_kn = blocks[k][n]
                contrib = jnp.einsum(
                    "gij,gjk->gik", T_mk * (1.0 - eps), A_kn, precision=_HIGHEST,
                )
                acc = sanitize(acc + contrib, config)
            A_mn = -jnp.einsum("gij,gjk->gik", A_mm, acc, precision=_HIGHEST)
            A_mn = sanitize(A_mn, config)
            blocks[m][n] = A_mn

    rows = []
    for m in range(N_MICRO):
        row_blocks = []
        for n in range(N_MICRO):
            if n > m:
                row_blocks.append(jnp.zeros((group, MB, MB), dtype=jnp.float32))
            else:
                row_blocks.append(blocks[m][n])
        rows.append(jnp.concatenate(row_blocks, axis=2))
    return jnp.concatenate(rows, axis=1)


# ---------- batched Kernel B body ----------
def _kernel_b_body_batched(akk_ref, a_ref, *, bt: int, bc: int, config: KernelConfig, group: int):
    # Same bt == 2*bc invariant as the non-batched path.
    assert bt == 2 * bc, (
        f"Kernel B (batched) поддерживает только двухблочный top-level split "
        f"(bt == 2*bc); получено bt={bt}, bc={bc}."
    )
    # akk_ref block: (group, bt, bt)
    Akk = akk_ref[0, 0].astype(jnp.float32)
    T00 = Akk[:, 0:bc, 0:bc]
    T11 = Akk[:, bc:2 * bc, bc:2 * bc]
    T10 = Akk[:, bc:2 * bc, 0:bc]

    A00 = _block_solve_batched(T00, config, group)
    A11 = _block_solve_batched(T11, config, group)

    # Same single (1-eps) damping point as the non-batched path -- see
    # the long NOTE in gdn2_fwd._kernel_b_body. Do not pre-damp Akk
    # before calling wy_solve_pallas_batched.
    eps = config.wy_eps
    tmp = sanitize(
        jnp.einsum("gij,gjk->gik", T10 * (1.0 - eps), A00, precision=_HIGHEST),
        config,
    )
    A10 = sanitize(-jnp.einsum("gij,gjk->gik", A11, tmp, precision=_HIGHEST), config)

    # Write slices into a_ref exactly the way the working non-batched
    # _kernel_b_body does. Do NOT build a separate `out` tensor and
    # bulk-write -- that pattern is what the MB8 report associates with
    # the batched Kernel B failing to lower on Mosaic.
    a_ref[0, 0] = jnp.zeros((group, bt, bt), dtype=jnp.float32)
    a_ref[0, 0, :, 0:bc, 0:bc] = A00
    a_ref[0, 0, :, bc:2 * bc, 0:bc] = A10
    a_ref[0, 0, :, bc:2 * bc, bc:2 * bc] = A11


# ---------- public entry point ----------
def wy_solve_pallas_batched(Akk, config: KernelConfig = DEFAULT_CONFIG,
                            group: int | None = None,
                            interpret: bool = False):
    """Batched version of gdn2_fwd.wy_solve_pallas.

    Akk: (bsz, H, n_chunks, bt, bt) -- RAW, undamped matrix.
    group: how many consecutive n_chunks to fuse into one VMEM batch.
           None => config.b_batch_group if set, else n_chunks (i.e. one
           group for the whole sequence). n_chunks must be divisible by
           group.
    interpret: forwarded to pl.pallas_call. Use True only on CPU.
    """
    bsz, H, n_chunks = Akk.shape[:3]

    assert config.bt == 2 * config.bc, (
        f"wy_solve_pallas_batched: bt должен быть == 2*bc, получено "
        f"bt={config.bt}, bc={config.bc}."
    )

    if group is None:
        group = config.b_batch_group if config.b_batch_group is not None else n_chunks

    if group > _MAX_VALIDATED_GROUP:
        warnings.warn(
            f"wy_solve_pallas_batched: group={group} превышает диапазон, "
            f"измеренный в MB8_status_report.md (до group={_MAX_VALIDATED_GROUP} "
            f"на train_shape). При n_chunks=64 группа=128 упала в VMEM OOM "
            f"в исходном эксперименте -- задайте b_batch_group явно и "
            f"подберите под ваш vmem_limit_bytes, не полагайтесь на дефолт "
            f"group=n_chunks для длинных последовательностей.",
            RuntimeWarning,
        )

    if n_chunks % group != 0:
        raise ValueError(
            f"wy_solve_pallas_batched: n_chunks={n_chunks} должен делиться "
            f"на group={group} (жёсткая проверка -- группа всегда покрывает "
            f"ровное число чанков)."
        )
    n_groups = n_chunks // group

    grid = (bsz, H, n_groups)
    spec = pl.BlockSpec(
        (1, 1, group, config.bt, config.bt),
        lambda i, h, g: (i, h, g, 0, 0),
    )

    A = pl.pallas_call(
        lambda *refs: _kernel_b_body_batched(
            *refs, bt=config.bt, bc=config.bc, config=config, group=group,
        ),
        grid=grid,
        in_specs=[spec],
        out_specs=spec,
        out_shape=jax.ShapeDtypeStruct(Akk.shape, jnp.float32),
        compiler_params=pltpu.CompilerParams(vmem_limit_bytes=96 * 1024 * 1024),
        interpret=interpret,
    )(Akk)
    return A
