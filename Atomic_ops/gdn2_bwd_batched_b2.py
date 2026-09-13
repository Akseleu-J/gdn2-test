"""
Hypothesis G (MB8_status_report.md "Следующие шаги" #3): apply the same
batch-axis-instead-of-grid pattern already validated bit-identical for
Kernel B to dav_backward_pallas (B2). B2 is the cheapest backward target:
two direct matmuls, no python loop over (si,sj) pairs, no scatter --
none of the risk class that broke Kernel B originally (.at[].set() not
lowering) or that B3/B4 carry (scatter writes inside a loop).

grid=(bsz,H,n_chunks) -> grid=(bsz,H), n_chunks batched inside the kernel
body via an einsum with a leading batch axis instead of a Pallas grid
dimension -- exactly the transformation MB8 already did for Kernel B.

Do NOT wire this into gdn2_pipeline.py's _gdn2_core_bwd until Gate 1 (A)
below passes on real TPU (interpret=False), per the same discipline
already applied to Kernel B and the fused A+B path.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from .configs import KernelConfig, DEFAULT_CONFIG, sanitize

_HIGHEST = jax.lax.Precision.HIGHEST


def _kernel_b2_body_batched(aqk_ref, vnew_ref, do_ref, daqk_ref, dvnew_ref,
                            *, bt: int, config: KernelConfig, group: int):
    Aqk = aqk_ref[0, 0].astype(jnp.float32)      # (group, bt, bt)
    v_new = vnew_ref[0, 0].astype(jnp.float32)   # (group, bt, D)
    do = do_ref[0, 0].astype(jnp.float32)        # (group, bt, D)

    idx = jnp.arange(bt)
    causal = (idx[:, None] >= idx[None, :]).astype(jnp.float32)

    dAqk = jnp.einsum("gid,gjd->gij", do, v_new, precision=_HIGHEST) * causal[None, :, :]
    dv_new = jnp.einsum("gij,gid->gjd", Aqk, do, precision=_HIGHEST)

    daqk_ref[0, 0] = sanitize(dAqk, config)
    dvnew_ref[0, 0] = sanitize(dv_new, config)


def dav_backward_pallas_batched(Aqk, v_new, do, config: KernelConfig = DEFAULT_CONFIG,
                                group: int | None = None, interpret: bool = False):
    """Batched version of gdn2_bwd.dav_backward_pallas.

    Aqk: (bsz, H, n_chunks, bt, bt)
    v_new, do: (bsz, H, n_chunks, bt, D)
    group: chunks fused per grid cell. None => config.b_batch_group or
           n_chunks (same convention as wy_solve_pallas_batched).
    """
    bsz, H, n_chunks, _bt, D = v_new.shape

    if group is None:
        group = config.b_batch_group if config.b_batch_group is not None else n_chunks
    if n_chunks % group != 0:
        raise ValueError(
            f"dav_backward_pallas_batched: n_chunks={n_chunks} must be "
            f"divisible by group={group}."
        )
    n_groups = n_chunks // group

    grid = (bsz, H, n_groups)
    aqk_spec = pl.BlockSpec((1, 1, group, config.bt, config.bt), lambda i, h, gi: (i, h, gi, 0, 0))
    io_spec = pl.BlockSpec((1, 1, group, config.bt, D), lambda i, h, gi: (i, h, gi, 0, 0))

    dAqk, dv_new = pl.pallas_call(
        lambda *refs: _kernel_b2_body_batched(*refs, bt=config.bt, config=config, group=group),
        grid=grid,
        in_specs=[aqk_spec, io_spec, io_spec],
        out_specs=[aqk_spec, io_spec],
        out_shape=[
            jax.ShapeDtypeStruct(Aqk.shape, jnp.float32),
            jax.ShapeDtypeStruct(v_new.shape, jnp.float32),
        ],
        compiler_params=pltpu.CompilerParams(vmem_limit_bytes=96 * 1024 * 1024),
        interpret=interpret,
    )(Aqk, v_new, do)
    return dAqk, dv_new
