"""
Фаза 1 fix (см. §1.2 / §4 uploaded roadmap): _kernel_ab_body_batched
переписан без _ScratchAkkRef/.at[].set()-на-значении-затем-bulk-write.

Паттерн: Aqk пишется НАПРЯМУЮ в срезы aqk_ref (в точности как в оригинальном
_kernel_a_body -- этот путь никогда не был сломан). Akk собирается как
value через list-of-blocks + jnp.concatenate (в точности как уже работает
_block_solve/_block_solve_batched) -- НЕ через .at[].set(). Итоговый A
пишется в a_ref срезами (в точности как в уже рабочем
_kernel_b_body_batched), а не одним bulk-write после .at[].set().

Ничего в МАТЕМАТИКЕ не меняется относительно оригинального
_kernel_a_body/_kernel_b_body_batched -- меняется только то, ЧЕРЕЗ КАКОЙ
Pallas-паттерн она записывается. Поэтому bit-identical со старым
(interpret=True) прототипом ожидается по построению; главная цель этого
файла -- дать версию, которая имеет разумный шанс лечь на interpret=False
(реальный Mosaic), т.к. не содержит паттерна, который отчёт связывает с
поломкой batched Kernel B.

ВАЖНО: это всё ещё требует прогона с interpret=False на реальном TPU
(§4 Фаза 1, п.2) -- локально (CPU, interpret=True) можно проверить только
"не изменилась ли математика", не "лежит ли это на Mosaic".
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from .configs import KernelConfig, DEFAULT_CONFIG, sanitize
from .gdn2_fwd import _reshape_to_chunks
from .gdn2_fwd_batched import _block_solve_batched

_HIGHEST = jax.lax.Precision.HIGHEST


def _compute_aqk_akk_single_chunk(q_full, k_full, b_full, g_raw, *, scale, bt, bc, n_sub,
                                   use_centering, config):
    """Та же математика, что _kernel_a_body, но возвращает Akk как VALUE
    (list-of-blocks + concatenate), а не пишет его в ref -- нужно для
    последующего jnp.stack по группе чанков перед батченым solve."""
    bt_idx = jnp.arange(bt)
    tril_ones_bt = (bt_idx[:, None] >= bt_idx[None, :]).astype(jnp.float32)
    gc = jnp.dot(tril_ones_bt, g_raw, precision=_HIGHEST)

    aqk_blocks = [[jnp.zeros((bc, bc), dtype=jnp.float32) for _ in range(n_sub)] for _ in range(n_sub)]
    akk_blocks = [[jnp.zeros((bc, bc), dtype=jnp.float32) for _ in range(n_sub)] for _ in range(n_sub)]

    for si in range(n_sub):
        for sj in range(si + 1):
            i0, i1 = si * bc, (si + 1) * bc
            j0, j1 = sj * bc, (sj + 1) * bc

            q_i = q_full[i0:i1]
            k_i = k_full[i0:i1]
            k_j = k_full[j0:j1]
            b_i = b_full[i0:i1]
            gc_i = gc[i0:i1]
            gc_j = gc[j0:j1]

            if use_centering:
                gn_i = gc[i0]
                gn_j = gc[j0]
                gq_i = jnp.clip(gc_i - gn_i[None, :], -20.0, 20.0)
                gk_j = jnp.clip(gn_j[None, :] - gc_j, -20.0, 20.0)
                gcross = jnp.clip(gn_i - gn_j, -20.0, 20.0)

                eq_i = jnp.exp(gq_i)
                ek_j = jnp.exp(gk_j)
                ecross = jnp.exp(gcross)

                q_scaled = q_i * eq_i
                k_scaled = (k_j * ek_j) * ecross[None, :]
                bk_scaled = (b_i * k_i) * eq_i

                aqk_blk = scale * jnp.dot(q_scaled, k_scaled.T, precision=_HIGHEST)
                akk_blk = jnp.dot(bk_scaled, k_scaled.T, precision=_HIGHEST)
            else:
                decay_diff = gc_i[:, None, :] - gc_j[None, :, :]
                edecay = jnp.exp(jnp.clip(decay_diff, -20.0, 20.0))
                tmp_q = q_i[:, None, :] * edecay * k_j[None, :, :]
                aqk_blk = scale * jnp.sum(tmp_q, axis=-1)
                bk_i = b_i * k_i
                tmp_k = bk_i[:, None, :] * edecay * k_j[None, :, :]
                akk_blk = jnp.sum(tmp_k, axis=-1)

            if si == sj:
                idx = jnp.arange(bc)
                causal = (idx[:, None] >= idx[None, :]).astype(jnp.float32)
                strict = (idx[:, None] > idx[None, :]).astype(jnp.float32)
                aqk_blk = aqk_blk * causal
                akk_blk = akk_blk * strict

            aqk_blocks[si][sj] = sanitize(aqk_blk, config)
            akk_blocks[si][sj] = sanitize(akk_blk, config)

    aqk_rows = [jnp.concatenate(row, axis=1) for row in aqk_blocks]
    akk_rows = [jnp.concatenate(row, axis=1) for row in akk_blocks]
    return jnp.concatenate(aqk_rows, axis=0), jnp.concatenate(akk_rows, axis=0)


def _kernel_ab_body_batched_fixed(q_ref, k_ref, b_ref, g_ref, aqk_ref, a_ref, *,
                                   scale: float, bt: int, bc: int, n_sub: int,
                                   use_centering: bool, config: KernelConfig, group: int):
    akk_list = []
    for gi in range(group):
        q_full = q_ref[0, 0, gi].astype(jnp.float32)
        k_full = k_ref[0, 0, gi].astype(jnp.float32)
        b_full = b_ref[0, 0, gi].astype(jnp.float32)
        g_raw = g_ref[0, 0, gi].astype(jnp.float32)

        aqk_full, akk_full = _compute_aqk_akk_single_chunk(
            q_full, k_full, b_full, g_raw, scale=scale, bt=bt, bc=bc, n_sub=n_sub,
            use_centering=use_centering, config=config,
        )
        # Прямая запись в HBM-ref -- тот же паттерн, что уже работает в
        # оригинальном _kernel_a_body. Никакого .at[].set() здесь нет.
        aqk_ref[0, 0, gi] = aqk_full
        akk_list.append(akk_full)

    Akk_group = jnp.stack(akk_list, axis=0)  # (group, bt, bt) -- value, не ref

    T00 = Akk_group[:, 0:bc, 0:bc]
    T11 = Akk_group[:, bc:2 * bc, bc:2 * bc]
    T10 = Akk_group[:, bc:2 * bc, 0:bc]
    A00 = _block_solve_batched(T00, config, group)
    A11 = _block_solve_batched(T11, config, group)
    eps = config.wy_eps
    tmp = sanitize(jnp.einsum("gij,gjk->gik", T10 * (1.0 - eps), A00, precision=_HIGHEST), config)
    A10 = sanitize(-jnp.einsum("gij,gjk->gik", A11, tmp, precision=_HIGHEST), config)

    # Прямая запись срезами в a_ref -- ТОЧНО тот же паттерн, что уже
    # доказанно рабочий в _kernel_b_body_batched (gdn2_fwd_batched.py).
    # НЕ строим промежуточный `out = zeros(...); out.at[...].set(...)`.
    a_ref[0, 0] = jnp.zeros((group, bt, bt), dtype=jnp.float32)
    a_ref[0, 0, :, 0:bc, 0:bc] = A00
    a_ref[0, 0, :, bc:2 * bc, 0:bc] = A10
    a_ref[0, 0, :, bc:2 * bc, bc:2 * bc] = A11


def build_and_solve_pallas_batched_fixed(q, k, b, g, scale, config: KernelConfig = DEFAULT_CONFIG,
                                          group: int | None = None,
                                          vmem_limit_bytes: int = 160 * 1024 * 1024,
                                          interpret: bool = False):
    """Как build_and_solve_pallas_batched (gdn2_fwd_batched.py), но без
    .at[].set()-на-значении-затем-bulk-write паттерна. Возвращает (Aqk, A);
    Akk по-прежнему не материализуется в HBM (существует только в VMEM как
    промежуточный jnp value внутри тела кернела)."""
    assert config.bt == 2 * config.bc
    bsz, L, H, D = q.shape
    n_chunks = L // config.bt
    if group is None:
        group = config.b_batch_group if config.b_batch_group is not None else n_chunks
    if n_chunks % group != 0:
        raise ValueError(f"n_chunks={n_chunks} должен делиться на group={group}")
    n_groups = n_chunks // group

    q_r, k_r, b_r, g_r = map(lambda t: _reshape_to_chunks(t, bsz, n_chunks, H, D, config.bt),
                              (q, k, b, g))

    grid = (bsz, H, n_groups)
    in_spec = pl.BlockSpec((1, 1, group, config.bt, D), lambda i, h, gi: (i, h, gi, 0, 0))
    aqk_spec = pl.BlockSpec((1, 1, group, config.bt, config.bt), lambda i, h, gi: (i, h, gi, 0, 0))
    a_spec = pl.BlockSpec((1, 1, group, config.bt, config.bt), lambda i, h, gi: (i, h, gi, 0, 0))

    aqk, A = pl.pallas_call(
        lambda *refs: _kernel_ab_body_batched_fixed(
            *refs, scale=scale, bt=config.bt, bc=config.bc, n_sub=config.n_sub,
            use_centering=config.use_centering, config=config, group=group,
        ),
        grid=grid,
        in_specs=[in_spec, in_spec, in_spec, in_spec],
        out_specs=[aqk_spec, a_spec],
        out_shape=[
            jax.ShapeDtypeStruct((bsz, H, n_chunks, config.bt, config.bt), jnp.float32),
            jax.ShapeDtypeStruct((bsz, H, n_chunks, config.bt, config.bt), jnp.float32),
        ],
        compiler_params=pltpu.CompilerParams(vmem_limit_bytes=vmem_limit_bytes),
        interpret=interpret,  # на TPU: False. interpret=True здесь -- только для CPU sanity-check.
    )(q_r, k_r, b_r, g_r)
    return aqk, A
