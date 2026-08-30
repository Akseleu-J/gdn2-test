"""
Forward kernels: A (scores) -> B (WY solve) -> C (recompute) -> D (inter-chunk scan).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from .configs import (
    KernelConfig, DEFAULT_CONFIG, sanitize, sanitize_h0,
    _stage_diag, validate_inputs,
)

_HIGHEST = jax.lax.Precision.HIGHEST


# ---------- reshape helpers ----------
def _reshape_to_chunks(t: jnp.ndarray, bsz: int, n_chunks: int, H: int, D: int, bt: int) -> jnp.ndarray:
    t = t.reshape(bsz, n_chunks, bt, H, D)
    return jnp.moveaxis(t, (1, 3), (2, 1))


def _reshape_from_chunks(t: jnp.ndarray, bsz: int, n_chunks: int, bt: int, H: int, D: int) -> jnp.ndarray:
    t2 = jnp.moveaxis(t, (1, 2, 3), (3, 1, 2))
    return t2.reshape(bsz, n_chunks * bt, H, D)


# ---------- Kernel A ----------
def _weighted_pair_sum(a_i, edecay, b_j):
    tmp = a_i[:, None, :] * edecay
    tmp = tmp * b_j[None, :, :]
    return jnp.sum(tmp, axis=-1)


def _kernel_a_body(q_ref, k_ref, b_ref, g_ref, aqk_ref, akk_ref, *, scale: float, bt: int, bc: int, n_sub: int, use_centering: bool):
    q_full = q_ref[0, 0, 0].astype(jnp.float32)
    k_full = k_ref[0, 0, 0].astype(jnp.float32)
    b_full = b_ref[0, 0, 0].astype(jnp.float32)
    g_raw = g_ref[0, 0, 0].astype(jnp.float32)

    bt_idx = jnp.arange(bt)
    tril_ones_bt = (bt_idx[:, None] >= bt_idx[None, :]).astype(jnp.float32)
    gc = jnp.dot(tril_ones_bt, g_raw, precision=_HIGHEST)

    aqk_ref[0, 0, 0] = jnp.zeros((bt, bt), dtype=jnp.float32)
    akk_ref[0, 0, 0] = jnp.zeros((bt, bt), dtype=jnp.float32)

    if use_centering:
        n_mid = bt // 2
        gn = gc[n_mid]   # (D,), shared reference point for the whole BT chunk

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
                gq_i = jnp.clip(gc_i - gn[None, :], -20.0, 20.0)
                gk_j = jnp.clip(gn[None, :] - gc_j, -20.0, 20.0)
                eq_i = jnp.exp(gq_i)
                ek_j = jnp.exp(gk_j)
                q_scaled = q_i * eq_i
                k_scaled = k_j * ek_j
                bk_scaled = (b_i * k_i) * eq_i
                aqk_blk = scale * jnp.dot(q_scaled, k_scaled.T, precision=_HIGHEST)
                akk_blk = jnp.dot(bk_scaled, k_scaled.T, precision=_HIGHEST)
            else:
                decay_diff = gc_i[:, None, :] - gc_j[None, :, :]
                edecay = jnp.exp(jnp.clip(decay_diff, -20.0, 20.0))
                aqk_blk = scale * _weighted_pair_sum(q_i, edecay, k_j)
                bk_i = b_i * k_i
                akk_blk = _weighted_pair_sum(bk_i, edecay, k_j)

            if si == sj:
                idx = jnp.arange(bc)
                causal = (idx[:, None] >= idx[None, :]).astype(jnp.float32)
                strict = (idx[:, None] > idx[None, :]).astype(jnp.float32)
                aqk_blk = aqk_blk * causal
                akk_blk = akk_blk * strict

            aqk_ref[0, 0, 0, i0:i1, j0:j1] = sanitize(aqk_blk)
            akk_ref[0, 0, 0, i0:i1, j0:j1] = sanitize(akk_blk)


def build_chunk_scores_pallas(q, k, b, g, scale, config: KernelConfig = DEFAULT_CONFIG, interpret: bool = False):
    bsz, L, H, D = q.shape
    n_chunks = L // config.bt
    q_r, k_r, b_r, g_r = map(
        lambda t: _reshape_to_chunks(t, bsz, n_chunks, H, D, config.bt),
        (q, k, b, g),
    )
    grid = (bsz, H, n_chunks)
    in_spec = pl.BlockSpec((1, 1, 1, config.bt, D), lambda i, h, c: (i, h, c, 0, 0))
    out_spec = pl.BlockSpec((1, 1, 1, config.bt, config.bt), lambda i, h, c: (i, h, c, 0, 0))

    aqk, akk = pl.pallas_call(
        lambda *refs: _kernel_a_body(
            *refs, scale=scale, bt=config.bt, bc=config.bc, n_sub=config.n_sub,
            use_centering=config.use_centering,
        ),
        grid=grid,
        in_specs=[in_spec, in_spec, in_spec, in_spec],
        out_specs=[out_spec, out_spec],
        out_shape=[
            jax.ShapeDtypeStruct((bsz, H, n_chunks, config.bt, config.bt), jnp.float32),
            jax.ShapeDtypeStruct((bsz, H, n_chunks, config.bt, config.bt), jnp.float32),
        ],
        compiler_params=pltpu.CompilerParams(vmem_limit_bytes=100 * 1024 * 1024),
        interpret=interpret,
    )(q_r, k_r, b_r, g_r)
    return aqk, akk


# ---------- Kernel B ----------
def _micro_forward_substitution(T_mb, mb: int, eps: float):
    idx = jnp.arange(mb)
    T_mb = T_mb * (1.0 - eps)

    def body(i, A):
        onehot_i = (idx == i).astype(jnp.float32)
        t_row = jnp.sum(T_mb * onehot_i[:, None], axis=0)
        contrib = jnp.sum(t_row[:, None] * A, axis=0)
        new_row = onehot_i - contrib
        new_row = sanitize(new_row)
        mask_col = onehot_i[:, None]
        A = A * (1.0 - mask_col) + mask_col * new_row[None, :]
        return A

    A0 = jnp.zeros((mb, mb), dtype=jnp.float32)
    return jax.lax.fori_loop(0, mb, body, A0)


def _block_solve(T_full, config: KernelConfig):
    N_MICRO = config.n_micro
    MB = config.mb
    eps = config.wy_eps
    blocks = [[None] * N_MICRO for _ in range(N_MICRO)]

    for m in range(N_MICRO):
        T_mm = T_full[m * MB:(m + 1) * MB, m * MB:(m + 1) * MB]
        A_mm = sanitize(_micro_forward_substitution(T_mm, MB, eps))
        blocks[m][m] = A_mm

        for n in range(m - 1, -1, -1):
            acc = jnp.zeros((MB, MB), dtype=jnp.float32)
            for k in range(n, m):
                T_mk = T_full[m * MB:(m + 1) * MB, k * MB:(k + 1) * MB]
                A_kn = blocks[k][n]
                contrib = jnp.dot(T_mk * (1.0 - eps), A_kn, precision=_HIGHEST)
                acc = sanitize(acc + contrib)
            A_mn = -jnp.dot(A_mm, acc, precision=_HIGHEST)
            A_mn = sanitize(A_mn)
            blocks[m][n] = A_mn

    rows = []
    for m in range(N_MICRO):
        row_blocks = []
        for n in range(N_MICRO):
            if n > m:
                row_blocks.append(jnp.zeros((MB, MB), dtype=jnp.float32))
            else:
                row_blocks.append(blocks[m][n])
        rows.append(jnp.concatenate(row_blocks, axis=1))
    return jnp.concatenate(rows, axis=0)


def _kernel_b_body(akk_ref, a_ref, *, bt: int, bc: int, config: KernelConfig):
    # FIX (найдено в grid_bt_bc_condition_diag.py, Часть 3 -- bc<bt/2
    # давал max|diff vs exact|=1.0 РОВНО, т.е. НЕ численную неточность,
    # а структурно нулевые блоки): этот top-level 2x2 T00/T11/T10 split
    # ЖЁСТКО предполагает bt == 2*bc (унаследовано из kernel_b_solve.py,
    # где это было явным assert'ом -- см. его докстринг "N_SUB=BT//BC=2";
    # assert потерялся при переносе в этот config-driven файл). При
    # bc < bt/2 блоки A[2*bc:, :] / A[:, 2*bc:] никогда не записываются
    # и остаются нулями из инициализации ниже. `bc` здесь -- ТОЛЬКО
    # размер top-level половины chunk'а, не тонкая настройка точности
    # решателя (эта роль -- у config.mb внутри _block_solve). Экспери-
    # ментально подтверждено (та же диагностика, Часть 3, корректный
    # повторный прогон): mb не влияет на точность решения вообще, только
    # на скорость -- поэтому единственный валидный способ варьировать
    # решатель это mb, не bc.
    assert bt == 2 * bc, (
        f"Kernel B поддерживает только двухблочный top-level split "
        f"(bt == 2*bc); получено bt={bt}, bc={bc}. Для варьирования "
        f"granularity решателя используйте config.mb, а не config.bc."
    )
    Akk = akk_ref[0, 0, 0].astype(jnp.float32)
    T00 = Akk[0:bc, 0:bc]
    T11 = Akk[bc:2*bc, bc:2*bc]
    T10 = Akk[bc:2*bc, 0:bc]

    A00 = _block_solve(T00, config)
    A11 = _block_solve(T11, config)

    eps = config.wy_eps
    tmp = jnp.dot(T10 * (1.0 - eps), A00, precision=_HIGHEST)
    tmp = sanitize(tmp)
    A10 = -jnp.dot(A11, tmp, precision=_HIGHEST)
    A10 = sanitize(A10)

    a_ref[0, 0, 0] = jnp.zeros((bt, bt), dtype=jnp.float32)
    a_ref[0, 0, 0, 0:bc, 0:bc] = A00
    a_ref[0, 0, 0, bc:2*bc, 0:bc] = A10
    a_ref[0, 0, 0, bc:2*bc, bc:2*bc] = A11


def wy_solve_pallas(Akk, config: KernelConfig = DEFAULT_CONFIG):
    bsz, H, n_chunks = Akk.shape[:3]
    # FIX: та же проверка, что и внутри _kernel_b_body -- дублируется
    # здесь намеренно, чтобы падать ДО трейсинга/компиляции Pallas-кернела
    # (при jit/vmap ошибка внутри _kernel_b_body может всплыть менее
    # прозрачно). configs.py's KernelConfig.__post_init__ также проверяет
    # этот инвариант при СОЗДАНИИ конфига -- это третья, самая ранняя
    # линия защиты.
    assert config.bt == 2 * config.bc, (
        f"wy_solve_pallas: bt должен быть == 2*bc (top-level 2-блочный "
        f"solve), получено bt={config.bt}, bc={config.bc}. Не варьируйте "
        f"bc независимо от bt -- для granularity решателя есть config.mb."
    )
    grid = (bsz, H, n_chunks)
    spec = pl.BlockSpec((1, 1, 1, config.bt, config.bt), lambda i, h, c: (i, h, c, 0, 0))
    A = pl.pallas_call(
        lambda *refs: _kernel_b_body(*refs, bt=config.bt, bc=config.bc, config=config),
        grid=grid,
        in_specs=[spec],
        out_specs=spec,
        out_shape=jax.ShapeDtypeStruct(Akk.shape, jnp.float32),
        compiler_params=pltpu.CompilerParams(vmem_limit_bytes=96 * 1024 * 1024),
    )(Akk)
    return A


# ---------- Kernel C ----------
def _kernel_c_body(q_ref, k_ref, v_ref, w_ref, b_ref, g_ref, a_ref,
                   w_pseudo_ref, u_ref, kg_ref, qg_ref, gc_last_ref, *, bt: int):
    q = q_ref[0, 0, 0].astype(jnp.float32)
    k = k_ref[0, 0, 0].astype(jnp.float32)
    v = v_ref[0, 0, 0].astype(jnp.float32)
    w = w_ref[0, 0, 0].astype(jnp.float32)
    b = b_ref[0, 0, 0].astype(jnp.float32)
    g_raw = g_ref[0, 0, 0].astype(jnp.float32)
    A = a_ref[0, 0, 0].astype(jnp.float32)

    bt_idx = jnp.arange(bt)
    tril_ones_bt = (bt_idx[:, None] >= bt_idx[None, :]).astype(jnp.float32)
    gc = jnp.dot(tril_ones_bt, g_raw, precision=_HIGHEST)

    kb_decayed = b * k * jnp.exp(gc)
    w_pseudo = jnp.dot(A, kb_decayed, precision=_HIGHEST)
    u = jnp.dot(A, w * v, precision=_HIGHEST)
    w_pseudo = sanitize(w_pseudo)
    u = sanitize(u)

    gc_last_row = gc[bt - 1]
    kg = k * jnp.exp(gc_last_row[None, :] - gc)
    qg = q * jnp.exp(gc)
    kg = sanitize(kg)
    qg = sanitize(qg)
    gc_last_row = sanitize(gc_last_row)

    w_pseudo_ref[0, 0, 0] = w_pseudo
    u_ref[0, 0, 0] = u
    kg_ref[0, 0, 0] = kg
    qg_ref[0, 0, 0] = qg
    gc_last_ref[0, 0, 0, 0] = gc_last_row


def recompute_wy_pallas(q, k, v, w, b, g, A, config: KernelConfig = DEFAULT_CONFIG):
    bsz, L, H, D = q.shape
    n_chunks = L // config.bt

    def reshape_in(t):
        return _reshape_to_chunks(t, bsz, n_chunks, H, D, config.bt)

    q_r, k_r, v_r, w_r, b_r, g_r = map(reshape_in, (q, k, v, w, b, g))

    grid = (bsz, H, n_chunks)
    io_spec = pl.BlockSpec((1, 1, 1, config.bt, D), lambda i, h, c: (i, h, c, 0, 0))
    a_spec = pl.BlockSpec((1, 1, 1, config.bt, config.bt), lambda i, h, c: (i, h, c, 0, 0))
    gclast_spec = pl.BlockSpec((1, 1, 1, 1, D), lambda i, h, c: (i, h, c, 0, 0))

    w_pseudo, u, kg, qg, gc_last = pl.pallas_call(
        lambda *refs: _kernel_c_body(*refs, bt=config.bt),
        grid=grid,
        in_specs=[io_spec, io_spec, io_spec, io_spec, io_spec, io_spec, a_spec],
        out_specs=[io_spec, io_spec, io_spec, io_spec, gclast_spec],
        out_shape=[
            jax.ShapeDtypeStruct((bsz, H, n_chunks, config.bt, D), jnp.float32),
            jax.ShapeDtypeStruct((bsz, H, n_chunks, config.bt, D), jnp.float32),
            jax.ShapeDtypeStruct((bsz, H, n_chunks, config.bt, D), jnp.float32),
            jax.ShapeDtypeStruct((bsz, H, n_chunks, config.bt, D), jnp.float32),
            jax.ShapeDtypeStruct((bsz, H, n_chunks, 1, D), jnp.float32),
        ],
        compiler_params=pltpu.CompilerParams(vmem_limit_bytes=64 * 1024 * 1024),
    )(q_r, k_r, v_r, w_r, b_r, g_r, A)

    gc_last = gc_last.reshape(bsz, H, n_chunks, D)
    return w_pseudo, u, kg, qg, gc_last


# ---------- Kernel D ----------
def gdn2_inter_chunk_combine(Aqk, w_pseudo, u, kg, qg, gc_last, scale, h0=None,
                              config: KernelConfig = DEFAULT_CONFIG, debug_tag: str = ""):
    bsz, H, n_chunks, _BT, D = w_pseudo.shape
    if h0 is None:
        h0 = jnp.zeros((bsz, H, D, D), dtype=jnp.float32)
    h0 = sanitize_h0(h0)

    to_scan = tuple(jnp.moveaxis(x, 2, 0) for x in (Aqk, w_pseudo, u, kg, qg, gc_last))

    def step(h_pre, inputs):
        Aqk_c, w_pseudo_c, u_c, kg_c, qg_c, gclast_c = inputs
        wh = jnp.einsum("bhid,bhdv->bhiv", w_pseudo_c, h_pre, precision=_HIGHEST)
        v_new = u_c - wh
        qh = jnp.einsum("bhid,bhdv->bhiv", qg_c, h_pre, precision=_HIGHEST)
        intra = jnp.einsum("bhij,bhjv->bhiv", Aqk_c, v_new, precision=_HIGHEST)
        o_c = scale * qh + intra

        decay_h = jnp.exp(gclast_c)[..., None]
        write = jnp.einsum("bhid,bhiv->bhdv", kg_c, v_new, precision=_HIGHEST)
        h_new = h_pre * decay_h + write
        h_new = sanitize(h_new)
        o_c = sanitize(o_c)
        return h_new, o_c

    h_final, o_scanned = jax.lax.scan(step, h0, to_scan)
    h_final = _stage_diag(f"{debug_tag}:kernel_D_h_final", h_final)
    o = jnp.moveaxis(o_scanned, 0, 2)
    o = _stage_diag(f"{debug_tag}:kernel_D_o", o)
    return o, h_final


def gdn2_inter_chunk_combine_with_state(Aqk, w_pseudo, u, kg, qg, gc_last, scale,
                                         h0=None, config: KernelConfig = DEFAULT_CONFIG,
                                         debug_tag: str = ""):
    bsz, H, n_chunks, _BT, D = w_pseudo.shape
    if h0 is None:
        h0 = jnp.zeros((bsz, H, D, D), dtype=jnp.float32)
    h0 = sanitize_h0(h0)

    to_scan = tuple(jnp.moveaxis(x, 2, 0) for x in (Aqk, w_pseudo, u, kg, qg, gc_last))

    def step(h_pre, inputs):
        Aqk_c, w_pseudo_c, u_c, kg_c, qg_c, gclast_c = inputs
        wh = jnp.einsum("bhid,bhdv->bhiv", w_pseudo_c, h_pre, precision=_HIGHEST)
        v_new = u_c - wh
        qh = jnp.einsum("bhid,bhdv->bhiv", qg_c, h_pre, precision=_HIGHEST)
        intra = jnp.einsum("bhij,bhjv->bhiv", Aqk_c, v_new, precision=_HIGHEST)
        o_c = scale * qh + intra

        decay_h = jnp.exp(gclast_c)[..., None]
        write = jnp.einsum("bhid,bhiv->bhdv", kg_c, v_new, precision=_HIGHEST)
        h_new = h_pre * decay_h + write
        h_new = sanitize(h_new)
        o_c = sanitize(o_c)
        return h_new, (o_c, h_pre, v_new)

    h_final, (o_scanned, h_pre_all, v_new_all) = jax.lax.scan(step, h0, to_scan)
    h_final = _stage_diag(f"{debug_tag}:kernel_D_h_final", h_final)
    o = jnp.moveaxis(o_scanned, 0, 2)
    o = _stage_diag(f"{debug_tag}:kernel_D_o", o)
    return o, h_final, h_pre_all, v_new_all


def gdn2_pallas_forward(q, k, v, w, b, g, scale, h0=None,
                        config: KernelConfig = DEFAULT_CONFIG, debug_tag: str = ""):
    bsz, L, H, D, n_chunks = validate_inputs(q, k, v, w, b, g, scale, h0, config)

    Aqk, Akk = build_chunk_scores_pallas(q, k, b, g, scale, config)
    Aqk = _stage_diag(f"{debug_tag}:kernel_A_Aqk", Aqk)
    Akk = _stage_diag(f"{debug_tag}:kernel_A_Akk", Akk)

    A = wy_solve_pallas(Akk, config)
    A = _stage_diag(f"{debug_tag}:kernel_B_wy_inverse_A", A)

    w_pseudo, u, kg, qg, gc_last = recompute_wy_pallas(q, k, v, w, b, g, A, config)
    w_pseudo = _stage_diag(f"{debug_tag}:kernel_C_w_pseudo", w_pseudo)
    u = _stage_diag(f"{debug_tag}:kernel_C_u", u)
    kg = _stage_diag(f"{debug_tag}:kernel_C_kg", kg)
    qg = _stage_diag(f"{debug_tag}:kernel_C_qg", qg)

    o_chunks, h_final = gdn2_inter_chunk_combine(
        Aqk, w_pseudo, u, kg, qg, gc_last, scale, h0=h0, config=config, debug_tag=debug_tag
    )
    o = _reshape_from_chunks(o_chunks, bsz, n_chunks, config.bt, H, D)
    return o, h_final


def gdn2_pallas_forward_with_residuals(q, k, v, w, b, g, scale, h0=None,
                                        config: KernelConfig = DEFAULT_CONFIG,
                                        debug_tag: str = ""):
    bsz, L, H, D, n_chunks = validate_inputs(q, k, v, w, b, g, scale, h0, config)

    Aqk, Akk = build_chunk_scores_pallas(q, k, b, g, scale, config)
    Aqk = _stage_diag(f"{debug_tag}:kernel_A_Aqk", Aqk)
    Akk = _stage_diag(f"{debug_tag}:kernel_A_Akk", Akk)

    Akk_damped = Akk * (1.0 - config.wy_eps)
    A = wy_solve_pallas(Akk_damped, config)
    A = _stage_diag(f"{debug_tag}:kernel_B_wy_inverse_A", A)

    w_pseudo, u, kg, qg, gc_last = recompute_wy_pallas(q, k, v, w, b, g, A, config)
    w_pseudo = _stage_diag(f"{debug_tag}:kernel_C_w_pseudo", w_pseudo)
    u = _stage_diag(f"{debug_tag}:kernel_C_u", u)
    kg = _stage_diag(f"{debug_tag}:kernel_C_kg", kg)
    qg = _stage_diag(f"{debug_tag}:kernel_C_qg", qg)

    o_chunks, h_final, h_pre_all, v_new_all = gdn2_inter_chunk_combine_with_state(
        Aqk, w_pseudo, u, kg, qg, gc_last, scale, h0=h0, config=config, debug_tag=debug_tag
    )
    h_pre_all = jnp.moveaxis(h_pre_all, 0, 2)
    v_new_all = jnp.moveaxis(v_new_all, 0, 2)

    o = _reshape_from_chunks(o_chunks, bsz, n_chunks, config.bt, H, D)

    residuals = {
        "Aqk": Aqk, "Akk": Akk, "A": A,
        "h_pre_all": h_pre_all, "v_new_all": v_new_all,
        "w_pseudo": w_pseudo, "u": u, "kg": kg, "qg": qg, "gc_last": gc_last,
    }
    return o, h_final, residuals
