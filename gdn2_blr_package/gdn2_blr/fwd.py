"""
gdn2_blr.fwd -- Pallas forward.

Состав и почему именно так (см. handbook §5, таблицу «что в Pallas, что в XLA»):

  Kernel A (BLR scores)   -- Pallas. Единственное исключение из правила
      «chunk-parallel WY-математику держать в XLA»: k~ зависит от блока
      запросов через reference-точку r, поэтому в XLA пришлось бы
      материализовать n_sub копий k (x8 памяти при score_bs=32), а в
      VMEM это один блок 128 KB. У внешнего прецедента такой проблемы нет,
      потому что у Qwen3-Next GDN гейт СКАЛЯРНЫЙ, а у GDN-2/KDA -- по-канальный.

  Kernel B (H9 ladder)    -- Pallas. Настоящая последовательная
      зависимость + у внешней команды это крупнейший из трёх выигрышей.

  Kernel C                -- есть и Pallas, и XLA-двойник; выбор за гейтом.

  Kernel D (inter-chunk)  -- Pallas, state в VMEM внутри одного вызова,
      батчинг голов. lax.scan гоняет carry через HBM на каждом шаге.

MOSAIC-ДИСЦИПЛИНА, соблюдаемая во всех телах:
  * запись в ref ТОЛЬКО срезом по sublane-оси (строки) на полную ширину
    lane-оси; сборка по lane-оси делается конкатенацией ЗНАЧЕНИЙ
    (паттерн, уже работающий в production `_block_solve`);
  * никаких `.at[].set()` на значении с последующим bulk-write -- именно
    этот паттерн исторически ломал batched Kernel B на Mosaic;
  * никаких dynamic advanced indexing.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from .config import BLRConfig, effective_group, fit_group, fit_heads_per_cell, \
    vmem_kernel_a, vmem_kernel_b, vmem_kernel_c, vmem_kernel_d_res, \
    vmem_kernel_ab_fused
from .precision import HIGHEST, make_dot, make_einsum, exp_nonpos, exp_clipped, \
    sanitize
from .reference import to_chunks, from_chunks, chunk_gc, causal_masks, \
    ladder_inverse, _place

import warnings


def _cparams(vmem_bytes: int):
    for nm in ("CompilerParams", "TPUCompilerParams"):
        c = getattr(pltpu, nm, None)
        if c is not None:
            return c(vmem_limit_bytes=int(vmem_bytes))
    return None


def _vmem_guard(cfg: BLRConfig, need: int, who: str):
    if need <= cfg.vmem_budget:
        return
    msg = (f"{who}: оценка VMEM {need / 2**20:.1f} MB превышает бюджет "
           f"{cfg.vmem_budget / 2**20:.1f} MB. Уменьшите group/heads_per_cell "
           f"или n_chunks на ячейку.")
    if cfg.vmem_strict:
        raise RuntimeError(msg)
    warnings.warn(msg, RuntimeWarning)


# ===========================================================================
# Kernel A -- BLR scores
# ===========================================================================
def _blr_rows_2d(q, k, bk, gc, scale, cfg: BLRConfig, dot):
    """Общее тело BLR для ОДНОГО чанка: (T,D) -> список (bs,T) строк.

    Переиспользуется и отдельным Kernel A, и fused A+B. Ни одного среза
    по lane-оси: off-diagonal считается на полную ширину T и гасится
    маской `before`, диагональ собирается конкатенацией значений.
    Плата -- 2x флопов на off-diagonal части (не пользуемся причинностью),
    итого ровно T^2*D на матрицу, как и у текущего VPU-кернела, но на MXU.
    """
    T, bs, n_sub = cfg.bt, cfg.score_bs, cfg.n_sub
    dc = cfg.diff_clip
    idx = jnp.arange(T)
    rows_q, rows_k = [], []
    for p in range(n_sub):
        p0, p1 = p * bs, (p + 1) * bs
        r = gc[p0]
        a, _ = exp_nonpos(gc[p0:p1] - r[None, :])
        qt, bkt = q[p0:p1] * a, bk[p0:p1] * a

        if p0 > 0:
            c, _ = exp_nonpos(r[None, :] - gc)
            kt = k * c
            before = (idx < p0).astype(jnp.float32)[None, :]
            off_q = scale * dot(qt, kt.T) * before
            off_k = dot(bkt, kt.T) * before
        else:
            off_q = jnp.zeros((bs, T), jnp.float32)
            off_k = off_q

        gd = gc[p0:p1]
        E, _ = exp_clipped(gd[:, None, :] - gd[None, :, :], -dc, dc)
        dia_q = scale * jnp.sum(q[p0:p1][:, None, :] * E * k[p0:p1][None, :, :], axis=-1)
        dia_k = jnp.sum(bk[p0:p1][:, None, :] * E * k[p0:p1][None, :, :], axis=-1)
        rows_q.append(off_q + _place(dia_q, p0, p1, T))
        rows_k.append(off_k + _place(dia_k, p0, p1, T))
    return rows_q, rows_k


def _kernel_a_body(q_ref, k_ref, b_ref, gc_ref, aqk_ref, akk_ref, *,
                   scale: float, cfg: BLRConfig):
    q = q_ref[0, 0, 0].astype(jnp.float32)
    k = k_ref[0, 0, 0].astype(jnp.float32)
    b = b_ref[0, 0, 0].astype(jnp.float32)
    gc = gc_ref[0, 0, 0].astype(jnp.float32)
    dot = make_dot(cfg.dot_mode)
    causal, strict = causal_masks(cfg.bt)
    rows_q, rows_k = _blr_rows_2d(q, k, b * k, gc, scale, cfg, dot)
    bs = cfg.score_bs
    for p in range(cfg.n_sub):
        p0, p1 = p * bs, (p + 1) * bs
        aqk_ref[0, 0, 0, p0:p1] = sanitize(rows_q[p] * causal[p0:p1], cfg.clip)
        akk_ref[0, 0, 0, p0:p1] = sanitize(rows_k[p] * strict[p0:p1], cfg.clip)


def build_scores(q, k, b, gc, scale, cfg: BLRConfig):
    """q,k,b: (bsz,L,H,D). gc: (bsz,H,n_chunks,bt,D), уже cumsum'нутый."""
    bsz, L, H, D = q.shape
    nc = L // cfg.bt
    _vmem_guard(cfg, vmem_kernel_a(cfg, D), "Kernel A")
    qr, kr, br = (to_chunks(t, bsz, nc, H, D, cfg.bt) for t in (q, k, b))
    io = pl.BlockSpec((1, 1, 1, cfg.bt, D), lambda i, h, c: (i, h, c, 0, 0))
    sc = pl.BlockSpec((1, 1, 1, cfg.bt, cfg.bt), lambda i, h, c: (i, h, c, 0, 0))
    return pl.pallas_call(
        lambda *refs: _kernel_a_body(*refs, scale=scale, cfg=cfg),
        grid=(bsz, H, nc),
        in_specs=[io, io, io, io],
        out_specs=[sc, sc],
        out_shape=[jax.ShapeDtypeStruct((bsz, H, nc, cfg.bt, cfg.bt), jnp.float32)] * 2,
        compiler_params=_cparams(cfg.vmem_budget),
        interpret=cfg.interpret,
    )(qr, kr, br, gc)


# ===========================================================================
# Kernel B -- H9 ladder, батч по тайлам чанков
# ===========================================================================
def _kernel_b_body(akk_ref, a_ref, *, cfg: BLRConfig):
    S = akk_ref[0, 0].astype(jnp.float32)          # (group, T, T)
    A = ladder_inverse(S, cfg.wy_eps, cfg.bt, cfg.mb, cfg.solve_dot_mode)
    a_ref[0, 0] = sanitize(A, cfg.clip)


def wy_solve(Akk, cfg: BLRConfig, group: int | None = None):
    bsz, H, nc = Akk.shape[:3]
    g = fit_group(cfg, nc) if group is None else effective_group(nc, group)
    _vmem_guard(cfg, vmem_kernel_b(cfg, g), f"Kernel B (group={g})")
    spec = pl.BlockSpec((1, 1, g, cfg.bt, cfg.bt), lambda i, h, c: (i, h, c, 0, 0))
    return pl.pallas_call(
        lambda *refs: _kernel_b_body(*refs, cfg=cfg),
        grid=(bsz, H, nc // g),
        in_specs=[spec], out_specs=spec,
        out_shape=jax.ShapeDtypeStruct(Akk.shape, jnp.float32),
        compiler_params=_cparams(cfg.vmem_budget),
        interpret=cfg.interpret,
    )(Akk)


# ===========================================================================
# Fused A+B -- Akk никогда не попадает в HBM (фаза 3.4)
# ===========================================================================
def _kernel_ab_body(q_ref, k_ref, b_ref, gc_ref, aqk_ref, a_ref, *,
                    scale: float, cfg: BLRConfig, group: int):
    dot = make_dot(cfg.dot_mode)
    causal, strict = causal_masks(cfg.bt)
    bs = cfg.score_bs
    akk_list = []
    for gi in range(group):
        q = q_ref[0, 0, gi].astype(jnp.float32)
        k = k_ref[0, 0, gi].astype(jnp.float32)
        b = b_ref[0, 0, gi].astype(jnp.float32)
        gc = gc_ref[0, 0, gi].astype(jnp.float32)
        rows_q, rows_k = _blr_rows_2d(q, k, b * k, gc, scale, cfg, dot)
        for p in range(cfg.n_sub):
            p0, p1 = p * bs, (p + 1) * bs
            aqk_ref[0, 0, gi, p0:p1] = sanitize(rows_q[p] * causal[p0:p1], cfg.clip)
        akk_list.append(sanitize(jnp.concatenate(rows_k, axis=0) * strict, cfg.clip))
    Akk = jnp.stack(akk_list, axis=0)              # (group, T, T), только VMEM
    A = ladder_inverse(Akk, cfg.wy_eps, cfg.bt, cfg.mb, cfg.solve_dot_mode)
    a_ref[0, 0] = sanitize(A, cfg.clip)


def build_scores_and_solve(q, k, b, gc, scale, cfg: BLRConfig,
                           group: int | None = None):
    bsz, L, H, D = q.shape
    nc = L // cfg.bt
    g = fit_group(cfg, nc) if group is None else effective_group(nc, group)
    _vmem_guard(cfg, vmem_kernel_ab_fused(cfg, g, D), f"fused A+B (group={g})")
    qr, kr, br = (to_chunks(t, bsz, nc, H, D, cfg.bt) for t in (q, k, b))
    io = pl.BlockSpec((1, 1, g, cfg.bt, D), lambda i, h, c: (i, h, c, 0, 0))
    sc = pl.BlockSpec((1, 1, g, cfg.bt, cfg.bt), lambda i, h, c: (i, h, c, 0, 0))
    return pl.pallas_call(
        lambda *refs: _kernel_ab_body(*refs, scale=scale, cfg=cfg, group=g),
        grid=(bsz, H, nc // g),
        in_specs=[io, io, io, io],
        out_specs=[sc, sc],
        out_shape=[jax.ShapeDtypeStruct((bsz, H, nc, cfg.bt, cfg.bt), jnp.float32)] * 2,
        compiler_params=_cparams(cfg.vmem_budget),
        interpret=cfg.interpret,
    )(qr, kr, br, gc)


# ===========================================================================
# Kernel C -- recompute (с фиксом H0.6: min(.,0) перед каждым exp)
# ===========================================================================
def _kernel_c_body(q_ref, k_ref, v_ref, w_ref, b_ref, gc_ref, a_ref,
                   wp_ref, u_ref, kg_ref, qg_ref, gcl_ref, *, cfg: BLRConfig):
    dot = make_dot(cfg.dot_mode)
    q = q_ref[0, 0, 0].astype(jnp.float32)
    k = k_ref[0, 0, 0].astype(jnp.float32)
    v = v_ref[0, 0, 0].astype(jnp.float32)
    w = w_ref[0, 0, 0].astype(jnp.float32)
    b = b_ref[0, 0, 0].astype(jnp.float32)
    gc = gc_ref[0, 0, 0].astype(jnp.float32)
    A = a_ref[0, 0, 0].astype(jnp.float32)

    egc, _ = exp_nonpos(gc)
    kb = b * k * egc
    wp = sanitize(dot(A, kb), cfg.clip)
    u = sanitize(dot(A, w * v), cfg.clip)
    gc_last = gc[cfg.bt - 1]
    ekg, _ = exp_nonpos(gc_last[None, :] - gc)
    wp_ref[0, 0, 0] = wp
    u_ref[0, 0, 0] = u
    kg_ref[0, 0, 0] = sanitize(k * ekg, cfg.clip)
    qg_ref[0, 0, 0] = sanitize(q * egc, cfg.clip)
    gcl_ref[0, 0, 0, 0] = gc_last


def recompute_wy(q, k, v, w, b, gc, A, cfg: BLRConfig):
    bsz, L, H, D = q.shape
    nc = L // cfg.bt
    _vmem_guard(cfg, vmem_kernel_c(cfg, D), "Kernel C")
    qr, kr, vr, wr, br = (to_chunks(t, bsz, nc, H, D, cfg.bt)
                          for t in (q, k, v, w, b))
    io = pl.BlockSpec((1, 1, 1, cfg.bt, D), lambda i, h, c: (i, h, c, 0, 0))
    sc = pl.BlockSpec((1, 1, 1, cfg.bt, cfg.bt), lambda i, h, c: (i, h, c, 0, 0))
    gl = pl.BlockSpec((1, 1, 1, 1, D), lambda i, h, c: (i, h, c, 0, 0))
    wp, u, kg, qg, gcl = pl.pallas_call(
        lambda *refs: _kernel_c_body(*refs, cfg=cfg),
        grid=(bsz, H, nc),
        in_specs=[io, io, io, io, io, io, sc],
        out_specs=[io, io, io, io, gl],
        out_shape=[jax.ShapeDtypeStruct((bsz, H, nc, cfg.bt, D), jnp.float32)] * 4
        + [jax.ShapeDtypeStruct((bsz, H, nc, 1, D), jnp.float32)],
        compiler_params=_cparams(cfg.vmem_budget),
        interpret=cfg.interpret,
    )(qr, kr, vr, wr, br, gc, A)
    return wp, u, kg, qg, gcl.reshape(bsz, H, nc, D)


# ===========================================================================
# Kernel D -- inter-chunk scan, state в VMEM, батч голов
# ===========================================================================
def _kernel_d_body(aqk, wp, u, kg, qg, gcl, h0,
                   o_ref, hf_ref, hpre_ref, vnew_ref, *,
                   n_chunks: int, hb: int, scale: float, cfg: BLRConfig):
    dot = make_dot(cfg.dot_mode)
    for hh in range(hb):
        h = h0[0, hh].astype(jnp.float32)          # (D, Dv) -- живёт в VMEM
        for c in range(n_chunks):
            Aq = aqk[0, hh, c].astype(jnp.float32)
            wp_ = wp[0, hh, c].astype(jnp.float32)
            u_ = u[0, hh, c].astype(jnp.float32)
            kg_ = kg[0, hh, c].astype(jnp.float32)
            qg_ = qg[0, hh, c].astype(jnp.float32)
            gl = gcl[0, hh, c].astype(jnp.float32)

            hpre_ref[0, hh, c] = h
            v_new = u_ - dot(wp_, h)
            o_c = scale * dot(qg_, h) + dot(Aq, v_new)
            dec, _ = exp_nonpos(gl)
            h = sanitize(h * dec[:, None] + dot(kg_.T, v_new), cfg.clip)
            o_ref[0, hh, c] = sanitize(o_c, cfg.clip)
            vnew_ref[0, hh, c] = v_new
        hf_ref[0, hh] = h


def inter_chunk_scan(Aqk, wp, u, kg, qg, gc_last, scale, h0, cfg: BLRConfig,
                     heads_per_cell: int | None = None):
    """Возвращает (o_chunks, h_final, h_pre_all, v_new_all)."""
    bsz, H, nc, T, D = wp.shape
    hb = heads_per_cell or cfg.heads_per_cell
    if H % hb:
        hb = fit_heads_per_cell(cfg, nc, H, vmem_kernel_d_res, D)
    _vmem_guard(cfg, vmem_kernel_d_res(cfg, nc, hb, D), f"Kernel D (hb={hb})")
    io = pl.BlockSpec((1, hb, nc, T, D), lambda i, h: (i, h, 0, 0, 0))
    sc = pl.BlockSpec((1, hb, nc, T, T), lambda i, h: (i, h, 0, 0, 0))
    gl = pl.BlockSpec((1, hb, nc, D), lambda i, h: (i, h, 0, 0))
    hs = pl.BlockSpec((1, hb, D, D), lambda i, h: (i, h, 0, 0))
    hp = pl.BlockSpec((1, hb, nc, D, D), lambda i, h: (i, h, 0, 0, 0))
    return pl.pallas_call(
        lambda *r: _kernel_d_body(*r, n_chunks=nc, hb=hb, scale=scale, cfg=cfg),
        grid=(bsz, H // hb),
        in_specs=[sc, io, io, io, io, gl, hs],
        out_specs=[io, hs, hp, io],
        out_shape=[
            jax.ShapeDtypeStruct((bsz, H, nc, T, D), jnp.float32),   # o
            jax.ShapeDtypeStruct((bsz, H, D, D), jnp.float32),       # h_final
            jax.ShapeDtypeStruct((bsz, H, nc, D, D), jnp.float32),   # h_pre_all
            jax.ShapeDtypeStruct((bsz, H, nc, T, D), jnp.float32),   # v_new_all
        ],
        compiler_params=_cparams(cfg.vmem_budget),
        interpret=cfg.interpret,
    )(Aqk, wp, u, kg, qg, gc_last, h0)


# ===========================================================================
# полный forward
# ===========================================================================
def forward(q, k, v, w, b, g, scale, h0, cfg: BLRConfig, fused_ab: bool = False):
    bsz, L, H, D = q.shape
    nc = L // cfg.bt
    gr = to_chunks(g, bsz, nc, H, D, cfg.bt)
    gc = chunk_gc(gr)
    if fused_ab:
        Aqk, A = build_scores_and_solve(q, k, b, gc, scale, cfg)
        Akk = None
    else:
        Aqk, Akk = build_scores(q, k, b, gc, scale, cfg)
        A = wy_solve(Akk, cfg)
    wp, u, kg, qg, gc_last = recompute_wy(q, k, v, w, b, gc, A, cfg)
    o_ch, h_final, hpre, vnew = inter_chunk_scan(
        Aqk, wp, u, kg, qg, gc_last, scale, h0, cfg)
    o = from_chunks(o_ch, bsz, nc, cfg.bt, H, D)
    res = dict(gc=gc, Aqk=Aqk, Akk=Akk, A=A, w_pseudo=wp, u=u, kg=kg, qg=qg,
               gc_last=gc_last, h_pre_all=hpre, v_new_all=vnew)
    return o, h_final, res
