"""
gdn2_blr.bwd -- backward.

Раскладка по исполнителям:
  B2 (dAqk/dv)      -- XLA. Два прямых матмула, батчатся сами; batched
                       Pallas измерялся как 1.05x (у потолка).
  B1 (inter-chunk)  -- Pallas, ОБРАТНЫЙ скан со state в VMEM. Зеркало
                       Kernel D. Настоящая последовательная зависимость.
  B3 (WY/dqkg)      -- XLA. Плюс УДАЛЁН мёртвый вход Akk (дыра H0.5:
                       Akk x100 не менял ни один из семи выходов, при этом
                       BlockSpec на него тянул ~201 MB HBM за backward).
  B4 (intra)        -- Pallas, BLR backward. Аналитика, без единого
                       деления; проверена против jax.vjp (S3: 1e-7..1e-5).
  B5 (reverse cumsum) -- XLA einsum. Оборачивание в Pallas измерялось как
                       1.07-1.21x ХУЖЕ plain JAX.

Каждая Pallas-функция имеет XLA-двойник с тем же именем + `_xla`. Это не
дублирование ради дублирования: двойник одновременно (а) цель гейта Gate 1,
(б) кандидат H_EXEC, (в) рабочий fallback, если Mosaic-lowering не зайдёт.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from .config import BLRConfig, fit_heads_per_cell, vmem_kernel_b1, vmem_kernel_b4
from .precision import HIGHEST, make_dot, make_einsum, exp_nonpos, exp_clipped, \
    sanitize
from .reference import to_chunks, causal_masks
from .fwd import _cparams, _vmem_guard
from .config import vmem_kernel_b4
# ===========================================================================
# B5 -- reverse cumsum (dgc -> dg)
# ===========================================================================
def reverse_cumsum(dgc, cfg: BLRConfig):
    T = dgc.shape[-2]
    i = jnp.arange(T)
    triu = (i[:, None] <= i[None, :]).astype(jnp.float32)
    return sanitize(jnp.einsum("ij,...jd->...id", triu, dgc.astype(jnp.float32),
                               precision=HIGHEST), cfg.clip)


# ===========================================================================
# B2 -- dav backward (XLA)
# ===========================================================================
def dav_backward(Aqk, v_new, do, cfg: BLRConfig):
    T = v_new.shape[-2]
    causal, _ = causal_masks(T)
    ein = make_einsum(cfg.dot_mode)
    dAqk = ein("...iv,...jv->...ij", do, v_new) * causal
    dv_partial = ein("...ji,...jv->...iv", Aqk, do)
    return sanitize(dAqk, cfg.clip), sanitize(dv_partial, cfg.clip)


# ===========================================================================
# B1 -- обратный inter-chunk скан, state в VMEM (Pallas)
# ===========================================================================
def _kernel_b1_body(do, dvp, wp, qg, kg, gcl, dht,
                    dhnext_ref, dv_ref, dh0_ref, *,
                    n_chunks: int, hb: int, scale: float, cfg: BLRConfig):
    dot = make_dot(cfg.dot_mode)
    for hh in range(hb):
        dh = dht[0, hh].astype(jnp.float32)       # dL/dh_final, живёт в VMEM
        for c in range(n_chunks - 1, -1, -1):
            # dh здесь == dL/dh_new_c == то, что B3 ждёт как dh_next[c]
            dhnext_ref[0, hh, c] = dh
            dec, _ = exp_nonpos(gcl[0, hh, c].astype(jnp.float32))
            dqh = scale * do[0, hh, c].astype(jnp.float32)
            from_out = dot(qg[0, hh, c].astype(jnp.float32).T, dqh)
            from_state = dh * dec[:, None]
            dv_write = dot(kg[0, hh, c].astype(jnp.float32), dh)
            dv_new = sanitize(dvp[0, hh, c].astype(jnp.float32) + dv_write, cfg.clip)
            from_vnew = -dot(wp[0, hh, c].astype(jnp.float32).T, dv_new)
            dh = sanitize(from_out + from_state + from_vnew, cfg.clip)
            dv_ref[0, hh, c] = dv_new
        dh0_ref[0, hh] = dh


def dhu_backward(do, dv_partial, w_pseudo, qg, kg, gc_last, scale, dht,
                 cfg: BLRConfig, heads_per_cell: int | None = None):
    """Возвращает (dh_next_all, dv_all, dh0).

    dh_next_all[c] = dL/dh_new_c -- именно то, что нужно B3, БЕЗ внешнего
    сдвига `_build_dh_next_all` (в старом коде это была отдельная склейка
    concat(dh_all[:,:,1:], dht)).
    """
    bsz, H, nc, T, D = qg.shape
    hb = heads_per_cell or cfg.heads_per_cell
    if H % hb:
        hb = fit_heads_per_cell(cfg, nc, H, vmem_kernel_b1, D)
    _vmem_guard(cfg, vmem_kernel_b1(cfg, nc, hb, D), f"Kernel B1 (hb={hb})")
    io = pl.BlockSpec((1, hb, nc, T, D), lambda i, h: (i, h, 0, 0, 0))
    gl = pl.BlockSpec((1, hb, nc, D), lambda i, h: (i, h, 0, 0))
    hs = pl.BlockSpec((1, hb, D, D), lambda i, h: (i, h, 0, 0))
    hp = pl.BlockSpec((1, hb, nc, D, D), lambda i, h: (i, h, 0, 0, 0))
    return pl.pallas_call(
        lambda *r: _kernel_b1_body(*r, n_chunks=nc, hb=hb, scale=scale, cfg=cfg),
        grid=(bsz, H // hb),
        in_specs=[io, io, io, io, io, gl, hs],
        out_specs=[hp, io, hs],
        out_shape=[
            jax.ShapeDtypeStruct((bsz, H, nc, D, D), jnp.float32),   # dh_next_all
            jax.ShapeDtypeStruct((bsz, H, nc, T, D), jnp.float32),   # dv_all
            jax.ShapeDtypeStruct((bsz, H, D, D), jnp.float32),       # dh0
        ],
        compiler_params=_cparams(cfg.vmem_budget),
        interpret=cfg.interpret,
    )(do, dv_partial, w_pseudo, qg, kg, gc_last, dht)


def dhu_backward_xla(do, dv_partial, w_pseudo, qg, kg, gc_last, scale, dht,
                     cfg: BLRConfig):
    """Двойник на lax.scan -- цель Gate 1 для Pallas-версии выше."""
    ein = make_einsum(cfg.dot_mode)
    to_scan = tuple(jnp.moveaxis(x, 2, 0)
                    for x in (do, dv_partial, w_pseudo, qg, kg, gc_last))

    def step(dh, xs):
        do_c, dvp_c, wp_c, qg_c, kg_c, gl_c = xs
        dec, _ = exp_nonpos(gl_c)
        dqh = scale * do_c
        from_out = ein("...id,...iv->...dv", qg_c, dqh)
        from_state = dh * dec[..., :, None]
        dv_write = ein("...id,...dv->...iv", kg_c, dh)
        dv_new = sanitize(dvp_c + dv_write, cfg.clip)
        from_vnew = -ein("...jd,...jv->...dv", wp_c, dv_new)
        dh_new = sanitize(from_out + from_state + from_vnew, cfg.clip)
        return dh_new, (dh, dv_new)

    dh0, (dhn, dv) = jax.lax.scan(step, dht, to_scan, reverse=True)
    return jnp.moveaxis(dhn, 0, 2), jnp.moveaxis(dv, 0, 2), dh0


# ===========================================================================
# B3 -- WY/dqkg backward (XLA, без мёртвого Akk)
# ===========================================================================
def wy_dqkg_backward(q, k, b, w, v, gc, A, h_pre, v_new, do, dv, dh_next,
                     scale, cfg: BLRConfig):
    """Все тензоры в чанк-разметке (bsz,H,nc,T,D) / (...,T,T) / (...,D,D).

    Маски `mgc`/`mkg`/`mdec` -- производные клампов min(.,0), введённых в
    Kernel C и D (фикс H0.6). Без них backward рассогласуется с forward
    ровно там, где клип срабатывает.
    """
    ein = make_einsum(cfg.dot_mode)
    T = q.shape[-2]
    i = jnp.arange(T)
    strict = (i[:, None] > i[None, :]).astype(jnp.float32)
    c_ = cfg.clip

    egc, mgc = exp_nonpos(gc)
    gc_last = gc[..., T - 1, :]
    ekg, mkg = exp_nonpos(gc_last[..., None, :] - gc)
    kb = b * k * egc
    kg = k * ekg
    qg = q * egc
    wv = w * v

    dqh = scale * do
    dqg = ein("...iv,...dv->...id", dqh, h_pre)
    dw_pseudo = ein("...iv,...dv->...id", -dv, h_pre)
    du = dv
    dkg = ein("...iv,...dv->...id", v_new, dh_next)

    dA_from_w = ein("...id,...jd->...ij", dw_pseudo, kb)
    dkb = ein("...ji,...jd->...id", A, dw_pseudo)
    dA_from_u = ein("...iv,...jv->...ij", du, wv)
    dwv = ein("...ji,...jv->...iv", A, du)

    dA_total = sanitize(dA_from_w + dA_from_u, c_)
    tmp = sanitize(ein("...ij,...kj->...ik", dA_total, A), c_)
    dAkk = -ein("...ji,...jk->...ik", A, tmp)
    # ЕДИНСТВЕННЫЙ (1-wy_eps) chain-rule множитель во всём пайплайне.
    # Не добавлять его нигде ниже по потоку (история double-damping бага).
    dAkk = sanitize(dAkk * (1.0 - cfg.wy_eps) * strict, c_)

    dk_from_kb = dkb * egc * b
    db = dkb * egc * k
    dgc_from_kb = dkb * kb * mgc

    dx = dkg * kg
    dk_from_kg = dkg * ekg
    dgc_from_kg = -dx * mkg
    dgc_last_contrib = jnp.sum(dx * mkg, axis=-2)

    dq = dqg * egc
    dgc_from_qg = dqg * qg * mgc

    dw = dwv * v
    dv_raw = dwv * w

    dk = dk_from_kb + dk_from_kg
    dgc = dgc_from_kb + dgc_from_qg + dgc_from_kg

    dec, mdec = exp_nonpos(gc_last)
    dgc_last_from_decay = dec * jnp.sum(dh_next * h_pre, axis=-1) * mdec
    dgc_last_total = dgc_last_contrib + dgc_last_from_decay
    row_last = (i == (T - 1)).astype(jnp.float32)[:, None]
    dgc = dgc + row_last * dgc_last_total[..., None, :]

    return dict(dq=sanitize(dq, c_), dk=sanitize(dk, c_), db=sanitize(db, c_),
                dw=sanitize(dw, c_), dv_raw=sanitize(dv_raw, c_),
                dgc=sanitize(dgc, c_), dAkk=dAkk)

# gdn2_blr_package/gdn2_blr/bwd.py -- добавить

def _kernel_b3_body_pallas(q_ref, k_ref, b_ref, w_ref, v_ref, gc_ref, a_ref,
                            hpre_ref, vnew_ref, do_ref, dv_ref, dhnext_ref,
                            dq_ref, dk_ref, db_ref, dw_ref, dvraw_ref, dgc_ref, dakk_ref,
                            *, scale: float, bt: int, wy_eps: float, cfg: BLRConfig):
    """Портировано 1:1 из Atomic_ops.gdn2_bwd._kernel_b3_body -- математика
    B3 не зависит от BLR/three-leg (только от A, h_pre, v_new), поэтому
    порт безопасен. H2: XLA-версия (wy_dqkg_backward) на TPU оказалась
    медленнее (~10.7ms) чем этот Pallas-кернел был в старом пакете
    (~4.6ms) -- вопреки общему правилу §5.1 плана "chunk-parallel в XLA
    быстрее". Держать обе реализации, выбор через cfg.backend."""
    dot = make_dot(cfg.dot_mode)
    q_c = q_ref[0, 0, 0].astype(jnp.float32)
    k_c = k_ref[0, 0, 0].astype(jnp.float32)
    b_c = b_ref[0, 0, 0].astype(jnp.float32)
    w_c = w_ref[0, 0, 0].astype(jnp.float32)
    v_c = v_ref[0, 0, 0].astype(jnp.float32)
    gc = gc_ref[0, 0, 0].astype(jnp.float32)
    A = a_ref[0, 0, 0].astype(jnp.float32)
    h_pre = hpre_ref[0, 0, 0].astype(jnp.float32)
    v_new = vnew_ref[0, 0, 0].astype(jnp.float32)
    do = do_ref[0, 0, 0].astype(jnp.float32)
    dv = dv_ref[0, 0, 0].astype(jnp.float32)
    dh_next = dhnext_ref[0, 0, 0].astype(jnp.float32)

    C = bt
    egc, mgc = exp_nonpos(gc)
    gc_last = gc[C - 1]
    ekg, mkg = exp_nonpos(gc_last[None, :] - gc)

    kb_decayed = b_c * k_c * egc
    kg = k_c * ekg
    qg = q_c * egc
    wv = w_c * v_c

    dqh_up = scale * do
    dqg = dot(dqh_up, h_pre.T)
    dwh = -dv
    dw_pseudo = dot(dwh, h_pre.T)
    du = dv
    dkg = dot(v_new, dh_next.T)

    dA_from_w = dot(dw_pseudo, kb_decayed.T)
    dkb_decayed = dot(A.T, dw_pseudo)
    dA_from_u = dot(du, wv.T)
    dwv = dot(A.T, du)

    dA_total = sanitize(dA_from_w + dA_from_u, cfg.clip)
    idx = jnp.arange(C)
    strict = (idx[:, None] > idx[None, :]).astype(jnp.float32)

    tmp = sanitize(dot(dA_total, A.T), cfg.clip)
    dAkk_raw = -dot(A.T, tmp) * (1.0 - wy_eps)
    dAkk = dAkk_raw * strict

    dk_from_kb = dkb_decayed * egc * b_c
    db = dkb_decayed * egc * k_c
    dgc_from_kb = dkb_decayed * kb_decayed * mgc

    dx = dkg * kg
    dk_from_kg = dkg * ekg
    dgc_from_kg = -dx * mkg
    dgc_last_contrib = jnp.sum(dx * mkg, axis=0)

    dq = dqg * egc
    dgc_from_qg = dqg * qg * mgc

    dw = dwv * v_c
    dv_raw = dwv * w_c

    dk = dk_from_kb + dk_from_kg
    dgc = dgc_from_kb + dgc_from_qg + dgc_from_kg

    decay_h_row, mdec = exp_nonpos(gc_last)
    dgc_last_from_decay = decay_h_row * jnp.sum(dh_next * h_pre, axis=-1) * mdec
    dgc_last_total = dgc_last_contrib + dgc_last_from_decay
    row_mask = (idx == (C - 1)).astype(jnp.float32)[:, None]
    dgc = dgc + row_mask * dgc_last_total[None, :]

    dq_ref[0, 0, 0] = sanitize(dq, cfg.clip)
    dk_ref[0, 0, 0] = sanitize(dk, cfg.clip)
    db_ref[0, 0, 0] = sanitize(db, cfg.clip)
    dw_ref[0, 0, 0] = sanitize(dw, cfg.clip)
    dvraw_ref[0, 0, 0] = sanitize(dv_raw, cfg.clip)
    dgc_ref[0, 0, 0] = sanitize(dgc, cfg.clip)
    dakk_ref[0, 0, 0] = sanitize(dAkk, cfg.clip)


def wy_dqkg_backward_pallas(q, k, b, w, v, gc, A, h_pre, v_new, do, dv, dh_next,
                            scale, cfg: BLRConfig):
    bsz, H, nc, T, D = q.shape
    _vmem_guard(cfg, vmem_kernel_b4(cfg, D) * 2, "Kernel B3 (Pallas)")  # грубая оценка, уточнить по факту
    io = pl.BlockSpec((1, 1, 1, T, D), lambda i, h, c: (i, h, c, 0, 0))
    sc = pl.BlockSpec((1, 1, 1, T, T), lambda i, h, c: (i, h, c, 0, 0))
    hs = pl.BlockSpec((1, 1, 1, D, D), lambda i, h, c: (i, h, c, 0, 0))
    out = pl.pallas_call(
        lambda *r: _kernel_b3_body_pallas(*r, scale=scale, bt=T, wy_eps=cfg.wy_eps, cfg=cfg),
        grid=(bsz, H, nc),
        in_specs=[io, io, io, io, io, io, sc, hs, io, io, io, hs],
        out_specs=[io, io, io, io, io, io, sc],
        out_shape=[jax.ShapeDtypeStruct((bsz, H, nc, T, D), jnp.float32)] * 5
        + [jax.ShapeDtypeStruct((bsz, H, nc, T, D), jnp.float32),
           jax.ShapeDtypeStruct((bsz, H, nc, T, T), jnp.float32)],
        compiler_params=_cparams(cfg.vmem_budget),
        interpret=cfg.interpret,
    )(q, k, b, w, v, gc, A, h_pre, v_new, do, dv, dh_next)
    dq, dk, db, dw, dv_raw, dgc, dAkk = out
    return dict(dq=dq, dk=dk, db=db, dw=dw, dv_raw=dv_raw, dgc=dgc, dAkk=dAkk) 
# ===========================================================================
# B4 -- BLR intra backward
# ===========================================================================
def blr_backward_ref(dAqk, dAkk, q, k, b, gc, scale, cfg: BLRConfig):
    """XLA-двойник B4 (батченый). Цель Gate 1 и кандидат H_EXEC.

    Вывод (без единого деления, поэтому устойчив при gamma -> 0):
        вне диагонали, блок P, r = gc[p0]:
            dgc_i   += (dq~_i*q~_i + dbk~_i*bk~_i) * 1{gc_i - r < 0}
            dgc_j   += -(dk~_j*k~_j)               * 1{r - gc_j < 0}
            dgc[p0] += -sum_i(первое) + sum_j(второе)     <- градиент по r
        на диагонали: формула non-centered пути с clipmask на +-diff_clip.
    """
    ein = make_einsum(cfg.dot_mode)
    T, bs, n_sub, dc = cfg.bt, cfg.score_bs, cfg.n_sub, cfg.diff_clip
    causal, strict = causal_masks(T)
    idx = jnp.arange(T)
    bk = b * k
    dAqk = dAqk * causal
    dAkk = dAkk * strict

    dq = jnp.zeros_like(q)
    dbk = jnp.zeros_like(q)
    dk = jnp.zeros_like(q)
    dgc = jnp.zeros_like(q)

    for p in range(n_sub):
        p0, p1 = p * bs, (p + 1) * bs
        r = gc[..., p0, :]
        ea = gc[..., p0:p1, :] - r[..., None, :]
        a, m_a = exp_nonpos(ea)
        qt, bkt = q[..., p0:p1, :] * a, bk[..., p0:p1, :] * a

        if p0 > 0:
            ec = r[..., None, :] - gc
            c, m_c = exp_nonpos(ec)
            kt = k * c
            before = (idx < p0).astype(jnp.float32)
            dMq = dAqk[..., p0:p1, :] * before
            dMk = dAkk[..., p0:p1, :] * before
            dqt = scale * ein("...ij,...jd->...id", dMq, kt)
            dbkt = ein("...ij,...jd->...id", dMk, kt)
            dkt = (scale * ein("...ij,...id->...jd", dMq, qt)
                   + ein("...ij,...id->...jd", dMk, bkt))

            dq = dq.at[..., p0:p1, :].add(dqt * a)
            dbk = dbk.at[..., p0:p1, :].add(dbkt * a)
            dk = dk + dkt * c
            d_a = (dqt * qt + dbkt * bkt) * m_a
            d_c = (dkt * kt) * m_c
            dgc = dgc.at[..., p0:p1, :].add(d_a)
            dgc = dgc - d_c
            dr = -jnp.sum(d_a, axis=-2) + jnp.sum(d_c, axis=-2)
            dgc = dgc.at[..., p0, :].add(dr)

        gd = gc[..., p0:p1, :]
        diff = gd[..., :, None, :] - gd[..., None, :, :]
        E, cm = exp_clipped(diff, -dc, dc)
        qd, kd, bkd = (q[..., p0:p1, :], k[..., p0:p1, :], bk[..., p0:p1, :])
        dMq_d = dAqk[..., p0:p1, p0:p1]
        dMk_d = dAkk[..., p0:p1, p0:p1]

        dq_d = scale * jnp.sum(dMq_d[..., :, :, None] * E * kd[..., None, :, :], axis=-2)
        dbk_d = jnp.sum(dMk_d[..., :, :, None] * E * kd[..., None, :, :], axis=-2)
        dk_d = (scale * jnp.sum(dMq_d[..., :, :, None] * E * qd[..., :, None, :], axis=-3)
                + jnp.sum(dMk_d[..., :, :, None] * E * bkd[..., :, None, :], axis=-3))
        wgt = ((dMq_d[..., :, :, None] * (scale * qd)[..., :, None, :]
                + dMk_d[..., :, :, None] * bkd[..., :, None, :])
               * kd[..., None, :, :] * E * cm)
        dgc_d = jnp.sum(wgt, axis=-2) - jnp.sum(wgt, axis=-3)

        dq = dq.at[..., p0:p1, :].add(dq_d)
        dbk = dbk.at[..., p0:p1, :].add(dbk_d)
        dk = dk.at[..., p0:p1, :].add(dk_d)
        dgc = dgc.at[..., p0:p1, :].add(dgc_d)

    dk = dk + dbk * b
    db = dbk * k
    c_ = cfg.clip
    return (sanitize(dq, c_), sanitize(dk, c_), sanitize(db, c_),
            sanitize(dgc, c_))


def _kernel_b4_body(q_ref, k_ref, b_ref, gc_ref, daqk_ref, dakk_ref,
                    dq_ref, dk_ref, db_ref, dgc_ref, *,
                    scale: float, cfg: BLRConfig):
    """Pallas-зеркало blr_backward_ref.

    db_ref используется как аккумулятор d(b*k) до самого конца -- ровно тот
    же приём, что в production _kernel_b4_body. Все записи -- срезы по
    sublane-оси на полную ширину lane-оси, кроме диагонального извлечения,
    для которого есть два режима (см. cfg.dia_extract).
    """
    dot = make_dot(cfg.dot_mode)
    T, bs, n_sub, dc = cfg.bt, cfg.score_bs, cfg.n_sub, cfg.diff_clip
    c_ = cfg.clip
    q = q_ref[0, 0, 0].astype(jnp.float32)
    k = k_ref[0, 0, 0].astype(jnp.float32)
    b = b_ref[0, 0, 0].astype(jnp.float32)
    gc = gc_ref[0, 0, 0].astype(jnp.float32)
    causal, strict = causal_masks(T)
    dAqk = daqk_ref[0, 0, 0].astype(jnp.float32) * causal
    dAkk = dakk_ref[0, 0, 0].astype(jnp.float32) * strict
    bk = b * k
    idx = jnp.arange(T)

    dq_ref[0, 0, 0] = jnp.zeros((T, q.shape[-1]), jnp.float32)
    dk_ref[0, 0, 0] = jnp.zeros((T, q.shape[-1]), jnp.float32)
    db_ref[0, 0, 0] = jnp.zeros((T, q.shape[-1]), jnp.float32)
    dgc_ref[0, 0, 0] = jnp.zeros((T, q.shape[-1]), jnp.float32)

    for p in range(n_sub):
        p0, p1 = p * bs, (p + 1) * bs
        r = gc[p0]
        ea = gc[p0:p1] - r[None, :]
        a, m_a = exp_nonpos(ea)
        qt, bkt = q[p0:p1] * a, bk[p0:p1] * a

        if p0 > 0:
            ec = r[None, :] - gc
            c, m_c = exp_nonpos(ec)
            kt = k * c
            before = (idx < p0).astype(jnp.float32)[None, :]
            dMq = dAqk[p0:p1] * before
            dMk = dAkk[p0:p1] * before
            dqt = scale * dot(dMq, kt)
            dbkt = dot(dMk, kt)
            dkt = scale * dot(dMq.T, qt) + dot(dMk.T, bkt)

            dq_ref[0, 0, 0, p0:p1] = sanitize(dq_ref[0, 0, 0, p0:p1] + dqt * a, c_)
            db_ref[0, 0, 0, p0:p1] = sanitize(db_ref[0, 0, 0, p0:p1] + dbkt * a, c_)
            dk_ref[0, 0, 0] = sanitize(dk_ref[0, 0, 0] + dkt * c, c_)
            d_a = (dqt * qt + dbkt * bkt) * m_a
            d_c = (dkt * kt) * m_c
            dgc_ref[0, 0, 0] = sanitize(dgc_ref[0, 0, 0] - d_c, c_)
            dgc_ref[0, 0, 0, p0:p1] = sanitize(dgc_ref[0, 0, 0, p0:p1] + d_a, c_)
            dr = -jnp.sum(d_a, axis=0) + jnp.sum(d_c, axis=0)
            dgc_ref[0, 0, 0, p0:p0 + 1] = sanitize(
                dgc_ref[0, 0, 0, p0:p0 + 1] + dr[None, :], c_)

        # --- диагональный блок ---
        if cfg.dia_extract == "matmul":
            # Выбор колонок [p0,p1) матмулом на (T,bs) one-hot вместо среза
            # по lane-оси. Безопасно при любом score_bs; стоит bs*T*bs флопов
            # на MXU (при bs=32, T=256 это 0.26 MFLOP на блок).
            sel = (idx[:, None] == (p0 + jnp.arange(bs))[None, :]).astype(jnp.float32)
            dMq_d = dot(dAqk[p0:p1], sel)
            dMk_d = dot(dAkk[p0:p1], sel)
        else:
            dMq_d = dAqk[p0:p1, p0:p1]
            dMk_d = dAkk[p0:p1, p0:p1]

        gd = gc[p0:p1]
        E, cm = exp_clipped(gd[:, None, :] - gd[None, :, :], -dc, dc)
        qd, kd, bkd = q[p0:p1], k[p0:p1], bk[p0:p1]
        dq_d = scale * jnp.sum(dMq_d[:, :, None] * E * kd[None, :, :], axis=1)
        dbk_d = jnp.sum(dMk_d[:, :, None] * E * kd[None, :, :], axis=1)
        dk_d = (scale * jnp.sum(dMq_d[:, :, None] * E * qd[:, None, :], axis=0)
                + jnp.sum(dMk_d[:, :, None] * E * bkd[:, None, :], axis=0))
        wgt = ((dMq_d[:, :, None] * (scale * qd)[:, None, :]
                + dMk_d[:, :, None] * bkd[:, None, :])
               * kd[None, :, :] * E * cm)
        dgc_d = jnp.sum(wgt, axis=1) - jnp.sum(wgt, axis=0)

        dq_ref[0, 0, 0, p0:p1] = sanitize(dq_ref[0, 0, 0, p0:p1] + dq_d, c_)
        db_ref[0, 0, 0, p0:p1] = sanitize(db_ref[0, 0, 0, p0:p1] + dbk_d, c_)
        dk_ref[0, 0, 0, p0:p1] = sanitize(dk_ref[0, 0, 0, p0:p1] + dk_d, c_)
        dgc_ref[0, 0, 0, p0:p1] = sanitize(dgc_ref[0, 0, 0, p0:p1] + dgc_d, c_)

    dbk_fin = db_ref[0, 0, 0]
    dk_ref[0, 0, 0] = sanitize(dk_ref[0, 0, 0] + dbk_fin * b, c_)
    db_ref[0, 0, 0] = sanitize(dbk_fin * k, c_)


def intra_backward(dAqk, dAkk, q, k, b, gc, scale, cfg: BLRConfig):
    """q,k,b: (bsz,L,H,D); gc/dAqk/dAkk уже в чанк-разметке."""
    bsz, L, H, D = q.shape
    nc = L // cfg.bt
    _vmem_guard(cfg, vmem_kernel_b4(cfg, D), "Kernel B4")
    qr, kr, br = (to_chunks(t, bsz, nc, H, D, cfg.bt) for t in (q, k, b))
    io = pl.BlockSpec((1, 1, 1, cfg.bt, D), lambda i, h, c: (i, h, c, 0, 0))
    sc = pl.BlockSpec((1, 1, 1, cfg.bt, cfg.bt), lambda i, h, c: (i, h, c, 0, 0))
    return pl.pallas_call(
        lambda *r: _kernel_b4_body(*r, scale=scale, cfg=cfg),
        grid=(bsz, H, nc),
        in_specs=[io, io, io, io, sc, sc],
        out_specs=[io, io, io, io],
        out_shape=[jax.ShapeDtypeStruct((bsz, H, nc, cfg.bt, D), jnp.float32)] * 4,
        compiler_params=_cparams(cfg.vmem_budget),
        interpret=cfg.interpret,
    )(qr, kr, br, gc, dAqk, dAkk)
