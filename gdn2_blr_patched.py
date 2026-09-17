"""
gdn2_blr_patched.py

Исправленная, изолированная версия трёх патчей:
  (1) Kernel A = BLR (F.build_scores), зафиксировано безусловно.
  (2) B3 = production Pallas (portированная 1:1 математика _kernel_b3_body_pallas),
      НЕ XLA.
  (3) Group-батчинг для B3 и B4 -- тот же приём, что дал Kernel B 45->7.9ms.

Отличия от предыдущей версии:
  - Убран effective_group (тихий подбор ближайшего делителя). Вместо него
    простой assert -- если group не делит n_chunks, падает явно на этапе
    вызова, а не подбирает что-то незаметно.
  - B4 batched kernel переписан на ПРЯМУЮ запись в срез ref на каждой
    итерации (dq_ref[0,0,gi,p0:p1] = sanitize(dq_ref[...] + delta)),
    вместо накопления в отдельной value-переменной с одним bulk-write в
    конце. Это тот самый паттерн (value-accumulate + bulk write), который
    в handbook (T-14, §6.2) прямо назван поломавшим batched Kernel B на
    Mosaic -- в прошлой версии файла B4 batched был написан неправильно.
  - Убраны мёртвые импорты внутри тела кернела (exp_clipped теперь
    импортирован один раз наверху, не на каждой итерации p).

Ничего в gdn2_blr_package не редактируется.
"""
from __future__ import annotations

import math
from functools import partial

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

import gdn2_blr_package.gdn2_blr.fwd as F
import gdn2_blr_package.gdn2_blr.bwd as B
import gdn2_blr_package.gdn2_blr.reference as R
from gdn2_blr_package.gdn2_blr.config import BLRConfig
from gdn2_blr_package.gdn2_blr.precision import make_dot, sanitize, exp_clipped


def _exp_nonpos(x):
    """Копия precision.exp_nonpos, локально -- чтобы не зависеть от
    внутреннего имени пакета при рефакторинге."""
    m = (x < 0.0).astype(jnp.float32)
    return jnp.exp(jnp.minimum(x, 0.0)), m


def _check_group(nc: int, group: int) -> int:
    """Простая проверка вместо effective_group. group ОБЯЗАН делить nc --
    никакого автоподбора."""
    assert group > 0, f"group must be positive, got {group}"
    assert nc % group == 0, f"n_chunks={nc} must be divisible by group={group}"
    return group


# ===========================================================================
# (2)+(3) B3 -- group-батченый Pallas, математика 1:1 из bwd._kernel_b3_body_pallas
# ===========================================================================
def _compute_b3_single_chunk(q_c, k_c, b_c, w_c, v_c, gc, A, h_pre, v_new,
                              do, dv, dh_next, *, scale, bt, wy_eps, cfg):
    dot = make_dot(cfg.dot_mode)
    C = bt
    egc, mgc = _exp_nonpos(gc)
    gc_last = gc[C - 1]
    ekg, mkg = _exp_nonpos(gc_last[None, :] - gc)

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

    decay_h_row, mdec = _exp_nonpos(gc_last)
    dgc_last_from_decay = decay_h_row * jnp.sum(dh_next * h_pre, axis=-1) * mdec
    dgc_last_total = dgc_last_contrib + dgc_last_from_decay
    row_mask = (idx == (C - 1)).astype(jnp.float32)[:, None]
    dgc = dgc + row_mask * dgc_last_total[None, :]

    return (sanitize(dq, cfg.clip), sanitize(dk, cfg.clip), sanitize(db, cfg.clip),
            sanitize(dw, cfg.clip), sanitize(dv_raw, cfg.clip), sanitize(dgc, cfg.clip),
            sanitize(dAkk, cfg.clip))


def _kernel_b3_body_batched(q_ref, k_ref, b_ref, w_ref, v_ref, gc_ref, a_ref,
                             hpre_ref, vnew_ref, do_ref, dv_ref, dhnext_ref,
                             dq_ref, dk_ref, db_ref, dw_ref, dvraw_ref, dgc_ref, dakk_ref,
                             *, scale, bt, wy_eps, cfg, group):
    for gi in range(group):
        q_c = q_ref[0, 0, gi].astype(jnp.float32)
        k_c = k_ref[0, 0, gi].astype(jnp.float32)
        b_c = b_ref[0, 0, gi].astype(jnp.float32)
        w_c = w_ref[0, 0, gi].astype(jnp.float32)
        v_c = v_ref[0, 0, gi].astype(jnp.float32)
        gc = gc_ref[0, 0, gi].astype(jnp.float32)
        A = a_ref[0, 0, gi].astype(jnp.float32)
        h_pre = hpre_ref[0, 0, gi].astype(jnp.float32)
        v_new = vnew_ref[0, 0, gi].astype(jnp.float32)
        do = do_ref[0, 0, gi].astype(jnp.float32)
        dv = dv_ref[0, 0, gi].astype(jnp.float32)
        dh_next = dhnext_ref[0, 0, gi].astype(jnp.float32)

        dq, dk, db, dw, dv_raw, dgc, dAkk = _compute_b3_single_chunk(
            q_c, k_c, b_c, w_c, v_c, gc, A, h_pre, v_new, do, dv, dh_next,
            scale=scale, bt=bt, wy_eps=wy_eps, cfg=cfg)

        # Прямая запись СРЕЗОМ по группе -- это единственный вывод на весь
        # чанк (не read-modify-write), поэтому bulk-write здесь безопасен:
        # тот же паттерн, что уже работает в build_and_solve_pallas_batched_fixed.
        dq_ref[0, 0, gi] = dq
        dk_ref[0, 0, gi] = dk
        db_ref[0, 0, gi] = db
        dw_ref[0, 0, gi] = dw
        dvraw_ref[0, 0, gi] = dv_raw
        dgc_ref[0, 0, gi] = dgc
        dakk_ref[0, 0, gi] = dAkk


def wy_dqkg_backward_pallas_batched(q, k, b, w, v, gc, A, h_pre, v_new,
                                     do, dv, dh_next, scale, cfg: BLRConfig,
                                     group: int):
    bsz, H, nc, T, D = q.shape
    group = _check_group(nc, group)
    n_groups = nc // group

    io = pl.BlockSpec((1, 1, group, T, D), lambda i, h, gi: (i, h, gi, 0, 0))
    sc = pl.BlockSpec((1, 1, group, T, T), lambda i, h, gi: (i, h, gi, 0, 0))
    hs = pl.BlockSpec((1, 1, group, D, D), lambda i, h, gi: (i, h, gi, 0, 0))

    out = pl.pallas_call(
        lambda *r: _kernel_b3_body_batched(*r, scale=scale, bt=T, wy_eps=cfg.wy_eps,
                                            cfg=cfg, group=group),
        grid=(bsz, H, n_groups),
        in_specs=[io, io, io, io, io, io, sc, hs, io, io, io, hs],
        out_specs=[io, io, io, io, io, io, sc],
        out_shape=[jax.ShapeDtypeStruct((bsz, H, nc, T, D), jnp.float32)] * 5
        + [jax.ShapeDtypeStruct((bsz, H, nc, T, D), jnp.float32),
           jax.ShapeDtypeStruct((bsz, H, nc, T, T), jnp.float32)],
        compiler_params=pltpu.CompilerParams(vmem_limit_bytes=cfg.vmem_budget),
        interpret=cfg.interpret,
    )(q, k, b, w, v, gc, A, h_pre, v_new, do, dv, dh_next)
    dq, dk, db, dw, dv_raw, dgc, dAkk = out
    return dict(dq=dq, dk=dk, db=db, dw=dw, dv_raw=dv_raw, dgc=dgc, dAkk=dAkk)


# ===========================================================================
# (3) B4 -- group-батченый, ИСПРАВЛЕН: прямая запись в ref-срез на каждом шаге
# ===========================================================================
def _kernel_b4_body_batched(q_ref, k_ref, b_ref, gc_ref, daqk_ref, dakk_ref,
                             dq_ref, dk_ref, db_ref, dgc_ref, *,
                             scale, cfg: BLRConfig, group: int):
    """Механическая обёртка bwd._kernel_b4_body: тело скопировано с заменой
    [0,0,0] -> [0,0,gi] внутри `for gi in range(group)`. КРИТИЧНО: каждая
    промежуточная сумма пишется ПРЯМО в срез ref (dq_ref[0,0,gi,p0:p1] = ...),
    а не в отдельную python-переменную с единственным bulk-write в конце --
    именно паттерн "value accumulate -> bulk write" исторически ломал
    batched Kernel B на Mosaic (handbook T-14)."""
    T, bs, n_sub, dc = cfg.bt, cfg.score_bs, cfg.n_sub, cfg.diff_clip
    c_ = cfg.clip
    dot = make_dot(cfg.dot_mode)
    causal, strict = R.causal_masks(T)

    for gi in range(group):
        q = q_ref[0, 0, gi].astype(jnp.float32)
        k = k_ref[0, 0, gi].astype(jnp.float32)
        b = b_ref[0, 0, gi].astype(jnp.float32)
        gc = gc_ref[0, 0, gi].astype(jnp.float32)
        dAqk = daqk_ref[0, 0, gi].astype(jnp.float32) * causal
        dAkk = dakk_ref[0, 0, gi].astype(jnp.float32) * strict
        bk = b * k
        idx = jnp.arange(T)

        dq_ref[0, 0, gi] = jnp.zeros((T, q.shape[-1]), jnp.float32)
        dk_ref[0, 0, gi] = jnp.zeros((T, q.shape[-1]), jnp.float32)
        db_ref[0, 0, gi] = jnp.zeros((T, q.shape[-1]), jnp.float32)
        dgc_ref[0, 0, gi] = jnp.zeros((T, q.shape[-1]), jnp.float32)

        for p in range(n_sub):
            p0, p1 = p * bs, (p + 1) * bs
            r = gc[p0]
            ea = gc[p0:p1] - r[None, :]
            a, m_a = _exp_nonpos(ea)
            qt, bkt = q[p0:p1] * a, bk[p0:p1] * a

            if p0 > 0:
                ec = r[None, :] - gc
                c, m_c = _exp_nonpos(ec)
                kt = k * c
                before = (idx < p0).astype(jnp.float32)[None, :]
                dMq = dAqk[p0:p1] * before
                dMk = dAkk[p0:p1] * before
                dqt = scale * dot(dMq, kt)
                dbkt = dot(dMk, kt)
                dkt = scale * dot(dMq.T, qt) + dot(dMk.T, bkt)

                dq_ref[0, 0, gi, p0:p1] = sanitize(dq_ref[0, 0, gi, p0:p1] + dqt * a, c_)
                db_ref[0, 0, gi, p0:p1] = sanitize(db_ref[0, 0, gi, p0:p1] + dbkt * a, c_)
                dk_ref[0, 0, gi] = sanitize(dk_ref[0, 0, gi] + dkt * c, c_)
                d_a = (dqt * qt + dbkt * bkt) * m_a
                d_c = (dkt * kt) * m_c
                dgc_ref[0, 0, gi] = sanitize(dgc_ref[0, 0, gi] - d_c, c_)
                dgc_ref[0, 0, gi, p0:p1] = sanitize(dgc_ref[0, 0, gi, p0:p1] + d_a, c_)
                dr = -jnp.sum(d_a, axis=0) + jnp.sum(d_c, axis=0)
                dgc_ref[0, 0, gi, p0:p0 + 1] = sanitize(
                    dgc_ref[0, 0, gi, p0:p0 + 1] + dr[None, :], c_)

            if cfg.dia_extract == "matmul":
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

            dq_ref[0, 0, gi, p0:p1] = sanitize(dq_ref[0, 0, gi, p0:p1] + dq_d, c_)
            db_ref[0, 0, gi, p0:p1] = sanitize(db_ref[0, 0, gi, p0:p1] + dbk_d, c_)
            dk_ref[0, 0, gi, p0:p1] = sanitize(dk_ref[0, 0, gi, p0:p1] + dk_d, c_)
            dgc_ref[0, 0, gi, p0:p1] = sanitize(dgc_ref[0, 0, gi, p0:p1] + dgc_d, c_)

        dbk_fin = db_ref[0, 0, gi]
        dk_ref[0, 0, gi] = sanitize(dk_ref[0, 0, gi] + dbk_fin * b, c_)
        db_ref[0, 0, gi] = sanitize(dbk_fin * k, c_)


def intra_backward_batched(dAqk, dAkk, q, k, b, gc, scale, cfg: BLRConfig,
                            group: int):
    bsz, L, H, D = q.shape
    nc = L // cfg.bt
    group = _check_group(nc, group)
    n_groups = nc // group

    qr = R.to_chunks(q, bsz, nc, H, D, cfg.bt)
    kr = R.to_chunks(k, bsz, nc, H, D, cfg.bt)
    br = R.to_chunks(b, bsz, nc, H, D, cfg.bt)

    io = pl.BlockSpec((1, 1, group, cfg.bt, D), lambda i, h, gi: (i, h, gi, 0, 0))
    sc = pl.BlockSpec((1, 1, group, cfg.bt, cfg.bt), lambda i, h, gi: (i, h, gi, 0, 0))

    return pl.pallas_call(
        lambda *r: _kernel_b4_body_batched(*r, scale=scale, cfg=cfg, group=group),
        grid=(bsz, H, n_groups),
        in_specs=[io, io, io, io, sc, sc],
        out_specs=[io, io, io, io],
        out_shape=[jax.ShapeDtypeStruct((bsz, H, nc, cfg.bt, D), jnp.float32)] * 4,
        compiler_params=pltpu.CompilerParams(vmem_limit_bytes=cfg.vmem_budget),
        interpret=cfg.interpret,
    )(qr, kr, br, gc, dAqk, dAkk)


# ===========================================================================
# Диспетчер: Kernel A = BLR всегда, B3 = Pallas (batched или non-batched),
# B4 = batched или non-batched. Настраивается явными флагами, без
# автоопределения по имени backend'а.
# ===========================================================================
def forward_with_residuals_fixed(q, k, v, w, b, g, scale, h0, cfg: BLRConfig):
    bsz, L, H, D = q.shape
    nc = L // cfg.bt
    qr, kr, vr, wr, br, gr = (R.to_chunks(t, bsz, nc, H, D, cfg.bt)
                              for t in (q, k, v, w, b, g))
    gc = R.chunk_gc(gr)

    Aqk, Akk = F.build_scores(q, k, b, gc, scale, cfg)         # (1) BLR, всегда
    A = F.wy_solve(Akk, cfg)
    wp, u, kg, qg, gc_last = F.recompute_wy(q, k, v, w, b, gc, A, cfg)
    o_ch, h_final, hpre, vnew = F.inter_chunk_scan(
        Aqk, wp, u, kg, qg, gc_last, scale, h0, cfg)

    o = R.from_chunks(o_ch, bsz, nc, cfg.bt, H, D)
    res = dict(gc=gc, Aqk=Aqk, A=A, w_pseudo=wp, u=u, kg=kg, qg=qg,
               gc_last=gc_last, h_pre_all=hpre, v_new_all=vnew,
               qr=qr, kr=kr, vr=vr, wr=wr, br=br)
    return o, h_final, res


def backward_from_residuals_fixed(q, k, v, w, b, g, h0, scale, cfg: BLRConfig,
                                   res, do, dh_final,
                                   b3_group: int | None,
                                   b4_group: int | None):
    """b3_group=None или b4_group=None -> non-batched Pallas версия
    (одна ячейка грида на чанк, как в bwd.py). Иначе -- group-батченая."""
    bsz, L, H, D = q.shape
    nc = L // cfg.bt
    do_r = R.to_chunks(do, bsz, nc, H, D, cfg.bt)
    gc = res["gc"]

    dAqk, dv_partial = B.dav_backward(res["Aqk"], res["v_new_all"], do_r, cfg)

    dh_next, dv_all, dh0 = B.dhu_backward(
        do_r, dv_partial, res["w_pseudo"], res["qg"], res["kg"],
        res["gc_last"], scale, dh_final, cfg)

    if b3_group is None:
        b3 = B.wy_dqkg_backward_pallas(                          # (2) Pallas, non-batched
            res["qr"], res["kr"], res["br"], res["wr"], res["vr"], gc, res["A"],
            res["h_pre_all"], res["v_new_all"], do_r, dv_all, dh_next, scale, cfg)
    else:
        b3 = wy_dqkg_backward_pallas_batched(                     # (2)+(3) Pallas, batched
            res["qr"], res["kr"], res["br"], res["wr"], res["vr"], gc, res["A"],
            res["h_pre_all"], res["v_new_all"], do_r, dv_all, dh_next, scale, cfg,
            group=b3_group)

    if b4_group is None:
        dq4, dk4, db4, dgc4 = B.intra_backward(dAqk, b3["dAkk"], q, k, b, gc, scale, cfg)
    else:
        dq4, dk4, db4, dgc4 = intra_backward_batched(
            dAqk, b3["dAkk"], q, k, b, gc, scale, cfg, group=b4_group)

    dg_r = B.reverse_cumsum(b3["dgc"] + dgc4, cfg)

    def out(x):
        return R.from_chunks(x, bsz, nc, cfg.bt, H, D)

    def fin(x, dt):
        return sanitize(x, 1e4).astype(dt)

    return (fin(out(b3["dq"] + dq4), q.dtype),
            fin(out(b3["dk"] + dk4), k.dtype),
            fin(out(b3["dv_raw"]), v.dtype),
            fin(out(b3["dw"]), w.dtype),
            fin(out(b3["db"] + db4), b.dtype),
            fin(out(dg_r), g.dtype),
            fin(dh0, h0.dtype))


# ===========================================================================
# custom_vjp обёртка. b3_group/b4_group -- nondiff, вшиты через closure
# фабрикой make_trainable, а не argnums-магией.
# ===========================================================================
def make_trainable(b3_group: int | None, b4_group: int | None):
    """Возвращает call_trainable(q,k,v,w,b,g,scale,h0=None,cfg=None) с
    зашитыми b3_group/b4_group. Явная фабрика вместо диспетчера по
    строковому имени backend'а -- нагляднее, что именно исполняется."""

    @partial(jax.custom_vjp, nondiff_argnums=(6, 7))
    def _core(q, k, v, w, b, g, scale, cfg, h0):
        o, h_final, _ = forward_with_residuals_fixed(q, k, v, w, b, g, scale, h0, cfg)
        return o, h_final

    def _fwd(q, k, v, w, b, g, scale, cfg, h0):
        o, h_final, res = forward_with_residuals_fixed(q, k, v, w, b, g, scale, h0, cfg)
        res = dict(res)
        res.update(q=q, k=k, v=v, w=w, b=b, g=g, h0=h0)
        return (o, h_final), res

    def _bwd(scale, cfg, res, cts):
        do, dh_final = cts
        return backward_from_residuals_fixed(
            res["q"], res["k"], res["v"], res["w"], res["b"], res["g"], res["h0"],
            scale, cfg, res, do, dh_final, b3_group=b3_group, b4_group=b4_group)

    _core.defvjp(_fwd, _bwd)

    def call_trainable(q, k, v, w, b, g, scale, h0=None, cfg: BLRConfig = None):
        cfg = cfg or BLRConfig()
        bsz, L, H, D = q.shape
        if h0 is None:
            h0 = jnp.zeros((bsz, H, D, D), jnp.float32)
        return _core(q, k, v, w, b, g, scale, cfg, h0)

    return call_trainable
