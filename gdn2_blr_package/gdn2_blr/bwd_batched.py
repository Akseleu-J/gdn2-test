"""
gdn2_blr.bwd_batched -- group-batched Pallas B3/B4.

Drop this file at: gdn2_blr_package/gdn2_blr/bwd_batched.py
(same package as fwd.py/bwd.py/config.py -- uses relative imports).

Closes two blockers:
  (1) B3 regression: backend="hybrid" currently pins B3 to XLA
      (wy_dqkg_backward) at ~10.7ms on train_shape, vs ~4.6ms for the old
      Pallas kernel this project already validated. wy_dqkg_backward_pallas_batched
      below ports _kernel_b3_body 1:1 (same math, same single (1-wy_eps)
      chain-rule factor -- see the long comment in _compute_b3_single_chunk)
      and adds the group-batching that already gave Kernel B 45ms->7.9ms.

  (2) B4 batched write pattern: the group-batched B4 prototype elsewhere
      in this project (gdn2_bwd_batched_b3_b4.py's _kernel_b4_body_batched
      equivalent for BLR) can end up doing "value-accumulate then one
      bulk write" inside the (si,sj)/p loop, which handbook.md T-14/§6.2
      names as the exact pattern that broke batched Kernel B on Mosaic.
      intra_backward_batched below writes DIRECTLY into the ref slice on
      every iteration (dq_ref[0,0,gi,p0:p1] = sanitize(dq_ref[...] + delta))
      -- the same pattern already proven to work in
      build_and_solve_pallas_batched_fixed.

Gate 1 (CPU, interpret=True) required before trusting this on TPU:
compare against bwd.wy_dqkg_backward_pallas / bwd.intra_backward
(non-batched) for group in (2,4,8) -- see the S7/S8 sections of
test_gdn2_blr_full_suite.py, which already exercises exactly this.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from .config import BLRConfig
from .precision import make_dot, sanitize, exp_clipped
from . import reference as R

_HIGHEST = jax.lax.Precision.HIGHEST


def _exp_nonpos(x):
    """Local copy of precision.exp_nonpos -- avoids depending on the
    internal name staying stable across refactors."""
    m = (x < 0.0).astype(jnp.float32)
    return jnp.exp(jnp.minimum(x, 0.0)), m


def _check_group(nc: int, group: int) -> int:
    """No silent auto-adjustment (unlike effective_group): group MUST
    divide n_chunks, or this raises immediately at the call site."""
    assert group > 0, f"group must be positive, got {group}"
    assert nc % group == 0, f"n_chunks={nc} must be divisible by group={group}"
    return group


# ===========================================================================
# B3 -- group-batched Pallas, math ported 1:1 from bwd._kernel_b3_body_pallas
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
    # ЕДИНСТВЕННЫЙ (1-wy_eps) chain-rule множитель -- нигде больше вниз по
    # потоку (B4/pipeline.py) не добавлять, история double-damping бага.
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

        # Один вывод на весь чанк (не read-modify-write) -- прямой
        # bulk-write в срез безопасен, тот же паттерн, что уже работает в
        # build_and_solve_pallas_batched_fixed.
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
# B4 -- group-batched, FIXED Mosaic pattern: direct per-iteration ref write
# ===========================================================================
def _kernel_b4_body_batched(q_ref, k_ref, b_ref, gc_ref, daqk_ref, dakk_ref,
                             dq_ref, dk_ref, db_ref, dgc_ref, *,
                             scale, cfg: BLRConfig, group: int):
    """Mechanical wrapper of bwd._kernel_b4_body: body copied with
    [0,0,0] -> [0,0,gi] inside `for gi in range(group)`. CRITICAL: every
    intermediate accumulation writes DIRECTLY into the ref slice
    (dq_ref[0,0,gi,p0:p1] = ...), never into a python-level value that
    gets one bulk write at the end -- that "accumulate then bulk write"
    pattern is what handbook.md T-14/§6.2 names as historically breaking
    batched Kernel B on Mosaic."""
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
