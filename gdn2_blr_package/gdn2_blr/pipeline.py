"""
gdn2_blr.pipeline -- полный custom_vjp пайплайн.

Диспетчер стадий: каждая стадия имеет Pallas- и XLA-реализацию, выбор по
`cfg.backend`:

    "pallas"  -- всё в Pallas (для изоляции и замера отдельных кернелов)
    "xla"     -- всё в XLA (Gate 1 target, ответ на H_EXEC, CPU fallback)
    "hybrid"  -- ЦЕЛЕВАЯ раскладка: Pallas только там, где есть настоящая
                 последовательная зависимость либо VMEM-специфика
                 (A, B, D, B1, B4), XLA для chunk-parallel матмулов
                 (C, B2, B3, B5).

Почему такая раскладка -- см. handbook §5. Коротко: внешняя production-
команда (MaxText #4348) измерила, что вплавление chunk-parallel матмулов
в Pallas-грид МЕДЛЕННЕЕ XLA, потому что ячейки грида на ядре TPU идут
последовательно. Наше единственное отклонение -- Kernel A, где
факторизация требует per-block k~, который в XLA пришлось бы
материализовать n_sub раз.

КОТАНГЕНСЫ ВОЗВРАЩАЮТСЯ В dtype ВХОДОВ forward. Иначе под bf16-autocast
на границе происходит тихая порча (зафиксировано в GDN2_MASTER.md §0).
"""
from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from .config import BLRConfig
from .precision import sanitize
from . import fwd as F
from . import bwd as B
from . import reference as R

_FINAL_CLIP = 1e4


def _use_pallas(cfg: BLRConfig, stage: str) -> bool:
    if cfg.backend == "pallas":
        return True
    if cfg.backend == "xla":
        return False
    return stage in ("A", "B", "D", "B1", "B4")   # hybrid


# ---------------------------------------------------------------------------
# forward со всеми residuals
# ---------------------------------------------------------------------------
def forward_with_residuals(q, k, v, w, b, g, scale, h0, cfg: BLRConfig):
    bsz, L, H, D = q.shape
    nc = L // cfg.bt
    qr, kr, vr, wr, br, gr = (R.to_chunks(t, bsz, nc, H, D, cfg.bt)
                              for t in (q, k, v, w, b, g))
    gc = R.chunk_gc(gr)

    if _use_pallas(cfg, "A"):
        Aqk, Akk = F.build_scores(q, k, b, gc, scale, cfg)
    else:
        Aqk, Akk = R.blr_scores_ref(qr, kr, br, gc, scale, cfg)

    if _use_pallas(cfg, "B"):
        A = F.wy_solve(Akk, cfg)
    else:
        A = sanitize(R.ladder_inverse(Akk, cfg.wy_eps, cfg.bt, cfg.mb,
                                      cfg.solve_dot_mode), cfg.clip)

    if _use_pallas(cfg, "C"):
        wp, u, kg, qg, gc_last = F.recompute_wy(q, k, v, w, b, gc, A, cfg)
    else:
        wp, u, kg, qg, gc_last = R.recompute_wy_ref(qr, kr, vr, wr, br, gc, A, cfg)

    if _use_pallas(cfg, "D"):
        o_ch, h_final, hpre, vnew = F.inter_chunk_scan(
            Aqk, wp, u, kg, qg, gc_last, scale, h0, cfg)
    else:
        o_ch, h_final, hpre, vnew = R.inter_chunk_scan_ref(
            Aqk, wp, u, kg, qg, gc_last, scale, h0, cfg)

    o = R.from_chunks(o_ch, bsz, nc, cfg.bt, H, D)
    res = dict(gc=gc, Aqk=Aqk, A=A, w_pseudo=wp, u=u, kg=kg, qg=qg,
               gc_last=gc_last, h_pre_all=hpre, v_new_all=vnew,
               qr=qr, kr=kr, vr=vr, wr=wr, br=br)
    return o, h_final, res


# ---------------------------------------------------------------------------
# backward
# ---------------------------------------------------------------------------
def backward_from_residuals(q, k, v, w, b, g, h0, scale, cfg: BLRConfig,
                            res, do, dh_final):
    bsz, L, H, D = q.shape
    nc = L // cfg.bt
    do_r = R.to_chunks(do, bsz, nc, H, D, cfg.bt)
    gc = res["gc"]

    # B2 -- всегда XLA (batched Pallas измерялся как 1.05x, у потолка)
    dAqk, dv_partial = B.dav_backward(res["Aqk"], res["v_new_all"], do_r, cfg)

    # B1
    if _use_pallas(cfg, "B1"):
        dh_next, dv_all, dh0 = B.dhu_backward(
            do_r, dv_partial, res["w_pseudo"], res["qg"], res["kg"],
            res["gc_last"], scale, dh_final, cfg)
    else:
        dh_next, dv_all, dh0 = B.dhu_backward_xla(
            do_r, dv_partial, res["w_pseudo"], res["qg"], res["kg"],
            res["gc_last"], scale, dh_final, cfg)

    # B3 -- всегда XLA, БЕЗ мёртвого входа Akk (дыра H0.5)
    b3 = B.wy_dqkg_backward(
        res["qr"], res["kr"], res["br"], res["wr"], res["vr"], gc, res["A"],
        res["h_pre_all"], res["v_new_all"], do_r, dv_all, dh_next, scale, cfg)

    # B4
    if _use_pallas(cfg, "B4"):
        dq4, dk4, db4, dgc4 = B.intra_backward(dAqk, b3["dAkk"], q, k, b, gc,
                                               scale, cfg)
    else:
        dq4, dk4, db4, dgc4 = B.blr_backward_ref(
            dAqk, b3["dAkk"], res["qr"], res["kr"], res["br"], gc, scale, cfg)

    # B5
    dg_r = B.reverse_cumsum(b3["dgc"] + dgc4, cfg)

    def out(x):
        return R.from_chunks(x, bsz, nc, cfg.bt, H, D)

    def fin(x, dt):
        return sanitize(x, _FINAL_CLIP).astype(dt)

    return (fin(out(b3["dq"] + dq4), q.dtype),
            fin(out(b3["dk"] + dk4), k.dtype),
            fin(out(b3["dv_raw"]), v.dtype),
            fin(out(b3["dw"]), w.dtype),
            fin(out(b3["db"] + db4), b.dtype),
            fin(out(dg_r), g.dtype),
            fin(dh0, h0.dtype))


# ---------------------------------------------------------------------------
# custom_vjp
# ---------------------------------------------------------------------------
@partial(jax.custom_vjp, nondiff_argnums=(6, 7))
def _core(q, k, v, w, b, g, scale, cfg, h0):
    o, h_final, _ = forward_with_residuals(q, k, v, w, b, g, scale, h0, cfg)
    return o, h_final


def _core_fwd(q, k, v, w, b, g, scale, cfg, h0):
    o, h_final, res = forward_with_residuals(q, k, v, w, b, g, scale, h0, cfg)
    res = dict(res)
    res.update(q=q, k=k, v=v, w=w, b=b, g=g, h0=h0)
    return (o, h_final), res


def _core_bwd(scale, cfg, res, cts):
    do, dh_final = cts
    return backward_from_residuals(
        res["q"], res["k"], res["v"], res["w"], res["b"], res["g"], res["h0"],
        scale, cfg, res, do, dh_final)


_core.defvjp(_core_fwd, _core_bwd)


# ---------------------------------------------------------------------------
# публичные точки входа
# ---------------------------------------------------------------------------
def validate_inputs(q, k, v, w, b, g, h0, cfg: BLRConfig):
    if q.ndim != 4:
        raise ValueError(f"q must be (batch, seq, heads, d_head); got {q.shape}")
    bsz, L, H, D = q.shape
    if L % cfg.bt:
        raise ValueError(f"seq_len={L} must be divisible by bt={cfg.bt}")
    for nm, t in (("k", k), ("v", v), ("w", w), ("b", b), ("g", g)):
        if t.shape != q.shape:
            raise ValueError(f"{nm}.shape={t.shape} must match q.shape={q.shape}")
    if h0 is not None and h0.shape != (bsz, H, D, D):
        raise ValueError(f"h0.shape={h0.shape} must be {(bsz, H, D, D)}")
    return bsz, L, H, D, L // cfg.bt


def blr_forward(q, k, v, w, b, g, scale, h0=None, cfg: BLRConfig = None):
    """Inference-only (без custom_vjp)."""
    cfg = cfg or BLRConfig()
    bsz, L, H, D, _ = validate_inputs(q, k, v, w, b, g, h0, cfg)
    if h0 is None:
        h0 = jnp.zeros((bsz, H, D, D), jnp.float32)
    o, h_final, _ = forward_with_residuals(q, k, v, w, b, g, scale, h0, cfg)
    return o, h_final


def blr_trainable(q, k, v, w, b, g, scale, h0=None, cfg: BLRConfig = None):
    """Тренировочная точка входа с custom_vjp."""
    cfg = cfg or BLRConfig()
    bsz, L, H, D, _ = validate_inputs(q, k, v, w, b, g, h0, cfg)
    if h0 is None:
        h0 = jnp.zeros((bsz, H, D, D), jnp.float32)
    return _core(q, k, v, w, b, g, scale, cfg, h0)
