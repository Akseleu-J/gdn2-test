"""
test_gdn2_blr.py -- гейты пакета gdn2_blr.

Gate 1 (CPU, interpret=True)  = "математика не изменилась".
Gate 2 (TPU, interpret=False) = "лежит на Mosaic" + "не течёт по времени".
Второе НЕ следует из первого -- в истории проекта H9, fused A+B и batched
B2/B3/B4 проходили CPU-гейт и требовали отдельного TPU-гейта.

Секции:
  G0  конфиг, VMEM-бюджет, effective_group
  G1  Kernel A: Pallas vs reference vs float64 (совпадающий sanitize!)
  G2  причинность forward
  G3  Kernel B: лесенка H9 vs построчная инверсия, все режимы точности
  G4  Kernel C: Pallas vs reference
  G5  Kernel D: Pallas vs reference, включая residuals
  G6  B1: Pallas обратный VMEM-скан vs lax.scan
  G7  B4: Pallas vs reference vs jax.vjp
  G8  полный custom_vjp: forward vs token-serial, градиенты vs jax.grad
  G9  (TPU) тайминги и свип вариантов

Запуск: положить рядом с пакетом gdn2_blr и выполнить файл целиком.
JSON пишется после КАЖДОЙ проверки.
"""
from __future__ import annotations

import os
import sys
import json
import math
import time
import traceback

import numpy as np
import jax
import jax.numpy as jnp

sys.path.insert(0, os.getcwd())
try:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
except NameError:
    pass

from gdn2_blr import (BLRConfig, vmem_report, effective_group, reference as R,
                      fwd as F, bwd as B, blr_trainable)
from gdn2_blr.precision import sanitize

OUT = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
REPORT_PATH = os.path.join(OUT, "gdn2_blr_gates.json")
REPORT = {"meta": {}, "results": [], "failures": [], "timings": []}
BACKEND = jax.default_backend()
INTERPRET = BACKEND != "tpu"


def _save():
    try:
        with open(REPORT_PATH, "w") as f:
            json.dump(REPORT, f, indent=2, default=str)
    except Exception as e:
        print(f"[WARN] {e}", flush=True)


def log(m):
    print(m, flush=True)


def check(name, ok, err=None, tol=None, extra=""):
    st = "PASS" if ok else "FAIL"
    e = f" rel_err={err:.3e}" if err is not None else ""
    t = f" (tol={tol:.1e})" if tol is not None else ""
    log(f"[{st}] {name}{e}{t} {extra}")
    REPORT["results"].append({"name": name, "ok": bool(ok),
                              "rel_err": float(err) if err is not None else None,
                              "tol": tol, "extra": extra})
    if not ok:
        REPORT["failures"].append(name)
    _save()
    return ok


def info(m):
    log(f"[INFO] {m}")


def rel_err(a, b):
    a = np.asarray(jax.device_get(a), np.float64)
    b = np.asarray(jax.device_get(b), np.float64)
    return float(np.linalg.norm((a - b).ravel()) / max(np.linalg.norm(b.ravel()), 1e-12))


def timeit(fn, *a, n_warmup=3, n_iters=10, label=None):
    j = jax.jit(fn)
    for _ in range(n_warmup):
        jax.block_until_ready(j(*a))
    ts = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        jax.block_until_ready(j(*a))
        ts.append(time.perf_counter() - t0)
    ms = float(np.mean(ts)) * 1e3
    if label:
        log(f"    [TIMING] {label:<58} {ms:8.3f} ms")
        REPORT["timings"].append({"label": label, "ms": ms})
        _save()
    return ms


# ---------------------------------------------------------------------------
SEEDS = [
    dict(name="benign_seed0",            seed=0, g=0.05,  k=1.0, b=0.5, ood=False),
    dict(name="benign_seed1",            seed=1, g=0.05,  k=1.0, b=0.5, ood=False),
    dict(name="extreme_strong_decay",    seed=2, g=3.0,   k=1.0, b=0.5, ood=False),
    dict(name="extreme_near_zero_decay", seed=3, g=0.001, k=1.0, b=0.5, ood=False),
    dict(name="extreme_large_kb",        seed=4, g=0.3,   k=8.0, b=4.0, ood=False),
    dict(name="extreme_b_near_one",      seed=5, g=0.3,   k=1.0, b=0.95, ood=False),
    dict(name="mixed_sign_g_OOD",        seed=6, g=1.5,   k=2.0, b=0.7, ood=True),
]


def make_inputs(spec, bsz, L, H, D=128):
    rng = np.random.default_rng(spec["seed"])
    sh = (bsz, L, H, D)
    q = rng.normal(size=sh).astype(np.float32) * 0.1
    k = rng.normal(size=sh).astype(np.float32) * 0.1 * spec["k"]
    v = rng.normal(size=sh).astype(np.float32) * 0.1
    w = np.ones(sh, dtype=np.float32)
    b = (spec["b"] * rng.uniform(0.5, 1.0, size=sh)).astype(np.float32)
    gg = rng.normal(size=sh).astype(np.float32) * spec["g"]
    g = gg if spec["ood"] else -np.abs(gg)
    return tuple(jnp.asarray(x) for x in (q, k, v, w, b, g))


def exact_scores_f64(q, k, b, g, scale, clip, db=16):
    """float64 ground truth + ТОТ ЖЕ sanitize, что у Pallas/reference.
    Без совпадающего sanitize сравнение расходится на входах, где значения
    выходят за clip -- ровно это дало два ложных FAIL в прогоне v3."""
    q, k, b, g = (np.asarray(jax.device_get(x), np.float64) for x in (q, k, b, g))
    T, D = q.shape
    gc = np.cumsum(g, axis=0)
    A = np.zeros((T, T)); K = np.zeros((T, T)); bk = b * k
    for d0 in range(0, D, db):
        d1 = min(d0 + db, D)
        e = np.exp(gc[:, None, d0:d1] - gc[None, :, d0:d1])
        A += np.einsum("id,ijd,jd->ij", q[:, d0:d1], e, k[:, d0:d1])
        K += np.einsum("id,ijd,jd->ij", bk[:, d0:d1], e, k[:, d0:d1])
    i = np.arange(T)
    A = scale * A * (i[:, None] >= i[None, :])
    K = K * (i[:, None] > i[None, :])
    # ВАЖНО: конечность проверяется ДО sanitize, иначе проверка тождественно
    # истинна (та же методологическая дыра, что была в старой decision table:
    # finite_frac после nan_to_num не может упасть).
    finite = bool(np.all(np.isfinite(A)) and np.all(np.isfinite(K)))

    def san(x):
        return np.nan_to_num(np.clip(x, -clip, clip), nan=0.0,
                             posinf=clip, neginf=-clip)

    return san(A), san(K), finite


def cfg_of(**kw):
    kw.setdefault("interpret", INTERPRET)
    return BLRConfig(**kw)


# ===========================================================================
# G0
# ===========================================================================
def g0_config():
    log("\n" + "=" * 78)
    log("G0 -- конфиг, VMEM-бюджет, effective_group")
    log("=" * 78)
    ok = True
    for nc, req, want in ((4, 16, 4), (16, 16, 16), (12, 16, 4), (16, 8, 8),
                          (16, None, 16), (7, 4, 1)):
        got = effective_group(nc, req)
        ok &= (got == want)
        if got != want:
            log(f"    effective_group({nc},{req}) = {got}, ожидалось {want}")
    check("G0.effective_group", ok, None, None,
          "-- ФИКС H0.4: делитель вместо ValueError")

    for name, cfg in (("SAFE(bs=128)", cfg_of(score_bs=128, group=8)),
                      ("FAST(bs=32)", cfg_of(score_bs=32, group=8,
                                             heads_per_cell=4))):
        rep = vmem_report(cfg, bsz=8, H=6, n_chunks=16)
        log(f"    {name}: " + "  ".join(
            f"{k}={v:.2f}MB" if isinstance(v, float) else f"{k}={v}"
            for k, v in rep.items()))
        REPORT["results"].append({"name": f"G0.vmem[{name}]", "report": rep})
        _save()
    info("Все оценки должны быть <= budget (16 MB). Если нет -- уменьшать "
         "group / heads_per_cell ДО компиляции, а не ловить RESOURCE_EXHAUSTED.")


# ===========================================================================
# G1 / G2 -- Kernel A
# ===========================================================================
def g1_kernel_a():
    log("\n" + "=" * 78)
    log("G1 -- Kernel A (BLR): Pallas vs reference vs float64")
    log("=" * 78)
    scale = 1.0 / math.sqrt(128)
    for bt in (128, 256):
        for bs in (32, 64, 128):
            if bs > bt:
                continue
            cfg = cfg_of(bt=bt, bc=bt // 2, score_bs=bs)
            for spec in SEEDS:
                tag = f"bt={bt},bs={bs},{spec['name']}"
                q, k, v, w, b, g = make_inputs(spec, 1, bt, 1)
                gr = R.to_chunks(g, 1, 1, 1, 128, bt)
                gc = R.chunk_gc(gr)
                qr = R.to_chunks(q, 1, 1, 1, 128, bt)
                kr = R.to_chunks(k, 1, 1, 1, 128, bt)
                br = R.to_chunks(b, 1, 1, 1, 128, bt)

                Aq_p, Ak_p = F.build_scores(q, k, b, gc, scale, cfg)
                Aq_r, Ak_r = R.blr_scores_ref(qr, kr, br, gc, scale, cfg)
                e_pr = rel_err(Aq_p[0, 0, 0], Aq_r[0, 0, 0])
                check(f"G1.pallas_vs_ref.Aqk[{tag}]", e_pr < 1e-5, e_pr, 1e-5)
                e_pk = rel_err(Ak_p[0, 0, 0], Ak_r[0, 0, 0])
                check(f"G1.pallas_vs_ref.Akk[{tag}]", e_pk < 1e-5, e_pk, 1e-5)

                gt_q, gt_k, gt_ok = exact_scores_f64(
                    q[0, :, 0], k[0, :, 0], b[0, :, 0], g[0, :, 0], scale, cfg.clip)
                if not gt_ok:
                    info(f"G1[{tag}]: float64-эталон переполнился ДО клипа "
                         f"-- сравнение не определено, пропуск")
                    continue
                e = rel_err(Aq_p[0, 0, 0], gt_q)
                if spec["ood"]:
                    info(f"G1.vs_f64[{tag}]: rel_err={e:.3e} -- "
                         f"OUT-OF-DISTRIBUTION (g>0 => alpha=exp(g)>1, не "
                         f"forget-gate; на этом входе ломается и baseline)")
                else:
                    check(f"G1.vs_f64.Aqk[{tag}]", e < 5e-4, e, 5e-4,
                          "-- ожидается паритет с production non-centered")


def g2_causality():
    log("\n" + "=" * 78)
    log("G2 -- причинность: возмущение g в будущем токене")
    log("=" * 78)
    scale = 1.0 / math.sqrt(128)
    for bt in (128, 256):
        cfg = cfg_of(bt=bt, bc=bt // 2, score_bs=64)
        for spec in SEEDS:
            q, k, v, w, b, g = make_inputs(spec, 1, bt, 1)
            gc = R.chunk_gc(R.to_chunks(g, 1, 1, 1, 128, bt))
            Aq, _ = F.build_scores(q, k, b, gc, scale, cfg)
            p = bt * 3 // 4
            delta = -jnp.abs(jax.random.normal(
                jax.random.PRNGKey(spec["seed"]), g[:, p, :, :].shape)) * 2.0
            g2 = g.at[:, p, :, :].add(delta)
            gc2 = R.chunk_gc(R.to_chunks(g2, 1, 1, 1, 128, bt))
            Aq2, _ = F.build_scores(q, k, b, gc2, scale, cfg)
            e = rel_err(Aq2[0, 0, 0, :p, :p], Aq[0, 0, 0, :p, :p])
            check(f"G2.causality[bt={bt},{spec['name']}]", e < 1e-6, e, 1e-6,
                  "-- Aqk[i<p,j<p] не должен зависеть от g[p]")


# ===========================================================================
# G3 -- Kernel B
# ===========================================================================
def g3_kernel_b():
    log("\n" + "=" * 78)
    log("G3 -- Kernel B: лесенка H9 vs построчная инверсия; режимы точности")
    log("=" * 78)
    for bt in (128, 256):
        for eps in (0.0, 1e-3):
            for mode in ("highest", "bf16x3", "default"):
                cfg = cfg_of(bt=bt, bc=bt // 2, wy_eps=eps,
                             solve_dot_mode=mode, group=4)
                key = jax.random.PRNGKey(2000 + bt)
                raw = jax.random.normal(key, (2, 2, 4, bt, bt)) * 0.12
                i = jnp.arange(bt)
                strict = (i[:, None] > i[None, :]).astype(jnp.float32)
                Akk = (raw * strict[None, None, None]).astype(jnp.float32)
                tag = f"bt={bt},eps={eps},dot={mode}"

                A = F.wy_solve(Akk, cfg)
                eye = jnp.broadcast_to(jnp.eye(bt), Akk.shape)
                resid = jnp.einsum("...ij,...jk->...ik",
                                   eye + (1.0 - eps) * Akk, A,
                                   precision=jax.lax.Precision.HIGHEST)
                er = rel_err(resid, eye)
                check(f"G3.residual[{tag}]", er < 1e-4, er, 1e-4,
                      "-- (I+(1-eps)Akk) @ A == I")
                if mode == "highest":
                    A_ref = R.row_by_row_inverse(Akk, eps)
                    e = rel_err(A, A_ref)
                    check(f"G3.ladder_vs_rowwise[{tag}]", e < 1e-4, e, 1e-4)
    info("На CPU 'default'=='highest' (нет MXU), а bf16x3 -- своя точность "
         "(~4.5e-06 против 2.9e-07 у f32). На TPU таблица может "
         "перевернуться: если 'highest' уедет к ~1e-3, значит Mosaic "
         "усекает f32-dot до bf16 и bf16x3 становится обязательным.")


# ===========================================================================
# G4 / G5 -- Kernel C, D
# ===========================================================================
def g4_kernel_c():
    log("\n" + "=" * 78)
    log("G4 -- Kernel C: Pallas vs reference (клип min(.,0) с обеих сторон)")
    log("=" * 78)
    bt, nc, H, bsz, D = 256, 2, 2, 1, 128
    cfg = cfg_of(bt=bt, bc=128, score_bs=64)
    for spec in SEEDS:
        q, k, v, w, b, g = make_inputs(spec, bsz, nc * bt, H)
        gc = R.chunk_gc(R.to_chunks(g, bsz, nc, H, D, bt))
        A = jnp.broadcast_to(jnp.eye(bt, dtype=jnp.float32),
                             (bsz, H, nc, bt, bt))
        out_p = F.recompute_wy(q, k, v, w, b, gc, A, cfg)
        qr, kr, vr, wr, br = (R.to_chunks(t, bsz, nc, H, D, bt)
                              for t in (q, k, v, w, b))
        out_r = R.recompute_wy_ref(qr, kr, vr, wr, br, gc, A, cfg)
        for nm, a_, b_ in zip(("w_pseudo", "u", "kg", "qg", "gc_last"), out_p, out_r):
            e = rel_err(a_, b_)
            check(f"G4.{nm}[{spec['name']}]", e < 1e-5, e, 1e-5)


def g5_kernel_d():
    log("\n" + "=" * 78)
    log("G5 -- Kernel D: Pallas VMEM-скан vs lax.scan, включая residuals")
    log("=" * 78)
    bt, D = 256, 128
    scale = 1.0 / math.sqrt(D)
    for bsz, H, nc, hb in ((1, 2, 4, 1), (1, 2, 4, 2), (2, 4, 4, 4)):
        cfg = cfg_of(bt=bt, bc=128, score_bs=64, heads_per_cell=hb)
        spec = SEEDS[0]
        q, k, v, w, b, g = make_inputs(spec, bsz, nc * bt, H)
        gc = R.chunk_gc(R.to_chunks(g, bsz, nc, H, D, bt))
        Aqk, Akk = F.build_scores(q, k, b, gc, scale, cfg)
        A = F.wy_solve(Akk, cfg)
        wp, u, kg, qg, gcl = F.recompute_wy(q, k, v, w, b, gc, A, cfg)
        h0 = jnp.zeros((bsz, H, D, D), jnp.float32)
        p = F.inter_chunk_scan(Aqk, wp, u, kg, qg, gcl, scale, h0, cfg)
        r = R.inter_chunk_scan_ref(Aqk, wp, u, kg, qg, gcl, scale, h0, cfg)
        tag = f"bsz={bsz},H={H},nc={nc},hb={hb}"
        for nm, a_, b_ in zip(("o", "h_final", "h_pre_all", "v_new_all"), p, r):
            e = rel_err(a_, b_)
            check(f"G5.{nm}[{tag}]", e < 1e-5, e, 1e-5)


# ===========================================================================
# G6 -- B1
# ===========================================================================
def g6_b1():
    log("\n" + "=" * 78)
    log("G6 -- B1: Pallas обратный VMEM-скан vs lax.scan")
    log("=" * 78)
    bt, D = 256, 128
    scale = 1.0 / math.sqrt(D)
    for bsz, H, nc, hb in ((1, 2, 4, 1), (1, 2, 4, 2)):
        cfg = cfg_of(bt=bt, bc=128, score_bs=64, heads_per_cell=hb)
        rng = np.random.default_rng(31)

        def rnd(*s):
            return jnp.asarray(rng.normal(size=s).astype(np.float32) * 0.05)

        do = rnd(bsz, H, nc, bt, D)
        dvp = rnd(bsz, H, nc, bt, D)
        wp = rnd(bsz, H, nc, bt, D)
        qg = rnd(bsz, H, nc, bt, D)
        kg = rnd(bsz, H, nc, bt, D)
        gcl = -jnp.abs(rnd(bsz, H, nc, D))
        dht = rnd(bsz, H, D, D)
        p = B.dhu_backward(do, dvp, wp, qg, kg, gcl, scale, dht, cfg)
        r = B.dhu_backward_xla(do, dvp, wp, qg, kg, gcl, scale, dht, cfg)
        tag = f"bsz={bsz},H={H},nc={nc},hb={hb}"
        for nm, a_, b_ in zip(("dh_next", "dv_all", "dh0"), p, r):
            e = rel_err(a_, b_)
            check(f"G6.{nm}[{tag}]", e < 1e-5, e, 1e-5)


# ===========================================================================
# G7 -- B4
# ===========================================================================
def g7_b4():
    log("\n" + "=" * 78)
    log("G7 -- B4 (BLR backward): Pallas vs reference vs jax.vjp")
    log("=" * 78)
    scale = 1.0 / math.sqrt(128)
    for bt in (128, 256):
        for bs in (64, 128):
            if bs > bt:
                continue
            cfg = cfg_of(bt=bt, bc=bt // 2, score_bs=bs)
            for spec in SEEDS:
                if spec["ood"]:
                    continue
                tag = f"bt={bt},bs={bs},{spec['name']}"
                q, k, v, w, b, g = make_inputs(spec, 1, bt, 1)
                qr, kr, br, gr = (R.to_chunks(t, 1, 1, 1, 128, bt)
                                  for t in (q, k, b, g))
                gc = R.chunk_gc(gr)
                key = jax.random.PRNGKey(800 + bt + spec["seed"])
                k1, k2 = jax.random.split(key)
                dAq = jax.random.normal(k1, (1, 1, 1, bt, bt)) * 0.01
                dAk = jax.random.normal(k2, (1, 1, 1, bt, bt)) * 0.01

                def fn(q_, k_, b_, g_):
                    gc_ = R.chunk_gc(g_)
                    return R.blr_scores_ref(q_, k_, b_, gc_, scale, cfg)

                (_, _), vjp = jax.vjp(fn, qr, kr, br, gr)
                dq_r, dk_r, db_r, dg_r = vjp((dAq, dAk))

                dq_x, dk_x, db_x, dgc_x = B.blr_backward_ref(
                    dAq, dAk, qr, kr, br, gc, scale, cfg)
                dg_x = B.reverse_cumsum(dgc_x, cfg)
                for nm, a_, b_ in (("dq", dq_x, dq_r), ("dk", dk_x, dk_r),
                                   ("db", db_x, db_r), ("dg", dg_x, dg_r)):
                    e = rel_err(a_, b_)
                    check(f"G7.ref_vs_vjp.{nm}[{tag}]", e < 1e-4, e, 1e-4)

                dq_p, dk_p, db_p, dgc_p = B.intra_backward(
                    dAq, dAk, q, k, b, gc, scale, cfg)
                dg_p = B.reverse_cumsum(dgc_p, cfg)
                for nm, a_, b_ in (("dq", dq_p, dq_r), ("dk", dk_p, dk_r),
                                   ("db", db_p, db_r), ("dg", dg_p, dg_r)):
                    e = rel_err(a_, b_)
                    check(f"G7.pallas_vs_vjp.{nm}[{tag}]", e < 5e-4, e, 5e-4)

                # причинность ТОЛЬКО по dq (dk обязан зависеть от будущих g)
                p_ = bt * 3 // 4
                g2 = gr.at[0, 0, 0, p_].add(
                    -jnp.abs(jax.random.normal(jax.random.PRNGKey(7), (128,))) * 2.0)
                gc2 = R.chunk_gc(g2)
                dq_p2, _, _, _ = B.intra_backward(dAq, dAk, q, k, b, gc2, scale, cfg)
                e = rel_err(dq_p2[0, 0, 0, :p_], dq_p[0, 0, 0, :p_])
                check(f"G7.causality.dq[{tag}]", e < 1e-6, e, 1e-6)


# ===========================================================================
# G8 -- полный пайплайн
# ===========================================================================
def g8_pipeline():
    log("\n" + "=" * 78)
    log("G8 -- полный custom_vjp: forward vs token-serial, градиенты vs jax.grad")
    log("=" * 78)
    bt, D, nc, bsz, H = 256, 128, 2, 1, 2
    scale = 1.0 / math.sqrt(D)
    L = nc * bt
    for backend in ("xla", "hybrid", "pallas"):
        for spec in SEEDS:
            if spec["ood"]:
                continue
            cfg = cfg_of(bt=bt, bc=128, score_bs=64, backend=backend, group=nc)
            tag = f"backend={backend},{spec['name']}"
            q, k, v, w, b, g = make_inputs(spec, bsz, L, H)
            h0 = jnp.zeros((bsz, H, D, D), jnp.float32)

            o, hf = blr_trainable(q, k, v, w, b, g, scale, h0=h0, cfg=cfg)
            o_ts, hf_ts = R.token_serial_ref(q, k, v, g, b, w, scale, h0=h0)
            if np.all(np.isfinite(np.asarray(jax.device_get(o_ts)))):
                e = rel_err(o, o_ts)
                check(f"G8.o_vs_token_serial[{tag}]", e < 5e-2, e, 5e-2)
            else:
                info(f"G8[{tag}]: token-serial сам разошёлся -- пропуск")

            o_ref, hf_ref, _ = R.forward_ref(q, k, v, w, b, g, scale, h0, cfg)
            e = rel_err(o, o_ref)
            check(f"G8.o_vs_reference[{tag}]", e < 1e-4, e, 1e-4)

            def loss_new(q_, k_, v_, w_, b_, g_):
                oo, hh = blr_trainable(q_, k_, v_, w_, b_, g_, scale, h0=h0, cfg=cfg)
                return jnp.sum(oo * oo) + jnp.sum(hh * hh)

            def loss_ref(q_, k_, v_, w_, b_, g_):
                oo, hh, _ = R.forward_ref(q_, k_, v_, w_, b_, g_, scale, h0, cfg)
                return jnp.sum(oo * oo) + jnp.sum(hh * hh)

            gn = jax.grad(loss_new, argnums=(0, 1, 2, 3, 4, 5))(q, k, v, w, b, g)
            gr_ = jax.grad(loss_ref, argnums=(0, 1, 2, 3, 4, 5))(q, k, v, w, b, g)
            for nm, a_, b_ in zip(("dq", "dk", "dv", "dw", "db", "dg"), gn, gr_):
                e = rel_err(a_, b_)
                check(f"G8.grad.{nm}[{tag}]", e < 5e-2, e, 5e-2,
                      "-- аналитический custom_vjp vs autodiff того же forward")


# ===========================================================================
# G9 -- TPU: тайминги и свип вариантов
# ===========================================================================
def g9_sweep():
    log("\n" + "=" * 78)
    log("G9 -- TPU: тайминги и свип вариантов")
    log("=" * 78)
    if BACKEND != "tpu":
        info("G9 пропущен: только TPU")
        return
    bsz, H, nc, bt, D = 8, 6, 16, 256, 128
    L = nc * bt
    scale = 1.0 / math.sqrt(D)
    q, k, v, w, b, g = make_inputs(SEEDS[0], bsz, L, H)
    h0 = jnp.zeros((bsz, H, D, D), jnp.float32)
    base = cfg_of(bt=bt, bc=128, group=8)
    gc = R.chunk_gc(R.to_chunks(g, bsz, nc, H, D, bt))

    # Kernel A: score_bs x dot_mode
    for bs in (128, 64, 32):
        for mode in ("highest", "bf16x3"):
            c = base.with_(score_bs=bs, dot_mode=mode)
            try:
                timeit(lambda *a, _c=c: F.build_scores(*a, scale, _c),
                       q, k, b, gc, label=f"A pallas bs={bs} dot={mode}")
            except Exception as ex:
                log(f"    [A bs={bs} {mode}] {type(ex).__name__}: {ex}")
        c = base.with_(score_bs=bs)
        qr, kr, br = (R.to_chunks(t, bsz, nc, H, D, bt) for t in (q, k, b))
        try:
            timeit(lambda *a, _c=c: R.blr_scores_ref(*a, scale, _c),
                   qr, kr, br, gc, label=f"A xla    bs={bs} (H_EXEC)")
        except Exception as ex:
            log(f"    [A xla bs={bs}] {type(ex).__name__}: {ex}")

    # Kernel B: group x dot_mode
    Aqk, Akk = F.build_scores(q, k, b, gc, scale, base)
    for grp in (1, 4, 8, 16):
        for mode in ("highest", "bf16x3"):
            c = base.with_(group=grp, solve_dot_mode=mode)
            try:
                timeit(lambda X, _c=c: F.wy_solve(X, _c), Akk,
                       label=f"B H9 group={grp} dot={mode}")
            except Exception as ex:
                log(f"    [B grp={grp} {mode}] {type(ex).__name__}: {ex}")

    # Kernel D / B1: heads_per_cell
    A = F.wy_solve(Akk, base)
    wp, u, kg, qg, gcl = F.recompute_wy(q, k, v, w, b, gc, A, base)
    for hb in (1, 2, 3, 6):
        if H % hb:
            continue
        c = base.with_(heads_per_cell=hb)
        try:
            timeit(lambda *a, _c=c: F.inter_chunk_scan(*a, scale, h0, _c),
                   Aqk, wp, u, kg, qg, gcl, label=f"D pallas hb={hb}")
        except Exception as ex:
            log(f"    [D hb={hb}] {type(ex).__name__}: {ex}")
    timeit(lambda *a: R.inter_chunk_scan_ref(*a, scale, h0, base),
           Aqk, wp, u, kg, qg, gcl, label="D lax.scan (baseline)")

    # fused A+B
    for grp in (1, 4, 8):
        c = base.with_(group=grp)
        try:
            timeit(lambda *a, _c=c: F.build_scores_and_solve(*a, scale, _c),
                   q, k, b, gc, label=f"A+B fused group={grp}")
        except Exception as ex:
            log(f"    [A+B grp={grp}] {type(ex).__name__}: {ex}")

    # e2e fwd+bwd по backend'ам
    for backend in ("hybrid", "pallas", "xla"):
        c = base.with_(backend=backend)

        def fb(q_, k_, v_, w_, b_, g_, _c=c):
            return jax.grad(lambda *a: jnp.sum(
                blr_trainable(*a, scale, h0=h0, cfg=_c)[0] ** 2),
                argnums=(0, 1, 2, 3, 4, 5))(q_, k_, v_, w_, b_, g_)

        try:
            timeit(fb, q, k, v, w, b, g, n_iters=5,
                   label=f"E2E fwd+bwd backend={backend}")
        except Exception as ex:
            log(f"    [E2E {backend}] {type(ex).__name__}: {ex}")


# ===========================================================================
SECTIONS = [("G0", g0_config), ("G1", g1_kernel_a), ("G2", g2_causality),
            ("G3", g3_kernel_b), ("G4", g4_kernel_c), ("G5", g5_kernel_d),
            ("G6", g6_b1), ("G7", g7_b4), ("G8", g8_pipeline), ("G9", g9_sweep)]


def main():
    log(f"jax {jax.__version__} | backend = {BACKEND} | interpret = {INTERPRET}")
    REPORT["meta"] = {"backend": BACKEND, "jax": jax.__version__,
                      "interpret": INTERPRET,
                      "ts": time.strftime("%Y-%m-%d %H:%M:%S")}
    _save()
    for name, fn in SECTIONS:
        t0 = time.time()
        try:
            fn()
        except Exception as e:
            log(f"[SECTION FAILED] {name}: {type(e).__name__}: {e}")
            traceback.print_exc()
            REPORT["failures"].append(f"{name}: {type(e).__name__}: {e}")
            _save()
        log(f"--- {name} done in {time.time() - t0:.1f}s ---")
    log("\n" + "=" * 78)
    log(f"RESULT: {len(REPORT['failures'])} failure(s). Report: {REPORT_PATH}")
    for f in REPORT["failures"]:
        log(f"  [FAIL] {f}")
    log("=" * 78)
    _save()


if __name__ == "__main__":
    main()
