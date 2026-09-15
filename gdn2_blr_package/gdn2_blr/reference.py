"""
gdn2_blr.reference -- чисто-JAX эталон каждой стадии.

Назначение:
  1. Источник истины для Gate 1 (Pallas vs этот файл, CPU interpret=True).
  2. Источник истины для градиентов: всё здесь autodiff-прозрачно, поэтому
     `jax.grad(forward_ref)` -- эталон для custom_vjp пайплайна.
  3. Кандидат на H_EXEC: батченые версии здесь -- это ровно те XLA-пути,
     которые внешний прецедент (MaxText #4348) рекомендует оставить в XLA
     вместо Pallas-грида.

ВАЖНО ПРО SANITIZE. Все стадии применяют тот же sanitize(., cfg.clip), что
и Pallas-тела. Без этого сравнение Pallas vs reference расходится на
входах, где значения выходят за clip -- ровно это дало два FAIL
`S1.pallas_vs_jax[extreme_mixed_sign_g] = 0.9946` в прогоне v3: математика
совпадала, а клип применялся только с одной стороны.

Все тензоры в чанк-разметке: (bsz, H, n_chunks, bt, D), ведущие оси
произвольны -- функции написаны через `...`.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from .config import BLRConfig
from .precision import (HIGHEST, make_einsum, exp_nonpos, exp_clipped,
                        sanitize)


# ---------------------------------------------------------------------------
# разметка
# ---------------------------------------------------------------------------
def to_chunks(t, bsz, n_chunks, H, D, bt):
    t = t.reshape(bsz, n_chunks, bt, H, D)
    return jnp.moveaxis(t, (1, 3), (2, 1))


def from_chunks(t, bsz, n_chunks, bt, H, D):
    t2 = jnp.moveaxis(t, (1, 2, 3), (3, 1, 2))
    return t2.reshape(bsz, n_chunks * bt, H, D)


def chunk_gc(g_r):
    """gc = cumsum(g) ВНУТРИ чанка. Считается ОДИН раз и прокидывается во
    все кернелы. В старом коде пересчитывался tril-матмулом в Kernel A, C,
    B4 и ещё раз в пайплайне -- четыре матмула bt^2*D на ровном месте."""
    return jnp.cumsum(g_r.astype(jnp.float32), axis=-2)


def causal_masks(T):
    i = jnp.arange(T)
    return ((i[:, None] >= i[None, :]).astype(jnp.float32),
            (i[:, None] > i[None, :]).astype(jnp.float32))


# ---------------------------------------------------------------------------
# Kernel A -- BLR scores
# ---------------------------------------------------------------------------
def blr_scores_ref(q, k, b, gc, scale, cfg: BLRConfig):
    """(..., T, D) -> Aqk, Akk (..., T, T).

    Для блока запросов P=[p0,p1) берётся ОДНА reference-точка r=gc[p0] и
    точное разложение на ДВЕ ноги (третьей, cross-ноги, не существует --
    именно она была источником утечки three-leg):

        gc_i - gc_j = (gc_i - r) + (r - gc_j)

    При g<=0 обе ноги <=0 для j<p0 => exp in (0,1] => переполнение
    невозможно. Диагональный блок [p0,p1)x[p0,p1) считается
    неф акторизованно с ОДНИМ клипом на истинную разность -- математика
    production non-centered пути, измеренно точная.
    """
    T, bs, n_sub = cfg.bt, cfg.score_bs, cfg.n_sub
    dc = cfg.diff_clip
    ein = make_einsum(cfg.dot_mode)
    causal, strict = causal_masks(T)
    idx = jnp.arange(T)
    bk = b * k

    rows_q, rows_k = [], []
    for p in range(n_sub):
        p0, p1 = p * bs, (p + 1) * bs
        r = gc[..., p0, :]
        a, _ = exp_nonpos(gc[..., p0:p1, :] - r[..., None, :])
        qt = q[..., p0:p1, :] * a
        bkt = bk[..., p0:p1, :] * a

        if p0 > 0:
            c, _ = exp_nonpos(r[..., None, :] - gc)
            kt = k * c
            before = (idx < p0).astype(jnp.float32)
            off_q = scale * ein("...id,...jd->...ij", qt, kt) * before
            off_k = ein("...id,...jd->...ij", bkt, kt) * before
        else:
            off_q = jnp.zeros(qt.shape[:-1] + (T,), jnp.float32)
            off_k = off_q

        gd = gc[..., p0:p1, :]
        E, _ = exp_clipped(gd[..., :, None, :] - gd[..., None, :, :], -dc, dc)
        dia_q = scale * jnp.einsum("...id,...ijd,...jd->...ij",
                                   q[..., p0:p1, :], E, k[..., p0:p1, :],
                                   precision=HIGHEST)
        dia_k = jnp.einsum("...id,...ijd,...jd->...ij",
                           bk[..., p0:p1, :], E, k[..., p0:p1, :],
                           precision=HIGHEST)
        rows_q.append(off_q + _place(dia_q, p0, p1, T))
        rows_k.append(off_k + _place(dia_k, p0, p1, T))

    Aqk = jnp.concatenate(rows_q, axis=-2) * causal
    Akk = jnp.concatenate(rows_k, axis=-2) * strict
    return sanitize(Aqk, cfg.clip), sanitize(Akk, cfg.clip)


def _place(block, p0, p1, T):
    """(..., bs, bs) -> (..., bs, T), блок стоит в колонках [p0, p1).
    Конкатенация ЗНАЧЕНИЙ по lane-оси -- тот же паттерн, что уже работает
    в production `_block_solve` (сборка из (mb,mb) блоков)."""
    parts = []
    if p0 > 0:
        parts.append(jnp.zeros(block.shape[:-1] + (p0,), jnp.float32))
    parts.append(block)
    if T - p1 > 0:
        parts.append(jnp.zeros(block.shape[:-1] + (T - p1,), jnp.float32))
    return jnp.concatenate(parts, axis=-1) if len(parts) > 1 else block


# ---------------------------------------------------------------------------
# Kernel B -- инверсии
# ---------------------------------------------------------------------------
def row_by_row_inverse(Akk, eps: float):
    """A = (I + (1-eps)Akk)^-1 построчной прямой подстановкой. Медленно,
    но это ground truth для лесенки H9."""
    C = Akk.shape[-1]
    eye = jnp.eye(C, dtype=jnp.float32)
    Ad = Akk * (1.0 - eps)

    def step(A, i):
        t_row = jnp.take(Ad, i, axis=-2)
        contrib = jnp.einsum("...j,...jk->...k", t_row, A, precision=HIGHEST)
        new_row = eye[i] - contrib
        A = jax.lax.dynamic_update_slice_in_dim(A, new_row[..., None, :], i, axis=-2)
        return A, None

    A0 = jnp.zeros(Akk.shape, jnp.float32)
    A, _ = jax.lax.scan(step, A0, jnp.arange(C))
    return A


def micro_base_inverse(T_mb, mb: int):
    """База лесенки: (mb,mb) прямой подстановкой, ведущие оси произвольны.
    T_mb должна быть УЖЕ демпфирована на (1-eps)."""
    idx = jnp.arange(mb)

    def body(i, A):
        oh = (idx == i).astype(jnp.float32)
        t_row = jnp.sum(T_mb * oh[:, None], axis=-2)
        contrib = jnp.sum(t_row[..., :, None] * A, axis=-2)
        new_row = oh - contrib
        col = oh[:, None]
        return A * (1.0 - col) + col * new_row[..., None, :]

    return jax.lax.fori_loop(0, mb, body, jnp.zeros(T_mb.shape, jnp.float32))


def ladder_inverse(S, eps: float, C: int, base: int, mode: str = "highest"):
    """H9: block-doubling. X_2b = [[X_t, 0], [-X_b S_l X_t, X_b]].

    Степени S никогда не образуются (они переполняют f32, как только
    |S|>1), поэтому промежуточные величины остаются в масштабе истинной
    инверсии. Длина цепочки: C -> 2*log2(C/base) матмулов.
    """
    assert C % base == 0 and (base & (base - 1)) == 0
    nb = C // base
    assert nb & (nb - 1) == 0, f"C/base={nb} must be a power of 2"
    ein = make_einsum(mode)
    Se = S * (1.0 - eps)
    X = [micro_base_inverse(Se[..., m * base:(m + 1) * base,
                               m * base:(m + 1) * base], base)
         for m in range(nb)]
    b = base
    while b < C:
        n2, new = 2 * b, []
        for m in range(C // n2):
            top, bot = X[2 * m], X[2 * m + 1]
            i0 = m * n2
            S_l = Se[..., i0 + b:i0 + n2, i0:i0 + b]
            mid = ein("...ij,...jk->...ik", S_l, top)
            ll = -ein("...ij,...jk->...ik", bot, mid)
            z = jnp.zeros_like(ll)
            new.append(jnp.concatenate(
                [jnp.concatenate([top, z], axis=-1),
                 jnp.concatenate([ll, bot], axis=-1)], axis=-2))
        X, b = new, n2
    return X[0]


# ---------------------------------------------------------------------------
# Kernel C -- recompute
# ---------------------------------------------------------------------------
def recompute_wy_ref(q, k, v, w, b, gc, A, cfg: BLRConfig):
    """ФИКС H0.6: все три экспоненты идут через min(.,0).
    Замерено на mixed-sign входе: без клипа 22.4% kg и 19.8% qg упирались
    в границу sanitize, то есть overflow был и маскировался санитайзером
    под 'конечное значение'."""
    ein = make_einsum(cfg.dot_mode)
    egc, _ = exp_nonpos(gc)
    kb = b * k * egc
    wp = ein("...ij,...jd->...id", A, kb)
    u = ein("...ij,...jd->...id", A, w * v)
    gc_last = gc[..., -1, :]
    ekg, _ = exp_nonpos(gc_last[..., None, :] - gc)
    kg = k * ekg
    qg = q * egc
    c = cfg.clip
    return (sanitize(wp, c), sanitize(u, c), sanitize(kg, c),
            sanitize(qg, c), gc_last)


# ---------------------------------------------------------------------------
# Kernel D -- inter-chunk scan
# ---------------------------------------------------------------------------
def inter_chunk_scan_ref(Aqk, wp, u, kg, qg, gc_last, scale, h0, cfg: BLRConfig):
    """Возвращает (o_chunks, h_final, h_pre_all, v_new_all) -- residuals
    нужны backward'у, поэтому эмитятся сразу (как в Tokamax/MaxText, где
    T^-1 и state кэшируются в residuals, а не пересчитываются)."""
    ein = make_einsum(cfg.dot_mode)
    to_scan = tuple(jnp.moveaxis(x, 2, 0) for x in (Aqk, wp, u, kg, qg, gc_last))
    c = cfg.clip

    def step(h, xs):
        Aq, wp_, u_, kg_, qg_, gl = xs
        wh = ein("...id,...dv->...iv", wp_, h)
        v_new = u_ - wh
        qh = ein("...id,...dv->...iv", qg_, h)
        intra = ein("...ij,...jv->...iv", Aq, v_new)
        o = scale * qh + intra
        dec, _ = exp_nonpos(gl)
        write = ein("...id,...iv->...dv", kg_, v_new)
        h_new = sanitize(h * dec[..., :, None] + write, c)
        return h_new, (sanitize(o, c), h, v_new)

    h_final, (o_s, hpre_s, vnew_s) = jax.lax.scan(step, h0, to_scan)
    return (jnp.moveaxis(o_s, 0, 2), h_final,
            jnp.moveaxis(hpre_s, 0, 2), jnp.moveaxis(vnew_s, 0, 2))


# ---------------------------------------------------------------------------
# полный forward (autodiff-прозрачный) -- эталон градиентов
# ---------------------------------------------------------------------------
def forward_ref(q, k, v, w, b, g, scale, h0, cfg: BLRConfig,
                use_ladder: bool = True):
    bsz, L, H, D = q.shape
    nc = L // cfg.bt
    qr, kr, vr, wr, br, gr = (to_chunks(t, bsz, nc, H, D, cfg.bt)
                              for t in (q, k, v, w, b, g))
    gc = chunk_gc(gr)
    Aqk, Akk = blr_scores_ref(qr, kr, br, gc, scale, cfg)
    if use_ladder:
        A = ladder_inverse(Akk, cfg.wy_eps, cfg.bt, cfg.mb, cfg.solve_dot_mode)
    else:
        A = row_by_row_inverse(Akk, cfg.wy_eps)
    A = sanitize(A, cfg.clip)
    wp, u, kg, qg, gc_last = recompute_wy_ref(qr, kr, vr, wr, br, gc, A, cfg)
    o_ch, h_final, hpre, vnew = inter_chunk_scan_ref(
        Aqk, wp, u, kg, qg, gc_last, scale, h0, cfg)
    o = from_chunks(o_ch, bsz, nc, cfg.bt, H, D)
    res = dict(gc=gc, Aqk=Aqk, Akk=Akk, A=A, w_pseudo=wp, u=u, kg=kg,
               qg=qg, gc_last=gc_last, h_pre_all=hpre, v_new_all=vnew)
    return o, h_final, res


# ---------------------------------------------------------------------------
# token-serial ground truth
# ---------------------------------------------------------------------------
def token_serial_ref(q, k, v, g, b, w, scale, h0=None):
    bsz, L, H, D = q.shape
    Dv = v.shape[-1]
    if h0 is None:
        h0 = jnp.zeros((bsz, H, D, Dv), jnp.float32)
    alpha = jnp.exp(jnp.minimum(g.astype(jnp.float32), 0.0))

    def step(h, xs):
        q_t, k_t, v_t, a_t, b_t, w_t = xs
        h = h * a_t[..., :, None]
        bk_t = (b_t * k_t).astype(jnp.float32)
        erase = jnp.einsum("bhd,bhdv->bhv", bk_t, h, precision=HIGHEST)
        v_new = (w_t * v_t).astype(jnp.float32) - erase
        h = h + jnp.einsum("bhd,bhv->bhdv", k_t.astype(jnp.float32), v_new,
                           precision=HIGHEST)
        o_t = jnp.einsum("bhdv,bhd->bhv", h, (q_t * scale).astype(jnp.float32),
                         precision=HIGHEST)
        return h, o_t

    to_scan = tuple(jnp.moveaxis(x, 1, 0) for x in (q, k, v, alpha, b, w))
    h_final, o_s = jax.lax.scan(step, h0, to_scan)
    return jnp.moveaxis(o_s, 0, 1), h_final
