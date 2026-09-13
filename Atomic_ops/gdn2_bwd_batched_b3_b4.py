"""
gdn2_bwd_batched_b3_b4.py

Закрывает "Дыру 4" ревью mxu_utilization_hypothesis_suite.py: B3
(wy_dqkg_backward_pallas) и B4 (intra_backward_pallas) до сих пор не
имели батченых прототипов -- только non-batched baseline. Это самая
большая упущенная выгода по объёму (B3 -- единственный compute-bound
кернел на floor; B3+B4 вместе -- заметная доля backward), и одновременно
САМЫЙ рискованный код для батчинга (MB8_status_report.md, "Гипотеза E",
и Fwdbwd_14ms_roadmap.md §4 Фаза 3 явно откладывают его "напоследок").

ПРИНЦИП (тот же, что уже провалидирован для Kernel A --
gdn2_fwd_batched_fixed.py -- и Kernel B2 -- gdn2_bwd_batched_b2.py):
    grid=(bsz,H,n_chunks) -> grid=(bsz,H,n_groups), n_chunks/group
    батчируется ВНУТРИ тела кернела.

Для B3: математика извлечена в pure-function _compute_b3_single_chunk
(идентична _kernel_b3_body из gdn2_bwd.py построчно, только без ref
чтения/записи), и батченое тело -- python-цикл `for gi in range(group)`,
вызывающий эту функцию и пишущий результат ОДНИМ прямым срезом
`out_ref[0, 0, gi] = ...` (тот же паттерн, что уже используется в
_kernel_a_only_batched_body / build_and_solve_pallas_batched_fixed).

Для B4 сделан МАКСИМАЛЬНО консервативный выбор: тело _kernel_b4_body
скопировано ПОЧТИ ДОСЛОВНО и обёрнуто в `for gi in range(group)`, с
заменой фиксированного индекса `[0, 0, 0]` на `[0, 0, gi]` везде.
Внутренняя read-modify-write аккумуляция (`dq_ref[..., i0:i1] =
clip_acc(dq_ref[..., i0:i1] + ..., config)`) НЕ переписана на
`.at[].set()` -- это ТОТ ЖЕ паттерн прямой записи в срез ref, который
уже сейчас (grid=768, non-batched) реально исполняется в продакшене
через intra_backward_pallas и, по всем имеющимся данным, лежит на
Mosaic. Единственное мех. изменение -- добавленная group-ось; сама
формула НЕ переписывалась заново (в отличие от, скажем, H9 -- новый
алгоритм), что снижает шанс тихо сломать double-damping-чувствительную
математику, которая уже один раз содержала баг именно на стыке
B3->B4 (см. комментарий "FIX (double-damping bug)" в gdn2_pipeline.py).

ЧТО ЭТО НЕ ЗАМЕНЯЕТ:
  - НЕ подключено к gdn2_pipeline.py::_gdn2_core_bwd. НЕ трогайте
    production backward, пока Gate 1 (multi-seed vs
    gdn2_token_serial_reference ЧЕРЕЗ ВЕСЬ custom_vjp, включая
    экстремальные g/k/b) и Gate 2 (causality-via-perturbation) не
    пройдены на interpret=False (реальный TPU).
  - tests/test_b3_b4_batched_vs_nonbatched.py в этом наборе -- ТОЛЬКО
    Gate-1-lite: batched vs non-batched на interpret=True (CPU).
    Доказывает "математика не изменилась при добавлении group-оси",
    НЕ "лежит на реальном Mosaic" (тот же caveat, что и во всех прочих
    batched-прототипах этого репо).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from .configs import KernelConfig, DEFAULT_CONFIG, sanitize, clip_acc
from .configs import _reshape_to_chunks as _r2c
from .gdn2_bwd import _dL_pair_sum, _dR_pair_sum, _dgc_pair_sum

_HIGHEST = jax.lax.Precision.HIGHEST


# ===========================================================================
# B3 -- wy_dqkg_backward_pallas, батченая версия
# ===========================================================================
def _compute_b3_single_chunk(q_c, k_c, b_c, w_c, v_c, gc, A, h_pre, v_new, do, dv, dh_next,
                              *, scale: float, bt: int, wy_eps: float, config: KernelConfig):
    """Математика _kernel_b3_body (gdn2_bwd.py), 1:1, но БЕЗ ref
    чтения/записи -- принимает и возвращает plain jnp arrays для ОДНОГО
    чанка. Используется и в non-batched теле (через тонкую ref-обёртку,
    см. Примечание ниже), и здесь, в batched теле -- чтобы одна и та же
    формула не имела шанса разойтись с оригиналом при копипасте.

    Akk_ref в оригинальном _kernel_b3_body передаётся как аргумент, но
    НИГДЕ не читается внутри тела -- это подтверждено построчным чтением
    gdn2_bwd.py. Здесь Akk сознательно не принимается вовсе, чтобы не
    поддерживать fantom-параметр.
    """
    C = bt
    gc_last = gc[C - 1]

    kb_decayed = b_c * k_c * jnp.exp(gc)
    kg = k_c * jnp.exp(gc_last[None, :] - gc)
    qg = q_c * jnp.exp(gc)
    wv = w_c * v_c

    dqh_up = scale * do
    dqg = jnp.dot(dqh_up, h_pre.T, precision=_HIGHEST)

    dwh = -dv
    dw_pseudo = jnp.dot(dwh, h_pre.T, precision=_HIGHEST)
    du = dv

    dkg = jnp.dot(v_new, dh_next.T, precision=_HIGHEST)

    dA_from_w = jnp.dot(dw_pseudo, kb_decayed.T, precision=_HIGHEST)
    dkb_decayed = jnp.dot(A.T, dw_pseudo, precision=_HIGHEST)

    dA_from_u = jnp.dot(du, wv.T, precision=_HIGHEST)
    dwv = jnp.dot(A.T, du, precision=_HIGHEST)

    dA_total = dA_from_w + dA_from_u
    dA_total = sanitize(dA_total, config)

    idx = jnp.arange(C)
    strict = (idx[:, None] > idx[None, :]).astype(jnp.float32)

    tmp = jnp.dot(dA_total, A.T, precision=_HIGHEST)
    tmp = sanitize(tmp, config)
    dAkk_raw = -jnp.dot(A.T, tmp, precision=_HIGHEST)
    # Тот же ЕДИНСТВЕННЫЙ (1-wy_eps) chain-rule множитель, что и в
    # non-batched _kernel_b3_body -- см. длинный комментарий там же
    # про double-damping bug. НЕ добавляйте это где-то ещё вниз по
    # потоку (B4/gdn2_pipeline.py).
    dAkk_raw = dAkk_raw * (1.0 - wy_eps)
    dAkk = dAkk_raw * strict

    dk_from_kb = dkb_decayed * jnp.exp(gc) * b_c
    db = dkb_decayed * jnp.exp(gc) * k_c
    dgc_from_kb = dkb_decayed * kb_decayed

    dx = dkg * kg
    dk_from_kg = dkg * jnp.exp(gc_last[None, :] - gc)
    dgc_from_kg = -dx
    dgc_last_contrib = jnp.sum(dx, axis=0)

    dq = dqg * jnp.exp(gc)
    dgc_from_qg = dqg * qg

    dw = dwv * v_c
    dv_raw = dwv * w_c

    dk = dk_from_kb + dk_from_kg
    dgc = dgc_from_kb + dgc_from_qg + dgc_from_kg

    decay_h_row = jnp.exp(gc_last)
    dgc_last_from_decay = decay_h_row * jnp.sum(dh_next * h_pre, axis=-1)
    dgc_last_total = dgc_last_contrib + dgc_last_from_decay

    row_mask = (idx == (C - 1)).astype(jnp.float32)[:, None]
    dgc = dgc + row_mask * dgc_last_total[None, :]

    return (
        sanitize(dq, config), sanitize(dk, config), sanitize(db, config),
        sanitize(dw, config), sanitize(dv_raw, config), sanitize(dgc, config),
        sanitize(dAkk, config),
    )


def _kernel_b3_body_batched(q_ref, k_ref, b_ref, w_ref, v_ref, gc_ref, a_ref, akk_ref,
                             hpre_ref, vnew_ref, do_ref, dv_ref, dhnext_ref,
                             dq_ref, dk_ref, db_ref, dw_ref, dvraw_ref, dgc_ref, dakk_ref,
                             *, scale: float, bt: int, wy_eps: float, config: KernelConfig,
                             group: int):
    # akk_ref принят для совпадения сигнатуры/BlockSpec с non-batched
    # версией (см. docstring _compute_b3_single_chunk) -- не читается.
    del akk_ref
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
            scale=scale, bt=bt, wy_eps=wy_eps, config=config,
        )
        # Прямая запись срезом -- НЕ .at[].set() (см. MB8_status_report.md
        # про то, почему это важно на Pallas TPU).
        dq_ref[0, 0, gi] = dq
        dk_ref[0, 0, gi] = dk
        db_ref[0, 0, gi] = db
        dw_ref[0, 0, gi] = dw
        dvraw_ref[0, 0, gi] = dv_raw
        dgc_ref[0, 0, gi] = dgc
        dakk_ref[0, 0, gi] = dAkk


def wy_dqkg_backward_pallas_batched(q, k, b, w, v, gc, A, Akk, h_pre_all, v_new_all,
                                     do, dv, dh_next_all, scale,
                                     config: KernelConfig = DEFAULT_CONFIG,
                                     group: int | None = None, interpret: bool = False):
    """Батченая версия gdn2_bwd.wy_dqkg_backward_pallas (B3).

    ВНИМАНИЕ (см. модульный docstring): НЕ подключать в
    gdn2_pipeline.py до полного Gate 1/2 на interpret=False.
    Сигнатура намеренно идентична non-batched версии (плюс `group`,
    `interpret`), чтобы быть drop-in заменой после валидации.
    """
    bsz, H, n_chunks, _BT, D = q.shape
    if group is None:
        group = config.b_batch_group if config.b_batch_group is not None else n_chunks
    if n_chunks % group != 0:
        raise ValueError(
            f"wy_dqkg_backward_pallas_batched: n_chunks={n_chunks} должен "
            f"делиться на group={group}."
        )
    n_groups = n_chunks // group

    grid = (bsz, H, n_groups)
    io_spec = pl.BlockSpec((1, 1, group, config.bt, D), lambda i, h, gi: (i, h, gi, 0, 0))
    score_spec = pl.BlockSpec((1, 1, group, config.bt, config.bt), lambda i, h, gi: (i, h, gi, 0, 0))
    h_spec = pl.BlockSpec((1, 1, group, D, D), lambda i, h, gi: (i, h, gi, 0, 0))

    dq, dk, db, dw, dv_raw, dgc, dAkk = pl.pallas_call(
        lambda *refs: _kernel_b3_body_batched(
            *refs, scale=scale, bt=config.bt, wy_eps=config.wy_eps, config=config, group=group,
        ),
        grid=grid,
        in_specs=[io_spec, io_spec, io_spec, io_spec, io_spec, io_spec,
                  score_spec, score_spec, h_spec, io_spec, io_spec, io_spec, h_spec],
        out_specs=[io_spec, io_spec, io_spec, io_spec, io_spec, io_spec, score_spec],
        out_shape=[
            jax.ShapeDtypeStruct((bsz, H, n_chunks, config.bt, D), jnp.float32),
            jax.ShapeDtypeStruct((bsz, H, n_chunks, config.bt, D), jnp.float32),
            jax.ShapeDtypeStruct((bsz, H, n_chunks, config.bt, D), jnp.float32),
            jax.ShapeDtypeStruct((bsz, H, n_chunks, config.bt, D), jnp.float32),
            jax.ShapeDtypeStruct((bsz, H, n_chunks, config.bt, D), jnp.float32),
            jax.ShapeDtypeStruct((bsz, H, n_chunks, config.bt, D), jnp.float32),
            jax.ShapeDtypeStruct((bsz, H, n_chunks, config.bt, config.bt), jnp.float32),
        ],
        compiler_params=pltpu.CompilerParams(vmem_limit_bytes=160 * 1024 * 1024),
        interpret=interpret,
    )(q, k, b, w, v, gc, A, Akk, h_pre_all, v_new_all, do, dv, dh_next_all)

    return dict(dq=dq, dk=dk, db=db, dw=dw, dv_raw=dv_raw, dgc=dgc, dAkk=dAkk)


# ===========================================================================
# B4 -- intra_backward_pallas, батченая версия
# ===========================================================================
def _kernel_b4_body_batched(q_ref, k_ref, b_ref, g_ref, daqk_ref, dakk_ref,
                             dq_ref, dk_ref, db_ref, dgc_ref, *, scale: float, bt: int, bc: int,
                             n_sub: int, use_centering: bool, config: KernelConfig, group: int):
    """МЕХАНИЧЕСКАЯ обёртка _kernel_b4_body (gdn2_bwd.py): тело --
    построчная копия оригинала внутри `for gi in range(group)`, с
    заменой фиксированного `[0, 0, 0]` на `[0, 0, gi]`. Формула НЕ
    переписана заново -- намеренно, это самый рискованный кернел в
    репо (double-damping история + scatter-подобный accumulate при
    use_centering=True), и минимальный diff от уже проверенного кода
    снижает шанс внести новый баг при батчинге."""
    for gi in range(group):
        q_full = q_ref[0, 0, gi].astype(jnp.float32)
        k_full = k_ref[0, 0, gi].astype(jnp.float32)
        b_full = b_ref[0, 0, gi].astype(jnp.float32)
        g_raw = g_ref[0, 0, gi].astype(jnp.float32)
        dAqk = daqk_ref[0, 0, gi].astype(jnp.float32)
        dAkk = dakk_ref[0, 0, gi].astype(jnp.float32)

        bt_idx = jnp.arange(bt)
        tril_ones_bt = (bt_idx[:, None] >= bt_idx[None, :]).astype(jnp.float32)
        gc = jnp.dot(tril_ones_bt, g_raw, precision=_HIGHEST)

        bk_full = b_full * k_full

        dq_ref[0, 0, gi] = jnp.zeros_like(q_full)
        dk_ref[0, 0, gi] = jnp.zeros_like(k_full)
        db_ref[0, 0, gi] = jnp.zeros_like(k_full)
        dgc_ref[0, 0, gi] = jnp.zeros_like(g_raw)

        if use_centering:
            D = q_full.shape[-1]
            dgn_i_acc = [jnp.zeros((D,), dtype=jnp.float32) for _ in range(n_sub)]
            dgn_j_acc = [jnp.zeros((D,), dtype=jnp.float32) for _ in range(n_sub)]

        for si in range(n_sub):
            for sj in range(si + 1):
                i0, i1 = si * bc, (si + 1) * bc
                j0, j1 = sj * bc, (sj + 1) * bc

                q_i = q_full[i0:i1]
                k_i = k_full[i0:i1]
                k_j = k_full[j0:j1]
                b_i = b_full[i0:i1]
                bk_i = bk_full[i0:i1]
                gc_i = gc[i0:i1]
                gc_j = gc[j0:j1]

                dM_qk = dAqk[i0:i1, j0:j1]
                dM_kk = dAkk[i0:i1, j0:j1]
                if si == sj:
                    idx = jnp.arange(bc)
                    causal = (idx[:, None] >= idx[None, :]).astype(jnp.float32)
                    strict = (idx[:, None] > idx[None, :]).astype(jnp.float32)
                    dM_qk = dM_qk * causal
                    dM_kk = dM_kk * strict

                if use_centering:
                    gn_i = gc[i0]
                    gn_j = gc[j0]

                    gq_i_raw = gc_i - gn_i[None, :]
                    gk_j_raw = gn_j[None, :] - gc_j
                    gcross_raw = gn_i - gn_j

                    gq_i = jnp.clip(gq_i_raw, -20.0, 20.0)
                    gk_j = jnp.clip(gk_j_raw, -20.0, 20.0)
                    gcross = jnp.clip(gcross_raw, -20.0, 20.0)

                    clipmask_q = ((gq_i_raw >= -20.0) & (gq_i_raw <= 20.0)).astype(jnp.float32)
                    clipmask_k = ((gk_j_raw >= -20.0) & (gk_j_raw <= 20.0)).astype(jnp.float32)
                    clipmask_cross = ((gcross_raw >= -20.0) & (gcross_raw <= 20.0)).astype(jnp.float32)

                    eq_i = jnp.exp(gq_i)
                    ek_j = jnp.exp(gk_j)
                    ecross = jnp.exp(gcross)

                    q_scaled = q_i * eq_i
                    k_scaled = (k_j * ek_j) * ecross[None, :]
                    bk_scaled = bk_i * eq_i

                    dq_scaled = scale * jnp.dot(dM_qk, k_scaled, precision=_HIGHEST)
                    dk_scaled_qk = scale * jnp.dot(dM_qk.T, q_scaled, precision=_HIGHEST)
                    dbk_scaled = jnp.dot(dM_kk, k_scaled, precision=_HIGHEST)
                    dk_scaled_kk = jnp.dot(dM_kk.T, bk_scaled, precision=_HIGHEST)
                    dk_scaled = dk_scaled_qk + dk_scaled_kk

                    dk_j_from_scaled = dk_scaled * ek_j * ecross[None, :]
                    dB = dk_scaled * k_j * ecross[None, :]
                    dgk_j = (dB * ek_j) * clipmask_k
                    dC = jnp.sum(dk_scaled * k_j * ek_j, axis=0)
                    dgcross = (dC * ecross) * clipmask_cross

                    d_eq_i_total = dq_scaled * q_i + dbk_scaled * bk_i
                    dgq_i = (d_eq_i_total * eq_i) * clipmask_q
                    dq_i_from_scaled = dq_scaled * eq_i
                    dbk_i_from_scaled = dbk_scaled * eq_i

                    dq_ref[0, 0, gi, i0:i1] = clip_acc(dq_ref[0, 0, gi, i0:i1] + dq_i_from_scaled, config)
                    db_ref[0, 0, gi, i0:i1] = clip_acc(db_ref[0, 0, gi, i0:i1] + dbk_i_from_scaled, config)
                    dk_ref[0, 0, gi, j0:j1] = clip_acc(dk_ref[0, 0, gi, j0:j1] + dk_j_from_scaled, config)
                    dgc_ref[0, 0, gi, i0:i1] = clip_acc(dgc_ref[0, 0, gi, i0:i1] + dgq_i, config)
                    dgc_ref[0, 0, gi, j0:j1] = clip_acc(dgc_ref[0, 0, gi, j0:j1] - dgk_j, config)

                    dgn_i_acc[si] = dgn_i_acc[si] + jnp.sum(-dgq_i, axis=0) + dgcross
                    dgn_j_acc[sj] = dgn_j_acc[sj] + jnp.sum(dgk_j, axis=0) - dgcross
                else:
                    decay_diff = gc_i[:, None, :] - gc_j[None, :, :]
                    clipmask = ((decay_diff >= -20.0) & (decay_diff <= 20.0)).astype(jnp.float32)
                    edecay = jnp.exp(jnp.clip(decay_diff, -20.0, 20.0))

                    L_qk = scale * q_i
                    R_qk = k_j
                    dL_qk = _dL_pair_sum(dM_qk, edecay, R_qk)
                    dR_qk = _dR_pair_sum(dM_qk, edecay, L_qk)
                    dgc_i_qk, dgc_j_qk = _dgc_pair_sum(dM_qk, edecay, L_qk, R_qk, clipmask)

                    L_kk = bk_i
                    R_kk = k_j
                    dL_kk = _dL_pair_sum(dM_kk, edecay, R_kk)
                    dR_kk = _dR_pair_sum(dM_kk, edecay, L_kk)
                    dgc_i_kk, dgc_j_kk = _dgc_pair_sum(dM_kk, edecay, L_kk, R_kk, clipmask)

                    dq_ref[0, 0, gi, i0:i1] = clip_acc(dq_ref[0, 0, gi, i0:i1] + dL_qk * scale, config)
                    db_ref[0, 0, gi, i0:i1] = clip_acc(db_ref[0, 0, gi, i0:i1] + dL_kk, config)
                    dk_ref[0, 0, gi, j0:j1] = clip_acc(dk_ref[0, 0, gi, j0:j1] + dR_qk + dR_kk, config)
                    dgc_ref[0, 0, gi, i0:i1] = clip_acc(dgc_ref[0, 0, gi, i0:i1] + dgc_i_qk + dgc_i_kk, config)
                    dgc_ref[0, 0, gi, j0:j1] = clip_acc(dgc_ref[0, 0, gi, j0:j1] + dgc_j_qk + dgc_j_kk, config)

        if use_centering:
            for si in range(n_sub):
                i0 = si * bc
                dgc_ref[0, 0, gi, i0] = clip_acc(dgc_ref[0, 0, gi, i0] + dgn_i_acc[si], config)
            for sj in range(n_sub):
                j0 = sj * bc
                dgc_ref[0, 0, gi, j0] = clip_acc(dgc_ref[0, 0, gi, j0] + dgn_j_acc[sj], config)

        dbk_final = db_ref[0, 0, gi]
        dk_final = dk_ref[0, 0, gi] + dbk_final * b_full
        db_final = dbk_final * k_full
        dq_final = dq_ref[0, 0, gi]
        dgc_final = dgc_ref[0, 0, gi]

        dq_ref[0, 0, gi] = sanitize(dq_final, config)
        dk_ref[0, 0, gi] = sanitize(dk_final, config)
        db_ref[0, 0, gi] = sanitize(db_final, config)
        dgc_ref[0, 0, gi] = sanitize(dgc_final, config)


def intra_backward_pallas_batched(dAqk, dAkk, q, k, b, g, scale,
                                   config: KernelConfig = DEFAULT_CONFIG,
                                   group: int | None = None, interpret: bool = False):
    """Батченая версия gdn2_bwd.intra_backward_pallas (B4).

    ВНИМАНИЕ (см. модульный docstring): НЕ подключать в
    gdn2_pipeline.py до полного Gate 1/2 на interpret=False. При
    use_centering=True несёт дополнительный риск (dgn_i_acc/dgn_j_acc
    scatter-подобный паттерн) сверх общего для B3/B4 риска.
    """
    bsz, L, H, D = q.shape
    n_chunks = L // config.bt
    if group is None:
        group = config.b_batch_group if config.b_batch_group is not None else n_chunks
    if n_chunks % group != 0:
        raise ValueError(
            f"intra_backward_pallas_batched: n_chunks={n_chunks} должен "
            f"делиться на group={group}."
        )
    n_groups = n_chunks // group

    def reshape_in(t):
        return _r2c(t, bsz, n_chunks, H, D, config.bt)

    q_r, k_r, b_r, g_r = map(reshape_in, (q, k, b, g))

    grid = (bsz, H, n_groups)
    io_spec = pl.BlockSpec((1, 1, group, config.bt, D), lambda i, h, gi: (i, h, gi, 0, 0))
    score_spec = pl.BlockSpec((1, 1, group, config.bt, config.bt), lambda i, h, gi: (i, h, gi, 0, 0))

    dq, dk, db, dgc = pl.pallas_call(
        lambda *refs: _kernel_b4_body_batched(
            *refs, scale=scale, bt=config.bt, bc=config.bc, n_sub=config.n_sub,
            use_centering=config.use_centering, config=config, group=group,
        ),
        grid=grid,
        in_specs=[io_spec, io_spec, io_spec, io_spec, score_spec, score_spec],
        out_specs=[io_spec, io_spec, io_spec, io_spec],
        out_shape=[
            jax.ShapeDtypeStruct((bsz, H, n_chunks, config.bt, D), jnp.float32),
            jax.ShapeDtypeStruct((bsz, H, n_chunks, config.bt, D), jnp.float32),
            jax.ShapeDtypeStruct((bsz, H, n_chunks, config.bt, D), jnp.float32),
            jax.ShapeDtypeStruct((bsz, H, n_chunks, config.bt, D), jnp.float32),
        ],
        compiler_params=pltpu.CompilerParams(vmem_limit_bytes=200 * 1024 * 1024),
        interpret=interpret,
    )(q_r, k_r, b_r, g_r, dAqk, dAkk)

    return dq, dk, db, dgc
