"""
Atomic_ops/gdn2_fwd_batched.py

MB8: batched Kernel B (wy_solve) и fused Kernel A+B -- альтернативные
entrypoint'ы, устраняющие grid-dispatch overhead
(bsz*H*n_chunks=768 grid-точек -> bsz*H*n_groups).

НЕ заменяет wy_solve_pallas / build_chunk_scores_pallas в gdn2_fwd.py --
это отдельный путь для контролируемой A/B-валидации (Gate 1/2) перед
интеграцией в KAGGLE_* пресеты, как того требует MB8_status_report.md.

Инвариант: math ИДЕНТИЧНА _block_solve/_micro_forward_substitution из
gdn2_fwd.py, единственное отличие -- ведущая batch-ось `group` вместо
скалярной обработки одного чанка за вызов кернела. НЕ используется
.at[].set() (не lowерится в Pallas TPU, см. отчёт) -- вместо scatter
используется select через onehot-маску (уже был этот паттерн в
_micro_forward_substitution, просто расширен на batch-ось).

ВНИМАНИЕ: backward (custom_vjp) для batched-пути НЕ реализован в этом
патче -- только forward. Значит НЕЛЬЗЯ пропускать batched-путь через
gdn2_pallas_forward_trainable / jax.grad, пока B1-B5 backward для него
не написан отдельно. Годится для: (a) forward-only инференса, (b)
измерения chunk-A/B correctness и speed в рамках Gate 1/2 ниже.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from .configs import KernelConfig, DEFAULT_CONFIG, sanitize
from .gdn2_fwd import _reshape_to_chunks, _kernel_a_body

_HIGHEST = jax.lax.Precision.HIGHEST


# ==========================================================================
# Batched Kernel B
# ==========================================================================
def _micro_forward_substitution_batched(T_mb, mb: int, eps: float, config: KernelConfig):
    """Тот же алгоритм, что _micro_forward_substitution в gdn2_fwd.py,
    но с ведущей batch-осью `group` вместо скалярного тела. T_mb: (group, mb, mb)."""
    group = T_mb.shape[0]
    idx = jnp.arange(mb)
    T_mb = T_mb * (1.0 - eps)

    def body(i, A):
        onehot_i = (idx == i).astype(jnp.float32)              # (mb,)
        t_row = jnp.sum(T_mb * onehot_i[None, :, None], axis=1)  # (group, mb)
        contrib = jnp.sum(t_row[:, :, None] * A, axis=1)         # (group, mb)
        new_row = onehot_i[None, :] - contrib                    # (group, mb)
        new_row = sanitize(new_row, config)
        mask_col = onehot_i[None, :, None]                       # (1, mb, 1)
        A = A * (1.0 - mask_col) + mask_col * new_row[:, None, :]
        return A

    A0 = jnp.zeros((group, mb, mb), dtype=jnp.float32)
    return jax.lax.fori_loop(0, mb, body, A0)


def _block_solve_batched(T_full, config: KernelConfig, group: int):
    """Батч-версия _block_solve. T_full: (group, C, C) -> (group, C, C)."""
    N_MICRO = config.n_micro
    MB = config.mb
    eps = config.wy_eps
    blocks = [[None] * N_MICRO for _ in range(N_MICRO)]

    for m in range(N_MICRO):
        T_mm = T_full[:, m * MB:(m + 1) * MB, m * MB:(m + 1) * MB]
        A_mm = sanitize(_micro_forward_substitution_batched(T_mm, MB, eps, config), config)
        blocks[m][m] = A_mm

        for n in range(m - 1, -1, -1):
            acc = jnp.zeros((group, MB, MB), dtype=jnp.float32)
            for k in range(n, m):
                T_mk = T_full[:, m * MB:(m + 1) * MB, k * MB:(k + 1) * MB]
                A_kn = blocks[k][n]
                contrib = jnp.einsum("gij,gjk->gik", T_mk * (1.0 - eps), A_kn, precision=_HIGHEST)
                acc = sanitize(acc + contrib, config)
            A_mn = -jnp.einsum("gij,gjk->gik", A_mm, acc, precision=_HIGHEST)
            A_mn = sanitize(A_mn, config)
            blocks[m][n] = A_mn

    rows = []
    for m in range(N_MICRO):
        row_blocks = []
        for n in range(N_MICRO):
            if n > m:
                row_blocks.append(jnp.zeros((group, MB, MB), dtype=jnp.float32))
            else:
                row_blocks.append(blocks[m][n])
        rows.append(jnp.concatenate(row_blocks, axis=2))
    return jnp.concatenate(rows, axis=1)


def _kernel_b_body_batched(akk_ref, a_ref, *, bt: int, bc: int, config: KernelConfig, group: int):
    assert bt == 2 * bc, (
        f"batched Kernel B поддерживает только двухблочный top-level split "
        f"(bt == 2*bc); получено bt={bt}, bc={bc}."
    )
    Akk = akk_ref[0, 0].astype(jnp.float32)   # (group, bt, bt)
    T00 = Akk[:, 0:bc, 0:bc]
    T11 = Akk[:, bc:2 * bc, bc:2 * bc]
    T10 = Akk[:, bc:2 * bc, 0:bc]

    A00 = _block_solve_batched(T00, config, group)
    A11 = _block_solve_batched(T11, config, group)

    eps = config.wy_eps
    tmp = jnp.einsum("gij,gjk->gik", T10 * (1.0 - eps), A00, precision=_HIGHEST)
    tmp = sanitize(tmp, config)
    A10 = -jnp.einsum("gij,gjk->gik", A11, tmp, precision=_HIGHEST)
    A10 = sanitize(A10, config)

    a_ref[0, 0] = jnp.zeros((group, bt, bt), dtype=jnp.float32)
    a_ref[0, 0, :, 0:bc, 0:bc] = A00
    a_ref[0, 0, :, bc:2 * bc, 0:bc] = A10
    a_ref[0, 0, :, bc:2 * bc, bc:2 * bc] = A11


def wy_solve_pallas_batched(Akk, config: KernelConfig = DEFAULT_CONFIG, group: int | None = None,
                             vmem_limit_bytes: int = 128 * 1024 * 1024):
    """Forward-only, batched (grid=(bsz,H,n_groups)) замена wy_solve_pallas.
    Требует Akk формы (bsz,H,n_chunks,bt,bt), как и оригинал. `group` --
    сколько чанков обрабатывать за один вызов тела кернела; n_chunks %
    group == 0. Если group=None, используется config.b_batch_group, а
    если и он None -- весь n_chunks одной группой (может упереться в
    VMEM на длинных последовательностях, см. MB8_status_report.md --
    в этом случае явно передайте меньший group)."""
    assert config.bt == 2 * config.bc, (
        f"wy_solve_pallas_batched: bt должен быть == 2*bc, получено "
        f"bt={config.bt}, bc={config.bc}."
    )
    bsz, H, n_chunks = Akk.shape[:3]
    if group is None:
        group = config.b_batch_group if config.b_batch_group is not None else n_chunks
    if n_chunks % group != 0:
        raise ValueError(f"n_chunks={n_chunks} должен делиться на group={group}")
    n_groups = n_chunks // group

    grid = (bsz, H, n_groups)
    spec = pl.BlockSpec((1, 1, group, config.bt, config.bt), lambda i, h, gi: (i, h, gi, 0, 0))

    A = pl.pallas_call(
        lambda *refs: _kernel_b_body_batched(*refs, bt=config.bt, bc=config.bc, config=config, group=group),
        grid=grid,
        in_specs=[spec],
        out_specs=spec,
        out_shape=jax.ShapeDtypeStruct(Akk.shape, jnp.float32),
        compiler_params=pltpu.CompilerParams(vmem_limit_bytes=vmem_limit_bytes),
    )(Akk)
    return A


# ==========================================================================
# Fused Kernel A+B (прототип)
#
# Убирает ОДНОВРЕМЕННО grid-overhead Kernel A и HBM roundtrip Akk между
# A->B (MB8_status_report.md, "Kernel A+B: слияние в один pallas_call").
# Реализация: внутри одного pallas-тела на группу из `group` чанков
# сначала честно пересчитывается Aqk/Akk по существующей математике
# _kernel_a_body (без изменений -- в т.ч. корректно работает
# use_centering=True), затем сразу передаётся в _block_solve_batched
# без прохода через HBM. Aqk по-прежнему пишется в HBM (нужен
# downstream'у Kernel D), Akk -- НЕТ (существует только как
# промежуточная переменная в VMEM).
# ==========================================================================
def _kernel_ab_body_batched(q_ref, k_ref, b_ref, g_ref, aqk_ref, a_ref, *,
                             scale: float, bt: int, bc: int, n_sub: int,
                             use_centering: bool, config: KernelConfig, group: int):
    akk_list = []
    for gi in range(group):
        # Переиспользуем СУЩЕСТВУЮЩУЮ математику Kernel A как есть --
        # один "виртуальный" вызов на chunk gi внутри группы, пишем Aqk
        # напрямую в выходной ref (как и раньше, нужен Kernel D),
        # а Akk оставляем в VMEM-списке для батчевого solve.
        class _ScratchAkkRef:
            """Мини-обёртка, чтобы переиспользовать _kernel_a_body
            без изменений: перехватывает запись akk_ref[0,0,0,...]
            в локальный VMEM-массив вместо HBM ref."""
            def __init__(self, bt):
                self.buf = jnp.zeros((bt, bt), dtype=jnp.float32)

            def __setitem__(self, idx, val):
                # idx ожидается вида (0,0,0, slice_i, slice_j) или (0,0,0)
                if idx == (0, 0, 0):
                    self.buf = val
                else:
                    i0, i1 = idx[3].start, idx[3].stop
                    j0, j1 = idx[4].start, idx[4].stop
                    self.buf = self.buf.at[i0:i1, j0:j1].set(val)

        akk_scratch = _ScratchAkkRef(bt)

        class _Q1:
            def __getitem__(self, idx):
                return q_ref[0, 0, gi]
        class _K1:
            def __getitem__(self, idx):
                return k_ref[0, 0, gi]
        class _B1:
            def __getitem__(self, idx):
                return b_ref[0, 0, gi]
        class _G1:
            def __getitem__(self, idx):
                return g_ref[0, 0, gi]

        class _AqkOut:
            def __setitem__(self, idx, val):
                if idx == (0, 0, 0):
                    aqk_ref[0, 0, gi] = val
                else:
                    i0, i1 = idx[3].start, idx[3].stop
                    j0, j1 = idx[4].start, idx[4].stop
                    aqk_ref[0, 0, gi, i0:i1, j0:j1] = val

        _kernel_a_body(_Q1(), _K1(), _B1(), _G1(), _AqkOut(), akk_scratch,
                        scale=scale, bt=bt, bc=bc, n_sub=n_sub,
                        use_centering=use_centering, config=config)
        akk_list.append(akk_scratch.buf)

    Akk_group = jnp.stack(akk_list, axis=0)  # (group, bt, bt)
    A_group = _kernel_b_solve_only(Akk_group, config, group, bt, bc)
    a_ref[0, 0] = A_group


def _kernel_b_solve_only(Akk_group, config, group, bt, bc):
    T00 = Akk_group[:, 0:bc, 0:bc]
    T11 = Akk_group[:, bc:2 * bc, bc:2 * bc]
    T10 = Akk_group[:, bc:2 * bc, 0:bc]
    A00 = _block_solve_batched(T00, config, group)
    A11 = _block_solve_batched(T11, config, group)
    eps = config.wy_eps
    tmp = sanitize(jnp.einsum("gij,gjk->gik", T10 * (1.0 - eps), A00, precision=_HIGHEST), config)
    A10 = sanitize(-jnp.einsum("gij,gjk->gik", A11, tmp, precision=_HIGHEST), config)
    out = jnp.zeros((group, bt, bt), dtype=jnp.float32)
    out = out.at[:, 0:bc, 0:bc].set(A00)
    out = out.at[:, bc:2 * bc, 0:bc].set(A10)
    out = out.at[:, bc:2 * bc, bc:2 * bc].set(A11)
    return out


def build_and_solve_pallas_batched(q, k, b, g, scale, config: KernelConfig = DEFAULT_CONFIG,
                                    group: int | None = None,
                                    vmem_limit_bytes: int = 160 * 1024 * 1024):
    """ПРОТОТИП. Возвращает (Aqk, A) -- Akk НЕ материализуется в HBM.
    ВАЖНО (см. MB8_status_report.md, шаг 1 "Следующие шаги"): тело
    кернела на group чанков тяжелее, чем чистый Kernel B -- потолок по
    group будет ниже 64/128, найденных для одиночного B. Замерять
    отдельно, не экстраполировать из wy_solve_pallas_batched.
    Использует Python-цикл (unrolled) по `group` для честного
    переиспользования _kernel_a_body -- НЕ векторизованная версия
    Kernel A; выигрыш здесь только от одного pallas_call вместо двух
    (устраняет grid-overhead A + HBM roundtrip Akk), а не от
    батчинга самой scores-математики. Полная векторизация Kernel A по
    batch-оси -- отдельная, более рискованная работа (см. Гипотезу E/F
    в отчёте про B3/B4)."""
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
        lambda *refs: _kernel_ab_body_batched(
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
        interpret=True,  # см. NOTE ниже
    )(q_r, k_r, b_r, g_r)
    return aqk, A
