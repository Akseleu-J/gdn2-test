"""
gdn2_blr.config -- единая конфигурация нового ядра.

Отличия от Atomic_ops.KernelConfig (и почему):

  * `interpret` живёт В КОНФИГЕ, а не в сигнатуре каждой функции. Дыра
    H0.2/H0.3: в старом коде `interpret` доведён только до части точек
    входа, поэтому Gate 1 (CPU) для backward и training-пути был
    физически невозможен. Конфиг уже hashable и уже проходит через
    custom_vjp как nondiff -- это правильное место.

  * `score_bs` -- гранулярность BLR-перешкалирования в Kernel A/B4.
    НЕ связан с `bc` (тот жёстко bt == 2*bc для WY-солвера).

  * `vmem_budget` по умолчанию 16 MB, а не 64-200 MB. Внешний
    production-прецедент (MaxText #4348) батчит головы/тайлы именно под
    16 MB scoped-VMEM; наш собственный RESOURCE_EXHAUSTED на B5 при
    лимите 64 MB -- то же самое наблюдение с другой стороны.

  * `group` берётся через `_effective_group`, а не через жёсткую
    проверку делимости. Дыра H0.4: KAGGLE_MEDIUM падал на L=1024.

  * `dot_mode` / `solve_dot_mode` выбирают реализацию матмула. На CPU
    измерено: default == highest == 2.87e-07, bf16x3 == 4.46e-06 (то
    есть на CPU bf16x3 ХУЖЕ, это его собственная точность). На TPU
    ожидается обратное, если Mosaic усекает f32-dot до bf16. Решение
    принимается гейтом G3/G9, а не декларацией.
"""
from __future__ import annotations

import dataclasses as dc
import math

_MB = 1024 * 1024

DOT_MODES = ("highest", "bf16x3", "default")
DIA_EXTRACT = ("matmul", "slice")


def effective_group(n_chunks: int, requested: int | None) -> int:
    """Наибольший делитель n_chunks, не превосходящий requested.

    ФИКС H0.4. Старый код бросал ValueError, из-за чего KAGGLE_MEDIUM
    (b_batch_group=16, bt=256) требовал L кратной 4096, а KAGGLE_SMALL --
    кратной 8192. L=1024 (n_chunks=4) падал.
    """
    if n_chunks <= 0:
        raise ValueError(f"n_chunks must be positive, got {n_chunks}")
    if requested is None or requested >= n_chunks:
        return n_chunks
    g = int(requested)
    while g > 1 and n_chunks % g != 0:
        g -= 1
    return max(1, g)


@dc.dataclass(frozen=True)
class BLRConfig:
    # --- геометрия чанка ---
    bt: int = 256                 # длина чанка (WY-солв инвертирует bt x bt)
    bc: int = 128                 # top-level split WY-солвера, bt == 2*bc
    mb: int = 16                  # база лесенки H9 (степень 2, делит bt)

    # --- BLR ---
    score_bs: int = 128           # гранулярность перешкалирования (см. handbook §3)
    dia_extract: str = "matmul"   # как вынимать диагональный блок в backward
    diff_clip: float = 20.0       # клип на ИСТИННУЮ разность внутри диагонали

    # --- численность ---
    clip: float = 1e4             # граница sanitize на выходах кернелов
    wy_eps: float = 1e-3          # демпфирование (I + (1-eps)Akk)
    dot_mode: str = "highest"     # матмулы score-кернелов
    solve_dot_mode: str = "highest"  # матмулы лесенки H9 (самое чувствительное место)

    # --- исполнение ---
    backend: str = "pallas"       # "pallas" | "xla" | "hybrid"
    """hybrid = Pallas там, где есть настоящая последовательная зависимость
    (A, B, D, B1, B4), XLA для chunk-parallel матмулов (C, B2, B3, B5).
    Это целевая раскладка (handbook §5). "xla" -- полный fallback, нужен
    и как цель Gate 1, и как ответ на H_EXEC."""

    group: int | None = None      # тайлов чанков на ячейку грида (Kernel B)
    heads_per_cell: int = 1       # голов на ячейку грида (Kernel D / B1)
    interpret: bool = False       # CPU-гейт
    vmem_budget: int = 16 * _MB
    vmem_strict: bool = False     # True -> падать, False -> предупреждать

    # --- архитектурный контракт ---
    assume_nonpositive_g: bool = True
    """g <= 0 покомпонентно (alpha = exp(g) in (0,1]) -- определение
    forget-gate'а. На этом держится вся безопасность BLR: обе ноги
    факторизации <= 0, значит exp in (0,1], значит переполнение
    невозможно. min(.,0) применяется ВСЕГДА: при g<=0 это тождественный
    no-op, при нарушении -- ограничивает вместо inf. Флаг существует
    только чтобы явно назвать инвариант, отключать его нельзя."""

    def __post_init__(self):
        if self.bt != 2 * self.bc:
            raise ValueError(f"bt={self.bt} must equal 2*bc, got bc={self.bc}")
        if self.bt % self.score_bs:
            raise ValueError(f"bt={self.bt} must be divisible by score_bs={self.score_bs}")
        if self.mb & (self.mb - 1) or self.bt % self.mb:
            raise ValueError(f"mb={self.mb} must be a power of 2 dividing bt={self.bt}")
        if self.bt & (self.bt - 1):
            raise ValueError(f"H9 ladder requires power-of-2 bt, got {self.bt}")
        nb = self.bt // self.mb
        if nb & (nb - 1):
            raise ValueError(f"bt/mb={nb} must be a power of 2 (doubling base->bt)")
        if not (0.0 <= self.wy_eps < 1.0):
            raise ValueError(f"wy_eps={self.wy_eps} must be in [0,1)")
        if self.dot_mode not in DOT_MODES or self.solve_dot_mode not in DOT_MODES:
            raise ValueError(f"dot_mode must be one of {DOT_MODES}")
        if self.dia_extract not in DIA_EXTRACT:
            raise ValueError(f"dia_extract must be one of {DIA_EXTRACT}")
        if self.backend not in ("pallas", "xla", "hybrid"):
            raise ValueError(f"backend must be pallas|xla|hybrid, got {self.backend}")
        if not self.assume_nonpositive_g:
            raise ValueError(
                "assume_nonpositive_g=False не поддерживается: вся схема "
                "предполагает alpha=exp(g) in (0,1]. См. handbook §2.4.")

    # ---- производные ----
    @property
    def n_sub(self) -> int:
        return self.bt // self.score_bs

    @property
    def n_micro(self) -> int:
        return self.bt // self.mb

    @property
    def ladder_levels(self) -> int:
        return int(math.log2(self.bt // self.mb))

    def with_(self, **kw) -> "BLRConfig":
        return dc.replace(self, **kw)


# ===========================================================================
# VMEM-бюджетирование. Считается АНАЛИТИЧЕСКИ, до компиляции, чтобы
# RESOURCE_EXHAUSTED не был сюрпризом в 3 часа ночи.
# Все числа -- байты на ОДНУ ячейку грида, float32.
# ===========================================================================
F32 = 4


def vmem_kernel_a(cfg: BLRConfig, D: int = 128) -> int:
    """q,k,b,gc (in) + Aqk,Akk (out) + рабочие: k~ (T,D), диагональ (bs,bs,D)."""
    T, bs = cfg.bt, cfg.score_bs
    inp = 4 * T * D
    out = 2 * T * T
    work = T * D + bs * bs * D + 2 * T * D
    return (inp + out + work) * F32


def vmem_kernel_b(cfg: BLRConfig, group: int) -> int:
    """Akk(in) + A(out) + лесенка держит ~2 уровня по group*T*T."""
    T = cfg.bt
    return (2 * group * T * T + 2 * group * T * T) * F32


def vmem_kernel_ab_fused(cfg: BLRConfig, group: int, D: int = 128) -> int:
    return vmem_kernel_a(cfg, D) * group + vmem_kernel_b(cfg, group)


def vmem_kernel_c(cfg: BLRConfig, D: int = 128) -> int:
    T = cfg.bt
    return (6 * T * D + T * T + 4 * T * D + 2 * T * D) * F32


def vmem_kernel_d(cfg: BLRConfig, n_chunks: int, hb: int, D: int = 128) -> int:
    """Aqk + w_pseudo,u,kg,qg (in) + o (out) + h (DxD) на голову."""
    T = cfg.bt
    per_head = n_chunks * (T * T + 4 * T * D + T * D) + D * D
    return hb * per_head * F32


def vmem_kernel_d_res(cfg: BLRConfig, n_chunks: int, hb: int, D: int = 128) -> int:
    """То же + residuals h_pre_all (N,D,D) и v_new_all (N,T,D)."""
    T = cfg.bt
    extra = n_chunks * (D * D + T * D)
    return vmem_kernel_d(cfg, n_chunks, hb, D) + hb * extra * F32


def vmem_kernel_b1(cfg: BLRConfig, n_chunks: int, hb: int, D: int = 128) -> int:
    """do, dv_partial, w_pseudo, qg, kg (in) + dh_next(N,D,D), dv(N,T,D) (out)."""
    T = cfg.bt
    per_head = n_chunks * (5 * T * D + D * D + T * D) + 2 * D * D
    return hb * per_head * F32


def vmem_kernel_b4(cfg: BLRConfig, D: int = 128) -> int:
    T, bs = cfg.bt, cfg.score_bs
    inp = 4 * T * D + 2 * T * T
    out = 4 * T * D
    work = 2 * T * D + bs * bs * D + bs * T
    return (inp + out + work) * F32


def fit_heads_per_cell(cfg: BLRConfig, n_chunks: int, H: int,
                       estimator, D: int = 128) -> int:
    """Наибольшее hb, делящее H, при котором estimator(...) <= vmem_budget."""
    hb = min(H, 8)
    while hb > 1:
        if H % hb == 0 and estimator(cfg, n_chunks, hb, D) <= cfg.vmem_budget:
            return hb
        hb -= 1
    return 1


def fit_group(cfg: BLRConfig, n_chunks: int) -> int:
    """Наибольший делитель n_chunks (<= cfg.group), влезающий в бюджет."""
    g = effective_group(n_chunks, cfg.group)
    while g > 1 and vmem_kernel_b(cfg, g) > cfg.vmem_budget:
        g -= 1
        while g > 1 and n_chunks % g:
            g -= 1
    return max(1, g)


def vmem_report(cfg: BLRConfig, bsz: int, H: int, n_chunks: int, D: int = 128) -> dict:
    """Полный отчёт по бюджету -- печатать ПЕРЕД первым прогоном на TPU."""
    g = fit_group(cfg, n_chunks)
    hb_d = fit_heads_per_cell(cfg, n_chunks, H, vmem_kernel_d_res, D)
    hb_b1 = fit_heads_per_cell(cfg, n_chunks, H, vmem_kernel_b1, D)
    rep = {
        "budget_MB": cfg.vmem_budget / _MB,
        "A": vmem_kernel_a(cfg, D) / _MB,
        "B(group=%d)" % g: vmem_kernel_b(cfg, g) / _MB,
        "AB_fused(group=%d)" % g: vmem_kernel_ab_fused(cfg, g, D) / _MB,
        "C": vmem_kernel_c(cfg, D) / _MB,
        "D(hb=%d)" % hb_d: vmem_kernel_d_res(cfg, n_chunks, hb_d, D) / _MB,
        "B1(hb=%d)" % hb_b1: vmem_kernel_b1(cfg, n_chunks, hb_b1, D) / _MB,
        "B4": vmem_kernel_b4(cfg, D) / _MB,
        "chosen_group": g,
        "chosen_hb_D": hb_d,
        "chosen_hb_B1": hb_b1,
        "grid_cells_A": bsz * H * n_chunks,
        "grid_cells_B": bsz * H * (n_chunks // g),
        "grid_cells_D": bsz * (H // hb_d),
    }
    return rep


# ---- пресеты ----
KAGGLE_SAFE = BLRConfig(bt=256, bc=128, mb=16, score_bs=128,
                        dot_mode="highest", solve_dot_mode="highest",
                        group=8, heads_per_cell=1)
"""Стартовая точка: score_bs=128 -> все срезы по lane-оси кратны 128,
нулевой риск Mosaic. 50% работы Kernel A на MXU (вместо 0% сейчас)."""

KAGGLE_FAST = BLRConfig(bt=256, bc=128, mb=16, score_bs=32,
                        dia_extract="matmul",
                        dot_mode="highest", solve_dot_mode="bf16x3",
                        group=8, heads_per_cell=4)
"""Цель фазы 1.3: 87.5% на MXU. Требует Gate 2 (Mosaic lowering при
score_bs<128) ДО использования в проде."""

KAGGLE_SMALL = BLRConfig(bt=128, bc=64, mb=16, score_bs=64,
                         group=16, heads_per_cell=1)
