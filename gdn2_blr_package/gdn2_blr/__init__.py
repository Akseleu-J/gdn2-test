"""
gdn2_blr -- GDN-2 ядра на BLR (Block-Local Rescaling).

Ядро идеи в одну строку: для блока запросов P с началом p0 берётся ОДНА
reference-точка r = gc[p0] и точное разложение затухания на ДВЕ ноги
(третьей, cross-ноги, не существует -- именно она была источником утечки
three-leg). При архитектурном инварианте g <= 0 обе ноги <= 0, значит
exp in (0,1], значит переполнение невозможно ПО ПОСТРОЕНИЮ. Диагональный
суб-блок считается неф акторизованно с ОДНИМ клипом на истинную разность --
математикой production non-centered пути, измеренно точной.

Что это даёт (измерено, numpy f32 vs float64, bt=256, 7 seed'ов):
  rel_err совпадает с production non-centered ДО ПОСЛЕДНЕЙ ЦИФРЫ на всех
  входах, включая extreme_strong_decay при range(gc)=698 (4.7e-06), тогда
  как Eq.19-global на том же входе даёт 8.8e+00. При этом 50-93.8% работы
  (в зависимости от score_bs) уходит с VPU на MXU.

Статус: фундамент. Gate 1 (CPU interpret=True) -- в test_gdn2_blr.py.
Gate 2 (TPU interpret=False) НЕ ПРОЙДЕН -- ничего отсюда не подключать в
production до него. См. GDN2_HANDBOOK.md §9.
"""
from .config import (BLRConfig, effective_group, fit_group,
                     fit_heads_per_cell, vmem_report,
                     KAGGLE_SAFE, KAGGLE_FAST, KAGGLE_SMALL)
from .precision import (dot_bf16x3, make_dot, make_einsum, exp_nonpos,
                        exp_clipped, sanitize)
from . import reference
from . import fwd
from . import bwd
from .pipeline import blr_forward, blr_trainable, forward_with_residuals, \
    backward_from_residuals

__all__ = [
    "BLRConfig", "effective_group", "fit_group", "fit_heads_per_cell",
    "vmem_report", "KAGGLE_SAFE", "KAGGLE_FAST", "KAGGLE_SMALL",
    "dot_bf16x3", "make_dot", "make_einsum", "exp_nonpos", "exp_clipped",
    "sanitize", "reference", "fwd", "bwd",
    "blr_forward", "blr_trainable",
    "forward_with_residuals", "backward_from_residuals",
]

__version__ = "0.1.0.dev0"
