"""
gdn2_blr.precision -- точность матмулов и безопасные экспоненты.

ЗАЧЕМ bf16x3. Внешний production-прецедент (MaxText PR #4348, Qwen3-Next
GDN на v5e-256) сообщает: дефолтный f32-dot в Mosaic усекается до bf16, а
Precision.HIGH не поддержан, поэтому для треугольной инверсии они делают
РУЧНОЕ трёхпроходное разложение операндов. Весь Atomic_ops полагается на
`precision=jax.lax.Precision.HIGHEST` внутри pallas_call.

ЧТО ИЗМЕРЕНО У НАС (CPU, 256x256 против float64):
    default  2.874e-07
    highest  2.874e-07
    bf16x3   4.459e-06
На CPU нет MXU, поэтому default==highest, а bf16x3 -- это его собственная
точность (~15x хуже честного f32, но ~1000x лучше однопроходного bf16).
ЭТО КАЛИБРОВОЧНАЯ СТРОКА, а не вердикт. Вердикт даёт тот же замер на TPU:
если там `highest` уедет в ~1e-3, значит HIGHEST внутри Mosaic не
работает, и bf16x3 становится обязательным для Akk/A.

ЧТО НЕЛЬЗЯ ДЕЛАТЬ: выбирать режим "по умолчанию самый точный". На TPU
"самый точный по названию" и "самый точный по факту" -- разные вещи.
Режим выбирается гейтом G3/G9 по остатку (I + (1-eps)Akk) @ A == I.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

HIGHEST = jax.lax.Precision.HIGHEST
DEFAULT = jax.lax.Precision.DEFAULT


def split_bf16(x):
    """x ~= hi + lo, обе половины точно представимы в bf16."""
    hi = x.astype(jnp.bfloat16).astype(jnp.float32)
    lo = (x - hi).astype(jnp.bfloat16).astype(jnp.float32)
    return hi, lo


def dot_bf16x3(a, b):
    """a @ b тремя bf16-проходами. Каждый внутренний dot идёт с DEFAULT --
    ровно то, что MXU делает нативно, без надежды на флаг precision.
    Опущен член lo*lo: он даёт ~2^-16 относительного вклада, что ниже
    разрешения f32-аккумулятора."""
    a_hi, a_lo = split_bf16(a)
    b_hi, b_lo = split_bf16(b)
    return (jnp.dot(a_hi, b_hi, precision=DEFAULT)
            + jnp.dot(a_hi, b_lo, precision=DEFAULT)
            + jnp.dot(a_lo, b_hi, precision=DEFAULT))


def einsum_bf16x3(spec, a, b):
    a_hi, a_lo = split_bf16(a)
    b_hi, b_lo = split_bf16(b)
    return (jnp.einsum(spec, a_hi, b_hi, precision=DEFAULT)
            + jnp.einsum(spec, a_hi, b_lo, precision=DEFAULT)
            + jnp.einsum(spec, a_lo, b_hi, precision=DEFAULT))


def make_dot(mode: str):
    """2-D матмул по имени режима."""
    if mode == "bf16x3":
        return dot_bf16x3
    p = HIGHEST if mode == "highest" else DEFAULT
    return lambda a, b: jnp.dot(a, b, precision=p)


def make_einsum(mode: str):
    """einsum по имени режима (для батченых путей)."""
    if mode == "bf16x3":
        return einsum_bf16x3
    p = HIGHEST if mode == "highest" else DEFAULT
    return lambda spec, a, b: jnp.einsum(spec, a, b, precision=p)


# ===========================================================================
# Безопасные экспоненты.
#
# ЦЕНТРАЛЬНЫЙ ИНВАРИАНТ BLR: все показатели экспонент, которые уходят в
# матмул, <= 0. Тогда каждый множитель лежит в (0, 1], переполнение
# невозможно ПО ПОСТРОЕНИЮ, а underflow в 0 -- правильный ответ (полный
# распад), а не потеря информации.
#
# min(x, 0) при g <= 0 -- тождественный no-op. Он существует не как
# аппроксимация, а как страховка от нарушения архитектурного инварианта
# (например, багом в параметризации гейта): вместо inf/nan получится
# "без усиления".
#
# Маска возвращается отдельно, потому что backward должен домножать на
# 1{x < 0} -- производную клампа.
# ===========================================================================
def exp_nonpos(x):
    """exp(min(x, 0)), плюс маска 1{x < 0} для chain rule."""
    m = (x < 0.0).astype(jnp.float32)
    return jnp.exp(jnp.minimum(x, 0.0)), m


def exp_clipped(x, lo, hi):
    """exp(clip(x, lo, hi)) + маска 1{lo <= x <= hi}.
    Используется ТОЛЬКО на истинной разности gc_i - gc_j внутри
    диагонального блока -- клипуется СУММА, а не отдельные ноги, поэтому
    exp(-20) ~ 2e-9 и есть правильный ответ. Это математика production
    non-centered пути, измеренно точная (1.65e-06 против float64 при
    range(gc)=345)."""
    m = ((x >= lo) & (x <= hi)).astype(jnp.float32)
    return jnp.exp(jnp.clip(x, lo, hi)), m


def sanitize(x, clip: float):
    return jnp.nan_to_num(jnp.clip(x, -clip, clip), nan=0.0,
                          posinf=clip, neginf=-clip)
