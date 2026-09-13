#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# apply_batch2_patches_standalone.sh
#
# Самодостаточная версия: весь код 9 файлов батча зашит прямо в скрипт
# (heredoc'ами) -- никакой incoming_patches/ директории не нужно.
#
# ТОЛЬКО ПАТЧИТ И СОЗДАЁТ ФАЙЛЫ. Ничего не прогоняет. Список тестов для
# ручного прогона -- в конце вывода скрипта и в ответе рядом с ним.
#
# Что делает:
#   ШАГ 1  -- unsafe_allow_centering=True в 3 внешних файлах (если найдены)
#   ШАГ 2  -- _max_validated_group(bt) формула в gdn2_fwd_batched.py
#             (Current_state.md "срочное" #1 -- ни в одном из 9 файлов
#             самой функции нет, дописана по формуле из тестов)
#   ШАГ 3  -- Variant A (fixed b_batch_group per preset) в configs.py
#             + tests/test_preset_group_within_validated_bound.py
#             ИЛИ Variant B (autosearch) -- Atomic_ops/group_autosearch.py
#             (выбор через GROUP_STRATEGY=fixed|autosearch, default fixed)
#   ШАГ 4  -- use_fused_ab flag в gdn2_pallas_forward (inference-only)
#   ШАГ 5  -- tests/test_use_fused_ab.py
#   ШАГ 6  -- Atomic_ops/gdn2_bwd_batched_b2.py (B2 batched backward, Hypothesis G)
#   ШАГ 7  -- tests/test_b2_batched_vs_nonbatched.py (import-путь уже поправлен)
#   ШАГ 8  -- tests/test_regressions_bundle.py (group guard + P0.1/P0.2/P0.3 + N5)
#
# Ожидания: ./Atomic_ops/ уже существует рядом со скриптом.
# =============================================================================

ROOT_DIR="$(pwd)"
PKG_DIR="Atomic_ops"
GROUP_STRATEGY="${GROUP_STRATEGY:-fixed}"   # fixed | autosearch

TS="$(date +%Y%m%d_%H%M%S)"
BACKUP_DIR="_backup_batch2_${TS}"

if [ ! -d "$PKG_DIR" ]; then
    echo "!!! $PKG_DIR не найден в $(pwd). Запускай из директории, где рядом лежит Atomic_ops."
    exit 1
fi

mkdir -p "$BACKUP_DIR" tests

backup_file() {
    local f="$1"
    if [ -f "$f" ]; then
        mkdir -p "$BACKUP_DIR/$(dirname "$f")"
        cp "$f" "$BACKUP_DIR/$f"
    fi
}

echo ">>> Бэкапы изменяемых файлов: $BACKUP_DIR/"

# conftest.py, чтобы pytest видел Atomic_ops без pip install -e
if [ ! -f "conftest.py" ]; then
    cat > conftest.py <<'PYEOF'
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
PYEOF
    echo ">>> Добавлен conftest.py"
fi

# -----------------------------------------------------------------------------
# ШАГ 1 -- N1/N2 fix: unsafe_allow_centering=True в 3 внешних файлах.
#          Эти файлы -- НЕ часть текущего батча, должны уже существовать
#          в проекте с прошлых раундов (gate1/sweep скрипты). Если их нет,
#          шаг пропускается с явным предупреждением, не молча.
# -----------------------------------------------------------------------------
echo ""
echo "=================================================================="
echo ">>> ШАГ 1: N1/N2 fix -- unsafe_allow_centering=True"
echo "=================================================================="

patch_add_ack() {
    local file="$1" old="$2" new="$3"
    if [ ! -f "$file" ]; then
        echo "    !!! ПРОПУСК: $file не найден (должен уже существовать в проекте)."
        return
    fi
    backup_file "$file"
    python3 - "$file" "$old" "$new" <<'PYEOF'
import sys, pathlib
path, old, new = pathlib.Path(sys.argv[1]), sys.argv[2], sys.argv[3]
src = path.read_text()
if old not in src:
    print(f"    !!! WARNING: искомая строка не найдена в {path} -- правь вручную.")
    sys.exit(0)
if src.count(old) != 1:
    print(f"    !!! WARNING: строка встречается не 1 раз в {path} -- правь вручную.")
    sys.exit(0)
path.write_text(src.replace(old, new, 1))
print(f"    OK: {path}")
PYEOF
}

patch_add_ack "test_gate1_wy_solve_batched.py" \
  "CFG_BASE = dict(bt=128, bc=64, mb=16, clip=1e4, wy_eps=1e-3, use_centering=True)" \
  "CFG_BASE = dict(bt=128, bc=64, mb=16, clip=1e4, wy_eps=1e-3,
                use_centering=True, unsafe_allow_centering=True)"

patch_add_ack "gate1_wy_solve_batched.py" \
  "CFG_BASE = dict(bt=128, bc=64, mb=16, clip=1e4, wy_eps=1e-3, use_centering=True)" \
  "CFG_BASE = dict(bt=128, bc=64, mb=16, clip=1e4, wy_eps=1e-3,
                use_centering=True, unsafe_allow_centering=True)"

patch_add_ack "sweep_centering_leak.py" \
  "cfg_cent = KernelConfig(**CFG_BASE, use_centering=True)" \
  "cfg_cent = KernelConfig(**CFG_BASE, use_centering=True,
                         unsafe_allow_centering=True)"

# -----------------------------------------------------------------------------
# ШАГ 2 -- _max_validated_group(bt) формула (Current_state.md "срочное" #1).
# -----------------------------------------------------------------------------
echo ""
echo "=================================================================="
echo ">>> ШАГ 2: _max_validated_group(bt) формула в gdn2_fwd_batched.py"
echo "=================================================================="

FWD_BATCHED="$PKG_DIR/gdn2_fwd_batched.py"
backup_file "$FWD_BATCHED"

python3 - "$FWD_BATCHED" <<'PYEOF'
import sys, pathlib
path = pathlib.Path(sys.argv[1])
src = path.read_text()

OLD_CONST = "_MAX_VALIDATED_GROUP = 16\n"
NEW_CONST = '''def _max_validated_group(bt: int) -> int:
    """Largest `group` for which VMEM safety has actually been measured,
    as a FUNCTION of `bt` (Current_state.md "срочное" #1 -- replaces the
    single hardcoded constant that goes stale whenever bt changes). The
    one real anchor point is bt=256 -> group=16 (MB8_status_report.md,
    train_shape). The rest follows the same group*bt*bt ~ const scaling
    implied by the report's own VMEM-OOM curve (n_chunks=64, group=128
    OOMs at bt=256). This is a WARNING threshold, not a hard assert.
    """
    return max(1, (64 * 128 * 128) // (bt * bt))


# Kept for anyone importing the old name directly; now derived from the
# formula above at its one TPU-measured anchor point (bt=256).
_MAX_VALIDATED_GROUP = _max_validated_group(256)
'''

if OLD_CONST not in src:
    print("    !!! WARNING: '_MAX_VALIDATED_GROUP = 16' не найдено -- правь вручную.")
    sys.exit(0)
src = src.replace(OLD_CONST, NEW_CONST, 1)

OLD_WARN = '''    if group > _MAX_VALIDATED_GROUP:
        warnings.warn(
            f"wy_solve_pallas_batched: group={group} \u043f\u0440\u0435\u0432\u044b\u0448\u0430\u0435\u0442 \u0434\u0438\u0430\u043f\u0430\u0437\u043e\u043d, "
            f"\u0438\u0437\u043c\u0435\u0440\u0435\u043d\u043d\u044b\u0439 \u0432 MB8_status_report.md (\u0434\u043e group={_MAX_VALIDATED_GROUP} "
            f"\u043d\u0430 train_shape). \u041f\u0440\u0438 n_chunks=64 \u0433\u0440\u0443\u043f\u043f\u0430=128 \u0443\u043f\u0430\u043b\u0430 \u0432 VMEM OOM "
            f"\u0432 \u0438\u0441\u0445\u043e\u0434\u043d\u043e\u043c \u044d\u043a\u0441\u043f\u0435\u0440\u0438\u043c\u0435\u043d\u0442\u0435 -- \u0437\u0430\u0434\u0430\u0439\u0442\u0435 b_batch_group \u044f\u0432\u043d\u043e "
            f"\u0438 \u043f\u043e\u0434\u0431\u0435\u0440\u0438\u0442\u0435 \u043f\u043e\u0434 \u0432\u0430\u0448 vmem_limit_bytes, \u043d\u0435 \u043f\u043e\u043b\u0430\u0433\u0430\u0439\u0442\u0435\u0441\u044c \u043d\u0430 \u0434\u0435\u0444\u043e\u043b\u0442 "
            f"group=n_chunks \u0434\u043b\u044f \u0434\u043b\u0438\u043d\u043d\u044b\u0445 \u043f\u043e\u0441\u043b\u0435\u0434\u043e\u0432\u0430\u0442\u0435\u043b\u044c\u043d\u043e\u0441\u0442\u0435\u0439.",
            RuntimeWarning,
        )
'''
NEW_WARN = '''    max_validated = _max_validated_group(config.bt)
    if group > max_validated:
        warnings.warn(
            f"wy_solve_pallas_batched: group={group} превышает диапазон, "
            f"измеренный/выведенный по формуле для bt={config.bt} "
            f"(max_validated={max_validated}; см. MB8_status_report.md и "
            f"_max_validated_group()). При n_chunks=64 группа=128 упала в "
            f"VMEM OOM в исходном эксперименте (bt=256) -- задайте "
            f"b_batch_group явно и подберите под ваш vmem_limit_bytes, не "
            f"полагайтесь на дефолт group=n_chunks для длинных "
            f"последовательностей.",
            RuntimeWarning,
        )
'''

if OLD_WARN in src:
    src = src.replace(OLD_WARN, NEW_WARN, 1)
    print("    OK: warning-блок переведён на формулу.")
else:
    print("    !!! WARNING: warning-блок не найден дословно -- проверь вручную.")

path.write_text(src)
print(f"    OK: {path}")
PYEOF

# -----------------------------------------------------------------------------
# ШАГ 3 -- group strategy: Variant A (fixed) или Variant B (autosearch)
# -----------------------------------------------------------------------------
echo ""
echo "=================================================================="
echo ">>> ШАГ 3: group strategy = $GROUP_STRATEGY"
echo "=================================================================="

CONFIGS_PY="$PKG_DIR/configs.py"

if [ "$GROUP_STRATEGY" = "fixed" ]; then
    backup_file "$CONFIGS_PY"
    python3 - "$CONFIGS_PY" <<'PYEOF'
import sys, pathlib
path = pathlib.Path(sys.argv[1])
src = path.read_text()

OLD = (
    "KAGGLE_SMALL = KernelConfig(bt=128, bc=64, mb=16, clip=1e4, wy_eps=1e-3, use_centering=False)\n"
    "KAGGLE_MEDIUM = KernelConfig(bt=256, bc=128, mb=16, clip=1e4, wy_eps=1e-3, use_centering=False)\n"
    "KAGGLE_LARGE = KernelConfig(bt=256, bc=128, mb=16, clip=5e3, wy_eps=1e-3, use_centering=False)\n"
)
NEW = (
    "# b_batch_group values below come from the formula in\n"
    "# gdn2_fwd_batched._max_validated_group(bt), which is anchored to the\n"
    "# ONE real TPU measurement we have (bt=256 -> group=16, MB8_status_report.md,\n"
    "# train_shape n_chunks=16). They are NOT independently re-measured for\n"
    "# bt=128/512 -- if you retarget hardware or vmem_limit_bytes, re-run\n"
    "# Gate 1 (A) with the new group and update these, don't trust the formula\n"
    "# blindly for a preset it wasn't anchored on. See\n"
    "# tests/test_preset_group_within_validated_bound.py for the tripwire.\n"
    "KAGGLE_SMALL = KernelConfig(bt=128, bc=64, mb=16, clip=1e4, wy_eps=1e-3,\n"
    "                             use_centering=False, b_batch_group=64)\n"
    "KAGGLE_MEDIUM = KernelConfig(bt=256, bc=128, mb=16, clip=1e4, wy_eps=1e-3,\n"
    "                              use_centering=False, b_batch_group=16)\n"
    "KAGGLE_LARGE = KernelConfig(bt=256, bc=128, mb=16, clip=5e3, wy_eps=1e-3,\n"
    "                             use_centering=False, b_batch_group=16)\n"
)

if OLD not in src:
    print("    !!! WARNING: KAGGLE_SMALL/MEDIUM/LARGE блок не найден дословно -- правь configs.py вручную.")
    sys.exit(0)
if src.count(OLD) != 1:
    print("    !!! WARNING: блок встречается не 1 раз -- правь вручную.")
    sys.exit(0)

path.write_text(src.replace(OLD, NEW, 1))
print(f"    OK: {path} -- b_batch_group добавлен в KAGGLE_SMALL/MEDIUM/LARGE.")
PYEOF

    cat > "tests/test_preset_group_within_validated_bound.py" <<'TESTEOF'
"""
If you pick Variant A (fixed b_batch_group per preset), this test is
the tripwire for preset/formula drift: fails loudly instead of OOMing
quietly on a longer sequence length than train_shape was tuned for.
"""
import pytest
from Atomic_ops.configs import KAGGLE_SMALL, KAGGLE_MEDIUM, KAGGLE_LARGE
from Atomic_ops.gdn2_fwd_batched import _max_validated_group


@pytest.mark.parametrize("preset,name", [
    (KAGGLE_SMALL, "KAGGLE_SMALL"),
    (KAGGLE_MEDIUM, "KAGGLE_MEDIUM"),
    (KAGGLE_LARGE, "KAGGLE_LARGE"),
])
def test_preset_group_does_not_exceed_validated_bound(preset, name):
    assert preset.b_batch_group is not None, (
        f"{name}: b_batch_group is None -- falls back to group=n_chunks, "
        f"which OOM'd at n_chunks=64/group=128 in MB8_status_report.md. "
        f"Variant A requires every preset to set this explicitly."
    )
    bound = _max_validated_group(preset.bt)
    assert preset.b_batch_group <= bound, (
        f"{name}: b_batch_group={preset.b_batch_group} exceeds "
        f"_max_validated_group(bt={preset.bt})={bound}. Either the "
        f"preset drifted from the formula, or the formula's anchor "
        f"point needs a fresh TPU measurement at this bt."
    )
TESTEOF
    echo "    OK: tests/test_preset_group_within_validated_bound.py"

elif [ "$GROUP_STRATEGY" = "autosearch" ]; then
    cat > "$PKG_DIR/group_autosearch.py" <<'AUTOEOF'
"""
VARIANT B -- auto-search the largest safe `group` under a VMEM budget,
instead of hardcoding numbers per preset.

Roadmap ref: §1.6 / Current_state "срочное" #2.
Choose this variant if: you expect bt/bc/dtype/sequence-length/hardware
to change across sessions (Kaggle TPU allocation varies run to run), and
you'd rather pay a one-time calibration cost than maintain hand-tuned
constants that silently go stale (exactly what happened to the old
_MAX_VALIDATED_GROUP=16).

Mechanism: try candidate groups from largest to smallest (powers of two,
capped by n_chunks divisibility), actually call wy_solve_pallas_batched
with interpret=False and REAL data of the given shape, catch VMEM OOM
specifically (not swallow arbitrary exceptions), and return the first
one that doesn't OOM. Caches the result per (bt, bc, dtype, n_chunks,
vmem_limit_bytes) so this only pays the probe cost once per shape family,
not once per training step.

This does NOT replace Gate 1/Gate 2 -- it only searches for a group that
doesn't blow VMEM. Correctness (bit-identical vs non-batched) must still
be checked separately, same as always.
"""
from __future__ import annotations

import functools
import warnings

import jax
import jax.numpy as jnp

from .configs import KernelConfig


class GroupSearchFailed(RuntimeError):
    pass


def _is_oom_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "resource_exhausted" in msg or "out of memory" in msg or "oom" in msg


@functools.lru_cache(maxsize=32)
def _cached_search(bt: int, bc: int, mb: int, wy_eps: float, dtype_name: str,
                    n_chunks: int, vmem_limit_bytes: int) -> int:
    from .gdn2_fwd_batched import wy_solve_pallas_batched

    dtype = jnp.dtype(dtype_name)
    config = KernelConfig(bt=bt, bc=bc, mb=mb, wy_eps=wy_eps)

    candidates = []
    g = 1
    while g <= n_chunks:
        candidates.append(g)
        g *= 2
    candidates = [g for g in candidates if n_chunks % g == 0]
    candidates.sort(reverse=True)

    key = jax.random.PRNGKey(0)
    probe_shape = (1, 1, n_chunks, bt, bt)
    Akk_probe = jax.random.normal(key, probe_shape, dtype=jnp.float32) * 0.01

    last_err = None
    for group in candidates:
        try:
            fn = jax.jit(
                lambda akk: wy_solve_pallas_batched(akk, config, group=group)
            )
            out = jax.block_until_ready(fn(Akk_probe))
            if not bool(jnp.all(jnp.isfinite(out))):
                continue
            return group
        except Exception as e:  # noqa: BLE001 -- must inspect message, JAX OOM type varies by backend
            if _is_oom_error(e):
                last_err = e
                continue
            raise

    raise GroupSearchFailed(
        f"No group in {candidates} avoided VMEM OOM for bt={bt}, bc={bc}, "
        f"n_chunks={n_chunks}, vmem_limit_bytes={vmem_limit_bytes}. "
        f"Last error: {last_err}"
    )


def resolve_group_autosearch(config: KernelConfig, n_chunks: int,
                              vmem_limit_bytes: int = 96 * 1024 * 1024,
                              dtype=jnp.float32) -> int:
    """Drop-in replacement for the `group = config.b_batch_group or
    n_chunks` line in wy_solve_pallas_batched's caller. Explicit
    config.b_batch_group ALWAYS wins (manual override still respected);
    auto-search only kicks in when it's None.
    """
    if config.b_batch_group is not None:
        return config.b_batch_group

    try:
        group = _cached_search(
            config.bt, config.bc, config.mb, config.wy_eps,
            jnp.dtype(dtype).name, n_chunks, vmem_limit_bytes,
        )
    except GroupSearchFailed as e:
        warnings.warn(
            f"{e}. Falling back to group=1 (safest, slowest). Set "
            f"config.b_batch_group explicitly to skip this search.",
            RuntimeWarning,
        )
        group = 1
    return group
AUTOEOF
    echo "    OK: $PKG_DIR/group_autosearch.py"
    echo "    NOTE: resolve_group_autosearch() нужно вызвать вручную там, где"
    echo "          сейчас читается config.b_batch_group -- opt-in, не автоподключается."
else
    echo "    !!! Неизвестная GROUP_STRATEGY='$GROUP_STRATEGY' (ожидается fixed|autosearch)."
    exit 1
fi

# -----------------------------------------------------------------------------
# ШАГ 4 -- use_fused_ab flag в gdn2_pallas_forward (inference-only)
# -----------------------------------------------------------------------------
echo ""
echo "=================================================================="
echo ">>> ШАГ 4: use_fused_ab flag в gdn2_pallas_forward"
echo "=================================================================="

FWD_MAIN="$PKG_DIR/gdn2_fwd.py"
backup_file "$FWD_MAIN"

python3 - "$FWD_MAIN" <<'PYEOF'
import sys, pathlib
path = pathlib.Path(sys.argv[1])
src = path.read_text()

OLD = '''def gdn2_pallas_forward(q, k, v, w, b, g, scale, h0=None,
                        config: KernelConfig = DEFAULT_CONFIG, debug_tag: str = "",
                        interpret: bool = False):
    bsz, L, H, D, n_chunks = validate_inputs(q, k, v, w, b, g, scale, h0, config)

    # Local import to avoid a top-level circular import (gdn2_fwd_batched
    # imports from gdn2_fwd).
    from .gdn2_fwd_batched import wy_solve_pallas_batched

    Aqk, Akk = build_chunk_scores_pallas(q, k, b, g, scale, config, interpret=interpret)
    Aqk = _stage_diag(f"{debug_tag}:kernel_A_Aqk", Aqk)
    Akk = _stage_diag(f"{debug_tag}:kernel_A_Akk", Akk)

    A = wy_solve_pallas_batched(Akk, config, interpret=interpret)
    A = _stage_diag(f"{debug_tag}:kernel_B_wy_inverse_A", A)

    w_pseudo, u, kg, qg, gc_last = recompute_wy_pallas(
        q, k, v, w, b, g, A, config, interpret=interpret,
    )'''

NEW = '''def gdn2_pallas_forward(q, k, v, w, b, g, scale, h0=None,
                        config: KernelConfig = DEFAULT_CONFIG, debug_tag: str = "",
                        interpret: bool = False, use_fused_ab: bool = False):
    """
    use_fused_ab: if True, use build_and_solve_pallas_batched_fixed
    (Atomic_ops.gdn2_fwd_batched_fixed) instead of the separate
    build_chunk_scores_pallas + wy_solve_pallas_batched calls. This
    fuses Kernel A and Kernel B into one pallas_call (removes one HBM
    roundtrip of Akk between them). INFERENCE-ONLY: this function has
    no custom_vjp, so flipping this flag cannot silently change
    training-path gradients. Requires config.b_batch_group to be set
    (fused kernel takes an explicit group, same constraint as the
    non-fused batched path). Gate 1 (A2) in gate1_wy_solve_batched.py
    must PASS on interpret=False for your target shape before setting
    this True in anything user-facing. gdn2_pallas_forward_with_residuals
    / gdn2_pallas_forward_trainable are intentionally NOT touched --
    training stays on the already-integrated non-fused batched path.
    """
    bsz, L, H, D, n_chunks = validate_inputs(q, k, v, w, b, g, scale, h0, config)

    # Local imports to avoid a top-level circular import (both batched
    # modules import from gdn2_fwd).
    from .gdn2_fwd_batched import wy_solve_pallas_batched

    if use_fused_ab:
        from .gdn2_fwd_batched_fixed import build_and_solve_pallas_batched_fixed
        if config.b_batch_group is None:
            raise ValueError(
                "use_fused_ab=True requires config.b_batch_group to be "
                "set explicitly (same constraint as wy_solve_pallas_batched)."
            )
        Aqk, A = build_and_solve_pallas_batched_fixed(
            q, k, b, g, scale, config, group=config.b_batch_group,
            interpret=interpret,
        )
        Aqk = _stage_diag(f"{debug_tag}:kernel_AB_fused_Aqk", Aqk)
        A = _stage_diag(f"{debug_tag}:kernel_AB_fused_A", A)
    else:
        Aqk, Akk = build_chunk_scores_pallas(q, k, b, g, scale, config, interpret=interpret)
        Aqk = _stage_diag(f"{debug_tag}:kernel_A_Aqk", Aqk)
        Akk = _stage_diag(f"{debug_tag}:kernel_A_Akk", Akk)

        A = wy_solve_pallas_batched(Akk, config, interpret=interpret)
        A = _stage_diag(f"{debug_tag}:kernel_B_wy_inverse_A", A)

    w_pseudo, u, kg, qg, gc_last = recompute_wy_pallas(
        q, k, v, w, b, g, A, config, interpret=interpret,
    )'''

if OLD not in src:
    print("    !!! WARNING: искомый блок gdn2_pallas_forward не найден дословно -- правь вручную.")
    sys.exit(0)
if src.count(OLD) != 1:
    print("    !!! WARNING: блок встречается не 1 раз -- правь вручную.")
    sys.exit(0)

path.write_text(src.replace(OLD, NEW, 1))
print(f"    OK: {path} -- use_fused_ab добавлен, gdn2_pallas_forward_with_residuals не тронут.")
PYEOF

# -----------------------------------------------------------------------------
# ШАГ 5 -- tests/test_use_fused_ab.py
# -----------------------------------------------------------------------------
echo ""
echo "=================================================================="
echo ">>> ШАГ 5: tests/test_use_fused_ab.py"
echo "=================================================================="

cat > "tests/test_use_fused_ab.py" <<'FUSEDTESTEOF'
"""
CPU-only (interpret=True) smoke test for the use_fused_ab flag patch.
Does NOT replace Gate 1 (A2) on real TPU -- interpret=True only proves
"math unchanged by the flag", same caveat gate1_wy_solve_batched.py
already states for its own (A2) section.
"""
from __future__ import annotations

import dataclasses as dc

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from Atomic_ops.configs import KernelConfig
from Atomic_ops.gdn2_fwd import gdn2_pallas_forward


def _make_inputs(seed, bsz=1, L=256, H=1, D=128):
    rng = np.random.default_rng(seed)
    shape = (bsz, L, H, D)
    q = rng.normal(size=shape).astype(np.float32) * 0.1
    k = rng.normal(size=shape).astype(np.float32) * 0.1
    v = rng.normal(size=shape).astype(np.float32) * 0.1
    w = np.ones(shape, dtype=np.float32)
    b = (0.5 * rng.uniform(0.5, 1.0, size=shape)).astype(np.float32)
    g = -np.abs(rng.normal(size=shape)).astype(np.float32) * 0.05
    return tuple(jnp.asarray(x) for x in (q, k, v, w, b, g))


def _rel_err(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.linalg.norm((a - b).ravel()) / max(np.linalg.norm(b.ravel()), 1e-12))


@pytest.fixture
def base_config():
    return KernelConfig(bt=128, bc=64, mb=16, wy_eps=1e-3, b_batch_group=2)


def test_default_flag_preserves_old_behavior(base_config):
    q, k, v, w, b, g = _make_inputs(seed=0)
    o1, h1 = gdn2_pallas_forward(q, k, v, w, b, g, scale=1.0, config=base_config, interpret=True)
    o2, h2 = gdn2_pallas_forward(q, k, v, w, b, g, scale=1.0, config=base_config,
                                  interpret=True, use_fused_ab=False)
    assert jnp.allclose(o1, o2, atol=0.0, rtol=0.0)
    assert jnp.allclose(h1, h2, atol=0.0, rtol=0.0)


def test_fused_matches_nonfused_on_cpu(base_config):
    q, k, v, w, b, g = _make_inputs(seed=1)
    o_nonfused, h_nonfused = gdn2_pallas_forward(
        q, k, v, w, b, g, scale=1.0, config=base_config,
        interpret=True, use_fused_ab=False,
    )
    o_fused, h_fused = gdn2_pallas_forward(
        q, k, v, w, b, g, scale=1.0, config=base_config,
        interpret=True, use_fused_ab=True,
    )
    err_o = _rel_err(o_fused, o_nonfused)
    err_h = _rel_err(h_fused, h_nonfused)
    assert err_o < 1e-5, f"fused vs non-fused output diverges: rel_err={err_o:.2e}"
    assert err_h < 1e-5, f"fused vs non-fused h_final diverges: rel_err={err_h:.2e}"
    assert jnp.all(jnp.isfinite(o_fused))
    assert jnp.all(jnp.isfinite(h_fused))


def test_fused_requires_explicit_group():
    config_no_group = KernelConfig(bt=128, bc=64, mb=16, wy_eps=1e-3)  # b_batch_group=None
    q, k, v, w, b, g = _make_inputs(seed=2)
    with pytest.raises(ValueError, match="b_batch_group"):
        gdn2_pallas_forward(q, k, v, w, b, g, scale=1.0, config=config_no_group,
                             interpret=True, use_fused_ab=True)


@pytest.mark.parametrize("seed,g_scale,k_scale,b_scale", [
    (10, 3.0, 1.0, 0.5),    # extreme_strong_decay
    (11, 0.3, 8.0, 4.0),    # extreme_large_kb
])
def test_fused_matches_nonfused_extreme_seeds(base_config, seed, g_scale, k_scale, b_scale):
    rng = np.random.default_rng(seed)
    shape = (1, 256, 1, 128)
    q = rng.normal(size=shape).astype(np.float32) * 0.1
    k = rng.normal(size=shape).astype(np.float32) * 0.1 * k_scale
    v = rng.normal(size=shape).astype(np.float32) * 0.1
    w = np.ones(shape, dtype=np.float32)
    b = (b_scale * rng.uniform(0.5, 1.0, size=shape)).astype(np.float32)
    g = -np.abs(rng.normal(size=shape)).astype(np.float32) * g_scale
    q, k, v, w, b, g = (jnp.asarray(x) for x in (q, k, v, w, b, g))

    o_nonfused, h_nonfused = gdn2_pallas_forward(
        q, k, v, w, b, g, scale=1.0, config=base_config, interpret=True, use_fused_ab=False,
    )
    o_fused, h_fused = gdn2_pallas_forward(
        q, k, v, w, b, g, scale=1.0, config=base_config, interpret=True, use_fused_ab=True,
    )
    assert jnp.all(jnp.isfinite(o_fused)) and jnp.all(jnp.isfinite(h_fused)), (
        f"non-finite fused output on extreme seed={seed} "
        f"(g_scale={g_scale}, k_scale={k_scale}, b_scale={b_scale})"
    )
    assert _rel_err(o_fused, o_nonfused) < 1e-4
FUSEDTESTEOF
echo "    OK: tests/test_use_fused_ab.py"

# -----------------------------------------------------------------------------
# ШАГ 6 -- Atomic_ops/gdn2_bwd_batched_b2.py (Hypothesis G, MB8)
# -----------------------------------------------------------------------------
echo ""
echo "=================================================================="
echo ">>> ШАГ 6: Atomic_ops/gdn2_bwd_batched_b2.py"
echo "=================================================================="

cat > "$PKG_DIR/gdn2_bwd_batched_b2.py" <<'B2MODULEEOF'
"""
Hypothesis G (MB8_status_report.md "Следующие шаги" #3): apply the same
batch-axis-instead-of-grid pattern already validated bit-identical for
Kernel B to dav_backward_pallas (B2). B2 is the cheapest backward target:
two direct matmuls, no python loop over (si,sj) pairs, no scatter --
none of the risk class that broke Kernel B originally (.at[].set() not
lowering) or that B3/B4 carry (scatter writes inside a loop).

grid=(bsz,H,n_chunks) -> grid=(bsz,H), n_chunks batched inside the kernel
body via an einsum with a leading batch axis instead of a Pallas grid
dimension -- exactly the transformation MB8 already did for Kernel B.

Do NOT wire this into gdn2_pipeline.py's _gdn2_core_bwd until Gate 1 (A)
below passes on real TPU (interpret=False), per the same discipline
already applied to Kernel B and the fused A+B path.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from .configs import KernelConfig, DEFAULT_CONFIG, sanitize

_HIGHEST = jax.lax.Precision.HIGHEST


def _kernel_b2_body_batched(aqk_ref, vnew_ref, do_ref, daqk_ref, dvnew_ref,
                            *, bt: int, config: KernelConfig, group: int):
    Aqk = aqk_ref[0, 0].astype(jnp.float32)      # (group, bt, bt)
    v_new = vnew_ref[0, 0].astype(jnp.float32)   # (group, bt, D)
    do = do_ref[0, 0].astype(jnp.float32)        # (group, bt, D)

    idx = jnp.arange(bt)
    causal = (idx[:, None] >= idx[None, :]).astype(jnp.float32)

    dAqk = jnp.einsum("gid,gjd->gij", do, v_new, precision=_HIGHEST) * causal[None, :, :]
    dv_new = jnp.einsum("gij,gid->gjd", Aqk, do, precision=_HIGHEST)

    daqk_ref[0, 0] = sanitize(dAqk, config)
    dvnew_ref[0, 0] = sanitize(dv_new, config)


def dav_backward_pallas_batched(Aqk, v_new, do, config: KernelConfig = DEFAULT_CONFIG,
                                group: int | None = None, interpret: bool = False):
    """Batched version of gdn2_bwd.dav_backward_pallas.

    Aqk: (bsz, H, n_chunks, bt, bt)
    v_new, do: (bsz, H, n_chunks, bt, D)
    group: chunks fused per grid cell. None => config.b_batch_group or
           n_chunks (same convention as wy_solve_pallas_batched).
    """
    bsz, H, n_chunks, _bt, D = v_new.shape

    if group is None:
        group = config.b_batch_group if config.b_batch_group is not None else n_chunks
    if n_chunks % group != 0:
        raise ValueError(
            f"dav_backward_pallas_batched: n_chunks={n_chunks} must be "
            f"divisible by group={group}."
        )
    n_groups = n_chunks // group

    grid = (bsz, H, n_groups)
    aqk_spec = pl.BlockSpec((1, 1, group, config.bt, config.bt), lambda i, h, gi: (i, h, gi, 0, 0))
    io_spec = pl.BlockSpec((1, 1, group, config.bt, D), lambda i, h, gi: (i, h, gi, 0, 0))

    dAqk, dv_new = pl.pallas_call(
        lambda *refs: _kernel_b2_body_batched(*refs, bt=config.bt, config=config, group=group),
        grid=grid,
        in_specs=[aqk_spec, io_spec, io_spec],
        out_specs=[aqk_spec, io_spec],
        out_shape=[
            jax.ShapeDtypeStruct(Aqk.shape, jnp.float32),
            jax.ShapeDtypeStruct(v_new.shape, jnp.float32),
        ],
        compiler_params=pltpu.CompilerParams(vmem_limit_bytes=96 * 1024 * 1024),
        interpret=interpret,
    )(Aqk, v_new, do)
    return dAqk, dv_new
B2MODULEEOF
echo "    OK: $PKG_DIR/gdn2_bwd_batched_b2.py"
echo "    NOTE: НЕ подключён к gdn2_pipeline.py::_gdn2_core_bwd -- по дизайну,"
echo "          до прохождения Gate 1 (A) на реальном TPU (interpret=False)."

# -----------------------------------------------------------------------------
# ШАГ 7 -- tests/test_b2_batched_vs_nonbatched.py (import-путь уже под Atomic_ops)
# -----------------------------------------------------------------------------
echo ""
echo "=================================================================="
echo ">>> ШАГ 7: tests/test_b2_batched_vs_nonbatched.py"
echo "=================================================================="

cat > "tests/test_b2_batched_vs_nonbatched.py" <<'B2TESTEOF'
"""
Gate 1 (A)-equivalent for the B2 batching prototype (Atomic_ops/gdn2_bwd_batched_b2.py).
Same discipline as gate1_wy_solve_batched.py's section (A): batched vs
non-batched must be near bit-identical (same math, different Pallas
pattern), multi-seed including extreme g/k/b, checked separately from
any comparison against token-serial ground truth.

Run on CPU first (interpret=True, default here). Before wiring
dav_backward_pallas_batched into gdn2_pipeline.py's _gdn2_core_bwd,
re-run with interpret=False on real TPU -- interpret=True only proves
the math didn't change, not that it lowers on Mosaic (same caveat as
gate1_wy_solve_batched.py's (A2) section).
"""
from __future__ import annotations

import numpy as np
import jax.numpy as jnp
import pytest

from Atomic_ops.configs import KernelConfig
from Atomic_ops.gdn2_fwd import (
    build_chunk_scores_pallas, wy_solve_pallas, recompute_wy_pallas,
    gdn2_inter_chunk_combine_with_state,
)
from Atomic_ops.gdn2_bwd import dav_backward_pallas
from Atomic_ops.gdn2_bwd_batched_b2 import dav_backward_pallas_batched

INTERPRET = True

CFG_BASE = dict(bt=128, bc=64, mb=16, clip=1e4, wy_eps=1e-3)
BSZ, H, N_CHUNKS = 2, 2, 4
D = 128
L = N_CHUNKS * CFG_BASE["bt"]

SEEDS = [
    dict(name="benign_seed0", seed=0, g_scale=0.05, k_scale=1.0, b_scale=0.5),
    dict(name="extreme_strong_decay", seed=2, g_scale=3.0, k_scale=1.0, b_scale=0.5),
    dict(name="extreme_large_kb", seed=4, g_scale=0.3, k_scale=8.0, b_scale=4.0),
    dict(name="extreme_mixed_sign_g", seed=6, g_scale=1.5, k_scale=2.0, b_scale=0.7),
]

GROUP_CHOICES = [1, N_CHUNKS]


def make_inputs(seed, g_scale, k_scale, b_scale):
    rng = np.random.default_rng(seed)
    shape = (BSZ, L, H, D)
    q = rng.normal(size=shape).astype(np.float32) * 0.1
    k = rng.normal(size=shape).astype(np.float32) * 0.1 * k_scale
    v = rng.normal(size=shape).astype(np.float32) * 0.1
    w = np.ones(shape, dtype=np.float32)
    b = (b_scale * rng.uniform(0.5, 1.0, size=shape)).astype(np.float32)
    g = -np.abs(rng.normal(size=shape)).astype(np.float32) * g_scale
    return tuple(jnp.asarray(x) for x in (q, k, v, w, b, g))


def rel_err(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.linalg.norm((a - b).ravel()) / max(np.linalg.norm(b.ravel()), 1e-12))


def _build_real_residuals(q, k, v, w, b, g, scale, config):
    """Builds real Aqk/v_new/do (via a synthetic upstream cotangent) so
    the B2 comparison runs on actual forward residuals, not toy arrays --
    matching the honest-benchmarking principle already applied elsewhere
    in this codebase (full_fwd_bwd_efficiency.py's measure_all)."""
    Aqk, Akk = build_chunk_scores_pallas(q, k, b, g, scale, config, interpret=INTERPRET)
    A = wy_solve_pallas(Akk, config, interpret=INTERPRET)
    w_pseudo, u, kg, qg, gc_last = recompute_wy_pallas(
        q, k, v, w, b, g, A, config, interpret=INTERPRET,
    )
    o, h_final, h_pre_all, v_new_all = gdn2_inter_chunk_combine_with_state(
        Aqk, w_pseudo, u, kg, qg, gc_last, scale, config=config,
    )
    v_new_all = jnp.moveaxis(v_new_all, 0, 2)
    do = jnp.moveaxis(
        jnp.asarray(np.random.default_rng(999).normal(size=o.shape).astype(np.float32)) * 0.01,
        0, 0,
    )
    return Aqk, v_new_all, do


@pytest.mark.parametrize("spec", SEEDS, ids=lambda s: s["name"])
@pytest.mark.parametrize("group", GROUP_CHOICES)
def test_b2_batched_matches_nonbatched(spec, group):
    config = KernelConfig(**CFG_BASE, b_batch_group=group)
    scale = 1.0 / np.sqrt(D)
    q, k, v, w, b, g = make_inputs(spec["seed"], spec["g_scale"], spec["k_scale"], spec["b_scale"])

    Aqk, v_new, do = _build_real_residuals(q, k, v, w, b, g, scale, config)

    dAqk_nb, dv_nb = dav_backward_pallas(Aqk, v_new, do, config=config)
    dAqk_b, dv_b = dav_backward_pallas_batched(Aqk, v_new, do, config=config,
                                               group=group, interpret=INTERPRET)

    err_daqk = rel_err(dAqk_nb, dAqk_b)
    err_dv = rel_err(dv_nb, dv_b)

    assert np.all(np.isfinite(np.asarray(dAqk_b))), f"non-finite dAqk on {spec['name']}, group={group}"
    assert np.all(np.isfinite(np.asarray(dv_b))), f"non-finite dv_new on {spec['name']}, group={group}"
    assert err_daqk < 1e-4, f"dAqk diverges: rel_err={err_daqk:.2e} ({spec['name']}, group={group})"
    assert err_dv < 1e-4, f"dv_new diverges: rel_err={err_dv:.2e} ({spec['name']}, group={group})"
B2TESTEOF
echo "    OK: tests/test_b2_batched_vs_nonbatched.py (import уже Atomic_ops.gdn2_bwd_batched_b2)"

# -----------------------------------------------------------------------------
# ШАГ 8 -- tests/test_regressions_bundle.py
# -----------------------------------------------------------------------------
echo ""
echo "=================================================================="
echo ">>> ШАГ 8: tests/test_regressions_bundle.py"
echo "=================================================================="

cat > "tests/test_regressions_bundle.py" <<'REGRESSIONSEOF'
"""
Regression bundle covering Current_state.md / What_next.md "срочное" +
closed-P0 items that had no regression test attached yet:

  - Открыто #1: `_MAX_VALIDATED_GROUP = 16` устарел -> formula guard
  - Закрыто #3/#4 (P0.1/P0.2): wy_eps/clip plumbing fallback<->reference
  - Закрыто #5 (P0.3) + N3: centering gate restored, DEFAULT_CONFIG
    reverted to use_centering=False
  - N5: extreme_large_kb should no longer be nan after _sanitize_ref fix

Run: pytest tests/test_regressions_bundle.py -v
All tests run on CPU (interpret=True where relevant) -- no TPU needed.
"""
from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
import pytest

from Atomic_ops.configs import (
    KernelConfig, DEFAULT_CONFIG, KAGGLE_SMALL, KAGGLE_MEDIUM, KAGGLE_LARGE,
    KAGGLE_MEDIUM_CENTERED,
)
from Atomic_ops.fallback import gdn2_forward
from Atomic_ops.reference import gdn2_chunked_wy_reference
from Atomic_ops.gdn2_fwd_batched import _max_validated_group, wy_solve_pallas_batched


# ===========================================================================
# 1. Group guard formula (Current_state.md "срочное" #1)
# ===========================================================================
class TestGroupGuardFormula:
    @pytest.mark.parametrize("bt,expected", [
        (128, 64),
        (256, 16),   # <-- единственная реально измеренная на TPU точка
        (512, 4),
    ])
    def test_formula_matches_known_points(self, bt, expected):
        assert _max_validated_group(bt) == expected, (
            f"_max_validated_group({bt}) diverges from the MB8-measured "
            f"anchor point (bt=256 -> group=16). If this fails after an "
            f"edit, check whether the formula's bt=256 anchor changed "
            f"without a new TPU measurement to back it."
        )

    def test_at_limit_no_warning(self, recwarn):
        config = KernelConfig(bt=256, bc=128, mb=16, wy_eps=1e-3)
        Akk = jnp.zeros((1, 1, 16, config.bt, config.bt), dtype=jnp.float32)
        wy_solve_pallas_batched(Akk, config, group=16, interpret=True)
        assert not any(issubclass(w.category, RuntimeWarning) for w in recwarn.list)

    def test_over_limit_warns(self, recwarn):
        config = KernelConfig(bt=256, bc=128, mb=16, wy_eps=1e-3)
        Akk = jnp.zeros((1, 1, 32, config.bt, config.bt), dtype=jnp.float32)
        wy_solve_pallas_batched(Akk, config, group=32, interpret=True)
        assert any(issubclass(w.category, RuntimeWarning) for w in recwarn.list), (
            "group=32 > _max_validated_group(256)=16 should warn -- if this "
            "fails, the guard inside wy_solve_pallas_batched still checks "
            "against the old hardcoded constant instead of the formula."
        )


# ===========================================================================
# 2. fallback.py wy_eps/clip plumbing (P0.1/P0.2, closed -- regression guard)
# ===========================================================================
def _make_inputs(seed=0, bt=128):
    key = jax.random.PRNGKey(seed)
    ks = jax.random.split(key, 6)
    shape = (2, bt, 2, 128)
    q, k, v, w, b, g = (jax.random.normal(ks[i], shape) * 0.3 for i in range(6))
    g = -jnp.abs(g) * 0.3
    return q, k, v, w, b, g


class TestFallbackConfigPlumbing:
    @pytest.mark.parametrize("wy_eps,clip", [(1e-3, 1e4), (5e-2, 5e3)])
    def test_fallback_uses_config_wy_eps_and_clip(self, wy_eps, clip):
        config = KernelConfig(bt=128, bc=64, mb=16, wy_eps=wy_eps, clip=clip)
        q, k, v, w, b, g = _make_inputs(bt=config.bt)

        out_fallback, h_fallback = gdn2_forward(q, k, v, w, b, g, scale=1.0, config=config)
        out_ref, h_ref = gdn2_chunked_wy_reference(
            q, k, v, g, b, w, scale=1.0, chunk_size=config.bt,
            wy_eps=config.wy_eps, clip=config.clip,
        )
        assert jnp.allclose(out_fallback, out_ref, atol=0.0, rtol=0.0)
        assert jnp.allclose(h_fallback, h_ref, atol=0.0, rtol=0.0)

        wrong_eps = 0.0 if wy_eps != 0.0 else 1e-2
        out_wrong, _ = gdn2_chunked_wy_reference(
            q, k, v, g, b, w, scale=1.0, chunk_size=config.bt,
            wy_eps=wrong_eps, clip=config.clip,
        )
        assert not jnp.allclose(out_fallback, out_wrong, atol=1e-6), (
            "wy_eps has no visible effect on this input -- test cannot "
            "distinguish correct plumbing from broken plumbing; use a "
            "chunk with worse Akk conditioning."
        )

    def test_sanitize_ref_clips_not_just_nan_to_num_N5(self):
        """N5: extreme_large_kb should no longer produce nan after the
        real jnp.clip was added to _sanitize_ref (not just nan_to_num)."""
        key = jax.random.PRNGKey(1)
        ks = jax.random.split(key, 6)
        shape = (1, 128, 1, 128)
        scale_large = 50.0
        q = jax.random.normal(ks[0], shape) * 0.3
        k = jax.random.normal(ks[1], shape) * scale_large
        v = jax.random.normal(ks[2], shape) * 0.3
        w = jax.random.uniform(ks[3], shape, minval=0.5, maxval=1.0)
        b = jax.random.uniform(ks[4], shape, minval=0.5, maxval=1.0) * scale_large
        g = -jnp.abs(jax.random.normal(ks[5], shape)) * 0.1

        config = KernelConfig(bt=128, bc=64, mb=16, wy_eps=1e-3, clip=1e4)
        out, h_final = gdn2_forward(q, k, v, w, b, g, scale=1.0, config=config)

        assert jnp.all(jnp.isfinite(out)), (
            "N5 not closed: extreme_large_kb still produces nan/inf -- "
            "_sanitize_ref is not clipping w_pseudo/u the way the "
            "docstring in reference.py claims."
        )
        assert jnp.all(jnp.isfinite(h_final))
        assert jnp.max(jnp.abs(out)) <= config.clip * 10


# ===========================================================================
# 3. Centering gate (P0.3) + N3 (DEFAULT_CONFIG behavioral change)
# ===========================================================================
class TestCenteringGateAndDefault:
    def test_default_config_is_noncentered(self):
        assert DEFAULT_CONFIG.use_centering is False

    @pytest.mark.parametrize("preset", [KAGGLE_SMALL, KAGGLE_MEDIUM, KAGGLE_LARGE])
    def test_kaggle_presets_are_noncentered(self, preset):
        assert preset.use_centering is False

    def test_centering_requires_explicit_ack(self):
        with pytest.raises(NotImplementedError):
            KernelConfig(bt=256, bc=128, mb=16, wy_eps=1e-3, use_centering=True)

    def test_centering_with_ack_succeeds(self):
        cfg = KernelConfig(bt=256, bc=128, mb=16, wy_eps=1e-3,
                            use_centering=True, unsafe_allow_centering=True)
        assert cfg.use_centering is True

    def test_centered_preset_already_has_ack(self):
        assert KAGGLE_MEDIUM_CENTERED.use_centering is True
        assert KAGGLE_MEDIUM_CENTERED.unsafe_allow_centering is True
REGRESSIONSEOF
echo "    OK: tests/test_regressions_bundle.py"

cd "$ROOT_DIR"
echo ""
echo "=================================================================="
echo ">>> ГОТОВО. Патчи применены, тесты НЕ прогонялись."
echo "=================================================================="
echo "Бэкапы: $BACKUP_DIR/"