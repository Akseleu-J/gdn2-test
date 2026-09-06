#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------------------
# Ожидаемая структура на момент запуска:
#   ./Atomic_ops/            <- уже существует, это наш рабочий пакет
#   ./run_this_script.sh     <- этот скрипт лежит рядом
#
# Скрипт НЕ клонирует Atomic_ops. Он клонирует gdn2-pallas во временную
# папку только чтобы вытащить оттуда тесты, адаптирует импорты под
# "Atomic_ops" (наш регистр) и кладёт их в ./tests рядом с Atomic_ops.
# ---------------------------------------------------------------------------

ROOT_DIR="$(pwd)"
PKG_DIR="Atomic_ops"
CFG_FILE="$PKG_DIR/configs.py"
SRC_REPO_URL="https://github.com/Akseleu-J/gdn2-pallas"
TMP_CLONE_DIR="$(mktemp -d)"

if [ ! -d "$PKG_DIR" ]; then
    echo "!!! $PKG_DIR не найден в $(pwd). Запускай скрипт из директории, где рядом лежит Atomic_ops."
    exit 1
fi
if [ ! -f "$CFG_FILE" ]; then
    echo "!!! $CFG_FILE не найден."
    exit 1
fi

cleanup() { rm -rf "$TMP_CLONE_DIR"; }
trap cleanup EXIT

# ---------------------------------------------------------------------------
# 1. Клонируем исходный репо ТОЛЬКО ради тестов (во временную папку)
# ---------------------------------------------------------------------------
echo ">>> Клонирую $SRC_REPO_URL во временную папку (только за тестами)..."
git clone --depth 1 "$SRC_REPO_URL" "$TMP_CLONE_DIR/gdn2-pallas"
SRC="$TMP_CLONE_DIR/gdn2-pallas"

# ---------------------------------------------------------------------------
# 2. Бэкап и патч Atomic_ops/configs.py (снимаем gate use_centering)
# ---------------------------------------------------------------------------
cp "$CFG_FILE" "${CFG_FILE}.bak_pre_centering"
echo ">>> Бэкап: ${CFG_FILE}.bak_pre_centering"

python3 - "$CFG_FILE" <<'PYEOF'
import re, sys, pathlib

path = pathlib.Path(sys.argv[1])
src = path.read_text()

# 2a. use_centering: False -> True
new_src = src.replace(
    "    use_centering: bool = False\n",
    "    use_centering: bool = True\n",
    1,
)
if new_src == src:
    print("!!! WARNING: 'use_centering: bool = False' не найден -- проверь вручную.")
src = new_src

# 2b. Вырезаем NotImplementedError gate
pattern = re.compile(
    r"        if self\.use_centering and not self\.unsafe_allow_centering:\n"
    r"(?:.*\n)*?"
    r'            \)\n',
)
replacement = (
    "        # NOTE: use_centering=True больше не гейтится -- теперь дефолт\n"
    "        # во всех KAGGLE_* пресетах и DEFAULT_CONFIG. Для старого\n"
    "        # (pre-centering) VPU-пути передайте use_centering=False явно.\n"
)
new_src, n = pattern.subn(replacement, src, count=1)
if n == 0:
    print("!!! WARNING: gate-блок NotImplementedError не найден -- проверь configs.py вручную.")
else:
    src = new_src

# 2c. Пресеты -> use_centering=True + добавляем *_NOCENTER для A/B сравнения
old_presets = (
    'KAGGLE_SMALL = KernelConfig(bt=128, bc=64, mb=16, clip=1e4, wy_eps=1e-3)\n'
    'KAGGLE_MEDIUM = KernelConfig(bt=256, bc=128, mb=16, clip=1e4, wy_eps=1e-3)\n'
    'KAGGLE_LARGE = KernelConfig(bt=256, bc=128, mb=16, clip=5e3, wy_eps=1e-3)\n'
)
new_presets = (
    'KAGGLE_SMALL = KernelConfig(bt=128, bc=64, mb=16, clip=1e4, wy_eps=1e-3, use_centering=True)\n'
    'KAGGLE_MEDIUM = KernelConfig(bt=256, bc=128, mb=16, clip=1e4, wy_eps=1e-3, use_centering=True)\n'
    'KAGGLE_LARGE = KernelConfig(bt=256, bc=128, mb=16, clip=5e3, wy_eps=1e-3, use_centering=True)\n'
    '\n'
    '# Pre-centering путь -- для A/B сравнения (speed/memory "до/после").\n'
    '# Не используется дефолтными entrypoint-ами.\n'
    'KAGGLE_SMALL_NOCENTER = KernelConfig(bt=128, bc=64, mb=16, clip=1e4, wy_eps=1e-3, use_centering=False)\n'
    'KAGGLE_MEDIUM_NOCENTER = KernelConfig(bt=256, bc=128, mb=16, clip=1e4, wy_eps=1e-3, use_centering=False)\n'
    'KAGGLE_LARGE_NOCENTER = KernelConfig(bt=256, bc=128, mb=16, clip=5e3, wy_eps=1e-3, use_centering=False)\n'
)
if old_presets in src:
    src = src.replace(old_presets, new_presets, 1)
else:
    print("!!! WARNING: блок KAGGLE_* пресетов не найден в ожидаемом виде -- добавь use_centering=True вручную.")

path.write_text(src)
print(">>> configs.py пропатчен.")
PYEOF

# ---------------------------------------------------------------------------
# 3. Копируем тесты из клона в наш проект, адаптируя импорты под "Atomic_ops"
#    - atomic_ops.xxx      -> Atomic_ops.xxx
#    - from atomic_ops     -> from Atomic_ops
#    - import atomic_ops   -> import Atomic_ops
# ---------------------------------------------------------------------------
mkdir -p tests/extended

copy_and_patch() {
    local src_file="$1"
    local dst_file="$2"
    if [ ! -f "$src_file" ]; then
        echo "    (пропуск, нет в исходном репо: $src_file)"
        return
    fi
    mkdir -p "$(dirname "$dst_file")"
    sed -E \
        -e 's/from atomic_ops/from Atomic_ops/g' \
        -e 's/import atomic_ops/import Atomic_ops/g' \
        -e 's/\batomic_ops\./Atomic_ops./g' \
        "$src_file" > "$dst_file"
    echo "    OK: $dst_file"
}

echo ">>> Копирую и патчу важные тесты..."

# Базовые (Layer 0-1, дешёвые smoke)
copy_and_patch "$SRC/tests/test_imports.py"                 "tests/test_imports.py"
copy_and_patch "$SRC/tests/test_configs.py"                 "tests/test_configs.py"
copy_and_patch "$SRC/tests/test_reference.py"                "tests/test_reference.py"
copy_and_patch "$SRC/tests/test_pallas.py"                   "tests/test_pallas.py"
copy_and_patch "$SRC/tests/test_clip_config_plumbing.py"     "tests/test_clip_config_plumbing.py"
copy_and_patch "$SRC/tests/test_gdn2_full_math_correctness.py" "tests/test_gdn2_full_math_correctness.py"

# Extended (TPU deep-correctness)
copy_and_patch "$SRC/tests/extended/test_gdn2_deep_correctness_mini.py" \
                "tests/extended/test_gdn2_deep_correctness_mini.py"
copy_and_patch "$SRC/tests/extended/test_gdn2_deep_correctness.py" \
                "tests/extended/test_gdn2_deep_correctness.py"

# ---------------------------------------------------------------------------
# 4. Пишем centering-suite напрямую под Atomic_ops (этого файла с таким
#    содержимым в апстриме ещё нет / он приватный -- берём вашу версию)
# ---------------------------------------------------------------------------
CENTERING_TEST="tests/extended/test_gdn2_deep_correctness_centering.py"
cat > "$CENTERING_TEST" <<'PYEOF'
"""
test_gdn2_deep_correctness_centering.py

Полный deep-correctness suite для целого пайплайна
(gdn2_pallas_forward_trainable, весь custom_vjp: Kernel A/B/C/D + B1-B5)
с use_centering=True. Дополняет изолированные тесты Kernel A/B4.
"""
from __future__ import annotations

import dataclasses as dc
import sys

import jax
import jax.numpy as jnp

from Atomic_ops.configs import KernelConfig, KAGGLE_SMALL, KAGGLE_MEDIUM
from Atomic_ops.gdn2_pipeline import gdn2_pallas_forward_trainable
from Atomic_ops.gdn2_fwd import gdn2_pallas_forward
from Atomic_ops.reference import gdn2_token_serial_reference

_HIGHEST = jax.lax.Precision.HIGHEST
_FAILURES = []


def _check(name, rel_err, tol, extra=""):
    status = "PASS" if rel_err <= tol else "FAIL"
    print(f"[{status}] {name}: rel_err={rel_err:.3e}  (tol={tol:.1e}) {extra}")
    if rel_err > tol:
        _FAILURES.append(name)
    return rel_err <= tol


def _rel_err(a, b):
    a = jnp.asarray(a, dtype=jnp.float32)
    b = jnp.asarray(b, dtype=jnp.float32)
    num = jnp.max(jnp.abs(a - b))
    den = jnp.maximum(jnp.max(jnp.abs(b)), 1e-8)
    return float(num / den)


def _centered_config(base: KernelConfig, **overrides) -> KernelConfig:
    fields = {f.name: getattr(base, f.name) for f in dc.fields(base)}
    fields.update(overrides)
    fields["use_centering"] = True
    fields["unsafe_allow_centering"] = True
    return KernelConfig(**fields)


def _make_inputs(key, bsz, n_chunks, bt, H, D, decay_scale, h0_nonzero=False, dtype=jnp.float32):
    L = n_chunks * bt
    k1, k2, k3, k4, k5 = jax.random.split(key, 5)
    shape = (bsz, L, H, D)

    q = jax.random.normal(k1, shape)
    k = jax.random.normal(k2, shape)
    q = q / (jnp.linalg.norm(q, axis=-1, keepdims=True) + 1e-6)
    k = k / (jnp.linalg.norm(k, axis=-1, keepdims=True) + 1e-6)
    v = jax.random.normal(k3, shape) * 0.5
    w = jax.random.uniform(k4, shape, minval=0.2, maxval=1.0)
    b = jax.random.uniform(jax.random.fold_in(k4, 1), shape, minval=0.2, maxval=1.0)

    if decay_scale <= 0.0:
        g = jnp.zeros(shape, dtype=jnp.float32)
    else:
        g = -jnp.abs(jax.random.normal(k5, shape)) * decay_scale

    h0 = None
    if h0_nonzero:
        h0 = jax.random.normal(jax.random.fold_in(key, 99), (bsz, H, D, D)) * 0.1

    q, k, v, w, b = (t.astype(dtype) for t in (q, k, v, w, b))
    g = g.astype(jnp.float32)
    return q, k, v, w, b, g, h0


def test_multiseed_sweep_centered(cfg):
    print("\n--- C2: multi-seed sweep, use_centering=True (fwd+bwd vs token-serial) ---")
    bt = cfg["bt"]
    config = _centered_config(KAGGLE_MEDIUM, bt=bt, bc=bt // 2, mb=min(16, bt // 2), wy_eps=0.0)
    n_chunks = 2

    for seed in range(cfg["n_seeds"]):
        key = jax.random.PRNGKey(5000 + seed)
        q, k, v, w, b, g, h0 = _make_inputs(key, cfg["bsz"], n_chunks, bt,
                                             cfg["H"], cfg["D"], decay_scale=0.15,
                                             h0_nonzero=True)

        o_pallas, h_final_pallas = gdn2_pallas_forward(q, k, v, w, b, g, scale=1.0, h0=h0, config=config)
        o_ref, h_final_ref = gdn2_token_serial_reference(q, k, v, g, b, w, scale=1.0, h0=h0)
        _check(f"centered.sweep[seed={seed}].fwd.o", _rel_err(o_pallas, o_ref), 2e-2)
        _check(f"centered.sweep[seed={seed}].fwd.h_final", _rel_err(h_final_pallas, h_final_ref), 2e-2)

        rkey = jax.random.fold_in(key, 777)
        r1, r2 = jax.random.split(rkey)
        do_rand = jax.random.normal(r1, o_pallas.shape)
        dh_rand = jax.random.normal(r2, h_final_pallas.shape)

        def honest_loss(q_, k_, v_, w_, b_, g_, h0_):
            o, hf = gdn2_pallas_forward_trainable(q_, k_, v_, w_, b_, g_, scale=1.0, h0=h0_, config=config)
            return jnp.sum(o * do_rand) + jnp.sum(hf * dh_rand)

        def ref_loss(q_, k_, v_, w_, b_, g_, h0_):
            o, hf = gdn2_token_serial_reference(q_, k_, v_, g_, b_, w_, scale=1.0, h0=h0_)
            return jnp.sum(o * do_rand) + jnp.sum(hf * dh_rand)

        hg = jax.grad(honest_loss, argnums=(0, 1, 2, 3, 4, 5, 6))(q, k, v, w, b, g, h0)
        rg = jax.grad(ref_loss, argnums=(0, 1, 2, 3, 4, 5, 6))(q, k, v, w, b, g, h0)
        for name, h, r in zip(["dq", "dk", "dv", "dw", "db", "dg", "dh0"], hg, rg):
            _check(f"centered.sweep[seed={seed}].bwd.{name}", _rel_err(h, r), 5e-2)


def test_wy_eps_damping_centered(cfg):
    print("\n--- C3: wy_eps > 0 damping, use_centering=True ---")
    bt = cfg["bt"]
    for wy_eps in (1e-3, 1e-2):
        config = _centered_config(KAGGLE_MEDIUM, bt=bt, bc=bt // 2, mb=min(16, bt // 2), wy_eps=wy_eps)
        n_chunks = 2
        key = jax.random.PRNGKey(6000 + int(wy_eps * 1e5))
        q, k, v, w, b, g, h0 = _make_inputs(key, cfg["bsz"], n_chunks, bt, cfg["H"], cfg["D"],
                                             decay_scale=0.1, h0_nonzero=True)

        o_pallas, h_final_pallas = gdn2_pallas_forward(q, k, v, w, b, g, scale=1.0, h0=h0, config=config)
        o_ref, h_final_ref = gdn2_token_serial_reference(q, k, v, g, b, w, scale=1.0, h0=h0)
        tol = max(2e-2, wy_eps * 5)
        _check(f"centered.wy_eps={wy_eps}.fwd.o", _rel_err(o_pallas, o_ref), tol)


def test_bf16_inputs_centered(cfg):
    print("\n--- C4: bf16 inputs, use_centering=True ---")
    bt = cfg["bt"]
    config = _centered_config(KAGGLE_MEDIUM, bt=bt, bc=bt // 2, mb=min(16, bt // 2), wy_eps=0.0)
    n_chunks = 2
    key = jax.random.PRNGKey(7111)
    q, k, v, w, b, g, h0 = _make_inputs(key, cfg["bsz"], n_chunks, bt, cfg["H"], cfg["D"],
                                         decay_scale=0.1, h0_nonzero=True, dtype=jnp.bfloat16)

    o_pallas, h_final_pallas = gdn2_pallas_forward_trainable(q, k, v, w, b, g, scale=1.0, h0=h0, config=config)
    o_ref, h_final_ref = gdn2_token_serial_reference(
        q.astype(jnp.float32), k.astype(jnp.float32), v.astype(jnp.float32),
        g, b.astype(jnp.float32), w.astype(jnp.float32), scale=1.0, h0=h0,
    )
    _check("centered.bf16.fwd.o_vs_fp32_token_serial",
           _rel_err(o_pallas.astype(jnp.float32), o_ref), 8e-2)


def test_kaggle_small_config_centered(cfg):
    print("\n--- C5: KAGGLE_SMALL (bt=128), use_centering=True ---")
    config = _centered_config(KAGGLE_SMALL)
    n_chunks = 2
    key = jax.random.PRNGKey(8131)
    q, k, v, w, b, g, h0 = _make_inputs(key, cfg["bsz"], n_chunks, config.bt, cfg["H"], cfg["D"],
                                         decay_scale=0.1, h0_nonzero=True)

    o_pallas, h_final_pallas = gdn2_pallas_forward(q, k, v, w, b, g, scale=1.0, h0=h0, config=config)
    o_ref, h_final_ref = gdn2_token_serial_reference(q, k, v, g, b, w, scale=1.0, h0=h0)
    tol = max(2e-2, config.wy_eps * 5)
    _check("centered.kaggle_small.fwd.o", _rel_err(o_pallas, o_ref), tol)
    _check("centered.kaggle_small.fwd.h_final", _rel_err(h_final_pallas, h_final_ref), tol)


RUN_CONFIG = dict(bsz=2, H=2, D=128, bt=256, n_seeds=5)


def main(cfg=RUN_CONFIG):
    test_multiseed_sweep_centered(cfg)
    test_wy_eps_damping_centered(cfg)
    test_bf16_inputs_centered(cfg)
    test_kaggle_small_config_centered(cfg)

    print("\n" + "=" * 78)
    if _FAILURES:
        print(f"РЕЗУЛЬТАТ: {len(_FAILURES)} провал(ов):")
        for name in _FAILURES:
            print(f"  - {name}")
        sys.exit(1)
    else:
        print("РЕЗУЛЬТАТ: ВСЕ проверки прошли -- use_centering=True корректен "
              "через весь custom_vjp пайплайн.")
        sys.exit(0)


if __name__ == "__main__":
    main()
PYEOF
echo "    OK: $CENTERING_TEST (написан напрямую под Atomic_ops)"

# ---------------------------------------------------------------------------
# 5. conftest.py, чтобы pytest видел Atomic_ops без pip install -e
# ---------------------------------------------------------------------------
if [ ! -f "conftest.py" ]; then
    cat > conftest.py <<'PYEOF'
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
PYEOF
    echo ">>> Добавлен conftest.py (чтобы pytest видел Atomic_ops из корня проекта)."
fi

# ---------------------------------------------------------------------------
# 6. Шаг A: дешёвый CPU/interpret smoke
# ---------------------------------------------------------------------------
echo ""
echo "=================================================================="
echo ">>> ШАГ A: CPU / interpret smoke"
echo "=================================================================="
JAX_PLATFORMS=cpu pytest tests/test_gdn2_full_math_correctness.py -v --tb=short
pytest tests/test_configs.py tests/test_reference.py tests/test_imports.py \
       tests/test_clip_config_plumbing.py tests/test_pallas.py -v --tb=short

if [ -f tests/extended/test_gdn2_deep_correctness_mini.py ]; then
    python tests/extended/test_gdn2_deep_correctness_mini.py
fi

# ---------------------------------------------------------------------------
# 7. Шаг B: TPU deep-correctness (только если TPU доступен)
# ---------------------------------------------------------------------------
echo ""
echo "=================================================================="
echo ">>> ШАГ B: TPU deep-correctness (полный пайплайн + centering)"
echo "=================================================================="
if python3 -c "import jax; jax.devices('tpu')" 2>/dev/null; then
    pytest tests/extended/test_gdn2_deep_correctness.py -v
    python tests/extended/test_gdn2_deep_correctness_centering.py
else
    echo ">>> TPU не найден -- Шаг B пропущен (запусти на Kaggle TPU v5e-8)."
fi

cd "$ROOT_DIR"
echo ""
echo ">>> Готово. Бэкап оригинального configs.py: ${CFG_FILE}.bak_pre_centering"
echo ">>> Тесты лежат в: ./tests и ./tests/extended (рядом с Atomic_ops)."