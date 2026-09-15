"""
gdn2_blr.phase0_patches -- патчи Фазы 0 для СТАРОГО пакета Atomic_ops.

Все пять дыр подтверждены ИСПОЛНЕНИЕМ (прогон gdn2_hypothesis_sweep_v2.py):

  H0.1  reference.py: `return o, h_fina` -> NameError, весь fallback мёртв.
  H0.2/3 `interpret` не доведён до gdn2_pallas_forward_trainable,
         dav_backward_pallas, wy_dqkg_backward_pallas -> Gate 1 (CPU) для
         backward и training-пути физически невозможен.
  H0.4  KAGGLE_MEDIUM (b_batch_group=16) падает при n_chunks=4 (L=1024).
  H0.5  wy_dqkg_backward_pallas принимает Akk и НЕ читает его
         (Akk x100 -> все семь выходов бит-в-бит те же) -> ~201 MB
         мёртвого HBM-чтения за backward в compute-bound кернеле.
  H0.6  _kernel_c_body: exp(gc) без клипа -> на mixed-sign 22.4% kg и
         19.8% qg упираются в границу sanitize.

Запуск (dry-run, ничего не меняет):
    python -c "from gdn2_blr.phase0_patches import main; main('/path/to/repo')"
Реальное применение:
    python -c "from gdn2_blr.phase0_patches import main; main('/path/to/repo', dry_run=False)"

Бэкапы пишутся рядом с суффиксом .bak_phase0.
"""
from __future__ import annotations

import os
import shutil

Patch = tuple  # (relative_path, old, new, why)

PATCHES: list[Patch] = [
    (
        "reference.py",
        "    return o, h_fina",
        "    return o, h_final",
        "H0.1: NameError -- gdn2_chunked_wy_reference падала на последней "
        "строке, из-за чего оба входа fallback.py были мертвы.",
    ),
    (
        "configs.py",
        "    unsafe_allow_centering: bool = False",
        "    unsafe_allow_centering: bool = False\n"
        "    # PATCH Phase 0 (H0.2/H0.3): interpret живёт в конфиге, а не в\n"
        "    # сигнатуре каждой функции. Конфиг уже hashable и уже проходит\n"
        "    # через custom_vjp как nondiff -- это единственное место, из\n"
        "    # которого его видят ВСЕ pallas_call, включая backward.\n"
        "    interpret: bool = False",
        "H0.2/H0.3: без этого Gate 1 (CPU) для backward невозможен.",
    ),
    (
        "gdn2_fwd_batched.py",
        "    if n_chunks % group != 0:\n"
        "        raise ValueError(\n"
        "            f\"wy_solve_pallas_batched: n_chunks={n_chunks} должен делиться \"\n"
        "            f\"на group={group} (жёсткая проверка -- группа всегда покрывает \"\n"
        "            f\"ровное число чанков).\"\n"
        "        )",
        "    # PATCH Phase 0 (H0.4): вместо жёсткого ValueError -- наибольший\n"
        "    # делитель n_chunks, не превосходящий group. Старое поведение\n"
        "    # требовало L кратной group*bt (4096 для KAGGLE_MEDIUM, 8192 для\n"
        "    # KAGGLE_SMALL) и падало на совершенно обычной L=1024.\n"
        "    if n_chunks % group != 0:\n"
        "        _g = int(group)\n"
        "        while _g > 1 and n_chunks % _g != 0:\n"
        "            _g -= 1\n"
        "        group = max(1, _g)",
        "H0.4: KAGGLE_MEDIUM падал на L=1024.",
    ),
    (
        "gdn2_fwd.py",
        "    kb_decayed = b * k * jnp.exp(gc)\n"
        "    w_pseudo = jnp.dot(A, kb_decayed, precision=_HIGHEST)",
        "    # PATCH Phase 0 (H0.6): min(.,0) перед exp. При g<=0 (инвариант\n"
        "    # forget-gate'а, alpha=exp(g) in (0,1]) это тождественный no-op;\n"
        "    # при нарушении инварианта ограничивает вместо inf. Замерено:\n"
        "    # без клипа на mixed-sign входе 22.4% kg и 19.8% qg упирались в\n"
        "    # границу sanitize, то есть overflow был и маскировался.\n"
        "    kb_decayed = b * k * jnp.exp(jnp.minimum(gc, 0.0))\n"
        "    w_pseudo = jnp.dot(A, kb_decayed, precision=_HIGHEST)",
        "H0.6: Kernel C считал exp(gc) без клипа.",
    ),
    (
        "gdn2_fwd.py",
        "    kg = k * jnp.exp(gc_last_row[None, :] - gc)\n"
        "    qg = q * jnp.exp(gc)",
        "    kg = k * jnp.exp(jnp.minimum(gc_last_row[None, :] - gc, 0.0))\n"
        "    qg = q * jnp.exp(jnp.minimum(gc, 0.0))",
        "H0.6 (продолжение): те же клипы для kg/qg.",
    ),
]

# Патчи, которые НЕ делаются автоматически -- они требуют согласованного
# изменения нескольких файлов и должны пройти ревью руками.
MANUAL: list[tuple[str, str]] = [
    (
        "H0.5 -- удалить мёртвый вход Akk из B3",
        "gdn2_bwd.py: убрать `akk_ref` из сигнатуры _kernel_b3_body, убрать "
        "второй `score_spec` из in_specs в wy_dqkg_backward_pallas, убрать "
        "`Akk` из списка аргументов вызова; в gdn2_pipeline.py убрать `Akk` "
        "из вызова wy_dqkg_backward_pallas. Доказательство мёртвости: "
        "Akk x100 не меняет ни один из семи выходов (проба S0.5). Экономия "
        "на train_shape ~201 MB HBM-чтения за каждый backward.",
    ),
    (
        "H0.2/H0.3 -- протянуть config.interpret",
        "Во всех pl.pallas_call в gdn2_fwd.py / gdn2_bwd.py / "
        "gdn2_fwd_batched*.py / gdn2_bwd_batched*.py заменить "
        "`interpret=interpret` и отсутствующий аргумент на "
        "`interpret=config.interpret`. Отдельно: gdn2_pipeline."
        "gdn2_pallas_forward_trainable перестаёт нуждаться в новом "
        "параметре -- конфиг уже nondiff-аргумент custom_vjp.",
    ),
    (
        "H0.6 -- согласовать backward с клипами",
        "В _kernel_b3_body домножить dgc_from_kb, dgc_from_qg на "
        "1{gc<0}, а dgc_from_kg и dgc_last_contrib на "
        "1{gc_last-gc<0}; dgc_last_from_decay на 1{gc_last<0}. Без этого "
        "backward рассогласован с forward ровно там, где клип срабатывает. "
        "Готовая формула -- gdn2_blr/bwd.py::wy_dqkg_backward.",
    ),
    (
        "gc считается четыре раза",
        "gc = tril_ones @ g_raw пересчитывается в _kernel_a_body, "
        "_kernel_c_body, _kernel_b4_body и ещё раз в _gdn2_core_bwd -- "
        "четыре матмула bt^2*D. Считать ОДИН раз jnp.cumsum в XLA и "
        "прокидывать как вход (так сделано в gdn2_blr).",
    ),
]


def _find_pkg(root: str) -> str:
    for name in ("Atomic_ops", "atomic_ops"):
        p = os.path.join(root, name)
        if os.path.isdir(p):
            return p
    raise FileNotFoundError(f"ни Atomic_ops, ни atomic_ops не найдены в {root}")


def main(root: str = ".", dry_run: bool = True) -> dict:
    pkg = _find_pkg(root)
    report = {"pkg": pkg, "applied": [], "already": [], "missing": [],
              "manual": [m[0] for m in MANUAL]}
    for rel, old, new, why in PATCHES:
        path = os.path.join(pkg, rel)
        if not os.path.isfile(path):
            report["missing"].append(f"{rel} (файла нет)")
            continue
        src = open(path, encoding="utf-8").read()
        if new in src:
            report["already"].append(f"{rel}: {why}")
            continue
        if old not in src:
            report["missing"].append(f"{rel}: не найден якорь -- {why}")
            continue
        if not dry_run:
            bak = path + ".bak_phase0"
            if not os.path.exists(bak):
                shutil.copy2(path, bak)
            open(path, "w", encoding="utf-8").write(src.replace(old, new, 1))
        report["applied"].append(f"{rel}: {why}")

    print(f"пакет: {pkg}   dry_run={dry_run}")
    for key in ("applied", "already", "missing"):
        for line in report[key]:
            print(f"  [{key}] {line}")
    print("  --- требуют ручного ревью ---")
    for name, body in MANUAL:
        print(f"  [manual] {name}\n           {body}")
    return report
