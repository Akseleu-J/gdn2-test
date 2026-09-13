# audit_task.py
# ============================================================================
# Задача для Kaggle Benchmarks: глубокий аудит кодовой базы.
#
# Запуск через CLI:
#   kaggle b t push audit-task -f audit_task.py -d <user>/<codebase-dataset>
#   kaggle b t run audit-task -m gpt-6-astra --wait
#
# Ожидает, что файлы кодовой базы лежат в /kaggle/input/<dataset>/
# (путь можно переопределить через CODEBASE_ROOT).
# ============================================================================

import os
import textwrap

import kaggle_benchmarks as kbench


# ---------- Конфигурация ----------
CODE_EXTENSIONS = (".py", ".md", ".toml", ".cfg", ".txt")
EXCLUDE_DIRS = {"__pycache__", ".ipynb_checkpoints", ".git", ".venv"}
EXCLUDE_FILES = {
    "audit_task.py",
    "deep_codebase_audit.py",
    "test_gdn2_deep_correctness.py",
    "test_gdn2_deep_correctness_centering.py",
    "test_gdn2_deep_correctness_mini.py",
    "test_clip_config_plumbing.py",
    "test_configs.py",
    "test_gdn2_full_math_correctness.py",
    "test_imports.py",
    "test_pallas.py",
    "test_reference.py",
}

# Куда примонтирован датасет. Можно переопределить через env var.
CODEBASE_ROOT = os.environ.get("CODEBASE_ROOT", "/kaggle/input")
MAX_CHARS_PER_CHUNK = 700_000  # ~90k токенов


# ---------- Хелперы (обычные функции, вызываются из задачи) ----------
def _collect_files(root_dir):
    files = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS]
        for fn in filenames:
            if fn.endswith(CODE_EXTENSIONS) and fn not in EXCLUDE_FILES:
                full = os.path.join(dirpath, fn)
                rel = os.path.relpath(full, root_dir)
                files.append((rel, full))
    return sorted(files)


def _blob(file_list):
    parts = []
    for rel, full in file_list:
        try:
            with open(full, "r", encoding="utf-8", errors="replace") as fh:
                content = fh.read()
        except Exception as e:
            content = f"<не удалось прочитать файл: {e}>"
        lang = (
            "python" if rel.endswith(".py")
            else "markdown" if rel.endswith(".md")
            else "text"
        )
        parts.append(f"### FILE: {rel}\n```{lang}\n{content}\n```\n")
    return "\n".join(parts)


def _chunk(files, max_chars):
    chunks, current, current_len = [], [], 0
    for f in files:
        size = len(_blob([f]))
        if current and current_len + size > max_chars:
            chunks.append(current)
            current, current_len = [], 0
        current.append(f)
        current_len += size
    if current:
        chunks.append(current)
    return chunks


AUDIT_INSTRUCTIONS = textwrap.dedent("""
    Ты проводишь глубокий инженерный аудит кодовой базы.
    Ниже -- один или несколько файлов кодовой базы (может быть частичный
    набор, если кодовая база разбита на чанки; в таком случае явно помечай
    выводы, которые опираются только на этот кусок и могут не учитывать
    остальной код).

    Твоя задача:
    1. Найти потенциальные "дыры": баги, несоответствия между документацией/
       комментариями и реальным кодом, риски корректности -- особенно в местах,
       где в самом коде уже задокументированы прошлые баги/фиксы.
    2. Оценить производительность и архитектурные ограничения: избыточный
       dispatch/launch overhead, неиспользованный батчинг, лишние HBM/IO
       roundtrip'ы, избыточная численная точность.
    3. Дать конкретный, приоритизированный план действий: что чинить и что
       нужно ИЗМЕРИТЬ дальше, с указанием точного файла/функции/строки.

    Отвечай строго по фактам из предоставленного кода. Гипотезы помечай как
    гипотезы. Структура ответа:
    "Дыры", "Производительность", "План действий", "Нужные тесты".
""").strip()

SYNTH_INSTRUCTIONS = textwrap.dedent("""
    Ниже -- результаты аудита ОТДЕЛЬНЫХ частей одной и той же кодовой базы.
    Собери из них ОДИН связный отчёт: убери дублирование, явно укажи связи
    между находками из разных частей, дай единый приоритизированный план.
    Сохрани структуру: "Дыры", "Производительность", "План действий",
    "Нужные тесты".
""").strip()


# ---------- Задача, которую запускает CLI ----------
@kbench.task(name="audit_codebase")
def audit_codebase(llm) -> dict:
    """
    Полный аудит кодовой базы, смонтированной в CODEBASE_ROOT.
    Читает файлы, чанкует, прогоняет каждый чанк через `llm`,
    затем делает синтез-проход, если чанков больше одного.
    """
    files = _collect_files(CODEBASE_ROOT)
    if not files:
        return {
            "error": f"Не найдено файлов в {CODEBASE_ROOT}. "
                     f"Убедитесь, что датасет с кодовой базой примонтирован.",
            "report": "",
        }

    full_blob = _blob(files)
    if len(full_blob) <= MAX_CHARS_PER_CHUNK:
        chunks = [files]
    else:
        chunks = _chunk(files, MAX_CHARS_PER_CHUNK)

    chunk_reports = []
    for i, chunk in enumerate(chunks):
        chunk_label = f"{i + 1}/{len(chunks)}: " + ", ".join(
            rel for rel, _ in chunk
        )
        prompt = (
            f"{AUDIT_INSTRUCTIONS}\n\n"
            f"=== Кодовая база (часть: {chunk_label}) ===\n\n"
            f"{_blob(chunk)}"
        )
        # Каждый вызов llm.prompt() расходует квоту на сервере Kaggle
        chunk_reports.append({
            "chunk_index": i + 1,
            "files": ", ".join(rel for rel, _ in chunk),
            "report": llm.prompt(prompt),
        })

    # Синтез-проход, если чанков больше одного
    if len(chunk_reports) > 1:
        combined = "\n\n---\n\n".join(
            f"### Чанк {r['chunk_index']} ({r['files']}):\n{r['report']}"
            for r in chunk_reports
        )
        final_report = llm.prompt(f"{SYNTH_INSTRUCTIONS}\n\n{combined}")
    else:
        final_report = chunk_reports[0]["report"]

    return {
        "files_total": len(files),
        "chunks_total": len(chunks),
        "chunk_reports": chunk_reports,
        "final_report": final_report,
        "error": "",
    }
