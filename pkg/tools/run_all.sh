#!/usr/bin/env bash
# CPU (interpret) suite: T1,T2,T3,T4,T5,T7,T9 (+T6 tiny with -m slow). On TPU also run tools/bench.py (T8) and t0_diagnose.py (T0).
set -e; cd "$(dirname "$0")/.."
python -m pytest tests -q "$@"
