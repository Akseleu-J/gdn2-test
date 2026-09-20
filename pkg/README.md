# atomic_gdn2 — GDN-2 / BLR Pallas kernels (TPU v5e)

Attested path: bt=128, score_bs=128, bs2=32, mb=8, diag=btl, B3=bf16x3, B4/B5=highest.
`make_cfg()` ties `g_max = 160/bs2` so BTL half_span < 88 by construction.

```
atomic_gdn2/
  config.py precision.py layout.py domain.py      config, matmul modes, layout, guards/clamps
  fwd/   scores.py (A: BLR 2-level, BTL|lean)  solve.py (B: ladder, leaf-batched)  cd_slim.py (C+D fused)
  bwd/   b4.py (no clamp masks: O-1 fix)  mega.py (B2+B1+B3+B4+B5, one launch)
  pipeline.py    custom_vjp, segmentation of long L via h0/h_final
  layer.py       g = -min(softplus(.+bias), g_max), sigmoid beta, optional L2-norm q/k
  reference/     forward_ref (autodiff), token_serial_ref (ground truth), f64 B4
  training/      model, data, train (skip-step, no-wd on bias/embed, domain stats, canary, ckpt)
tests/ tools/
```
Run: `tools/run_all.sh` (CPU, interpret) — 30 tests, ~2 min. `-m slow` adds tiny training smoke.
TPU: `python tools/bench.py` (T8), `python tools/t0_diagnose.py --branch base` (T0).

## Test map
T0 tools/t0_diagnose.py (TPU) | T1 test_t1_finite_diff | T2 test_t2_domain | T3 test_t3_canary
T4 test_t4_phantom_params | T5/T7 test_t5_grad_vs_reference | T6 test_t6_train_smoke + full run on TPU
T8 tools/bench.py | T9 test_t9_package | segmentation test_segmentation

## Not verified here
Everything ran on CPU (`interpret=True`). Mosaic lowering / timing on TPU is NOT confirmed for this package:
the code is a consolidation of notebook code that did run on TPU, with these deliberate changes —
kernels take `cfg` for all modes (b4_dot_mode is now really used), the C+D and backward kernels use
grid=(bsz,H) with heads_per_cell=1 (as in the attested runs), long L is segmented instead of raising vmem.
First TPU step: run tools/bench.py and tests with JAX_PLATFORMS=tpu before trusting timings.
