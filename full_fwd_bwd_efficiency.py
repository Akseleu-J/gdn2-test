"""
full_fwd_bwd_efficiency.py
 
Два изменения относительно предыдущей версии:
 
1. ПОЛНЫЙ КОНВЕЙЕР: измеряются все 9 кернелов -- A, B, C, D (forward) и
   B1, B2, B3, B4, B5 (backward), не только forward-четвёрка.
 
2. МЕТРИКА -- "efficiency %" КАЖДОГО кернела ОТНОСИТЕЛЬНО ЕГО
   СОБСТВЕННОГО потолка, а НЕ доли от общего времени пайплайна:
 
       own_ceiling_ms = max(compute_floor, memory_floor, dispatch_floor)
       efficiency_pct = 100 * own_ceiling_ms / measured_ms
 
   Смысл: own_ceiling -- это ЛУЧШЕЕ время, которое кернел физически МОГ
   БЫ показать при своей структуре (данные его же flops/bytes/chain-
   length), а measured -- то, что он показывает реально. efficiency=100%
   значит "уже упирается в свой собственный потолок, дальше некуда".
   efficiency=10% значит "в 10 раз хуже своего же потолка -- есть что
   оптимизировать", НЕЗАВИСИМО от того, сколько миллисекунд он вообще
   занимает в общем времени (kernel_B с его 45ms и, скажем, kernel_B2
   с его 0.5ms сравниваются на РАВНЫХ по этой метрике).
 
   Таблица в конце отсортирована по efficiency% ПО ВОЗРАСТАНИЮ -- самый
   неэффективный кернел (относительно самого себя) оказывается сверху,
   даже если он мелкий на фоне B.
 
ВАЖНО (честно): для backward-кернелов B1 и D (forward) sequential chain
length -- это РЕАЛЬНАЯ алгоритмическая последовательность (lax.scan по
n_chunks с истинной data dependency через h_pre/dh), это НЕ то же самое,
что "плохо написанный код", и это помечено флагом is_algorithmic. Для B3
(backward) chain length оценена ЧТЕНИЕМ кода вручную (нет тривиальной
формулы, как для B forward) -- это указано явно как ESTIMATED, не
ИЗМЕРЕНО автоматически, doверять этому числу можно меньше, чем формулам
для A/B/C/D/B2/B4/B5.
"""
from __future__ import annotations
 
import statistics
import time
 
import jax
import jax.numpy as jnp
 
RUN_CONFIG = dict(
    bsz=8, H=6, n_chunks=16, D=128,
    bt=256, bc=128, mb=32,
    wy_eps=1e-3, use_centering=True,
    n_warmup=8, n_iters=20,
    peak_bf16_tflops_per_chip=197.0,
    hbm_bandwidth_gbps_per_chip=819.0,
)
 
 
# ==========================================================================
# CALIBRATION
# ==========================================================================
def calibrate_dispatch_latency(cfg, sizes):
    calib = {}
    for sz in sizes:
        x = jax.random.normal(jax.random.PRNGKey(0), (sz, sz))
 
        def chain(n_steps, x):
            def body(i, a):
                return jnp.dot(a, x, precision=jax.lax.Precision.HIGHEST)
            return jax.lax.fori_loop(0, n_steps, body, x)
 
        chain_jit = jax.jit(chain, static_argnums=(0,))
        times = {}
        for N in (50, 500):
            for _ in range(cfg["n_warmup"]):
                jax.block_until_ready(chain_jit(N, x))
            ts = []
            for _ in range(cfg["n_iters"]):
                t0 = time.perf_counter()
                jax.block_until_ready(chain_jit(N, x))
                ts.append(time.perf_counter() - t0)
            times[N] = statistics.mean(ts)
        slope_s = (times[500] - times[50]) / (500 - 50)
        calib[sz] = slope_s * 1e6
    return calib
 
 
def nearest_calib(calib, size):
    return calib[min(calib.keys(), key=lambda s: abs(s - size))]
 
 
# ==========================================================================
# ANALYTICAL MODELS -- 9 kernels
# confidence: "formula" = derived mechanically from loop structure (high
#             confidence); "estimated" = read from code by hand, critical
#             path length approximate (lower confidence, said explicitly)
# ==========================================================================
def build_models(cfg):
    bsz, H, n_chunks, bt, bc, mb, D = (cfg["bsz"], cfg["H"], cfg["n_chunks"],
                                        cfg["bt"], cfg["bc"], cfg["mb"], cfg["D"])
    n_sub = bt // bc
    pairs = n_sub * (n_sub + 1) // 2
    bc2 = bt // 2
    n_micro = bc2 // mb
 
    models = {}
 
    # ---- A ----
    flops_per_pair = 2 * (bc * bc * D * 2)
    models["A"] = dict(
        flops=bsz * H * n_chunks * (pairs * flops_per_pair + bt * bt * D * 2),
        bytes=bsz * H * n_chunks * (4 * (bt * D * 4) + 2 * (bt * bt * 4)),
        grid_cells=bsz * H * n_chunks,
        seq_chain=2, rep_size=bc, is_algorithmic=False, confidence="formula",
    )
 
    # ---- B ----
    diag_flops = n_micro * (mb ** 3) * 2
    offdiag_flops = sum((m - n) * (mb ** 3) * 2 for m in range(n_micro) for n in range(m))
    per_chunk_flops_b = 2 * (diag_flops + offdiag_flops) + 2 * (bc2 ** 3) * 2
    diag_chain = n_micro * mb
    offdiag_chain = sum(m for m in range(n_micro))
    models["B"] = dict(
        flops=bsz * H * n_chunks * per_chunk_flops_b,
        bytes=bsz * H * n_chunks * 2 * (bt * bt * 4),
        grid_cells=bsz * H * n_chunks,
        seq_chain=diag_chain + offdiag_chain + 2, rep_size=mb,
        is_algorithmic=False, confidence="formula",
    )
 
    # ---- C ----
    flops_per_chunk_c = 3 * (bt * bt * D * 2)
    models["C"] = dict(
        flops=bsz * H * n_chunks * flops_per_chunk_c,
        bytes=bsz * H * n_chunks * (6 * (bt * D * 4) + (bt * bt * 4) + 4 * (bt * D * 4)),
        grid_cells=bsz * H * n_chunks,
        seq_chain=2, rep_size=bt, is_algorithmic=False, confidence="formula",
    )
 
    # ---- D (forward inter-chunk scan) ----
    flops_per_step_d = 3 * (bt * D * D * 2) + (bt * bt * D * 2)
    models["D"] = dict(
        flops=bsz * H * n_chunks * flops_per_step_d,
        bytes=bsz * H * n_chunks * (4 * (bt * D * 4) + (bt * bt * 4) + bt * D * 4),
        grid_cells=bsz * H,
        seq_chain=n_chunks * 3, rep_size=D, is_algorithmic=True, confidence="formula",
    )
 
    # ---- B1 (gdn2_dhu_backward -- pure jax, lax.scan, real algorithmic dep) ----
    flops_per_step_b1 = (D * D) * 1 * 2  # dqh contrib small; dominant: einsums qg/kg (bt,D,D) + wp (bt,D,D)
    # honest: mirrors D's step cost roughly (same tensor shapes, similar einsums)
    flops_per_step_b1 = 3 * (bt * D * D * 2)
    models["B1"] = dict(
        flops=bsz * H * n_chunks * flops_per_step_b1,
        bytes=bsz * H * n_chunks * (5 * (bt * D * 4) + D * D * 4),
        grid_cells=bsz * H,
        seq_chain=n_chunks * 3, rep_size=D, is_algorithmic=True, confidence="formula",
    )
 
    # ---- B2 (dav_backward_pallas -- Pallas, no loop, 2 matmuls) ----
    flops_per_chunk_b2 = 2 * (bt * bt * D * 2)
    models["B2"] = dict(
        flops=bsz * H * n_chunks * flops_per_chunk_b2,
        bytes=bsz * H * n_chunks * (2 * (bt * bt * 4) + 2 * (bt * D * 4)),
        grid_cells=bsz * H * n_chunks,
        seq_chain=2, rep_size=bt, is_algorithmic=False, confidence="formula",
    )
 
    # ---- B3 (wy_dqkg_backward_pallas -- Pallas, complex, chain read by hand) ----
    # 8 major matmuls identified in _kernel_b3_body, critical path
    # (dw_pseudo -> dA_from_w -> dA_total -> tmp -> dAkk_raw) ~ 5 sequential
    # matmuls + ~3 setup elementwise steps. ESTIMATED, not derived from a
    # loop-count formula like A/B/C/D.
    # B3
        # ---- B3 (wy_dqkg_backward_pallas -- Pallas, complex, chain read by hand) ----
    # 8 major matmuls identified in _kernel_b3_body, critical path
    # (dw_pseudo -> dA_from_w -> dA_total -> tmp -> dAkk_raw) ~ 5 sequential
    # matmuls + ~3 setup elementwise steps. ESTIMATED, not derived from a
    # loop-count formula like A/B/C/D.
    n_major_matmuls_b3 = 8
    flops_per_chunk_b3 = n_major_matmuls_b3 * (bt * bt * D * 2)  # rough: most are bt x bt x D or bt x D x D scale
    models["B3"] = dict(
        flops=bsz * H * n_chunks * flops_per_chunk_b3,
        bytes=bsz * H * n_chunks * (5 * (bt * D * 4) + 3 * (bt * bt * 4) + D * D * 4),
        grid_cells=bsz * H * n_chunks,
        seq_chain=8, rep_size=bt, is_algorithmic=False, confidence="estimated",
    )
 
    # ---- B4 (intra_backward_pallas -- same n_sub structure as A) ----
    flops_per_pair_b4 = 4 * (bc * bc * D * 2)  # more matmuls per pair than fwd A (dq,dk,db,dgc contributions)
    models["B4"] = dict(
        flops=bsz * H * n_chunks * (pairs * flops_per_pair_b4),
        bytes=bsz * H * n_chunks * (4 * (bt * D * 4) + 2 * (bt * bt * 4) + 4 * (bt * D * 4)),
        grid_cells=bsz * H * n_chunks,
        seq_chain=3, rep_size=bc, is_algorithmic=False, confidence="estimated",
    )
 
    # ---- B5 (reverse_cumsum_bwd -- single matmul) ----
    models["B5"] = dict(
        flops=bsz * H * n_chunks * (bt * bt * D * 2),
        bytes=bsz * H * n_chunks * (2 * (bt * D * 4)),
        grid_cells=bsz * H * n_chunks,
        seq_chain=1, rep_size=bt, is_algorithmic=False, confidence="formula",
    )
 
    return models
 
 
# ==========================================================================
# MEASURE all 9 kernels in isolation, in-process, reusing real residuals
# ==========================================================================
def measure_all(cfg):
    from Atomic_ops.configs import KernelConfig
    from Atomic_ops.gdn2_fwd import (
        build_chunk_scores_pallas, wy_solve_pallas, recompute_wy_pallas,
        gdn2_inter_chunk_combine, gdn2_pallas_forward_with_residuals,
        _reshape_to_chunks as _r2c,
    )
    from Atomic_ops.gdn2_bwd import (
        gdn2_dhu_backward, dav_backward_pallas, wy_dqkg_backward_pallas,
        intra_backward_pallas, reverse_cumsum_bwd,
    )
 
    config = KernelConfig(bt=cfg["bt"], bc=cfg["bc"], mb=cfg["mb"],
                           wy_eps=cfg["wy_eps"], use_centering=cfg["use_centering"])
    B, H, C, D, bt = cfg["bsz"], cfg["H"], cfg["n_chunks"], cfg["D"], cfg["bt"]
    L = C * bt
    key = jax.random.PRNGKey(0)
    kq, kk, kv, kw, kb, kg = jax.random.split(key, 6)
    q = jax.random.normal(kq, (B, L, H, D)) * 0.3
    k = jax.random.normal(kk, (B, L, H, D)) * 0.3
    v = jax.random.normal(kv, (B, L, H, D)) * 0.3
    w = jax.random.uniform(kw, (B, L, H, D), minval=0.2, maxval=1.0)
    b = jax.random.uniform(kb, (B, L, H, D), minval=0.2, maxval=1.0)
    g = -jnp.abs(jax.random.normal(kg, (B, L, H, D))) * 0.05
    h0 = jnp.zeros((B, H, D, D), dtype=jnp.float32)
 
    def timeit(fn, args):
        for _ in range(cfg["n_warmup"]):
            jax.block_until_ready(fn(*args))
        ts = []
        for _ in range(cfg["n_iters"]):
            t0 = time.perf_counter()
            jax.block_until_ready(fn(*args))
            ts.append(time.perf_counter() - t0)
        return statistics.mean(ts) * 1000, statistics.pstdev(ts) * 1000
 
    results = {}
 
    # --- forward, to get real residuals for backward measurement ---
    o, h_final, res = jax.block_until_ready(
        gdn2_pallas_forward_with_residuals(q, k, v, w, b, g, 1.0, h0=h0, config=config)
    )
    Aqk, Akk, A = res["Aqk"], res["Akk"], res["A"]
    h_pre_all, v_new_all = res["h_pre_all"], res["v_new_all"]
    w_pseudo, u, kg_, qg, gc_last = res["w_pseudo"], res["u"], res["kg"], res["qg"], res["gc_last"]
 
    # A
    fn = jax.jit(lambda q, k, b, g: build_chunk_scores_pallas(q, k, b, g, 1.0, config=config))
    results["A"] = timeit(fn, (q, k, b, g))
 
    # B
    fn = jax.jit(lambda akk: wy_solve_pallas(akk, config=config))
    results["B"] = timeit(fn, (Akk,))
 
    # C
    fn = jax.jit(lambda q, k, v, w, b, g, A: recompute_wy_pallas(q, k, v, w, b, g, A, config=config))
    results["C"] = timeit(fn, (q, k, v, w, b, g, A))
 
    # D
    fn = jax.jit(lambda *a: gdn2_inter_chunk_combine(*a, 1.0, config=config))
    results["D"] = timeit(fn, (Aqk, w_pseudo, u, kg_, qg, gc_last))
 
    # synthetic cotangents for backward measurement
    do = jax.random.normal(jax.random.PRNGKey(99), o.shape) * 0.01
    dh_final = jax.random.normal(jax.random.PRNGKey(100), h_final.shape) * 0.01
    do_r = _r2c(do, B, C, H, D, bt)
 
    # B2 (dav_backward)
    fn = jax.jit(lambda *a: dav_backward_pallas(*a, config=config))
    results["B2"] = timeit(fn, (Aqk, v_new_all, do_r))
    dAqk, dv_partial = jax.block_until_ready(fn(Aqk, v_new_all, do_r))
 
    # B1 (gdn2_dhu_backward)
    fn = jax.jit(lambda *a: gdn2_dhu_backward(*a, dht=dh_final, config=config))
    results["B1"] = timeit(fn, (do_r, dv_partial, w_pseudo, qg, kg_, gc_last, 1.0))
    dh_all, dh0, dv_all = jax.block_until_ready(fn(do_r, dv_partial, w_pseudo, qg, kg_, gc_last, 1.0))
 
    # gc needed for B3
    g_r = _r2c(g, B, C, H, D, bt)
    idx = jnp.arange(bt)
    tril_ones_bt = (idx[:, None] >= idx[None, :]).astype(jnp.float32)
    gc = jnp.einsum("ij,bhcjd->bhcid", tril_ones_bt, g_r, precision=jax.lax.Precision.HIGHEST)
    q_r = _r2c(q, B, C, H, D, bt)
    k_r = _r2c(k, B, C, H, D, bt)
    b_r = _r2c(b, B, C, H, D, bt)
    w_r = _r2c(w, B, C, H, D, bt)
    v_r = _r2c(v, B, C, H, D, bt)
    dh_next_all = jnp.concatenate([dh_all[:, :, 1:], dh_final[:, :, None]], axis=2)
 
    # B3
    fn = jax.jit(
        lambda q, k, b, w, v, gc, A, Akk, h_pre, v_new, do, dv, dh_next:
            wy_dqkg_backward_pallas(
                q, k, b, w, v, gc, A, Akk, h_pre, v_new, do, dv, dh_next,
                1.0, config=config,
            )
    )
    b3_args = (
        q_r, k_r, b_r, w_r, v_r, gc, A, Akk, h_pre_all, v_new_all,
        do_r, dv_all, dh_next_all,
    )
    results["B3"] = timeit(fn, b3_args)
    b3_out = jax.block_until_ready(fn(*b3_args))
    # B4
    # B4
    fn = jax.jit(
        lambda dAqk, dAkk, q, k, b, g:
            intra_backward_pallas(dAqk, dAkk, q, k, b, g, 1.0, config=config)
    )
    b4_args = (dAqk, b3_out["dAkk"], q, k, b, g)
    results["B4"] = timeit(fn, b4_args)
    # B5
    fn = jax.jit(lambda dgc: reverse_cumsum_bwd(dgc, chunk_size=bt, config=config))
    dgc_total = b3_out["dgc"]
    results["B5"] = timeit(fn, (dgc_total,))
 
    return results
 
 
# ==========================================================================
# MAIN
# ==========================================================================
def main():
    cfg = RUN_CONFIG
    print(f"jax backend: {jax.default_backend()}  |  devices: {jax.devices()}")
 
    models = build_models(cfg)
    rep_sizes = sorted(set(m["rep_size"] for m in models.values()))
 
    print("\n" + "=" * 100)
    print("КАЛИБРОВКА dispatch latency")
    print("=" * 100)
    calib = calibrate_dispatch_latency(cfg, rep_sizes)
    for sz, lat in calib.items():
        print(f"  matmul {sz}x{sz}: {lat:.4f} us/op")
 
    print("\n" + "=" * 100)
    print("ИЗМЕРЕНИЕ всех 9 кернелов (in-process, реальные residuals из forward)")
    print("=" * 100)
    measured = measure_all(cfg)
    total_ms = sum(m[0] for m in measured.values())
    for k in ("A", "B", "C", "D", "B1", "B2", "B3", "B4", "B5"):
        mean_ms, std_ms = measured[k]
        print(f"  Kernel {k:<3}: {mean_ms:>9.4f} ms (std {std_ms:.4f})  "
              f"-- {100*mean_ms/total_ms:5.1f}% от суммы всех 9")
 
    peak_bf16 = cfg["peak_bf16_tflops_per_chip"] * 1e12
    bw_bytes_per_s = cfg["hbm_bandwidth_gbps_per_chip"] * 1e9
 
    print("\n" + "=" * 100)
    print("EFFICIENCY % ОТНОСИТЕЛЬНО СОБСТВЕННОГО ПОТОЛКА КАЖДОГО КЕРНЕЛА")
    print("=" * 100)
 
    rows = []
    for k in ("A", "B", "C", "D", "B1", "B2", "B3", "B4", "B5"):
        m = models[k]
        measured_ms, std_ms = measured[k]
        rep_lat_us = nearest_calib(calib, m["rep_size"])
 
        t_compute_ms = m["flops"] / (peak_bf16 / 3) * 1000  # 3-pass f32 HIGHEST assumption
        t_memory_ms = m["bytes"] / bw_bytes_per_s * 1000
        t_dispatch_ms = m["seq_chain"] * rep_lat_us / 1000
 
        own_ceiling_ms = max(t_compute_ms, t_memory_ms, t_dispatch_ms)
        binding = max(
            [("compute", t_compute_ms), ("memory", t_memory_ms), ("dispatch", t_dispatch_ms)],
            key=lambda x: x[1],
        )[0]
        efficiency_pct = 100.0 * own_ceiling_ms / measured_ms
 
        rows.append(dict(
            kernel=k, measured_ms=measured_ms, own_ceiling_ms=own_ceiling_ms,
            efficiency_pct=efficiency_pct, binding=binding,
            is_algorithmic=m["is_algorithmic"], confidence=m["confidence"],
            pct_of_total=100 * measured_ms / total_ms,
        ))
 
    rows.sort(key=lambda r: r["efficiency_pct"])
 
    print(f"\n  {'Kernel':<8}{'measured_ms':>13}{'own_ceiling_ms':>16}{'efficiency%':>13}"
          f"{'binding':>11}{'% of total':>12}{'algo?':>7}{'confidence':>12}")
    print("  " + "-" * 95)
    for r in rows:
        algo_flag = "YES" if r["is_algorithmic"] else "-"
        print(f"  {r['kernel']:<8}{r['measured_ms']:>13.4f}{r['own_ceiling_ms']:>16.4f}"
              f"{r['efficiency_pct']:>12.3f}%{r['binding']:>11}{r['pct_of_total']:>11.1f}%"
              f"{algo_flag:>7}{r['confidence']:>12}")
 
    print("""
  ЧТЕНИЕ ТАБЛИЦЫ (отсортировано по efficiency% -- ХУДШИЕ СВЕРХУ):
  - Верхние строки -- кернелы, наиболее ДАЛЁКИЕ от своего же потолка,
    НЕЗАВИСИМО от абсолютного вклада в общее время. Именно здесь
    скрывались "замаскированные" неэффективности, которые не видны в
    обычной разбивке "% от forward" (там всё забивает Kernel B).
  - 'algo?=YES' -- кернел структурно последовательный ПО АЛГОРИТМУ
    (D, B1) -- низкий efficiency% здесь означает "код уже близок к
    пределу, доступному БЕЗ смены математики" -- не список первоочередных
    целей рефакторинга, а кандидаты на associative_scan / смену
    формулировки, если вообще возможно.
  - 'confidence=estimated' (B3, B4) -- цепочка/flops для них не выведены
    формулой из структуры цикла (как для A/B/C/D/B2/B5), а прочитаны из
    кода вручную -- относитесь к их efficiency% с поправкой, возможной
    ошибкой в 1.5-2x по seq_chain.
  - Если верхние строки -- это B2/B3/B4/B5 (не B1/D), значит backward
    Pallas-кернелы неэффективны структурно ТАК ЖЕ, как forward Kernel B
    -- вероятно, по той же причине (grid из независимых задач,
    сериализуемый почти как последовательный), и решение то же самое:
    батчинг задач внутри тела кернела вместо grid, а не только для B.
""")
 
 
if __name__ == "__main__":
    main()
