Not part of the attested path (do not import from production code):
- fused A+B (`build_and_solve_lean`): 1 launch less, gave no attested gain
- reverse-grid backward (scratch persistence probed OK on TPU; blockspec rewrite pending:
  grid=(nc,bsz,H), scratch=(bsz,H,D,D), gc_last must be a 5D-compatible block)
- bf16 storage of Aqk / v_new (needs real-checkpoint gate)
