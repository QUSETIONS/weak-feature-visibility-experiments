# E4 TopK Held-Out Confirmation Protocol

Status: frozen before execution; no result has been generated.

This protocol tests a narrow question left open by E4-v2: whether the fixed-basis independence split is accompanied by an axis-specific recovery penalty when TopK capacity and sparsity change. It is an independent confirmation, not a public registration and not an all-objective SAE claim.

## Fixed design

- Centered Bernoulli feature in m=8, p=0.05, and lambda_eff=0.55.
- Eight held-out seeds starting at 20260901; detector N=4096, 400 alternative trials, 800 null draws, and alpha=0.05.
- TopK configurations: (d=8,k=1), (d=16,k=2), and (d=32,k=4).
- Training: 20,000 samples, 4,000 Adam steps, batch 512, learning rate 1e-3, and decoder-cosine recovery threshold 0.80.

## Decision gates

The detector split must remain present. Every TopK configuration must also satisfy: axis and dense recovery rates are each at least 0.75; their absolute recovery-rate difference is at most 0.25; and axis mean decoder cosine is no more than 0.05 below dense mean cosine.

A failure is informative and must not be folded into E4-v2. A pass supports only this three-configuration TopK grid, not other SAE objectives, widths outside the grid, or unknown-direction discovery.

## Outputs

run_e4_topk_confirmation.py writes raw rows and a summary with protocol/runner SHA256 hashes. It atomically updates raw output after each completed unit and supports explicit --resume after interruption.
