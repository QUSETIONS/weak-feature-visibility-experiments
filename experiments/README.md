# Experiments

Reproducible core for *Why Weak Features Hide: The Detector Axis Hole Is Not a Free-Decoder Constraint*.

| Script | What it writes |
|---|---|
| `run_e1_e2_synthetic.py` | same-λ pair, rotation of \(B\) |
| `run_v2.py` | 2-sparse imbalance; pre-specified detector/TopK control at \(\lambda_{\mathrm{eff}}\in\{0.35,0.55,0.80\}\) |
| `run_extra_synthetic.py` | \(N^\star(\lambda)\) slopes and recover-or-miss floor |
| `run_sae_wall.py` | TopK cosine vs \(\lambda\) on axis and dense |
| `run_gpt2_bridge.py` | GPT-2-small L6 WikiText residual, PCA-whitened injections |
| `run_sae_nstar.py` | fixed-step TopK \(N^\star(\lambda)\) (not headlined; small \(N\) over-trained) |
| `run_sae_exponent.py` | constant-epoch labeled / oracle / free TopK; wide grid is `--m 32` with \(N\le 131\mathrm{k}\) |
| `run_sae_capacity.py` | missed mass vs dictionary size (\(k=4\), \(N=16384\)) |
| `run_gpt2_natural_sae.py` | uninjected GPT-2 SAE atom geometry |
| `run_sae_floor.py` | trained TopK missed-mass vs ind vs quartic predictors |
| `run_e4_topk_confirmation.py` | independent held-out capacity/sparsity grid; original all-pass gate fails at d32-k4 |
| `run_e4_topk_budget_followup.py` | post-hoc same-trajectory d32-k4 checkpoints at 4k/8k/16k steps |
| `run_reconstruction_certificate.py` | objective-agnostic held-out reconstruction certificate (analytic + synthetic validation) |
| `run_reconstruction_certificate_stress.py` | corrected v3 objective/capacity stress rerun; 48 TopK training rows with held-out certificates |
| `run_spectral_anchor_sae.py` | target-free spectral-anchor SAE; residual objectives preserve a signed spectral anchor |
| `run_bounded_drift_spectral_anchor_sae.py` | relaxed v4 target-free spectral anchor; trainable anchor in a trust region with bounded residual leakage |
| `run_spectral_warm_start_matched.py` | paired vanilla/warm-start synthetic matrix; also executes the independent v9 fresh-seed protocol |
| `run_lm_spectral_warm_start_matched.py` | paired vanilla/warm-start matrix on GPT-2-small L6 and Pythia-70m L3 residual backgrounds |
| `lm_spectral_warm_start_scale_extension_budget_protocol.json` | frozen 6k-step Pythia-160M budget follow-up protocol |

Frozen constants: seed base `20260824`, SAE recovery if decoder cosine \(\ge 0.80\).

```
python run_e1_e2_synthetic.py
python run_v2.py --device cuda
python run_extra_synthetic.py
python run_sae_wall.py --out results_wall --device cuda
python run_gpt2_bridge.py --out results_gpt2 --device cuda
python run_spectral_warm_start_matched.py --protocol spectral_warm_start_fresh_replication_protocol.json --out results_spectral_warm_start_fresh_replication --device cuda
python run_lm_spectral_warm_start_matched.py --protocol lm_spectral_warm_start_matched_protocol.json --model pythia-70m-l3 --model-dir weights/pythia-70m --text-file /path/to/wiki.train.raw --device cuda
```

Paper figures are built from these JSON files by `paper/make_figures.py`.
The supplied public-atom diagnostic is stored as `paper/figures/fig_public_atoms_supplied.pdf` and `.png`; the supplied residual-floor figure is stored as `paper/figures/fig_sae_floor_supplied.pdf` and `.png`.

Completed E4-v2 evidence: at lambda_eff=0.55, axis/dense independence power is 0.081/1.000 and both TopK geometries meet the frozen cosine recovery criterion in all eight seeds; the complete three-strength raw result has 48 rows, with 19/24 recovered in each detector class. The source protocol is `../ICLR_experiment_design_v2.md`; summary and raw rows are `results_v2/e4v2_summary.json` and `results_v2/e4v2_raw.json`.

`e4_topk_confirmation_protocol.json` and `run_e4_topk_confirmation.py` define and ran an independent held-out confirmation across `(d,k)=(8,1),(16,2),(32,4)` and eight new seeds starting at 20260901. The complete `results_e4_topk_confirmation/` artifact has 16 detector and 48 training rows; its all-pass gate is false because `d32-k4` dense recovery is 0.625, below the frozen 0.75 minimum. `e4_topk_budget_followup_protocol.json` and `run_e4_topk_budget_followup.py` then run a transparent post-hoc same-trajectory d32-k4 budget trace: recovery at 4k/8k/16k is axis `0.750/0.875/1.000` and dense `0.625/0.750/0.875`, with its post-hoc gate passing at 16k. This supports a budget explanation but does not revise the original held-out decision; the paper's main claim remains the completed E4-v2 matched control. `run_reconstruction_certificate.py` implements the separate objective-agnostic certificate: for any frozen decoder subspace produced by any objective, an independent held-out reconstruction excess-risk bound yields a target-alignment lower bound under the stated isotropic rank-one covariance model. Its two-case artifact is analytic plus synthetic validation only, explicitly marked `evidence_eligible=false`; it is not a multi-objective training sweep or an optimization guarantee. The corrected v3 stress protocol (`reconstruction_certificate_stress_protocol.json`) reruns the frozen 48-cell matrix under the rank-2 projector-difference concentration bound: 48/48 rows are complete, certificate pass rate is 19/48=0.3958, direct recovery is 0.4792, and `evidence_eligible=false`. The v3 rerun supersedes the earlier v2 summary, which used the pre-correction finite-sample bound; it does not support universal training recovery.

Bounded-drift v4 evidence: `bounded_drift_spectral_anchor_sae_protocol.json` and `run_bounded_drift_spectral_anchor_sae.py` relax v3 by making the training-only spectral anchor trainable inside a `rho=0.20` radian trust region and permitting residual anchor leakage up to `eta=0.10`. The complete artifact in `results_bounded_drift_spectral_anchor_sae/` has 48 rows across three residual objectives, two capacities, two geometries, and four seeds. All rows pass: recovery `48/48`, minimum final cosine `0.989891`, maximum drift `0.128657` radians, maximum leakage `0.1000000015`, and certified cosine lower bound `0.854304`; `evidence_eligible=true`. This is a relaxed but still explicitly bounded anchor-preserving family, not an unconstrained-SAE theorem. The separate `spectral_anchor_sae_protocol.json` freezes a target-free signed spectral anchor, one active anchor slot, residual decoder orthogonality, a family-wise Gaussian-Wishart/Davis--Kahan gate, three residual objectives, two total capacities, two geometries, and four seeds. Its completed `results_spectral_anchor_sae/` artifact has 48/48 rows, anchor recovery rate 1.0, observed minimum cosine 0.9989, and family-wise certified cosine lower bound 0.9405; `evidence_eligible=true`. This supports the theorem for the anchor-preserving family, not unconstrained SAE training.

Current follow-up protocols are frozen separately. `spectral_warm_start_fresh_replication_protocol.json` repeats the full v6 objective-by-capacity-by-geometry matrix with eight fresh base seeds and analyzes the paired effect at the base-seed cluster level rather than treating 96 condition rows as independent replicates. `lm_spectral_warm_start_matched_protocol.json` tests the same initialization-only intervention on GPT-2-small L6 and Pythia-70m L3, with 48 pairs per model. It requires real WikiText, disjoint train/validation sequence indices, exact pair integrity, and no synthetic text fallback. Because empirical whitened residuals do not satisfy the stated rank-one Gaussian model, this LM protocol reports held-out reconstruction and cosine audits but explicitly forbids claiming the Gaussian certificate.

The independent model-scale extension is frozen in `lm_spectral_warm_start_scale_extension_protocol.json` and completed in `results_lm_spectral_warm_start_scale_extension/pythia-160m-l6/`. It repeats the 48-pair real-residual intervention on Pythia-160M layer 6 (hidden size 768). All 96 rows pass the provenance, finiteness, and pair-integrity gates. Recovery is `28/48` versus `33/48`, the base-seed clustered cosine gain is `0.0339` (bootstrap 95% CI `[0.0176, 0.0491]`), and held-out MSE increases by `0.000665` (CI `[0.000389, 0.000980]`). This is a model-scale boundary result with a reconstruction-cost trade-off, not a universal improvement claim or a Gaussian certificate.

The separately frozen `lm_spectral_warm_start_scale_extension_budget_protocol.json` tests the same Pythia-160M matrix at `6000` steps using the same residual cache, sampled indices, seeds, and validation samples. Its complete `results_lm_spectral_warm_start_scale_extension_budget/` artifact contains 48 pairs; vanilla recovery is `32/48` and warm-start recovery is `31/48`, with clustered cosine difference `-0.013` (95% CI `[-0.032, 0.009]`) and held-out MSE difference `+0.000258` (95% CI `[-0.000413, 0.001002]`). The follow-up removes the detectable 3k MSE gap but also removes the recovery advantage, so the intervention is reported as budget-sensitive rather than universally beneficial.
