# ICLR SAE Recovery Handoff

**Last updated:** 2026-09-02
**Workspace:** `LOCAL_WORKSPACE_PATH/Desktop/iclr/old`
**Paper:** *Why Weak Features Hide: The Detector Axis Hole Is Not a Free-Decoder Constraint*
**Purpose:** handoff for continuing the manuscript, experiments, audit, or submission packaging.

## Executive Summary

The project now has two defensible structured recovery results, but both quantifiers are intentionally restricted:

> **All objectives and capacities in the explicitly defined anchor-preserving or bounded-drift anchor-preserving SAE family inherit recovery of a target-free spectral anchor, under the stated finite-sample eigengap and constraint conditions.**

This is **not** a theorem that every unconstrained SAE objective, optimizer, or capacity recovers every unknown target direction. The unconstrained TopK stress test remains negative for that stronger claim: only `19/48` held-out certificates pass. Do not remove this limitation from the paper or rename the structured result as universal vanilla-SAE recovery.

The final submission PDF and audit package are built and verified after the bounded-drift v4 extension. No further GPU run is required for the current submission state.

## What Has Been Established

### 1. Detector-class axis hole

The paper separates a fixed-basis feature-independent detector from the model class of an unconstrained sparse decoder.

- For the diagonal/feature-independent Gaussian nuisance, the local coefficient is `1 - ||w||_4^4`.
- An axis-aligned feature can be invisible to that detector while remaining available to an unconstrained sparse decoder.
- This is a detector-class result, not a representation-level constraint on every SAE.
- The reconstruction rates are stated separately: linear reconstruction is quartic in feature strength; conditional isolation is quadratic once a separating direction is already available.
- The new proposition `Why a structural constraint is necessary` makes the boundary explicit: the unconstrained feasible decoder class alone gives no positive uniform alignment bound, while this does not rule out a data-dependent learner.

### 2. E4-v2 matched TopK control

Canonical result in `experiments/results_v2/`:

- At matched `lambda_eff=0.55`, independence power is axis `0.081` versus dense `1.000`.
- Both geometries meet the frozen cosine recovery criterion in all eight seeds at that strength.
- Across the full three-strength grid, axis and dense each recover `19/24` configurations.
- The result is a controlled non-implication: detector visibility does not automatically transfer to free-TopK recovery.

### 3. Independent held-out capacity/sparsity confirmation

Canonical result in `experiments/results_e4_topk_confirmation/`:

- Frozen grid: `(d,k)=(8,1),(16,2),(32,4)`, eight new seeds.
- The complete artifact contains 16 detector rows and 48 training rows.
- The pre-specified all-pass gate **fails** at `d32-k4` because dense recovery is `0.625 < 0.75`.
- A separate post-hoc same-trajectory budget trace reaches dense recovery `0.875` at 16k steps, supporting a training-budget explanation.
- The post-hoc trace does not revise the original failed gate and does not broaden the free-SAE claim.

### 4. Objective-agnostic reconstruction certificate, v3

Canonical result in `experiments/results_reconstruction_certificate_stress/`:

- The finite-sample bound was corrected for the difference of two rank-r projectors: the difference can have rank at most `2r`.
- The corrected v3 matrix is complete: `48/48` rows.
- Certificate pass rate: `19/48 = 0.3958`.
- Direct recovery rate: `0.4792`.
- This artifact is marked `evidence_eligible=false` because it tests objective/capacity dependence rather than universal training recovery.
- Earlier v2 stress output used the pre-correction bound and is superseded. It was moved outside the workspace and must not be used as canonical evidence.

The certificate theorem is a **post-training output certificate**: given a frozen decoder subspace, a valid independent held-out excess-risk estimate gives a target-alignment lower bound under the stated isotropic rank-one covariance model. It is not an optimization convergence theorem.

### 5. Target-free spectral-anchor SAE family, v3

Canonical protocol and outputs:

- Protocol: `experiments/spectral_anchor_sae_protocol.json`
- Runner: `experiments/run_spectral_anchor_sae.py`
- Tests: `experiments/tests/test_spectral_anchor_sae.py`
- Raw artifact: `experiments/results_spectral_anchor_sae/spectral_anchor_raw.json`
- Summary: `experiments/results_spectral_anchor_sae/spectral_anchor_summary.json`

Frozen design:

- Dimension `m=8`, signal strength `0.55`, rank-one Gaussian signal plus isotropic Gaussian noise.
- Three residual objectives: MSE, MSE+L1, MSE+L2.
- Two total capacities: `d16-k2` and `d32-k4`.
- Axis and dense geometry, four seeds, 100k independent validation samples per row.
- Total: `3 x 2 x 2 x 4 = 48` rows.
- Anchor is the leading eigenvector of the **training-only** second moment.
- Sign is fixed by a deterministic coordinate convention; target oracle and geometry are not passed to the estimator.
- A signed anchor pair uses one active TopK slot.
- Residual branch uses `k-1` active slots.
- Anchor columns are frozen; residual decoder columns are projected into the anchor orthogonal complement and renormalized after every optimizer step.
- Validation is never used for anchor or candidate selection.
- Family-wise confidence allocates `delta=0.05` over `G=8` seed/geometry anchors.

Results:

- Rows complete: `48/48`.
- Anchor recovery: `48/48 = 1.0`.
- Minimum observed target cosine: approximately `0.9989` (`0.998881` in the artifact audit).
- Theoretical family-wise covariance error bound: `0.1394568845`.
- Davis--Kahan sine bound: approximately `0.339689`.
- Theoretical cosine lower bound: `0.9405379003` (reported as `0.9405`).
- Maximum residual-anchor inner product: at most `1e-6`; observed audit value approximately `8.94e-8`.
- Summary flag: `evidence_eligible=true`.

The v3 proof relies on a frozen anchor; v4 relaxes this by allowing a trainable anchor to move within a spherical trust region and allowing bounded residual leakage. Recovery remains inherited across residual objectives and capacities because the trust-region and leakage projections are enforced after every optimizer update.

### 6. Relaxed bounded-drift spectral-anchor family, v4

This is the next scientific iteration beyond the exact-anchor v3 construction. It is deliberately weaker than v3 but still has an explicit data--target linkage.

Canonical files:

- Protocol: `experiments/bounded_drift_spectral_anchor_sae_protocol.json`
- Runner: `experiments/run_bounded_drift_spectral_anchor_sae.py`
- Tests: `experiments/tests/test_bounded_drift_spectral_anchor_sae.py`
- Raw artifact: `experiments/results_bounded_drift_spectral_anchor_sae/bounded_drift_spectral_anchor_raw.json`
- Summary: `experiments/results_bounded_drift_spectral_anchor_sae/bounded_drift_spectral_anchor_summary.json`

Relaxed design:

- The reference anchor is still estimated from the training-only leading eigenvector; target oracle and validation selection remain forbidden.
- The anchor is trainable, but after every optimizer step it is projected into a spherical cap of radius `rho=0.20` radians around the reference anchor.
- Residual decoder columns are unit norm with nonzero anchor leakage bounded by `eta=0.10`; exact orthogonality is no longer required.
- One signed anchor slot is reserved and the residual branch uses `k-1` active slots.
- The matrix remains `3 objectives x 2 capacities x 2 geometries x 4 seeds = 48` rows.

v4 results:

- Complete and gate-valid: `48/48` rows, `anchor_recovery_rate=1.0`.
- Minimum final target cosine: `0.989891`.
- Maximum anchor drift: `0.128657` radians, below the `0.20` trust-region cap.
- Maximum residual-anchor leakage: `0.1000000015`, within numerical tolerance of the `0.10` cap.
- Family-wise covariance error bound: `0.1394568845`; Davis--Kahan sine bound: approximately `0.339689`.
- Bounded-drift cosine lower bound: `0.854304`, above the frozen recovery threshold `0.80`.
- Artifact summary is marked `evidence_eligible=true`.

The v4 theorem uses the angular triangle inequality: final anchor error is at most the spectral estimation angle plus the allowed trust-region radius. It is universal only over the bounded-drift anchor-preserving family and does not establish recovery for unconstrained SAE training.

## Manuscript Changes

The current manuscript includes:

- The objective/capacity-uniform spectral-anchor theorem and proof.
- The exact anchor-preserving quantifier: “all anchor-preserving objectives and capacities”.
- Table 9 with the `48/48` result.
- Abstract, contributions, Discussion, and Limitations updates.
- Explicit separation between the structured spectral-anchor theorem and unconstrained SAE training.
- Added `Why a structural constraint is necessary` proposition and appendix proof: an unconstrained feasible decoder class alone has no positive uniform alignment bound; a data-dependent learner is not ruled out.
- Corrected v3 projector-difference concentration factor `2` in the certificate theorem/proof.
- Corrected Table 8 values and v3 semantics.
- Added the bounded-drift v4 theorem, proof, protocol, runner, artifact, and Table 10.
- Added the v5 dose-response paragraph and Table 11 (`tab:dose_response`): anchor recovery is mechanism-independent at this scale across nine mechanisms (`72/72` rows), the unconstrained control reaches maximum residual--anchor leakage `0.734` with the worst held-out loss, and the certified family is framed as provable leakage control and lower variance rather than raw recovery.

Important manuscript locations in `paper/main.tex`:

- The structural-constraint proposition: around lines 196--199.
- The exact spectral-anchor theorem: around lines 214--216.
- The bounded-drift theorem: around lines 218--221.
- Spectral-anchor proofs: around lines 460 onward, including the bounded-drift proof.
- Experimental protocol, Table 9, and Table 10: around lines 550--632.
- Dose-response paragraph and Table 11: around lines 614--637.
- Discussion and limitations: around lines 410--414, with the v5 dose-response boundary sentence inside the Limitations paragraph.

Current PDF layout:

- 21 pages.
- References start on page 14 (unchanged; the v5 addition is appendix-only plus one Limitations sentence).
- Exact spectral-anchor proof is on page 15.
- Held-out Table 6 is on page 16; experimental protocol is on page 17.
- Tables 7 and 8 are on page 18.
- Exact-anchor Table 9 is on page 19.
- Bounded-drift Table 10 is on page 20; the dose-response paragraph is also on page 20.
- Dose-response Table 11 is on page 21.
- The v4 bounded-drift theorem appears before the reconstruction-rate theorem; its proof is in the appendix before the existing reconstruction proof.

## Verification Evidence

Final focused verification passed:

- Python syntax check: `python3 -m py_compile experiments/*.py paper/make_figures.py`
- Existing focused contract suite: `40/40` tests passed, including the structural-constraint boundary guard.
- Bounded-drift v4 suite: `9/9` passed on the PyTorch A40 runtime; two tensor tests are skipped only in the local no-torch environment. Combined local suite: `49` tests run, `49` passed with `2` environment skips for unavailable local PyTorch.
- Bounded-drift v4 suite: `9/9` passed on the PyTorch A40 runtime; two tensor tests are skipped only in the local no-torch environment.
- PDF build at v4 close: 20 pages, theorem/Table 9/Table 10 text present.
- v5 dose-response suite: `13` tests, `10` passed with `3` no-torch environment skips; the artifact-gate test ran against the real `72/72` A40 artifact.
- Combined local suite after v5: `96` tests, all green (`11` environment skips).
- Current PDF build: `21` pages; dose-response paragraph (page 20) and Table 11 (page 21) text present; in-text `Table 11` references resolve.
- Benign rebuild note: `xdvipdfmx` emits `Object @table.N already defined` warnings on rerun builds; reproduced identically on a reverted pre-v5 baseline, so it is a tectonic rerun-pass artifact, not a v5 regression.
- Independent clean-directory build: page text identical to checked-in PDF.
- Independent clean-directory build: PyMuPDF raster pixels identical page by page.
- No overfull hbox or annotation-boundary warning.
- Literal stale scan passed.
- Remote spectral-anchor process on `remote GPU runtime` is stopped.
- Submission checksum passed.

The final clean build used:

`/tmp/iclr-bounded-drift-v4-clean-20260902-0730`

Recommended verification command from the workspace:

```bash
python3 -m py_compile experiments/*.py paper/make_figures.py
python3 -m unittest \\
  experiments.tests.test_e4_topk_confirmation \\
  experiments.tests.test_e4_topk_budget_followup \\
  experiments.tests.test_reconstruction_certificate \\
  experiments.tests.test_reconstruction_certificate_stress \\
  experiments.tests.test_spectral_anchor_sae \
  experiments.tests.test_bounded_drift_spectral_anchor_sae
```

Build command:

```bash
TECTONIC=LOCAL_WORKSPACE_PATH/Desktop/iclr/old/bin/tectonic make -B -C paper all
```

The build may report non-blocking underfull vboxes and duplicate hyperref object names. Treat overfull boxes, annotation-boundary warnings, missing references, or page-text/pixel mismatches as blocking and investigate them.

## Review-Consistency Fixes (2026-09-05)

Read-only adversarial re-review of the v5 manuscript (GLM line; Claude unavailable) returned 5/10 with a defect list. Nine factual defects were verified against `experiments/results_dose_response/dose_response_raw.json` and fixed in `paper/main.tex` (rollback: `paper/main.tex.bak-20260905-prereviewfix`):

- Removed leftover infrastructure sentence ("Do not redistribute remote-host credentials.") from Appendix B.
- Dose-response design sentence now states the actual 9-config layout (2 frozen + 5 hard + 1 soft + 1 unbounded = 9x8 = 72 rows); the old wording implied a crossed 4x2 hard grid.
- "caps leakage at <= 0.101" corrected: constrained arms at or below 0.102, with the soft arm's exact 0.10125 disclosed (hard/frozen arms sit at the nominal 0.10 cap).
- The lambda grid is now enumerated as the actual 7-point runner grid {0.30, 0.34, 0.38, 0.42, 0.48, 0.55, 0.65} (was written {0.30,...,0.65}).
- "stays above 0.9788" rephrased to "at or above" (exact minimum 0.978814).
- The monotonicity claim is scoped to the eta=0.10 hard series (the rho=0.20, eta=0 arm breaks global monotonicity).
- Added: the unbounded control's worst-group cosine (0.9901) beats four of the seven constrained arms; the loss-mean contrast is not significant at n=8 (Welch t ~ 1.5); leakage and variance are the robust differentiators (also noted in Limitations).
- Added: artifact metadata disclosure (evidence_eligible=false, anchor-group ablation only) and an explicit reconciliation of the confirm-grid vs dose-response recovery criteria.

Contracts: `experiments/tests/test_manuscript_review_consistency.py` (9 tests, red -> green). Full suite: 105 collected, 94 passed, 11 environment skips, 0 failures. PDF rebuilt: 21 pages, layout unchanged (main text ends p13, dose-response paragraph p20, Table 11 p21).

Pending strategic decisions (not yet executed): title/abstract reframing around calculus + certificate + boundary story; demoting the saeclass theorem to an observation; compressing main text from ~13 pages to the ICLR 2027 9-page submission limit (desk-reject threshold); optional GPT-2 known-direction anchor experiment.

## Delivery Artifacts

Submission package:

- Archive: `iclr2027_submission_pdf-only.tar.gz`
- PDF: `submission_bundle/main.pdf`
- Checksum file: `submission_bundle/SHA256SUMS.txt`
- Audit manifest: `audit_bundle/AUDIT_MANIFEST.md`
- Red-team report: `audit_bundle/RED_TEAM_REPORT.md`

Final hashes:

- PDF SHA256: `529e57b501d3567cc47e07fe170697d8f2d818e6f9d6a22c8204332f978c1245` (rebuilt 2026-09-05 after review-consistency fixes; supersedes earlier builds)
- Archive SHA256: `6fb935f7a34150f81c152596221482cd4075fdd14b5e4c60bf80c6f59d35fe6c`

The PDF-only upload intentionally excludes source and audit material. Source, JSON artifacts, protocol files, and audit reports remain in the workspace.

## Claim Boundaries to Preserve

Do not make any of the following claims without new proof and evidence:

- Every unconstrained SAE objective recovers every unknown direction.
- The `19/48` certificate stress result demonstrates universal training recovery.
- The conditional `lambda^-2` isolation benchmark is end-to-end trained-SAE convergence.
- The objective-agnostic certificate proves optimization success.
- The spectral anchor is a named-concept discovery method.
- The GPT-2/Pythia or public-atom results are unknown-direction semantic discovery.
- Residual mass is exactly equal to dark matter.
- The post-hoc 16k budget trace changes the failed held-out confirmation decision.

The safe wording is:

> The unconstrained feasible decoder class alone supplies no positive uniform alignment bound, while data-dependent learners remain possible. The unconstrained experiments establish a detector/training non-implication. Separate target-free exact-anchor and bounded-drift anchor-preserving SAE constructions admit objective/capacity-uniform anchor recovery under explicit eigengap and constraint conditions; the bounded-drift 48-row matrix also satisfies its relaxed gate in every tested row.

## Suggested Next Steps

1. **Submission-only continuation:** run the checksum command and upload `iclr2027_submission_pdf-only.tar.gz`; no code changes are needed.
2. **Scientific strengthening:** v4 is the current relaxed result. Any further relaxation, especially removing the trust region or allowing unconstrained residual leakage, requires a new theorem and a new pre-registered experiment; do not infer it from the current v4 `48/48` result.
3. **Reviewer preparation:** prepare a short explanation that the anchor theorem is a structural construction with target-free spectral estimation, not a vanilla optimizer convergence theorem. Keep the negative `19/48` stress result visible as the control.
4. **Future extensions:** possible directions are multiple anchors/subspaces, non-Gaussian concentration, a larger trust region, approximate leakage accounting, or an unconstrained trainable anchor with a stability theorem. Each would need fresh contract tests and a new artifact version.
5. **Artifact discipline:** any change to the protocol, theorem, table, or runner requires rerunning the focused tests, rebuilding the PDF, refreshing the clean-build comparison, and updating hashes in `audit_bundle/AUDIT_MANIFEST.md`.

## Operational Notes

- Working directory is `LOCAL_WORKSPACE_PATH/Desktop/iclr/old`; do not infer it from the DSH installation path.
- The v3 run used `remote GPU runtime`, physical GPU 5; the v4 bounded-drift run used physical GPU 6. Both completed processes have been stopped.
- The source directory is not currently recognized as a Git repository by `git status`; do not use Git reset/checkout commands as a cleanup mechanism.
- Earlier protocol backups may exist, including `experiments/spectral_anchor_sae_protocol.json.bak-20260901-1500`. The canonical exact-anchor protocol is the unsuffixed v3 JSON file; the canonical relaxed iteration is `experiments/bounded_drift_spectral_anchor_sae_protocol.json`.
- Never copy credentials or remote configuration into this document or into experiment logs.

## Canonical Starting Point

For a new session, read this document first, then inspect in this order:

1. `paper/main.tex`
2. `experiments/spectral_anchor_sae_protocol.json` and `experiments/bounded_drift_spectral_anchor_sae_protocol.json`
3. `experiments/run_spectral_anchor_sae.py` and `experiments/run_bounded_drift_spectral_anchor_sae.py`
4. `experiments/results_spectral_anchor_sae/spectral_anchor_summary.json` and `experiments/results_bounded_drift_spectral_anchor_sae/bounded_drift_spectral_anchor_summary.json`
5. `audit_bundle/RED_TEAM_REPORT.md`

The current project is complete for the stated exact-anchor and relaxed bounded-drift structured recovery claims and submission package. Further work should be treated as a new scientific iteration, not as a missing cleanup step.
