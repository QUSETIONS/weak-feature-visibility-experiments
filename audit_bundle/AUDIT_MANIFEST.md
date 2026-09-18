# ICLR 2027 Final Artifact Manifest

Generated: 2026-08-31
Workspace: LOCAL_WORKSPACE_PATH/Desktop/iclr/old

## Submission Upload

- Archive: iclr2027_submission_pdf-only.tar.gz
- Archive SHA256: 3039407eaa2ecbb34ab672bbdf66f29622f649804c0b232314073889adb14256
- Archive members: main.pdf, SHA256SUMS.txt
- PDF SHA256: eddc413a130bb38d4b7bf962f480ea4af81ddbb123fc3b3a66d5c5bdc310a45b
- PDF pages: 21
- References start: page 14 in the v4 layout
- E4-v2 control figure: page 9; residual-floor figure: page 10; GPT-2 bridge: page 11; public-Bloom figure, Bloom Table 5, and Discussion: page 12; References: page 14; AI disclosure and Proofs: page 15; held-out Table 6 and post-hoc budget Table 7: page 18; experimental protocol: page 17; corrected objective/capacity Table 8 and exact-anchor Table 9: page 19; bounded-drift Table 10: page 20
- Anonymous author line: page 1 only

## Source Outputs

- paper/main.tex
- paper/make_figures.py
- paper/Makefile
- paper/refs.bib
- experiments/results_v2/e4v2_summary.json
- experiments/results_v2/e4v2_raw.json
- ICLR_experiment_design_v2.md
- experiments/README.md
- paper/figures/fig_public_atoms_supplied.pdf
- paper/figures/fig_public_atoms_supplied.png
- paper/figures/fig_sae_floor_supplied.pdf
- paper/figures/fig_sae_floor_supplied.png
- experiments/results_public_atoms/atoms.json
- experiments/results_public_atoms/summary.json
- audit_bundle/RED_TEAM_REPORT.md
- experiments/e4_topk_confirmation_protocol.json
- experiments/run_e4_topk_confirmation.py
- experiments/tests/test_e4_topk_confirmation.py
- experiments/results_e4_topk_confirmation/e4_topk_confirmation_raw.json
- experiments/results_e4_topk_confirmation/e4_topk_confirmation_summary.json
- experiments/e4_topk_budget_followup_protocol.json
- experiments/run_e4_topk_budget_followup.py
- experiments/tests/test_e4_topk_budget_followup.py
- experiments/results_e4_topk_budget_followup/e4_topk_budget_followup_raw.json
- experiments/results_e4_topk_budget_followup/e4_topk_budget_followup_summary.json
- experiments/run_reconstruction_certificate.py
- experiments/tests/test_reconstruction_certificate.py
- experiments/results_reconstruction_certificate/certificate.json
- experiments/results_reconstruction_certificate/summary.json
- experiments/reconstruction_certificate_stress_protocol.json
- experiments/run_reconstruction_certificate_stress.py
- experiments/tests/test_reconstruction_certificate_stress.py
- experiments/results_reconstruction_certificate_stress/stress_raw.json (canonical v3)
- experiments/results_reconstruction_certificate_stress/stress_summary.json (canonical v3; corrected projector-difference bound)
- experiments/spectral_anchor_sae_protocol.json
- experiments/run_spectral_anchor_sae.py
- experiments/tests/test_spectral_anchor_sae.py
- experiments/results_spectral_anchor_sae/spectral_anchor_raw.json
- experiments/results_spectral_anchor_sae/spectral_anchor_summary.json
- experiments/bounded_drift_spectral_anchor_sae_protocol.json
- experiments/run_bounded_drift_spectral_anchor_sae.py
- experiments/tests/test_bounded_drift_spectral_anchor_sae.py
- experiments/results_bounded_drift_spectral_anchor_sae/bounded_drift_spectral_anchor_raw.json
- experiments/results_bounded_drift_spectral_anchor_sae/bounded_drift_spectral_anchor_summary.json

## Verification

- Tectonic build: PASS
- Objective-agnostic certificate demo: 2/2 held-out synthetic cases certified under stated assumptions; artifact marked synthetic-validation-demo and evidence-ineligible
- Cross-objective/capacity stress v3: 48/48 rows complete and gate-consistent under projector-difference rank factor 2; certificate pass rate 19/48=0.3958, direct recovery 0.4792, evidence-eligible=false; earlier v2 summary superseded and archived outside the workspace
- Spectral-anchor SAE v3: 48/48 rows complete; target-free training spectral anchor, family-wise certified cosine lower bound 0.9405, observed minimum cosine 0.9989, residual-anchor inner product at most 1e-6, evidence-eligible=true
- Bounded-drift spectral-anchor SAE v4: 48/48 rows complete; trainable training-only anchor with rho=0.20 rad trust region, residual leakage cap eta=0.10, certified cosine lower bound 0.8543, minimum observed cosine 0.9899, maximum drift 0.1287 rad, evidence-eligible=true
- Fresh clean-directory Tectonic build: PASS (20 pages; page text and rendered pixels match after corrected v3 Table 8 disclosure, finite-sample bound correction, spectral-anchor Tables 9--10, the unconstrained-class boundary proposition, and bounded-drift theorem)
- Fresh clean build page text and PyMuPDF raster pixels: identical to paper/main.pdf; clean directory: /tmp/iclr-bounded-drift-v4-clean-20260902-0730
- Python syntax and contract tests: PASS after corrected v3 update; held-out confirmation 4/4, budget trace 2/2, reconstruction certificate 15/15, stress protocol 7/7, spectral-anchor protocol/artifact 12/12, bounded-drift v4 protocol/artifact 9/9 on the PyTorch A40 runtime; 49 tests total across the final focused suites (40 existing local tests plus 9 v4 tests)
- Semantic red-team guards: PASS
- Citation/cross-reference/figure closure: PASS (8 figures; all figure labels referenced)
- Public atom JSON audit: PASS, 9 atoms; required IDs 23123, 14698, 1, 1834, 0 present
- PDF local-path scan: PASS
- PDF visual inspection: PASS on final pages including held-out Tables 6, 7, 8, exact-anchor Table 9, and bounded-drift Table 10, no visible clipping or overlap

## Known Non-Blocking Warnings

Tectonic reports several underfull vboxes and duplicate hyperref object names. No overfull hbox or annotation-boundary warning remains; final page renders show no clipping or overlap. Pages 17--20 contain the experimental protocol, held-out confirmation, post-hoc budget/certificate stress, exact-anchor Table 9, and bounded-drift Table 10; the v3/v4 results are separate family-level claims, not unconstrained-SAE claims.
## v5 Dose-Response Update (2026-09-05)

Experiment and artifact:

- v5 dose-response protocol: nine mechanisms x eight independent anchor groups = 72/72 rows complete on remote GPU runtime (torch 2.3.1+cu121, GPU 5); artifact gate passed against the real artifact.
- Canonical local artifacts: experiments/results_dose_response/dose_response_raw.json and dose_response_summary.json; protocol at experiments/dose_response_spectral_anchor_protocol.json (backup .bak-20260902-0900).
- Key results: anchor recovery mechanism-independent at this scale (8/8 for every mechanism, worst-group cosine >= 0.9788); unbounded control max residual--anchor leak 0.734 with worst held-out loss 0.1414+-0.0077; every constrained mechanism caps leak at <= 0.101 with loss <= 0.1373; frozen anchor best worst-group cosine 0.9983.

Manuscript (paper/main.tex):

- Appendix paragraph "Dose-response and matched-compute baselines" and Table 11 (tab:dose_response): lines 614--637.
- Limitations sentence added (line 413): recovery is mechanism-independent at this scale; the certified family buys provable leakage control and lower variance, not recovery itself.
- Main-text pagination unchanged: References still start on page 14.

Verification:

- v5 suite: 13 tests, 10 passed + 3 no-torch environment skips (artifact-gate test ran against the real 72/72 artifact); combined local suite 96 tests all green (11 environment skips).
- Clean-directory rebuild: /tmp/iclr-v5-clean-20260905, 0 errors, 21 pages, page text identical to paper/main.pdf, Table 11 on page 21.
- Rollback path: paper/main.tex.bak-20260905-pre-v5 (pre-v5 main.tex).

Submission bundle refresh:

- submission_bundle/main.pdf: 966852 bytes, 21 pages, SHA256 eddc413a130bb38d4b7bf962f480ea4af81ddbb123fc3b3a66d5c5bdc310a45b
- iclr2027_submission_pdf-only.tar.gz rebuilt (members: main.pdf, SHA256SUMS.txt); in-archive checksum self-verification PASS.

## Review-Consistency Fix Rebuild (2026-09-05)

- paper/main.tex: nine factual defects from the adversarial review corrected (details in HANDOFF "Review-Consistency Fixes"); rollback main.tex.bak-20260905-prereviewfix.
- submission_bundle/main.pdf rebuilt: 968082 bytes, 21 pages, SHA256 529e57b501d3567cc47e07fe170697d8f2d818e6f9d6a22c8204332f978c1245.
- iclr2027_submission_pdf-only.tar.gz rebuilt (SHA256 6fb935f7a34150f81c152596221482cd4075fdd14b5e4c60bf80c6f59d35fe6c); clean-dir checksum self-verification PASS.
- Contracts: test_manuscript_review_consistency.py 9/9 green; full suite 105 collected, 94 passed, 11 environment skips.

Warning characterization (historical): the xdvipdfmx "Object @table.N already defined" warnings reproduce identically on rerun builds of the reverted pre-v5 baseline (18 occurrences), confirming they are a tectonic rerun-pass artifact, not a v5 regression; output rendering verified correct.
