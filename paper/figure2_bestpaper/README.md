# Figure 2 — detector–decoder decoupling

This package rebuilds Figure 2 from the frozen E4-v2 raw artifact. It does not use mock values or infer missing observations.

## Build

```bash
python3 build_figure2.py
```

The command validates the 48-row long-form source, forms all 24 seed × strength geometry pairs, reproduces the frozen seed-level bootstrap intervals, writes traceable CSV files, and exports the figure.

## Outputs

- `output/figure2_detector_decoder_decoupling.svg` — editable vector artwork and text.
- `output/figure2_detector_decoder_decoupling.pdf` — vector PDF.
- `output/figure2_detector_decoder_decoupling.png` — 300-dpi preview at 7.0 × 6.0 inches.
- `data/figure2_raw_long_48.csv` — all source observations.
- `data/figure2_paired_24.csv` — the requested 24 real seed × strength pairs.
- `data/figure2_primary_endpoint_8.csv` — the eight pre-specified $\lambda_{\mathrm{eff}}=0.55$ pairs.
- `data/figure2_bootstrap_summary.csv` — reproduced means and percentile-bootstrap intervals.
- `data/figure2_audit.json` — row counts, endpoint deltas, paired-audit counts, and source hash.
- `protocol.json` — detector, TopK, bootstrap, and evidence-status definitions.
- `figure2_caption.tex` — LaTeX caption with claim boundaries.

## Statistical definition

The original analysis resamples the eight seed-level values with replacement 1,000 times using `numpy.random.default_rng(0)`. Confidence intervals are the 2.5th and 97.5th percentiles of the bootstrap means. The generator is reused sequentially in the metric order recorded in `protocol.json`; this package verifies the results against the frozen summary before plotting.

Panels a–b are the pre-specified endpoint. Panels c–d reuse the full three-strength grid and are explicitly labeled cross-strength/descriptive analyses.
