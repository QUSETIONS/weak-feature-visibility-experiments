# Figure 2 QA record

- Core conclusion: a fixed-basis detector split is large at the pre-specified endpoint, while matched TopK recovery remains high in the same runs.
- Archetype: asymmetric quantitative grid with a dominant decoupling panel.
- Source integrity: 48/48 raw rows retained; 24/24 geometry pairs complete; no exclusions or imputation.
- Primary endpoint: eight seeds at `lambda_eff = 0.55`.
- Secondary status: the cross-strength plane and 24-pair recovery audit are labeled descriptive.
- Bootstrap: seed-level nonparametric percentile bootstrap, 1,000 resamples, `numpy.random.default_rng(0)`, reproduced exactly against the frozen summary.
- Endpoint audit: mean dense-minus-axis independence difference `0.91875`; mean decoder-cosine difference `0.008123746880640825`.
- Paired audit: detector class flips `24/24`; binary recovery agrees `22/24`.
- Export: SVG retains editable text; PDF is vector; PNG is 2100 × 1800 pixels at 300 dpi; PDF media box is exactly 7.0 × 6.0 inches.
- Visual inspection: panel labels, thresholds, callouts, raw points, paired lines, and audit cells are visible without clipping at final size.
- Validator: no failures. The remaining warnings are expected because the requested raster is 300-dpi PNG rather than 600-dpi TIFF, and the validator conservatively flags the bootstrap RNG as possible simulation.
