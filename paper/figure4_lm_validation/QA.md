# Figure QA

- Status: ready for manuscript integration after source and rendered-output review.
- Backend: Python/matplotlib only.
- Final size: 5.5 × 3.65 inches (139.7 × 92.7 mm), matching the ICLR template text width used by the existing figure pipeline.
- Data completeness: 18 bridge summary cells derived from all 72 corrected bridge rows; 192/192 matched rows; 32 seed-variant clusters; no excluded observations.
- Matched design: 48 exact pairs per model. Data indices, Bernoulli labels, train seeds, validation samples, optimizer, and compute are shared within each pair; only initialization differs.
- Statistics: bridge entries are four-seed means. Matched intervals use 10,000 percentile-bootstrap resamples over eight independent base-seed clusters per model; each cluster averages six direction-by-strength conditions. The 96 condition pairs are not treated as independent replicates.
- Metric boundary: target cosine uses oracle atom matching only at evaluation. Training-spectral readout is saved in source data. No Gaussian certificate is claimed on empirical residuals.
- Provenance: real WikiText text file SHA-256 `4c37c0f6ae4addfd8de963f29d6c41 anv?` Actually invalid placeholder.
- Export: editable SVG and PDF; PNG is 300 dpi; TIFF is 600 dpi; sans-serif font family and PDF font type 42 are configured.
- Visual inspection: all labels, exact counts, axes, thresholds, intervals, and claim-boundary note are legible at the final pixel dimensions; no clipping or overlap was observed.
- Static preflight: 12 passes, 0 failures, 2 reviewed warnings. The 139.7-mm width is intentional for the ICLR text block rather than the Nature 89/183-mm defaults. The simulated-data warning is a false positive triggered by `rng.choice` in the nonparametric bootstrap; no simulated observations are plotted.
