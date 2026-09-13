# Figure contract

Core conclusion: The detector-axis hole transfers to two empirical residual backgrounds, while a training-only spectral warm start improves known injected-direction recovery in both models with model-dependent effect size and no detectable held-out reconstruction cost.

Figure archetype: asymmetric quantitative grid.

Target/output: ICLR full-width main-text figure; Python/matplotlib; 5.5 × 3.65 inches; editable SVG and PDF, 300-dpi PNG, and 600-dpi TIFF.

Panel map:

- a: corrected real-WikiText detector power at lambda 0.35, separating PC axes from the dense direction.
- b: corrected vanilla TopK recovery over direction and strength, showing that the empirical backgrounds are not interchangeable.
- c: hero panel with all eight base-seed cluster recovery rates for vanilla versus warm start, split by model.
- d: model-specific seed-cluster bootstrap intervals for target-cosine and held-out reconstruction-MSE differences.

Evidence hierarchy: panel c is the primary intervention evidence; panel d is the effect-size and cost audit; panels a and b establish the transfer setting and baseline heterogeneity.

Statistics: bridge values are means over four seeds. Matched effects use eight base-seed clusters per model, each averaging six direction-by-strength conditions. Intervals are 10,000-resample percentile bootstrap intervals over the eight clusters. Condition rows are not treated as independent replicates.

Source data: all 72 corrected bridge rows and all 192 matched rows are used; no row or model is excluded.

Reviewer risks: all targets are known injected directions; oracle atom matching is evaluation-only; the spectral readout is training-only; empirical residual backgrounds are not assigned the Gaussian certificate; the two-model pooled effect is descriptive only.
