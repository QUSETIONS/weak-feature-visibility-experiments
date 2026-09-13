# Figure 3 QA

- Core conclusion: a training-only spectral signed-pair warm start improves recovery and held-out certification under matched compute, but does not establish universal convergence.
- Archetype: quantitative grid with one paired-observation hero panel and two audit panels.
- Backend: Python/matplotlib only.
- Final size: 7.0 x 2.15 inches before manuscript scaling.
- Source observations: 96 raw rows forming 48 complete matched pairs; no exclusions.
- Panel a: all 48 paired target-cosine observations; dashed identity and frozen 0.80 recovery thresholds.
- Panel b: exact certificate counts over eight geometry-seed rows in each objective-capacity cell.
- Panel c: recovery and certificate counts for three training-only readouts on the same 48 warm-start models.
- Statistics: descriptive paired observations and exact counts; no confidence interval or hypothesis test is claimed.
- Integrity: target, validation data, and validation metrics were not used for training or readout selection.
- Reviewer risk: the result is an initialization ablation at one rank-one Gaussian signal strength, not an unknown-feature discovery or universal SAE result.
