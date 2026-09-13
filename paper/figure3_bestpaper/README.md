# Figure 3: matched-compute spectral warm start

This figure is generated from the complete paired matched-compute experiment and the target-free readout audit. The 48 conditions span three objectives, two capacities, two geometries, and four seeds. Every condition uses matched data, training, validation, and random seeds across the vanilla and warm-start variants; the only variant difference is initialization.

Run:

```bash
python3 build_figure3.py
```

The script validates row count, pair integrity, seed matching, summary consistency, and evidence eligibility before exporting PDF, editable SVG, and 600-dpi PNG/TIFF files. It also exports the plotted observations to CSV. No observations are filtered or simulated.
