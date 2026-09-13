# Real-residual validation figure

Run `python3 paper/figure4_lm_validation/build_lm_validation_figure.py` from the repository root. The script reads the corrected real-WikiText GPT-2/Pythia bridge artifacts and the completed two-model matched warm-start artifacts, writes three source-data CSV files, and exports editable SVG/PDF, a 300-dpi PNG, and a 600-dpi TIFF.

The bridge panels use four seeds. The matched panels use 48 pairs per model and summarize uncertainty over eight independent base-seed clusters. The target is never passed to training; oracle decoder matching is an evaluation metric. No Gaussian certificate is claimed for empirical residual backgrounds.
