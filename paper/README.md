# Anonymous paper source and figures

This directory contains the anonymous manuscript source, the editable figure
source/data packages, rendered figures, and the latest anonymous PDF.

## Build

From a machine with Tectonic or a compatible LaTeX installation:

```bash
make -C paper TECTONIC=tectonic
```

The build target is `paper/main.pdf`. Figure-specific source packages are in
`figure2_bestpaper/`, `figure3_bestpaper/`, and `figure4_lm_validation/`.
Each package includes its plotting script, source data or protocol, and QA
notes. The author-version source and PDF are intentionally excluded from this
anonymous repository.
