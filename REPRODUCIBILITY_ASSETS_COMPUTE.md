# Reproducibility, Assets, and Compute

This note records the concrete information needed for the ICLR 2027 checklist and anonymous supplemental package.

## Code and data access

The anonymous review package contains:

- `experiments/`: Python scripts for the formal synthetic experiments, GPT-2 activation preprocessing, controlled injection tests, natural-event probes, SAE-facing audits, and summary generation.
- `results/`: CSV/JSON/PDF/PNG artifacts used to produce the paper tables, figures, and reported numbers.
- `assets_metadata/`: license/config metadata for the pretrained GPT-2 SAE artifacts used by the C17/C19/C20/C21 audits.
- `paper/`: anonymous LaTeX source, editable figure source/data, rendered figures, and the latest anonymous PDF.

Large model files, SAE weights, activation caches, `.npy` caches, `.pt` checkpoints, and `__pycache__` files are intentionally excluded from the portable archive. The scripts document how the excluded activation caches were produced from public artifacts.

## Main reproducibility commands

Formal synthetic validation:

```bash
python experiments/run_server_experiments.py --outdir results/server_formal_v2 --seed 20260502 --m 8 --trials 700 --seeds 8 --alpha 0.05 --target-power 0.8 --bootstrap 1000
```

GPT-2 layer-6 residual-stream activation collection:

```bash
python experiments/real_activation_bridge.py collect --model-name assets/gpt2 --text-file assets/wikitext_train.txt --layer 6 --hook-position resid_pre --skip-first-token --tokens 100000 --device cuda:0 --outdir results/real_activation/c1_100k_resid_pre_clean
```

PCA whitening and bridge audits are run by the preprocessing and audit scripts in `experiments/`, with paths recorded in the corresponding JSON summaries under `results/real_activation/`.

## External assets and licenses

| Asset | How it is used | Local path / source | License status |
|---|---|---|---|
| GPT-2 small / `openai-community/gpt2` | Generate layer-6 residual-stream activations; no generated text is released. | `assets/gpt2` cache; Hugging Face model card | MIT license on the model card. |
| WikiText-2 raw v1 | Public text stream for activation collection and natural-event token labels. | `assets/wikitext_train.txt`; `Salesforce/wikitext`, `wikitext-2-raw-v1` | Hugging Face dataset card lists CC BY-SA 4.0. |
| GPT-2 small SAEs, `jbloom/GPT2-Small-SAEs-Reformatted`, `blocks.{4,6,8}.hook_resid_pre` | Pretrained SAE natural-event audits and second-order bridge audit. | `assets/gpt2_sae_layer{4,6,8}/blocks.{4,6,8}.hook_resid_pre` | Hugging Face model page and local README indicate MIT. We do not redistribute weights in the portable archive. |
| NLTK POS/NER resources | Define external POS and named-entity events. | `assets/nltk_data` cache | NLTK data repository is Apache-2.0 at the repository level and notes heterogeneous dataset licenses; the archive includes result summaries, not a redistributed NLTK data package. |
| Python scientific stack | Numerical experiments and plots. | NumPy, PyTorch, matplotlib, safetensors, transformers, datasets, NLTK | Standard open-source dependencies; exact versions should be frozen in the final anonymous code release if required. |

## Compute actually used

The formal synthetic experiment is CPU-only by design. The recorded formal run used:

- Platform: Linux CPU run
- Configuration: `m=8`, `seed=20260502`, `8` independent seeds, `700` Monte Carlo trials, `alpha=0.05`, `1000` bootstrap resamples
- Wall time: `545.277` seconds
- Source artifact: `results/server_formal_v2/formal_summary.json`

The real-activation bridge uses cached GPT-2-small layer-6 activations:

- Main activation cache: `100000` tokens, `d_model=768`, layer 6, `resid_pre`
- Whitening/audit cache: `30000` tokens for PCA-whitened bridge experiments
- C6 controlled detector collapse: `735` configurations over `30000` whitened tokens
- C7/C8 geometry and detector-class audits: `400` trials at `n=8192`
- C13 natural-event bank: `47` externally defined events, 50/50 train/test splits, Benjamini-Hochberg correction
- C16 controlled TopK SAE audit: `18` configurations, `30000` tokens, `2362.36` seconds in the recorded summary
- C17 pretrained SAE audit on GPT-2 layer 6: `41` evaluated events, `d_sae=24576`, `110.48` seconds in the recorded CUDA summary
- C20/C21 pretrained SAE robustness audits on GPT-2 layers 8 and 4: `41` evaluated events each, `d_sae=24576`, sparse-calibrated random baselines, `283.91` and `266.23` seconds in the recorded CUDA summaries
- C19 bridge-assumption audit: `12000` tokens, 50/50 train/test split, CPU for the recorded C17-atom audit

These are audit-scale experiments, not large foundation-model training runs.

The Pythia-160M scale extension is recorded in `experiments/results_lm_spectral_warm_start_scale_extension/`; its separately frozen
6k-step same-data budget follow-up is in `experiments/results_lm_spectral_warm_start_scale_extension_budget/`. The 6k follow-up
contains 48 complete pairs and reports a null-compatible cosine difference and held-out MSE difference, so the paper frames
the warm-start effect as model- and budget-sensitive rather than universal.

## Anonymous manuscript build

```bash
make -C paper TECTONIC=tectonic
```

This produces the anonymous `paper/main.pdf`. The author-version source and PDF are deliberately not part of the public repository.
