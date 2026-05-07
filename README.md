# Weak-Feature Visibility Experiments

This anonymous repository contains the experiment code and lightweight result artifacts for the paper
`Why Weak Features Hide: A Local Detector-Class Visibility Calculus for Superposition`.

The repository is intentionally lightweight. It includes scripts, CSV/JSON summaries, and paper figures,
but excludes large model weights, GPT-2 activation caches, SAE weights, Python caches, and local logs.
Public artifacts used by the experiments are documented in `REPRODUCIBILITY_ASSETS_COMPUTE.md`.

## Repository Layout

- `experiments/`: scripts for synthetic validation, GPT-2 bridge experiments, natural-event probes, and SAE-facing audits.
- `results/`: selected result summaries and figures used by the paper.
- `assets_metadata/`: license/config metadata for public pretrained SAE artifacts; weights are not included.
- `EXPERIMENTS_OVERVIEW.md`: claim-to-experiment index.
- `REPRODUCIBILITY_ASSETS_COMPUTE.md`: data, asset, and compute notes.

## Main Evidence Chain

The experiments are organized as a staged validation:

1. Matched synthetic data verifies the detector-class law, exponent separation, geometry penalty, evidence collapse, and residual-floor prediction.
2. Controlled GPT-2 injections test whether the evidence coordinate transfers to real activation backgrounds.
3. Externally defined natural events test whether the same ordering appears without injected labels.
4. SAE-facing audits test whether recovery by controlled or pretrained sparse decoders becomes more favorable as detector-law evidence increases.

## Reproducing Key Results

Formal synthetic validation:

```bash
python experiments/run_server_experiments.py \
  --outdir results/server_formal_v2 \
  --seed 20260502 \
  --m 8 \
  --trials 700 \
  --seeds 8 \
  --alpha 0.05 \
  --target-power 0.8 \
  --bootstrap 1000
```

GPT-2 activation collection example:

```bash
python experiments/real_activation_bridge.py collect \
  --model-name assets/gpt2 \
  --text-file assets/wikitext_train.txt \
  --layer 6 \
  --hook-position resid_pre \
  --skip-first-token \
  --tokens 100000 \
  --device cuda:0 \
  --outdir results/real_activation/c1_100k_resid_pre_clean
```

Pretrained-SAE natural-event audit example:

```bash
python experiments/pretrained_sae_natural_event_audit.py \
  --activation-path results/real_activation/c1_10k_resid_pre_clean/gpt2_layer6_resid_pre_10000tok.npy \
  --sae-dir assets/gpt2_sae_layer6/blocks.6.hook_resid_pre \
  --bank-csv results/real_activation/c13_preregistered_feature_bank/preregistered_feature_bank.csv \
  --outdir results/real_activation/c17_pretrained_sae_natural_event_audit_sparse \
  --model-name assets/gpt2 \
  --text-file assets/wikitext_train.txt \
  --nltk-data-dir assets/nltk_data \
  --random-baseline sparse_calibrated
```

The commands above reference large public assets and generated caches that are not redistributed here.
The included summaries under `results/` record the reported numerical outputs.

## Dependencies

See `requirements.txt` for a minimal Python dependency list. Exact versions are not required for reading
the result summaries, but should be pinned in a final archival release if full reruns are required.

## Anonymity Notes

This repository is prepared for anonymous review. It should not contain author names, institution names,
local machine paths, private server addresses, model weights, activation caches, or funding statements.

