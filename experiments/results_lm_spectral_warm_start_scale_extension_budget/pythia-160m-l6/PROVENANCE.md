# Pythia-160M 6000-step budget follow-up provenance

- Protocol: `experiments/lm_spectral_warm_start_scale_extension_budget_protocol.json`
- Protocol SHA-256: `e06996fd409849b1a934ec1513aaae3e95c991ba7379e4a2bfd860339d33e554`
- Parent protocol: `experiments/lm_spectral_warm_start_scale_extension_protocol.json`
- Parent protocol SHA-256: `dff375051f6892d58a9572ff2f8e8e7d7b36ae7cbbf5f435f381fbe9ade44ffb`
- Model: `EleutherAI/pythia-160m`, layer 6, hidden size 768
- Hugging Face snapshot revision: `50f5173d932e8e61f858120bcb800b97af589f46`
- Corpus: explicit real WikiText-2 train text; SHA-256 `4c37c0f6ae4addfd8de963f29d6c41f65f41f9626de6013e5da889cc5cc20a0c`
- Split: 4,000 train and 1,500 validation sequences of length 128; zero exact sequence-content overlap
- Cache: reused the parent experiment's train-only whitening residual cache; no data or split changes
- SAE: dictionary 64, TopK 4, 24,000 train and 24,000 held-out samples, 6,000 Adam steps, learning rate 0.001
- Result raw JSON SHA-256: `199e96e8535320b970d701086d5a862a803fbe4f7097f883e4de810ab159453a`
- Result summary JSON SHA-256: `760ba439216c441a36398979b731b5b4ddbb418192a90c453677329ee5156668`
- Aggregate summary JSON SHA-256: `79422ecd3b098a4043d3893805e4760ff95ec80ab42f44516231301dc62636b0`
- Completed rows: 96; paired conditions: 48; data-quality gates: all passed
- Budget result: vanilla recovery 32/48 versus warm recovery 31/48; mean paired cosine delta -0.012775 (95% bootstrap CI [-0.031673, 0.008854]); mean held-out MSE delta +0.000258 (95% bootstrap CI [-0.000413, 0.001002])
- Reporting boundary: same-data compute-budget follow-up for the known-direction Pythia-160M extension; it tests persistence of the 3000-step effect and is not an independent data replication, Gaussian certificate, or universal scale claim
