# Pythia-160M scale-extension provenance

- Protocol: `experiments/lm_spectral_warm_start_scale_extension_protocol.json`
- Protocol SHA-256: `dff375051f6892d58a9572ff2f8e8e7d7b36ae7cbbf5f435f381fbe9ade44ffb`
- Model: `EleutherAI/pythia-160m`, layer 6, hidden size 768
- Hugging Face snapshot revision: `50f5173d932e8e61f858120bcb800b97af589f46`
- Downloaded config SHA-256: `76eb275107220e450d31258f792a2efcbee109d8b62ae0088260057dec06362f`
- Downloaded `model.safetensors` SHA-256: `29d2e457a664e41c12c735f20a36dc0956a665f614a54ce5db21a32e75965270`
- Corpus: explicit real WikiText-2 train text; SHA-256 `4c37c0f6ae4addfd8de963f29d6c41f65f41f9626de6013e5da889cc5cc20a0c`
- Split: 4,000 train and 1,500 validation sequences of length 128; zero exact sequence-content overlap
- Result raw JSON SHA-256: `f9b06be62b8882596513ad8dfc833775dcdff88718b543ab9f50cbc866d8efd4`
- Result summary JSON SHA-256: `8e5f5b899666679795cd8acff644642e1409aa20b5464c279230b907f2668ee1`
- Completed rows: 96; paired conditions: 48; data-quality gates: all passed
- Reporting boundary: known injected directions on one larger empirical residual background; no Gaussian certificate or unknown-direction semantic-discovery claim
