#!/usr/bin/env python3
import os
import subprocess
from pathlib import Path

root = Path("/mnt/liuzelin/weak_feature_visibility")
out = root / "experiments/results_public_sae"
out.mkdir(parents=True, exist_ok=True)
log = open(out / "run.log", "w")
env = os.environ.copy()
env["CUDA_VISIBLE_DEVICES"] = os.environ.get("EXP_GPU", "2")
env["HF_HOME"] = "/mnt/e1_runs/hf_cache"
env["HUGGINGFACE_HUB_CACHE"] = "/mnt/e1_runs/hf_cache/hub"
env["TRANSFORMERS_OFFLINE"] = "1"
env["HF_HUB_OFFLINE"] = "1"
npy = root / "experiments/weights/jbloom_gpt2_l6/W_dec.npy"
sft = root / "experiments/weights/jbloom_gpt2_l6/sae_weights.safetensors"
cmd = [
    "/mnt/miniconda3/bin/python",
    "-u",
    str(root / "experiments/run_public_sae_c.py"),
    "--out",
    str(out),
    "--device",
    "cuda",
]
if npy.is_file():
    cmd.extend(["--weights", str(npy)])
elif sft.is_file():
    cmd.extend(["--weights", str(sft)])
p = subprocess.Popen(
    cmd,
    cwd=str(root),
    env=env,
    stdout=log,
    stderr=subprocess.STDOUT,
    stdin=subprocess.DEVNULL,
    start_new_session=True,
)
(out / "pid").write_text(str(p.pid))
print("PUBLIC_SAE_C_PID", p.pid, flush=True)
