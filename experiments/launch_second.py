#!/usr/bin/env python3
import os
import subprocess
from pathlib import Path

root = Path("/mnt/liuzelin/weak_feature_visibility")
out = root / "experiments/results_pythia"
out.mkdir(parents=True, exist_ok=True)
log = open(out / "run.log", "w")
env = os.environ.copy()
env["CUDA_VISIBLE_DEVICES"] = os.environ.get("EXP_GPU", "5")
env["HF_HOME"] = "/mnt/e1_runs/hf_cache"
env["HUGGINGFACE_HUB_CACHE"] = "/mnt/e1_runs/hf_cache/hub"
env["TRANSFORMERS_OFFLINE"] = "1"
env["HF_HUB_OFFLINE"] = "1"
weights = root / "experiments/weights/pythia-70m"
cmd = [
    "/mnt/miniconda3/bin/python",
    "-u",
    str(root / "experiments/run_second_lm.py"),
    "--out",
    str(out),
    "--device",
    "cuda",
]
if (weights / "config.json").is_file():
    cmd.extend(["--model-dir", str(weights)])
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
print("PYTHIA_PID", p.pid, flush=True)
