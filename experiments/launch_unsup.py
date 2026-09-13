#!/usr/bin/env python3
import os
import subprocess
from pathlib import Path

root = Path("/mnt/liuzelin/weak_feature_visibility")
out = root / "experiments/results_unsup"
out.mkdir(parents=True, exist_ok=True)
log = open(out / "run.log", "w")
env = os.environ.copy()
env["CUDA_VISIBLE_DEVICES"] = os.environ.get("EXP_GPU", "5")
env["HF_HOME"] = "/mnt/e1_runs/hf_cache"
env["HUGGINGFACE_HUB_CACHE"] = "/mnt/e1_runs/hf_cache/hub"
env["TRANSFORMERS_OFFLINE"] = "1"
env["HF_HUB_OFFLINE"] = "1"
p = subprocess.Popen(
    [
        "/mnt/miniconda3/bin/python",
        "-u",
        str(root / "experiments/run_unsup_natural.py"),
    ],
    cwd=str(root),
    env=env,
    stdout=log,
    stderr=subprocess.STDOUT,
    stdin=subprocess.DEVNULL,
    start_new_session=True,
)
(out / "pid").write_text(str(p.pid))
print("UNSUP_PID", p.pid, flush=True)
