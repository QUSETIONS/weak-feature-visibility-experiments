#!/usr/bin/env python3
import os
import subprocess
from pathlib import Path

root = Path("/mnt/liuzelin/weak_feature_visibility")
out = root / "experiments/results_gpt2"
out.mkdir(parents=True, exist_ok=True)
log = open(out / "nat_sae.log", "w")
env = os.environ.copy()
env["CUDA_VISIBLE_DEVICES"] = "3"
p = subprocess.Popen(
    ["/mnt/miniconda3/bin/python", "-u", str(root / "experiments/run_gpt2_natural_sae.py")],
    cwd=str(root),
    env=env,
    stdout=log,
    stderr=subprocess.STDOUT,
    stdin=subprocess.DEVNULL,
    start_new_session=True,
)
print("NAT_SAE_PID", p.pid, flush=True)
