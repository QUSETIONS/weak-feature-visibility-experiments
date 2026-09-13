#!/usr/bin/env python3
import os
import subprocess
from pathlib import Path

root = Path("/mnt/liuzelin/weak_feature_visibility")
out = root / "experiments/results_exponent_wide"
out.mkdir(parents=True, exist_ok=True)
log = open(out / "run.log", "w")
env = os.environ.copy()
env["CUDA_VISIBLE_DEVICES"] = os.environ.get("EXP_GPU", "3")
p = subprocess.Popen(
    [
        "/mnt/miniconda3/bin/python",
        "-u",
        str(root / "experiments/run_sae_exponent.py"),
        "--out",
        str(out),
        "--device",
        "cuda",
        "--m",
        "32",
        "--dict-size",
        "16",
        "--n-seeds",
        "6",
        "--n-epochs",
        "80",
        "--lambdas",
        "0.30,0.34,0.38,0.42,0.48,0.55,0.65",
        "--n-grid",
        "1024,4096,16384,32768,65536,131072",
    ],
    cwd=str(root),
    env=env,
    stdout=log,
    stderr=subprocess.STDOUT,
    stdin=subprocess.DEVNULL,
    start_new_session=True,
)
(out / "pid").write_text(str(p.pid))
print("EXPONENT_WIDE_PID", p.pid, flush=True)
