#!/usr/bin/env python3
import os
import subprocess
from pathlib import Path

root = Path("/mnt/liuzelin/weak_feature_visibility")
out = root / "experiments/results_label_nstar"
out.mkdir(parents=True, exist_ok=True)
log = open(out / "run.log", "w")
env = os.environ.copy()
p = subprocess.Popen(
    [
        "/mnt/miniconda3/bin/python",
        "-u",
        str(root / "experiments/run_label_nstar.py"),
        "--out",
        str(out),
    ],
    cwd=str(root),
    env=env,
    stdout=log,
    stderr=subprocess.STDOUT,
    stdin=subprocess.DEVNULL,
    start_new_session=True,
)
(out / "pid").write_text(str(p.pid))
print("LABEL_NSTAR_PID", p.pid, flush=True)
