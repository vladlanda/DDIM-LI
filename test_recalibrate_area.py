"""Check for recalibrate_area.py (synthetic, CPU, seconds): python test_recalibrate_area.py
A CNN-max biased low by a fixed monotone map is restored by the validation-fitted isotonic fit."""
import csv, os, subprocess, sys, tempfile
import numpy as np
from evaluate import area_collect, save_area_outputs

rng, root = np.random.default_rng(0), tempfile.mkdtemp()
def make(d, name, n, f):
    store = {}
    for s in range(n):
        for t in (0, 1):
            truth = rng.random((16, 16)) * 0.6
            obs = rng.random((16, 16)) < truth
            area_collect(store, f(truth), obs, 2, t, s)
    os.makedirs(d, exist_ok=True); save_area_outputs(d, name, store)
# same seed stream for test CNN and diffusion so labels align: build both from one draw
make(f"{root}/val", "cnn_nmax", 60, lambda p: p ** 2)              # biased low
rng = np.random.default_rng(1); make(f"{root}/cnn", "cnn_nmax", 40, lambda p: p ** 2)
rng = np.random.default_rng(1); make(f"{root}/dif", "diffusion", 40, lambda p: p)  # calibrated
r = subprocess.run([sys.executable, "recalibrate_area.py", "--val_dir", f"{root}/val", "--cnn_test_dir",
                    f"{root}/cnn", "--diffusion_test_dir", f"{root}/dif", "--out_dir", f"{root}/out",
                    "--n_boot", "50"], capture_output=True, text=True)
assert r.returncode == 0, r.stderr
row = next(csv.DictReader(open(f"{root}/out/area_recal_comparison.csv")))
b = {k: float(v) for k, v in row.items()}
assert b["brier_cnn_recal"] < b["brier_cnn_nmax"]                  # recalibration helps
assert abs(b["fbias_cnn_recal"] - 1) < abs(b["fbias_cnn_nmax"] - 1)  # and removes the bias
assert os.path.exists(f"{root}/out/area_cnn_nmax_recal_r2px.npz")
print("ok")
