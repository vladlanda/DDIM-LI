"""Check: python baseline_lightgbm/test_permutation_importance.py -- the informative feature
gets the largest AP drop, a noise feature ~0, and the IR group includes its members."""
import json, os, sys, tempfile
import numpy as np
import lightgbm as lgb
here = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [here, os.path.dirname(here)]
import permutation_importance as P
from features import FEATURE_NAMES

rng, d = np.random.default_rng(0), tempfile.mkdtemp()
n, k = 20000, len(FEATURE_NAMES)
def make():
    X = rng.normal(size=(n, k)).astype(np.float32); X[:, -1] = rng.integers(0, 6, n)
    y = (X[:, FEATURE_NAMES.index("ir_trend_30min")] + 0.3 * rng.normal(size=n) > 1.2).astype(np.int8)
    return X, y
Xtr, ytr = make()
b = lgb.train({"objective": "binary", "verbose": -1}, lgb.Dataset(Xtr, ytr), 50)
b.save_model(f"{d}/model.txt"); json.dump({"feature_names": FEATURE_NAMES}, open(f"{d}/meta.json", "w"))
X, y = make()
P.collect = lambda args, meta: ([X[X[:, -1] == t] for t in range(6)], [y[X[:, -1] == t] for t in range(6)],
                                [np.ones((X[:, -1] == t).sum(), bool) for t in range(6)])
sys.argv = ["x", "--model_dir", d, "--test_roots", "unused", "--output", f"{d}/out.csv", "--n_repeats", "2"]
P.main()
import csv
r = {row["feature"]: float(row["drop_all"]) for row in csv.DictReader(open(f"{d}/out.csv"))}
assert max(r, key=r.get) in ("ir_trend_30min", "GROUP: IR cooling trends (30 min, 2 h)", "GROUP: all IR (cloud-top) features")
assert r["ir_trend_30min"] > 0.3 and abs(r["li_now_phys"]) < 0.02 and r["GROUP: all IR (cloud-top) features"] >= r["ir_trend_30min"] - 0.02
print("ok")
