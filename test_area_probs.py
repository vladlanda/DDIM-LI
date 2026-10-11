"""Check of the area-probability helpers in evaluate.py: python test_area_probs.py"""
import numpy as np
from evaluate import area_nmax as _nmax, area_p_indep as _p_indep

p = np.random.default_rng(0).uniform(0, .3, (2, 9, 9))
assert abs(_p_indep(p, 2)[1, 5, 5] - (1 - np.prod(1 - p[1, 3:8, 3:8]))) < 1e-9  # no mixing across batch
assert _nmax(np.eye(5)[None], 1)[0].sum() == 25 - 6                              # 3x3 dilation of a diagonal
print("ok")

# saved area samples load in bootstrap_pr_auc_ci.py unchanged (same npz schema)
import tempfile
from evaluate import area_collect, save_area_outputs
from bootstrap_pr_auc_ci import load_run
rng, store, d = np.random.default_rng(1), {}, tempfile.mkdtemp()
for seq in range(3):
    for t in range(2):
        obs = rng.random((32, 32)) < 0.05
        area_collect(store, rng.random((32, 32)), obs, 2, t, seq)
rows = save_area_outputs(d, "x", store)
run = load_run(f"{d}/area_x_r2px.npz")
assert set(run) == {0, 1} and run[0][0].size == 3 * 8 * 8 and set(run[0][2]) == {0, 1, 2}
assert len(rows) == 2 and 0 < rows[0]["base_rate"] < 1
print("ok (bootstrap-compatible)")
