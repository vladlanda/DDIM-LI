"""Check for merge_eval_halves.py: python test_merge_halves.py (synthetic, ~2 s)."""
import csv, os, tempfile
import numpy as np
from evaluate import area_collect, save_area_outputs
from merge_eval_halves import main
from bootstrap_pr_auc_ci import load_run

rng, root = np.random.default_rng(0), tempfile.mkdtemp()
full = {}   # what a single full run would contain, in merged order
for half, n in (("a", 3), ("b", 2)):
    d = os.path.join(root, half); os.makedirs(d)
    pay, store = {"pr_steps": np.array([0, 1]), "dt_min": np.array(10)}, {}
    for t in (0, 1):
        p = rng.random(n * 16).astype(np.float32); l = (rng.random(n * 16) < .2).astype(np.int32)
        pay.update({f"pr_prob_{t}": p, f"pr_label_{t}": l, f"pr_seqid_{t}": np.repeat(np.arange(n), 16).astype(np.int32),
                    f"auc_{t}": np.array(rng.random())})                         # a summary: must be dropped
        full.setdefault(t, []).append(p)
        for s in range(n):
            area_collect(store, rng.random((8, 8)), rng.random((8, 8)) < .1, 1, t, s)
    np.savez_compressed(os.path.join(d, "x_pr_curves.npz"), **pay)
    save_area_outputs(d, "m", store)
    np.savez_compressed(os.path.join(d, "meta.npz"), n_sequences=np.array(n), T_out=np.array(2),
                        seqid=np.repeat(np.arange(n), 16).astype(np.int32), seq_time_utc=np.array([f"{half}{i}" for i in range(n)]))
    with open(os.path.join(d, "x_metrics.csv"), "w", newline="") as f:
        csv.writer(f).writerows([["lead", "crps", "model"], [10, 1.0 if half == "a" else 2.0, "x"]])

out = os.path.join(root, "m"); main(os.path.join(root, "a"), os.path.join(root, "b"), out)
run = load_run(os.path.join(out, "x_pr_curves.npz"))
assert np.array_equal(run[0][0], np.concatenate(full[0]))               # samples in merged order
assert set(run[0][2]) == {0, 1, 2, 3, 4}                                # ids 0-2 then 3-4
z = np.load(os.path.join(out, "x_pr_curves.npz")); assert "auc_0" not in z and int(z["dt_min"]) == 10
m = np.load(os.path.join(out, "meta.npz")); assert int(m["n_sequences"]) == 5 and list(m["seq_time_utc"]) == ["a0", "a1", "a2", "b0", "b1"]
assert set(np.load(os.path.join(out, "area_m_r1px.npz"))["pr_seqid_0"]) == {0, 1, 2, 3, 4}
assert os.path.exists(os.path.join(out, "area_m.csv"))
r = next(csv.DictReader(open(os.path.join(out, "x_metrics.csv"))))
assert abs(float(r["crps"]) - (3 * 1 + 2 * 2) / 5) < 1e-9 and r["model"] == "x"   # 3:2 weighting
print("ok")
