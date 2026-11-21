"""Check: fast weighted AP == sklearn AP on the explicitly duplicated resample (ties included)."""
import numpy as np
from sklearn.metrics import average_precision_score
from bootstrap_pr_auc_ci import _WeightedAP

rng = np.random.default_rng(0)
n_seq, per = 40, 50
seqid = np.repeat(np.arange(n_seq), per)
prob = np.round(rng.random(seqid.size), 2)            # many ties, like ensemble fractions
lbl = (rng.random(seqid.size) < prob * 0.6).astype(int)
uniq = np.unique(seqid)
ap = _WeightedAP(prob, lbl, seqid, uniq)
for _ in range(20):
    w = np.bincount(rng.integers(0, n_seq, n_seq), minlength=n_seq)
    idx = np.concatenate([np.flatnonzero(seqid == s) for s in np.repeat(uniq, w)])
    assert abs(ap(w) - average_precision_score(lbl[idx], prob[idx])) < 1e-12
assert abs(ap(np.ones(n_seq, int)) - average_precision_score(lbl, prob)) < 1e-12
print("ok")
