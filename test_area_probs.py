"""Check of the area-probability helpers in tune_sampling_on_val.py: python test_area_probs.py"""
import numpy as np
from tune_sampling_on_val import _nmax, _p_indep

p = np.random.default_rng(0).uniform(0, .3, (2, 9, 9))
assert abs(_p_indep(p, 2)[1, 5, 5] - (1 - np.prod(1 - p[1, 3:8, 3:8]))) < 1e-9  # no mixing across batch
assert _nmax(np.eye(5)[None], 1)[0].sum() == 25 - 6                              # 3x3 dilation of a diagonal
print("ok")
