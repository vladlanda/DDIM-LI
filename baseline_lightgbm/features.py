"""
Feature extraction for the LightGBM baseline (see PAPER_TODO.md Phase 3
and FINDINGS.md F1/F3 for scope/framing rationale).

Shared between train_lightgbm.py and evaluate_lightgbm.py -- imported by
both, never duplicated, so train-time and inference-time features are
computed by the exact same code path. Feature-computation skew between
training and serving is one of the most common, hardest-to-notice bugs
in this kind of pipeline; a shared module makes that class of bug
structurally impossible rather than something to remember to avoid.

Design intent: a feature-engineered gradient-boosted-tree baseline "in
the methodological spirit of" Song et al. 2023 (npj Clim Atmos Sci
6:126) -- they combine meteorological/aerosol/satellite features into
LightGBM at coarse (0.25 deg/hourly) resolution; we don't have their
aerosol/reanalysis inputs, so this uses features derived purely from
the same ir/li satellite channels the diffusion model and CNN baseline
already use, at our own pixel-exact 4km/10min resolution (see F1 on why
that resolution choice matters for comparability).

Deliberately NOT included: absolute pixel row/col position. Everything
here is translation-invariant (temporal history + LOCAL spatial
neighbourhood only), matching the fully-convolutional CNN and diffusion
model's translation equivariance, and keeping this baseline honest for
the planned held-out-region generalization check (PAPER_TODO.md) --
including absolute coordinates would let the model memorize
region-specific geography rather than learn a transferable local
storm-behaviour signal, undermining that check before it's even run.

All functions operate on PHYSICAL-space li (after _li_to_physical) and
raw normalized ir -- ir doesn't need physical conversion for a
tree-based model (monotonic transforms don't change what LightGBM can
learn from a feature), but li DOES need physical-space thresholding
here, because several features are built around a physical activity
threshold, and that threshold must match the target label's own
definition (li_event_threshold in physical space) or "was this pixel
active in the past" would use a different definition of "active" than
"is this pixel active at the target lead", which would be a real,
confusing inconsistency.
"""
from typing import Dict, List

import numpy as np
from scipy.ndimage import maximum_filter, minimum_filter, uniform_filter

# Ordered feature list -- this exact order is what gets fed to LightGBM
# and must stay in sync with build_feature_matrix()'s output columns.
FEATURE_NAMES: List[str] = [
    "ir_now", "ir_mean_ctx", "ir_std_ctx", "ir_min_ctx", "ir_max_ctx",
    "ir_trend_30min", "ir_trend_2h", "ir_local_mean_9", "ir_local_min_9",
    "li_now_phys", "li_active_now", "li_recent_active_30min",
    "li_count_active_ctx", "li_frac_active_ctx", "li_time_since_last_active",
    "li_local_max_9", "li_local_count_9", "li_local_max_recent_9",
    "lead_idx",
]


def _li_to_physical(arr: np.ndarray, stats: Dict, ch: str = "li") -> np.ndarray:
    """Same inverse-transform as baseline_cnn/evaluate_cnn.py's _li_to_physical
    -- duplicated here (not imported) to keep this module dependency-free of
    baseline_cnn/, since it's a pure, trivial elementwise function and this
    baseline shouldn't need to import across sibling baseline directories."""
    if ch not in stats:
        return arr
    x = arr * stats[ch]["std"] + stats[ch]["mean"]
    if stats[ch].get("transform") == "cbrt":
        x = np.power(np.clip(x, 0.0, None), 3)
    return np.clip(x, 0.0, 1.0)


def compute_feature_maps(
    ir_ctx: np.ndarray,          # (T_in, H, W) raw normalized ir, one sequence
    li_ctx_phys: np.ndarray,     # (T_in, H, W) PHYSICAL-space li, one sequence
    li_event_threshold: float,
    short_lag: int = 3,          # ~30min at dt_min=10
    long_lag: int = 12,          # ~2h at dt_min=10, connects to E4
    local_size: int = 9,         # spatial neighbourhood window, pixels
) -> Dict[str, np.ndarray]:
    """
    Compute every feature in FEATURE_NAMES (except lead_idx, added by the
    caller since it's constant per-image, not spatially varying) as a dict
    of (H, W) maps for ONE sequence. Fully vectorized -- no per-pixel loops.

    ir_ctx/li_ctx_phys index 0..T_in-1 in chronological order (index -1 /
    T_in-1 is "now", the most recent observed frame).
    """
    T_in, H, W = ir_ctx.shape
    assert li_ctx_phys.shape == ir_ctx.shape

    ir_now = ir_ctx[-1]
    short_lag = min(short_lag, T_in - 1)
    long_lag = min(long_lag, T_in - 1)

    active = li_ctx_phys >= li_event_threshold  # (T_in, H, W) bool

    # "Time since last active" (in frames), vectorized: for each pixel, the
    # most recent frame index where active is True; T_in (sentinel) if never.
    time_idx = np.arange(T_in, dtype=np.float32).reshape(T_in, 1, 1)
    active_time_idx = np.where(active, time_idx, -1.0)
    last_active_idx = active_time_idx.max(axis=0)  # (H, W), -1 if never active
    time_since_last_active = np.where(
        last_active_idx >= 0, (T_in - 1) - last_active_idx, float(T_in)
    ).astype(np.float32)

    feats = {
        "ir_now":          ir_now,
        "ir_mean_ctx":     ir_ctx.mean(axis=0),
        "ir_std_ctx":      ir_ctx.std(axis=0),
        "ir_min_ctx":      ir_ctx.min(axis=0),
        "ir_max_ctx":      ir_ctx.max(axis=0),
        "ir_trend_30min":  ir_now - ir_ctx[-1 - short_lag],
        "ir_trend_2h":     ir_now - ir_ctx[-1 - long_lag],
        "ir_local_mean_9": uniform_filter(ir_now, size=local_size, mode="nearest"),
        "ir_local_min_9":  minimum_filter(ir_now, size=local_size, mode="nearest"),

        "li_now_phys":               li_ctx_phys[-1],
        "li_active_now":             active[-1].astype(np.float32),
        "li_recent_active_30min":    active[-1 - short_lag:].any(axis=0).astype(np.float32),
        "li_count_active_ctx":       active.sum(axis=0).astype(np.float32),
        "li_frac_active_ctx":        active.mean(axis=0).astype(np.float32),
        "li_time_since_last_active": time_since_last_active,
        "li_local_max_9":            maximum_filter(li_ctx_phys[-1], size=local_size, mode="nearest"),
        "li_local_count_9":          uniform_filter(active[-1].astype(np.float32), size=local_size, mode="nearest"),
        "li_local_max_recent_9":     maximum_filter(
            li_ctx_phys[-1 - short_lag:].max(axis=0), size=local_size, mode="nearest"
        ),
    }
    assert set(feats.keys()) == set(FEATURE_NAMES) - {"lead_idx"}, (
        "compute_feature_maps output doesn't match FEATURE_NAMES -- keep these in sync"
    )
    return feats


def feature_maps_to_matrix(
    feats: Dict[str, np.ndarray],
    lead_idx: int,
    pixel_indices: np.ndarray = None,
) -> np.ndarray:
    """Cheap step: turn an already-computed feats dict (from
    compute_feature_maps, called ONCE per sequence) into the (N, F) matrix
    for a specific lead_idx. Splitting this out from build_feature_matrix
    matters because compute_feature_maps does the expensive spatial-filter
    work and does NOT depend on lead_idx -- callers that need all T_out
    lead times per sequence (both train_lightgbm.py and
    evaluate_lightgbm.py do) should call compute_feature_maps ONCE and
    this function T_out times, not build_feature_matrix T_out times
    (which would redundantly recompute the same filters every time).
    """
    H, W = feats["ir_now"].shape
    if pixel_indices is None:
        cols = [feats[name].reshape(-1) for name in FEATURE_NAMES if name != "lead_idx"]
        n = H * W
    else:
        rows, cols_idx = pixel_indices[:, 0], pixel_indices[:, 1]
        cols = [feats[name][rows, cols_idx] for name in FEATURE_NAMES if name != "lead_idx"]
        n = pixel_indices.shape[0]
    lead_col = np.full(n, lead_idx, dtype=np.float32)
    X = np.stack(cols + [lead_col], axis=1).astype(np.float32)
    assert X.shape == (n, len(FEATURE_NAMES))
    return X


def build_feature_matrix(
    ir_ctx: np.ndarray,
    li_ctx_phys: np.ndarray,
    lead_idx: int,
    li_event_threshold: float,
    pixel_indices: np.ndarray = None,  # optional (N,2) array of (row, col) to subsample; None = all pixels
    **feature_kwargs,
) -> np.ndarray:
    """
    Convenience one-shot version: compute_feature_maps + feature_maps_to_matrix
    in one call, for callers that only need a SINGLE lead_idx per sequence.
    If you need multiple lead times for the same sequence (both training and
    evaluation scripts do, for all T_out steps), call compute_feature_maps
    once yourself and feature_maps_to_matrix per lead_idx instead -- see
    that function's docstring for why.
    """
    feats = compute_feature_maps(ir_ctx, li_ctx_phys, li_event_threshold, **feature_kwargs)
    return feature_maps_to_matrix(feats, lead_idx, pixel_indices)
